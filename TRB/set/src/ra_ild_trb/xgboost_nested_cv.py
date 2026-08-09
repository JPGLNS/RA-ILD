#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Leakage-controlled XGBoost nested CV reusing the validated TRB V2 feature layer."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import brier_score_loss, log_loss

from .metrics import classification_metrics, safe_pr_auc, safe_roc_auc
from .nested_cv import (
    InnerPreparation,
    OuterTaskSplit,
    PreparedInnerFold,
    _annotate_public_audit,
    _binary_labels,
    _indexed_base,
    assemble_model_dataframe,
    prepare_inner_folds,
    sample_role_frame,
    validate_fixed_assignments,
)
from .preprocessing import PreparedDesign, prepare_design_matrices
from .public_reference import ThresholdScheme, external_public_features, leave_one_out_public_features
from .specifications import ResolvedModelSpecification
from .thresholds import apply_threshold, choose_threshold_youden
from .xgboost_model import XGBoostCandidate, derive_xgb_seed, fit_xgboost


class XGBoostNestedCVError(ValueError):
    pass


@dataclass(frozen=True)
class XGBoostNestedCVOptions:
    base_seed: int
    tuning_primary_metric: str = "roc_auc"
    zero_sd_tolerance: float = 1.0e-12
    epsilon: float = 1.0e-8
    positive_label: str = "ILD"
    negative_label: str = "RA"
    label_column: str = "cohort"
    sample_id_column: str = "sample_id"
    n_jobs_per_fit: int = 1
    fixed_parameters: Mapping[str, Any] = None

    def __post_init__(self) -> None:
        if isinstance(self.base_seed, bool) or not isinstance(self.base_seed, int) or self.base_seed < 0:
            raise XGBoostNestedCVError("base_seed must be an integer >= 0")
        metric = str(self.tuning_primary_metric).strip().lower()
        if metric not in {"roc_auc", "log_loss"}:
            raise XGBoostNestedCVError("tuning_primary_metric must be roc_auc or log_loss")
        object.__setattr__(self, "tuning_primary_metric", metric)
        if float(self.zero_sd_tolerance) < 0 or float(self.epsilon) <= 0:
            raise XGBoostNestedCVError("Invalid zero_sd_tolerance/epsilon")
        if isinstance(self.n_jobs_per_fit, bool) or not isinstance(self.n_jobs_per_fit, int) or self.n_jobs_per_fit < 1:
            raise XGBoostNestedCVError("n_jobs_per_fit must be an integer >= 1")
        object.__setattr__(self, "fixed_parameters", dict(self.fixed_parameters or {}))


@dataclass(frozen=True)
class SelectedXGBoostTuning:
    candidate: XGBoostCandidate
    threshold: float
    inner_roc_auc: float
    inner_pr_auc: float
    inner_log_loss: float
    inner_brier_score: float
    tuning_primary_metric: str
    candidate_selection_policy: str
    threshold_source: str = "inner_oof_youden"


@dataclass(frozen=True)
class XGBoostInnerTuningResult:
    tuning: pd.DataFrame
    selected_oof_predictions: pd.DataFrame
    selected: Mapping[str, SelectedXGBoostTuning]


@dataclass(frozen=True)
class XGBoostOuterFitResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    preprocessing_audit: pd.DataFrame
    outer_train_public: pd.DataFrame
    outer_validation_public: pd.DataFrame
    public_reference_audit: pd.DataFrame
    public_loo_assignments: pd.DataFrame


@dataclass(frozen=True)
class XGBoostNestedCVTaskResult:
    split: OuterTaskSplit
    inner_tuning: pd.DataFrame
    inner_selected_oof_predictions: pd.DataFrame
    selected: Mapping[str, SelectedXGBoostTuning]
    outer_predictions: pd.DataFrame
    outer_metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    preprocessing_audit: pd.DataFrame
    outer_train_public: pd.DataFrame
    outer_validation_public: pd.DataFrame
    public_reference_audit: pd.DataFrame
    public_loo_assignments: pd.DataFrame
    sample_roles: pd.DataFrame


def _selection_policy(metric: str) -> str:
    return "pooled_roc_pr_logloss_candidate" if metric == "roc_auc" else "pooled_logloss_roc_pr_candidate"


def _rank_candidates(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    required = {
        "candidate_id", "pooled_inner_roc_auc", "pooled_inner_pr_auc",
        "pooled_inner_log_loss", "candidate_sample_order",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise XGBoostNestedCVError(f"Tuning results missing columns: {missing}")
    if metric == "roc_auc":
        columns = ["pooled_inner_roc_auc", "pooled_inner_pr_auc", "pooled_inner_log_loss", "candidate_sample_order", "candidate_id"]
        ascending = [False, False, True, True, True]
    else:
        columns = ["pooled_inner_log_loss", "pooled_inner_roc_auc", "pooled_inner_pr_auc", "candidate_sample_order", "candidate_id"]
        ascending = [True, False, False, True, True]
    return frame.sort_values(columns, ascending=ascending, kind="mergesort").reset_index(drop=True)


def tune_xgboost_models(
    prepared: Mapping[str, Mapping[int, PreparedInnerFold]],
    candidates: Sequence[XGBoostCandidate],
    options: XGBoostNestedCVOptions,
    *,
    outer_repeat: int,
    outer_fold: int,
) -> XGBoostInnerTuningResult:
    if not candidates:
        raise XGBoostNestedCVError("At least one XGBoost candidate is required")
    if isinstance(outer_repeat, bool) or int(outer_repeat) < 1:
        raise XGBoostNestedCVError("outer_repeat must be an integer >= 1")
    if isinstance(outer_fold, bool) or int(outer_fold) < 1:
        raise XGBoostNestedCVError("outer_fold must be an integer >= 1")
    tuning_rows: List[Dict[str, Any]] = []
    prediction_cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    policy = _selection_policy(options.tuning_primary_metric)

    for model_index, (model_name, folds) in enumerate(prepared.items()):
        if not folds:
            raise XGBoostNestedCVError(f"No prepared inner folds for {model_name}")
        for candidate in candidates:
            prediction_rows: List[Dict[str, Any]] = []
            fold_roc: List[float] = []
            fold_pr: List[float] = []
            fold_ll: List[float] = []
            fold_brier: List[float] = []
            n_features: List[int] = []
            seeds: List[int] = []
            for inner_fold, fold_data in folds.items():
                seed = derive_xgb_seed(
                    options.base_seed,
                    "inner",
                    int(outer_repeat),
                    int(outer_fold),
                    model_index,
                    int(inner_fold),
                    candidate.candidate_id,
                )
                fit = fit_xgboost(
                    fold_data.X_train,
                    fold_data.y_train,
                    candidate=candidate,
                    fixed_parameters=options.fixed_parameters,
                    random_state=seed,
                    n_jobs=options.n_jobs_per_fit,
                )
                probability = fit.predict_probability(fold_data.X_valid)
                fold_roc.append(safe_roc_auc(fold_data.y_valid, probability))
                fold_pr.append(safe_pr_auc(fold_data.y_valid, probability))
                fold_ll.append(float(log_loss(fold_data.y_valid, probability, labels=[0, 1])))
                fold_brier.append(float(brier_score_loss(fold_data.y_valid, probability)))
                n_features.append(int(fold_data.X_train.shape[1]))
                seeds.append(int(seed))
                for sample_id, truth, value in zip(fold_data.valid_sample_ids, fold_data.y_valid, probability):
                    row: Dict[str, Any] = {
                        "model": model_name,
                        "inner_fold": int(inner_fold),
                        "sample_id": str(sample_id),
                        "true_label": int(truth),
                        "probability_ILD": float(value),
                        "prediction_probability": float(value),
                        "prediction_score": float(value),
                        "score_type": "probability",
                    }
                    row.update(candidate.as_dict())
                    prediction_rows.append(row)
            prediction_frame = pd.DataFrame(prediction_rows)
            pooled_y = prediction_frame["true_label"].to_numpy(int)
            pooled_probability = prediction_frame["probability_ILD"].to_numpy(float)
            row = {
                "model": model_name,
                **candidate.as_dict(),
                "pooled_inner_roc_auc": safe_roc_auc(pooled_y, pooled_probability),
                "pooled_inner_pr_auc": safe_pr_auc(pooled_y, pooled_probability),
                "pooled_inner_average_precision": safe_pr_auc(pooled_y, pooled_probability),
                "pooled_inner_log_loss": float(log_loss(pooled_y, pooled_probability, labels=[0, 1])),
                "pooled_inner_brier_score": float(brier_score_loss(pooled_y, pooled_probability)),
                "mean_fold_roc_auc": float(np.mean(fold_roc)),
                "sd_fold_roc_auc": float(np.std(fold_roc, ddof=1)) if len(fold_roc) > 1 else 0.0,
                "mean_fold_pr_auc": float(np.mean(fold_pr)),
                "sd_fold_pr_auc": float(np.std(fold_pr, ddof=1)) if len(fold_pr) > 1 else 0.0,
                "mean_fold_log_loss": float(np.mean(fold_ll)),
                "sd_fold_log_loss": float(np.std(fold_ll, ddof=1)) if len(fold_ll) > 1 else 0.0,
                "mean_fold_brier_score": float(np.mean(fold_brier)),
                "sd_fold_brier_score": float(np.std(fold_brier, ddof=1)) if len(fold_brier) > 1 else 0.0,
                "min_final_predictors": int(min(n_features)),
                "max_final_predictors": int(max(n_features)),
                "fit_seed_min": int(min(seeds)),
                "fit_seed_max": int(max(seeds)),
                "candidate_selection_policy": policy,
                "tuning_primary_metric": options.tuning_primary_metric,
            }
            tuning_rows.append(row)
            prediction_cache[(model_name, candidate.candidate_id)] = prediction_rows

    tuning = pd.DataFrame(tuning_rows)
    selected: Dict[str, SelectedXGBoostTuning] = {}
    selected_oof_parts: List[pd.DataFrame] = []
    candidate_lookup = {c.candidate_id: c for c in candidates}
    for model_name in prepared:
        ranked = _rank_candidates(tuning.loc[tuning["model"] == model_name].copy(), options.tuning_primary_metric)
        best = ranked.iloc[0]
        candidate = candidate_lookup[str(best["candidate_id"])]
        prediction = pd.DataFrame(prediction_cache[(model_name, candidate.candidate_id)])
        threshold = choose_threshold_youden(
            prediction["true_label"].to_numpy(int), prediction["probability_ILD"].to_numpy(float)
        )
        prediction["selected_threshold"] = float(threshold)
        prediction["threshold_source"] = "inner_oof_youden"
        prediction["candidate_selection_policy"] = policy
        prediction["tuning_primary_metric"] = options.tuning_primary_metric
        prediction["outer_repeat"] = int(outer_repeat)
        prediction["outer_fold"] = int(outer_fold)
        selected_oof_parts.append(prediction)
        selected[model_name] = SelectedXGBoostTuning(
            candidate=candidate,
            threshold=float(threshold),
            inner_roc_auc=float(best["pooled_inner_roc_auc"]),
            inner_pr_auc=float(best["pooled_inner_pr_auc"]),
            inner_log_loss=float(best["pooled_inner_log_loss"]),
            inner_brier_score=float(best["pooled_inner_brier_score"]),
            tuning_primary_metric=options.tuning_primary_metric,
            candidate_selection_policy=policy,
        )
    return XGBoostInnerTuningResult(
        tuning=tuning,
        selected_oof_predictions=pd.concat(selected_oof_parts, ignore_index=True),
        selected=MappingProxyType(selected),
    )


def fit_outer_xgboost_models(
    base: pd.DataFrame,
    split: OuterTaskSplit,
    row_lookup: Mapping[str, int],
    presence: sparse.spmatrix,
    frequency: sparse.spmatrix,
    labels: Sequence[object],
    aa_clone_numbers: Sequence[float],
    model_specifications: Mapping[str, ResolvedModelSpecification],
    selected: Mapping[str, SelectedXGBoostTuning],
    scheme: ThresholdScheme,
    options: XGBoostNestedCVOptions,
) -> XGBoostOuterFitResult:
    indexed = _indexed_base(
        base,
        sample_id_column=options.sample_id_column,
        label_column=options.label_column,
        positive_label=options.positive_label,
        negative_label=options.negative_label,
    )
    train_rows = np.asarray([row_lookup[x] for x in split.outer_train_ids], dtype=int)
    valid_rows = np.asarray([row_lookup[x] for x in split.outer_valid_ids], dtype=int)
    train_public_result = leave_one_out_public_features(
        presence, frequency, train_rows, labels, aa_clone_numbers, scheme, options.epsilon,
        context="outer_training_leave_one_out",
    )
    valid_public_result = external_public_features(
        presence, frequency, train_rows, valid_rows, labels, aa_clone_numbers, scheme, options.epsilon,
        context="outer_validation_application",
    )
    ref_train, assignment_train = _annotate_public_audit(train_public_result, split=split, inner_fold=None)
    ref_valid, _ = _annotate_public_audit(valid_public_result, split=split, inner_fold=None)
    train_frame = assemble_model_dataframe(indexed, split.outer_train_ids, train_public_result.features, row_lookup)
    valid_frame = assemble_model_dataframe(indexed, split.outer_valid_ids, valid_public_result.features, row_lookup)
    y_train = _binary_labels(train_frame, options)
    y_valid = _binary_labels(valid_frame, options)
    split_id = f"repeat_{split.outer_repeat:02d}_fold_{split.outer_fold:02d}"

    prediction_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    importance_parts: List[pd.DataFrame] = []
    preprocessing_parts: List[pd.DataFrame] = []

    for model_index, (model_name, specification) in enumerate(model_specifications.items()):
        if model_name not in selected:
            raise XGBoostNestedCVError(f"Selected tuning parameters missing for {model_name}")
        design: PreparedDesign = prepare_design_matrices(
            train_frame, valid_frame, specification.numeric, specification.categorical,
            zero_sd_tolerance=options.zero_sd_tolerance,
        )
        audit = design.audit.copy()
        audit["stage"] = "outer_final"
        audit["model"] = model_name
        audit["inner_fold"] = ""
        preprocessing_parts.append(audit)

        tuning = selected[model_name]
        seed = derive_xgb_seed(options.base_seed, "outer", split.outer_repeat, split.outer_fold, model_index, tuning.candidate.candidate_id)
        fit = fit_xgboost(
            design.X_train, y_train,
            candidate=tuning.candidate,
            fixed_parameters=options.fixed_parameters,
            random_state=seed,
            n_jobs=options.n_jobs_per_fit,
        )
        probability = fit.predict_probability(design.X_valid)
        threshold = float(tuning.threshold)
        metrics: Dict[str, Any] = classification_metrics(
            y_valid, probability, threshold, include_brier=True, include_log_loss=True, include_sample_summary=True
        )
        metrics["average_precision"] = metrics["pr_auc"]
        selected_payload = tuning.candidate.as_dict(prefix="selected_")
        metrics.update(
            {
                "model": model_name,
                "model_definition": specification.description,
                "model_engine": "xgboost",
                "score_type": "probability",
                "split_id": split_id,
                "threshold_source": tuning.threshold_source,
                "selected_threshold": threshold,
                "outer_repeat": split.outer_repeat,
                "outer_fold": split.outer_fold,
                "n_outer_train": split.n_outer_train,
                "n_outer_validation": split.n_outer_valid,
                "inner_selected_roc_auc": tuning.inner_roc_auc,
                "inner_selected_pr_auc": tuning.inner_pr_auc,
                "inner_selected_log_loss": tuning.inner_log_loss,
                "inner_selected_brier_score": tuning.inner_brier_score,
                "tuning_primary_metric": tuning.tuning_primary_metric,
                "candidate_selection_policy": tuning.candidate_selection_policy,
                "probability_metrics_available": True,
                "fit_random_state": int(seed),
                "n_final_predictors": int(len(design.feature_names)),
                **selected_payload,
            }
        )
        metric_rows.append(metrics)

        predicted = apply_threshold(probability, threshold)
        for sample_id, truth, value, prediction in zip(split.outer_valid_ids, y_valid, probability, predicted):
            prediction_rows.append(
                {
                    "model": model_name,
                    "model_engine": "xgboost",
                    "score_type": "probability",
                    "split_id": split_id,
                    "outer_repeat": split.outer_repeat,
                    "outer_fold": split.outer_fold,
                    "sample_id": str(sample_id),
                    "true_label": int(truth),
                    "true_cohort": options.positive_label if truth == 1 else options.negative_label,
                    "probability_ILD": float(value),
                    "prediction_probability": float(value),
                    "prediction_score": float(value),
                    "threshold": threshold,
                    "threshold_source": tuning.threshold_source,
                    "predicted_label": int(prediction),
                    "predicted_cohort": options.positive_label if prediction == 1 else options.negative_label,
                }
            )
        importance = fit.importance_frame(design.feature_names, model_name=model_name)
        importance["split_id"] = split_id
        importance["outer_repeat"] = split.outer_repeat
        importance["outer_fold"] = split.outer_fold
        importance["selected_candidate_id"] = tuning.candidate.candidate_id
        importance_parts.append(importance)

    train_public = train_public_result.features.loc[train_rows].copy()
    train_public.insert(0, options.sample_id_column, list(split.outer_train_ids))
    valid_public = valid_public_result.features.loc[valid_rows].copy()
    valid_public.insert(0, options.sample_id_column, list(split.outer_valid_ids))
    return XGBoostOuterFitResult(
        predictions=pd.DataFrame(prediction_rows),
        metrics=pd.DataFrame(metric_rows),
        feature_importance=pd.concat(importance_parts, ignore_index=True),
        preprocessing_audit=pd.concat(preprocessing_parts, ignore_index=True),
        outer_train_public=train_public.reset_index(drop=True),
        outer_validation_public=valid_public.reset_index(drop=True),
        public_reference_audit=pd.concat([f for f in (ref_train, ref_valid) if not f.empty], ignore_index=True),
        public_loo_assignments=assignment_train.reset_index(drop=True),
    )


def run_xgboost_outer_task(
    base: pd.DataFrame,
    outer_assignments: pd.DataFrame,
    inner_assignments: pd.DataFrame,
    row_lookup: Mapping[str, int],
    presence: sparse.spmatrix,
    frequency: sparse.spmatrix,
    labels: Sequence[object],
    aa_clone_numbers: Sequence[float],
    model_specifications: Mapping[str, ResolvedModelSpecification],
    candidates: Sequence[XGBoostCandidate],
    scheme: ThresholdScheme,
    options: XGBoostNestedCVOptions,
    *,
    outer_repeat: int,
    outer_fold: int,
    outer_folds: int,
    inner_folds: int,
) -> XGBoostNestedCVTaskResult:
    split = validate_fixed_assignments(
        base[options.sample_id_column].astype(str).tolist(), outer_assignments, inner_assignments,
        outer_repeat=outer_repeat, outer_fold=outer_fold, outer_folds=outer_folds,
        inner_folds=inner_folds, sample_id_column=options.sample_id_column,
    )
    inner: InnerPreparation = prepare_inner_folds(
        base, split, row_lookup, presence, frequency, labels, aa_clone_numbers,
        model_specifications, scheme, options,
    )
    tuning = tune_xgboost_models(
        inner.prepared,
        candidates,
        options,
        outer_repeat=split.outer_repeat,
        outer_fold=split.outer_fold,
    )
    outer = fit_outer_xgboost_models(
        base, split, row_lookup, presence, frequency, labels, aa_clone_numbers,
        model_specifications, tuning.selected, scheme, options,
    )
    return XGBoostNestedCVTaskResult(
        split=split,
        inner_tuning=tuning.tuning,
        inner_selected_oof_predictions=tuning.selected_oof_predictions,
        selected=tuning.selected,
        outer_predictions=outer.predictions,
        outer_metrics=outer.metrics,
        feature_importance=outer.feature_importance,
        preprocessing_audit=pd.concat([inner.preprocessing_audit, outer.preprocessing_audit], ignore_index=True),
        outer_train_public=outer.outer_train_public,
        outer_validation_public=outer.outer_validation_public,
        public_reference_audit=pd.concat([inner.public_reference_audit, outer.public_reference_audit], ignore_index=True),
        public_loo_assignments=pd.concat([inner.public_loo_assignments, outer.public_loo_assignments], ignore_index=True),
        sample_roles=sample_role_frame(base, split, options),
    )
