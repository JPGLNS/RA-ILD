#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Leakage-controlled Linear SVM nested CV using the validated TRB V2 feature layer."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from .linear_svm import (
    LinearSVMCandidate,
    add_natural_zero_metrics,
    apply_score_threshold,
    choose_youden_threshold,
    count_nonzero_coefficients,
    derive_svm_seed,
    fit_linear_svm,
    safe_average_precision,
    safe_roc_auc,
    score_classification_metrics,
)
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
from .public_reference import (
    ThresholdScheme,
    external_public_features,
    leave_one_out_public_features,
)
from .specifications import ResolvedModelSpecification


class LinearSVMNestedCVError(ValueError):
    """Raised when the Linear SVM nested-CV task is invalid."""


@dataclass(frozen=True)
class LinearSVMNestedCVOptions:
    base_seed: int
    max_iter: int = 10000
    tolerance: float = 1.0e-4
    zero_sd_tolerance: float = 1.0e-12
    epsilon: float = 1.0e-8
    coefficient_nonzero_tolerance: float = 1.0e-12
    fit_intercept: bool = True
    intercept_scaling: float = 1.0
    positive_label: str = "ILD"
    negative_label: str = "RA"
    label_column: str = "cohort"
    sample_id_column: str = "sample_id"

    def __post_init__(self) -> None:
        if isinstance(self.base_seed, bool) or int(self.base_seed) < 0:
            raise LinearSVMNestedCVError("base_seed must be an integer >= 0")
        if isinstance(self.max_iter, bool) or int(self.max_iter) < 1:
            raise LinearSVMNestedCVError("max_iter must be an integer >= 1")
        if float(self.tolerance) <= 0 or float(self.intercept_scaling) <= 0:
            raise LinearSVMNestedCVError("Invalid tolerance/intercept_scaling")
        if float(self.zero_sd_tolerance) < 0:
            raise LinearSVMNestedCVError("zero_sd_tolerance must be >= 0")
        if float(self.epsilon) <= 0:
            raise LinearSVMNestedCVError("epsilon must be > 0")
        if float(self.coefficient_nonzero_tolerance) < 0:
            raise LinearSVMNestedCVError(
                "coefficient_nonzero_tolerance must be >= 0"
            )
        if self.fit_intercept is not True:
            raise LinearSVMNestedCVError("Batch 15 requires fit_intercept=True")


@dataclass(frozen=True)
class SelectedLinearSVMTuning:
    candidate: LinearSVMCandidate
    threshold: float
    youden_j: float
    inner_roc_auc: float
    inner_average_precision: float
    candidate_selection_policy: str = "pooled_roc_ap_C_priority"
    threshold_source: str = "inner_oof_youden"


@dataclass(frozen=True)
class LinearSVMInnerTuningResult:
    tuning: pd.DataFrame
    selected_oof_predictions: pd.DataFrame
    selected: Mapping[str, SelectedLinearSVMTuning]


@dataclass(frozen=True)
class LinearSVMOuterFitResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    coefficients: pd.DataFrame
    preprocessing_audit: pd.DataFrame
    outer_train_public: pd.DataFrame
    outer_validation_public: pd.DataFrame
    public_reference_audit: pd.DataFrame
    public_loo_assignments: pd.DataFrame


@dataclass(frozen=True)
class LinearSVMNestedCVTaskResult:
    split: OuterTaskSplit
    inner_tuning: pd.DataFrame
    inner_selected_oof_predictions: pd.DataFrame
    selected: Mapping[str, SelectedLinearSVMTuning]
    outer_predictions: pd.DataFrame
    outer_metrics: pd.DataFrame
    coefficients: pd.DataFrame
    preprocessing_audit: pd.DataFrame
    outer_train_public: pd.DataFrame
    outer_validation_public: pd.DataFrame
    public_reference_audit: pd.DataFrame
    public_loo_assignments: pd.DataFrame
    sample_roles: pd.DataFrame


def _rank_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "candidate_id",
        "pooled_inner_roc_auc",
        "pooled_inner_average_precision",
        "C",
        "candidate_priority",
        "eligible_for_selection",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise LinearSVMNestedCVError(f"Tuning results missing columns: {missing}")
    eligible = frame.loc[frame["eligible_for_selection"].astype(bool)].copy()
    if eligible.empty:
        raise LinearSVMNestedCVError(
            "All Linear SVM candidates were ineligible because at least one inner fit did not converge"
        )
    return eligible.sort_values(
        [
            "pooled_inner_roc_auc",
            "pooled_inner_average_precision",
            "C",
            "candidate_priority",
            "candidate_id",
        ],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def tune_linear_svm_models(
    prepared: Mapping[str, Mapping[int, PreparedInnerFold]],
    candidates: Sequence[LinearSVMCandidate],
    options: LinearSVMNestedCVOptions,
) -> LinearSVMInnerTuningResult:
    if not candidates:
        raise LinearSVMNestedCVError("At least one Linear SVM candidate is required")
    tuning_rows: List[Dict[str, object]] = []
    prediction_cache: Dict[Tuple[str, str], List[Dict[str, object]]] = {}

    for model_index, (model_name, folds) in enumerate(prepared.items()):
        if not folds:
            raise LinearSVMNestedCVError(f"No prepared inner folds for {model_name}")
        for candidate in candidates:
            prediction_rows: List[Dict[str, object]] = []
            fold_aucs: List[float] = []
            fold_aps: List[float] = []
            convergence: List[bool] = []
            iterations: List[int] = []
            dual_resolved: List[bool] = []
            n_train_values: List[int] = []
            n_feature_values: List[int] = []
            for inner_fold, fold_data in folds.items():
                seed = derive_svm_seed(
                    options.base_seed,
                    "inner",
                    model_index,
                    int(inner_fold),
                    candidate.candidate_id,
                )
                fit = fit_linear_svm(
                    fold_data.X_train,
                    fold_data.y_train,
                    candidate=candidate,
                    max_iter=options.max_iter,
                    tolerance=options.tolerance,
                    fit_intercept=options.fit_intercept,
                    intercept_scaling=options.intercept_scaling,
                    random_state=seed,
                )
                score = fit.decision_score(fold_data.X_valid)
                fold_aucs.append(safe_roc_auc(fold_data.y_valid, score))
                fold_aps.append(safe_average_precision(fold_data.y_valid, score))
                convergence.append(bool(fit.converged))
                iterations.append(int(fit.n_iter))
                dual_resolved.append(bool(fit.dual_resolved))
                n_train_values.append(int(fold_data.X_train.shape[0]))
                n_feature_values.append(int(fold_data.X_train.shape[1]))
                for sample_id, truth, value in zip(
                    fold_data.valid_sample_ids, fold_data.y_valid, score
                ):
                    row: Dict[str, object] = {
                        "model": model_name,
                        "inner_fold": int(inner_fold),
                        "sample_id": str(sample_id),
                        "true_label": int(truth),
                        "prediction_score": float(value),
                        "decision_score_ILD": float(value),
                        "score_type": "decision_function",
                    }
                    row.update(candidate.as_dict())
                    prediction_rows.append(row)

            prediction_frame = pd.DataFrame(prediction_rows)
            pooled_y = prediction_frame["true_label"].to_numpy(int)
            pooled_score = prediction_frame["prediction_score"].to_numpy(float)
            all_converged = bool(all(convergence))
            row = {
                "model": model_name,
                **candidate.as_dict(),
                "pooled_inner_roc_auc": safe_roc_auc(pooled_y, pooled_score),
                "pooled_inner_pr_auc": safe_average_precision(pooled_y, pooled_score),
                "pooled_inner_average_precision": safe_average_precision(
                    pooled_y, pooled_score
                ),
                "mean_fold_roc_auc": float(np.mean(fold_aucs)),
                "sd_fold_roc_auc": float(np.std(fold_aucs, ddof=1)),
                "mean_fold_average_precision": float(np.mean(fold_aps)),
                "sd_fold_average_precision": float(np.std(fold_aps, ddof=1)),
                "all_fits_converged": all_converged,
                "eligible_for_selection": all_converged,
                "ineligibility_reason": "" if all_converged else "inner_fit_nonconvergence",
                "max_iterations_used": int(max(iterations)),
                "dual_resolved_values": ",".join(
                    sorted({str(value).lower() for value in dual_resolved})
                ),
                "min_inner_train_samples": int(min(n_train_values)),
                "max_inner_train_samples": int(max(n_train_values)),
                "min_final_predictors": int(min(n_feature_values)),
                "max_final_predictors": int(max(n_feature_values)),
                "candidate_selection_policy": "pooled_roc_ap_C_priority",
                "tuning_primary_metric": "roc_auc",
            }
            tuning_rows.append(row)
            prediction_cache[(model_name, candidate.candidate_id)] = prediction_rows

    tuning = pd.DataFrame(tuning_rows)
    selected: Dict[str, SelectedLinearSVMTuning] = {}
    selected_oof_parts: List[pd.DataFrame] = []
    candidate_lookup = {candidate.candidate_id: candidate for candidate in candidates}
    for model_name in prepared:
        ranked = _rank_candidates(tuning.loc[tuning["model"] == model_name].copy())
        best = ranked.iloc[0]
        candidate = candidate_lookup[str(best["candidate_id"])]
        prediction = pd.DataFrame(
            prediction_cache[(model_name, candidate.candidate_id)]
        )
        threshold, youden_j = choose_youden_threshold(
            prediction["true_label"].to_numpy(int),
            prediction["prediction_score"].to_numpy(float),
        )
        prediction["selected_threshold"] = float(threshold)
        prediction["threshold_source"] = "inner_oof_youden"
        prediction["candidate_selection_policy"] = "pooled_roc_ap_C_priority"
        selected_oof_parts.append(prediction)
        selected[model_name] = SelectedLinearSVMTuning(
            candidate=candidate,
            threshold=float(threshold),
            youden_j=float(youden_j),
            inner_roc_auc=float(best["pooled_inner_roc_auc"]),
            inner_average_precision=float(
                best["pooled_inner_average_precision"]
            ),
        )
    return LinearSVMInnerTuningResult(
        tuning=tuning,
        selected_oof_predictions=pd.concat(selected_oof_parts, ignore_index=True),
        selected=MappingProxyType(selected),
    )


def fit_outer_linear_svm_models(
    base: pd.DataFrame,
    split: OuterTaskSplit,
    row_lookup: Mapping[str, int],
    presence: sparse.spmatrix,
    frequency: sparse.spmatrix,
    labels: Sequence[object],
    aa_clone_numbers: Sequence[float],
    model_specifications: Mapping[str, ResolvedModelSpecification],
    selected: Mapping[str, SelectedLinearSVMTuning],
    scheme: ThresholdScheme,
    options: LinearSVMNestedCVOptions,
) -> LinearSVMOuterFitResult:
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
        presence,
        frequency,
        train_rows,
        labels,
        aa_clone_numbers,
        scheme,
        options.epsilon,
        context="outer_training_leave_one_out",
    )
    valid_public_result = external_public_features(
        presence,
        frequency,
        train_rows,
        valid_rows,
        labels,
        aa_clone_numbers,
        scheme,
        options.epsilon,
        context="outer_validation_application",
    )
    ref_train, assignment_train = _annotate_public_audit(
        train_public_result, split=split, inner_fold=None
    )
    ref_valid, _ = _annotate_public_audit(
        valid_public_result, split=split, inner_fold=None
    )
    train_frame = assemble_model_dataframe(
        indexed, split.outer_train_ids, train_public_result.features, row_lookup
    )
    valid_frame = assemble_model_dataframe(
        indexed, split.outer_valid_ids, valid_public_result.features, row_lookup
    )
    y_train = _binary_labels(train_frame, options)
    y_valid = _binary_labels(valid_frame, options)

    prediction_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []
    coefficient_parts: List[pd.DataFrame] = []
    preprocessing_parts: List[pd.DataFrame] = []

    for model_index, (model_name, specification) in enumerate(
        model_specifications.items()
    ):
        if model_name not in selected:
            raise LinearSVMNestedCVError(
                f"Selected tuning parameters missing for {model_name}"
            )
        design: PreparedDesign = prepare_design_matrices(
            train_frame,
            valid_frame,
            specification.numeric,
            specification.categorical,
            zero_sd_tolerance=options.zero_sd_tolerance,
        )
        audit = design.audit.copy()
        audit["stage"] = "outer_final"
        audit["model"] = model_name
        audit["inner_fold"] = ""
        preprocessing_parts.append(audit)

        tuning = selected[model_name]
        seed = derive_svm_seed(options.base_seed, "outer", model_index)
        fit = fit_linear_svm(
            design.X_train,
            y_train,
            candidate=tuning.candidate,
            max_iter=options.max_iter,
            tolerance=options.tolerance,
            fit_intercept=options.fit_intercept,
            intercept_scaling=options.intercept_scaling,
            random_state=seed,
        )
        if not fit.converged:
            raise LinearSVMNestedCVError(
                f"Final outer model did not converge: {model_name}"
            )
        score = fit.decision_score(design.X_valid)
        threshold = float(tuning.threshold)
        metrics: Dict[str, object] = score_classification_metrics(
            y_valid, score, threshold
        )
        metrics = add_natural_zero_metrics(metrics, y_valid, score)
        metrics.update(
            {
                "model": model_name,
                "model_definition": specification.description,
                "model_engine": "linear_svc",
                "score_type": "decision_function",
                "threshold_source": tuning.threshold_source,
                "selected_threshold": threshold,
                "selected_youden_j": float(tuning.youden_j),
                "outer_repeat": split.outer_repeat,
                "outer_fold": split.outer_fold,
                "n_outer_train": split.n_outer_train,
                "n_outer_validation": split.n_outer_valid,
                "inner_selected_roc_auc": float(tuning.inner_roc_auc),
                "inner_selected_pr_auc": float(tuning.inner_average_precision),
                "inner_selected_average_precision": float(
                    tuning.inner_average_precision
                ),
                "candidate_selection_policy": tuning.candidate_selection_policy,
                "selected_candidate_id": tuning.candidate.candidate_id,
                "selected_candidate_block": tuning.candidate.block_id,
                "selected_candidate_priority": int(tuning.candidate.priority),
                "selected_C": float(tuning.candidate.C),
                "selected_penalty": tuning.candidate.penalty,
                "selected_loss": tuning.candidate.loss,
                "selected_class_weight": tuning.candidate.class_weight,
                "selected_dual_requested": tuning.candidate.dual,
                "selected_dual_resolved": bool(fit.dual_resolved),
                "fit_converged": bool(fit.converged),
                "iterations_used": int(fit.n_iter),
                "n_final_predictors": int(len(design.feature_names)),
                "n_nonzero_coefficients": count_nonzero_coefficients(
                    fit.model, options.coefficient_nonzero_tolerance
                ),
                "coefficient_sparsity_interpretable": bool(
                    tuning.candidate.penalty == "l1"
                ),
                "log_loss": np.nan,
                "brier_score": np.nan,
                "probability_metrics_available": False,
            }
        )
        metric_rows.append(metrics)

        predicted = apply_score_threshold(score, threshold)
        zero_predicted = apply_score_threshold(score, 0.0)
        for sample_id, truth, value, prediction, zero_prediction in zip(
            split.outer_valid_ids,
            y_valid,
            score,
            predicted,
            zero_predicted,
        ):
            prediction_rows.append(
                {
                    "model": model_name,
                    "model_engine": "linear_svc",
                    "outer_repeat": split.outer_repeat,
                    "outer_fold": split.outer_fold,
                    "sample_id": str(sample_id),
                    "true_label": int(truth),
                    "true_cohort": options.positive_label
                    if truth == 1
                    else options.negative_label,
                    "prediction_score": float(value),
                    "decision_score_ILD": float(value),
                    "score_type": "decision_function",
                    "threshold": threshold,
                    "threshold_source": tuning.threshold_source,
                    "predicted_label": int(prediction),
                    "predicted_cohort": options.positive_label
                    if prediction == 1
                    else options.negative_label,
                    "natural_zero_threshold": 0.0,
                    "zero_threshold_predicted_label": int(zero_prediction),
                    "zero_threshold_predicted_cohort": options.positive_label
                    if zero_prediction == 1
                    else options.negative_label,
                }
            )
        coefficient_parts.append(
            fit.coefficient_frame(
                design.feature_names,
                nonzero_tolerance=options.coefficient_nonzero_tolerance,
                model_name=model_name,
            )
        )

    train_public = train_public_result.features.loc[train_rows].copy()
    train_public.insert(0, options.sample_id_column, list(split.outer_train_ids))
    valid_public = valid_public_result.features.loc[valid_rows].copy()
    valid_public.insert(0, options.sample_id_column, list(split.outer_valid_ids))
    return LinearSVMOuterFitResult(
        predictions=pd.DataFrame(prediction_rows),
        metrics=pd.DataFrame(metric_rows),
        coefficients=pd.concat(coefficient_parts, ignore_index=True),
        preprocessing_audit=pd.concat(preprocessing_parts, ignore_index=True),
        outer_train_public=train_public.reset_index(drop=True),
        outer_validation_public=valid_public.reset_index(drop=True),
        public_reference_audit=pd.concat(
            [frame for frame in (ref_train, ref_valid) if not frame.empty],
            ignore_index=True,
        ),
        public_loo_assignments=assignment_train.reset_index(drop=True),
    )


def run_linear_svm_outer_task(
    base: pd.DataFrame,
    outer_assignments: pd.DataFrame,
    inner_assignments: pd.DataFrame,
    row_lookup: Mapping[str, int],
    presence: sparse.spmatrix,
    frequency: sparse.spmatrix,
    labels: Sequence[object],
    aa_clone_numbers: Sequence[float],
    model_specifications: Mapping[str, ResolvedModelSpecification],
    candidates: Sequence[LinearSVMCandidate],
    scheme: ThresholdScheme,
    options: LinearSVMNestedCVOptions,
    *,
    outer_repeat: int,
    outer_fold: int,
    outer_folds: int,
    inner_folds: int,
) -> LinearSVMNestedCVTaskResult:
    split = validate_fixed_assignments(
        base[options.sample_id_column].astype(str).tolist(),
        outer_assignments,
        inner_assignments,
        outer_repeat=outer_repeat,
        outer_fold=outer_fold,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        sample_id_column=options.sample_id_column,
    )
    inner: InnerPreparation = prepare_inner_folds(
        base,
        split,
        row_lookup,
        presence,
        frequency,
        labels,
        aa_clone_numbers,
        model_specifications,
        scheme,
        options,
    )
    tuning = tune_linear_svm_models(inner.prepared, candidates, options)
    outer = fit_outer_linear_svm_models(
        base,
        split,
        row_lookup,
        presence,
        frequency,
        labels,
        aa_clone_numbers,
        model_specifications,
        tuning.selected,
        scheme,
        options,
    )
    return LinearSVMNestedCVTaskResult(
        split=split,
        inner_tuning=tuning.tuning,
        inner_selected_oof_predictions=tuning.selected_oof_predictions,
        selected=tuning.selected,
        outer_predictions=outer.predictions,
        outer_metrics=outer.metrics,
        coefficients=outer.coefficients,
        preprocessing_audit=pd.concat(
            [inner.preprocessing_audit, outer.preprocessing_audit],
            ignore_index=True,
        ),
        outer_train_public=outer.outer_train_public,
        outer_validation_public=outer.outer_validation_public,
        public_reference_audit=pd.concat(
            [inner.public_reference_audit, outer.public_reference_audit],
            ignore_index=True,
        ),
        public_loo_assignments=pd.concat(
            [inner.public_loo_assignments, outer.public_loo_assignments],
            ignore_index=True,
        ),
        sample_roles=sample_role_frame(base, split, options),
    )
