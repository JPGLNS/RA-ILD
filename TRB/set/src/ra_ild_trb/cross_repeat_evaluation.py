#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen-model n×n cross-repeat evaluation for TRB repeated holdouts."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from .metrics import classification_metrics
from .nested_cv import NestedCVOptions, assemble_model_dataframe, validate_fixed_assignments
from .preprocessing import FittedPreprocessor, fit_preprocessor
from .public_reference import ALL_PUBLIC_FEATURES, ThresholdScheme, external_public_features
from .specifications import ResolvedModelSpecification
from .thresholds import apply_threshold


class CrossRepeatEvaluationError(ValueError):
    """Raised when frozen cross-repeat evaluation is inconsistent."""


REPORT_METRICS = (
    "roc_auc", "pr_auc", "accuracy", "sensitivity_recall",
    "specificity", "precision", "f1", "brier_score",
)


@dataclass(frozen=True)
class SelectedModelParameters:
    alpha: float
    lambda_value: float
    threshold: float
    inner_roc_auc: float
    inner_pr_auc: float


@dataclass(frozen=True)
class FrozenTaskArtifacts:
    selected: Mapping[str, SelectedModelParameters]
    coefficients: pd.DataFrame
    preprocessing: pd.DataFrame
    outer_train_public: pd.DataFrame
    outer_validation_public: pd.DataFrame
    native_predictions: pd.DataFrame


@dataclass(frozen=True)
class FrozenLinearModel:
    model_name: str
    model_definition: str
    preprocessor: FittedPreprocessor
    intercept: float
    coefficients: np.ndarray
    feature_names: Tuple[str, ...]
    selected: SelectedModelParameters


@dataclass(frozen=True)
class CrossRepeatEvaluationResult:
    predictions: pd.DataFrame
    full_metrics: pd.DataFrame
    unseen_only_metrics: pd.DataFrame
    overlap_audit: pd.DataFrame
    preprocessing_reproduction_audit: pd.DataFrame
    native_public_reproduction_audit: pd.DataFrame
    native_prediction_reproduction_audit: pd.DataFrame
    prediction_consistency_audit: pd.DataFrame


def expected_cross_repeat_counts(repeats: int, models: int, holdout_size: int) -> Mapping[str, int]:
    values = (repeats, models, holdout_size)
    if any(isinstance(v, bool) or int(v) < 1 for v in values):
        raise CrossRepeatEvaluationError("repeats, models, and holdout_size must be positive integers")
    r, m, h = map(int, values)
    return MappingProxyType({
        "model_instances": r * m,
        "evaluation_pairs_per_model": r * r,
        "full_metric_rows": r * r * m,
        "unseen_only_metric_rows": r * r * m,
        "prediction_rows": r * r * m * h,
        "overlap_rows": r * r,
        "native_prediction_rows": r * m * h,
    })


def evaluation_role(*, fit_repeat: int, evaluation_repeat: int, sample_id: str,
                    fit_training_ids: Sequence[str]) -> str:
    if int(fit_repeat) == int(evaluation_repeat):
        return "native_holdout"
    if str(sample_id) in set(map(str, fit_training_ids)):
        return "cross_holdout_seen_training"
    return "cross_holdout_unseen"


def _ids(values: Sequence[object], label: str) -> Tuple[str, ...]:
    out = tuple(str(v) for v in values)
    if not out or len(out) != len(set(out)):
        raise CrossRepeatEvaluationError(f"{label} is empty or duplicated")
    return out


def build_overlap_audit(split_membership: Mapping[int, Mapping[str, Sequence[str]]]) -> pd.DataFrame:
    repeats = tuple(sorted(int(v) for v in split_membership))
    if repeats != tuple(range(1, len(repeats) + 1)):
        raise CrossRepeatEvaluationError("repeat IDs must be consecutive starting at 1")
    rows, full = [], None
    for fit_repeat in repeats:
        train = set(_ids(split_membership[fit_repeat]["training"], f"repeat {fit_repeat} training"))
        holdout = set(_ids(split_membership[fit_repeat]["holdout"], f"repeat {fit_repeat} holdout"))
        if train & holdout:
            raise CrossRepeatEvaluationError(f"repeat {fit_repeat} training/holdout overlap")
        cohort = train | holdout
        if full is None:
            full = cohort
        elif cohort != full:
            raise CrossRepeatEvaluationError("all repeats must use the same cohort")
        for eval_repeat in repeats:
            test = set(_ids(split_membership[eval_repeat]["holdout"], f"repeat {eval_repeat} holdout"))
            seen, unseen = test & train, test - train
            shared, union = test & holdout, test | holdout
            rows.append({
                "fit_repeat": fit_repeat,
                "evaluation_repeat": eval_repeat,
                "is_native_pair": fit_repeat == eval_repeat,
                "n_evaluation_samples": len(test),
                "n_seen_in_fit_training": len(seen),
                "n_unseen_to_fit_model": len(unseen),
                "seen_fraction": len(seen) / len(test),
                "unseen_fraction": len(unseen) / len(test),
                "n_shared_with_fit_native_holdout": len(shared),
                "holdout_jaccard": len(shared) / len(union) if union else np.nan,
            })
    return pd.DataFrame(rows)


def _indexed_base(base: pd.DataFrame, options: NestedCVOptions) -> pd.DataFrame:
    required = (options.sample_id_column, options.label_column)
    missing = [c for c in required if c not in base.columns]
    if missing:
        raise CrossRepeatEvaluationError(f"base matrix missing columns: {missing}")
    out = base.copy()
    out[options.sample_id_column] = out[options.sample_id_column].astype(str)
    if out[options.sample_id_column].duplicated().any() or out.isna().any().any():
        raise CrossRepeatEvaluationError("base matrix has duplicate IDs or missing values")
    out[options.label_column] = out[options.label_column].astype(str).str.strip().str.upper()
    expected = {options.positive_label.upper(), options.negative_label.upper()}
    if set(out[options.label_column]) != expected:
        raise CrossRepeatEvaluationError("base labels do not match configured binary labels")
    return out.set_index(options.sample_id_column, drop=False)


def _binary_labels(frame: pd.DataFrame, options: NestedCVOptions) -> np.ndarray:
    y = (frame[options.label_column].astype(str).str.upper() == options.positive_label.upper()).astype(int).to_numpy()
    if set(np.unique(y).tolist()) != {0, 1}:
        raise CrossRepeatEvaluationError("complete holdout must contain both classes")
    return y


def _saved_public(saved: pd.DataFrame, sample_ids: Sequence[str], row_lookup: Mapping[str, int], label: str) -> pd.DataFrame:
    required = {"sample_id", *ALL_PUBLIC_FEATURES}
    missing = sorted(required - set(saved.columns))
    if missing:
        raise CrossRepeatEvaluationError(f"{label} missing columns: {missing}")
    frame = saved[["sample_id", *ALL_PUBLIC_FEATURES]].copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    expected = tuple(map(str, sample_ids))
    if frame["sample_id"].duplicated().any() or set(frame["sample_id"]) != set(expected):
        raise CrossRepeatEvaluationError(f"{label} sample membership mismatch")
    frame = frame.set_index("sample_id").loc[list(expected)].reset_index()
    for column in ALL_PUBLIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(float)).all():
            raise CrossRepeatEvaluationError(f"{label}.{column} contains non-finite values")
    frame.index = [int(row_lookup[sid]) for sid in expected]
    return frame[list(ALL_PUBLIC_FEATURES)]


def _hybrid_public(*, presence: sparse.spmatrix, frequency: sparse.spmatrix,
                   fit_train_rows: np.ndarray, evaluation_rows: np.ndarray,
                   labels: Sequence[object], aa_clone_numbers: Sequence[float],
                   scheme: ThresholdScheme, epsilon: float,
                   saved_train_loo: pd.DataFrame) -> Tuple[pd.DataFrame, Mapping[int, str]]:
    """Assemble public features without passing overlapping rows to the external transform.

    Cross-repeat holdouts can contain samples that participated in the source
    model's training. ``external_public_features`` correctly rejects those rows
    because its reference and target sets must be disjoint. Therefore the target
    holdout is partitioned first:

    * source-training rows replay their saved exact LOO public values;
    * source-unseen rows are transformed from the source-training reference.
    """
    train_set = set(map(int, fit_train_rows))
    ordered_rows = tuple(map(int, evaluation_rows))
    seen_rows = tuple(row for row in ordered_rows if row in train_set)
    unseen_rows = tuple(row for row in ordered_rows if row not in train_set)

    features = pd.DataFrame(
        index=pd.Index(ordered_rows, name="row_index"),
        columns=list(ALL_PUBLIC_FEATURES),
        dtype=float,
    )
    methods: Dict[int, str] = {}

    if seen_rows:
        missing_seen = [row for row in seen_rows if row not in saved_train_loo.index]
        if missing_seen:
            raise CrossRepeatEvaluationError(
                "Saved fitting LOO public values are missing cache rows: "
                f"{missing_seen[:20]}"
            )
        features.loc[list(seen_rows), list(ALL_PUBLIC_FEATURES)] = (
            saved_train_loo.loc[list(seen_rows), list(ALL_PUBLIC_FEATURES)]
            .to_numpy(float)
        )
        methods.update(
            {row: "saved_fit_training_exact_loo" for row in seen_rows}
        )

    if unseen_rows:
        result = external_public_features(
            presence, frequency, fit_train_rows, unseen_rows, labels,
            aa_clone_numbers, scheme, epsilon,
            context="cross_repeat_evaluation_application",
        )
        features.loc[list(unseen_rows), list(ALL_PUBLIC_FEATURES)] = (
            result.features.loc[list(unseen_rows), list(ALL_PUBLIC_FEATURES)]
            .to_numpy(float)
        )
        methods.update(
            {row: "source_training_reference_only" for row in unseen_rows}
        )

    if features.isna().any().any():
        bad = features.columns[features.isna().any()].tolist()
        raise CrossRepeatEvaluationError(
            f"Hybrid public-feature assembly produced missing values: {bad}"
        )
    values = features.to_numpy(float)
    if not np.isfinite(values).all():
        raise CrossRepeatEvaluationError(
            "Hybrid public-feature assembly produced non-finite values."
        )
    return features, MappingProxyType(methods)


def _compare_preprocessing(generated: pd.DataFrame, saved: pd.DataFrame, *, fit_repeat: int,
                           model_name: str, tolerance: float) -> Dict[str, object]:
    cols = ("feature_name", "source_type", "training_mean", "training_sd", "kept_after_zero_variance_filter")
    missing = [c for c in cols if c not in saved.columns]
    if missing:
        raise CrossRepeatEvaluationError(f"saved preprocessing missing columns: {missing}")
    observed = saved.copy()
    if "stage" in observed.columns:
        observed = observed.loc[observed["stage"].astype(str) == "outer_final"]
    if "model" in observed.columns:
        observed = observed.loc[observed["model"].astype(str) == model_name]
    observed, expected = observed[list(cols)].reset_index(drop=True), generated[list(cols)].reset_index(drop=True)
    if len(observed) != len(expected):
        raise CrossRepeatEvaluationError(f"preprocessing row mismatch repeat={fit_repeat}, model={model_name}")
    names = observed["feature_name"].astype(str).tolist() == expected["feature_name"].astype(str).tolist()
    types = observed["source_type"].astype(str).tolist() == expected["source_type"].astype(str).tolist()
    keep = np.array_equal(observed["kept_after_zero_variance_filter"].astype(bool).to_numpy(), expected["kept_after_zero_variance_filter"].astype(bool).to_numpy())
    mean_diff = np.abs(pd.to_numeric(observed["training_mean"], errors="raise").to_numpy(float) - expected["training_mean"].to_numpy(float))
    sd_diff = np.abs(pd.to_numeric(observed["training_sd"], errors="raise").to_numpy(float) - expected["training_sd"].to_numpy(float))
    max_mean = float(mean_diff.max()) if len(mean_diff) else 0.0
    max_sd = float(sd_diff.max()) if len(sd_diff) else 0.0
    passed = names and types and keep and max_mean <= tolerance and max_sd <= tolerance
    if not passed:
        raise CrossRepeatEvaluationError(
            f"preprocessing reproduction failed repeat={fit_repeat}, model={model_name}; "
            f"names={names}, types={types}, keep={keep}, max_mean={max_mean}, max_sd={max_sd}"
        )
    return {
        "fit_repeat": fit_repeat, "model": model_name,
        "n_raw_features": len(expected),
        "n_kept_features": int(expected["kept_after_zero_variance_filter"].astype(bool).sum()),
        "feature_names_match": names, "source_types_match": types,
        "keep_mask_match": keep, "max_training_mean_abs_diff": max_mean,
        "max_training_sd_abs_diff": max_sd, "tolerance": tolerance, "passed": True,
    }


def _load_model(*, fit_repeat: int, model_name: str, specification: ResolvedModelSpecification,
                selected: SelectedModelParameters, train_frame: pd.DataFrame,
                coefficients: pd.DataFrame, preprocessing: pd.DataFrame,
                zero_sd_tolerance: float, tolerance: float) -> Tuple[FrozenLinearModel, Dict[str, object]]:
    preprocessor = fit_preprocessor(train_frame, specification.numeric, specification.categorical,
                                    zero_sd_tolerance=zero_sd_tolerance)
    audit = _compare_preprocessing(preprocessor.audit_frame(), preprocessing,
                                   fit_repeat=fit_repeat, model_name=model_name, tolerance=tolerance)
    rows = coefficients.loc[coefficients["model"].astype(str) == model_name].copy()
    if not {"feature_name", "coefficient"}.issubset(rows.columns):
        raise CrossRepeatEvaluationError("coefficient table missing required columns")
    rows["feature_name"] = rows["feature_name"].astype(str)
    if rows["feature_name"].duplicated().any():
        raise CrossRepeatEvaluationError(f"duplicate coefficients repeat={fit_repeat}, model={model_name}")
    intercept_row = rows.loc[rows["feature_name"] == "__INTERCEPT__"]
    coef_rows = rows.loc[rows["feature_name"] != "__INTERCEPT__"]
    if len(intercept_row) != 1 or tuple(coef_rows["feature_name"]) != tuple(preprocessor.kept_feature_names):
        raise CrossRepeatEvaluationError(f"coefficient/preprocessing order mismatch repeat={fit_repeat}, model={model_name}")
    intercept = float(pd.to_numeric(intercept_row["coefficient"], errors="raise").iloc[0])
    coef = pd.to_numeric(coef_rows["coefficient"], errors="raise").to_numpy(float)
    if not math.isfinite(intercept) or not np.isfinite(coef).all():
        raise CrossRepeatEvaluationError("non-finite frozen coefficients")
    return FrozenLinearModel(model_name, specification.description, preprocessor, intercept, coef,
                             tuple(preprocessor.kept_feature_names), selected), audit


def frozen_probability(model: FrozenLinearModel, frame: pd.DataFrame) -> np.ndarray:
    transformed = model.preprocessor.transform(frame)
    if tuple(transformed.feature_names) != model.feature_names:
        raise CrossRepeatEvaluationError(f"prediction feature order mismatch for {model.model_name}")
    z = model.intercept + transformed.matrix @ model.coefficients
    p = np.empty_like(z, dtype=float)
    positive = z >= 0
    p[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    expz = np.exp(z[~positive])
    p[~positive] = expz / (1.0 + expz)
    if not np.isfinite(p).all():
        raise CrossRepeatEvaluationError("frozen model produced non-finite probabilities")
    return p


def _native_public_audit(generated: pd.DataFrame, saved: pd.DataFrame, *, fit_repeat: int,
                         evaluation_ids: Sequence[str], row_lookup: Mapping[str, int],
                         tolerance: float) -> Dict[str, object]:
    expected = _saved_public(saved, evaluation_ids, row_lookup, f"repeat {fit_repeat} native public")
    rows = [row_lookup[sid] for sid in evaluation_ids]
    diff = np.abs(generated.loc[rows, list(ALL_PUBLIC_FEATURES)].to_numpy(float) - expected.loc[rows, list(ALL_PUBLIC_FEATURES)].to_numpy(float))
    maximum = float(diff.max()) if diff.size else 0.0
    if maximum > tolerance:
        raise CrossRepeatEvaluationError(f"native public reproduction failed repeat={fit_repeat}, max_diff={maximum}")
    return {"fit_repeat": fit_repeat, "evaluation_repeat": fit_repeat,
            "n_samples": len(evaluation_ids), "n_public_features": len(ALL_PUBLIC_FEATURES),
            "max_abs_diff": maximum, "tolerance": tolerance, "passed": True}


def _native_prediction_audit(generated: pd.DataFrame, saved: pd.DataFrame, *, fit_repeat: int,
                             tolerance: float) -> pd.DataFrame:
    required = {"model", "sample_id", "probability_ILD", "threshold"}
    for frame, label in ((generated, "generated"), (saved, "saved")):
        if not required.issubset(frame.columns):
            raise CrossRepeatEvaluationError(f"{label} native predictions missing columns")
    left, right = generated[list(required)].copy(), saved[list(required)].copy()
    for frame in (left, right):
        frame["model"] = frame["model"].astype(str)
        frame["sample_id"] = frame["sample_id"].astype(str)
    merged = left.merge(right, on=["model", "sample_id"], how="outer", validate="one_to_one",
                        suffixes=("_generated", "_saved"), indicator=True)
    if not merged["_merge"].eq("both").all():
        raise CrossRepeatEvaluationError(f"native prediction membership mismatch repeat={fit_repeat}")
    pdiff = np.abs(merged["probability_ILD_generated"].to_numpy(float) - merged["probability_ILD_saved"].to_numpy(float))
    tdiff = np.abs(merged["threshold_generated"].to_numpy(float) - merged["threshold_saved"].to_numpy(float))
    passed = (pdiff <= tolerance) & (tdiff <= tolerance)
    audit = pd.DataFrame({
        "fit_repeat": fit_repeat, "evaluation_repeat": fit_repeat,
        "model": merged["model"], "sample_id": merged["sample_id"],
        "probability_abs_diff": pdiff, "threshold_abs_diff": tdiff,
        "tolerance": tolerance, "passed": passed,
    })
    if not passed.all():
        raise CrossRepeatEvaluationError(f"native prediction reproduction failed: {audit.loc[~audit['passed']].head(10).to_dict('records')}")
    return audit


def _unseen_metrics(predictions: pd.DataFrame, common: Mapping[str, object]) -> Dict[str, object]:
    unseen = predictions.loc[~predictions["was_in_fit_training"].astype(bool)]
    y = unseen["true_label"].to_numpy(int)
    p = unseen["probability_ILD"].to_numpy(float)
    threshold = float(predictions["threshold"].iloc[0])
    row = dict(common)
    row.update({
        "metric_scope": "unseen_only", "n_test": len(unseen),
        "n_RA": int(np.sum(y == 0)), "n_ILD": int(np.sum(y == 1)),
        "positive_prevalence": float(np.mean(y)) if len(y) else np.nan,
    })
    if len(unseen) < 2 or set(np.unique(y).tolist()) != {0, 1}:
        row.update({m: np.nan for m in REPORT_METRICS})
        row.update({"threshold": threshold, "TN": np.nan, "FP": np.nan, "FN": np.nan, "TP": np.nan,
                    "metric_status": "not_estimable_both_classes_required"})
    else:
        row.update(classification_metrics(y, p, threshold, include_brier=True, include_sample_summary=True))
        row["metric_status"] = "estimated"
    return row


def _consistency(predictions: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    rows = []
    for (model, fit_repeat, sample_id), frame in predictions.groupby(["model", "fit_repeat", "sample_id"], sort=False):
        p, t = frame["probability_ILD"].to_numpy(float), frame["threshold"].to_numpy(float)
        pdiff, tdiff = float(p.max() - p.min()), float(t.max() - t.min())
        rows.append({"model": model, "fit_repeat": int(fit_repeat), "sample_id": sample_id,
                     "evaluation_occurrences": len(frame), "max_probability_difference": pdiff,
                     "max_threshold_difference": tdiff, "tolerance": tolerance,
                     "passed": pdiff <= tolerance and tdiff <= tolerance})
    audit = pd.DataFrame(rows)
    if not audit["passed"].all():
        raise CrossRepeatEvaluationError(f"same frozen model produced inconsistent repeated-sample predictions: {audit.loc[~audit['passed']].head(10).to_dict('records')}")
    return audit


def evaluate_cross_repeat_matrix(*, base: pd.DataFrame, outer_assignments: pd.DataFrame,
                                 inner_assignments: pd.DataFrame, row_lookup: Mapping[str, int],
                                 presence: sparse.spmatrix, frequency: sparse.spmatrix,
                                 labels: Sequence[object], aa_clone_numbers: Sequence[float],
                                 model_specifications: Mapping[str, ResolvedModelSpecification],
                                 artifacts_by_repeat: Mapping[int, FrozenTaskArtifacts],
                                 scheme: ThresholdScheme, options: NestedCVOptions,
                                 repeats: int, outer_folds: int, inner_folds: int,
                                 executed_outer_fold: int, selected_models: Optional[Sequence[str]] = None,
                                 reproduction_tolerance: float = 1e-8) -> CrossRepeatEvaluationResult:
    configured = tuple(model_specifications)
    models = configured if selected_models is None else tuple(map(str, selected_models))
    if not models or len(models) != len(set(models)) or set(models) - set(configured):
        raise CrossRepeatEvaluationError("selected models are empty, duplicated, or unknown")
    expected_repeats = set(range(1, int(repeats) + 1))
    if set(artifacts_by_repeat) != expected_repeats:
        raise CrossRepeatEvaluationError("task artifacts do not cover every repeat")
    indexed, base_ids = _indexed_base(base, options), base[options.sample_id_column].astype(str).tolist()
    splits, membership = {}, {}
    for repeat in range(1, int(repeats) + 1):
        split = validate_fixed_assignments(base_ids, outer_assignments, inner_assignments,
                                           outer_repeat=repeat, outer_fold=executed_outer_fold,
                                           outer_folds=outer_folds, inner_folds=inner_folds,
                                           sample_id_column=options.sample_id_column)
        splits[repeat] = split
        membership[repeat] = {"training": split.outer_train_ids, "holdout": split.outer_valid_ids}
    overlap = build_overlap_audit(membership)

    prediction_parts, full_rows, unseen_rows = [], [], []
    prep_rows, native_public_rows, native_pred_parts = [], [], []
    for fit_repeat in range(1, int(repeats) + 1):
        split, artifacts = splits[fit_repeat], artifacts_by_repeat[fit_repeat]
        train_rows = np.asarray([row_lookup[s] for s in split.outer_train_ids], dtype=int)
        train_public = _saved_public(artifacts.outer_train_public, split.outer_train_ids, row_lookup,
                                     f"repeat {fit_repeat} outer-train LOO public")
        train_frame = assemble_model_dataframe(indexed, split.outer_train_ids, train_public, row_lookup)
        frozen_models = {}
        for model_name in models:
            frozen, audit = _load_model(
                fit_repeat=fit_repeat, model_name=model_name,
                specification=model_specifications[model_name], selected=artifacts.selected[model_name],
                train_frame=train_frame, coefficients=artifacts.coefficients,
                preprocessing=artifacts.preprocessing, zero_sd_tolerance=options.zero_sd_tolerance,
                tolerance=reproduction_tolerance,
            )
            frozen_models[model_name], prep_rows = frozen, prep_rows + [audit]
        train_set, native_set = set(split.outer_train_ids), set(split.outer_valid_ids)

        for eval_repeat in range(1, int(repeats) + 1):
            eval_split = splits[eval_repeat]
            eval_ids = tuple(eval_split.outer_valid_ids)
            eval_rows = np.asarray([row_lookup[s] for s in eval_ids], dtype=int)
            public_values, methods = _hybrid_public(
                presence=presence, frequency=frequency, fit_train_rows=train_rows,
                evaluation_rows=eval_rows, labels=labels, aa_clone_numbers=aa_clone_numbers,
                scheme=scheme, epsilon=options.epsilon, saved_train_loo=train_public,
            )
            if fit_repeat == eval_repeat:
                native_public_rows.append(_native_public_audit(
                    public_values, artifacts.outer_validation_public, fit_repeat=fit_repeat,
                    evaluation_ids=eval_ids, row_lookup=row_lookup, tolerance=reproduction_tolerance,
                ))
            eval_frame = assemble_model_dataframe(indexed, eval_ids, public_values, row_lookup)
            y = _binary_labels(eval_frame, options)
            seen = np.asarray([s in train_set for s in eval_ids], dtype=bool)
            pair_parts = []
            for model_name in models:
                frozen = frozen_models[model_name]
                probability = frozen_probability(frozen, eval_frame)
                predicted = apply_threshold(probability, frozen.selected.threshold)
                rows = []
                for sid, truth, prob, pred, cache_row, was_seen in zip(eval_ids, y, probability, predicted, eval_rows, seen):
                    rows.append({
                        "model": model_name, "model_definition": frozen.model_definition,
                        "fit_repeat": fit_repeat, "fit_outer_fold": executed_outer_fold,
                        "evaluation_repeat": eval_repeat, "evaluation_outer_fold": executed_outer_fold,
                        "is_native_pair": fit_repeat == eval_repeat, "sample_id": sid,
                        "true_label": int(truth), "true_cohort": options.positive_label if truth == 1 else options.negative_label,
                        "probability_ILD": float(prob), "threshold": float(frozen.selected.threshold),
                        "predicted_label": int(pred), "predicted_cohort": options.positive_label if pred == 1 else options.negative_label,
                        "evaluation_role": evaluation_role(fit_repeat=fit_repeat, evaluation_repeat=eval_repeat,
                                                           sample_id=sid, fit_training_ids=split.outer_train_ids),
                        "was_in_fit_training": bool(was_seen), "was_in_fit_native_holdout": sid in native_set,
                        "public_generation_method": methods[int(cache_row)],
                        "selected_l1_ratio_alpha": frozen.selected.alpha,
                        "selected_lambda": frozen.selected.lambda_value,
                        "inner_selected_roc_auc": frozen.selected.inner_roc_auc,
                        "inner_selected_pr_auc": frozen.selected.inner_pr_auc,
                        "n_final_predictors": len(frozen.feature_names),
                    })
                pair = pd.DataFrame(rows)
                pair_parts.append(pair)
                common = {
                    "model": model_name, "model_definition": frozen.model_definition,
                    "fit_repeat": fit_repeat, "fit_outer_fold": executed_outer_fold,
                    "evaluation_repeat": eval_repeat, "evaluation_outer_fold": executed_outer_fold,
                    "is_native_pair": fit_repeat == eval_repeat,
                    "n_seen_in_fit_training": int(seen.sum()), "n_unseen_to_fit_model": int((~seen).sum()),
                    "seen_fraction": float(seen.mean()), "unseen_fraction": float((~seen).mean()),
                    "selected_l1_ratio_alpha": frozen.selected.alpha,
                    "selected_lambda": frozen.selected.lambda_value,
                    "inner_selected_roc_auc": frozen.selected.inner_roc_auc,
                    "inner_selected_pr_auc": frozen.selected.inner_pr_auc,
                    "n_final_predictors": len(frozen.feature_names),
                }
                full = dict(common)
                full.update({"metric_scope": "full_requested_holdout", "metric_status": "estimated"})
                full.update(classification_metrics(y, probability, frozen.selected.threshold,
                                                   include_brier=True, include_sample_summary=True))
                full_rows.append(full)
                unseen_rows.append(_unseen_metrics(pair, common))
            pair_frame = pd.concat(pair_parts, ignore_index=True)
            prediction_parts.append(pair_frame)
            if fit_repeat == eval_repeat:
                native_pred_parts.append(_native_prediction_audit(
                    pair_frame[["model", "sample_id", "probability_ILD", "threshold"]],
                    artifacts.native_predictions.loc[artifacts.native_predictions["model"].astype(str).isin(models)],
                    fit_repeat=fit_repeat, tolerance=reproduction_tolerance,
                ))

    predictions = pd.concat(prediction_parts, ignore_index=True)
    full_metrics, unseen_metrics = pd.DataFrame(full_rows), pd.DataFrame(unseen_rows)
    prep_audit, native_public_audit = pd.DataFrame(prep_rows), pd.DataFrame(native_public_rows)
    native_pred_audit = pd.concat(native_pred_parts, ignore_index=True)
    consistency = _consistency(predictions, reproduction_tolerance)
    counts = expected_cross_repeat_counts(repeats, len(models), len(splits[1].outer_valid_ids))
    observed = {
        "full_metric_rows": len(full_metrics), "unseen_only_metric_rows": len(unseen_metrics),
        "prediction_rows": len(predictions), "overlap_rows": len(overlap),
        "native_prediction_rows": len(native_pred_audit),
    }
    for key, value in observed.items():
        if value != counts[key]:
            raise CrossRepeatEvaluationError(f"{key}={value}, expected={counts[key]}")
    if predictions.duplicated(["model", "fit_repeat", "evaluation_repeat", "sample_id"]).any():
        raise CrossRepeatEvaluationError("duplicate cross-repeat prediction keys")
    return CrossRepeatEvaluationResult(predictions, full_metrics, unseen_metrics, overlap, prep_audit,
                                       native_public_audit, native_pred_audit, consistency)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "model"


def write_cross_repeat_evaluation(result: CrossRepeatEvaluationResult, output_dir: Path, *,
                                  experiment_id: str, split_set_id: str, assignment_sha256: str,
                                  expected_repeats: int, expected_models: Sequence[str],
                                  expected_holdout_size: int, reproduction_tolerance: float,
                                  overwrite: bool = False) -> Mapping[str, Path]:
    out = Path(output_dir).expanduser().resolve()
    paths = {
        "predictions": out / "04_cross_repeat_predictions.csv.gz",
        "full_metrics": out / "04_cross_repeat_full_holdout_metrics.csv",
        "unseen_metrics": out / "04_cross_repeat_unseen_only_metrics.csv",
        "overlap": out / "04_cross_repeat_overlap_audit.csv",
        "preprocessing": out / "04_preprocessing_reproduction_audit.csv",
        "native_public": out / "04_native_public_reproduction_audit.csv",
        "native_predictions": out / "04_native_prediction_reproduction_audit.csv",
        "prediction_consistency": out / "04_repeated_sample_prediction_consistency.csv",
        "complete": out / "04_CROSS_REPEAT_EVALUATION_COMPLETE.json",
    }
    matrix_paths = {}
    for scope in ("full", "unseen_only"):
        for model in expected_models:
            for metric in REPORT_METRICS:
                matrix_paths[f"matrix::{scope}::{model}::{metric}"] = out / f"04_matrix_{scope}_{_safe(model)}_{metric}.csv"
    existing = [p for p in (*paths.values(), *matrix_paths.values()) if p.exists()]
    if existing and not overwrite:
        raise FileExistsError("cross-repeat outputs already exist; use --overwrite only for documented rerun:\n" + "\n".join(f"  - {p}" for p in existing))
    out.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(paths["predictions"], index=False, compression="gzip")
    result.full_metrics.to_csv(paths["full_metrics"], index=False)
    result.unseen_only_metrics.to_csv(paths["unseen_metrics"], index=False)
    result.overlap_audit.to_csv(paths["overlap"], index=False)
    result.preprocessing_reproduction_audit.to_csv(paths["preprocessing"], index=False)
    result.native_public_reproduction_audit.to_csv(paths["native_public"], index=False)
    result.native_prediction_reproduction_audit.to_csv(paths["native_predictions"], index=False)
    result.prediction_consistency_audit.to_csv(paths["prediction_consistency"], index=False)
    frames = {"full": result.full_metrics, "unseen_only": result.unseen_only_metrics}
    for scope, frame in frames.items():
        for model in expected_models:
            subset = frame.loc[frame["model"].astype(str) == str(model)]
            for metric in REPORT_METRICS:
                matrix = subset.pivot(index="fit_repeat", columns="evaluation_repeat", values=metric).sort_index().sort_index(axis=1)
                matrix.index.name = "fit_repeat"
                matrix.columns = [f"evaluation_repeat_{int(v):02d}" for v in matrix.columns]
                matrix.reset_index().to_csv(matrix_paths[f"matrix::{scope}::{model}::{metric}"], index=False)
    counts = expected_cross_repeat_counts(expected_repeats, len(tuple(expected_models)), expected_holdout_size)
    all_paths = {**paths, **matrix_paths}
    marker = {
        "status": "COMPLETE", "experiment_id": experiment_id,
        "analysis_mode": "frozen_model_cross_repeat_holdout_matrix",
        "split_set_id": split_set_id, "assignment_sha256": assignment_sha256,
        "expected_repeats": expected_repeats, "expected_models": list(expected_models),
        "expected_holdout_size": expected_holdout_size, "expected_counts": dict(counts),
        "observed_counts": {
            "full_metric_rows": len(result.full_metrics),
            "unseen_only_metric_rows": len(result.unseen_only_metrics),
            "prediction_rows": len(result.predictions), "overlap_rows": len(result.overlap_audit),
            "native_prediction_rows": len(result.native_prediction_reproduction_audit),
        },
        "model_replay": {
            "elastic_net_refit": False, "retuning": False,
            "coefficients_source": "saved_final_task_coefficients",
            "intercept_source": "saved_final_task_coefficients",
            "threshold_source": "saved_selected_task_configuration",
            "preprocessing": "reconstructed_from_source_training_and_verified_against_saved_audit",
            "seen_training_public": "saved_exact_leave_one_out",
            "unseen_public": "source_training_reference_only",
        },
        "reproduction_tolerance": reproduction_tolerance,
        "preprocessing_reproduction_pass": bool(result.preprocessing_reproduction_audit["passed"].all()),
        "native_public_reproduction_pass": bool(result.native_public_reproduction_audit["passed"].all()),
        "native_prediction_reproduction_pass": bool(result.native_prediction_reproduction_audit["passed"].all()),
        "repeated_sample_prediction_consistency_pass": bool(result.prediction_consistency_audit["passed"].all()),
        "primary_result": "full_requested_holdout_n_by_n_matrix",
        "supplementary_result": "unseen_only_n_by_n_matrix",
        "independent_test_read": False, "automatic_final_model_selection": False,
        "output_sha256": {k: _sha256(p) for k, p in all_paths.items() if k != "complete"},
    }
    paths["complete"].write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return MappingProxyType({**paths, **matrix_paths})
