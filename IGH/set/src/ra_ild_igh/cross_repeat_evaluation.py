#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen-model n×n cross-repeat evaluation for IGH repeated holdouts.

This module is an IGH-namespace port of the validated TRB Batch 06 v1.0.1
cross-repeat evaluator. It replays saved repeat-specific final models without
retuning or refitting and evaluates every fit repeat against every frozen
holdout repeat.
"""
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
    "roc_auc",
    "pr_auc",
    "accuracy",
    "sensitivity_recall",
    "specificity",
    "precision",
    "f1",
    "brier_score",
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


def expected_cross_repeat_counts(
    repeats: int,
    models: int,
    holdout_size: int,
) -> Mapping[str, int]:
    values = (repeats, models, holdout_size)
    if any(isinstance(value, bool) or int(value) < 1 for value in values):
        raise CrossRepeatEvaluationError(
            "repeats, models, and holdout_size must be positive integers"
        )
    repeat_count, model_count, holdout_count = map(int, values)
    return MappingProxyType(
        {
            "model_instances": repeat_count * model_count,
            "evaluation_pairs_per_model": repeat_count * repeat_count,
            "full_metric_rows": repeat_count * repeat_count * model_count,
            "unseen_only_metric_rows": repeat_count * repeat_count * model_count,
            "prediction_rows": repeat_count
            * repeat_count
            * model_count
            * holdout_count,
            "overlap_rows": repeat_count * repeat_count,
            "native_prediction_rows": repeat_count * model_count * holdout_count,
        }
    )


def evaluation_role(
    *,
    fit_repeat: int,
    evaluation_repeat: int,
    sample_id: str,
    fit_training_ids: Sequence[str],
) -> str:
    if int(fit_repeat) == int(evaluation_repeat):
        return "native_holdout"
    if str(sample_id) in set(map(str, fit_training_ids)):
        return "cross_holdout_seen_training"
    return "cross_holdout_unseen"


def _ids(values: Sequence[object], label: str) -> Tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result or len(result) != len(set(result)):
        raise CrossRepeatEvaluationError(f"{label} is empty or duplicated")
    return result


def build_overlap_audit(
    split_membership: Mapping[int, Mapping[str, Sequence[str]]],
) -> pd.DataFrame:
    repeats = tuple(sorted(int(value) for value in split_membership))
    if repeats != tuple(range(1, len(repeats) + 1)):
        raise CrossRepeatEvaluationError("repeat IDs must be consecutive starting at 1")

    rows: List[Dict[str, object]] = []
    complete_cohort: Optional[set[str]] = None
    for fit_repeat in repeats:
        training = set(
            _ids(
                split_membership[fit_repeat]["training"],
                f"repeat {fit_repeat} training",
            )
        )
        native_holdout = set(
            _ids(
                split_membership[fit_repeat]["holdout"],
                f"repeat {fit_repeat} holdout",
            )
        )
        if training & native_holdout:
            raise CrossRepeatEvaluationError(
                f"repeat {fit_repeat} training/holdout overlap"
            )
        cohort = training | native_holdout
        if complete_cohort is None:
            complete_cohort = cohort
        elif cohort != complete_cohort:
            raise CrossRepeatEvaluationError("all repeats must use the same cohort")

        for evaluation_repeat in repeats:
            evaluation_holdout = set(
                _ids(
                    split_membership[evaluation_repeat]["holdout"],
                    f"repeat {evaluation_repeat} holdout",
                )
            )
            seen = evaluation_holdout & training
            unseen = evaluation_holdout - training
            shared = evaluation_holdout & native_holdout
            union = evaluation_holdout | native_holdout
            rows.append(
                {
                    "fit_repeat": fit_repeat,
                    "evaluation_repeat": evaluation_repeat,
                    "is_native_pair": fit_repeat == evaluation_repeat,
                    "n_evaluation_samples": len(evaluation_holdout),
                    "n_seen_in_fit_training": len(seen),
                    "n_unseen_to_fit_model": len(unseen),
                    "seen_fraction": len(seen) / len(evaluation_holdout),
                    "unseen_fraction": len(unseen) / len(evaluation_holdout),
                    "n_shared_with_fit_native_holdout": len(shared),
                    "holdout_jaccard": len(shared) / len(union) if union else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _indexed_base(base: pd.DataFrame, options: NestedCVOptions) -> pd.DataFrame:
    required = (options.sample_id_column, options.label_column)
    missing = [column for column in required if column not in base.columns]
    if missing:
        raise CrossRepeatEvaluationError(f"base matrix missing columns: {missing}")

    result = base.copy()
    result[options.sample_id_column] = result[options.sample_id_column].astype(str)
    if result[options.sample_id_column].duplicated().any() or result.isna().any().any():
        raise CrossRepeatEvaluationError(
            "base matrix has duplicate IDs or missing values"
        )
    result[options.label_column] = (
        result[options.label_column].astype(str).str.strip().str.upper()
    )
    expected = {options.positive_label.upper(), options.negative_label.upper()}
    if set(result[options.label_column]) != expected:
        raise CrossRepeatEvaluationError(
            "base labels do not match configured binary labels"
        )
    return result.set_index(options.sample_id_column, drop=False)


def _binary_labels(frame: pd.DataFrame, options: NestedCVOptions) -> np.ndarray:
    labels = (
        frame[options.label_column].astype(str).str.upper()
        == options.positive_label.upper()
    ).astype(int).to_numpy()
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise CrossRepeatEvaluationError("complete holdout must contain both classes")
    return labels


def _saved_public(
    saved: pd.DataFrame,
    sample_ids: Sequence[str],
    row_lookup: Mapping[str, int],
    label: str,
) -> pd.DataFrame:
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
            raise CrossRepeatEvaluationError(
                f"{label}.{column} contains non-finite values"
            )
    frame.index = [int(row_lookup[sample_id]) for sample_id in expected]
    return frame[list(ALL_PUBLIC_FEATURES)]


def _hybrid_public(
    *,
    presence: sparse.spmatrix,
    frequency: sparse.spmatrix,
    fit_train_rows: np.ndarray,
    evaluation_rows: np.ndarray,
    labels: Sequence[object],
    aa_clone_numbers: Sequence[float],
    scheme: ThresholdScheme,
    epsilon: float,
    saved_train_loo: pd.DataFrame,
) -> Tuple[pd.DataFrame, Mapping[int, str]]:
    """Assemble public features while preserving reference/target disjointness.

    Source-training samples replay their saved exact leave-one-out public values.
    Only source-unseen samples are passed to the external transformation using the
    source training partition as reference. This includes the validated TRB
    v1.0.1 hotfix and avoids reference/target overlap.
    """
    training_set = set(map(int, fit_train_rows))
    ordered_rows = tuple(map(int, evaluation_rows))
    seen_rows = tuple(row for row in ordered_rows if row in training_set)
    unseen_rows = tuple(row for row in ordered_rows if row not in training_set)

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
        external = external_public_features(
            presence,
            frequency,
            fit_train_rows,
            unseen_rows,
            labels,
            aa_clone_numbers,
            scheme,
            epsilon,
            context="cross_repeat_evaluation_application",
        )
        features.loc[list(unseen_rows), list(ALL_PUBLIC_FEATURES)] = (
            external.features.loc[list(unseen_rows), list(ALL_PUBLIC_FEATURES)]
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
    if not np.isfinite(features.to_numpy(float)).all():
        raise CrossRepeatEvaluationError(
            "Hybrid public-feature assembly produced non-finite values"
        )
    return features, MappingProxyType(methods)


def _compare_preprocessing(
    generated: pd.DataFrame,
    saved: pd.DataFrame,
    *,
    fit_repeat: int,
    model_name: str,
    tolerance: float,
) -> Dict[str, object]:
    columns = (
        "feature_name",
        "source_type",
        "training_mean",
        "training_sd",
        "kept_after_zero_variance_filter",
    )
    missing = [column for column in columns if column not in saved.columns]
    if missing:
        raise CrossRepeatEvaluationError(
            f"saved preprocessing missing columns: {missing}"
        )

    observed = saved.copy()
    if "stage" in observed.columns:
        observed = observed.loc[observed["stage"].astype(str) == "outer_final"]
    if "model" in observed.columns:
        observed = observed.loc[observed["model"].astype(str) == model_name]
    observed = observed[list(columns)].reset_index(drop=True)
    expected = generated[list(columns)].reset_index(drop=True)
    if len(observed) != len(expected):
        raise CrossRepeatEvaluationError(
            f"preprocessing row mismatch repeat={fit_repeat}, model={model_name}"
        )

    names_match = (
        observed["feature_name"].astype(str).tolist()
        == expected["feature_name"].astype(str).tolist()
    )
    source_types_match = (
        observed["source_type"].astype(str).tolist()
        == expected["source_type"].astype(str).tolist()
    )
    keep_match = np.array_equal(
        observed["kept_after_zero_variance_filter"].astype(bool).to_numpy(),
        expected["kept_after_zero_variance_filter"].astype(bool).to_numpy(),
    )

    observed_mean = pd.to_numeric(observed["training_mean"], errors="raise").to_numpy(float)
    expected_mean = pd.to_numeric(expected["training_mean"], errors="raise").to_numpy(float)
    observed_sd = pd.to_numeric(observed["training_sd"], errors="raise").to_numpy(float)
    expected_sd = pd.to_numeric(expected["training_sd"], errors="raise").to_numpy(float)

    mean_difference = np.abs(observed_mean - expected_mean)
    sd_difference = np.abs(observed_sd - expected_sd)
    # Categorical audit rows may use NaN for numeric statistics in both frames.
    mean_difference = np.nan_to_num(mean_difference, nan=0.0, posinf=np.inf, neginf=np.inf)
    sd_difference = np.nan_to_num(sd_difference, nan=0.0, posinf=np.inf, neginf=np.inf)
    max_mean = float(mean_difference.max()) if len(mean_difference) else 0.0
    max_sd = float(sd_difference.max()) if len(sd_difference) else 0.0

    passed = (
        names_match
        and source_types_match
        and keep_match
        and max_mean <= tolerance
        and max_sd <= tolerance
    )
    if not passed:
        raise CrossRepeatEvaluationError(
            f"preprocessing reproduction failed repeat={fit_repeat}, "
            f"model={model_name}; names={names_match}, "
            f"types={source_types_match}, keep={keep_match}, "
            f"max_mean={max_mean}, max_sd={max_sd}"
        )
    return {
        "fit_repeat": fit_repeat,
        "model": model_name,
        "n_raw_features": len(expected),
        "n_kept_features": int(
            expected["kept_after_zero_variance_filter"].astype(bool).sum()
        ),
        "feature_names_match": names_match,
        "source_types_match": source_types_match,
        "keep_mask_match": keep_match,
        "max_training_mean_abs_diff": max_mean,
        "max_training_sd_abs_diff": max_sd,
        "tolerance": tolerance,
        "passed": True,
    }


def _load_model(
    *,
    fit_repeat: int,
    model_name: str,
    specification: ResolvedModelSpecification,
    selected: SelectedModelParameters,
    train_frame: pd.DataFrame,
    coefficients: pd.DataFrame,
    preprocessing: pd.DataFrame,
    zero_sd_tolerance: float,
    tolerance: float,
) -> Tuple[FrozenLinearModel, Dict[str, object]]:
    preprocessor = fit_preprocessor(
        train_frame,
        specification.numeric,
        specification.categorical,
        zero_sd_tolerance=zero_sd_tolerance,
    )
    audit = _compare_preprocessing(
        preprocessor.audit_frame(),
        preprocessing,
        fit_repeat=fit_repeat,
        model_name=model_name,
        tolerance=tolerance,
    )

    if "model" not in coefficients.columns:
        raise CrossRepeatEvaluationError("coefficient table missing model column")
    rows = coefficients.loc[coefficients["model"].astype(str) == model_name].copy()
    if not {"feature_name", "coefficient"}.issubset(rows.columns):
        raise CrossRepeatEvaluationError(
            "coefficient table missing required columns"
        )
    rows["feature_name"] = rows["feature_name"].astype(str)
    if rows["feature_name"].duplicated().any():
        raise CrossRepeatEvaluationError(
            f"duplicate coefficients repeat={fit_repeat}, model={model_name}"
        )

    intercept_row = rows.loc[rows["feature_name"] == "__INTERCEPT__"]
    coefficient_rows = rows.loc[rows["feature_name"] != "__INTERCEPT__"]
    if len(intercept_row) != 1 or tuple(coefficient_rows["feature_name"]) != tuple(
        preprocessor.kept_feature_names
    ):
        raise CrossRepeatEvaluationError(
            f"coefficient/preprocessing order mismatch repeat={fit_repeat}, "
            f"model={model_name}"
        )

    intercept = float(
        pd.to_numeric(intercept_row["coefficient"], errors="raise").iloc[0]
    )
    coefficient_vector = pd.to_numeric(
        coefficient_rows["coefficient"], errors="raise"
    ).to_numpy(float)
    if not math.isfinite(intercept) or not np.isfinite(coefficient_vector).all():
        raise CrossRepeatEvaluationError("non-finite frozen coefficients")

    return (
        FrozenLinearModel(
            model_name=model_name,
            model_definition=specification.description,
            preprocessor=preprocessor,
            intercept=intercept,
            coefficients=coefficient_vector,
            feature_names=tuple(preprocessor.kept_feature_names),
            selected=selected,
        ),
        audit,
    )


def frozen_probability(model: FrozenLinearModel, frame: pd.DataFrame) -> np.ndarray:
    transformed = model.preprocessor.transform(frame)
    if tuple(transformed.feature_names) != model.feature_names:
        raise CrossRepeatEvaluationError(
            f"prediction feature order mismatch for {model.model_name}"
        )
    linear_predictor = model.intercept + transformed.matrix @ model.coefficients
    probability = np.empty_like(linear_predictor, dtype=float)
    nonnegative = linear_predictor >= 0
    probability[nonnegative] = 1.0 / (
        1.0 + np.exp(-linear_predictor[nonnegative])
    )
    exponent = np.exp(linear_predictor[~nonnegative])
    probability[~nonnegative] = exponent / (1.0 + exponent)
    if not np.isfinite(probability).all():
        raise CrossRepeatEvaluationError(
            "frozen model produced non-finite probabilities"
        )
    return probability


def _native_public_audit(
    generated: pd.DataFrame,
    saved: pd.DataFrame,
    *,
    fit_repeat: int,
    evaluation_ids: Sequence[str],
    row_lookup: Mapping[str, int],
    tolerance: float,
) -> Dict[str, object]:
    expected = _saved_public(
        saved,
        evaluation_ids,
        row_lookup,
        f"repeat {fit_repeat} native public",
    )
    rows = [row_lookup[sample_id] for sample_id in evaluation_ids]
    difference = np.abs(
        generated.loc[rows, list(ALL_PUBLIC_FEATURES)].to_numpy(float)
        - expected.loc[rows, list(ALL_PUBLIC_FEATURES)].to_numpy(float)
    )
    maximum = float(difference.max()) if difference.size else 0.0
    if maximum > tolerance:
        raise CrossRepeatEvaluationError(
            f"native public reproduction failed repeat={fit_repeat}, "
            f"max_diff={maximum}"
        )
    return {
        "fit_repeat": fit_repeat,
        "evaluation_repeat": fit_repeat,
        "n_samples": len(evaluation_ids),
        "n_public_features": len(ALL_PUBLIC_FEATURES),
        "max_abs_diff": maximum,
        "tolerance": tolerance,
        "passed": True,
    }


def _native_prediction_audit(
    generated: pd.DataFrame,
    saved: pd.DataFrame,
    *,
    fit_repeat: int,
    tolerance: float,
) -> pd.DataFrame:
    required = {"model", "sample_id", "probability_ILD", "threshold"}
    for frame, label in ((generated, "generated"), (saved, "saved")):
        if not required.issubset(frame.columns):
            raise CrossRepeatEvaluationError(
                f"{label} native predictions missing columns"
            )

    columns = ["model", "sample_id", "probability_ILD", "threshold"]
    left = generated[columns].copy()
    right = saved[columns].copy()
    for frame in (left, right):
        frame["model"] = frame["model"].astype(str)
        frame["sample_id"] = frame["sample_id"].astype(str)

    merged = left.merge(
        right,
        on=["model", "sample_id"],
        how="outer",
        validate="one_to_one",
        suffixes=("_generated", "_saved"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise CrossRepeatEvaluationError(
            f"native prediction membership mismatch repeat={fit_repeat}"
        )

    probability_difference = np.abs(
        merged["probability_ILD_generated"].to_numpy(float)
        - merged["probability_ILD_saved"].to_numpy(float)
    )
    threshold_difference = np.abs(
        merged["threshold_generated"].to_numpy(float)
        - merged["threshold_saved"].to_numpy(float)
    )
    passed = (probability_difference <= tolerance) & (
        threshold_difference <= tolerance
    )
    audit = pd.DataFrame(
        {
            "fit_repeat": fit_repeat,
            "evaluation_repeat": fit_repeat,
            "model": merged["model"],
            "sample_id": merged["sample_id"],
            "probability_abs_diff": probability_difference,
            "threshold_abs_diff": threshold_difference,
            "tolerance": tolerance,
            "passed": passed,
        }
    )
    if not passed.all():
        raise CrossRepeatEvaluationError(
            "native prediction reproduction failed: "
            f"{audit.loc[~audit['passed']].head(10).to_dict('records')}"
        )
    return audit


def _unseen_metrics(
    predictions: pd.DataFrame,
    common: Mapping[str, object],
) -> Dict[str, object]:
    unseen = predictions.loc[~predictions["was_in_fit_training"].astype(bool)]
    labels = unseen["true_label"].to_numpy(int)
    probability = unseen["probability_ILD"].to_numpy(float)
    threshold = float(predictions["threshold"].iloc[0])

    row = dict(common)
    row.update(
        {
            "metric_scope": "unseen_only",
            "n_test": len(unseen),
            "n_RA": int(np.sum(labels == 0)),
            "n_ILD": int(np.sum(labels == 1)),
            "positive_prevalence": (
                float(np.mean(labels)) if len(labels) else np.nan
            ),
        }
    )
    if len(unseen) < 2 or set(np.unique(labels).tolist()) != {0, 1}:
        row.update({metric: np.nan for metric in REPORT_METRICS})
        row.update(
            {
                "threshold": threshold,
                "TN": np.nan,
                "FP": np.nan,
                "FN": np.nan,
                "TP": np.nan,
                "metric_status": "not_estimable_both_classes_required",
            }
        )
    else:
        row.update(
            classification_metrics(
                labels,
                probability,
                threshold,
                include_brier=True,
                include_sample_summary=True,
            )
        )
        row["metric_status"] = "estimated"
    return row


def _consistency(predictions: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (model, fit_repeat, sample_id), frame in predictions.groupby(
        ["model", "fit_repeat", "sample_id"],
        sort=False,
    ):
        probabilities = frame["probability_ILD"].to_numpy(float)
        thresholds = frame["threshold"].to_numpy(float)
        probability_difference = float(
            probabilities.max() - probabilities.min()
        )
        threshold_difference = float(thresholds.max() - thresholds.min())
        rows.append(
            {
                "model": model,
                "fit_repeat": int(fit_repeat),
                "sample_id": sample_id,
                "evaluation_occurrences": len(frame),
                "max_probability_difference": probability_difference,
                "max_threshold_difference": threshold_difference,
                "tolerance": tolerance,
                "passed": probability_difference <= tolerance
                and threshold_difference <= tolerance,
            }
        )
    audit = pd.DataFrame(rows)
    if not audit["passed"].all():
        raise CrossRepeatEvaluationError(
            "same frozen model produced inconsistent repeated-sample "
            "predictions: "
            f"{audit.loc[~audit['passed']].head(10).to_dict('records')}"
        )
    return audit


def evaluate_cross_repeat_matrix(
    *,
    base: pd.DataFrame,
    outer_assignments: pd.DataFrame,
    inner_assignments: pd.DataFrame,
    row_lookup: Mapping[str, int],
    presence: sparse.spmatrix,
    frequency: sparse.spmatrix,
    labels: Sequence[object],
    aa_clone_numbers: Sequence[float],
    model_specifications: Mapping[str, ResolvedModelSpecification],
    artifacts_by_repeat: Mapping[int, FrozenTaskArtifacts],
    scheme: ThresholdScheme,
    options: NestedCVOptions,
    repeats: int,
    outer_folds: int,
    inner_folds: int,
    executed_outer_fold: int,
    selected_models: Optional[Sequence[str]] = None,
    reproduction_tolerance: float = 1.0e-8,
) -> CrossRepeatEvaluationResult:
    configured_models = tuple(model_specifications)
    models = (
        configured_models
        if selected_models is None
        else tuple(map(str, selected_models))
    )
    if (
        not models
        or len(models) != len(set(models))
        or set(models) - set(configured_models)
    ):
        raise CrossRepeatEvaluationError(
            "selected models are empty, duplicated, or unknown"
        )

    expected_repeats = set(range(1, int(repeats) + 1))
    if set(artifacts_by_repeat) != expected_repeats:
        raise CrossRepeatEvaluationError(
            "task artifacts do not cover every repeat"
        )

    indexed = _indexed_base(base, options)
    base_ids = base[options.sample_id_column].astype(str).tolist()
    splits: Dict[int, object] = {}
    membership: Dict[int, Dict[str, Sequence[str]]] = {}
    for repeat in range(1, int(repeats) + 1):
        split = validate_fixed_assignments(
            base_ids,
            outer_assignments,
            inner_assignments,
            outer_repeat=repeat,
            outer_fold=executed_outer_fold,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            sample_id_column=options.sample_id_column,
        )
        splits[repeat] = split
        membership[repeat] = {
            "training": split.outer_train_ids,
            "holdout": split.outer_valid_ids,
        }
    overlap = build_overlap_audit(membership)

    prediction_parts: List[pd.DataFrame] = []
    full_rows: List[Dict[str, object]] = []
    unseen_rows: List[Dict[str, object]] = []
    preprocessing_rows: List[Dict[str, object]] = []
    native_public_rows: List[Dict[str, object]] = []
    native_prediction_parts: List[pd.DataFrame] = []

    for fit_repeat in range(1, int(repeats) + 1):
        split = splits[fit_repeat]
        artifacts = artifacts_by_repeat[fit_repeat]
        train_rows = np.asarray(
            [row_lookup[sample_id] for sample_id in split.outer_train_ids],
            dtype=int,
        )
        train_public = _saved_public(
            artifacts.outer_train_public,
            split.outer_train_ids,
            row_lookup,
            f"repeat {fit_repeat} outer-train LOO public",
        )
        train_frame = assemble_model_dataframe(
            indexed,
            split.outer_train_ids,
            train_public,
            row_lookup,
        )

        frozen_models: Dict[str, FrozenLinearModel] = {}
        for model_name in models:
            if model_name not in artifacts.selected:
                raise CrossRepeatEvaluationError(
                    f"repeat {fit_repeat} lacks selected parameters for {model_name}"
                )
            frozen, audit = _load_model(
                fit_repeat=fit_repeat,
                model_name=model_name,
                specification=model_specifications[model_name],
                selected=artifacts.selected[model_name],
                train_frame=train_frame,
                coefficients=artifacts.coefficients,
                preprocessing=artifacts.preprocessing,
                zero_sd_tolerance=options.zero_sd_tolerance,
                tolerance=reproduction_tolerance,
            )
            frozen_models[model_name] = frozen
            preprocessing_rows.append(audit)

        training_set = set(split.outer_train_ids)
        native_holdout_set = set(split.outer_valid_ids)

        for evaluation_repeat in range(1, int(repeats) + 1):
            evaluation_split = splits[evaluation_repeat]
            evaluation_ids = tuple(evaluation_split.outer_valid_ids)
            evaluation_rows = np.asarray(
                [row_lookup[sample_id] for sample_id in evaluation_ids],
                dtype=int,
            )
            public_values, generation_methods = _hybrid_public(
                presence=presence,
                frequency=frequency,
                fit_train_rows=train_rows,
                evaluation_rows=evaluation_rows,
                labels=labels,
                aa_clone_numbers=aa_clone_numbers,
                scheme=scheme,
                epsilon=options.epsilon,
                saved_train_loo=train_public,
            )
            if fit_repeat == evaluation_repeat:
                native_public_rows.append(
                    _native_public_audit(
                        public_values,
                        artifacts.outer_validation_public,
                        fit_repeat=fit_repeat,
                        evaluation_ids=evaluation_ids,
                        row_lookup=row_lookup,
                        tolerance=reproduction_tolerance,
                    )
                )

            evaluation_frame = assemble_model_dataframe(
                indexed,
                evaluation_ids,
                public_values,
                row_lookup,
            )
            true_labels = _binary_labels(evaluation_frame, options)
            was_seen = np.asarray(
                [sample_id in training_set for sample_id in evaluation_ids],
                dtype=bool,
            )

            pair_parts: List[pd.DataFrame] = []
            for model_name in models:
                frozen = frozen_models[model_name]
                probability = frozen_probability(frozen, evaluation_frame)
                predicted = apply_threshold(
                    probability,
                    frozen.selected.threshold,
                )
                rows: List[Dict[str, object]] = []
                for (
                    sample_id,
                    truth,
                    probability_value,
                    predicted_value,
                    cache_row,
                    seen_value,
                ) in zip(
                    evaluation_ids,
                    true_labels,
                    probability,
                    predicted,
                    evaluation_rows,
                    was_seen,
                ):
                    rows.append(
                        {
                            "model": model_name,
                            "model_definition": frozen.model_definition,
                            "fit_repeat": fit_repeat,
                            "fit_outer_fold": executed_outer_fold,
                            "evaluation_repeat": evaluation_repeat,
                            "evaluation_outer_fold": executed_outer_fold,
                            "is_native_pair": fit_repeat == evaluation_repeat,
                            "sample_id": sample_id,
                            "true_label": int(truth),
                            "true_cohort": (
                                options.positive_label
                                if truth == 1
                                else options.negative_label
                            ),
                            "probability_ILD": float(probability_value),
                            "threshold": float(frozen.selected.threshold),
                            "predicted_label": int(predicted_value),
                            "predicted_cohort": (
                                options.positive_label
                                if predicted_value == 1
                                else options.negative_label
                            ),
                            "evaluation_role": evaluation_role(
                                fit_repeat=fit_repeat,
                                evaluation_repeat=evaluation_repeat,
                                sample_id=sample_id,
                                fit_training_ids=split.outer_train_ids,
                            ),
                            "was_in_fit_training": bool(seen_value),
                            "was_in_fit_native_holdout": (
                                sample_id in native_holdout_set
                            ),
                            "public_generation_method": generation_methods[
                                int(cache_row)
                            ],
                            "selected_l1_ratio_alpha": frozen.selected.alpha,
                            "selected_lambda": frozen.selected.lambda_value,
                            "inner_selected_roc_auc": (
                                frozen.selected.inner_roc_auc
                            ),
                            "inner_selected_pr_auc": (
                                frozen.selected.inner_pr_auc
                            ),
                            "n_final_predictors": len(frozen.feature_names),
                        }
                    )
                pair = pd.DataFrame(rows)
                pair_parts.append(pair)

                common = {
                    "model": model_name,
                    "model_definition": frozen.model_definition,
                    "fit_repeat": fit_repeat,
                    "fit_outer_fold": executed_outer_fold,
                    "evaluation_repeat": evaluation_repeat,
                    "evaluation_outer_fold": executed_outer_fold,
                    "is_native_pair": fit_repeat == evaluation_repeat,
                    "n_seen_in_fit_training": int(was_seen.sum()),
                    "n_unseen_to_fit_model": int((~was_seen).sum()),
                    "seen_fraction": float(was_seen.mean()),
                    "unseen_fraction": float((~was_seen).mean()),
                    "selected_l1_ratio_alpha": frozen.selected.alpha,
                    "selected_lambda": frozen.selected.lambda_value,
                    "inner_selected_roc_auc": frozen.selected.inner_roc_auc,
                    "inner_selected_pr_auc": frozen.selected.inner_pr_auc,
                    "n_final_predictors": len(frozen.feature_names),
                }
                full = dict(common)
                full.update(
                    {
                        "metric_scope": "full_requested_holdout",
                        "metric_status": "estimated",
                    }
                )
                full.update(
                    classification_metrics(
                        true_labels,
                        probability,
                        frozen.selected.threshold,
                        include_brier=True,
                        include_sample_summary=True,
                    )
                )
                full_rows.append(full)
                unseen_rows.append(_unseen_metrics(pair, common))

            pair_frame = pd.concat(pair_parts, ignore_index=True)
            prediction_parts.append(pair_frame)
            if fit_repeat == evaluation_repeat:
                native_prediction_parts.append(
                    _native_prediction_audit(
                        pair_frame[
                            ["model", "sample_id", "probability_ILD", "threshold"]
                        ],
                        artifacts.native_predictions.loc[
                            artifacts.native_predictions["model"]
                            .astype(str)
                            .isin(models)
                        ],
                        fit_repeat=fit_repeat,
                        tolerance=reproduction_tolerance,
                    )
                )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    full_metrics = pd.DataFrame(full_rows)
    unseen_metrics = pd.DataFrame(unseen_rows)
    preprocessing_audit = pd.DataFrame(preprocessing_rows)
    native_public_audit = pd.DataFrame(native_public_rows)
    native_prediction_audit = pd.concat(
        native_prediction_parts,
        ignore_index=True,
    )
    consistency_audit = _consistency(
        predictions,
        reproduction_tolerance,
    )

    counts = expected_cross_repeat_counts(
        repeats,
        len(models),
        len(splits[1].outer_valid_ids),
    )
    observed_counts = {
        "full_metric_rows": len(full_metrics),
        "unseen_only_metric_rows": len(unseen_metrics),
        "prediction_rows": len(predictions),
        "overlap_rows": len(overlap),
        "native_prediction_rows": len(native_prediction_audit),
    }
    for key, observed in observed_counts.items():
        if observed != counts[key]:
            raise CrossRepeatEvaluationError(
                f"{key}={observed}, expected={counts[key]}"
            )
    if predictions.duplicated(
        ["model", "fit_repeat", "evaluation_repeat", "sample_id"]
    ).any():
        raise CrossRepeatEvaluationError(
            "duplicate cross-repeat prediction keys"
        )

    return CrossRepeatEvaluationResult(
        predictions=predictions,
        full_metrics=full_metrics,
        unseen_only_metrics=unseen_metrics,
        overlap_audit=overlap,
        preprocessing_reproduction_audit=preprocessing_audit,
        native_public_reproduction_audit=native_public_audit,
        native_prediction_reproduction_audit=native_prediction_audit,
        prediction_consistency_audit=consistency_audit,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "model"


def write_cross_repeat_evaluation(
    result: CrossRepeatEvaluationResult,
    output_dir: Path,
    *,
    experiment_id: str,
    split_set_id: str,
    assignment_sha256: str,
    expected_repeats: int,
    expected_models: Sequence[str],
    expected_holdout_size: int,
    reproduction_tolerance: float,
    overwrite: bool = False,
) -> Mapping[str, Path]:
    output = Path(output_dir).expanduser().resolve()
    paths = {
        "predictions": output / "04_cross_repeat_predictions.csv.gz",
        "full_metrics": output / "04_cross_repeat_full_holdout_metrics.csv",
        "unseen_metrics": output / "04_cross_repeat_unseen_only_metrics.csv",
        "overlap": output / "04_cross_repeat_overlap_audit.csv",
        "preprocessing": output / "04_preprocessing_reproduction_audit.csv",
        "native_public": output / "04_native_public_reproduction_audit.csv",
        "native_predictions": output
        / "04_native_prediction_reproduction_audit.csv",
        "prediction_consistency": output
        / "04_repeated_sample_prediction_consistency.csv",
        "complete": output / "04_CROSS_REPEAT_EVALUATION_COMPLETE.json",
    }
    matrix_paths: Dict[str, Path] = {}
    for scope in ("full", "unseen_only"):
        for model in expected_models:
            for metric in REPORT_METRICS:
                matrix_paths[f"matrix::{scope}::{model}::{metric}"] = (
                    output
                    / f"04_matrix_{scope}_{_safe(model)}_{metric}.csv"
                )

    existing = [
        path
        for path in (*paths.values(), *matrix_paths.values())
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "cross-repeat outputs already exist; use --overwrite only for a "
            "documented rerun:\n"
            + "\n".join(f"  - {path}" for path in existing)
        )

    output.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(paths["predictions"], index=False, compression="gzip")
    result.full_metrics.to_csv(paths["full_metrics"], index=False)
    result.unseen_only_metrics.to_csv(paths["unseen_metrics"], index=False)
    result.overlap_audit.to_csv(paths["overlap"], index=False)
    result.preprocessing_reproduction_audit.to_csv(
        paths["preprocessing"], index=False
    )
    result.native_public_reproduction_audit.to_csv(
        paths["native_public"], index=False
    )
    result.native_prediction_reproduction_audit.to_csv(
        paths["native_predictions"], index=False
    )
    result.prediction_consistency_audit.to_csv(
        paths["prediction_consistency"], index=False
    )

    frames = {
        "full": result.full_metrics,
        "unseen_only": result.unseen_only_metrics,
    }
    for scope, frame in frames.items():
        for model in expected_models:
            subset = frame.loc[frame["model"].astype(str) == str(model)]
            for metric in REPORT_METRICS:
                matrix = (
                    subset.pivot(
                        index="fit_repeat",
                        columns="evaluation_repeat",
                        values=metric,
                    )
                    .sort_index()
                    .sort_index(axis=1)
                )
                matrix.index.name = "fit_repeat"
                matrix.columns = [
                    f"evaluation_repeat_{int(value):02d}"
                    for value in matrix.columns
                ]
                matrix.reset_index().to_csv(
                    matrix_paths[f"matrix::{scope}::{model}::{metric}"],
                    index=False,
                )

    counts = expected_cross_repeat_counts(
        expected_repeats,
        len(tuple(expected_models)),
        expected_holdout_size,
    )
    all_paths = {**paths, **matrix_paths}
    marker = {
        "status": "COMPLETE",
        "receptor": "IGH",
        "experiment_id": experiment_id,
        "analysis_mode": "frozen_model_cross_repeat_holdout_matrix",
        "port_source": "validated_TRB_Batch06_v1.0.1",
        "split_set_id": split_set_id,
        "assignment_sha256": assignment_sha256,
        "expected_repeats": expected_repeats,
        "expected_models": list(expected_models),
        "expected_holdout_size": expected_holdout_size,
        "expected_counts": dict(counts),
        "observed_counts": {
            "full_metric_rows": len(result.full_metrics),
            "unseen_only_metric_rows": len(result.unseen_only_metrics),
            "prediction_rows": len(result.predictions),
            "overlap_rows": len(result.overlap_audit),
            "native_prediction_rows": len(
                result.native_prediction_reproduction_audit
            ),
        },
        "model_replay": {
            "elastic_net_refit": False,
            "retuning": False,
            "coefficients_source": "saved_final_task_coefficients",
            "intercept_source": "saved_final_task_coefficients",
            "threshold_source": "saved_selected_task_configuration",
            "preprocessing": (
                "reconstructed_from_source_training_and_verified_against_"
                "saved_audit"
            ),
            "seen_training_public": "saved_exact_leave_one_out",
            "unseen_public": "source_training_reference_only",
        },
        "reproduction_tolerance": reproduction_tolerance,
        "preprocessing_reproduction_pass": bool(
            result.preprocessing_reproduction_audit["passed"].all()
        ),
        "native_public_reproduction_pass": bool(
            result.native_public_reproduction_audit["passed"].all()
        ),
        "native_prediction_reproduction_pass": bool(
            result.native_prediction_reproduction_audit["passed"].all()
        ),
        "repeated_sample_prediction_consistency_pass": bool(
            result.prediction_consistency_audit["passed"].all()
        ),
        "primary_result": "full_requested_holdout_n_by_n_matrix",
        "supplementary_result": "unseen_only_n_by_n_matrix",
        "independent_test_read": False,
        "automatic_final_model_selection": False,
        "output_sha256": {
            key: _sha256(path)
            for key, path in all_paths.items()
            if key != "complete"
        },
    }
    paths["complete"].write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return MappingProxyType({**paths, **matrix_paths})
