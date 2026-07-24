#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Leakage-controlled nested cross-validation orchestration for TRB V2.

This module connects the independently regression-tested V2 components without
changing their algorithms:

* fixed outer and inner fold assignments;
* exact leave-one-out public features for every model-fitting sample;
* training-reference-only public features for validation samples;
* fold-local categorical encoding, zero-variance filtering, and scaling;
* V1-compatible Elastic Net fitting and deterministic random seeds;
* pooled inner-CV candidate ranking and Youden threshold selection;
* final outer-training refit and outer-validation evaluation.

The engine operates on in-memory pandas/sparse objects. File loading and output
writing are handled by command-line wrappers so the core can be unit tested with
small synthetic fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from .metrics import classification_metrics, safe_pr_auc, safe_roc_auc
from .modeling import (
    coefficient_frame,
    count_nonzero_coefficients,
    derive_seed,
    fit_elastic_net,
)
from .preprocessing import PreparedDesign, prepare_design_matrices
from .public_reference import (
    ALL_PUBLIC_FEATURES,
    PublicFeatureResult,
    ThresholdScheme,
    external_public_features,
    leave_one_out_public_features,
)
from .specifications import (
    HyperparameterCandidate,
    ResolvedModelSpecification,
    select_best_tuning_candidate,
)
from .thresholds import apply_threshold, choose_threshold_youden


class NestedCVError(ValueError):
    """Raised when nested-CV assignments or execution inputs are invalid."""


@dataclass(frozen=True)
class NestedCVOptions:
    """Execution settings that are fixed by the experiment configuration."""

    base_seed: int
    class_weight: str = "balanced"
    max_iter: int = 10000
    tolerance: float = 1.0e-4
    zero_sd_tolerance: float = 1.0e-12
    epsilon: float = 1.0e-8
    coefficient_nonzero_tolerance: float = 1.0e-12
    positive_label: str = "ILD"
    negative_label: str = "RA"
    label_column: str = "cohort"
    sample_id_column: str = "sample_id"

    def __post_init__(self) -> None:
        if isinstance(self.base_seed, bool) or int(self.base_seed) < 0:
            raise NestedCVError("base_seed must be an integer >= 0.")
        if self.class_weight not in {"balanced", "none"}:
            raise NestedCVError("class_weight must be balanced or none.")
        if int(self.max_iter) < 1 or float(self.tolerance) <= 0:
            raise NestedCVError("Invalid model iteration/tolerance settings.")
        if float(self.zero_sd_tolerance) < 0:
            raise NestedCVError("zero_sd_tolerance must be >= 0.")
        if float(self.epsilon) <= 0:
            raise NestedCVError("epsilon must be > 0.")
        if float(self.coefficient_nonzero_tolerance) < 0:
            raise NestedCVError("coefficient_nonzero_tolerance must be >= 0.")
        if not self.positive_label.strip() or not self.negative_label.strip():
            raise NestedCVError("Class labels must be non-empty.")
        if self.positive_label.upper() == self.negative_label.upper():
            raise NestedCVError("Positive and negative labels must differ.")


@dataclass(frozen=True)
class OuterTaskSplit:
    outer_repeat: int
    outer_fold: int
    outer_train_ids: Tuple[str, ...]
    outer_valid_ids: Tuple[str, ...]
    inner_valid_ids: Mapping[int, Tuple[str, ...]]

    @property
    def n_outer_train(self) -> int:
        return len(self.outer_train_ids)

    @property
    def n_outer_valid(self) -> int:
        return len(self.outer_valid_ids)

    @property
    def inner_folds(self) -> Tuple[int, ...]:
        return tuple(self.inner_valid_ids)


@dataclass(frozen=True)
class PreparedInnerFold:
    X_train: np.ndarray
    X_valid: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    train_sample_ids: Tuple[str, ...]
    valid_sample_ids: Tuple[str, ...]
    feature_names: Tuple[str, ...]
    preprocessing_audit: pd.DataFrame


@dataclass(frozen=True)
class InnerPreparation:
    prepared: Mapping[str, Mapping[int, PreparedInnerFold]]
    preprocessing_audit: pd.DataFrame
    public_reference_audit: pd.DataFrame
    public_loo_assignments: pd.DataFrame


@dataclass(frozen=True)
class SelectedTuning:
    candidate: HyperparameterCandidate
    threshold: float
    inner_roc_auc: float
    inner_pr_auc: float


@dataclass(frozen=True)
class InnerTuningResult:
    tuning: pd.DataFrame
    selected_oof_predictions: pd.DataFrame
    selected: Mapping[str, SelectedTuning]


@dataclass(frozen=True)
class OuterFitResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    coefficients: pd.DataFrame
    preprocessing_audit: pd.DataFrame
    outer_train_public: pd.DataFrame
    outer_validation_public: pd.DataFrame
    public_reference_audit: pd.DataFrame
    public_loo_assignments: pd.DataFrame


@dataclass(frozen=True)
class NestedCVTaskResult:
    split: OuterTaskSplit
    inner_tuning: pd.DataFrame
    inner_selected_oof_predictions: pd.DataFrame
    selected: Mapping[str, SelectedTuning]
    outer_predictions: pd.DataFrame
    outer_metrics: pd.DataFrame
    coefficients: pd.DataFrame
    preprocessing_audit: pd.DataFrame
    outer_train_public: pd.DataFrame
    outer_validation_public: pd.DataFrame
    public_reference_audit: pd.DataFrame
    public_loo_assignments: pd.DataFrame
    sample_roles: pd.DataFrame


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise NestedCVError(f"{label} is missing columns: {missing}")


def _normalize_sample_ids(values: Sequence[object], label: str) -> Tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result:
        raise NestedCVError(f"{label} must not be empty.")
    if len(result) != len(set(result)):
        raise NestedCVError(f"{label} contains duplicate sample IDs.")
    return result


def validate_fixed_assignments(
    base_sample_ids: Sequence[object],
    outer_assignments: pd.DataFrame,
    inner_assignments: pd.DataFrame,
    *,
    outer_repeat: int,
    outer_fold: int,
    outer_folds: int,
    inner_folds: int,
    sample_id_column: str = "sample_id",
) -> OuterTaskSplit:
    """Validate one fixed outer task and preserve the V1 sample ordering rules."""
    base_ids = _normalize_sample_ids(base_sample_ids, "base_sample_ids")
    if int(outer_repeat) < 1 or int(outer_fold) < 1:
        raise NestedCVError("outer_repeat and outer_fold must be >= 1.")
    if int(outer_folds) < 2 or not 1 <= int(outer_fold) <= int(outer_folds):
        raise NestedCVError("outer_fold is outside the configured fold range.")
    if int(inner_folds) < 2:
        raise NestedCVError("inner_folds must be >= 2.")

    _require_columns(
        outer_assignments,
        (sample_id_column, "outer_repeat", "outer_fold"),
        "outer assignments",
    )
    outer = outer_assignments.copy()
    outer[sample_id_column] = outer[sample_id_column].astype(str)
    outer["outer_repeat"] = pd.to_numeric(
        outer["outer_repeat"], errors="raise"
    ).astype(int)
    outer["outer_fold"] = pd.to_numeric(outer["outer_fold"], errors="raise").astype(int)
    task = outer.loc[outer["outer_repeat"] == int(outer_repeat)].copy()
    if len(task) != len(base_ids):
        raise NestedCVError(
            f"Outer repeat {outer_repeat} has {len(task)} rows; expected {len(base_ids)}."
        )
    if task[sample_id_column].duplicated().any():
        raise NestedCVError("Outer assignment contains duplicated sample IDs.")
    if set(task[sample_id_column]) != set(base_ids):
        raise NestedCVError("Outer assignment sample IDs do not match the base matrix.")
    observed_outer_folds = set(task["outer_fold"].tolist())
    expected_outer_folds = set(range(1, int(outer_folds) + 1))
    if observed_outer_folds != expected_outer_folds:
        raise NestedCVError(
            f"Outer fold coverage={sorted(observed_outer_folds)}, "
            f"expected={sorted(expected_outer_folds)}."
        )

    train_set = set(
        task.loc[task["outer_fold"] != int(outer_fold), sample_id_column]
    )
    valid_set = set(
        task.loc[task["outer_fold"] == int(outer_fold), sample_id_column]
    )
    if train_set & valid_set or train_set | valid_set != set(base_ids):
        raise NestedCVError("Outer training and validation partition is invalid.")
    train_ids = tuple(sample_id for sample_id in base_ids if sample_id in train_set)
    valid_ids = tuple(sample_id for sample_id in base_ids if sample_id in valid_set)

    _require_columns(
        inner_assignments,
        (sample_id_column, "outer_repeat", "outer_fold", "inner_fold"),
        "inner assignments",
    )
    inner = inner_assignments.copy()
    inner[sample_id_column] = inner[sample_id_column].astype(str)
    for column in ("outer_repeat", "outer_fold", "inner_fold"):
        inner[column] = pd.to_numeric(inner[column], errors="raise").astype(int)
    inner_task = inner.loc[
        (inner["outer_repeat"] == int(outer_repeat))
        & (inner["outer_fold"] == int(outer_fold))
    ].copy()
    if inner_task[sample_id_column].duplicated().any():
        raise NestedCVError("Inner assignment contains duplicated sample IDs.")
    if set(inner_task[sample_id_column]) != train_set:
        raise NestedCVError(
            "Inner assignment IDs are not exactly the selected outer-training IDs."
        )
    if set(inner_task[sample_id_column]) & valid_set:
        raise NestedCVError("Outer validation samples leaked into inner assignments.")
    observed_inner_folds = set(inner_task["inner_fold"].tolist())
    expected_inner_folds = set(range(1, int(inner_folds) + 1))
    if observed_inner_folds != expected_inner_folds:
        raise NestedCVError(
            f"Inner fold coverage={sorted(observed_inner_folds)}, "
            f"expected={sorted(expected_inner_folds)}."
        )

    inner_valid: Dict[int, Tuple[str, ...]] = {}
    for fold in range(1, int(inner_folds) + 1):
        # V1 used the row order in the fixed inner-assignment file.
        values = tuple(
            inner_task.loc[
                inner_task["inner_fold"] == fold, sample_id_column
            ].astype(str)
        )
        if not values:
            raise NestedCVError(f"Inner fold {fold} has no validation samples.")
        inner_valid[fold] = values

    return OuterTaskSplit(
        outer_repeat=int(outer_repeat),
        outer_fold=int(outer_fold),
        outer_train_ids=train_ids,
        outer_valid_ids=valid_ids,
        inner_valid_ids=MappingProxyType(inner_valid),
    )


def _indexed_base(
    base: pd.DataFrame,
    *,
    sample_id_column: str,
    label_column: str,
    positive_label: str,
    negative_label: str,
) -> pd.DataFrame:
    _require_columns(base, (sample_id_column, label_column), "base matrix")
    output = base.copy()
    output[sample_id_column] = output[sample_id_column].astype(str)
    if output[sample_id_column].duplicated().any():
        raise NestedCVError("Base matrix contains duplicate sample IDs.")
    if output.isna().any().any():
        bad = output.isna().sum()
        raise NestedCVError(
            f"Base matrix contains missing values: {bad[bad > 0].to_dict()}"
        )
    output[label_column] = output[label_column].astype(str).str.strip().str.upper()
    expected = {positive_label.upper(), negative_label.upper()}
    observed = set(output[label_column])
    if observed != expected:
        raise NestedCVError(
            f"Base labels={sorted(observed)}, expected={sorted(expected)}."
        )
    return output.set_index(sample_id_column, drop=False)


def _binary_labels(frame: pd.DataFrame, options: NestedCVOptions) -> np.ndarray:
    labels = frame[options.label_column].astype(str).str.upper()
    unexpected = sorted(
        set(labels) - {options.positive_label.upper(), options.negative_label.upper()}
    )
    if unexpected:
        raise NestedCVError(f"Unexpected class labels: {unexpected}")
    values = (labels == options.positive_label.upper()).astype(int).to_numpy()
    if set(np.unique(values).tolist()) != {0, 1}:
        raise NestedCVError("Every model fitting/evaluation partition must contain both classes.")
    return values


def assemble_model_dataframe(
    base: pd.DataFrame,
    sample_ids: Sequence[str],
    public_values: Optional[pd.DataFrame],
    row_lookup: Mapping[str, int],
) -> pd.DataFrame:
    """Align base rows and optional public values using V1 sample order."""
    ids = tuple(map(str, sample_ids))
    missing = [sample_id for sample_id in ids if sample_id not in base.index]
    if missing:
        raise NestedCVError(f"Requested samples are absent from base: {missing[:10]}")
    output = base.loc[list(ids)].copy()
    if public_values is None:
        return output
    row_ids = [int(row_lookup[sample_id]) for sample_id in ids]
    try:
        aligned = public_values.loc[row_ids, list(ALL_PUBLIC_FEATURES)].copy()
    except KeyError as exc:
        raise NestedCVError("Public feature rows/columns cannot be aligned.") from exc
    aligned.index = list(ids)
    collisions = sorted(set(output.columns) & set(ALL_PUBLIC_FEATURES))
    if collisions:
        raise NestedCVError(f"Public features collide with base columns: {collisions}")
    for column in ALL_PUBLIC_FEATURES:
        output[column] = pd.to_numeric(aligned[column], errors="raise").to_numpy(float)
    return output


def _annotate_public_audit(
    result: PublicFeatureResult,
    *,
    split: OuterTaskSplit,
    inner_fold: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    reference = result.reference_audit.copy()
    assignment = result.assignment_audit.copy()
    for frame in (reference, assignment):
        if frame.empty:
            continue
        frame["outer_repeat"] = split.outer_repeat
        frame["outer_fold"] = split.outer_fold
        frame["inner_fold"] = "" if inner_fold is None else int(inner_fold)
    return reference, assignment


def prepare_inner_folds(
    base: pd.DataFrame,
    split: OuterTaskSplit,
    row_lookup: Mapping[str, int],
    presence: sparse.spmatrix,
    frequency: sparse.spmatrix,
    labels: Sequence[object],
    aa_clone_numbers: Sequence[float],
    model_specifications: Mapping[str, ResolvedModelSpecification],
    scheme: ThresholdScheme,
    options: NestedCVOptions,
) -> InnerPreparation:
    """Build every fold-local public feature and standardized model matrix."""
    indexed = _indexed_base(
        base,
        sample_id_column=options.sample_id_column,
        label_column=options.label_column,
        positive_label=options.positive_label,
        negative_label=options.negative_label,
    )
    prepared: Dict[str, Dict[int, PreparedInnerFold]] = {
        model: {} for model in model_specifications
    }
    preprocessing_rows: List[pd.DataFrame] = []
    reference_rows: List[pd.DataFrame] = []
    assignment_rows: List[pd.DataFrame] = []

    for inner_fold, valid_ids in split.inner_valid_ids.items():
        valid_set = set(valid_ids)
        train_ids = tuple(
            sample_id
            for sample_id in split.outer_train_ids
            if sample_id not in valid_set
        )
        if set(train_ids) & valid_set or set(train_ids) | valid_set != set(split.outer_train_ids):
            raise NestedCVError(f"Invalid inner partition for fold {inner_fold}.")
        train_rows = np.asarray([row_lookup[x] for x in train_ids], dtype=int)
        valid_rows = np.asarray([row_lookup[x] for x in valid_ids], dtype=int)

        train_public_result = leave_one_out_public_features(
            presence,
            frequency,
            train_rows,
            labels,
            aa_clone_numbers,
            scheme,
            options.epsilon,
            context="inner_training_leave_one_out",
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
            context="inner_validation_application",
        )
        for public_result in (train_public_result, valid_public_result):
            ref, assignment = _annotate_public_audit(
                public_result,
                split=split,
                inner_fold=inner_fold,
            )
            if not ref.empty:
                reference_rows.append(ref)
            # V1's LOO-assignment file records only fitting-sample LOO rows.
            if (
                not assignment.empty
                and set(assignment.get("generation_method", [])) == {"leave_one_out"}
            ):
                assignment_rows.append(assignment)

        train_frame = assemble_model_dataframe(
            indexed,
            train_ids,
            train_public_result.features,
            row_lookup,
        )
        valid_frame = assemble_model_dataframe(
            indexed,
            valid_ids,
            valid_public_result.features,
            row_lookup,
        )
        y_train = _binary_labels(train_frame, options)
        y_valid = _binary_labels(valid_frame, options)

        for model_name, specification in model_specifications.items():
            design = prepare_design_matrices(
                train_frame,
                valid_frame,
                specification.numeric,
                specification.categorical,
                zero_sd_tolerance=options.zero_sd_tolerance,
            )
            audit = design.audit.copy()
            audit["stage"] = "inner"
            audit["model"] = model_name
            audit["inner_fold"] = int(inner_fold)
            preprocessing_rows.append(audit)
            prepared[model_name][int(inner_fold)] = PreparedInnerFold(
                X_train=design.X_train,
                X_valid=design.X_valid,
                y_train=y_train,
                y_valid=y_valid,
                train_sample_ids=train_ids,
                valid_sample_ids=tuple(valid_ids),
                feature_names=tuple(design.feature_names),
                preprocessing_audit=audit,
            )

    frozen_prepared = MappingProxyType(
        {
            model: MappingProxyType(dict(folds))
            for model, folds in prepared.items()
        }
    )
    return InnerPreparation(
        prepared=frozen_prepared,
        preprocessing_audit=(
            pd.concat(preprocessing_rows, ignore_index=True)
            if preprocessing_rows
            else pd.DataFrame()
        ),
        public_reference_audit=(
            pd.concat(reference_rows, ignore_index=True)
            if reference_rows
            else pd.DataFrame()
        ),
        public_loo_assignments=(
            pd.concat(assignment_rows, ignore_index=True)
            if assignment_rows
            else pd.DataFrame()
        ),
    )


def tune_models(
    prepared: Mapping[str, Mapping[int, PreparedInnerFold]],
    candidates: Sequence[HyperparameterCandidate],
    options: NestedCVOptions,
) -> InnerTuningResult:
    """Fit every inner-fold candidate and apply the frozen V1 ranking rule."""
    if not candidates:
        raise NestedCVError("At least one hyperparameter candidate is required.")
    tuning_rows: List[Dict[str, object]] = []
    prediction_cache: Dict[
        Tuple[str, float, float], List[Dict[str, object]]
    ] = {}

    for model_index, (model_name, folds) in enumerate(prepared.items()):
        if not folds:
            raise NestedCVError(f"No prepared inner folds for {model_name}.")
        for candidate in candidates:
            prediction_rows: List[Dict[str, object]] = []
            fold_aucs: List[float] = []
            convergence: List[bool] = []
            iterations: List[int] = []
            for inner_fold, fold_data in folds.items():
                seed = derive_seed(
                    options.base_seed,
                    200,
                    model_index,
                    int(inner_fold),
                    int(round(candidate.alpha * 1000)),
                    int(round(candidate.lambda_value * 1000)),
                )
                fit = fit_elastic_net(
                    fold_data.X_train,
                    fold_data.y_train,
                    l1_ratio=candidate.alpha,
                    lambda_value=candidate.lambda_value,
                    class_weight=options.class_weight,
                    max_iter=options.max_iter,
                    tolerance=options.tolerance,
                    random_state=seed,
                )
                probability = fit.predict_probability(fold_data.X_valid)
                fold_aucs.append(safe_roc_auc(fold_data.y_valid, probability))
                convergence.append(bool(fit.converged))
                iterations.append(int(fit.n_iter))
                for sample_id, truth, value in zip(
                    fold_data.valid_sample_ids,
                    fold_data.y_valid,
                    probability,
                ):
                    prediction_rows.append(
                        {
                            "model": model_name,
                            "inner_fold": int(inner_fold),
                            "sample_id": sample_id,
                            "true_label": int(truth),
                            "probability": float(value),
                            "l1_ratio_alpha": float(candidate.alpha),
                            "lambda": float(candidate.lambda_value),
                        }
                    )

            prediction_frame = pd.DataFrame(prediction_rows)
            pooled_auc = safe_roc_auc(
                prediction_frame["true_label"].to_numpy(int),
                prediction_frame["probability"].to_numpy(float),
            )
            pooled_pr = safe_pr_auc(
                prediction_frame["true_label"].to_numpy(int),
                prediction_frame["probability"].to_numpy(float),
            )
            tuning_rows.append(
                {
                    "model": model_name,
                    "l1_ratio_alpha": float(candidate.alpha),
                    "lambda": float(candidate.lambda_value),
                    "C_inverse_lambda": float(candidate.inverse_lambda_c),
                    "pooled_inner_roc_auc": float(pooled_auc),
                    "pooled_inner_pr_auc": float(pooled_pr),
                    "mean_fold_roc_auc": float(np.mean(fold_aucs)),
                    "sd_fold_roc_auc": float(np.std(fold_aucs, ddof=1)),
                    "all_fits_converged": bool(all(convergence)),
                    "max_iterations_used": int(max(iterations)),
                }
            )
            prediction_cache[
                (model_name, float(candidate.alpha), float(candidate.lambda_value))
            ] = prediction_rows

    tuning = pd.DataFrame(tuning_rows)
    selected: Dict[str, SelectedTuning] = {}
    selected_oof_parts: List[pd.DataFrame] = []
    for model_name in prepared:
        candidate_rows = tuning.loc[tuning["model"] == model_name].copy()
        best_candidate = select_best_tuning_candidate(candidate_rows)
        ranked = candidate_rows.sort_values(
            [
                "pooled_inner_roc_auc",
                "pooled_inner_pr_auc",
                "lambda",
                "l1_ratio_alpha",
            ],
            ascending=[False, False, False, False],
            kind="mergesort",
        )
        best = ranked.iloc[0]
        prediction = pd.DataFrame(
            prediction_cache[
                (
                    model_name,
                    float(best_candidate.alpha),
                    float(best_candidate.lambda_value),
                )
            ]
        )
        threshold = choose_threshold_youden(
            prediction["true_label"].to_numpy(int),
            prediction["probability"].to_numpy(float),
        )
        prediction["selected_threshold"] = float(threshold)
        selected_oof_parts.append(prediction)
        selected[model_name] = SelectedTuning(
            candidate=best_candidate,
            threshold=float(threshold),
            inner_roc_auc=float(best["pooled_inner_roc_auc"]),
            inner_pr_auc=float(best["pooled_inner_pr_auc"]),
        )

    return InnerTuningResult(
        tuning=tuning,
        selected_oof_predictions=pd.concat(selected_oof_parts, ignore_index=True),
        selected=MappingProxyType(selected),
    )


def fit_outer_models(
    base: pd.DataFrame,
    split: OuterTaskSplit,
    row_lookup: Mapping[str, int],
    presence: sparse.spmatrix,
    frequency: sparse.spmatrix,
    labels: Sequence[object],
    aa_clone_numbers: Sequence[float],
    model_specifications: Mapping[str, ResolvedModelSpecification],
    selected: Mapping[str, SelectedTuning],
    scheme: ThresholdScheme,
    options: NestedCVOptions,
) -> OuterFitResult:
    """Refit selected models on outer training and evaluate outer validation."""
    indexed = _indexed_base(
        base,
        sample_id_column=options.sample_id_column,
        label_column=options.label_column,
        positive_label=options.positive_label,
        negative_label=options.negative_label,
    )
    train_rows = np.asarray(
        [row_lookup[x] for x in split.outer_train_ids], dtype=int
    )
    valid_rows = np.asarray(
        [row_lookup[x] for x in split.outer_valid_ids], dtype=int
    )
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
        indexed,
        split.outer_train_ids,
        train_public_result.features,
        row_lookup,
    )
    valid_frame = assemble_model_dataframe(
        indexed,
        split.outer_valid_ids,
        valid_public_result.features,
        row_lookup,
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
            raise NestedCVError(f"Selected tuning parameters missing for {model_name}.")
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
        seed = derive_seed(options.base_seed, 400, model_index)
        fit = fit_elastic_net(
            design.X_train,
            y_train,
            l1_ratio=tuning.candidate.alpha,
            lambda_value=tuning.candidate.lambda_value,
            class_weight=options.class_weight,
            max_iter=options.max_iter,
            tolerance=options.tolerance,
            random_state=seed,
        )
        probability = fit.predict_probability(design.X_valid)
        threshold = float(tuning.threshold)
        metrics = classification_metrics(y_valid, probability, threshold)
        metrics.update(
            {
                "model": model_name,
                "model_definition": specification.description,
                "outer_repeat": split.outer_repeat,
                "outer_fold": split.outer_fold,
                "n_outer_train": split.n_outer_train,
                "n_outer_validation": split.n_outer_valid,
                "selected_l1_ratio_alpha": float(tuning.candidate.alpha),
                "selected_lambda": float(tuning.candidate.lambda_value),
                "inner_selected_roc_auc": float(tuning.inner_roc_auc),
                "inner_selected_pr_auc": float(tuning.inner_pr_auc),
                "fit_converged": bool(fit.converged),
                "iterations_used": int(fit.n_iter),
                "n_final_predictors": int(len(design.feature_names)),
                "n_nonzero_coefficients": count_nonzero_coefficients(
                    fit.model,
                    tolerance=options.coefficient_nonzero_tolerance,
                ),
            }
        )
        metric_rows.append(metrics)

        predicted = apply_threshold(probability, threshold)
        for sample_id, truth, value, prediction in zip(
            split.outer_valid_ids,
            y_valid,
            probability,
            predicted,
        ):
            prediction_rows.append(
                {
                    "model": model_name,
                    "outer_repeat": split.outer_repeat,
                    "outer_fold": split.outer_fold,
                    "sample_id": sample_id,
                    "true_label": int(truth),
                    "true_cohort": (
                        options.positive_label if truth == 1 else options.negative_label
                    ),
                    "probability_ILD": float(value),
                    "threshold": threshold,
                    "predicted_label": int(prediction),
                    "predicted_cohort": (
                        options.positive_label
                        if prediction == 1
                        else options.negative_label
                    ),
                }
            )
        coefficient_parts.append(
            coefficient_frame(
                fit.model,
                design.feature_names,
                nonzero_tolerance=options.coefficient_nonzero_tolerance,
                model_name=model_name,
            )
        )

    train_public = train_public_result.features.loc[train_rows].copy()
    train_public.insert(0, options.sample_id_column, list(split.outer_train_ids))
    valid_public = valid_public_result.features.loc[valid_rows].copy()
    valid_public.insert(0, options.sample_id_column, list(split.outer_valid_ids))

    return OuterFitResult(
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


def sample_role_frame(
    base: pd.DataFrame,
    split: OuterTaskSplit,
    options: NestedCVOptions,
) -> pd.DataFrame:
    indexed = _indexed_base(
        base,
        sample_id_column=options.sample_id_column,
        label_column=options.label_column,
        positive_label=options.positive_label,
        negative_label=options.negative_label,
    )
    train_set = set(split.outer_train_ids)
    rows = []
    for sample_id in indexed.index:
        rows.append(
            {
                "sample_id": sample_id,
                "cohort": indexed.loc[sample_id, options.label_column],
                "outer_repeat": split.outer_repeat,
                "outer_fold": split.outer_fold,
                "outer_role": "training" if sample_id in train_set else "validation",
            }
        )
    return pd.DataFrame(rows)


def run_nested_outer_task(
    base: pd.DataFrame,
    outer_assignments: pd.DataFrame,
    inner_assignments: pd.DataFrame,
    row_lookup: Mapping[str, int],
    presence: sparse.spmatrix,
    frequency: sparse.spmatrix,
    labels: Sequence[object],
    aa_clone_numbers: Sequence[float],
    model_specifications: Mapping[str, ResolvedModelSpecification],
    candidates: Sequence[HyperparameterCandidate],
    scheme: ThresholdScheme,
    options: NestedCVOptions,
    *,
    outer_repeat: int,
    outer_fold: int,
    outer_folds: int,
    inner_folds: int,
) -> NestedCVTaskResult:
    """Run one complete V1-compatible nested-CV outer task in memory."""
    _require_columns(base, (options.sample_id_column,), "base matrix")
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
    inner = prepare_inner_folds(
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
    tuning = tune_models(inner.prepared, candidates, options)
    outer = fit_outer_models(
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
    return NestedCVTaskResult(
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
