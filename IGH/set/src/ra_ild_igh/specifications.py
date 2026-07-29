#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model specifications, static-feature selection, and candidate management.

This module extracts the non-fitting decisions that were previously embedded in
V1 modeling scripts:

* selection of static IGH candidate predictors from the step-04 manifest;
* expansion of configured model definitions into concrete predictor columns;
* deterministic construction of the Elastic Net alpha/lambda grid;
* V1-compatible ranking of inner-CV tuning candidates;
* validation of locked final-model candidate pairs.

The module does not fit a model and therefore has no scikit-learn dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class SpecificationError(ValueError):
    """Raised when feature or model specifications are invalid."""


STATIC_FEATURE_GROUP = "static_igh_candidate_predictors"

TUNING_SORT_COLUMNS: Tuple[str, ...] = (
    "pooled_inner_roc_auc",
    "pooled_inner_pr_auc",
    "lambda",
    "l1_ratio_alpha",
)
TUNING_SORT_ASCENDING: Tuple[bool, ...] = (False, False, False, False)


@dataclass(frozen=True)
class HyperparameterCandidate:
    """One Elastic Net tuning candidate."""

    alpha: float
    lambda_value: float

    @property
    def inverse_lambda_c(self) -> float:
        return 1.0 / self.lambda_value

    def as_dict(self) -> Dict[str, float]:
        return {
            "l1_ratio_alpha": float(self.alpha),
            "lambda": float(self.lambda_value),
            "C_inverse_lambda": float(self.inverse_lambda_c),
        }


@dataclass(frozen=True)
class ModelSpecification:
    """Unresolved model definition from configuration."""

    name: str
    description: str
    numeric: Tuple[str, ...]
    categorical: Tuple[str, ...]
    static_feature_groups: Tuple[str, ...]
    dynamic_public: Tuple[str, ...]


@dataclass(frozen=True)
class ResolvedModelSpecification:
    """Concrete predictor columns for one model."""

    name: str
    description: str
    numeric: Tuple[str, ...]
    categorical: Tuple[str, ...]

    @property
    def raw_predictor_count(self) -> int:
        return len(self.numeric) + len(self.categorical)


def _string_tuple(value: Any, label: str) -> Tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SpecificationError(f"{label} must be a sequence of strings.")
    result: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise SpecificationError(f"{label} contains an invalid value: {item!r}")
        result.append(item.strip())
    if len(result) != len(set(result)):
        raise SpecificationError(f"{label} contains duplicated values.")
    return tuple(result)


def _boolean_series(series: pd.Series, label: str) -> pd.Series:
    """Interpret V1 manifest booleans stored as bool, 0/1, or text."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    accepted_true = {"true", "1", "yes"}
    accepted_false = {"false", "0", "no", "nan", "none", ""}
    invalid = sorted(set(normalized) - accepted_true - accepted_false)
    if invalid:
        raise SpecificationError(
            f"{label} contains unrecognized boolean values: {invalid[:10]}"
        )
    return normalized.isin(accepted_true)


def select_static_igh_features(
    manifest: pd.DataFrame,
    *,
    available_columns: Optional[Iterable[str]] = None,
    require_all_available: bool = False,
) -> List[str]:
    """Reproduce the V1 static-IGH feature-manifest selection rule.

    A feature is selected when all of the following hold:

    * ``present_in_train_base`` is true;
    * ``feature_group`` starts with ``igh_``;
    * ``feature_role`` equals ``candidate_predictor``;
    * ``reference_drop`` is false.

    Manifest order is preserved. V1 subsequently retained only features present
    in the base matrix; this behavior is reproduced when ``available_columns``
    is supplied. Set ``require_all_available=True`` for stricter validation.
    """
    required = {
        "feature_name",
        "feature_group",
        "feature_role",
        "present_in_train_base",
        "reference_drop",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise SpecificationError(f"Feature manifest missing columns: {missing}")

    if manifest["feature_name"].isna().any():
        raise SpecificationError("Feature manifest contains missing feature names.")
    names = manifest["feature_name"].astype(str)
    if names.duplicated().any():
        examples = names[names.duplicated(keep=False)].unique().tolist()[:10]
        raise SpecificationError(f"Feature manifest has duplicated names: {examples}")

    present = _boolean_series(
        manifest["present_in_train_base"], "present_in_train_base"
    )
    reference_drop = _boolean_series(manifest["reference_drop"], "reference_drop")

    selected = manifest.loc[
        present
        & manifest["feature_group"].astype(str).str.startswith("igh_")
        & manifest["feature_role"].astype(str).eq("candidate_predictor")
        & (~reference_drop),
        "feature_name",
    ].astype(str).tolist()

    if available_columns is not None:
        available = set(map(str, available_columns))
        unavailable = [name for name in selected if name not in available]
        if unavailable and require_all_available:
            raise SpecificationError(
                "Selected manifest features are missing from the base matrix: "
                f"{unavailable[:20]}"
            )
        selected = [name for name in selected if name in available]

    if not selected:
        raise SpecificationError("No static IGH candidate predictors were selected.")
    return selected


def parse_model_specifications(
    mapping: Mapping[str, Mapping[str, Any]],
) -> Dict[str, ModelSpecification]:
    """Parse configured model definitions while preserving mapping order."""
    if not isinstance(mapping, Mapping) or not mapping:
        raise SpecificationError("models must be a non-empty mapping.")

    parsed: Dict[str, ModelSpecification] = {}
    for model_name, raw in mapping.items():
        if not isinstance(model_name, str) or not model_name.strip():
            raise SpecificationError("Model names must be non-empty strings.")
        if not isinstance(raw, Mapping):
            raise SpecificationError(f"Model {model_name} must be a mapping.")
        description = str(raw.get("description", "")).strip()
        if not description:
            raise SpecificationError(f"Model {model_name} lacks a description.")
        spec = ModelSpecification(
            name=model_name.strip(),
            description=description,
            numeric=_string_tuple(raw.get("numeric", ()), f"{model_name}.numeric"),
            categorical=_string_tuple(
                raw.get("categorical", ()), f"{model_name}.categorical"
            ),
            static_feature_groups=_string_tuple(
                raw.get("static_feature_groups", ()),
                f"{model_name}.static_feature_groups",
            ),
            dynamic_public=_string_tuple(
                raw.get("dynamic_public", ()), f"{model_name}.dynamic_public"
            ),
        )
        if not (
            spec.numeric
            or spec.categorical
            or spec.static_feature_groups
            or spec.dynamic_public
        ):
            raise SpecificationError(f"Model {model_name} has no predictors.")
        parsed[spec.name] = spec
    return parsed


def resolve_model_specification(
    specification: ModelSpecification,
    *,
    static_feature_groups: Mapping[str, Sequence[str]],
    allowed_dynamic_public: Optional[Sequence[str]] = None,
) -> ResolvedModelSpecification:
    """Expand feature-group references into concrete model columns."""
    numeric: List[str] = list(specification.numeric)
    for group in specification.static_feature_groups:
        if group not in static_feature_groups:
            raise SpecificationError(
                f"Model {specification.name} references unknown static group {group!r}."
            )
        numeric.extend(map(str, static_feature_groups[group]))

    if allowed_dynamic_public is not None:
        allowed = set(map(str, allowed_dynamic_public))
        unknown = [name for name in specification.dynamic_public if name not in allowed]
        if unknown:
            raise SpecificationError(
                f"Model {specification.name} has unknown dynamic public features: {unknown}"
            )
    numeric.extend(specification.dynamic_public)
    categorical = list(specification.categorical)

    if len(numeric) != len(set(numeric)):
        raise SpecificationError(
            f"Resolved numeric columns for {specification.name} contain duplicates."
        )
    if len(categorical) != len(set(categorical)):
        raise SpecificationError(
            f"Resolved categorical columns for {specification.name} contain duplicates."
        )
    overlap = sorted(set(numeric) & set(categorical))
    if overlap:
        raise SpecificationError(
            f"Model {specification.name} uses columns as both numeric and categorical: {overlap}"
        )
    if not numeric and not categorical:
        raise SpecificationError(f"Resolved model {specification.name} is empty.")

    return ResolvedModelSpecification(
        name=specification.name,
        description=specification.description,
        numeric=tuple(numeric),
        categorical=tuple(categorical),
    )


def resolve_all_model_specifications(
    mapping: Mapping[str, Mapping[str, Any]],
    *,
    static_features: Sequence[str],
    allowed_dynamic_public: Sequence[str],
    static_group_name: str = STATIC_FEATURE_GROUP,
) -> Dict[str, ResolvedModelSpecification]:
    parsed = parse_model_specifications(mapping)
    groups = {static_group_name: tuple(map(str, static_features))}
    return {
        name: resolve_model_specification(
            specification,
            static_feature_groups=groups,
            allowed_dynamic_public=allowed_dynamic_public,
        )
        for name, specification in parsed.items()
    }


def build_hyperparameter_grid(
    alpha_grid: Sequence[float],
    lambda_grid: Sequence[float],
) -> List[HyperparameterCandidate]:
    """Build candidates in the same alpha-then-lambda order used by V1."""
    if not alpha_grid or not lambda_grid:
        raise SpecificationError("Alpha and lambda grids must both be non-empty.")
    alphas = [float(value) for value in alpha_grid]
    lambdas = [float(value) for value in lambda_grid]
    if any(not np.isfinite(value) or value < 0 or value > 1 for value in alphas):
        raise SpecificationError("Every alpha must be finite and in [0, 1].")
    if any(not np.isfinite(value) or value <= 0 for value in lambdas):
        raise SpecificationError("Every lambda must be finite and > 0.")
    if len(alphas) != len(set(alphas)):
        raise SpecificationError("Alpha grid contains duplicates.")
    if len(lambdas) != len(set(lambdas)):
        raise SpecificationError("Lambda grid contains duplicates.")
    return [
        HyperparameterCandidate(alpha=alpha, lambda_value=lambda_value)
        for alpha, lambda_value in product(alphas, lambdas)
    ]


def candidate_pairs(candidates: Sequence[HyperparameterCandidate]) -> set[Tuple[float, float]]:
    return {(float(item.alpha), float(item.lambda_value)) for item in candidates}


def rank_tuning_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort one model's tuning rows using the exact V1 tie-break order."""
    required = {
        "pooled_inner_roc_auc",
        "pooled_inner_pr_auc",
        "lambda",
        "l1_ratio_alpha",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SpecificationError(f"Tuning results missing columns: {missing}")
    if frame.empty:
        raise SpecificationError("Tuning results are empty.")

    ranked = frame.copy()
    for column in required:
        ranked[column] = pd.to_numeric(ranked[column], errors="raise")
    values = ranked[list(required)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise SpecificationError("Tuning results contain non-finite ranking values.")
    return ranked.sort_values(
        list(TUNING_SORT_COLUMNS),
        ascending=list(TUNING_SORT_ASCENDING),
        kind="mergesort",
    ).reset_index(drop=True)


def select_best_tuning_candidate(frame: pd.DataFrame) -> HyperparameterCandidate:
    best = rank_tuning_candidates(frame).iloc[0]
    return HyperparameterCandidate(
        alpha=float(best["l1_ratio_alpha"]),
        lambda_value=float(best["lambda"]),
    )


def validate_tuning_grid_frame(
    frame: pd.DataFrame,
    *,
    model_names: Sequence[str],
    candidates: Sequence[HyperparameterCandidate],
    inverse_lambda_tolerance: float = 1e-12,
) -> None:
    """Validate that every model contains every configured candidate once."""
    required = {"model", "l1_ratio_alpha", "lambda", "C_inverse_lambda"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SpecificationError(f"Tuning results missing columns: {missing}")

    expected_models = list(map(str, model_names))
    observed_models = frame["model"].astype(str).unique().tolist()
    if set(observed_models) != set(expected_models):
        raise SpecificationError(
            f"Tuning model set mismatch: observed={observed_models}, expected={expected_models}"
        )

    expected_pairs = candidate_pairs(candidates)
    for model in expected_models:
        subset = frame.loc[frame["model"].astype(str) == model].copy()
        alpha = pd.to_numeric(subset["l1_ratio_alpha"], errors="raise").astype(float)
        lambda_value = pd.to_numeric(subset["lambda"], errors="raise").astype(float)
        pairs = list(zip(alpha, lambda_value))
        if len(pairs) != len(set(pairs)):
            raise SpecificationError(f"Model {model} has duplicated tuning pairs.")
        if set(pairs) != expected_pairs:
            missing_pairs = sorted(expected_pairs - set(pairs))
            extra_pairs = sorted(set(pairs) - expected_pairs)
            raise SpecificationError(
                f"Model {model} tuning grid mismatch; missing={missing_pairs}, extra={extra_pairs}"
            )
        observed_c = pd.to_numeric(
            subset["C_inverse_lambda"], errors="raise"
        ).to_numpy(float)
        expected_c = 1.0 / lambda_value.to_numpy(float)
        if not np.allclose(
            observed_c,
            expected_c,
            rtol=inverse_lambda_tolerance,
            atol=inverse_lambda_tolerance,
        ):
            raise SpecificationError(f"Model {model} has incorrect C=1/lambda values.")


def selected_pairs_from_oof(frame: pd.DataFrame) -> Dict[str, HyperparameterCandidate]:
    """Extract one unique selected pair per model from V1 OOF predictions."""
    required = {"model", "l1_ratio_alpha", "lambda"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SpecificationError(f"OOF predictions missing columns: {missing}")
    selected: Dict[str, HyperparameterCandidate] = {}
    for model, subset in frame.groupby(frame["model"].astype(str), sort=False):
        pairs = subset[["l1_ratio_alpha", "lambda"]].copy()
        pairs["l1_ratio_alpha"] = pd.to_numeric(
            pairs["l1_ratio_alpha"], errors="raise"
        )
        pairs["lambda"] = pd.to_numeric(pairs["lambda"], errors="raise")
        unique = pairs.drop_duplicates()
        if len(unique) != 1:
            raise SpecificationError(
                f"OOF predictions for {model} contain {len(unique)} selected pairs."
            )
        row = unique.iloc[0]
        selected[str(model)] = HyperparameterCandidate(
            alpha=float(row["l1_ratio_alpha"]),
            lambda_value=float(row["lambda"]),
        )
    return selected


def parse_candidate_pair_mapping(
    values: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> List[HyperparameterCandidate]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise SpecificationError(f"{label} must be a non-empty list.")
    result: List[HyperparameterCandidate] = []
    for index, raw in enumerate(values):
        if not isinstance(raw, Mapping):
            raise SpecificationError(f"{label}[{index}] must be a mapping.")
        if "alpha" not in raw or "lambda" not in raw:
            raise SpecificationError(f"{label}[{index}] requires alpha and lambda.")
        candidate = HyperparameterCandidate(
            alpha=float(raw["alpha"]),
            lambda_value=float(raw["lambda"]),
        )
        # Reuse grid validation on a singleton for bounds.
        build_hyperparameter_grid([candidate.alpha], [candidate.lambda_value])
        result.append(candidate)
    if len(candidate_pairs(result)) != len(result):
        raise SpecificationError(f"{label} contains duplicated candidate pairs.")
    return result


def validate_locked_candidate_selection(
    *,
    full_grid: Sequence[HyperparameterCandidate],
    candidate_subset: Sequence[HyperparameterCandidate],
    selected: HyperparameterCandidate,
) -> None:
    grid_pairs = candidate_pairs(full_grid)
    subset_pairs = candidate_pairs(candidate_subset)
    selected_pair = (float(selected.alpha), float(selected.lambda_value))
    if not subset_pairs.issubset(grid_pairs):
        raise SpecificationError(
            f"Final candidate subset contains pairs outside the tuning grid: "
            f"{sorted(subset_pairs - grid_pairs)}"
        )
    if selected_pair not in subset_pairs:
        raise SpecificationError(
            f"Locked selected pair {selected_pair} is not in the final candidate subset."
        )
