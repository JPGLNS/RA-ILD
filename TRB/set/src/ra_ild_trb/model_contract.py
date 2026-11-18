#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict scientific contract for the TRB V3 179 hierarchical models.

This module validates the frozen M0-M3 predictor hierarchy and the existing
outer-repeat-only 3-mer selection policy. It does not fit models or change any
feature-generation algorithm.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


class ModelContractError(ValueError):
    """Raised when a resolved TRB model violates the frozen hierarchy."""


PUBLIC_FEATURES = (
    "all_ref_public_clone_ratio",
    "all_ref_public_frequency_sum",
    "shared_ref_clone_ratio",
    "shared_ref_frequency_sum",
    "RA_specific_ref_clone_ratio",
    "RA_specific_ref_frequency_sum",
    "ILD_specific_ref_clone_ratio",
    "ILD_specific_ref_frequency_sum",
)

MODEL_SPECS = {
    "M0_clinical": {
        "numeric": ("age",),
        "categorical": ("sex",),
        "static_feature_groups": (),
        "dynamic_public": (),
    },
    "M1_tcr": {
        "numeric": ("age",),
        "categorical": ("sex",),
        "static_feature_groups": ("core83", "repeat_3mer_both_top500"),
        "dynamic_public": (),
    },
    "M2_tcr_clinical": {
        "numeric": ("age",),
        "categorical": ("sex",),
        "static_feature_groups": ("core83", "repeat_3mer_both_top500"),
        "dynamic_public": PUBLIC_FEATURES,
    },
    "M3_tcr_clinical_material": {
        "numeric": ("age",),
        "categorical": ("sex", "material"),
        "static_feature_groups": ("core83", "repeat_3mer_both_top500"),
        "dynamic_public": PUBLIC_FEATURES,
    },
}

COHORT_MODELS = {
    "total": tuple(MODEL_SPECS),
    "pbmc": tuple(MODEL_SPECS)[:3],
    "buffycoat": tuple(MODEL_SPECS)[:3],
}

REPEAT_3MER_EXPECTED = {
    "enabled": True,
    "selection_scope": "outer_repeat_train_only",
    "ranking_rule": "prevalence_then_total_variance",
    "sequence_column": "cdr3_aa",
    "min_sample_count": 5,
    "min_prevalence": 0.05,
    "min_unweighted_variance": 1.0e-12,
    "min_weighted_variance": 1.0e-12,
    "max_kmers": 500,
    "top_k_values": (50, 100, 200, 500),
    "expected_core_feature_count": 83,
    "core_group_name": "core83",
    "legacy_group_name": "static_tcr_candidate_predictors",
}

_GROUP_COUNTS = {
    "core83": 83,
    "repeat_3mer_both_top500": 1000,
}


def _as_tuple(value: Any, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ModelContractError(f"{label} must be a list/tuple of strings")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise ModelContractError(f"{label} contains an empty value")
    if len(result) != len(set(result)):
        raise ModelContractError(f"{label} contains duplicate values")
    return result


def expected_model_names(cohort: str) -> tuple[str, ...]:
    key = str(cohort).strip().lower()
    if key not in COHORT_MODELS:
        raise ModelContractError(
            f"Unknown cohort {cohort!r}; expected one of {tuple(COHORT_MODELS)}"
        )
    return COHORT_MODELS[key]


def _validate_repeat_3mer_options(options: Mapping[str, Any]) -> None:
    if not isinstance(options, Mapping):
        raise ModelContractError("repeat_3mer_features must be a mapping")
    missing = sorted(set(REPEAT_3MER_EXPECTED) - set(options))
    if missing:
        raise ModelContractError(
            f"repeat_3mer_features missing required contract fields: {missing}"
        )
    if options.get("enabled") is not True:
        raise ModelContractError("repeat_3mer_features.enabled must be true")
    if options.get("selection_scope") != REPEAT_3MER_EXPECTED["selection_scope"]:
        raise ModelContractError(
            "repeat_3mer_features.selection_scope must remain "
            "'outer_repeat_train_only'; a different scope is a methodology change"
        )
    for key in ("ranking_rule", "sequence_column", "core_group_name", "legacy_group_name"):
        if str(options.get(key)) != str(REPEAT_3MER_EXPECTED[key]):
            raise ModelContractError(
                f"repeat_3mer_features.{key} mismatch: "
                f"expected={REPEAT_3MER_EXPECTED[key]!r}, observed={options.get(key)!r}"
            )
    for key in ("min_sample_count", "max_kmers", "expected_core_feature_count"):
        try:
            observed = int(options[key])
        except (TypeError, ValueError) as exc:
            raise ModelContractError(f"repeat_3mer_features.{key} must be an integer") from exc
        if observed != int(REPEAT_3MER_EXPECTED[key]):
            raise ModelContractError(
                f"repeat_3mer_features.{key} mismatch: "
                f"expected={REPEAT_3MER_EXPECTED[key]}, observed={observed}"
            )
    for key in ("min_prevalence", "min_unweighted_variance", "min_weighted_variance"):
        try:
            observed = float(options[key])
        except (TypeError, ValueError) as exc:
            raise ModelContractError(f"repeat_3mer_features.{key} must be numeric") from exc
        expected = float(REPEAT_3MER_EXPECTED[key])
        if not math.isclose(observed, expected, rel_tol=1.0e-12, abs_tol=0.0):
            raise ModelContractError(
                f"repeat_3mer_features.{key} mismatch: expected={expected}, observed={observed}"
            )
    top_k = _as_tuple(options["top_k_values"], "repeat_3mer_features.top_k_values")
    try:
        top_k_int = tuple(int(value) for value in top_k)
    except ValueError as exc:
        raise ModelContractError("repeat_3mer_features.top_k_values must contain integers") from exc
    if top_k_int != REPEAT_3MER_EXPECTED["top_k_values"]:
        raise ModelContractError(
            "repeat_3mer_features.top_k_values mismatch: "
            f"expected={REPEAT_3MER_EXPECTED['top_k_values']}, observed={top_k_int}"
        )
    manifest = options.get("aa_manifest")
    if not isinstance(manifest, str) or not manifest.strip():
        raise ModelContractError("repeat_3mer_features.aa_manifest must be a non-empty path")


def validate_hierarchical_model_contract(
    models: Mapping[str, Mapping[str, Any]],
    *,
    cohort: str,
    repeat_3mer_options: Mapping[str, Any],
    expected_resolved_columns: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, int]]:
    """Validate exact M0-M3 predictors and return raw predictor counts.

    Counts are before one-hot encoding, zero-variance filtering and scaling.
    """
    if not isinstance(models, Mapping):
        raise ModelContractError("models must be a mapping")
    expected_names = expected_model_names(cohort)
    observed_names = tuple(str(name) for name in models)
    if observed_names != expected_names:
        raise ModelContractError(
            f"{cohort} model order/names mismatch: expected={expected_names}, "
            f"observed={observed_names}"
        )
    _validate_repeat_3mer_options(repeat_3mer_options)

    counts: dict[str, dict[str, int]] = {}
    for model_name in expected_names:
        raw = models[model_name]
        if not isinstance(raw, Mapping):
            raise ModelContractError(f"models.{model_name} must be a mapping")
        expected = MODEL_SPECS[model_name]
        for field in ("numeric", "categorical", "static_feature_groups", "dynamic_public"):
            observed = _as_tuple(raw.get(field, []), f"models.{model_name}.{field}")
            if observed != expected[field]:
                raise ModelContractError(
                    f"models.{model_name}.{field} mismatch: "
                    f"expected={expected[field]}, observed={observed}"
                )

        numeric_count = len(expected["numeric"]) + len(expected["dynamic_public"])
        for group in expected["static_feature_groups"]:
            if group not in _GROUP_COUNTS:
                raise ModelContractError(f"No predictor-count rule for static group {group!r}")
            numeric_count += _GROUP_COUNTS[group]
        categorical_count = len(expected["categorical"])
        counts[model_name] = {
            "numeric": numeric_count,
            "categorical": categorical_count,
        }

    if expected_resolved_columns is not None:
        if not isinstance(expected_resolved_columns, Mapping):
            raise ModelContractError("model_selection.expected_resolved_columns must be a mapping")
        for model_name, observed in counts.items():
            raw_count = expected_resolved_columns.get(model_name)
            if not isinstance(raw_count, Mapping):
                raise ModelContractError(
                    f"model_selection.expected_resolved_columns missing {model_name}"
                )
            try:
                supplied = {
                    "numeric": int(raw_count["numeric"]),
                    "categorical": int(raw_count["categorical"]),
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise ModelContractError(
                    f"Invalid expected_resolved_columns for {model_name}"
                ) from exc
            if supplied != observed:
                raise ModelContractError(
                    f"Resolved predictor-count mismatch for {model_name}: "
                    f"expected={observed}, configured={supplied}"
                )
    return counts
