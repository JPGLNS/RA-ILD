#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Leakage-controlled public CDR3-AA reference construction and transforms.

This module centralizes the deterministic public-reference definitions currently
implemented in the TRB V1 step-03 and step-05 scripts. Batch 02 introduces it as
an independently tested V2 component; the V1 execution scripts are not yet
rewired to call it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


ALL_PUBLIC_FEATURES: Tuple[str, ...] = (
    "all_ref_public_clone_number",
    "all_ref_public_clone_ratio",
    "all_ref_public_frequency_sum",
    "all_ref_public_set_coverage",
    "RA_specific_ref_clone_number",
    "RA_specific_ref_clone_ratio",
    "RA_specific_ref_frequency_sum",
    "RA_specific_ref_set_coverage",
    "ILD_specific_ref_clone_number",
    "ILD_specific_ref_clone_ratio",
    "ILD_specific_ref_frequency_sum",
    "ILD_specific_ref_set_coverage",
    "shared_ref_clone_number",
    "shared_ref_clone_ratio",
    "shared_ref_frequency_sum",
    "shared_ref_set_coverage",
    "ILD_RA_specific_ref_frequency_delta",
    "ILD_RA_specific_ref_frequency_log_ratio",
)

PUBLIC_FEATURE_SETS: Mapping[str, Tuple[str, ...]] = MappingProxyType(
    {
        "raw_bilateral": (
            "all_ref_public_clone_ratio",
            "all_ref_public_frequency_sum",
            "shared_ref_clone_ratio",
            "shared_ref_frequency_sum",
            "RA_specific_ref_clone_ratio",
            "RA_specific_ref_frequency_sum",
            "ILD_specific_ref_clone_ratio",
            "ILD_specific_ref_frequency_sum",
        ),
        "log_ratio": (
            "all_ref_public_clone_ratio",
            "all_ref_public_frequency_sum",
            "shared_ref_clone_ratio",
            "shared_ref_frequency_sum",
            "ILD_RA_specific_ref_frequency_log_ratio",
        ),
        "all18": ALL_PUBLIC_FEATURES,
    }
)

_REFERENCE_MASK_NAMES: Tuple[str, ...] = (
    "global_mask",
    "ra_specific_mask",
    "ild_specific_mask",
    "shared_mask",
)

_PREFIX_TO_MASK_ATTRIBUTE: Mapping[str, str] = MappingProxyType(
    {
        "all_ref_public": "global_mask",
        "RA_specific_ref": "ra_specific_mask",
        "ILD_specific_ref": "ild_specific_mask",
        "shared_ref": "shared_mask",
    }
)


@dataclass(frozen=True)
class ThresholdScheme:
    """Count/prevalence rules used to define public sequence sets."""

    name: str
    global_min_prevalence: float
    global_min_count: int
    group_min_prevalence: float
    group_min_count: int
    specific_prevalence_delta: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ThresholdScheme.name must be non-empty.")
        for field_name in ("global_min_prevalence", "group_min_prevalence"):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1].")
        if not 0.0 <= float(self.specific_prevalence_delta) <= 1.0:
            raise ValueError("specific_prevalence_delta must be in [0, 1].")
        if int(self.global_min_count) < 1 or int(self.group_min_count) < 1:
            raise ValueError("Minimum public-reference counts must be >= 1.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ThresholdScheme":
        """Create a scheme from the ``public_reference`` YAML section."""
        required = {
            "global_min_prevalence",
            "global_min_count",
            "group_min_prevalence",
            "group_min_count",
            "specific_prevalence_delta",
        }
        missing = sorted(required - set(values))
        if missing:
            raise ValueError(f"Threshold mapping is missing keys: {missing}")
        return cls(
            name=str(values.get("scheme_name", values.get("name", "unnamed"))),
            global_min_prevalence=float(values["global_min_prevalence"]),
            global_min_count=int(values["global_min_count"]),
            group_min_prevalence=float(values["group_min_prevalence"]),
            group_min_count=int(values["group_min_count"]),
            specific_prevalence_delta=float(values["specific_prevalence_delta"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "global_min_prevalence": float(self.global_min_prevalence),
            "global_min_count": int(self.global_min_count),
            "group_min_prevalence": float(self.group_min_prevalence),
            "group_min_count": int(self.group_min_count),
            "specific_prevalence_delta": float(self.specific_prevalence_delta),
        }


@dataclass(frozen=True)
class PublicReference:
    """Boolean sequence masks plus the reference cohort sizes and thresholds."""

    global_mask: np.ndarray
    ra_specific_mask: np.ndarray
    ild_specific_mask: np.ndarray
    shared_mask: np.ndarray
    n_reference: int
    n_ra: int
    n_ild: int
    global_threshold: int
    ra_threshold: int
    ild_threshold: int

    def __post_init__(self) -> None:
        masks = []
        for name in _REFERENCE_MASK_NAMES:
            mask = np.asarray(getattr(self, name), dtype=bool)
            if mask.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional mask.")
            object.__setattr__(self, name, mask)
            masks.append(mask)
        lengths = {len(mask) for mask in masks}
        if len(lengths) != 1:
            raise ValueError("All public-reference masks must have equal length.")
        if self.n_reference != self.n_ra + self.n_ild:
            raise ValueError("n_reference must equal n_ra + n_ild.")
        if self.n_ra < 1 or self.n_ild < 1:
            raise ValueError("Both RA and ILD references must be non-empty.")
        if (
            np.any(self.ra_specific_mask & self.ild_specific_mask)
            or np.any(self.ra_specific_mask & self.shared_mask)
            or np.any(self.ild_specific_mask & self.shared_mask)
        ):
            raise ValueError(
                "RA-specific, ILD-specific, and shared masks must be disjoint."
            )

    @property
    def catalog_size(self) -> int:
        return len(self.global_mask)

    def sizes(self) -> Dict[str, int]:
        return {
            "global": int(np.count_nonzero(self.global_mask)),
            "RA_specific": int(np.count_nonzero(self.ra_specific_mask)),
            "ILD_specific": int(np.count_nonzero(self.ild_specific_mask)),
            "shared": int(np.count_nonzero(self.shared_mask)),
        }

    def thresholds(self) -> Dict[str, int]:
        return {
            "global": int(self.global_threshold),
            "RA": int(self.ra_threshold),
            "ILD": int(self.ild_threshold),
        }

    def mask(self, name: str) -> np.ndarray:
        aliases = {
            "global": "global_mask",
            "global_public": "global_mask",
            "all_ref_public": "global_mask",
            "RA_specific": "ra_specific_mask",
            "RA_specific_ref": "ra_specific_mask",
            "ILD_specific": "ild_specific_mask",
            "ILD_specific_ref": "ild_specific_mask",
            "shared": "shared_mask",
            "shared_ref": "shared_mask",
        }
        try:
            return getattr(self, aliases[name])
        except KeyError as exc:
            raise KeyError(f"Unknown public-reference mask name: {name}") from exc


@dataclass(frozen=True)
class PublicFeatureResult:
    """Features and audit rows produced by a public-reference transform."""

    features: pd.DataFrame
    reference_audit: pd.DataFrame
    assignment_audit: pd.DataFrame


MAIN_THRESHOLD_SCHEME = ThresholdScheme(
    name="main",
    global_min_prevalence=0.02,
    global_min_count=3,
    group_min_prevalence=0.05,
    group_min_count=3,
    specific_prevalence_delta=0.05,
)


def public_feature_columns(feature_set: str) -> Tuple[str, ...]:
    """Return the ordered columns for a configured feature-set name."""
    try:
        return PUBLIC_FEATURE_SETS[feature_set]
    except KeyError as exc:
        raise KeyError(
            f"Unknown public feature set '{feature_set}'. "
            f"Available: {sorted(PUBLIC_FEATURE_SETS)}"
        ) from exc


def effective_count_threshold(
    n_reference: int,
    min_prevalence: float,
    min_count: int,
) -> int:
    """Return ``max(min_count, ceil(min_prevalence * n_reference))``."""
    if n_reference < 1:
        raise ValueError("n_reference must be >= 1.")
    if not 0.0 <= min_prevalence <= 1.0:
        raise ValueError("min_prevalence must be in [0, 1].")
    if min_count < 1:
        raise ValueError("min_count must be >= 1.")
    return max(int(min_count), int(math.ceil(min_prevalence * n_reference)))


def calculate_thresholds(
    scheme: ThresholdScheme,
    n_ra_reference: int,
    n_ild_reference: int,
) -> Dict[str, int]:
    """Calculate effective full, RA, and ILD count thresholds."""
    if n_ra_reference < scheme.group_min_count:
        raise ValueError(
            "Too few RA reference samples: "
            f"observed={n_ra_reference}, required>={scheme.group_min_count}."
        )
    if n_ild_reference < scheme.group_min_count:
        raise ValueError(
            "Too few ILD reference samples: "
            f"observed={n_ild_reference}, required>={scheme.group_min_count}."
        )
    n_reference = n_ra_reference + n_ild_reference
    return {
        "global": effective_count_threshold(
            n_reference,
            scheme.global_min_prevalence,
            scheme.global_min_count,
        ),
        "RA": effective_count_threshold(
            n_ra_reference,
            scheme.group_min_prevalence,
            scheme.group_min_count,
        ),
        "ILD": effective_count_threshold(
            n_ild_reference,
            scheme.group_min_prevalence,
            scheme.group_min_count,
        ),
    }


def _validate_count_arrays(
    ra_counts: np.ndarray,
    ild_counts: np.ndarray,
    n_ra_reference: int,
    n_ild_reference: int,
    *,
    allow_template_overflow: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    ra = np.asarray(ra_counts)
    ild = np.asarray(ild_counts)
    if ra.ndim != 1 or ild.ndim != 1:
        raise ValueError("RA and ILD count arrays must be one-dimensional.")
    if ra.shape != ild.shape:
        raise ValueError("RA and ILD count arrays must have identical shapes.")
    if not np.issubdtype(ra.dtype, np.number) or not np.issubdtype(
        ild.dtype, np.number
    ):
        raise TypeError("RA and ILD count arrays must be numeric.")
    if not np.isfinite(ra).all() or not np.isfinite(ild).all():
        raise ValueError("RA and ILD count arrays contain non-finite values.")
    if np.any(ra < 0) or np.any(ild < 0):
        raise ValueError("RA and ILD count arrays cannot contain negative values.")
    if not allow_template_overflow and (
        np.any(ra > n_ra_reference) or np.any(ild > n_ild_reference)
    ):
        raise ValueError("A sequence count exceeds its cohort reference size.")
    return ra.astype(np.int64, copy=False), ild.astype(np.int64, copy=False)


def build_reference_from_counts(
    ra_counts: np.ndarray,
    ild_counts: np.ndarray,
    n_ra_reference: int,
    n_ild_reference: int,
    scheme: ThresholdScheme,
    *,
    allow_template_overflow: bool = False,
) -> PublicReference:
    """Build V1-compatible public masks from per-cohort sample counts."""
    thresholds = calculate_thresholds(
        scheme,
        n_ra_reference=n_ra_reference,
        n_ild_reference=n_ild_reference,
    )
    ra, ild = _validate_count_arrays(
        ra_counts,
        ild_counts,
        n_ra_reference,
        n_ild_reference,
        allow_template_overflow=allow_template_overflow,
    )

    total = ra + ild
    ra_prevalence = ra.astype(np.float64) / n_ra_reference
    ild_prevalence = ild.astype(np.float64) / n_ild_reference

    global_mask = total >= thresholds["global"]
    ra_group = ra >= thresholds["RA"]
    ild_group = ild >= thresholds["ILD"]

    ra_specific = ra_group & (
        (ra_prevalence - ild_prevalence) >= scheme.specific_prevalence_delta
    )
    ild_specific = ild_group & (
        (ild_prevalence - ra_prevalence) >= scheme.specific_prevalence_delta
    )
    if np.any(ra_specific & ild_specific):
        raise RuntimeError("RA-specific and ILD-specific masks overlap.")

    shared = ra_group & ild_group & ~ra_specific & ~ild_specific

    return PublicReference(
        global_mask=global_mask,
        ra_specific_mask=ra_specific,
        ild_specific_mask=ild_specific,
        shared_mask=shared,
        n_reference=n_ra_reference + n_ild_reference,
        n_ra=n_ra_reference,
        n_ild=n_ild_reference,
        global_threshold=thresholds["global"],
        ra_threshold=thresholds["RA"],
        ild_threshold=thresholds["ILD"],
    )


def _as_csr(matrix: sparse.spmatrix, label: str) -> sparse.csr_matrix:
    if not sparse.issparse(matrix):
        raise TypeError(f"{label} must be a scipy sparse matrix.")
    output = matrix.tocsr(copy=False)
    if output.ndim != 2:
        raise ValueError(f"{label} must be two-dimensional.")
    return output


def _normalize_labels(labels: Sequence[Any], expected_rows: int) -> np.ndarray:
    values = np.asarray(labels).astype(str)
    if values.ndim != 1 or len(values) != expected_rows:
        raise ValueError(
            f"labels must have length {expected_rows}; observed shape={values.shape}."
        )
    values = np.char.upper(np.char.strip(values))
    unexpected = sorted(set(values) - {"RA", "ILD"})
    if unexpected:
        raise ValueError(f"labels contains values other than RA/ILD: {unexpected}")
    return values


def _normalize_rows(rows: Sequence[int], n_rows: int, label: str) -> np.ndarray:
    values = np.asarray(rows, dtype=int)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional sequence.")
    if np.any(values < 0) or np.any(values >= n_rows):
        raise IndexError(f"{label} contains an out-of-range row index.")
    if len(values) != len(np.unique(values)):
        raise ValueError(f"{label} contains duplicate row indices.")
    return values


def build_reference_from_presence(
    presence: sparse.spmatrix,
    reference_rows: Sequence[int],
    labels: Sequence[Any],
    scheme: ThresholdScheme,
) -> PublicReference:
    """Build a public reference from selected rows of a presence matrix."""
    presence_csr = _as_csr(presence, "presence")
    rows = _normalize_rows(reference_rows, presence_csr.shape[0], "reference_rows")
    normalized_labels = _normalize_labels(labels, presence_csr.shape[0])
    local_labels = normalized_labels[rows]
    ra_rows = rows[local_labels == "RA"]
    ild_rows = rows[local_labels == "ILD"]

    ra_counts = np.asarray(presence_csr[ra_rows].sum(axis=0)).ravel()
    ild_counts = np.asarray(presence_csr[ild_rows].sum(axis=0)).ravel()
    return build_reference_from_counts(
        ra_counts=ra_counts,
        ild_counts=ild_counts,
        n_ra_reference=len(ra_rows),
        n_ild_reference=len(ild_rows),
        scheme=scheme,
    )


def _validate_transform_inputs(
    presence: sparse.spmatrix,
    frequency: sparse.spmatrix,
    aa_clone_numbers: Sequence[float],
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray]:
    presence_csr = _as_csr(presence, "presence")
    frequency_csr = _as_csr(frequency, "frequency")
    if presence_csr.shape != frequency_csr.shape:
        raise ValueError("presence and frequency matrices must have equal shape.")
    clone_numbers = np.asarray(aa_clone_numbers, dtype=float)
    if clone_numbers.ndim != 1 or len(clone_numbers) != presence_csr.shape[0]:
        raise ValueError(
            "aa_clone_numbers must have one value per sparse-matrix row."
        )
    if not np.isfinite(clone_numbers).all() or np.any(clone_numbers < 0):
        raise ValueError("aa_clone_numbers contains invalid values.")
    if frequency_csr.nnz and (
        not np.isfinite(frequency_csr.data).all()
        or np.any(frequency_csr.data < 0)
    ):
        raise ValueError("frequency contains negative or non-finite values.")
    return presence_csr, frequency_csr, clone_numbers


def apply_reference(
    presence: sparse.spmatrix,
    frequency: sparse.spmatrix,
    target_rows: Sequence[int],
    aa_clone_numbers: Sequence[float],
    reference: PublicReference,
    epsilon: float,
) -> pd.DataFrame:
    """Apply a fixed public reference to target samples."""
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0.")
    presence_csr, frequency_csr, clone_numbers = _validate_transform_inputs(
        presence,
        frequency,
        aa_clone_numbers,
    )
    rows = _normalize_rows(target_rows, presence_csr.shape[0], "target_rows")
    if reference.catalog_size != presence_csr.shape[1]:
        raise ValueError(
            "Reference mask length does not match sparse-matrix column count."
        )

    target_presence = presence_csr[rows]
    target_frequency = frequency_csr[rows]
    result: Dict[str, np.ndarray] = {}

    for prefix, attribute in _PREFIX_TO_MASK_ATTRIBUTE.items():
        mask = getattr(reference, attribute)
        set_size = int(np.count_nonzero(mask))
        clone_number = np.asarray(
            target_presence[:, mask].sum(axis=1)
        ).ravel().astype(float)
        frequency_sum = np.asarray(
            target_frequency[:, mask].sum(axis=1)
        ).ravel().astype(float)

        result[f"{prefix}_clone_number"] = clone_number
        result[f"{prefix}_clone_ratio"] = np.divide(
            clone_number,
            clone_numbers[rows],
            out=np.zeros_like(clone_number, dtype=float),
            where=clone_numbers[rows] > 0,
        )
        result[f"{prefix}_frequency_sum"] = frequency_sum
        result[f"{prefix}_set_coverage"] = (
            clone_number / set_size if set_size > 0 else np.zeros_like(clone_number)
        )

    ild_sum = result["ILD_specific_ref_frequency_sum"]
    ra_sum = result["RA_specific_ref_frequency_sum"]
    result["ILD_RA_specific_ref_frequency_delta"] = ild_sum - ra_sum
    result["ILD_RA_specific_ref_frequency_log_ratio"] = np.log(
        (ild_sum + epsilon) / (ra_sum + epsilon)
    )

    output = pd.DataFrame(result, index=rows)
    output.index.name = "row_index"
    return output.loc[:, list(ALL_PUBLIC_FEATURES)]


def reference_audit_row(
    reference: PublicReference,
    *,
    context: str,
    generation_method: str,
    held_out_row_index: Optional[int] = None,
    held_out_label: Optional[str] = None,
    set_sizes: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    """Create one normalized audit row for a reference construction."""
    sizes = reference.sizes()
    if set_sizes is not None:
        sizes = {
            "global": int(set_sizes["all_ref_public"]),
            "RA_specific": int(set_sizes["RA_specific_ref"]),
            "ILD_specific": int(set_sizes["ILD_specific_ref"]),
            "shared": int(set_sizes["shared_ref"]),
        }
    return {
        "context": context,
        "generation_method": generation_method,
        "held_out_row_index": "" if held_out_row_index is None else int(held_out_row_index),
        "held_out_label": "" if held_out_label is None else str(held_out_label),
        "n_reference": int(reference.n_reference),
        "n_RA_reference": int(reference.n_ra),
        "n_ILD_reference": int(reference.n_ild),
        "global_threshold": int(reference.global_threshold),
        "RA_threshold": int(reference.ra_threshold),
        "ILD_threshold": int(reference.ild_threshold),
        "all_ref_public_size": sizes["global"],
        "RA_specific_ref_size": sizes["RA_specific"],
        "ILD_specific_ref_size": sizes["ILD_specific"],
        "shared_ref_size": sizes["shared"],
    }


def _single_sample_from_loo_templates(
    presence_row: sparse.csr_matrix,
    frequency_row: sparse.csr_matrix,
    aa_clone_number: float,
    absent_reference: PublicReference,
    present_reference: PublicReference,
    epsilon: float,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Compute exact LOO values using absent/present mask templates."""
    presence_row = presence_row.tocsr(copy=True)
    frequency_row = frequency_row.tocsr(copy=True)
    presence_row.sort_indices()
    frequency_row.sort_indices()

    columns = presence_row.indices
    frequency_columns = frequency_row.indices
    if np.array_equal(columns, frequency_columns):
        frequency_values = frequency_row.data.astype(np.float64, copy=False)
    else:
        frequency_values = np.asarray(
            frequency_row[:, columns].toarray()
        ).ravel()

    result: Dict[str, float] = {}
    set_sizes: Dict[str, int] = {}

    for prefix, attribute in _PREFIX_TO_MASK_ATTRIBUTE.items():
        absent_mask = getattr(absent_reference, attribute)
        present_mask = getattr(present_reference, attribute)
        present_status = present_mask[columns]
        absent_status_at_sample = absent_mask[columns]

        set_size = int(np.count_nonzero(absent_mask)) + int(
            np.sum(
                present_status.astype(np.int8)
                - absent_status_at_sample.astype(np.int8)
            )
        )
        if set_size < 0:
            raise RuntimeError(f"Negative LOO set size for {prefix}.")

        clone_number = float(np.count_nonzero(present_status))
        frequency_sum = float(np.sum(frequency_values[present_status]))
        result[f"{prefix}_clone_number"] = clone_number
        result[f"{prefix}_clone_ratio"] = (
            clone_number / aa_clone_number if aa_clone_number > 0 else 0.0
        )
        result[f"{prefix}_frequency_sum"] = frequency_sum
        result[f"{prefix}_set_coverage"] = (
            clone_number / set_size if set_size > 0 else 0.0
        )
        set_sizes[prefix] = set_size

    ild_sum = result["ILD_specific_ref_frequency_sum"]
    ra_sum = result["RA_specific_ref_frequency_sum"]
    result["ILD_RA_specific_ref_frequency_delta"] = ild_sum - ra_sum
    result["ILD_RA_specific_ref_frequency_log_ratio"] = float(
        np.log((ild_sum + epsilon) / (ra_sum + epsilon))
    )
    return result, set_sizes


def leave_one_out_public_features(
    presence: sparse.spmatrix,
    frequency: sparse.spmatrix,
    sample_rows: Sequence[int],
    labels: Sequence[Any],
    aa_clone_numbers: Sequence[float],
    scheme: ThresholdScheme,
    epsilon: float,
    *,
    context: str = "training_leave_one_out",
) -> PublicFeatureResult:
    """Generate exact LOO public features for model-fitting samples."""
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0.")
    presence_csr, frequency_csr, clone_numbers = _validate_transform_inputs(
        presence,
        frequency,
        aa_clone_numbers,
    )
    rows = _normalize_rows(sample_rows, presence_csr.shape[0], "sample_rows")
    normalized_labels = _normalize_labels(labels, presence_csr.shape[0])
    local_labels = normalized_labels[rows]
    ra_rows = rows[local_labels == "RA"]
    ild_rows = rows[local_labels == "ILD"]

    if len(ra_rows) <= scheme.group_min_count:
        raise ValueError(
            "Too few RA samples for leave-one-out reference: "
            f"observed={len(ra_rows)}, need>{scheme.group_min_count}."
        )
    if len(ild_rows) <= scheme.group_min_count:
        raise ValueError(
            "Too few ILD samples for leave-one-out reference: "
            f"observed={len(ild_rows)}, need>{scheme.group_min_count}."
        )

    ra_counts = np.asarray(presence_csr[ra_rows].sum(axis=0)).ravel().astype(np.int64)
    ild_counts = np.asarray(presence_csr[ild_rows].sum(axis=0)).ravel().astype(np.int64)

    templates: Dict[str, Tuple[PublicReference, PublicReference]] = {}
    for held_label in ("RA", "ILD"):
        n_ra_loo = len(ra_rows) - (1 if held_label == "RA" else 0)
        n_ild_loo = len(ild_rows) - (1 if held_label == "ILD" else 0)

        absent_reference = build_reference_from_counts(
            ra_counts,
            ild_counts,
            n_ra_loo,
            n_ild_loo,
            scheme,
            allow_template_overflow=True,
        )
        if held_label == "RA":
            present_ra = np.maximum(ra_counts - 1, 0)
            present_ild = ild_counts
        else:
            present_ra = ra_counts
            present_ild = np.maximum(ild_counts - 1, 0)
        present_reference = build_reference_from_counts(
            present_ra,
            present_ild,
            n_ra_loo,
            n_ild_loo,
            scheme,
        )
        templates[held_label] = (absent_reference, present_reference)

    feature_rows = []
    reference_rows = []
    assignment_rows = []
    for row in rows:
        held_label = str(normalized_labels[row])
        absent_reference, present_reference = templates[held_label]
        values, set_sizes = _single_sample_from_loo_templates(
            presence_csr.getrow(row),
            frequency_csr.getrow(row),
            float(clone_numbers[row]),
            absent_reference,
            present_reference,
            epsilon,
        )
        feature_rows.append(values)
        reference_rows.append(
            reference_audit_row(
                absent_reference,
                context=context,
                generation_method="leave_one_out",
                held_out_row_index=int(row),
                held_out_label=held_label,
                set_sizes=set_sizes,
            )
        )
        assignment_rows.append(
            {
                "context": context,
                "generation_method": "leave_one_out",
                "held_out_row_index": int(row),
                "held_out_label": held_label,
                "n_partition_samples": int(len(rows)),
                "n_reference_samples": int(len(rows) - 1),
            }
        )

    features = pd.DataFrame(feature_rows, index=rows)
    features.index.name = "row_index"
    features = features.loc[:, list(ALL_PUBLIC_FEATURES)].sort_index()
    if features.isna().any().any():
        raise RuntimeError(f"Missing LOO public values in {context}.")
    return PublicFeatureResult(
        features=features,
        reference_audit=pd.DataFrame(reference_rows),
        assignment_audit=pd.DataFrame(assignment_rows),
    )


def external_public_features(
    presence: sparse.spmatrix,
    frequency: sparse.spmatrix,
    reference_rows: Sequence[int],
    target_rows: Sequence[int],
    labels: Sequence[Any],
    aa_clone_numbers: Sequence[float],
    scheme: ThresholdScheme,
    epsilon: float,
    *,
    context: str = "external_application",
) -> PublicFeatureResult:
    """Build from reference rows and apply once to external/validation rows."""
    presence_csr, _, _ = _validate_transform_inputs(
        presence,
        frequency,
        aa_clone_numbers,
    )
    ref_rows = _normalize_rows(
        reference_rows,
        presence_csr.shape[0],
        "reference_rows",
    )
    targets = _normalize_rows(target_rows, presence_csr.shape[0], "target_rows")
    if set(ref_rows).intersection(set(targets)):
        raise ValueError("reference_rows and target_rows must be disjoint.")

    reference = build_reference_from_presence(
        presence_csr,
        ref_rows,
        labels,
        scheme,
    )
    features = apply_reference(
        presence,
        frequency,
        targets,
        aa_clone_numbers,
        reference,
        epsilon,
    )
    audit = pd.DataFrame(
        [
            reference_audit_row(
                reference,
                context=context,
                generation_method="external_reference_application",
            )
        ]
    )
    assignment = pd.DataFrame(
        [
            {
                "context": context,
                "generation_method": "external_reference_application",
                "n_reference_samples": int(len(ref_rows)),
                "n_target_samples": int(len(targets)),
            }
        ]
    )
    return PublicFeatureResult(
        features=features,
        reference_audit=audit,
        assignment_audit=assignment,
    )
