#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared utilities for step 03 public CDR3-AA analyses.

This module contains no train/test file paths and does not inspect outcome
performance. It provides deterministic definitions used by:
  1) threshold-stability evaluation;
  2) final public-feature construction;
  3) future fold-specific model preprocessing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


CATEGORY_ORDER = (
    "global_public",
    "RA_specific",
    "ILD_specific",
    "shared",
)


@dataclass(frozen=True)
class ThresholdScheme:
    name: str
    global_min_prevalence: float
    global_min_count: int
    group_min_prevalence: float
    group_min_count: int
    specific_prevalence_delta: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


DEFAULT_SCHEMES: Tuple[ThresholdScheme, ...] = (
    ThresholdScheme(
        name="loose",
        global_min_prevalence=0.01,
        global_min_count=2,
        group_min_prevalence=0.03,
        group_min_count=2,
        specific_prevalence_delta=0.03,
    ),
    ThresholdScheme(
        name="main",
        global_min_prevalence=0.02,
        global_min_count=3,
        group_min_prevalence=0.05,
        group_min_count=3,
        specific_prevalence_delta=0.05,
    ),
    ThresholdScheme(
        name="strict",
        global_min_prevalence=0.05,
        global_min_count=5,
        group_min_prevalence=0.10,
        group_min_count=4,
        specific_prevalence_delta=0.10,
    ),
)


def effective_count_threshold(
    n_reference: int,
    min_prevalence: float,
    min_count: int,
) -> int:
    """Return max(min_count, ceil(min_prevalence * n_reference))."""
    if n_reference < 1:
        raise ValueError("n_reference must be >= 1.")
    if not 0.0 <= min_prevalence <= 1.0:
        raise ValueError("min_prevalence must be in [0, 1].")
    if min_count < 1:
        raise ValueError("min_count must be >= 1.")
    return max(int(min_count), int(math.ceil(min_prevalence * n_reference)))


def calculate_scheme_thresholds(
    scheme: ThresholdScheme,
    n_total: int,
    n_ra: int,
    n_ild: int,
) -> Dict[str, int]:
    if n_total != n_ra + n_ild:
        raise ValueError(
            f"n_total ({n_total}) != n_ra + n_ild ({n_ra + n_ild})."
        )
    if n_ra < 1 or n_ild < 1:
        raise ValueError("Both RA and ILD reference groups must be non-empty.")

    return {
        "global_count_threshold": effective_count_threshold(
            n_total,
            scheme.global_min_prevalence,
            scheme.global_min_count,
        ),
        "RA_count_threshold": effective_count_threshold(
            n_ra,
            scheme.group_min_prevalence,
            scheme.group_min_count,
        ),
        "ILD_count_threshold": effective_count_threshold(
            n_ild,
            scheme.group_min_prevalence,
            scheme.group_min_count,
        ),
    }


def build_reference_masks(
    ra_sample_counts: np.ndarray,
    ild_sample_counts: np.ndarray,
    n_ra_reference: int,
    n_ild_reference: int,
    scheme: ThresholdScheme,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """
    Build mutually interpretable reference masks.

    Definitions
    -----------
    global_public:
        total reference sample count reaches the global threshold.

    RA_specific:
        RA sample count reaches the RA group-public threshold AND
        RA prevalence - ILD prevalence >= specific delta.

    ILD_specific:
        ILD sample count reaches the ILD group-public threshold AND
        ILD prevalence - RA prevalence >= specific delta.

    shared:
        reaches both RA and ILD group-public thresholds, but is not classified
        as either specific set.

    RA_specific and ILD_specific cannot overlap when delta > 0.
    shared is explicitly made disjoint from the two specific sets.
    """
    ra_counts = np.asarray(ra_sample_counts)
    ild_counts = np.asarray(ild_sample_counts)

    if ra_counts.shape != ild_counts.shape:
        raise ValueError("RA and ILD count arrays must have identical shapes.")
    if n_ra_reference < 1 or n_ild_reference < 1:
        raise ValueError("Both reference groups must contain at least one sample.")

    n_total = n_ra_reference + n_ild_reference
    thresholds = calculate_scheme_thresholds(
        scheme=scheme,
        n_total=n_total,
        n_ra=n_ra_reference,
        n_ild=n_ild_reference,
    )

    total_counts = ra_counts + ild_counts
    ra_prevalence = ra_counts.astype(np.float64) / n_ra_reference
    ild_prevalence = ild_counts.astype(np.float64) / n_ild_reference

    global_public = total_counts >= thresholds["global_count_threshold"]
    ra_group_public = ra_counts >= thresholds["RA_count_threshold"]
    ild_group_public = ild_counts >= thresholds["ILD_count_threshold"]

    ra_specific = (
        ra_group_public
        & ((ra_prevalence - ild_prevalence) >= scheme.specific_prevalence_delta)
    )
    ild_specific = (
        ild_group_public
        & ((ild_prevalence - ra_prevalence) >= scheme.specific_prevalence_delta)
    )

    # Defensive disjointness check.
    if np.any(ra_specific & ild_specific):
        raise RuntimeError("RA_specific and ILD_specific unexpectedly overlap.")

    shared = (
        ra_group_public
        & ild_group_public
        & ~ra_specific
        & ~ild_specific
    )

    masks = {
        "global_public": global_public,
        "RA_specific": ra_specific,
        "ILD_specific": ild_specific,
        "shared": shared,
    }
    return masks, thresholds


def jaccard_similarity(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Jaccard index for two boolean masks; two empty sets are defined as 1."""
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("Jaccard masks must have identical shapes.")
    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return 1.0 if union == 0 else intersection / union


def stratified_reference_split(
    metadata: pd.DataFrame,
    reference_fraction: float,
    rng: np.random.Generator,
    strata_columns: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return reference and held-out row indices using stratified subsampling.

    For strata with n >= 2, at least one sample is left held out and at least
    one remains in reference. A singleton stratum is kept in reference because
    it cannot be split while preserving the stratum.
    """
    if not 0.0 < reference_fraction < 1.0:
        raise ValueError("reference_fraction must be between 0 and 1.")
    if not strata_columns:
        raise ValueError("At least one strata column is required.")

    missing = [c for c in strata_columns if c not in metadata.columns]
    if missing:
        raise ValueError(f"Missing strata columns: {missing}")

    ref_indices: List[int] = []
    held_indices: List[int] = []

    grouped = metadata.groupby(
        list(strata_columns),
        sort=True,
        dropna=False,
        observed=False,
    )

    for _, group in grouped:
        indices = group.index.to_numpy(dtype=int)
        shuffled = rng.permutation(indices)
        n = len(shuffled)

        if n == 1:
            n_ref = 1
        else:
            n_ref = int(round(reference_fraction * n))
            n_ref = min(n - 1, max(1, n_ref))

        ref_indices.extend(shuffled[:n_ref].tolist())
        held_indices.extend(shuffled[n_ref:].tolist())

    ref = np.array(sorted(ref_indices), dtype=int)
    held = np.array(sorted(held_indices), dtype=int)

    if len(ref) == 0 or len(held) == 0:
        raise RuntimeError(
            "Stratified split produced an empty reference or held-out subset."
        )
    if set(ref).intersection(set(held)):
        raise RuntimeError("Reference and held-out indices overlap.")
    if len(ref) + len(held) != len(metadata):
        raise RuntimeError("Split does not cover all metadata rows.")

    return ref, held


def classify_full_reference_category(
    masks: Mapping[str, np.ndarray],
) -> np.ndarray:
    """
    Return one category string per sequence using an exclusive priority order.

    Priority:
      RA_specific -> ILD_specific -> shared -> other_global_public -> not_public
    """
    required = set(CATEGORY_ORDER)
    missing = required - set(masks)
    if missing:
        raise ValueError(f"Missing masks: {sorted(missing)}")

    n = len(masks["global_public"])
    category = np.full(n, "not_public", dtype=object)
    category[np.asarray(masks["global_public"], dtype=bool)] = "other_global_public"
    category[np.asarray(masks["shared"], dtype=bool)] = "shared"
    category[np.asarray(masks["ILD_specific"], dtype=bool)] = "ILD_specific"
    category[np.asarray(masks["RA_specific"], dtype=bool)] = "RA_specific"
    return category
