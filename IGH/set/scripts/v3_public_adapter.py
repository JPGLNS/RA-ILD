#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fold-local 8-column V2 Public-reference adapter for IGH V3.

Builds an unlabeled sequence index once for efficiency, then uses only the
*current* training partition to fit public-reference thresholds and membership.
Training samples receive exact leave-one-out features; validation/holdout uses
training-only references without consulting validation labels.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from .dynamic_repertoire import DynamicRepertoireSource
from .public_reference import (
    ThresholdScheme,
    build_reference_from_presence,
    external_public_features,
    leave_one_out_public_features,
    public_feature_columns,
)

V2_PUBLIC_COLUMNS: Tuple[str, ...] = public_feature_columns("raw_bilateral")


class V3PublicAdapterError(ValueError):
    pass


@dataclass(frozen=True)
class PublicInput:
    sample_ids: Tuple[str, ...]
    labels: np.ndarray
    presence: sparse.csr_matrix
    frequency: sparse.csr_matrix
    clone_numbers: np.ndarray
    catalog_size: int
    source_id: str


@dataclass(frozen=True)
class PublicFold:
    train: pd.DataFrame
    valid: pd.DataFrame
    audit: Mapping[str, object]


def _read_clone_frequencies(path: Path, weight_column: str) -> Tuple[Tuple[str, ...], np.ndarray]:
    """Read same repertoire source as V3 3-mers, normalize to a unit sum.

    CDR3-AA presence is per sample; duplicate CDR3-AA rows are combined. A
    Top10000 repertoire is therefore internally consistent for clone ratios
    and abundance sums, instead of silently mixing full-depth denominators.
    """
    try:
        f = pd.read_csv(path, usecols=["cdr3_aa", weight_column])
    except (OSError, ValueError) as exc:
        raise V3PublicAdapterError(f"Cannot read public inputs from {path}: {exc}") from exc
    if f.empty:
        raise V3PublicAdapterError(f"Empty clone table: {path}")
    s = f["cdr3_aa"].astype("string").str.strip().str.upper()
    w = pd.to_numeric(f[weight_column], errors="coerce")
    if s.isna().any() or s.eq("").any() or w.isna().any():
        raise V3PublicAdapterError(f"Missing/invalid CDR3-AA or {weight_column}: {path}")
    if not np.isfinite(w.to_numpy(float)).all() or (w < 0).any():
        raise V3PublicAdapterError(f"Non-finite/negative repertoire weights: {path}")
    group = pd.DataFrame({"cdr3_aa": s.astype(str), "frequency": w.astype(float)})
    group = group.groupby("cdr3_aa", sort=True, as_index=False)["frequency"].sum()
    vals = group["frequency"].to_numpy(dtype=np.float64)
    total = float(vals.sum())
    if not np.isfinite(total) or total <= 0:
        raise V3PublicAdapterError(f"Repertoire frequency sum not positive: {path}")
    vals = vals / total
    return tuple(group["cdr3_aa"].tolist()), vals


def load_public_input(
    sample_ids: Sequence[str], labels: Sequence[str], source: DynamicRepertoireSource,
) -> PublicInput:
    ids = tuple(map(str, sample_ids))
    ys = np.asarray(list(map(str, labels)), dtype="U16")
    if not ids or len(ids) != len(set(ids)) or len(ids) != len(ys):
        raise V3PublicAdapterError("Sample IDs must be unique and match labels")
    ys = np.char.upper(np.char.strip(ys))
    if not set(ys).issubset({"RA", "ILD"}) or set(ys) != {"RA", "ILD"}:
        raise V3PublicAdapterError("Expected RA/ILD labels for public-reference input")

    # The global union is only an unlabeled COLUMN INDEX. Every reference mask,
    # threshold, or RA/ILD membership is fitted on train_rows only.
    lookup: Dict[str, int] = {}
    row: list[int] = []
    col: list[int] = []
    weight: list[float] = []
    clones = np.empty(len(ids), dtype=np.float64)
    for i, sid in enumerate(ids):
        seqs, values = _read_clone_frequencies(source.sample_path(sid), source.weight_column)
        if source.top_k is not None and len(seqs) != int(source.top_k):
            raise V3PublicAdapterError(
                f"Top-K clone count mismatch for {sid}: {len(seqs)} != {source.top_k}"
            )
        clones[i] = len(seqs)
        for seq, value in zip(seqs, values):
            j = lookup.setdefault(seq, len(lookup))
            row.append(i)
            col.append(j)
            weight.append(float(value))
    rows = np.asarray(row, dtype=np.int32)
    cols = np.asarray(col, dtype=np.int32)
    weights = np.asarray(weight, dtype=np.float64)
    shape = (len(ids), len(lookup))
    presence = sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=shape)
    frequency = sparse.csr_matrix((weights, (rows, cols)), shape=shape)
    presence.sort_indices()
    frequency.sort_indices()
    if presence.nnz != len(row):
        raise V3PublicAdapterError("Duplicate sample/CDR3 index in public input")
    if not np.array_equal(presence.indices, frequency.indices):
        raise V3PublicAdapterError("Public presence/frequency CSR layouts differ")
    return PublicInput(ids, ys, presence, frequency, clones, len(lookup), source.source_id)


def _threshold_scheme(config) -> ThresholdScheme:
    return ThresholdScheme(
        name="igh_v3_v2_compatible_public",
        global_min_prevalence=float(config.global_min_prevalence),
        global_min_count=int(config.global_min_count),
        group_min_prevalence=float(config.group_min_prevalence),
        group_min_count=int(config.group_min_count),
        specific_prevalence_delta=float(config.specific_prevalence_delta),
    )


def _frame_for_rows(frame: pd.DataFrame, idx: Sequence[int], ids: Tuple[str, ...]) -> pd.DataFrame:
    missing = set(V2_PUBLIC_COLUMNS) - set(frame.columns)
    if missing:
        raise V3PublicAdapterError(f"Missing public features: {sorted(missing)}")
    data = frame.reindex(list(idx)).loc[:, list(V2_PUBLIC_COLUMNS)]
    if data.isna().any().any():
        raise V3PublicAdapterError("Incorrect public fold features or missing values")
    out = data.reset_index(drop=True)
    out.insert(0, "sample_id", [ids[i] for i in idx])
    return out


def build_public_fold(
    inputs: PublicInput, config, train_ids: Sequence[str], valid_ids: Sequence[str],
    *, context: str,
) -> PublicFold:
    if not config.enabled:
        raise V3PublicAdapterError("Public reference is disabled")
    if config.feature_set != "raw_bilateral":
        raise V3PublicAdapterError("Only the eight-column raw_bilateral V2 public feature set is supported")
    index = {sid: i for i, sid in enumerate(inputs.sample_ids)}
    train = tuple(map(str, train_ids))
    valid = tuple(map(str, valid_ids))
    if not train or not valid or len(train) != len(set(train)) or len(valid) != len(set(valid)):
        raise V3PublicAdapterError("Empty/duplicate fold sample IDs")
    if set(train) & set(valid):
        raise V3PublicAdapterError("Training and validation IDs overlap")
    if (set(train) | set(valid)) - index.keys():
        raise V3PublicAdapterError("Unknown sample IDs")
    tr = [index[s] for s in train]
    va = [index[s] for s in valid]
    scheme = _threshold_scheme(config)
    eps = float(config.epsilon)
    # LOO uses only labels of the current training partition.
    train_result = leave_one_out_public_features(
        inputs.presence, inputs.frequency, tr, inputs.labels,
        inputs.clone_numbers, scheme, eps, context=f"{context}_train_loo",
    )
    # IMPORTANT: external_public_features uses labels for reference_rows only;
    # validation/holdout labels are never consulted for membership/thresholds.
    valid_result = external_public_features(
        inputs.presence, inputs.frequency, tr, va, inputs.labels,
        inputs.clone_numbers, scheme, eps, context=f"{context}_valid_train_reference_only",
    )
    ref = build_reference_from_presence(inputs.presence, tr, inputs.labels, scheme)
    h = hashlib.sha256()
    h.update("\n".join(sorted(train)).encode("utf-8"))
    for name in ("global_mask", "ra_specific_mask", "ild_specific_mask", "shared_mask"):
        h.update(np.packbits(getattr(ref, name)).tobytes())
    audit = {
        "context": context,
        "feature_family": "public_reference",
        "fit_policy": "current_train_exact_LOO;validation_training_reference_only",
        "fit_sample_count": len(train),
        "validation_sample_count": len(valid),
        "fit_sample_hash": hashlib.sha256("\n".join(sorted(train)).encode()).hexdigest(),
        "validation_sample_hash": hashlib.sha256("\n".join(sorted(valid)).encode()).hexdigest(),
        "feature_count": len(V2_PUBLIC_COLUMNS),
        "reference_size": ";".join(f"{k}={v}" for k, v in ref.sizes().items()),
        "reference_hash": h.hexdigest(),
        "reference_global_threshold": ref.global_threshold,
        "reference_RA_threshold": ref.ra_threshold,
        "reference_ILD_threshold": ref.ild_threshold,
        "catalog_size": inputs.catalog_size,
        "source_id": inputs.source_id,
        "denominator_policy": "unique_clones_within_selected_repertoire_source",
        "frequency_policy": "per_sample_normalized_source_weights",
    }
    return PublicFold(
        train=_frame_for_rows(train_result.features, tr, inputs.sample_ids),
        valid=_frame_for_rows(valid_result.features, va, inputs.sample_ids),
        audit=audit,
    )
