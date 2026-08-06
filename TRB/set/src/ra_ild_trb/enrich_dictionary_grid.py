#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Threshold-grid repeated 5-fold validation for enrich dictionaries.

Batch 17 extends Batch 16 without changing the single-threshold workflow.
For every scope and repeat, one optimized stratified fold partition is reused
for every threshold pair.  This makes threshold combinations directly
comparable and avoids threshold-specific split noise.

Parallelism is Linux process-based.  The unit of work is one complete repeat
within one scope, containing all five folds and every valid threshold pair.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
import shutil
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .enrich_dictionary_cv import (
    ALL_METRICS,
    DIAGNOSTIC_METRICS,
    EXPECTED_SCOPE_COUNTS,
    METRIC_LABELS,
    PRIMARY_METRICS,
    ScopeDataset,
    aligned_presence_counts,
    build_aa_file_map,
    build_scope_dataset,
    derived_seed,
    direction_fields,
    file_sha256,
    fold_balance_rows,
    load_metadata,
    load_repertoires,
    optimize_split,
    rankdata_average,
    validate_fold_assignment,
    _axis_boxplot,
    _import_matplotlib,
)

SCRIPT_VERSION = "1.0.0"
MAX_WORKERS = 16
POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint8)


@dataclass(frozen=True)
class ThresholdPair:
    threshold_id: str
    threshold_pct: float
    delta_pct: float


@dataclass(frozen=True)
class SplitSpec:
    repeat: int
    folds: np.ndarray
    split_quality_score: float
    chosen_seed: int
    partition_signature: str


@dataclass
class GridScopeResult:
    scope: str
    threshold_grid: pd.DataFrame
    assignments: pd.DataFrame
    sample_averaged_scores: pd.DataFrame
    auc_by_repeat: pd.DataFrame
    auc_by_fold: pd.DataFrame
    threshold_summary: pd.DataFrame
    best_thresholds: pd.DataFrame
    fold_dictionary_summary: pd.DataFrame
    stability: pd.DataFrame
    stability_summary: pd.DataFrame
    richness_correlation: pd.DataFrame
    runtime_seconds: float


_WORKER_DATASET: Optional[ScopeDataset] = None
_WORKER_PAIRS: Optional[Tuple[ThresholdPair, ...]] = None
_WORKER_FOLDS: int = 5
_WORKER_SAVE_FOLD_AUC: bool = True


def _decimal(value: object, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("Invalid {}: {!r}".format(name, value)) from exc
    if not result.is_finite():
        raise ValueError("{} must be finite".format(name))
    return result


def inclusive_decimal_range(start: object, stop: object, step: object) -> List[float]:
    start_d = _decimal(start, "range start")
    stop_d = _decimal(stop, "range stop")
    step_d = _decimal(step, "range step")
    if step_d <= 0:
        raise ValueError("Range step must be > 0")
    if stop_d < start_d:
        raise ValueError("Range stop must be >= start")
    values: List[float] = []
    current = start_d
    while current <= stop_d:
        values.append(float(current))
        current += step_d
    return values


def parse_numeric_list(text: Optional[str], name: str) -> Optional[List[float]]:
    if text is None or str(text).strip() == "":
        return None
    values: List[float] = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            values.append(float(_decimal(token, name)))
    if not values:
        raise ValueError("{} produced no values".format(name))
    return sorted(set(values))


def format_number(value: float) -> str:
    text = ("{:.10f}".format(float(value))).rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def generate_threshold_pairs(
    threshold_values: Sequence[float],
    delta_values: Sequence[float],
) -> Tuple[ThresholdPair, ...]:
    t_values = sorted(set(float(x) for x in threshold_values))
    d_values = sorted(set(float(x) for x in delta_values))
    if not t_values or not d_values:
        raise ValueError("Threshold and delta value sets must be non-empty")
    if min(t_values) <= 0 or min(d_values) < 0:
        raise ValueError("Threshold values must be >0 and delta values must be >=0")
    pairs: List[ThresholdPair] = []
    for threshold_pct in t_values:
        for delta_pct in d_values:
            if delta_pct <= threshold_pct + 1e-12:
                pairs.append(
                    ThresholdPair(
                        threshold_id="T{}_D{}".format(
                            format_number(threshold_pct), format_number(delta_pct)
                        ),
                        threshold_pct=float(threshold_pct),
                        delta_pct=float(delta_pct),
                    )
                )
    if not pairs:
        raise ValueError("No valid threshold pairs satisfy delta <= threshold")
    return tuple(pairs)


def threshold_grid_frame(pairs: Sequence[ThresholdPair]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "threshold_index": index,
                "threshold_id": pair.threshold_id,
                "threshold_pct": pair.threshold_pct,
                "delta_pct": pair.delta_pct,
                "valid_rule": "delta_pct <= threshold_pct",
            }
            for index, pair in enumerate(pairs, start=1)
        ]
    )


def validate_workers(workers: int) -> int:
    value = int(workers)
    if value < 1 or value > MAX_WORKERS:
        raise ValueError("workers must be between 1 and {} inclusive; got {}".format(MAX_WORKERS, value))
    return value


def _metric_matrix(
    ra_count: np.ndarray,
    ild_count: np.ndarray,
    ra_rf: np.ndarray,
    ild_rf: np.ndarray,
    ra_size: np.ndarray,
    ild_size: np.ndarray,
    sample_total: int,
) -> np.ndarray:
    ra_count_f = ra_count.astype(np.float64)
    ild_count_f = ild_count.astype(np.float64)
    ra_hit_rate = np.divide(
        ra_count_f,
        ra_size,
        out=np.full(ra_count_f.shape, np.nan, dtype=np.float64),
        where=ra_size > 0,
    )
    ild_hit_rate = np.divide(
        ild_count_f,
        ild_size,
        out=np.full(ild_count_f.shape, np.nan, dtype=np.float64),
        where=ild_size > 0,
    )
    if sample_total > 0:
        ra_sample_fraction = ra_count_f / float(sample_total)
        ild_sample_fraction = ild_count_f / float(sample_total)
    else:
        ra_sample_fraction = np.full(ra_count_f.shape, np.nan, dtype=np.float64)
        ild_sample_fraction = np.full(ild_count_f.shape, np.nan, dtype=np.float64)
    matrix = np.column_stack(
        [
            ra_count_f,
            ild_count_f,
            ra_hit_rate,
            ild_hit_rate,
            ra_rf,
            ild_rf,
            ra_count_f - ild_count_f,
            ra_rf - ild_rf,
            ra_hit_rate - ild_hit_rate,
            ra_sample_fraction,
            ild_sample_fraction,
            ra_sample_fraction - ild_sample_fraction,
        ]
    )
    return matrix


def _auc_one(labels: np.ndarray, values: np.ndarray) -> Tuple[float, int]:
    valid = np.isfinite(values)
    y = labels[valid]
    x = values[valid]
    n = int(len(x))
    ra = x[y == "RA"]
    ild = x[y == "ILD"]
    if ra.size == 0 or ild.size == 0:
        return float("nan"), n
    comparisons = ra[:, None] - ild[None, :]
    auc = float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)
    return auc, n


def auc_matrix(labels: Sequence[str], score_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    labels_array = np.asarray(labels, dtype=str)
    scores = np.asarray(score_matrix, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores[:, None]
    n_columns = scores.shape[1]
    aucs = np.full(n_columns, np.nan, dtype=np.float64)
    n_valid = np.zeros(n_columns, dtype=np.int32)
    finite_all = np.isfinite(scores).all(axis=0)
    if np.any(finite_all):
        valid_scores = scores[:, finite_all]
        ra = valid_scores[labels_array == "RA"]
        ild = valid_scores[labels_array == "ILD"]
        if ra.shape[0] and ild.shape[0]:
            # Chunk columns to cap temporary pairwise arrays.
            selected = np.flatnonzero(finite_all)
            chunk_size = 256
            for start in range(0, valid_scores.shape[1], chunk_size):
                stop = min(start + chunk_size, valid_scores.shape[1])
                diff = ra[:, None, start:stop] - ild[None, :, start:stop]
                block = (np.sum(diff > 0, axis=(0, 1)) + 0.5 * np.sum(diff == 0, axis=(0, 1))) / float(ra.shape[0] * ild.shape[0])
                aucs[selected[start:stop]] = block
                n_valid[selected[start:stop]] = len(labels_array)
    for column in np.flatnonzero(~finite_all):
        aucs[column], n_valid[column] = _auc_one(labels_array, scores[:, column])
    return aucs, n_valid


def _packed_jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter = POPCOUNT[np.bitwise_and(a, b)].sum(axis=0, dtype=np.int64)
    union = POPCOUNT[np.bitwise_or(a, b)].sum(axis=0, dtype=np.int64)
    return np.divide(
        inter,
        union,
        out=np.ones(inter.shape, dtype=np.float64),
        where=union > 0,
    )


def prepare_splits(
    dataset: ScopeDataset,
    folds: int,
    repeats: int,
    candidates: int,
    seed: int,
) -> List[SplitSpec]:
    balance_cols = [c for c in ("batch", "material", "sex") if c in dataset.metadata.columns]
    if dataset.scope != "total":
        balance_cols = [c for c in balance_cols if c != "material"]
    used: Set[str] = set()
    specs: List[SplitSpec] = []
    for repeat in range(1, repeats + 1):
        fold_ids, score, chosen_seed, signature = optimize_split(
            dataset.metadata,
            folds,
            candidates,
            derived_seed(seed, dataset.scope, repeat),
            balance_cols,
            forbidden=used,
        )
        used.add(signature)
        validate_fold_assignment(dataset.metadata, fold_ids, folds)
        specs.append(
            SplitSpec(
                repeat=repeat,
                folds=fold_ids.astype(np.int16, copy=True),
                split_quality_score=float(score),
                chosen_seed=int(chosen_seed),
                partition_signature=str(signature),
            )
        )
    return specs


def _set_worker_context(
    dataset: ScopeDataset,
    pairs: Sequence[ThresholdPair],
    folds: int,
    save_fold_auc: bool,
) -> None:
    global _WORKER_DATASET, _WORKER_PAIRS, _WORKER_FOLDS, _WORKER_SAVE_FOLD_AUC
    _WORKER_DATASET = dataset
    _WORKER_PAIRS = tuple(pairs)
    _WORKER_FOLDS = int(folds)
    _WORKER_SAVE_FOLD_AUC = bool(save_fold_auc)


def _run_repeat(spec: SplitSpec) -> Dict[str, object]:
    dataset = _WORKER_DATASET
    pairs = _WORKER_PAIRS
    folds = _WORKER_FOLDS
    if dataset is None or pairs is None:
        raise RuntimeError("Worker context was not initialized")
    n_pairs = len(pairs)
    n_metrics = len(ALL_METRICS)
    thresholds = np.asarray([pair.threshold_pct for pair in pairs], dtype=np.float64)
    deltas = np.asarray([pair.delta_pct for pair in pairs], dtype=np.float64)
    scores = np.full((dataset.n_samples, n_pairs, n_metrics), np.nan, dtype=np.float64)
    fold_aucs = np.full((folds, n_pairs, n_metrics), np.nan, dtype=np.float64)
    fold_n_valid = np.zeros((folds, n_pairs, n_metrics), dtype=np.int32)
    fold_rows: List[dict] = []
    packed_ra: List[np.ndarray] = []
    packed_ild: List[np.ndarray] = []

    for fold in range(1, folds + 1):
        validation_indices = np.flatnonzero(spec.folds == fold)
        training_indices = np.flatnonzero(spec.folds != fold)
        train_labels = dataset.labels[training_indices]
        ra_indices = training_indices[train_labels == "RA"]
        ild_indices = training_indices[train_labels == "ILD"]
        n_ra = int(len(ra_indices))
        n_ild = int(len(ild_indices))
        ra_counts = aligned_presence_counts(dataset, ra_indices)
        ild_counts = aligned_presence_counts(dataset, ild_indices)
        ra_pct = ra_counts.astype(np.float64) / float(n_ra) * 100.0
        ild_pct = ild_counts.astype(np.float64) / float(n_ild) * 100.0
        delta = ra_pct - ild_pct
        ra_members = (ra_pct[:, None] >= thresholds[None, :]) & (delta[:, None] >= deltas[None, :])
        ild_members = (ild_pct[:, None] >= thresholds[None, :]) & ((-delta)[:, None] >= deltas[None, :])
        if np.any(ra_members & ild_members):
            raise RuntimeError("RA and RA-ILD dictionary overlap in grid computation")
        ra_sizes = ra_members.sum(axis=0, dtype=np.int64)
        ild_sizes = ild_members.sum(axis=0, dtype=np.int64)
        packed_ra.append(np.packbits(ra_members, axis=0))
        packed_ild.append(np.packbits(ild_members, axis=0))

        ra_threshold_counts = np.ceil(n_ra * thresholds / 100.0).astype(np.int32)
        ild_threshold_counts = np.ceil(n_ild * thresholds / 100.0).astype(np.int32)
        validation_labels = dataset.labels[validation_indices]
        for pair_index, pair in enumerate(pairs):
            fold_rows.append(
                {
                    "scope": dataset.scope,
                    "repeat": spec.repeat,
                    "fold": fold,
                    "threshold_id": pair.threshold_id,
                    "threshold_pct": pair.threshold_pct,
                    "delta_pct": pair.delta_pct,
                    "split_quality_score": spec.split_quality_score,
                    "chosen_seed": spec.chosen_seed,
                    "partition_signature": spec.partition_signature,
                    "n_train": int(len(training_indices)),
                    "n_validation": int(len(validation_indices)),
                    "n_train_RA": n_ra,
                    "n_train_ILD": n_ild,
                    "n_validation_RA": int(np.sum(validation_labels == "RA")),
                    "n_validation_ILD": int(np.sum(validation_labels == "ILD")),
                    "RA_threshold_count": int(ra_threshold_counts[pair_index]),
                    "ILD_threshold_count": int(ild_threshold_counts[pair_index]),
                    "RA_dict_size": int(ra_sizes[pair_index]),
                    "ILD_dict_size": int(ild_sizes[pair_index]),
                }
            )

        for sample_index in validation_indices.tolist():
            ids = dataset.clone_ids[sample_index]
            rf = dataset.read_fractions[sample_index]
            if ids.size:
                sample_ra = ra_members[ids, :]
                sample_ild = ild_members[ids, :]
                ra_count = sample_ra.sum(axis=0, dtype=np.int64)
                ild_count = sample_ild.sum(axis=0, dtype=np.int64)
                ra_rf = np.einsum("i,ij->j", rf, sample_ra, optimize=True)
                ild_rf = np.einsum("i,ij->j", rf, sample_ild, optimize=True)
            else:
                ra_count = np.zeros(n_pairs, dtype=np.int64)
                ild_count = np.zeros(n_pairs, dtype=np.int64)
                ra_rf = np.zeros(n_pairs, dtype=np.float64)
                ild_rf = np.zeros(n_pairs, dtype=np.float64)
            scores[sample_index, :, :] = _metric_matrix(
                ra_count,
                ild_count,
                ra_rf,
                ild_rf,
                ra_sizes,
                ild_sizes,
                int(dataset.total_unique_clones[sample_index]),
            )

        if _WORKER_SAVE_FOLD_AUC:
            matrix = scores[validation_indices].reshape(len(validation_indices), n_pairs * n_metrics)
            auc_values, n_valid = auc_matrix(validation_labels, matrix)
            fold_aucs[fold - 1] = auc_values.reshape(n_pairs, n_metrics)
            fold_n_valid[fold - 1] = n_valid.reshape(n_pairs, n_metrics)

        # Release large bool matrices before the next fold; packed copies remain.
        del ra_members, ild_members

    repeat_matrix = scores.reshape(dataset.n_samples, n_pairs * n_metrics)
    repeat_auc, repeat_n_valid = auc_matrix(dataset.labels, repeat_matrix)
    repeat_auc = repeat_auc.reshape(n_pairs, n_metrics)
    repeat_n_valid = repeat_n_valid.reshape(n_pairs, n_metrics)

    stability = np.empty((n_pairs, 2, int(folds * (folds - 1) / 2)), dtype=np.float64)
    pair_position = 0
    for fold_a, fold_b in itertools.combinations(range(folds), 2):
        stability[:, 0, pair_position] = _packed_jaccard(packed_ra[fold_a], packed_ra[fold_b])
        stability[:, 1, pair_position] = _packed_jaccard(packed_ild[fold_a], packed_ild[fold_b])
        pair_position += 1

    return {
        "repeat": spec.repeat,
        "scores": scores,
        "repeat_auc": repeat_auc,
        "repeat_n_valid": repeat_n_valid,
        "fold_auc": fold_aucs,
        "fold_n_valid": fold_n_valid,
        "fold_rows": fold_rows,
        "stability": stability,
    }


def _direction_arrays(raw_auc: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(raw_auc, dtype=np.float64)
    directional = np.where(raw >= 0.5, raw, 1.0 - raw)
    direction = np.where(raw >= 0.5, "RA_higher", "RA-ILD_higher").astype(object)
    direction[~np.isfinite(raw)] = "not_available"
    ild_auc = 1.0 - raw
    directional[~np.isfinite(raw)] = np.nan
    ild_auc[~np.isfinite(raw)] = np.nan
    return directional, direction, ild_auc


def _auc_rows_from_arrays(
    pairs: Sequence[ThresholdPair],
    repeat: int,
    aucs: np.ndarray,
    n_valid: np.ndarray,
    fold: Optional[int] = None,
) -> List[dict]:
    directional, direction, ild_auc = _direction_arrays(aucs)
    rows: List[dict] = []
    for pair_index, pair in enumerate(pairs):
        for metric_index, metric in enumerate(ALL_METRICS):
            row = {
                "repeat": int(repeat),
                "threshold_id": pair.threshold_id,
                "threshold_pct": pair.threshold_pct,
                "delta_pct": pair.delta_pct,
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "auc_RA_positive_raw": float(aucs[pair_index, metric_index]),
                "auc_RAILD_positive_raw": float(ild_auc[pair_index, metric_index]),
                "directional_auc": float(directional[pair_index, metric_index]),
                "higher_score_cohort": str(direction[pair_index, metric_index]),
                "n_valid": int(n_valid[pair_index, metric_index]),
            }
            if fold is None:
                row.update(
                    {
                        "aggregation": "pooled_OOF_across_all_5_folds",
                        "is_primary_repeat_result": True,
                    }
                )
            else:
                row.update(
                    {
                        "fold": int(fold),
                        "role": "diagnostic_fold_auc_not_independent",
                    }
                )
            rows.append(row)
    return rows


def _sample_average_frame(
    dataset: ScopeDataset,
    pairs: Sequence[ThresholdPair],
    score_sum: np.ndarray,
    score_count: np.ndarray,
) -> pd.DataFrame:
    means = np.divide(
        score_sum,
        score_count,
        out=np.full(score_sum.shape, np.nan, dtype=np.float64),
        where=score_count > 0,
    )
    frames: List[pd.DataFrame] = []
    for pair_index, pair in enumerate(pairs):
        frame = pd.DataFrame(means[:, pair_index, :], columns=list(ALL_METRICS))
        frame.insert(0, "delta_pct", pair.delta_pct)
        frame.insert(0, "threshold_pct", pair.threshold_pct)
        frame.insert(0, "threshold_id", pair.threshold_id)
        frame.insert(0, "cohort", dataset.labels)
        frame.insert(0, "sample_id", dataset.sample_ids)
        frame["sample_unique_clone_count"] = dataset.total_unique_clones.astype(float)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    valid = np.isfinite(x) & np.isfinite(y)
    a = x[valid]
    b = y[valid]
    n = int(len(a))
    if n < 3 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan"), n
    ra = rankdata_average(a)
    rb = rankdata_average(b)
    return float(np.corrcoef(ra, rb)[0, 1]), n


def summarize_grid(
    dataset: ScopeDataset,
    pairs: Sequence[ThresholdPair],
    auc_by_repeat: pd.DataFrame,
    auc_by_fold: pd.DataFrame,
    sample_averaged: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: List[dict] = []
    richness_rows: List[dict] = []
    for pair in pairs:
        pair_repeat = auc_by_repeat.loc[auc_by_repeat["threshold_id"] == pair.threshold_id]
        if auc_by_fold.empty:
            pair_fold = pd.DataFrame(columns=["metric", "auc_RA_positive_raw"])
        else:
            pair_fold = auc_by_fold.loc[auc_by_fold["threshold_id"] == pair.threshold_id]
        pair_scores = sample_averaged.loc[sample_averaged["threshold_id"] == pair.threshold_id]
        for metric in ALL_METRICS:
            repeat_sub = pair_repeat.loc[pair_repeat["metric"] == metric]
            fold_sub = pair_fold.loc[pair_fold["metric"] == metric]
            raw = repeat_sub["auc_RA_positive_raw"].to_numpy(dtype=float)
            directional = repeat_sub["directional_auc"].to_numpy(dtype=float)
            valid_raw = raw[np.isfinite(raw)]
            valid_directional = directional[np.isfinite(directional)]
            fold_raw = fold_sub["auc_RA_positive_raw"].to_numpy(dtype=float)
            valid_fold = fold_raw[np.isfinite(fold_raw)]
            sample_auc_array, sample_n = auc_matrix(
                pair_scores["cohort"].to_numpy(dtype=str),
                pair_scores[[metric]].to_numpy(dtype=float),
            )
            sample_auc = float(sample_auc_array[0])
            sample_directional, sample_direction, sample_ild_auc = direction_fields(sample_auc)
            if valid_raw.size:
                q025, q975 = np.quantile(valid_raw, [0.025, 0.975])
                fraction_ra = float(np.mean(valid_raw >= 0.5))
            else:
                q025 = q975 = fraction_ra = float("nan")
            if valid_fold.size:
                fold_q025, fold_q975 = np.quantile(valid_fold, [0.025, 0.975])
            else:
                fold_q025 = fold_q975 = float("nan")
            rows.append(
                {
                    "threshold_id": pair.threshold_id,
                    "threshold_pct": pair.threshold_pct,
                    "delta_pct": pair.delta_pct,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "primary_resampling_unit": "complete_5fold_repeat",
                    "repeats_valid": int(valid_raw.size),
                    "repeat_auc_mean": float(np.mean(valid_raw)) if valid_raw.size else float("nan"),
                    "repeat_auc_sd": float(np.std(valid_raw, ddof=1)) if valid_raw.size > 1 else float("nan"),
                    "repeat_auc_median": float(np.median(valid_raw)) if valid_raw.size else float("nan"),
                    "repeat_auc_q025": float(q025),
                    "repeat_auc_q975": float(q975),
                    "repeat_directional_auc_median": float(np.median(valid_directional)) if valid_directional.size else float("nan"),
                    "fraction_repeats_RA_higher": fraction_ra,
                    "fold_auc_diagnostic_n": int(valid_fold.size),
                    "fold_auc_diagnostic_median": float(np.median(valid_fold)) if valid_fold.size else float("nan"),
                    "fold_auc_diagnostic_q025": float(fold_q025),
                    "fold_auc_diagnostic_q975": float(fold_q975),
                    "sample_averaged_auc_RA_positive_raw": sample_auc,
                    "sample_averaged_auc_RAILD_positive_raw": sample_ild_auc,
                    "sample_averaged_directional_auc": sample_directional,
                    "sample_averaged_higher_score_cohort": sample_direction,
                    "sample_averaged_n_valid": int(sample_n[0]),
                    "threshold_screening_role": "exploratory_threshold_grid_not_unbiased_final_performance",
                }
            )
            rho, n_corr = _spearman(
                pair_scores["sample_unique_clone_count"].to_numpy(dtype=float),
                pair_scores[metric].to_numpy(dtype=float),
            )
            richness_rows.append(
                {
                    "threshold_id": pair.threshold_id,
                    "threshold_pct": pair.threshold_pct,
                    "delta_pct": pair.delta_pct,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "spearman_rho_with_sample_unique_clones": rho,
                    "n_valid": n_corr,
                }
            )
    summary = pd.DataFrame(rows)
    summary["rank_directional_within_metric"] = summary.groupby("metric")["repeat_directional_auc_median"].rank(method="min", ascending=False)
    summary["rank_raw_RA_positive_within_metric"] = summary.groupby("metric")["repeat_auc_median"].rank(method="min", ascending=False)
    best = (
        summary.sort_values(
            ["metric", "repeat_directional_auc_median", "fraction_repeats_RA_higher", "threshold_pct", "delta_pct"],
            ascending=[True, False, False, True, True],
        )
        .groupby("metric", as_index=False)
        .head(1)
        .copy()
    )
    best.insert(0, "selection_rule", "max_median_directional_auc_then_direction_consistency")
    best["selection_warning"] = "Selected on the same repeated-CV grid; requires nested validation for unbiased final performance."
    return summary, best, pd.DataFrame(richness_rows)


def summarize_stability(
    stability: pd.DataFrame,
) -> pd.DataFrame:
    if stability.empty:
        return pd.DataFrame()
    rows: List[dict] = []
    for keys, frame in stability.groupby(
        ["threshold_id", "threshold_pct", "delta_pct", "dictionary"], sort=True
    ):
        threshold_id, threshold_pct, delta_pct, dictionary = keys
        values = frame["jaccard"].to_numpy(dtype=float)
        rows.append(
            {
                "threshold_id": threshold_id,
                "threshold_pct": threshold_pct,
                "delta_pct": delta_pct,
                "dictionary": dictionary,
                "n_pairs": int(len(values)),
                "jaccard_mean": float(np.mean(values)),
                "jaccard_median": float(np.median(values)),
                "jaccard_q025": float(np.quantile(values, 0.025)),
                "jaccard_q975": float(np.quantile(values, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def _repeat_result_iterator(
    specs: Sequence[SplitSpec],
    workers: int,
) -> Iterable[Dict[str, object]]:
    if workers == 1:
        for spec in specs:
            yield _run_repeat(spec)
        return
    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError("workers>1 requires Linux multiprocessing start method 'fork'")
    context = mp.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {executor.submit(_run_repeat, spec): spec.repeat for spec in specs}
        for future in concurrent.futures.as_completed(futures):
            yield future.result()


def run_scope_grid(
    dataset: ScopeDataset,
    pairs: Sequence[ThresholdPair],
    folds: int = 5,
    repeats: int = 100,
    candidates: int = 300,
    seed: int = 20260711,
    workers: int = 1,
    save_fold_auc: bool = True,
    progress_every: int = 5,
) -> GridScopeResult:
    started = time.time()
    workers = validate_workers(workers)
    if folds < 2 or repeats < 1 or candidates < 1:
        raise ValueError("folds>=2, repeats>=1 and candidates>=1 are required")
    if len(pairs) < 1:
        raise ValueError("At least one threshold pair is required")
    specs = prepare_splits(dataset, folds, repeats, candidates, seed)
    _set_worker_context(dataset, pairs, folds, save_fold_auc)

    assignment_rows: List[dict] = []
    balance_rows: List[dict] = []
    for spec in specs:
        balance_rows.extend(fold_balance_rows(dataset.metadata, spec.folds, spec.repeat, dataset.scope))
        for sample_index, sample_id in enumerate(dataset.sample_ids):
            assignment_rows.append(
                {
                    "scope": dataset.scope,
                    "repeat": spec.repeat,
                    "fold": int(spec.folds[sample_index]),
                    "sample_id": sample_id,
                    "cohort": dataset.labels[sample_index],
                    "split_quality_score": spec.split_quality_score,
                    "chosen_seed": spec.chosen_seed,
                    "partition_signature": spec.partition_signature,
                }
            )

    n_pairs = len(pairs)
    n_metrics = len(ALL_METRICS)
    score_sum = np.zeros((dataset.n_samples, n_pairs, n_metrics), dtype=np.float64)
    score_count = np.zeros((dataset.n_samples, n_pairs, n_metrics), dtype=np.int32)
    repeat_rows: List[dict] = []
    fold_auc_rows: List[dict] = []
    fold_dictionary_rows: List[dict] = []
    stability_rows: List[dict] = []
    completed = 0
    pending_results = _repeat_result_iterator(specs, workers)
    for result in pending_results:
        repeat = int(result["repeat"])
        scores = np.asarray(result["scores"], dtype=np.float64)
        valid = np.isfinite(scores)
        score_sum[valid] += scores[valid]
        score_count[valid] += 1
        repeat_rows.extend(
            _auc_rows_from_arrays(
                pairs,
                repeat,
                np.asarray(result["repeat_auc"], dtype=float),
                np.asarray(result["repeat_n_valid"], dtype=int),
            )
        )
        if save_fold_auc:
            fold_auc = np.asarray(result["fold_auc"], dtype=float)
            fold_n = np.asarray(result["fold_n_valid"], dtype=int)
            for fold in range(1, folds + 1):
                fold_auc_rows.extend(
                    _auc_rows_from_arrays(
                        pairs,
                        repeat,
                        fold_auc[fold - 1],
                        fold_n[fold - 1],
                        fold=fold,
                    )
                )
        fold_dictionary_rows.extend(result["fold_rows"])
        stability_values = np.asarray(result["stability"], dtype=float)
        pair_number = int(folds * (folds - 1) / 2)
        for threshold_index, pair in enumerate(pairs):
            for dictionary_index, dictionary_name in enumerate(("RA", "RA-ILD")):
                for pair_index in range(pair_number):
                    stability_rows.append(
                        {
                            "scope": dataset.scope,
                            "repeat": repeat,
                            "threshold_id": pair.threshold_id,
                            "threshold_pct": pair.threshold_pct,
                            "delta_pct": pair.delta_pct,
                            "dictionary": dictionary_name,
                            "fold_pair_index": pair_index + 1,
                            "jaccard": float(stability_values[threshold_index, dictionary_index, pair_index]),
                        }
                    )
        completed += 1
        if completed == 1 or completed % max(1, progress_every) == 0 or completed == repeats:
            print(
                "[{}] completed repeats {}/{} | threshold pairs={} | workers={}".format(
                    dataset.scope, completed, repeats, len(pairs), workers
                ),
                flush=True,
            )

    auc_by_repeat = pd.DataFrame(repeat_rows).sort_values(
        ["repeat", "threshold_pct", "delta_pct", "metric"]
    ).reset_index(drop=True)
    if fold_auc_rows:
        auc_by_fold = pd.DataFrame(fold_auc_rows).sort_values(
            ["repeat", "fold", "threshold_pct", "delta_pct", "metric"]
        ).reset_index(drop=True)
    else:
        auc_by_fold = pd.DataFrame(columns=[
            "repeat", "threshold_id", "threshold_pct", "delta_pct",
            "metric", "metric_label", "auc_RA_positive_raw",
            "auc_RAILD_positive_raw", "directional_auc",
            "higher_score_cohort", "n_valid", "fold", "role",
        ])
    sample_averaged = _sample_average_frame(dataset, pairs, score_sum, score_count)
    threshold_summary, best_thresholds, richness = summarize_grid(
        dataset,
        pairs,
        auc_by_repeat,
        auc_by_fold,
        sample_averaged,
    )
    stability = pd.DataFrame(stability_rows)
    stability_summary = summarize_stability(stability)
    assignments = pd.DataFrame(assignment_rows)
    fold_dictionary_summary = pd.DataFrame(fold_dictionary_rows)
    return GridScopeResult(
        scope=dataset.scope,
        threshold_grid=threshold_grid_frame(pairs),
        assignments=assignments,
        sample_averaged_scores=sample_averaged,
        auc_by_repeat=auc_by_repeat,
        auc_by_fold=auc_by_fold,
        threshold_summary=threshold_summary,
        best_thresholds=best_thresholds,
        fold_dictionary_summary=fold_dictionary_summary,
        stability=stability,
        stability_summary=stability_summary,
        richness_correlation=richness,
        runtime_seconds=time.time() - started,
    )


def _heatmap_matrix(frame: pd.DataFrame, value_column: str) -> Tuple[np.ndarray, List[float], List[float]]:
    thresholds = sorted(frame["threshold_pct"].unique())
    deltas = sorted(frame["delta_pct"].unique())
    matrix = np.full((len(deltas), len(thresholds)), np.nan, dtype=float)
    t_map = {value: index for index, value in enumerate(thresholds)}
    d_map = {value: index for index, value in enumerate(deltas)}
    for row in frame.itertuples(index=False):
        matrix[d_map[row.delta_pct], t_map[row.threshold_pct]] = getattr(row, value_column)
    return matrix, thresholds, deltas


def plot_scope_grid(result: GridScopeResult, output_dir: Path) -> List[Path]:
    plt = _import_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []

    for metric in PRIMARY_METRICS:
        frame = result.threshold_summary.loc[result.threshold_summary["metric"] == metric]
        matrix, thresholds, deltas = _heatmap_matrix(frame, "repeat_auc_median")
        fig, axis = plt.subplots(figsize=(10, 6))
        image = axis.imshow(matrix, origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
        axis.set_xticks(range(len(thresholds)), ["{:g}".format(x) for x in thresholds])
        axis.set_yticks(range(len(deltas)), ["{:g}".format(x) for x in deltas])
        axis.set_xlabel("T prevalence threshold (%)")
        axis.set_ylabel("Delta threshold (%)")
        axis.set_title("{}: median repeated 5-fold pooled-OOF AUC (RA+)".format(METRIC_LABELS[metric]))
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                if np.isfinite(matrix[y, x]):
                    axis.text(x, y, "{:.2f}".format(matrix[y, x]), ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=axis, label="Median AUC")
        fig.tight_layout()
        path = output_dir / "grid_auc_heatmap_{}.png".format(metric)
        fig.savefig(path, dpi=180)
        plt.close(fig)
        created.append(path)

    best_primary = result.best_thresholds.loc[result.best_thresholds["metric"].isin(PRIMARY_METRICS)].copy()
    best_primary["metric_order"] = best_primary["metric"].map({m: i for i, m in enumerate(PRIMARY_METRICS)})
    best_primary = best_primary.sort_values("metric_order")

    fig, axis = plt.subplots(figsize=(12, 6))
    positions = np.arange(len(best_primary))
    axis.bar(positions, best_primary["repeat_directional_auc_median"].to_numpy(dtype=float))
    axis.set_xticks(positions, [METRIC_LABELS[m] for m in best_primary["metric"]], rotation=35, ha="right")
    axis.set_ylim(0.45, 1.0)
    axis.set_ylabel("Median directional AUC")
    axis.set_title("Best threshold pair per primary metric (exploratory grid)")
    for x, row in enumerate(best_primary.itertuples(index=False)):
        axis.text(x, row.repeat_directional_auc_median + 0.01, "T={:g},D={:g}".format(row.threshold_pct, row.delta_pct), ha="center", fontsize=8)
    fig.tight_layout()
    path = output_dir / "grid_best_threshold_directional_auc.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    created.append(path)

    fig, axis = plt.subplots(figsize=(8, 8))
    for row in best_primary.itertuples(index=False):
        frame = result.sample_averaged_scores.loc[result.sample_averaged_scores["threshold_id"] == row.threshold_id]
        y = (frame["cohort"].to_numpy(dtype=str) == "RA").astype(int)
        values = frame[row.metric].to_numpy(dtype=float)
        valid = np.isfinite(values)
        if valid.sum() < 2 or np.unique(y[valid]).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y[valid], values[valid])
        axis.plot(fpr, tpr, label="{} | T={:g},D={:g} | AUC={:.3f}".format(
            METRIC_LABELS[row.metric], row.threshold_pct, row.delta_pct, row.sample_averaged_auc_RA_positive_raw
        ))
    axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    axis.set_xlabel("1 - Specificity")
    axis.set_ylabel("Sensitivity")
    axis.set_title("Best-threshold sample-averaged OOF ROC (RA positive)")
    axis.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    path = output_dir / "grid_best_threshold_roc_primary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    created.append(path)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    colors = {"RA": "#0072B2", "ILD": "#D55E00"}
    for axis, metric in zip(axes.flat, PRIMARY_METRICS):
        row = best_primary.loc[best_primary["metric"] == metric].iloc[0]
        frame = result.sample_averaged_scores.loc[result.sample_averaged_scores["threshold_id"] == row["threshold_id"]]
        ild = frame.loc[frame["cohort"] == "ILD", metric].dropna().to_numpy()
        ra = frame.loc[frame["cohort"] == "RA", metric].dropna().to_numpy()
        artists = _axis_boxplot(axis, [ild, ra], tick_labels=["RA-ILD", "RA"], patch_artist=True, showfliers=False)
        artists["boxes"][0].set_facecolor(colors["ILD"])
        artists["boxes"][1].set_facecolor(colors["RA"])
        axis.set_title("{}\nT={:g}, D={:g}, AUC={:.3f}".format(
            METRIC_LABELS[metric], row["threshold_pct"], row["delta_pct"], row["sample_averaged_auc_RA_positive_raw"]
        ))
    fig.suptitle("Best-threshold sample-averaged OOF scores: {}".format(result.scope), fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = output_dir / "grid_best_threshold_boxplot_primary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    created.append(path)
    return created


def write_scope_outputs(
    result: GridScopeResult,
    result_dir: Path,
    plot_dir: Path,
    make_plots: bool = True,
) -> Dict[str, str]:
    scope_dir = result_dir / result.scope
    scope_plot_dir = plot_dir / result.scope
    scope_dir.mkdir(parents=True, exist_ok=True)
    scope_plot_dir.mkdir(parents=True, exist_ok=True)
    files: Dict[str, Path] = {
        "assignments": scope_dir / "grid_fold_assignments_{}.csv.gz".format(result.scope),
        "sample_averaged_scores": scope_dir / "grid_sample_averaged_scores_{}.csv.gz".format(result.scope),
        "auc_by_repeat": scope_dir / "grid_auc_by_repeat_{}.csv.gz".format(result.scope),
        "auc_by_fold": scope_dir / "grid_auc_by_fold_diagnostic_{}.csv.gz".format(result.scope),
        "threshold_summary": scope_dir / "grid_metric_threshold_summary_{}.csv".format(result.scope),
        "best_thresholds": scope_dir / "grid_best_thresholds_by_metric_{}.csv".format(result.scope),
        "fold_dictionary_summary": scope_dir / "grid_fold_dictionary_summary_{}.csv.gz".format(result.scope),
        "stability": scope_dir / "grid_dictionary_stability_{}.csv.gz".format(result.scope),
        "stability_summary": scope_dir / "grid_dictionary_stability_summary_{}.csv".format(result.scope),
        "richness_correlation": scope_dir / "grid_richness_correlation_{}.csv".format(result.scope),
    }
    result.assignments.to_csv(files["assignments"], index=False, compression="gzip")
    result.sample_averaged_scores.to_csv(files["sample_averaged_scores"], index=False, compression="gzip")
    result.auc_by_repeat.to_csv(files["auc_by_repeat"], index=False, compression="gzip")
    result.auc_by_fold.to_csv(files["auc_by_fold"], index=False, compression="gzip")
    result.threshold_summary.to_csv(files["threshold_summary"], index=False)
    result.best_thresholds.to_csv(files["best_thresholds"], index=False)
    result.fold_dictionary_summary.to_csv(files["fold_dictionary_summary"], index=False, compression="gzip")
    result.stability.to_csv(files["stability"], index=False, compression="gzip")
    result.stability_summary.to_csv(files["stability_summary"], index=False)
    result.richness_correlation.to_csv(files["richness_correlation"], index=False)
    if make_plots:
        for index, path in enumerate(plot_scope_grid(result, scope_plot_dir), start=1):
            files["plot_{}".format(index)] = path
    return {name: str(path) for name, path in files.items()}


def write_grid_summary(
    path: Path,
    config: Mapping[str, object],
    datasets: Mapping[str, ScopeDataset],
    results: Mapping[str, GridScopeResult],
    runtime_seconds: float,
) -> None:
    lines: List[str] = [
        "# Enrich-dictionary threshold-grid repeated 5-fold summary",
        "",
        "## Design",
        "",
        "- Every valid threshold pair uses the same fold assignments within each scope and repeat.",
        "- Dictionaries are rebuilt from each fold's training samples only.",
        "- One complete five-fold repeat produces one pooled OOF AUC per threshold pair and metric.",
        "- Fold-level AUC values are diagnostic and are not treated as independent repeats.",
        "- Threshold pairs satisfy `delta_pct <= threshold_pct`.",
        "- Parallel workers operate on complete repeats. The supported range is 1–16; default is 1.",
        "- Best-threshold tables are exploratory because threshold selection and performance estimation use the same repeated-CV grid.",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Scope results",
        "",
    ]
    for scope, result in results.items():
        dataset = datasets[scope]
        counts = dataset.metadata["cohort"].value_counts()
        lines.extend(
            [
                "### {}".format(scope),
                "",
                "- Samples: **{}** (RA {}, RA-ILD {})".format(
                    dataset.n_samples, int(counts.get("RA", 0)), int(counts.get("ILD", 0))
                ),
                "- Valid threshold pairs: **{}**".format(len(result.threshold_grid)),
                "- Prescreened clone universe: **{} / {}** (minimum total presence {})".format(
                    dataset.n_universe, dataset.prescreen_original_universe, dataset.prescreen_min_presence
                ),
                "- Scope runtime: **{:.2f} seconds**".format(result.runtime_seconds),
                "",
                "| Metric | Selected T | Selected Delta | Median directional AUC | Raw RA-positive median AUC | Direction consistency |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        best = result.best_thresholds.set_index("metric")
        for metric in PRIMARY_METRICS:
            row = best.loc[metric]
            lines.append(
                "| {} | {:.1f} | {:.1f} | {:.3f} | {:.3f} | {:.1%} RA-higher |".format(
                    METRIC_LABELS[metric],
                    row["threshold_pct"],
                    row["delta_pct"],
                    row["repeat_directional_auc_median"],
                    row["repeat_auc_median"],
                    row["fraction_repeats_RA_higher"],
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation safeguards",
            "",
            "- The threshold grid is a screening analysis, not an unbiased final model assessment.",
            "- Choosing the best threshold from this grid creates selection optimism; use nested CV or an untouched validation set for a final performance claim.",
            "- A RA-ILD-named dictionary metric may still be higher in RA; inspect the direction fields.",
            "- Dictionary coverage corrects dictionary size but not sample repertoire richness.",
            "",
            "Total runtime: **{:.2f} seconds**".format(runtime_seconds),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_project_grid_analysis(
    project_root: Path,
    scopes: Sequence[str],
    folds: int,
    repeats: int,
    candidates: int,
    seed: int,
    threshold_values: Sequence[float],
    delta_values: Sequence[float],
    workers: int = 1,
    allow_count_mismatch: bool = False,
    save_fold_auc: bool = True,
    make_plots: bool = True,
    overwrite: bool = False,
    result_dir: Optional[Path] = None,
    plot_dir: Optional[Path] = None,
) -> Dict[str, object]:
    started = time.time()
    workers = validate_workers(workers)
    pairs = generate_threshold_pairs(threshold_values, delta_values)
    root = project_root.expanduser().resolve()
    metadata_path = root / "TRB/metadata.csv"
    aa_dir = root / "TRB/result/01_AA_clone_table"
    result_dir = (result_dir or (root / "TRB/result/enrich_dictionary_threshold_grid_cv_v1")).resolve()
    plot_dir = (plot_dir or (root / "TRB/gradient/enrich_dictionary_threshold_grid_cv_v1")).resolve()
    normalized_scopes = [str(scope).strip().lower() for scope in scopes]
    invalid = sorted(set(normalized_scopes).difference(EXPECTED_SCOPE_COUNTS))
    if invalid:
        raise ValueError("Unsupported scopes: {}".format(invalid))
    if result_dir.exists() and any(result_dir.iterdir()) and not overwrite:
        raise FileExistsError("Result directory is not empty; use --overwrite: {}".format(result_dir))
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = result_dir / "threshold_grid_run_manifest.json"
    summary_path = result_dir / "threshold_grid_summary.md"
    grid_path = result_dir / "threshold_grid.csv"
    if overwrite:
        for scope in normalized_scopes:
            for base in (result_dir, plot_dir):
                target = base / scope
                if target.exists():
                    shutil.rmtree(target)
        for path in (manifest_path, summary_path, grid_path):
            if path.exists():
                path.unlink()

    metadata = load_metadata(metadata_path)
    file_map = build_aa_file_map(aa_dir, metadata["libraryid"].tolist())
    repertoires = load_repertoires(metadata, file_map)
    minimum_threshold = min(pair.threshold_pct for pair in pairs)
    datasets: Dict[str, ScopeDataset] = {}
    results: Dict[str, GridScopeResult] = {}
    outputs: Dict[str, Dict[str, str]] = {}
    threshold_grid_frame(pairs).to_csv(grid_path, index=False)

    for scope in normalized_scopes:
        dataset = build_scope_dataset(
            metadata,
            repertoires,
            scope,
            folds,
            minimum_threshold,
            allow_count_mismatch=allow_count_mismatch,
        )
        datasets[scope] = dataset
        print(
            "Starting scope={} samples={} threshold_pairs={} repeats={} folds={} workers={}".format(
                scope, dataset.n_samples, len(pairs), repeats, folds, workers
            ),
            flush=True,
        )
        result = run_scope_grid(
            dataset,
            pairs,
            folds=folds,
            repeats=repeats,
            candidates=candidates,
            seed=seed,
            workers=workers,
            save_fold_auc=save_fold_auc,
        )
        results[scope] = result
        outputs[scope] = write_scope_outputs(result, result_dir, plot_dir, make_plots=make_plots)

    config: Dict[str, object] = {
        "script_version": SCRIPT_VERSION,
        "project_root": str(root),
        "metadata": str(metadata_path),
        "metadata_sha256": file_sha256(metadata_path),
        "aa_dir": str(aa_dir),
        "scopes": normalized_scopes,
        "folds": int(folds),
        "repeats": int(repeats),
        "candidates": int(candidates),
        "seed": int(seed),
        "workers": int(workers),
        "workers_default": 1,
        "workers_maximum": MAX_WORKERS,
        "threshold_values": sorted(set(float(x) for x in threshold_values)),
        "delta_values": sorted(set(float(x) for x in delta_values)),
        "valid_threshold_pair_count": len(pairs),
        "valid_pair_rule": "delta_pct <= threshold_pct",
        "allow_count_mismatch": bool(allow_count_mismatch),
        "save_fold_auc": bool(save_fold_auc),
        "make_plots": bool(make_plots),
        "result_dir": str(result_dir),
        "plot_dir": str(plot_dir),
        "positive_class": "RA",
        "score_direction": "raw",
        "primary_metrics": list(PRIMARY_METRICS),
        "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
        "threshold_selection_status": "exploratory_screening_requires_nested_validation_for_unbiased_final_claim",
    }
    runtime = time.time() - started
    manifest = {
        "configuration": config,
        "scope_counts": {
            scope: {
                "samples": dataset.n_samples,
                "RA": int(np.sum(dataset.labels == "RA")),
                "ILD": int(np.sum(dataset.labels == "ILD")),
                "prescreened_universe": dataset.n_universe,
                "original_universe": dataset.prescreen_original_universe,
            }
            for scope, dataset in datasets.items()
        },
        "outputs": outputs,
        "runtime_seconds": runtime,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_grid_summary(summary_path, config, datasets, results, runtime)
    return manifest
