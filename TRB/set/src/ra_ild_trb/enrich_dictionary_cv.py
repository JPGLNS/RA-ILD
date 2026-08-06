#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repeated stratified 5-fold validation for CDR3-AA enrich dictionaries.

The module is deliberately independent of the Elastic Net model.  For every
outer fold, RA- and RA-ILD-enriched clone dictionaries are rebuilt using only
that fold's training samples, and every validation sample in the fold is scored
with the same fold-specific dictionaries.

Input conventions match the current RA-ILD TRB project:
  * metadata: TRB/metadata.csv
  * repertoire tables: TRB/result/01_AA_clone_table/*_AA_clone_table.csv
  * cohort labels: RA and ILD (ILD means RA-ILD)
  * AA table columns: cdr3_aa and read_fraction
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

SCRIPT_VERSION = "1.1.1"
BALANCE_WEIGHTS: Mapping[str, float] = {"batch": 1.0, "material": 0.75, "sex": 0.5}

PRIMARY_METRICS: Tuple[str, ...] = (
    "RA_dict_clone_count",
    "ILD_dict_clone_count",
    "RA_dict_hit_rate",
    "ILD_dict_hit_rate",
    "RA_dict_read_fraction_sum",
    "ILD_dict_read_fraction_sum",
    "RA_minus_ILD_count",
    "RA_minus_ILD_read_fraction",
)

DIAGNOSTIC_METRICS: Tuple[str, ...] = (
    "RA_minus_ILD_hit_rate",
    "RA_hit_fraction_of_sample",
    "ILD_hit_fraction_of_sample",
    "RA_minus_ILD_sample_fraction",
)

ALL_METRICS: Tuple[str, ...] = PRIMARY_METRICS + DIAGNOSTIC_METRICS

METRIC_LABELS: Mapping[str, str] = {
    "RA_dict_clone_count": "RA dictionary clone count",
    "ILD_dict_clone_count": "RA-ILD dictionary clone count",
    "RA_dict_hit_rate": "RA dictionary coverage",
    "ILD_dict_hit_rate": "RA-ILD dictionary coverage",
    "RA_dict_read_fraction_sum": "RA dictionary cumulative abundance",
    "ILD_dict_read_fraction_sum": "RA-ILD dictionary cumulative abundance",
    "RA_minus_ILD_count": "RA minus RA-ILD clone-count contrast",
    "RA_minus_ILD_read_fraction": "RA minus RA-ILD abundance contrast",
    "RA_minus_ILD_hit_rate": "RA minus RA-ILD coverage contrast",
    "RA_hit_fraction_of_sample": "RA hits / sample unique clones",
    "ILD_hit_fraction_of_sample": "RA-ILD hits / sample unique clones",
    "RA_minus_ILD_sample_fraction": "RA minus RA-ILD sample-hit fraction",
}

EXPECTED_SCOPE_COUNTS: Mapping[str, Mapping[str, int]] = {
    "total": {"total": 174, "RA": 108, "ILD": 66},
    "pbmc": {"total": 111, "RA": 67, "ILD": 44},
    "buffycoat": {"total": 63, "RA": 41, "ILD": 22},
}


@dataclass(frozen=True)
class RawRepertoire:
    sample_id: str
    clones: Tuple[str, ...]
    read_fractions: np.ndarray
    total_unique_clones: int


@dataclass
class ScopeDataset:
    scope: str
    metadata: pd.DataFrame
    sample_ids: List[str]
    labels: np.ndarray
    clone_names: np.ndarray
    clone_ids: List[np.ndarray]
    read_fractions: List[np.ndarray]
    total_unique_clones: np.ndarray
    prescreen_min_presence: int
    prescreen_original_universe: int

    @property
    def n_samples(self) -> int:
        return len(self.sample_ids)

    @property
    def n_universe(self) -> int:
        return int(len(self.clone_names))


@dataclass(frozen=True)
class DictionaryResult:
    ra_ids: np.ndarray
    ild_ids: np.ndarray
    ra_threshold_count: int
    ild_threshold_count: int
    n_ra_train: int
    n_ild_train: int


@dataclass
class ScopeRunResult:
    scope: str
    assignments: pd.DataFrame
    predictions: pd.DataFrame
    sample_averaged: pd.DataFrame
    auc_by_repeat: pd.DataFrame
    auc_by_fold: pd.DataFrame
    metric_summary: pd.DataFrame
    fold_summary: pd.DataFrame
    fold_balance: pd.DataFrame
    stability: pd.DataFrame
    stability_summary: pd.DataFrame
    richness_correlation: pd.DataFrame
    dictionary_members: Optional[pd.DataFrame]


def derived_seed(base: int, *parts: object) -> int:
    text = ":".join(str(x) for x in (base, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") % (2**32 - 1)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_cohort(value: object) -> str:
    text = str(value).strip().upper()
    if text in {"RA", "ILD"}:
        return text
    if text in {"RA-ILD", "RA_ILD", "RAILD"}:
        return "ILD"
    raise ValueError("Unsupported cohort label: {!r}".format(value))


def normalize_material(value: object) -> str:
    text = str(value).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if text == "pbmc":
        return "pbmc"
    if text in {"buffycoat", "buffercoat", "buffcoat"}:
        return "buffycoat"
    return text


def parse_sample_id_from_filename(filename: str, suffix: str = "_AA_clone_table.csv") -> str:
    name = Path(filename).name
    if not name.endswith(suffix):
        raise ValueError("Unexpected AA clone-table filename: {}".format(name))
    sample_id = name[: -len(suffix)]
    if not sample_id:
        raise ValueError("Empty sample ID parsed from {}".format(name))
    return sample_id


def build_aa_file_map(
    aa_dir: Path,
    metadata_ids: Sequence[str],
    suffix: str = "_AA_clone_table.csv",
) -> Dict[str, Path]:
    if not aa_dir.is_dir():
        raise FileNotFoundError(str(aa_dir))
    metadata_set = set(str(x).strip() for x in metadata_ids)
    result: Dict[str, Path] = {}
    for path in sorted(aa_dir.glob("*{}".format(suffix))):
        sample_id = parse_sample_id_from_filename(path.name, suffix=suffix)
        if sample_id.startswith("01_") or sample_id not in metadata_set:
            continue
        if sample_id in result:
            raise ValueError("Duplicate AA clone table for sample {}".format(sample_id))
        result[sample_id] = path.resolve()
    missing = sorted(metadata_set.difference(result))
    if missing:
        raise FileNotFoundError(
            "Metadata samples missing AA clone tables ({}): {}".format(
                len(missing), ", ".join(missing[:20])
            )
        )
    if not result:
        raise ValueError("No metadata-matched AA clone tables found in {}".format(aa_dir))
    return result


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.is_file():
        raise FileNotFoundError(str(metadata_path))
    df = pd.read_csv(metadata_path, dtype=str)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed:")], errors="ignore")
    required = ["libraryid", "cohort", "material"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Metadata missing columns: {}".format(missing))
    for col in required:
        if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            raise ValueError("Metadata contains empty values in {}".format(col))
    df = df.copy()
    df["libraryid"] = df["libraryid"].astype(str).str.strip()
    if df["libraryid"].duplicated().any():
        dup = df.loc[df["libraryid"].duplicated(keep=False), "libraryid"].unique()[:20]
        raise ValueError("Duplicated metadata libraryid values: {}".format(dup.tolist()))
    df["cohort"] = [normalize_cohort(x) for x in df["cohort"]]
    df["material_normalized"] = [normalize_material(x) for x in df["material"]]
    if "patient" in df.columns:
        patient = df["patient"].astype(str).str.strip()
        if patient.eq("").any() or patient.isna().any():
            raise ValueError("Metadata contains empty patient values")
        if patient.duplicated().any():
            dup = patient[patient.duplicated(keep=False)].unique()[:20]
            raise ValueError(
                "Multiple samples per patient detected; grouped CV is required. Examples: {}".format(
                    dup.tolist()
                )
            )
    return df.sort_values("libraryid").reset_index(drop=True)


def read_repertoire(path: Path, sample_id: str) -> RawRepertoire:
    usecols = ["cdr3_aa", "read_fraction"]
    try:
        df = pd.read_csv(path, usecols=usecols)
    except ValueError as exc:
        raise ValueError("{} lacks required columns {}".format(path, usecols)) from exc
    df["cdr3_aa"] = df["cdr3_aa"].astype("string").str.strip()
    df["read_fraction"] = pd.to_numeric(df["read_fraction"], errors="coerce")
    mask = (
        df["cdr3_aa"].notna()
        & df["cdr3_aa"].ne("")
        & np.isfinite(df["read_fraction"].to_numpy(dtype=float))
        & df["read_fraction"].ge(0)
    )
    df = df.loc[mask, usecols]
    if df.empty:
        return RawRepertoire(sample_id, tuple(), np.asarray([], dtype=np.float64), 0)
    agg = df.groupby("cdr3_aa", sort=False, as_index=False)["read_fraction"].sum()
    clones = tuple(agg["cdr3_aa"].astype(str).tolist())
    rf = agg["read_fraction"].to_numpy(dtype=np.float64)
    return RawRepertoire(sample_id, clones, rf, len(clones))


def load_repertoires(metadata: pd.DataFrame, file_map: Mapping[str, Path]) -> Dict[str, RawRepertoire]:
    result: Dict[str, RawRepertoire] = {}
    for row in metadata.itertuples(index=False):
        sample_id = str(row.libraryid)
        path = file_map[sample_id]
        parsed = parse_sample_id_from_filename(path.name)
        if parsed != sample_id:
            raise ValueError("Sample/file mismatch: {} vs {}".format(sample_id, path.name))
        result[sample_id] = read_repertoire(path, sample_id)
    return result


def scope_metadata(metadata: pd.DataFrame, scope: str) -> pd.DataFrame:
    scope = scope.lower()
    if scope == "total":
        out = metadata.copy()
    elif scope in {"pbmc", "buffycoat"}:
        out = metadata.loc[metadata["material_normalized"] == scope].copy()
    else:
        raise ValueError("Unsupported scope: {}".format(scope))
    if out.empty:
        raise ValueError("Scope {} contains no samples".format(scope))
    if set(out["cohort"]) != {"RA", "ILD"}:
        raise ValueError("Scope {} must contain both RA and ILD".format(scope))
    return out.sort_values("libraryid").reset_index(drop=True)


def validate_expected_counts(meta: pd.DataFrame, scope: str, allow_mismatch: bool) -> None:
    if allow_mismatch:
        return
    expected = EXPECTED_SCOPE_COUNTS[scope]
    observed = meta["cohort"].value_counts().to_dict()
    if len(meta) != expected["total"] or observed.get("RA", 0) != expected["RA"] or observed.get("ILD", 0) != expected["ILD"]:
        raise ValueError(
            "Unexpected {} counts. Expected total/RA/ILD={}/{}/{}, observed={}/{}/{}. "
            "Use --allow-count-mismatch only after review.".format(
                scope,
                expected["total"], expected["RA"], expected["ILD"],
                len(meta), observed.get("RA", 0), observed.get("ILD", 0),
            )
        )


def training_group_minimum(group_size: int, folds: int) -> int:
    max_validation = int(math.ceil(group_size / float(folds)))
    return group_size - max_validation


def build_scope_dataset(
    metadata: pd.DataFrame,
    repertoires: Mapping[str, RawRepertoire],
    scope: str,
    folds: int,
    threshold_pct: float,
    allow_count_mismatch: bool = False,
) -> ScopeDataset:
    meta = scope_metadata(metadata, scope)
    validate_expected_counts(meta, scope, allow_count_mismatch)
    sample_ids = meta["libraryid"].astype(str).tolist()
    labels = meta["cohort"].to_numpy(dtype=str)
    n_ra = int(np.sum(labels == "RA"))
    n_ild = int(np.sum(labels == "ILD"))
    min_ra = training_group_minimum(n_ra, folds)
    min_ild = training_group_minimum(n_ild, folds)
    min_presence = min(
        int(math.ceil(min_ra * threshold_pct / 100.0)),
        int(math.ceil(min_ild * threshold_pct / 100.0)),
    )
    min_presence = max(1, min_presence)

    occurrence: Counter[str] = Counter()
    for sid in sample_ids:
        occurrence.update(repertoires[sid].clones)
    original_universe = len(occurrence)
    kept = sorted(clone for clone, count in occurrence.items() if count >= min_presence)
    clone_to_id = {clone: idx for idx, clone in enumerate(kept)}

    clone_ids: List[np.ndarray] = []
    rf_arrays: List[np.ndarray] = []
    total_unique: List[int] = []
    for sid in sample_ids:
        rep = repertoires[sid]
        ids: List[int] = []
        rfs: List[float] = []
        for clone, rf in zip(rep.clones, rep.read_fractions):
            clone_id = clone_to_id.get(clone)
            if clone_id is not None:
                ids.append(clone_id)
                rfs.append(float(rf))
        clone_ids.append(np.asarray(ids, dtype=np.int32))
        rf_arrays.append(np.asarray(rfs, dtype=np.float64))
        total_unique.append(rep.total_unique_clones)

    return ScopeDataset(
        scope=scope,
        metadata=meta,
        sample_ids=sample_ids,
        labels=labels,
        clone_names=np.asarray(kept, dtype=object),
        clone_ids=clone_ids,
        read_fractions=rf_arrays,
        total_unique_clones=np.asarray(total_unique, dtype=np.int32),
        prescreen_min_presence=min_presence,
        prescreen_original_universe=original_universe,
    )


def stratified_candidate(labels: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    assign = np.full(len(labels), -1, dtype=np.int16)
    current_sizes = np.zeros(k, dtype=int)
    label_order = sorted(np.unique(labels), key=lambda x: (-int(np.sum(labels == x)), str(x)))
    for label in label_order:
        idx = rng.permutation(np.flatnonzero(labels == label))
        base, remainder = divmod(len(idx), k)
        counts = np.full(k, base, dtype=int)
        if remainder:
            jitter = rng.random(k) * 1e-6
            chosen = np.argsort(current_sizes + jitter)[:remainder]
            counts[chosen] += 1
        fold_ids = np.concatenate(
            [np.full(counts[f], f + 1, dtype=np.int16) for f in range(k)]
        )
        assign[idx] = rng.permutation(fold_ids)
        current_sizes += counts
    if np.any(assign < 1):
        raise RuntimeError("Incomplete fold assignment")
    if current_sizes.max() - current_sizes.min() > 1:
        raise RuntimeError("Unable to balance fold sizes within one sample")
    return assign


def category_score(values: np.ndarray, folds: np.ndarray, k: int) -> float:
    score = 0.0
    for category in sorted(np.unique(values)):
        mask = values == category
        expected = float(mask.sum()) / float(k)
        observed = np.asarray([(mask & (folds == f)).sum() for f in range(1, k + 1)], dtype=float)
        score += float(np.sum(((observed - expected) / max(expected, 1.0)) ** 2))
    return score


def split_score(meta: pd.DataFrame, folds: np.ndarray, k: int, balance_cols: Sequence[str]) -> float:
    expected_size = len(meta) / float(k)
    sizes = np.asarray([(folds == f).sum() for f in range(1, k + 1)], dtype=float)
    score = 0.25 * float(np.sum(((sizes - expected_size) / max(expected_size, 1.0)) ** 2))
    labels = meta["cohort"].astype(str)
    for col in balance_cols:
        if col not in meta.columns or meta[col].nunique(dropna=False) <= 1:
            continue
        weight = BALANCE_WEIGHTS.get(col, 0.5)
        values = meta[col].astype(str).to_numpy()
        score += weight * category_score(values, folds, k)
        joint = (labels + "|" + meta[col].astype(str)).to_numpy()
        score += 1.5 * weight * category_score(joint, folds, k)
    return score


def fold_signature(sample_ids: Sequence[str], folds: np.ndarray) -> str:
    text = "\n".join("{}\t{}".format(sid, int(fold)) for sid, fold in sorted(zip(sample_ids, folds)))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def optimize_split(
    meta: pd.DataFrame,
    k: int,
    candidates: int,
    base_seed: int,
    balance_cols: Sequence[str],
    forbidden: Optional[Set[str]] = None,
) -> Tuple[np.ndarray, float, int, str]:
    labels = meta["cohort"].to_numpy(dtype=str)
    sample_ids = meta["libraryid"].astype(str).tolist()
    best: Optional[Tuple[float, int, np.ndarray, str]] = None
    fallback: Optional[Tuple[float, int, np.ndarray, str]] = None
    for candidate in range(1, candidates + 1):
        seed = derived_seed(base_seed, candidate)
        folds = stratified_candidate(labels, k, np.random.default_rng(seed))
        score = split_score(meta, folds, k, balance_cols)
        signature = fold_signature(sample_ids, folds)
        record = (score, seed, folds.copy(), signature)
        if fallback is None or score < fallback[0]:
            fallback = record
        if forbidden and signature in forbidden:
            continue
        if best is None or score < best[0]:
            best = record
    chosen = best or fallback
    if chosen is None:
        raise RuntimeError("No fold candidate generated")
    return chosen[2], float(chosen[0]), int(chosen[1]), chosen[3]


def validate_fold_assignment(meta: pd.DataFrame, folds: np.ndarray, k: int) -> None:
    if len(folds) != len(meta):
        raise RuntimeError("Fold assignment length mismatch")
    if set(int(x) for x in np.unique(folds)) != set(range(1, k + 1)):
        raise RuntimeError("Missing fold IDs")
    sizes = np.asarray([(folds == f).sum() for f in range(1, k + 1)])
    if int(sizes.max() - sizes.min()) > 1:
        raise RuntimeError("Fold sizes differ by more than one")
    labels = meta["cohort"].to_numpy(dtype=str)
    for label in ("RA", "ILD"):
        counts = np.asarray([np.sum((folds == f) & (labels == label)) for f in range(1, k + 1)])
        if int(counts.max() - counts.min()) > 1:
            raise RuntimeError("{} fold counts differ by more than one".format(label))
        if np.any(counts == 0):
            raise RuntimeError("A validation fold lacks {} samples".format(label))


def aligned_presence_counts(dataset: ScopeDataset, indices: np.ndarray) -> np.ndarray:
    counts = np.zeros(dataset.n_universe, dtype=np.int32)
    for index in indices.tolist():
        ids = dataset.clone_ids[index]
        if ids.size:
            counts[ids] += 1
    return counts


def build_dictionary(
    dataset: ScopeDataset,
    train_indices: np.ndarray,
    threshold_pct: float,
    delta_pct: float,
) -> DictionaryResult:
    train_labels = dataset.labels[train_indices]
    ra_indices = train_indices[train_labels == "RA"]
    ild_indices = train_indices[train_labels == "ILD"]
    n_ra = int(len(ra_indices))
    n_ild = int(len(ild_indices))
    if n_ra < 1 or n_ild < 1:
        raise ValueError("Training fold must contain both cohorts")
    ra_counts = aligned_presence_counts(dataset, ra_indices)
    ild_counts = aligned_presence_counts(dataset, ild_indices)
    ra_threshold = int(math.ceil(n_ra * threshold_pct / 100.0))
    ild_threshold = int(math.ceil(n_ild * threshold_pct / 100.0))
    candidate = (ra_counts >= ra_threshold) | (ild_counts >= ild_threshold)
    candidate_ids = np.flatnonzero(candidate)
    if candidate_ids.size == 0:
        return DictionaryResult(
            np.asarray([], dtype=np.int32), np.asarray([], dtype=np.int32),
            ra_threshold, ild_threshold, n_ra, n_ild,
        )
    ra_pct = ra_counts[candidate_ids] / float(n_ra) * 100.0
    ild_pct = ild_counts[candidate_ids] / float(n_ild) * 100.0
    delta = ra_pct - ild_pct
    ra_dict = candidate_ids[(ra_pct >= threshold_pct) & (delta >= delta_pct)].astype(np.int32)
    ild_dict = candidate_ids[(ild_pct >= threshold_pct) & (delta <= -delta_pct)].astype(np.int32)
    if np.intersect1d(ra_dict, ild_dict).size:
        raise RuntimeError("RA and RA-ILD dictionaries overlap")
    return DictionaryResult(ra_dict, ild_dict, ra_threshold, ild_threshold, n_ra, n_ild)


def safe_divide(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(numerator) / float(denominator)


def score_sample(
    dataset: ScopeDataset,
    sample_index: int,
    dictionary: DictionaryResult,
) -> Dict[str, float]:
    ids = dataset.clone_ids[sample_index]
    rf = dataset.read_fractions[sample_index]
    if ids.size != rf.size:
        raise RuntimeError("Clone/read-fraction alignment failure")
    ra_mask = np.zeros(dataset.n_universe, dtype=bool)
    ild_mask = np.zeros(dataset.n_universe, dtype=bool)
    ra_mask[dictionary.ra_ids] = True
    ild_mask[dictionary.ild_ids] = True
    sample_ra = ra_mask[ids] if ids.size else np.asarray([], dtype=bool)
    sample_ild = ild_mask[ids] if ids.size else np.asarray([], dtype=bool)
    ra_count = int(sample_ra.sum())
    ild_count = int(sample_ild.sum())
    ra_rf = float(rf[sample_ra].sum()) if rf.size else 0.0
    ild_rf = float(rf[sample_ild].sum()) if rf.size else 0.0
    sample_total = int(dataset.total_unique_clones[sample_index])
    ra_hit_rate = safe_divide(ra_count, len(dictionary.ra_ids))
    ild_hit_rate = safe_divide(ild_count, len(dictionary.ild_ids))
    ra_sample_fraction = safe_divide(ra_count, sample_total)
    ild_sample_fraction = safe_divide(ild_count, sample_total)
    return {
        "RA_dict_clone_count": float(ra_count),
        "ILD_dict_clone_count": float(ild_count),
        "RA_dict_hit_rate": ra_hit_rate,
        "ILD_dict_hit_rate": ild_hit_rate,
        "RA_dict_read_fraction_sum": ra_rf,
        "ILD_dict_read_fraction_sum": ild_rf,
        "RA_minus_ILD_count": float(ra_count - ild_count),
        "RA_minus_ILD_read_fraction": ra_rf - ild_rf,
        "RA_minus_ILD_hit_rate": ra_hit_rate - ild_hit_rate if np.isfinite(ra_hit_rate) and np.isfinite(ild_hit_rate) else float("nan"),
        "RA_hit_fraction_of_sample": ra_sample_fraction,
        "ILD_hit_fraction_of_sample": ild_sample_fraction,
        "RA_minus_ILD_sample_fraction": ra_sample_fraction - ild_sample_fraction if np.isfinite(ra_sample_fraction) and np.isfinite(ild_sample_fraction) else float("nan"),
        "sample_unique_clone_count": float(sample_total),
        "RA_dict_size": float(len(dictionary.ra_ids)),
        "ILD_dict_size": float(len(dictionary.ild_ids)),
    }


def auc_from_scores(labels: Sequence[str], scores: Sequence[float]) -> Tuple[float, int]:
    y = np.asarray([1 if str(x) == "RA" else 0 for x in labels], dtype=int)
    values = np.asarray(scores, dtype=float)
    valid = np.isfinite(values)
    y = y[valid]
    values = values[valid]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan"), int(len(y))
    if len(np.unique(values)) < 2:
        return 0.5, int(len(y))
    return float(roc_auc_score(y, values)), int(len(y))


def direction_fields(raw_auc: float) -> Tuple[float, str, float]:
    if not np.isfinite(raw_auc):
        return float("nan"), "not_available", float("nan")
    if raw_auc >= 0.5:
        return raw_auc, "RA_higher", 1.0 - raw_auc
    return 1.0 - raw_auc, "RA-ILD_higher", 1.0 - raw_auc


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    set_a = set(int(x) for x in a.tolist())
    set_b = set(int(x) for x in b.tolist())
    if not set_a and not set_b:
        return 1.0
    return len(set_a.intersection(set_b)) / float(len(set_a.union(set_b)))


def rankdata_average(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    return series.rank(method="average").to_numpy(dtype=float)


def spearman_pair(x: Sequence[float], y: Sequence[float]) -> Tuple[float, int]:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    n = len(a)
    if n < 3 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return float("nan"), n
    ra = rankdata_average(a)
    rb = rankdata_average(b)
    corr = np.corrcoef(ra, rb)[0, 1]
    return float(corr), n


def fold_balance_rows(meta: pd.DataFrame, folds: np.ndarray, repeat: int, scope: str) -> List[dict]:
    rows: List[dict] = []
    variables = ["cohort"] + [c for c in ("material", "batch", "sex") if c in meta.columns]
    for fold in sorted(np.unique(folds)):
        subset = meta.loc[folds == fold]
        for variable in variables:
            values = subset[variable].astype(str).value_counts(dropna=False)
            for category, count in values.items():
                rows.append({
                    "scope": scope,
                    "repeat": repeat,
                    "fold": int(fold),
                    "fold_size": int(len(subset)),
                    "variable": variable,
                    "category": str(category),
                    "count": int(count),
                    "proportion": float(count) / float(len(subset)),
                })
    return rows


def summarize_predictions(
    predictions: pd.DataFrame,
    metrics: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize repeated 5-fold OOF predictions.

    The primary resampling unit is one complete 5-fold repeat.  Within each
    repeat, the five validation folds are concatenated so every sample has one
    out-of-fold score, and one pooled OOF AUC is calculated from all samples.
    Fold-level AUC values are retained only as diagnostics; the five folds are
    not independent replicates and their AUCs are not averaged for the primary
    performance estimate.
    """
    metadata_cols = [c for c in ("sample_id", "cohort", "material", "batch", "sex") if c in predictions.columns]
    numeric_cols = list(metrics) + ["sample_unique_clone_count", "RA_dict_size", "ILD_dict_size"]
    sample_avg = predictions.groupby(metadata_cols, as_index=False)[numeric_cols].mean()

    fold_auc_rows: List[dict] = []
    for (repeat, fold), frame in predictions.groupby(["repeat", "fold"], sort=True):
        n_ra = int(np.sum(frame["cohort"].to_numpy(dtype=str) == "RA"))
        n_ild = int(np.sum(frame["cohort"].to_numpy(dtype=str) == "ILD"))
        for metric in metrics:
            raw_auc, n_valid = auc_from_scores(frame["cohort"], frame[metric])
            directional_auc, direction, ild_auc = direction_fields(raw_auc)
            fold_auc_rows.append({
                "repeat": int(repeat),
                "fold": int(fold),
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "auc_RA_positive_raw": raw_auc,
                "auc_RAILD_positive_raw": ild_auc,
                "directional_auc": directional_auc,
                "higher_score_cohort": direction,
                "n_validation": int(len(frame)),
                "n_validation_RA": n_ra,
                "n_validation_ILD": n_ild,
                "n_valid": n_valid,
                "role": "diagnostic_fold_auc_not_independent",
            })
    auc_by_fold = pd.DataFrame(fold_auc_rows)

    auc_rows: List[dict] = []
    for repeat, frame in predictions.groupby("repeat", sort=True):
        for metric in metrics:
            raw_auc, n_valid = auc_from_scores(frame["cohort"], frame[metric])
            directional_auc, direction, ild_auc = direction_fields(raw_auc)
            auc_rows.append({
                "repeat": int(repeat),
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "auc_RA_positive_raw": raw_auc,
                "auc_RAILD_positive_raw": ild_auc,
                "directional_auc": directional_auc,
                "higher_score_cohort": direction,
                "n_valid": n_valid,
                "n_expected": int(len(frame)),
                "aggregation": "pooled_OOF_across_5_folds",
                "is_primary_repeat_result": True,
            })
    auc_by_repeat = pd.DataFrame(auc_rows)

    summary_rows: List[dict] = []
    for metric in metrics:
        sub = auc_by_repeat.loc[auc_by_repeat["metric"] == metric]
        fold_sub = auc_by_fold.loc[auc_by_fold["metric"] == metric]
        raw = sub["auc_RA_positive_raw"].to_numpy(dtype=float)
        directional = sub["directional_auc"].to_numpy(dtype=float)
        fold_raw = fold_sub["auc_RA_positive_raw"].to_numpy(dtype=float)
        valid_raw = raw[np.isfinite(raw)]
        valid_directional = directional[np.isfinite(directional)]
        valid_fold_raw = fold_raw[np.isfinite(fold_raw)]
        aggregate_auc, aggregate_n = auc_from_scores(sample_avg["cohort"], sample_avg[metric])
        aggregate_directional, aggregate_direction, aggregate_ild_auc = direction_fields(aggregate_auc)
        if valid_raw.size:
            q025, q975 = np.quantile(valid_raw, [0.025, 0.975])
            direction_ra_fraction = float(np.mean(valid_raw >= 0.5))
        else:
            q025 = q975 = direction_ra_fraction = float("nan")
        if valid_fold_raw.size:
            fold_q025, fold_q975 = np.quantile(valid_fold_raw, [0.025, 0.975])
        else:
            fold_q025 = fold_q975 = float("nan")
        summary_rows.append({
            "metric": metric,
            "metric_label": METRIC_LABELS[metric],
            "primary_resampling_unit": "complete_5fold_repeat",
            "primary_auc_estimate": "median_of_repeat_pooled_OOF_AUC",
            "repeats_valid": int(valid_raw.size),
            "repeat_auc_mean": float(np.mean(valid_raw)) if valid_raw.size else float("nan"),
            "repeat_auc_sd": float(np.std(valid_raw, ddof=1)) if valid_raw.size > 1 else float("nan"),
            "repeat_auc_median": float(np.median(valid_raw)) if valid_raw.size else float("nan"),
            "repeat_auc_q025": float(q025),
            "repeat_auc_q975": float(q975),
            "repeat_directional_auc_median": float(np.median(valid_directional)) if valid_directional.size else float("nan"),
            "fraction_repeats_RA_higher": direction_ra_fraction,
            "fold_auc_diagnostic_n": int(valid_fold_raw.size),
            "fold_auc_diagnostic_median": float(np.median(valid_fold_raw)) if valid_fold_raw.size else float("nan"),
            "fold_auc_diagnostic_q025": float(fold_q025),
            "fold_auc_diagnostic_q975": float(fold_q975),
            "sample_averaged_auc_RA_positive_raw": aggregate_auc,
            "sample_averaged_auc_RAILD_positive_raw": aggregate_ild_auc,
            "sample_averaged_directional_auc": aggregate_directional,
            "sample_averaged_higher_score_cohort": aggregate_direction,
            "sample_averaged_n_valid": aggregate_n,
            "sample_averaged_role": "secondary_bagged_descriptive_result",
        })
    return sample_avg, auc_by_repeat, auc_by_fold, pd.DataFrame(summary_rows)


def stability_summary(stability: pd.DataFrame) -> pd.DataFrame:
    if stability.empty:
        return pd.DataFrame(columns=["dictionary", "n_pairs", "jaccard_mean", "jaccard_median", "jaccard_q025", "jaccard_q975"])
    rows: List[dict] = []
    for dictionary, frame in stability.groupby("dictionary"):
        values = frame["jaccard"].to_numpy(dtype=float)
        rows.append({
            "dictionary": dictionary,
            "n_pairs": int(len(values)),
            "jaccard_mean": float(np.mean(values)),
            "jaccard_median": float(np.median(values)),
            "jaccard_q025": float(np.quantile(values, 0.025)),
            "jaccard_q975": float(np.quantile(values, 0.975)),
        })
    return pd.DataFrame(rows)


def richness_correlations(sample_avg: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    rows: List[dict] = []
    for metric in metrics:
        rho, n_valid = spearman_pair(sample_avg["sample_unique_clone_count"], sample_avg[metric])
        rows.append({
            "metric": metric,
            "metric_label": METRIC_LABELS[metric],
            "spearman_rho_with_sample_unique_clones": rho,
            "n_valid": n_valid,
        })
    return pd.DataFrame(rows)


def run_scope_cv(
    dataset: ScopeDataset,
    folds: int = 5,
    repeats: int = 100,
    candidates: int = 300,
    seed: int = 20260711,
    threshold_pct: float = 20.0,
    delta_pct: float = 10.0,
    save_dictionaries: bool = False,
) -> ScopeRunResult:
    if folds < 2 or repeats < 1 or candidates < 1:
        raise ValueError("folds>=2, repeats>=1 and candidates>=1 are required")
    balance_cols = [c for c in ("batch", "material", "sex") if c in dataset.metadata.columns]
    if dataset.scope != "total":
        balance_cols = [c for c in balance_cols if c != "material"]

    assignment_rows: List[dict] = []
    prediction_rows: List[dict] = []
    fold_rows: List[dict] = []
    balance_rows: List[dict] = []
    stability_rows: List[dict] = []
    dictionary_rows: List[dict] = []
    used_signatures: Set[str] = set()

    for repeat in range(1, repeats + 1):
        fold_ids, split_quality, chosen_seed, signature = optimize_split(
            dataset.metadata,
            folds,
            candidates,
            derived_seed(seed, dataset.scope, repeat),
            balance_cols,
            forbidden=used_signatures,
        )
        used_signatures.add(signature)
        validate_fold_assignment(dataset.metadata, fold_ids, folds)
        balance_rows.extend(fold_balance_rows(dataset.metadata, fold_ids, repeat, dataset.scope))

        repeat_dictionaries: Dict[int, DictionaryResult] = {}
        for sample_index, sample_id in enumerate(dataset.sample_ids):
            assignment_rows.append({
                "scope": dataset.scope,
                "repeat": repeat,
                "fold": int(fold_ids[sample_index]),
                "sample_id": sample_id,
                "cohort": dataset.labels[sample_index],
                "split_quality_score": split_quality,
                "chosen_seed": chosen_seed,
                "partition_signature": signature,
            })

        for fold in range(1, folds + 1):
            validation_indices = np.flatnonzero(fold_ids == fold)
            training_indices = np.flatnonzero(fold_ids != fold)
            dictionary = build_dictionary(dataset, training_indices, threshold_pct, delta_pct)
            repeat_dictionaries[fold] = dictionary
            validation_labels = dataset.labels[validation_indices]
            fold_rows.append({
                "scope": dataset.scope,
                "repeat": repeat,
                "fold": fold,
                "split_quality_score": split_quality,
                "chosen_seed": chosen_seed,
                "n_train": int(len(training_indices)),
                "n_validation": int(len(validation_indices)),
                "n_train_RA": dictionary.n_ra_train,
                "n_train_ILD": dictionary.n_ild_train,
                "n_validation_RA": int(np.sum(validation_labels == "RA")),
                "n_validation_ILD": int(np.sum(validation_labels == "ILD")),
                "RA_threshold_count": dictionary.ra_threshold_count,
                "ILD_threshold_count": dictionary.ild_threshold_count,
                "RA_dict_size": int(len(dictionary.ra_ids)),
                "ILD_dict_size": int(len(dictionary.ild_ids)),
            })
            if save_dictionaries:
                for dictionary_type, ids in (("RA", dictionary.ra_ids), ("RA-ILD", dictionary.ild_ids)):
                    for clone_id in ids.tolist():
                        dictionary_rows.append({
                            "scope": dataset.scope,
                            "repeat": repeat,
                            "fold": fold,
                            "dictionary": dictionary_type,
                            "clone_id": int(clone_id),
                            "cdr3_aa": str(dataset.clone_names[clone_id]),
                        })

            for sample_index in validation_indices.tolist():
                values = score_sample(dataset, sample_index, dictionary)
                meta_row = dataset.metadata.iloc[sample_index]
                row = {
                    "scope": dataset.scope,
                    "repeat": repeat,
                    "fold": fold,
                    "sample_id": dataset.sample_ids[sample_index],
                    "cohort": dataset.labels[sample_index],
                    "material": str(meta_row.get("material", "")),
                    "batch": str(meta_row.get("batch", "")),
                    "sex": str(meta_row.get("sex", "")),
                }
                row.update(values)
                prediction_rows.append(row)

        for fold_a, fold_b in itertools.combinations(range(1, folds + 1), 2):
            a = repeat_dictionaries[fold_a]
            b = repeat_dictionaries[fold_b]
            stability_rows.append({
                "scope": dataset.scope,
                "repeat": repeat,
                "fold_a": fold_a,
                "fold_b": fold_b,
                "dictionary": "RA",
                "jaccard": jaccard(a.ra_ids, b.ra_ids),
            })
            stability_rows.append({
                "scope": dataset.scope,
                "repeat": repeat,
                "fold_a": fold_a,
                "fold_b": fold_b,
                "dictionary": "RA-ILD",
                "jaccard": jaccard(a.ild_ids, b.ild_ids),
            })

    assignments = pd.DataFrame(assignment_rows)
    predictions = pd.DataFrame(prediction_rows)
    fold_summary = pd.DataFrame(fold_rows)
    fold_balance = pd.DataFrame(balance_rows)
    stability = pd.DataFrame(stability_rows)
    members = pd.DataFrame(dictionary_rows) if save_dictionaries else None

    expected_predictions = dataset.n_samples * repeats
    if len(predictions) != expected_predictions:
        raise RuntimeError("Expected {} OOF rows; got {}".format(expected_predictions, len(predictions)))
    per_repeat_counts = predictions.groupby(["repeat", "sample_id"]).size()
    if not per_repeat_counts.eq(1).all():
        raise RuntimeError("Each sample must have exactly one OOF score per repeat")
    if predictions.duplicated(["repeat", "sample_id"]).any():
        raise RuntimeError("Duplicated OOF predictions detected")

    sample_avg, auc_by_repeat, auc_by_fold, metric_summary = summarize_predictions(predictions, ALL_METRICS)
    stab_summary = stability_summary(stability)
    richness = richness_correlations(sample_avg, ALL_METRICS)
    return ScopeRunResult(
        scope=dataset.scope,
        assignments=assignments,
        predictions=predictions,
        sample_averaged=sample_avg,
        auc_by_repeat=auc_by_repeat,
        auc_by_fold=auc_by_fold,
        metric_summary=metric_summary,
        fold_summary=fold_summary,
        fold_balance=fold_balance,
        stability=stability,
        stability_summary=stab_summary,
        richness_correlation=richness,
        dictionary_members=members,
    )


def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _axis_boxplot(axis, data, tick_labels=None, horizontal=False, **kwargs):
    """Call Axes.boxplot across Matplotlib API generations.

    Matplotlib 3.9 renamed ``labels`` to ``tick_labels`` and newer releases
    removed the old keyword. Matplotlib 3.10 also introduced ``orientation``
    as the replacement for ``vert``. Select supported keywords at runtime so
    the Linux workflow works with both older and newer environments.
    """
    import inspect

    parameters = inspect.signature(axis.boxplot).parameters
    if tick_labels is not None:
        if "tick_labels" in parameters:
            kwargs["tick_labels"] = tick_labels
        elif "labels" in parameters:
            kwargs["labels"] = tick_labels

    if horizontal:
        if "orientation" in parameters:
            kwargs["orientation"] = "horizontal"
        elif "vert" in parameters:
            kwargs["vert"] = False

    artists = axis.boxplot(data, **kwargs)

    # Extremely old or wrapped implementations may expose neither keyword.
    if tick_labels is not None and "tick_labels" not in parameters and "labels" not in parameters:
        positions = list(range(1, len(tick_labels) + 1))
        if horizontal:
            axis.set_yticks(positions, tick_labels)
        else:
            axis.set_xticks(positions, tick_labels)
    return artists


def plot_scope_result(result: ScopeRunResult, output_dir: Path) -> List[Path]:
    plt = _import_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []
    colors = {"RA": "#0072B2", "ILD": "#D55E00"}

    # Sample-averaged OOF boxplots: primary metrics only.
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for axis, metric in zip(axes.flat, PRIMARY_METRICS):
        values_ild = result.sample_averaged.loc[result.sample_averaged["cohort"] == "ILD", metric].dropna().to_numpy()
        values_ra = result.sample_averaged.loc[result.sample_averaged["cohort"] == "RA", metric].dropna().to_numpy()
        bp = _axis_boxplot(axis, [values_ild, values_ra], tick_labels=["RA-ILD", "RA"], patch_artist=True, showfliers=False)
        bp["boxes"][0].set_facecolor(colors["ILD"])
        bp["boxes"][1].set_facecolor(colors["RA"])
        for i, vals in enumerate((values_ild, values_ra), start=1):
            if len(vals):
                jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.asarray([0.0])
                axis.scatter(np.full(len(vals), i) + jitter, vals, s=10, alpha=0.35, color="0.35")
        summary = result.metric_summary.loc[result.metric_summary["metric"] == metric].iloc[0]
        axis.set_title("{}\nAUC={:.3f}, {}".format(
            METRIC_LABELS[metric],
            summary["sample_averaged_auc_RA_positive_raw"],
            summary["sample_averaged_higher_score_cohort"],
        ), fontsize=9)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "{}: repeated 5-fold OOF dictionary scores\nSample-level mean across repeats; RA is the ROC-positive class".format(result.scope),
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = output_dir / "enrich_5fold_boxplot_{}.png".format(result.scope)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    # Raw-direction ROC curves from sample-averaged OOF scores.
    fig, axis = plt.subplots(figsize=(10, 8))
    y = (result.sample_averaged["cohort"].to_numpy(dtype=str) == "RA").astype(int)
    for metric in PRIMARY_METRICS:
        score = result.sample_averaged[metric].to_numpy(dtype=float)
        valid = np.isfinite(score)
        if valid.sum() < 2 or len(np.unique(y[valid])) < 2 or len(np.unique(score[valid])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y[valid], score[valid])
        summary = result.metric_summary.loc[result.metric_summary["metric"] == metric].iloc[0]
        label = "{} (AUC {:.3f}; {})".format(
            METRIC_LABELS[metric],
            summary["sample_averaged_auc_RA_positive_raw"],
            summary["sample_averaged_higher_score_cohort"],
        )
        axis.plot(fpr, tpr, linewidth=1.5, label=label)
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.6", linewidth=1)
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("1 - specificity")
    axis.set_ylabel("Sensitivity")
    axis.set_title(
        "{}: repeated 5-fold OOF ROC\nRA is positive; raw score direction is preserved".format(result.scope)
    )
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    axis.grid(alpha=0.15)
    fig.tight_layout()
    path = output_dir / "enrich_5fold_ROC_RA_positive_raw_{}.png".format(result.scope)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    # AUC distribution over repeats.
    fig, axis = plt.subplots(figsize=(12, 7))
    data = [
        result.auc_by_repeat.loc[result.auc_by_repeat["metric"] == metric, "auc_RA_positive_raw"].dropna().to_numpy()
        for metric in PRIMARY_METRICS
    ]
    _axis_boxplot(axis, data, tick_labels=[METRIC_LABELS[m] for m in PRIMARY_METRICS], horizontal=True, showfliers=True)
    axis.axvline(0.5, linestyle="--", color="0.5", linewidth=1)
    axis.set_xlim(0, 1)
    axis.set_xlabel("AUC, RA positive, raw direction")
    axis.set_title("{}: AUC variation across repeated 5-fold partitions".format(result.scope))
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    path = output_dir / "enrich_5fold_AUC_distribution_{}.png".format(result.scope)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    # Dictionary size and stability diagnostics.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _axis_boxplot(
        axes[0],
        [result.fold_summary["RA_dict_size"].to_numpy(), result.fold_summary["ILD_dict_size"].to_numpy()],
        tick_labels=["RA", "RA-ILD"],
        showfliers=True,
    )
    axes[0].set_ylabel("Dictionary clone count")
    axes[0].set_title("Fold-specific dictionary sizes")
    axes[0].grid(axis="y", alpha=0.2)
    stab_data = [
        result.stability.loc[result.stability["dictionary"] == name, "jaccard"].to_numpy()
        for name in ("RA", "RA-ILD")
    ]
    _axis_boxplot(axes[1], stab_data, tick_labels=["RA", "RA-ILD"], showfliers=True)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Pairwise Jaccard")
    axes[1].set_title("Dictionary stability across folds")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("{}: dictionary diagnostics".format(result.scope))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = output_dir / "enrich_5fold_dictionary_diagnostics_{}.png".format(result.scope)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    # Richness correlations.
    fig, axis = plt.subplots(figsize=(10, 7))
    corr = result.richness_correlation.set_index("metric").loc[list(ALL_METRICS)]
    values = corr["spearman_rho_with_sample_unique_clones"].to_numpy(dtype=float)
    axis.barh([METRIC_LABELS[m] for m in ALL_METRICS], values)
    axis.axvline(0, color="0.5", linewidth=1)
    axis.set_xlim(-1, 1)
    axis.set_xlabel("Spearman rho with sample unique-clone count")
    axis.set_title("{}: repertoire-richness diagnostic".format(result.scope))
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    path = output_dir / "enrich_5fold_richness_correlation_{}.png".format(result.scope)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    created.append(path)
    return created


def write_scope_outputs(result: ScopeRunResult, result_dir: Path, plot_dir: Path, make_plots: bool = True) -> Dict[str, str]:
    scope_dir = result_dir / result.scope
    scope_plot_dir = plot_dir / result.scope
    scope_dir.mkdir(parents=True, exist_ok=True)
    files: Dict[str, Path] = {
        "assignments": scope_dir / "enrich_5fold_assignments_{}.csv.gz".format(result.scope),
        "oof_predictions": scope_dir / "enrich_5fold_oof_predictions_{}.csv.gz".format(result.scope),
        "sample_averaged": scope_dir / "enrich_5fold_sample_averaged_{}.csv".format(result.scope),
        "auc_by_repeat": scope_dir / "enrich_5fold_auc_by_repeat_{}.csv".format(result.scope),
        "auc_by_fold_diagnostic": scope_dir / "enrich_5fold_auc_by_fold_diagnostic_{}.csv".format(result.scope),
        "metric_summary": scope_dir / "enrich_5fold_metric_summary_{}.csv".format(result.scope),
        "fold_summary": scope_dir / "enrich_5fold_fold_summary_{}.csv".format(result.scope),
        "fold_balance": scope_dir / "enrich_5fold_fold_balance_{}.csv.gz".format(result.scope),
        "dictionary_stability": scope_dir / "enrich_5fold_dictionary_stability_{}.csv".format(result.scope),
        "dictionary_stability_summary": scope_dir / "enrich_5fold_dictionary_stability_summary_{}.csv".format(result.scope),
        "richness_correlation": scope_dir / "enrich_5fold_richness_correlation_{}.csv".format(result.scope),
    }
    result.assignments.to_csv(files["assignments"], index=False, compression="gzip")
    result.predictions.to_csv(files["oof_predictions"], index=False, compression="gzip")
    result.sample_averaged.to_csv(files["sample_averaged"], index=False)
    result.auc_by_repeat.to_csv(files["auc_by_repeat"], index=False)
    result.auc_by_fold.to_csv(files["auc_by_fold_diagnostic"], index=False)
    result.metric_summary.to_csv(files["metric_summary"], index=False)
    result.fold_summary.to_csv(files["fold_summary"], index=False)
    result.fold_balance.to_csv(files["fold_balance"], index=False, compression="gzip")
    result.stability.to_csv(files["dictionary_stability"], index=False)
    result.stability_summary.to_csv(files["dictionary_stability_summary"], index=False)
    result.richness_correlation.to_csv(files["richness_correlation"], index=False)
    if result.dictionary_members is not None:
        dict_path = scope_dir / "enrich_5fold_dictionary_members_{}.csv.gz".format(result.scope)
        result.dictionary_members.to_csv(dict_path, index=False, compression="gzip")
        files["dictionary_members"] = dict_path
    if make_plots:
        for index, path in enumerate(plot_scope_result(result, scope_plot_dir), start=1):
            files["plot_{}".format(index)] = path
    return {name: str(path) for name, path in files.items()}


def write_run_summary(
    path: Path,
    config: Mapping[str, object],
    datasets: Mapping[str, ScopeDataset],
    results: Mapping[str, ScopeRunResult],
    runtime_seconds: float,
) -> None:
    lines: List[str] = [
        "# Enrich-dictionary repeated 5-fold validation summary",
        "",
        "## Design",
        "",
        "- Each validation fold is scored with dictionaries rebuilt from that fold's training samples only.",
        "- All validation samples in one fold use the same RA and RA-ILD dictionaries.",
        "- Cohort is hard-stratified. Batch/material/sex are secondary balance targets when available.",
        "- Primary AUC unit: one complete 5-fold repeat, pooling all five OOF folds so every sample contributes exactly once.",
        "- With {} repeats, the primary AUC table contains {} values per metric.".format(config.get("repeats"), config.get("repeats")),
        "- The {} fold-level AUC values per metric are diagnostic only because folds within and across repeats are correlated and not independent replicates.".format(int(config.get("folds", 5)) * int(config.get("repeats", 1))),
        "- AUC uses RA as the positive class and preserves the raw score direction.",
        "- Repeat percentile ranges are descriptive split-sensitivity ranges, not independent confidence intervals.",
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
        lines.extend([
            "### {}".format(scope),
            "",
            "- Samples: **{}** (RA {}, RA-ILD {})".format(
                dataset.n_samples, int(counts.get("RA", 0)), int(counts.get("ILD", 0))
            ),
            "- Prescreened clone universe: **{} / {}** (minimum total presence {})".format(
                dataset.n_universe, dataset.prescreen_original_universe, dataset.prescreen_min_presence
            ),
            "- Median fold RA dictionary size: **{:.1f}**".format(result.fold_summary["RA_dict_size"].median()),
            "- Median fold RA-ILD dictionary size: **{:.1f}**".format(result.fold_summary["ILD_dict_size"].median()),
        ])
        for dictionary_name in ("RA", "RA-ILD"):
            sub = result.stability_summary.loc[result.stability_summary["dictionary"] == dictionary_name]
            if not sub.empty:
                lines.append("- Median {} dictionary pairwise Jaccard: **{:.3f}**".format(dictionary_name, float(sub.iloc[0]["jaccard_median"])))
        lines.extend(["", "| Metric | Secondary mean-score AUC (RA+) | Direction | Primary median repeat pooled-OOF AUC | 2.5%–97.5% repeat range |", "|---|---:|---|---:|---:|"])
        for metric in PRIMARY_METRICS:
            row = result.metric_summary.loc[result.metric_summary["metric"] == metric].iloc[0]
            lines.append(
                "| {} | {:.3f} | {} | {:.3f} | {:.3f}–{:.3f} |".format(
                    METRIC_LABELS[metric],
                    row["sample_averaged_auc_RA_positive_raw"],
                    row["sample_averaged_higher_score_cohort"],
                    row["repeat_auc_median"],
                    row["repeat_auc_q025"],
                    row["repeat_auc_q975"],
                )
            )
        lines.append("")
    lines.extend([
        "## Interpretation safeguards",
        "",
        "- Non-cross-validated in-sample AUC is not a generalization estimate.",
        "- A metric named for the RA-ILD dictionary may still be higher in RA; use the explicit direction column.",
        "- Dictionary coverage corrects dictionary size but not sample repertoire richness.",
        "- Review richness correlations and dictionary Jaccard stability before biological interpretation.",
        "",
        "Runtime: **{:.2f} seconds**".format(runtime_seconds),
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_project_analysis(
    project_root: Path,
    scopes: Sequence[str] = ("total", "pbmc", "buffycoat"),
    folds: int = 5,
    repeats: int = 100,
    candidates: int = 300,
    seed: int = 20260711,
    threshold_pct: float = 20.0,
    delta_pct: float = 10.0,
    allow_count_mismatch: bool = False,
    save_dictionaries: bool = False,
    make_plots: bool = True,
    overwrite: bool = False,
    result_dir: Optional[Path] = None,
    plot_dir: Optional[Path] = None,
) -> Dict[str, object]:
    started = time.time()
    root = project_root.expanduser().resolve()
    metadata_path = root / "TRB/metadata.csv"
    aa_dir = root / "TRB/result/01_AA_clone_table"
    result_dir = (result_dir or (root / "TRB/result/enrich_dictionary_5fold_cv_v1")).resolve()
    plot_dir = (plot_dir or (root / "TRB/gradient/enrich_dictionary_5fold_cv_v1")).resolve()
    manifest_path = result_dir / "enrich_5fold_run_manifest.json"
    summary_path = result_dir / "enrich_5fold_summary.md"
    normalized_scopes = [str(scope).lower() for scope in scopes]
    invalid_scopes = sorted(set(normalized_scopes).difference(EXPECTED_SCOPE_COUNTS))
    if invalid_scopes:
        raise ValueError("Unsupported scopes: {}".format(invalid_scopes))
    if result_dir.exists() and any(result_dir.iterdir()) and not overwrite:
        raise FileExistsError("Result directory is not empty; use --overwrite: {}".format(result_dir))
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        # Remove only Batch-16 outputs for requested scopes; never touch earlier analyses.
        for scope in normalized_scopes:
            for base in (result_dir, plot_dir):
                target = base / scope
                if target.exists():
                    shutil.rmtree(target)
        for path in (manifest_path, summary_path):
            if path.exists():
                path.unlink()

    metadata = load_metadata(metadata_path)
    file_map = build_aa_file_map(aa_dir, metadata["libraryid"].tolist())
    repertoires = load_repertoires(metadata, file_map)

    datasets: Dict[str, ScopeDataset] = {}
    results: Dict[str, ScopeRunResult] = {}
    outputs: Dict[str, Dict[str, str]] = {}
    for scope in normalized_scopes:
        dataset = build_scope_dataset(
            metadata,
            repertoires,
            scope,
            folds,
            threshold_pct,
            allow_count_mismatch=allow_count_mismatch,
        )
        datasets[scope] = dataset
        result = run_scope_cv(
            dataset,
            folds=folds,
            repeats=repeats,
            candidates=candidates,
            seed=seed,
            threshold_pct=threshold_pct,
            delta_pct=delta_pct,
            save_dictionaries=save_dictionaries,
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
        "folds": folds,
        "repeats": repeats,
        "candidates": candidates,
        "seed": seed,
        "threshold_pct": threshold_pct,
        "delta_pct": delta_pct,
        "allow_count_mismatch": allow_count_mismatch,
        "save_dictionaries": save_dictionaries,
        "make_plots": make_plots,
        "result_dir": str(result_dir),
        "plot_dir": str(plot_dir),
        "positive_class": "RA",
        "score_direction": "raw",
        "primary_metrics": list(PRIMARY_METRICS),
        "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
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
    write_run_summary(summary_path, config, datasets, results, runtime)
    return manifest
