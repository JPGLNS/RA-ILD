#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fold-local feature construction for Phase-6 unified TRB modeling.

Static repertoire features are label-independent and are loaded once from the
Phase-2 core tables.  Training-derived features are rebuilt for every current
training partition:

* 3-mer vocabulary is ranked on current training samples only;
* enriched-dictionary validation scores use a dictionary fit on current
  training samples only;
* enriched-dictionary training scores use exact leave-one-out references so a
  sample never contributes to the dictionary used to score itself.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from .dynamic_repertoire import (
    DynamicRepertoireSource,
    read_and_filter_metadata,
    resolve_dynamic_repertoire,
    validate_dynamic_source,
)
from .enriched_dictionary import (
    ALL_SCORE_COLUMNS,
    SampleRepertoire,
    fit_enriched_dictionary,
    load_repertoires,
    transform_repertoires,
)
from .feature_framework import FeatureRegistry, load_feature_registry
from .unified_experiment import UnifiedExperimentSpec

PathLike = Union[str, Path]
AA_SET = frozenset("ACDEFGHIKLMNPQRSTVWY")


class FoldFeatureError(ValueError):
    pass


def unique_3mers(sequence: str) -> Set[str]:
    seq = str(sequence).strip().upper()
    if len(seq) < 3 or any(aa not in AA_SET for aa in seq):
        return set()
    return {seq[i : i + 3] for i in range(len(seq) - 2)}


@dataclass(frozen=True)
class FoldFeatureInputs:
    metadata: pd.DataFrame
    static_values: pd.DataFrame
    static_feature_names: Tuple[str, ...]
    clinical_numeric: Tuple[str, ...]
    clinical_categorical: Tuple[str, ...]
    source: DynamicRepertoireSource
    kmer_unweighted: Mapping[str, Mapping[str, float]]
    kmer_weighted: Mapping[str, Mapping[str, float]]
    enrich_repertoires: Mapping[str, SampleRepertoire]

    @property
    def sample_ids(self) -> Tuple[str, ...]:
        return tuple(self.metadata["sample_id"].astype(str).tolist())


@dataclass(frozen=True)
class FoldFeatureBuild:
    train: pd.DataFrame
    valid: pd.DataFrame
    numeric_columns: Tuple[str, ...]
    categorical_columns: Tuple[str, ...]
    kmer_vocabulary: pd.DataFrame
    audit: pd.DataFrame


def _read_metadata_pair(
    train_path: PathLike,
    test_path: PathLike,
    scope: str,
) -> pd.DataFrame:
    train = read_and_filter_metadata(train_path, scope)
    test = read_and_filter_metadata(test_path, scope)
    train = train.copy()
    test = test.copy()
    train["_original_partition"] = "train70"
    test["_original_partition"] = "test30"
    frame = pd.concat([train, test], ignore_index=True, sort=False)
    if frame["libraryid"].astype(str).duplicated().any():
        raise FoldFeatureError("combined metadata contains duplicate libraryid")
    frame = frame.rename(columns={"libraryid": "sample_id"})
    frame["sample_id"] = frame["sample_id"].astype(str)
    required = {"sample_id", "cohort", "age", "sex"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise FoldFeatureError(f"metadata missing columns required by unified ML: {missing}")
    frame["cohort"] = frame["cohort"].astype(str).str.upper()
    if set(frame["cohort"]) != {"RA", "ILD"}:
        raise FoldFeatureError("combined scope must contain both RA and ILD")
    return frame.reset_index(drop=True)


def _verify_step02_source(step02_dir: PathLike, expected_source_id: str, label: str) -> bool:
    path = Path(step02_dir).expanduser().resolve() / "02_resolved_feature_source.json"
    if not path.is_file():
        return False
    import json
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_id = (payload.get("repertoire_input") or {}).get("source_id")
    if source_id != expected_source_id:
        raise FoldFeatureError(
            f"{label} Step02 repertoire source mismatch: observed={source_id!r}, expected={expected_source_id!r}"
        )
    return True


def _read_core(step02_dir: PathLike, label: str) -> pd.DataFrame:
    path = Path(step02_dir).expanduser().resolve() / "02_core_sample_features.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{label} core features not found: {path}")
    frame = pd.read_csv(path)
    if "sample_id" not in frame.columns:
        raise FoldFeatureError(f"{path} lacks sample_id")
    frame["sample_id"] = frame["sample_id"].astype(str)
    if frame["sample_id"].duplicated().any():
        raise FoldFeatureError(f"{path} contains duplicate sample_id")
    return frame


def select_static_features(
    registry: FeatureRegistry,
    core_columns: Sequence[str],
    group_ids: Sequence[str],
) -> Tuple[str, ...]:
    columns = tuple(str(x) for x in core_columns if str(x) != "sample_id")
    selected: List[str] = []
    for group_id in group_ids:
        if group_id not in registry.feature_groups:
            raise FoldFeatureError(f"unknown static group {group_id!r}")
        group = registry.feature_groups[group_id]
        group_columns: List[str] = []
        for exact in group.selector.exact:
            if exact in columns:
                group_columns.append(exact)
        for prefix in group.selector.prefixes:
            group_columns.extend([c for c in columns if c.startswith(prefix) and c not in group_columns])
        if not group_columns:
            raise FoldFeatureError(f"static group {group_id!r} selected no core columns")
        drops = set(group.reference_drop_columns)
        group_columns = [c for c in group_columns if c not in drops]
        selected.extend(group_columns)
    if len(selected) != len(set(selected)):
        duplicates = [x for x in selected if selected.count(x) > 1]
        raise FoldFeatureError(f"static feature selection contains duplicates: {sorted(set(duplicates))[:20]}")
    return tuple(selected)


def _clinical_columns(spec: UnifiedExperimentSpec) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    numeric: List[str] = []
    categorical: List[str] = []
    for group in spec.base.features.clinical_groups:
        if group == "clinical_age_sex":
            numeric.append("age")
            categorical.append("sex")
        elif group == "clinical_material":
            categorical.append("material")
        else:
            raise FoldFeatureError(f"unsupported clinical group: {group}")
    return tuple(numeric), tuple(categorical)


def _load_kmer_map(path: Path, weight_column: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    try:
        frame = pd.read_csv(path, usecols=["cdr3_aa", weight_column])
    except ValueError as exc:
        raise FoldFeatureError(f"{path} lacks cdr3_aa/{weight_column}") from exc
    frame["cdr3_aa"] = frame["cdr3_aa"].astype("string").str.strip().str.upper()
    frame[weight_column] = pd.to_numeric(frame[weight_column], errors="coerce")
    valid = frame["cdr3_aa"].notna() & frame["cdr3_aa"].ne("") & frame[weight_column].notna()
    frame = frame.loc[valid].copy()
    if frame.empty:
        raise FoldFeatureError(f"no valid sequences in {path}")
    if (~np.isfinite(frame[weight_column].to_numpy(float))).any() or (frame[weight_column] < 0).any():
        raise FoldFeatureError(f"invalid {weight_column} in {path}")
    frame = frame.groupby("cdr3_aa", sort=False, as_index=False)[weight_column].sum()
    weight = frame[weight_column].to_numpy(float)
    total = float(weight.sum())
    if total <= 0:
        raise FoldFeatureError(f"zero repertoire weight in {path}")
    weight = weight / total
    n_clones = int(len(frame))
    counts: Dict[str, int] = {}
    sums: Dict[str, float] = {}
    for sequence, clone_weight in zip(frame["cdr3_aa"].astype(str), weight):
        for kmer in unique_3mers(sequence):
            counts[kmer] = counts.get(kmer, 0) + 1
            sums[kmer] = sums.get(kmer, 0.0) + float(clone_weight)
    return (
        {k: value / n_clones for k, value in counts.items()},
        sums,
    )


def load_fold_feature_inputs(
    spec: UnifiedExperimentSpec,
    *,
    repository_root: PathLike,
    train_metadata: PathLike,
    test_metadata: PathLike,
    train_step02_dir: PathLike,
    test_step02_dir: PathLike,
    registry_path: Optional[PathLike] = None,
    validate_source_files: bool = True,
) -> FoldFeatureInputs:
    root = Path(repository_root).expanduser().resolve()
    metadata = _read_metadata_pair(train_metadata, test_metadata, spec.base.cohort.scope)
    train_core = _read_core(train_step02_dir, "TRAIN")
    test_core = _read_core(test_step02_dir, "TEST")
    core = pd.concat([train_core, test_core], ignore_index=True, sort=False)
    if core["sample_id"].duplicated().any():
        raise FoldFeatureError("combined core features contain duplicate sample_id")
    ids = metadata["sample_id"].astype(str).tolist()
    core = core.set_index("sample_id")
    missing = [sid for sid in ids if sid not in core.index]
    if missing:
        raise FoldFeatureError(f"core features missing scope samples: {missing[:20]}")
    core = core.loc[ids].reset_index()
    if core.isna().any().any():
        bad = core.isna().sum()
        raise FoldFeatureError(f"core features contain missing values: {bad[bad>0].to_dict()}")

    registry_file = (
        Path(registry_path).expanduser().resolve()
        if registry_path is not None
        else root / "TRB/set/configs/feature_framework/trb_feature_registry_v1.yaml"
    )
    registry = load_feature_registry(registry_file)
    static_names = select_static_features(
        registry,
        core.columns,
        spec.base.features.static_tcr_groups,
    )
    static = core.loc[:, ["sample_id", *static_names]].copy()
    clinical_numeric, clinical_categorical = _clinical_columns(spec)
    for column in (*clinical_numeric, *clinical_categorical):
        if column not in metadata.columns:
            raise FoldFeatureError(f"metadata lacks selected clinical column {column}")

    source = resolve_dynamic_repertoire(spec.base, root)
    _verify_step02_source(train_step02_dir, source.source_id, "TRAIN")
    _verify_step02_source(test_step02_dir, source.source_id, "TEST")
    if validate_source_files:
        validate_dynamic_source(source, ids, deep=False)

    kmer_u: Dict[str, Mapping[str, float]] = {}
    kmer_w: Dict[str, Mapping[str, float]] = {}
    if spec.base.features.kmer.enabled:
        for sid in ids:
            u, w = _load_kmer_map(source.sample_path(sid), source.weight_column)
            kmer_u[sid] = u
            kmer_w[sid] = w

    enrich: Mapping[str, SampleRepertoire] = {}
    if spec.base.features.enriched_dictionary.enabled:
        enrich_meta = metadata.rename(columns={"sample_id": "libraryid"})
        enrich = load_repertoires(enrich_meta, source, id_col="libraryid")

    return FoldFeatureInputs(
        metadata=metadata,
        static_values=static,
        static_feature_names=static_names,
        clinical_numeric=clinical_numeric,
        clinical_categorical=clinical_categorical,
        source=source,
        kmer_unweighted=kmer_u,
        kmer_weighted=kmer_w,
        enrich_repertoires=enrich,
    )


def rank_training_kmers(
    train_ids: Sequence[str],
    inputs: FoldFeatureInputs,
    spec: UnifiedExperimentSpec,
) -> pd.DataFrame:
    kspec = spec.base.features.kmer
    if not kspec.enabled:
        return pd.DataFrame(columns=["selection_rank", "kmer", "keep"])
    ids = tuple(map(str, train_ids))
    if not ids or len(ids) != len(set(ids)):
        raise FoldFeatureError("kmer training IDs must be non-empty and unique")
    all_kmers: Set[str] = set()
    for sid in ids:
        all_kmers.update(inputs.kmer_unweighted[sid])
        all_kmers.update(inputs.kmer_weighted[sid])
    rows: List[Dict[str, Any]] = []
    n = len(ids)
    for kmer in sorted(all_kmers):
        u = np.array([inputs.kmer_unweighted[sid].get(kmer, 0.0) for sid in ids], dtype=float)
        w = np.array([inputs.kmer_weighted[sid].get(kmer, 0.0) for sid in ids], dtype=float)
        count = int(np.count_nonzero(u > 0))
        prevalence = count / n
        var_u = float(np.var(u, ddof=0))
        var_w = float(np.var(w, ddof=0))
        reasons = []
        if count < kspec.min_sample_count:
            reasons.append("low_sample_count")
        if prevalence < kspec.min_prevalence:
            reasons.append("low_prevalence")
        if var_u < kspec.min_unweighted_variance and var_w < kspec.min_weighted_variance:
            reasons.append("low_variance_both")
        rows.append({
            "kmer": kmer,
            "training_sample_count": count,
            "training_sample_prevalence": prevalence,
            "variance_unweighted_value": var_u,
            "variance_weighted_value": var_w,
            "ranking_total_variance": var_u + var_w,
            "passes_filters": not reasons,
            "filter_reason": "retained_candidate" if not reasons else ";".join(reasons),
        })
    if not rows:
        raise FoldFeatureError("no 3-mers were found in current training partition")
    frame = pd.DataFrame(rows)
    candidates = frame.loc[frame["passes_filters"]].sort_values(
        ["training_sample_prevalence", "ranking_total_variance", "kmer"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    top_n = kspec.top_n
    if top_n is not None and len(candidates) < int(top_n):
        raise FoldFeatureError(
            f"only {len(candidates)} 3-mers passed current-fold filters; requested top_n={top_n}"
        )
    keep_n = len(candidates) if top_n is None else int(top_n)
    selected = candidates.head(keep_n).copy()
    selected.insert(0, "selection_rank", np.arange(1, len(selected) + 1))
    selected["keep"] = True
    return selected


def _kmer_feature_frame(
    sample_ids: Sequence[str],
    selected_kmers: Sequence[str],
    inputs: FoldFeatureInputs,
    representation: str,
) -> pd.DataFrame:
    ids = tuple(map(str, sample_ids))
    kmers = tuple(map(str, selected_kmers))
    values = inputs.kmer_unweighted if representation == "unweighted" else inputs.kmer_weighted
    matrix = np.zeros((len(ids), len(kmers)), dtype=float)
    lookup = {k: j for j, k in enumerate(kmers)}
    for i, sid in enumerate(ids):
        for kmer, value in values[sid].items():
            j = lookup.get(kmer)
            if j is not None:
                matrix[i, j] = float(value)
    prefix = f"{representation}_3mer_"
    out = pd.DataFrame(matrix, columns=[prefix + k for k in kmers])
    out.insert(0, "sample_id", ids)
    return out


def _presence_counts(ids: Sequence[str], repertoires: Mapping[str, SampleRepertoire]) -> Counter:
    out: Counter[str] = Counter()
    for sid in ids:
        out.update(repertoires[str(sid)].clones)
    return out


def exact_loo_enriched_scores(
    metadata: pd.DataFrame,
    sample_ids: Sequence[str],
    repertoires: Mapping[str, SampleRepertoire],
    *,
    threshold_pct: float,
    delta_pct: float,
) -> pd.DataFrame:
    """Exact but vectorized leave-one-out enriched scores for training samples.

    For each class, dictionary membership has only two states for an excluded
    sample: the clone was present in that sample or absent.  Precomputing those
    two membership masks makes the result exactly equivalent to refitting the
    dictionary N times while avoiding N full universe scans.
    """
    ids = tuple(map(str, sample_ids))
    meta = metadata.set_index("sample_id").loc[list(ids)]
    labels = meta["cohort"].astype(str).str.upper().to_dict()
    ra_ids = [sid for sid in ids if labels[sid] == "RA"]
    ild_ids = [sid for sid in ids if labels[sid] == "ILD"]
    if len(ra_ids) < 2 or len(ild_ids) < 2:
        raise FoldFeatureError("exact LOO enrich requires at least two RA and two ILD training samples")
    ra_counts_map = _presence_counts(ra_ids, repertoires)
    ild_counts_map = _presence_counts(ild_ids, repertoires)
    universe = tuple(sorted(set(ra_counts_map) | set(ild_counts_map)))
    if not universe:
        raise FoldFeatureError("enriched dictionary universe is empty")
    index = {clone: i for i, clone in enumerate(universe)}
    ra_counts = np.fromiter((ra_counts_map.get(c, 0) for c in universe), dtype=np.int32)
    ild_counts = np.fromiter((ild_counts_map.get(c, 0) for c in universe), dtype=np.int32)
    n_ra = len(ra_ids)
    n_ild = len(ild_ids)

    def states(excluded_class: str):
        nra = n_ra - (excluded_class == "RA")
        nild = n_ild - (excluded_class == "ILD")
        if nra < 1 or nild < 1:
            raise FoldFeatureError("LOO enrich left an empty cohort")
        # absent from the excluded sample: group counts stay unchanged
        ra_abs = ra_counts.astype(float) / nra * 100.0
        ild_abs = ild_counts.astype(float) / nild * 100.0
        # present in the excluded sample: decrement its own cohort count
        ra_pres_counts = ra_counts - (1 if excluded_class == "RA" else 0)
        ild_pres_counts = ild_counts - (1 if excluded_class == "ILD" else 0)
        ra_pres = ra_pres_counts.astype(float) / nra * 100.0
        ild_pres = ild_pres_counts.astype(float) / nild * 100.0
        ra_dict_abs = (ra_abs >= threshold_pct) & ((ra_abs - ild_abs) >= delta_pct)
        ild_dict_abs = (ild_abs >= threshold_pct) & ((ild_abs - ra_abs) >= delta_pct)
        ra_dict_pres = (ra_pres >= threshold_pct) & ((ra_pres - ild_pres) >= delta_pct)
        ild_dict_pres = (ild_pres >= threshold_pct) & ((ild_pres - ra_pres) >= delta_pct)
        return ra_dict_abs, ra_dict_pres, ild_dict_abs, ild_dict_pres

    state = {"RA": states("RA"), "ILD": states("ILD")}
    rows: List[Dict[str, float]] = []
    for sid in ids:
        rep = repertoires[sid]
        idx = np.fromiter((index[c] for c in rep.clones if c in index), dtype=np.int64)
        rf = np.fromiter((rep.read_fraction.get(c, 0.0) for c in rep.clones if c in index), dtype=float)
        ra_abs, ra_pres, ild_abs, ild_pres = state[labels[sid]]
        # dictionary size = absent-state baseline + corrections for clones present in excluded sample
        ra_size = int(ra_abs.sum() + np.sum(ra_pres[idx].astype(np.int8) - ra_abs[idx].astype(np.int8)))
        ild_size = int(ild_abs.sum() + np.sum(ild_pres[idx].astype(np.int8) - ild_abs[idx].astype(np.int8)))
        ra_hit_mask = ra_pres[idx]
        ild_hit_mask = ild_pres[idx]
        ra_count = int(ra_hit_mask.sum())
        ild_count = int(ild_hit_mask.sum())
        ra_rf = float(rf[ra_hit_mask].sum()) if len(rf) else 0.0
        ild_rf = float(rf[ild_hit_mask].sum()) if len(rf) else 0.0
        sample_total = rep.unique_clone_count
        ra_rate = float(ra_count / ra_size) if ra_size > 0 else float("nan")
        ild_rate = float(ild_count / ild_size) if ild_size > 0 else float("nan")
        ra_sample = float(ra_count / sample_total) if sample_total > 0 else float("nan")
        ild_sample = float(ild_count / sample_total) if sample_total > 0 else float("nan")
        rows.append({
            "sample_id": sid,
            "RA_dict_clone_count": float(ra_count),
            "ILD_dict_clone_count": float(ild_count),
            "RA_dict_hit_rate": ra_rate,
            "ILD_dict_hit_rate": ild_rate,
            "RA_dict_read_fraction_sum": ra_rf,
            "ILD_dict_read_fraction_sum": ild_rf,
            "RA_minus_ILD_count": float(ra_count - ild_count),
            "RA_minus_ILD_read_fraction": ra_rf - ild_rf,
            "RA_minus_ILD_hit_rate": ra_rate - ild_rate if np.isfinite(ra_rate) and np.isfinite(ild_rate) else float("nan"),
            "RA_hit_fraction_of_sample": ra_sample,
            "ILD_hit_fraction_of_sample": ild_sample,
            "RA_minus_ILD_sample_fraction": ra_sample - ild_sample if np.isfinite(ra_sample) and np.isfinite(ild_sample) else float("nan"),
            "sample_unique_clone_count": float(sample_total),
            "RA_dict_size": float(ra_size),
            "ILD_dict_size": float(ild_size),
        })
    return pd.DataFrame(rows)


def _base_rows(inputs: FoldFeatureInputs, sample_ids: Sequence[str]) -> pd.DataFrame:
    ids = tuple(map(str, sample_ids))
    metadata = inputs.metadata.set_index("sample_id").loc[list(ids)].reset_index()
    static = inputs.static_values.set_index("sample_id").loc[list(ids)].reset_index()
    keep_meta = ["sample_id", "cohort", *inputs.clinical_numeric, *inputs.clinical_categorical]
    keep_meta = list(dict.fromkeys(keep_meta))
    out = metadata.loc[:, keep_meta].merge(static, on="sample_id", how="left", validate="one_to_one")
    return out


def _ids_hash(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode("utf-8")).hexdigest()


def build_fold_features(
    inputs: FoldFeatureInputs,
    spec: UnifiedExperimentSpec,
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    *,
    context: str,
) -> FoldFeatureBuild:
    train_ids = tuple(map(str, train_ids))
    valid_ids = tuple(map(str, valid_ids))
    if set(train_ids) & set(valid_ids):
        raise FoldFeatureError("training and validation IDs overlap")
    available = set(inputs.sample_ids)
    unknown = sorted((set(train_ids) | set(valid_ids)) - available)
    if unknown:
        raise FoldFeatureError(f"fold IDs absent from feature inputs: {unknown[:20]}")
    train = _base_rows(inputs, train_ids)
    valid = _base_rows(inputs, valid_ids)
    numeric = list(inputs.clinical_numeric) + list(inputs.static_feature_names)
    categorical = list(inputs.clinical_categorical)
    audit_rows: List[Dict[str, Any]] = []

    vocabulary = pd.DataFrame(columns=["selection_rank", "kmer", "keep"])
    if spec.base.features.kmer.enabled:
        vocabulary = rank_training_kmers(train_ids, inputs, spec)
        selected = vocabulary.loc[vocabulary["keep"].astype(bool), "kmer"].astype(str).tolist()
        rep = spec.base.features.kmer.representation
        train_kmer = _kmer_feature_frame(train_ids, selected, inputs, rep)
        valid_kmer = _kmer_feature_frame(valid_ids, selected, inputs, rep)
        train = train.merge(train_kmer, on="sample_id", how="left", validate="one_to_one")
        valid = valid.merge(valid_kmer, on="sample_id", how="left", validate="one_to_one")
        kcols = [c for c in train_kmer.columns if c != "sample_id"]
        numeric.extend(kcols)
        vocab_hash = hashlib.sha256("\n".join(selected).encode("utf-8")).hexdigest()
        audit_rows.append({
            "context": context,
            "feature_family": "3mer",
            "fit_policy": "fit_vocabulary_on_current_training_only",
            "fit_sample_count": len(train_ids),
            "validation_sample_count": len(valid_ids),
            "fit_sample_hash": _ids_hash(train_ids),
            "validation_sample_hash": _ids_hash(valid_ids),
            "feature_count": len(kcols),
            "reference_size": "",
            "reference_hash": vocab_hash,
        })

    espec = spec.base.features.enriched_dictionary
    if espec.enabled:
        train_scores = exact_loo_enriched_scores(
            inputs.metadata,
            train_ids,
            inputs.enrich_repertoires,
            threshold_pct=espec.threshold_pct,
            delta_pct=espec.delta_pct,
        )
        train_meta = inputs.metadata.set_index("sample_id").loc[list(train_ids)].reset_index()
        train_meta_for_fit = train_meta.rename(columns={"sample_id": "libraryid"})
        reference = fit_enriched_dictionary(
            train_meta_for_fit,
            inputs.enrich_repertoires,
            source_id=inputs.source.source_id,
            threshold_pct=espec.threshold_pct,
            delta_pct=espec.delta_pct,
            id_col="libraryid",
            cohort_col="cohort",
        )
        valid_meta = inputs.metadata.set_index("sample_id").loc[list(valid_ids)].reset_index()
        valid_scores = transform_repertoires(
            valid_meta.rename(columns={"sample_id": "libraryid"}),
            inputs.enrich_repertoires,
            reference,
            id_col="libraryid",
        )
        ecols = list(espec.features)
        missing = [c for c in ecols if c not in ALL_SCORE_COLUMNS]
        if missing:
            raise FoldFeatureError(f"unsupported enriched score columns: {missing}")
        train = train.merge(train_scores[["sample_id", *ecols]], on="sample_id", how="left", validate="one_to_one")
        valid = valid.merge(valid_scores[["sample_id", *ecols]], on="sample_id", how="left", validate="one_to_one")
        numeric.extend(ecols)
        audit_rows.append({
            "context": context,
            "feature_family": "enriched_dictionary",
            "fit_policy": "training_exact_LOO;validation_training_reference_only",
            "fit_sample_count": len(train_ids),
            "validation_sample_count": len(valid_ids),
            "fit_sample_hash": _ids_hash(train_ids),
            "validation_sample_hash": _ids_hash(valid_ids),
            "feature_count": len(ecols),
            "reference_size": f"RA={reference.ra_size};ILD={reference.ild_size}",
            "reference_hash": hashlib.sha256(
                ("\n".join(reference.ra_clones) + "\n--ILD--\n" + "\n".join(reference.ild_clones)).encode("utf-8")
            ).hexdigest(),
        })

    required = [*numeric, *categorical]
    if len(required) != len(set(required)):
        raise FoldFeatureError("assembled predictor names contain duplicates")
    for label, frame in (("train", train), ("valid", valid)):
        missing_cols = [c for c in required if c not in frame.columns]
        if missing_cols:
            raise FoldFeatureError(f"{label} assembled frame missing predictors: {missing_cols[:20]}")
        if frame[required].isna().any().any():
            bad = frame[required].isna().sum()
            raise FoldFeatureError(f"{label} assembled predictors contain missing values: {bad[bad>0].to_dict()}")
    audit_rows.insert(0, {
        "context": context,
        "feature_family": "static_and_clinical",
        "fit_policy": "label_independent_static_or_metadata",
        "fit_sample_count": len(train_ids),
        "validation_sample_count": len(valid_ids),
        "fit_sample_hash": _ids_hash(train_ids),
        "validation_sample_hash": _ids_hash(valid_ids),
        "feature_count": len(inputs.static_feature_names) + len(inputs.clinical_numeric) + len(inputs.clinical_categorical),
        "reference_size": "",
        "reference_hash": "",
    })
    return FoldFeatureBuild(
        train=train,
        valid=valid,
        numeric_columns=tuple(numeric),
        categorical_columns=tuple(categorical),
        kmer_vocabulary=vocabulary,
        audit=pd.DataFrame(audit_rows),
    )
