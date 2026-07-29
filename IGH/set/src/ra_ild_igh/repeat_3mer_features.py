#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Outer-repeat-specific IGH 3-mer feature construction.

This module implements the pragmatic feature-ablation policy used by the
repeated-holdout framework:

* one 3-mer vocabulary is fitted from the current outer-repeat training set;
* the fitted vocabulary is frozen and applied to every sample in that repeat;
* the same ranked 3-mer sequence list is shared by weighted and unweighted
  representations;
* Top 50/100/200/500 groups are nested prefixes of one deterministic ranking;
* no cohort label or other clinical metadata is used in vocabulary selection.

The ranking reproduces the original IGH step-02 rule:

1. minimum training-sample count;
2. minimum training-sample prevalence;
3. reject only when both weighted and unweighted variances are below threshold;
4. rank by prevalence descending;
5. break prevalence ties by weighted + unweighted variance descending;
6. break remaining ties by amino-acid 3-mer lexicographic order.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


AA_SET = frozenset("ACDEFGHIKLMNPQRSTVWY")
UNWEIGHTED_PREFIX = "unweighted_3mer_"
WEIGHTED_PREFIX = "weighted_3mer_"
DEFAULT_CORE_GROUP = "core83"
DEFAULT_LEGACY_GROUP = "static_igh_candidate_predictors"


class Repeat3merError(ValueError):
    """Raised when repeat-specific 3-mer construction is invalid."""


@dataclass(frozen=True)
class Repeat3merOptions:
    aa_manifest: Path
    sequence_column: str = "cdr3_aa"
    min_sample_count: int = 5
    min_prevalence: float = 0.05
    min_unweighted_variance: float = 1.0e-12
    min_weighted_variance: float = 1.0e-12
    max_kmers: int = 500
    top_k_values: Tuple[int, ...] = (50, 100, 200, 500)
    expected_core_feature_count: int = 83
    core_group_name: str = DEFAULT_CORE_GROUP
    legacy_group_name: str = DEFAULT_LEGACY_GROUP

    def __post_init__(self) -> None:
        if not self.aa_manifest.is_file():
            raise FileNotFoundError(f"AA clone-table manifest not found: {self.aa_manifest}")
        if not self.sequence_column.strip():
            raise Repeat3merError("sequence_column must be non-empty")
        if self.min_sample_count < 1:
            raise Repeat3merError("min_sample_count must be >= 1")
        if not 0 <= self.min_prevalence <= 1:
            raise Repeat3merError("min_prevalence must be in [0, 1]")
        if self.min_unweighted_variance < 0 or self.min_weighted_variance < 0:
            raise Repeat3merError("variance thresholds must be >= 0")
        if self.max_kmers < 1:
            raise Repeat3merError("max_kmers must be >= 1")
        if not self.top_k_values:
            raise Repeat3merError("top_k_values must not be empty")
        if tuple(sorted(set(self.top_k_values))) != self.top_k_values:
            raise Repeat3merError("top_k_values must be unique and strictly increasing")
        if any(value < 1 or value > self.max_kmers for value in self.top_k_values):
            raise Repeat3merError("every top_k value must be in [1, max_kmers]")
        if self.max_kmers not in self.top_k_values:
            raise Repeat3merError("top_k_values must include max_kmers")
        if self.expected_core_feature_count < 1:
            raise Repeat3merError("expected_core_feature_count must be >= 1")
        for name in (self.core_group_name, self.legacy_group_name):
            if not name.strip():
                raise Repeat3merError("feature-group names must be non-empty")


@dataclass(frozen=True)
class Repeat3merBuild:
    augmented_base: pd.DataFrame
    vocabulary: pd.DataFrame
    feature_matrix: pd.DataFrame
    feature_group_manifest: pd.DataFrame
    static_feature_groups: Mapping[str, Tuple[str, ...]]
    audit: Mapping[str, Any]


def _as_string_list(value: Any, label: str) -> Tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise Repeat3merError(f"{label} must be a list")
    result: List[str] = []
    for item in value:
        if not isinstance(item, (str, int)) or not str(item).strip():
            raise Repeat3merError(f"{label} contains an invalid value: {item!r}")
        result.append(str(item).strip())
    if len(result) != len(set(result)):
        raise Repeat3merError(f"{label} contains duplicates")
    return tuple(result)


def options_from_config(config, *, repository_root: Optional[Path] = None) -> Optional[Repeat3merOptions]:
    """Parse optional ``repeat_3mer_features`` configuration.

    Returns ``None`` for legacy schemes that do not enable repeat-specific
    3-mers, preserving backward compatibility.
    """

    raw = config.raw.get("repeat_3mer_features")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise Repeat3merError("repeat_3mer_features must be a mapping")
    if raw.get("enabled", False) is not True:
        return None
    if str(raw.get("selection_scope", "outer_repeat_train_only")) != "outer_repeat_train_only":
        raise Repeat3merError(
            "repeat_3mer_features.selection_scope must be outer_repeat_train_only"
        )
    if str(raw.get("ranking_rule", "prevalence_then_total_variance")) != (
        "prevalence_then_total_variance"
    ):
        raise Repeat3merError(
            "repeat_3mer_features.ranking_rule must be prevalence_then_total_variance"
        )

    path_value = raw.get("aa_manifest")
    if not isinstance(path_value, str) or not path_value.strip():
        raise Repeat3merError("repeat_3mer_features.aa_manifest must be a path string")
    root = Path(repository_root or config.repository_root).expanduser().resolve()
    candidate = Path(path_value).expanduser()
    aa_manifest = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

    top_values = tuple(int(value) for value in raw.get("top_k_values", [50, 100, 200, 500]))
    return Repeat3merOptions(
        aa_manifest=aa_manifest,
        sequence_column=str(raw.get("sequence_column", "cdr3_aa")),
        min_sample_count=int(raw.get("min_sample_count", 5)),
        min_prevalence=float(raw.get("min_prevalence", 0.05)),
        min_unweighted_variance=float(raw.get("min_unweighted_variance", 1.0e-12)),
        min_weighted_variance=float(raw.get("min_weighted_variance", 1.0e-12)),
        max_kmers=int(raw.get("max_kmers", 500)),
        top_k_values=top_values,
        expected_core_feature_count=int(raw.get("expected_core_feature_count", 83)),
        core_group_name=str(raw.get("core_group_name", DEFAULT_CORE_GROUP)),
        legacy_group_name=str(raw.get("legacy_group_name", DEFAULT_LEGACY_GROUP)),
    )


def is_3mer_feature(name: str) -> bool:
    value = str(name)
    return value.startswith(UNWEIGHTED_PREFIX) or value.startswith(WEIGHTED_PREFIX)


def split_legacy_static_features(
    static_features: Sequence[str],
    *,
    expected_core_count: int = 83,
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    """Split legacy 1083 static features into core, unweighted and weighted."""

    ordered = tuple(map(str, static_features))
    core = tuple(name for name in ordered if not is_3mer_feature(name))
    unweighted = tuple(name for name in ordered if name.startswith(UNWEIGHTED_PREFIX))
    weighted = tuple(name for name in ordered if name.startswith(WEIGHTED_PREFIX))
    if len(core) != int(expected_core_count):
        raise Repeat3merError(
            f"Expected {expected_core_count} core static features; observed {len(core)}"
        )
    if not unweighted or not weighted:
        raise Repeat3merError("Legacy static feature list does not contain both 3-mer representations")
    unweighted_kmers = {name[len(UNWEIGHTED_PREFIX) :] for name in unweighted}
    weighted_kmers = {name[len(WEIGHTED_PREFIX) :] for name in weighted}
    if unweighted_kmers != weighted_kmers:
        missing_u = sorted(weighted_kmers - unweighted_kmers)[:20]
        missing_w = sorted(unweighted_kmers - weighted_kmers)[:20]
        raise Repeat3merError(
            "Legacy weighted/unweighted 3-mer vocabularies differ; "
            f"missing_unweighted={missing_u}, missing_weighted={missing_w}"
        )
    return core, unweighted, weighted


def unique_3mers(sequence: str) -> Set[str]:
    seq = str(sequence).strip().upper()
    if len(seq) < 3 or any(aa not in AA_SET for aa in seq):
        return set()
    return {seq[index : index + 3] for index in range(len(seq) - 2)}


def _read_aa_manifest(path: Path, expected_sample_ids: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"sample_id", "path", "frequency_column"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise Repeat3merError(f"AA manifest is missing columns: {missing}")
    frame = frame.loc[:, ["sample_id", "path", "frequency_column"]].copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    if frame["sample_id"].duplicated().any():
        duplicates = frame.loc[
            frame["sample_id"].duplicated(keep=False), "sample_id"
        ].unique().tolist()
        raise Repeat3merError(f"AA manifest duplicates sample IDs: {duplicates[:20]}")
    expected = tuple(map(str, expected_sample_ids))
    observed = set(frame["sample_id"])
    missing_ids = sorted(set(expected) - observed)
    extra_ids = sorted(observed - set(expected))
    if missing_ids or extra_ids:
        raise Repeat3merError(
            "AA manifest sample coverage mismatch; "
            f"missing={missing_ids[:20]}, extra={extra_ids[:20]}"
        )
    frame = frame.set_index("sample_id").loc[list(expected)].reset_index()
    resolved_paths = []
    for value in frame["path"]:
        candidate = Path(str(value)).expanduser()
        if not candidate.is_absolute():
            candidate = (path.parent / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"AA clone table not found: {candidate}")
        resolved_paths.append(str(candidate))
    frame["path"] = resolved_paths
    return frame


def _load_sample_kmers(
    path: Path,
    *,
    sequence_column: str,
    frequency_column: str,
    retained_kmers: Optional[Set[str]] = None,
) -> Tuple[Dict[str, float], Dict[str, float], int]:
    try:
        frame = pd.read_csv(path, usecols=[sequence_column, frequency_column])
    except ValueError as exc:
        raise Repeat3merError(
            f"AA clone table {path} lacks {sequence_column!r} or {frequency_column!r}"
        ) from exc
    frame = frame.dropna(subset=[sequence_column, frequency_column]).copy()
    frame[sequence_column] = frame[sequence_column].astype(str).str.strip().str.upper()
    frame[frequency_column] = pd.to_numeric(frame[frequency_column], errors="coerce")
    frame = frame.dropna(subset=[frequency_column])
    valid_sequence = frame[sequence_column].map(
        lambda value: len(value) >= 3 and all(aa in AA_SET for aa in value)
    )
    frame = frame.loc[valid_sequence].copy()
    if frame.empty:
        raise Repeat3merError(f"AA clone table has no valid CDR3-AA rows: {path}")
    values = frame[frequency_column].to_numpy(float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise Repeat3merError(f"AA clone table has invalid frequencies: {path}")
    frame = (
        frame.groupby(sequence_column, sort=False, as_index=False)[frequency_column]
        .sum()
    )
    total = float(frame[frequency_column].sum())
    if total <= 0:
        raise Repeat3merError(f"AA clone table has zero total frequency: {path}")
    frame[frequency_column] = frame[frequency_column].astype(float) / total
    aa_clone_number = int(len(frame))

    clone_counts: Dict[str, int] = {}
    weight_sums: Dict[str, float] = {}
    retained = retained_kmers
    for sequence, clone_weight in frame[[sequence_column, frequency_column]].itertuples(
        index=False, name=None
    ):
        kmers = unique_3mers(sequence)
        if retained is not None:
            kmers.intersection_update(retained)
        for kmer in kmers:
            clone_counts[kmer] = clone_counts.get(kmer, 0) + 1
            weight_sums[kmer] = weight_sums.get(kmer, 0.0) + float(clone_weight)
    unweighted = {
        kmer: count / aa_clone_number for kmer, count in clone_counts.items()
    }
    return unweighted, weight_sums, aa_clone_number


def rank_training_kmers(
    train_sample_ids: Sequence[str],
    unweighted_by_sample: Mapping[str, Mapping[str, float]],
    weighted_by_sample: Mapping[str, Mapping[str, float]],
    options: Repeat3merOptions,
) -> pd.DataFrame:
    """Create a deterministic full ranking and retain at most ``max_kmers``."""

    sample_ids = tuple(map(str, train_sample_ids))
    if len(sample_ids) != len(set(sample_ids)) or not sample_ids:
        raise Repeat3merError("Training sample IDs must be non-empty and unique")
    all_kmers: Set[str] = set()
    for sample_id in sample_ids:
        all_kmers.update(unweighted_by_sample[sample_id])
        all_kmers.update(weighted_by_sample[sample_id])
    n_samples = len(sample_ids)
    rows: List[Dict[str, Any]] = []
    for kmer in sorted(all_kmers):
        unweighted_values = np.array(
            [unweighted_by_sample[sample_id].get(kmer, 0.0) for sample_id in sample_ids],
            dtype=float,
        )
        weighted_values = np.array(
            [weighted_by_sample[sample_id].get(kmer, 0.0) for sample_id in sample_ids],
            dtype=float,
        )
        sample_count = int(np.count_nonzero(unweighted_values > 0))
        prevalence = sample_count / n_samples
        var_u = float(np.var(unweighted_values, ddof=0))
        var_w = float(np.var(weighted_values, ddof=0))
        reasons: List[str] = []
        if sample_count < options.min_sample_count:
            reasons.append("low_sample_count")
        if prevalence < options.min_prevalence:
            reasons.append("low_prevalence")
        if (
            var_u < options.min_unweighted_variance
            and var_w < options.min_weighted_variance
        ):
            reasons.append("low_variance_both")
        rows.append(
            {
                "kmer": kmer,
                "training_sample_count": sample_count,
                "training_sample_prevalence": prevalence,
                "mean_unweighted_value": float(np.mean(unweighted_values)),
                "variance_unweighted_value": var_u,
                "mean_weighted_value": float(np.mean(weighted_values)),
                "variance_weighted_value": var_w,
                "ranking_total_variance": var_u + var_w,
                "passes_filters": not reasons,
                "filter_reason": "retained_candidate" if not reasons else ";".join(reasons),
            }
        )
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        raise Repeat3merError("No valid amino-acid 3-mers were found in outer training")
    candidates = ranking.loc[ranking["passes_filters"]].copy()
    candidates = candidates.sort_values(
        ["training_sample_prevalence", "ranking_total_variance", "kmer"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(candidates) < options.max_kmers:
        raise Repeat3merError(
            f"Only {len(candidates)} 3-mers passed filters; max_kmers={options.max_kmers}"
        )
    selected = candidates.head(options.max_kmers).copy()
    selected.insert(0, "selection_rank", np.arange(1, len(selected) + 1))
    selected["keep"] = True

    excluded_candidates = candidates.iloc[options.max_kmers :].copy()
    if not excluded_candidates.empty:
        excluded_candidates.insert(
            0,
            "selection_rank",
            np.arange(options.max_kmers + 1, len(candidates) + 1),
        )
        excluded_candidates["keep"] = False
        excluded_candidates["filter_reason"] = "excluded_by_max_kmers"
    failed = ranking.loc[~ranking["passes_filters"]].copy()
    failed = failed.sort_values(
        ["training_sample_prevalence", "ranking_total_variance", "kmer"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    failed.insert(0, "selection_rank", np.nan)
    failed["keep"] = False

    out = pd.concat([selected, excluded_candidates, failed], ignore_index=True, sort=False)
    out.insert(0, "vocabulary_row", np.arange(1, len(out) + 1))
    return out


def _feature_groups(
    core_features: Sequence[str],
    selected_kmers: Sequence[str],
    options: Repeat3merOptions,
) -> Mapping[str, Tuple[str, ...]]:
    groups: Dict[str, Tuple[str, ...]] = {
        options.core_group_name: tuple(map(str, core_features)),
    }
    selected = tuple(map(str, selected_kmers))
    for k in options.top_k_values:
        kmers = selected[:k]
        unweighted = tuple(f"{UNWEIGHTED_PREFIX}{kmer}" for kmer in kmers)
        weighted = tuple(f"{WEIGHTED_PREFIX}{kmer}" for kmer in kmers)
        groups[f"repeat_3mer_unweighted_top{k}"] = unweighted
        groups[f"repeat_3mer_weighted_top{k}"] = weighted
        groups[f"repeat_3mer_both_top{k}"] = unweighted + weighted
    return MappingProxyType(groups)


def _group_manifest(groups: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for group_order, (group, features) in enumerate(groups.items(), start=1):
        for feature_order, feature in enumerate(features, start=1):
            rows.append(
                {
                    "group_order": group_order,
                    "feature_group": group,
                    "feature_order": feature_order,
                    "feature_name": str(feature),
                }
            )
    return pd.DataFrame(rows)


def _sha256_strings(values: Iterable[str]) -> str:
    payload = "\n".join(map(str, values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_repeat_3mer_features(
    base: pd.DataFrame,
    *,
    legacy_static_features: Sequence[str],
    outer_train_ids: Sequence[str],
    outer_repeat: int,
    options: Repeat3merOptions,
) -> Repeat3merBuild:
    """Fit one outer-repeat vocabulary and augment the complete cohort matrix."""

    if "sample_id" not in base.columns:
        raise Repeat3merError("Base matrix is missing sample_id")
    matrix = base.copy()
    matrix["sample_id"] = matrix["sample_id"].astype(str)
    if matrix["sample_id"].duplicated().any():
        raise Repeat3merError("Base matrix contains duplicate sample IDs")
    sample_ids = tuple(matrix["sample_id"].tolist())
    train_ids = tuple(map(str, outer_train_ids))
    if len(train_ids) != len(set(train_ids)) or not train_ids:
        raise Repeat3merError("Outer training sample IDs must be non-empty and unique")
    unknown = sorted(set(train_ids) - set(sample_ids))
    if unknown:
        raise Repeat3merError(f"Outer training IDs are absent from base matrix: {unknown[:20]}")

    core, old_unweighted, old_weighted = split_legacy_static_features(
        legacy_static_features,
        expected_core_count=options.expected_core_feature_count,
    )
    missing_core = sorted(set(core) - set(matrix.columns))
    if missing_core:
        raise Repeat3merError(f"Core static features are missing from base matrix: {missing_core[:20]}")
    old_3mer_columns = [
        name for name in (*old_unweighted, *old_weighted) if name in matrix.columns
    ]
    matrix = matrix.drop(columns=old_3mer_columns)

    aa_manifest = _read_aa_manifest(options.aa_manifest, sample_ids)
    manifest_lookup = aa_manifest.set_index("sample_id")
    train_unweighted: Dict[str, Mapping[str, float]] = {}
    train_weighted: Dict[str, Mapping[str, float]] = {}
    clone_numbers: Dict[str, int] = {}
    for sample_id in train_ids:
        row = manifest_lookup.loc[sample_id]
        unweighted, weighted, clone_number = _load_sample_kmers(
            Path(str(row["path"])).expanduser(),
            sequence_column=options.sequence_column,
            frequency_column=str(row["frequency_column"]),
        )
        train_unweighted[sample_id] = unweighted
        train_weighted[sample_id] = weighted
        clone_numbers[sample_id] = clone_number

    vocabulary = rank_training_kmers(
        train_ids,
        train_unweighted,
        train_weighted,
        options,
    )
    selected_kmers = tuple(
        vocabulary.loc[vocabulary["keep"].astype(bool), "kmer"]
        .astype(str)
        .head(options.max_kmers)
        .tolist()
    )
    if len(selected_kmers) != options.max_kmers:
        raise Repeat3merError(
            f"Expected {options.max_kmers} selected 3-mers; observed {len(selected_kmers)}"
        )
    retained_set = set(selected_kmers)

    rows: List[Dict[str, Any]] = []
    for sample_id in sample_ids:
        if sample_id in train_unweighted:
            unweighted = train_unweighted[sample_id]
            weighted = train_weighted[sample_id]
        else:
            row = manifest_lookup.loc[sample_id]
            unweighted, weighted, clone_number = _load_sample_kmers(
                Path(str(row["path"])).expanduser(),
                sequence_column=options.sequence_column,
                frequency_column=str(row["frequency_column"]),
                retained_kmers=retained_set,
            )
            clone_numbers[sample_id] = clone_number
        record: Dict[str, Any] = {"sample_id": sample_id}
        for kmer in selected_kmers:
            record[f"{UNWEIGHTED_PREFIX}{kmer}"] = float(unweighted.get(kmer, 0.0))
            record[f"{WEIGHTED_PREFIX}{kmer}"] = float(weighted.get(kmer, 0.0))
        rows.append(record)
    feature_matrix = pd.DataFrame(rows)
    if feature_matrix.isna().any().any():
        raise Repeat3merError("Repeat-specific 3-mer matrix contains missing values")
    expected_columns = 1 + 2 * options.max_kmers
    if feature_matrix.shape != (len(sample_ids), expected_columns):
        raise Repeat3merError(
            f"Unexpected 3-mer matrix shape {feature_matrix.shape}; "
            f"expected {(len(sample_ids), expected_columns)}"
        )
    augmented = matrix.merge(feature_matrix, on="sample_id", how="left", validate="one_to_one")
    if augmented["sample_id"].tolist() != list(sample_ids):
        raise Repeat3merError("3-mer augmentation changed base sample order")

    groups = dict(_feature_groups(core, selected_kmers, options))
    # Backward-compatible legacy alias now means this repeat's core + both Top max.
    groups[options.legacy_group_name] = (
        tuple(core) + groups[f"repeat_3mer_both_top{options.max_kmers}"]
    )
    groups_proxy = MappingProxyType(groups)
    group_manifest = _group_manifest(groups_proxy)
    audit = MappingProxyType(
        {
            "status": "BUILT",
            "selection_scope": "outer_repeat_train_only",
            "outer_repeat": int(outer_repeat),
            "training_sample_count": len(train_ids),
            "full_sample_count": len(sample_ids),
            "core_feature_count": len(core),
            "selected_kmer_count": len(selected_kmers),
            "generated_3mer_feature_count": 2 * len(selected_kmers),
            "top_k_values": list(options.top_k_values),
            "ranking_rule": (
                "training_prevalence_desc_then_unweighted_plus_weighted_"
                "variance_desc_then_kmer_asc"
            ),
            "min_sample_count": options.min_sample_count,
            "min_prevalence": options.min_prevalence,
            "min_unweighted_variance": options.min_unweighted_variance,
            "min_weighted_variance": options.min_weighted_variance,
            "outer_train_ids_sha256": _sha256_strings(train_ids),
            "selected_kmers_sha256": _sha256_strings(selected_kmers),
            "aa_manifest": str(options.aa_manifest),
            "clone_number_min": int(min(clone_numbers.values())),
            "clone_number_max": int(max(clone_numbers.values())),
        }
    )
    return Repeat3merBuild(
        augmented_base=augmented,
        vocabulary=vocabulary,
        feature_matrix=feature_matrix,
        feature_group_manifest=group_manifest,
        static_feature_groups=groups_proxy,
        audit=audit,
    )


def write_repeat_3mer_outputs(
    build: Repeat3merBuild,
    *,
    vocabulary_path: Path,
    matrix_path: Path,
    group_manifest_path: Path,
    audit_path: Path,
) -> None:
    vocabulary_path.parent.mkdir(parents=True, exist_ok=True)
    build.vocabulary.to_csv(vocabulary_path, index=False)
    build.feature_matrix.to_csv(matrix_path, index=False, compression="gzip")
    build.feature_group_manifest.to_csv(group_manifest_path, index=False)
    audit_path.write_text(
        json.dumps(dict(build.audit), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
