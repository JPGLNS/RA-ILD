#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic metadata-only repeated-holdout generation for RA/RA-ILD IGH.

The splitting unit is the patient. Candidate assignments are ranked only by
prespecified metadata-balance criteria. No repertoire feature, model output,
AUC, independent-test result, or other downstream result is read.

IGH contains one singleton ``cohort × batch`` stratum. The optional
``singleton_strata_policy: fixed_train`` contract removes explicitly audited
singleton patients from the randomised pool and places them in training for
every repeat. This preserves exact stratification among splittable strata and
prevents the singleton from entering a holdout by an arbitrary rule.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .paths import resolve_project_path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


SPLIT_SET_VERSION = "1.1"
SPLIT_SET_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")
DEFAULT_SPLIT_ROOT = Path("IGH/set/split_sets")
SINGLETON_POLICIES = frozenset({"error", "fixed_train"})


class SplitGenerationError(ValueError):
    """Raised when split input, configuration, or assignments are unsafe."""


@dataclass(frozen=True)
class RepeatedHoldoutSpec:
    source_path: Path
    repository_root: Path
    split_set_id: str
    description: str
    output_root: Path
    metadata_files: Tuple[Path, ...]
    sample_id_column: str
    patient_id_column: str
    require_one_sample_per_patient: bool
    expected_samples: int
    expected_patients: int
    label_column: str
    repeats: int
    train_size: int
    holdout_size: int
    exact_strata: Tuple[str, ...]
    singleton_strata_policy: str
    expected_fixed_train_sample_ids: Tuple[str, ...]
    balance_categorical: Tuple[str, ...]
    balance_numeric: Tuple[str, ...]
    candidates_per_repeat: int
    base_seed: int
    diversity_weight: float
    max_pairwise_holdout_jaccard: float
    train_role: str
    holdout_role: str
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class SplitPool:
    eligible_indices: np.ndarray
    fixed_train_indices: np.ndarray
    eligible_strata: pd.Series
    singleton_strata: Mapping[str, int]


@dataclass(frozen=True)
class ChosenSplit:
    repeat_index: int
    split_id: str
    candidate_seed: int
    train_indices: np.ndarray
    holdout_indices: np.ndarray
    fixed_train_indices: np.ndarray
    base_balance_score: float
    mean_previous_jaccard: float
    max_previous_jaccard: float
    total_score: float
    eligible_candidates: int


@dataclass(frozen=True)
class SplitGenerationResult:
    metadata: pd.DataFrame
    assignments: pd.DataFrame
    split_counts: pd.DataFrame
    balance_audit: pd.DataFrame
    selected_candidates: pd.DataFrame
    pairwise_overlap: pd.DataFrame
    sample_holdout_frequency: pd.DataFrame
    fixed_train_audit: pd.DataFrame
    chosen_splits: Tuple[ChosenSplit, ...]


def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required. Install with: mamba install -c conda-forge pyyaml"
        )


def _mapping(parent: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise SplitGenerationError(f"{context}.{key} must be a mapping")
    return value


def _string(parent: Mapping[str, Any], key: str, context: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SplitGenerationError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _integer(
    parent: Mapping[str, Any], key: str, context: str, minimum: int = 0
) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SplitGenerationError(f"{context}.{key} must be an integer >= {minimum}")
    return int(value)


def _number(
    parent: Mapping[str, Any],
    key: str,
    context: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SplitGenerationError(f"{context}.{key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise SplitGenerationError(f"{context}.{key} must be finite")
    if minimum is not None and number < minimum:
        raise SplitGenerationError(f"{context}.{key} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise SplitGenerationError(f"{context}.{key} must be <= {maximum}")
    return number


def _string_list(
    value: Any, context: str, *, allow_empty: bool = True
) -> Tuple[str, ...]:
    if value is None and allow_empty:
        return tuple()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SplitGenerationError(f"{context} must be a list of strings")
    out: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise SplitGenerationError(f"{context} must contain non-empty strings")
        out.append(item.strip())
    if not allow_empty and not out:
        raise SplitGenerationError(f"{context} must not be empty")
    if len(out) != len(set(out)):
        raise SplitGenerationError(f"{context} contains duplicates")
    return tuple(out)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _repo_relative(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError as exc:
        raise SplitGenerationError(
            f"Path must remain inside repository root: {path}"
        ) from exc


def load_repeated_holdout_spec(
    path: Path,
    *,
    repository_root: Path,
) -> RepeatedHoldoutSpec:
    """Load and validate a repeated-holdout split-set YAML specification."""

    _require_yaml()
    source_path = Path(path).expanduser().resolve()
    repository_root = Path(repository_root).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Split-set specification not found: {source_path}")
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise SplitGenerationError("Split-set YAML root must be a mapping")
    if str(raw.get("split_set_version")) != SPLIT_SET_VERSION:
        raise SplitGenerationError(
            f"split_set_version must be '{SPLIT_SET_VERSION}'"
        )

    split_set = _mapping(raw, "split_set", "root")
    split_set_id = _string(split_set, "id", "split_set")
    if not SPLIT_SET_ID_PATTERN.fullmatch(split_set_id):
        raise SplitGenerationError(
            "split_set.id must be 3-80 lowercase characters using only "
            "a-z, 0-9, '.', '_' or '-'"
        )
    description = str(split_set.get("description", "")).strip()
    output_value = split_set.get(
        "output_root", str(DEFAULT_SPLIT_ROOT / split_set_id)
    )
    if not isinstance(output_value, str) or not output_value.strip():
        raise SplitGenerationError("split_set.output_root must be a non-empty path")
    output_path = Path(output_value.strip())
    if output_path.is_absolute():
        raise SplitGenerationError(
            "split_set.output_root must be repository-relative"
        )
    output_root = resolve_project_path(output_path, repository_root)
    allowed_root = resolve_project_path(DEFAULT_SPLIT_ROOT, repository_root)
    if not _is_within(output_root, allowed_root):
        raise SplitGenerationError(
            f"split_set.output_root must be inside {DEFAULT_SPLIT_ROOT.as_posix()}"
        )
    if output_root.name != split_set_id:
        raise SplitGenerationError(
            "split_set.output_root must end with the exact split_set.id"
        )

    source = _mapping(raw, "source", "root")
    metadata_values = source.get("metadata_files")
    if not isinstance(metadata_values, Sequence) or isinstance(
        metadata_values, (str, bytes)
    ):
        raise SplitGenerationError("source.metadata_files must be a non-empty list")
    metadata_files: List[Path] = []
    for index, value in enumerate(metadata_values):
        if not isinstance(value, str) or not value.strip():
            raise SplitGenerationError(
                f"source.metadata_files[{index}] must be a non-empty "
                "repository-relative path"
            )
        candidate = Path(value.strip())
        if candidate.is_absolute():
            raise SplitGenerationError(
                "source.metadata_files entries must be repository-relative"
            )
        metadata_files.append(resolve_project_path(candidate, repository_root))
    if not metadata_files:
        raise SplitGenerationError("source.metadata_files must not be empty")

    sample_id_column = _string(source, "sample_id_column", "source")
    patient_id_column = _string(source, "patient_id_column", "source")
    require_one = source.get("require_one_sample_per_patient", True)
    if not isinstance(require_one, bool):
        raise SplitGenerationError(
            "source.require_one_sample_per_patient must be boolean"
        )
    expected_samples = _integer(source, "expected_samples", "source", 1)
    expected_patients = _integer(source, "expected_patients", "source", 1)

    split = _mapping(raw, "split", "root")
    if _string(split, "strategy", "split") != "repeated_holdout":
        raise SplitGenerationError("split.strategy must be repeated_holdout")
    label_column = _string(split, "label_column", "split")
    repeats = _integer(split, "repeats", "split", 1)
    train_size = _integer(split, "train_size", "split", 1)
    holdout_size = _integer(split, "holdout_size", "split", 1)
    if train_size + holdout_size != expected_patients:
        raise SplitGenerationError(
            "split.train_size + split.holdout_size must equal "
            "source.expected_patients"
        )
    exact_strata = _string_list(
        split.get("exact_strata", []),
        "split.exact_strata",
        allow_empty=False,
    )
    if label_column not in exact_strata:
        raise SplitGenerationError(
            "split.label_column must be included in split.exact_strata"
        )

    singleton_policy = str(
        split.get("singleton_strata_policy", "error")
    ).strip()
    if singleton_policy not in SINGLETON_POLICIES:
        raise SplitGenerationError(
            "split.singleton_strata_policy must be one of: "
            + ", ".join(sorted(SINGLETON_POLICIES))
        )
    expected_fixed = _string_list(
        split.get("expected_fixed_train_sample_ids", []),
        "split.expected_fixed_train_sample_ids",
    )
    if singleton_policy == "error" and expected_fixed:
        raise SplitGenerationError(
            "expected_fixed_train_sample_ids must be empty when "
            "singleton_strata_policy=error"
        )

    balance_categorical = _string_list(
        split.get("balance_categorical", []), "split.balance_categorical"
    )
    balance_numeric = _string_list(
        split.get("balance_numeric", []), "split.balance_numeric"
    )
    candidates = _integer(split, "candidates_per_repeat", "split", 1)
    base_seed = _integer(split, "base_seed", "split", 0)
    diversity_weight = _number(split, "diversity_weight", "split", 0.0)
    max_jaccard = _number(
        split,
        "max_pairwise_holdout_jaccard",
        "split",
        0.0,
        1.0,
    )
    if max_jaccard >= 1.0:
        raise SplitGenerationError(
            "split.max_pairwise_holdout_jaccard must be < 1"
        )
    roles = _mapping(split, "role_names", "split")
    train_role = _string(roles, "train", "split.role_names")
    holdout_role = _string(roles, "holdout", "split.role_names")
    if train_role == holdout_role:
        raise SplitGenerationError(
            "split.role_names.train and holdout must differ"
        )

    all_balance = (*exact_strata, *balance_categorical, *balance_numeric)
    if len(all_balance) != len(set(all_balance)):
        duplicates = sorted(
            {name for name in all_balance if all_balance.count(name) > 1}
        )
        raise SplitGenerationError(
            "Stratification/balance variables must not be repeated across "
            f"lists: {duplicates}"
        )

    return RepeatedHoldoutSpec(
        source_path=source_path,
        repository_root=repository_root,
        split_set_id=split_set_id,
        description=description,
        output_root=output_root,
        metadata_files=tuple(metadata_files),
        sample_id_column=sample_id_column,
        patient_id_column=patient_id_column,
        require_one_sample_per_patient=require_one,
        expected_samples=expected_samples,
        expected_patients=expected_patients,
        label_column=label_column,
        repeats=repeats,
        train_size=train_size,
        holdout_size=holdout_size,
        exact_strata=exact_strata,
        singleton_strata_policy=singleton_policy,
        expected_fixed_train_sample_ids=expected_fixed,
        balance_categorical=balance_categorical,
        balance_numeric=balance_numeric,
        candidates_per_repeat=candidates,
        base_seed=base_seed,
        diversity_weight=diversity_weight,
        max_pairwise_holdout_jaccard=max_jaccard,
        train_role=train_role,
        holdout_role=holdout_role,
        raw=raw,
    )


def _read_metadata_file(path: Path, index: int) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    frame = pd.read_csv(path)
    duplicated = frame.columns[frame.columns.duplicated()].astype(str).tolist()
    if duplicated:
        raise SplitGenerationError(
            f"Metadata file contains duplicated columns: {duplicated}"
        )
    frame = frame.copy()
    frame["_source_metadata_index"] = int(index)
    frame["_source_metadata_path"] = str(path)
    frame["_source_row"] = np.arange(1, len(frame) + 1, dtype=int)
    return frame


def load_combined_metadata(spec: RepeatedHoldoutSpec) -> pd.DataFrame:
    """Combine source metadata and validate the patient-level splitting unit."""

    frames = [
        _read_metadata_file(path, index)
        for index, path in enumerate(spec.metadata_files, 1)
    ]
    required = {
        spec.sample_id_column,
        spec.patient_id_column,
        spec.label_column,
        *spec.exact_strata,
        *spec.balance_categorical,
        *spec.balance_numeric,
    }
    for index, (path, frame) in enumerate(zip(spec.metadata_files, frames), 1):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise SplitGenerationError(
                f"Metadata file {index} is missing required columns {missing}: {path}"
            )
    metadata = pd.concat(frames, ignore_index=True, sort=False)

    for column in required:
        if metadata[column].isna().any():
            raise SplitGenerationError(
                f"Combined metadata column {column!r} contains missing values"
            )

    metadata = metadata.copy()
    metadata[spec.sample_id_column] = (
        metadata[spec.sample_id_column].astype(str).str.strip()
    )
    metadata[spec.patient_id_column] = (
        metadata[spec.patient_id_column].astype(str).str.strip()
    )
    if (metadata[spec.sample_id_column] == "").any():
        raise SplitGenerationError(
            "Combined metadata contains blank sample identifiers"
        )
    if (metadata[spec.patient_id_column] == "").any():
        raise SplitGenerationError(
            "Combined metadata contains blank patient identifiers"
        )

    duplicate_samples = metadata.loc[
        metadata[spec.sample_id_column].duplicated(keep=False),
        spec.sample_id_column,
    ].unique().tolist()
    if duplicate_samples:
        raise SplitGenerationError(
            f"Combined metadata contains duplicate sample IDs: {duplicate_samples[:20]}"
        )
    patient_counts = metadata.groupby(spec.patient_id_column, sort=False).size()
    if spec.require_one_sample_per_patient and (patient_counts != 1).any():
        bad = patient_counts[patient_counts != 1].head(20).to_dict()
        raise SplitGenerationError(
            "RA/RA-ILD IGH repeated holdout requires exactly one sample per "
            f"patient; violations: {bad}"
        )

    if len(metadata) != spec.expected_samples:
        raise SplitGenerationError(
            f"Expected {spec.expected_samples} samples, observed {len(metadata)}"
        )
    observed_patients = int(metadata[spec.patient_id_column].nunique())
    if observed_patients != spec.expected_patients:
        raise SplitGenerationError(
            f"Expected {spec.expected_patients} patients, observed {observed_patients}"
        )

    for column in (*spec.exact_strata, *spec.balance_categorical, spec.label_column):
        metadata[column] = metadata[column].astype(str).str.strip()
        if (metadata[column] == "").any():
            raise SplitGenerationError(
                f"Metadata column {column!r} contains blank values"
            )
    for column in spec.balance_numeric:
        converted = pd.to_numeric(metadata[column], errors="coerce")
        bad = converted.isna() | ~np.isfinite(converted.to_numpy(float))
        if bad.any():
            examples = metadata.loc[bad, column].astype(str).head(10).tolist()
            raise SplitGenerationError(
                f"Metadata numeric balance column {column!r} contains invalid "
                f"values: {examples}"
            )
        metadata[column] = converted.astype(float)

    metadata.insert(0, "sample_id", metadata[spec.sample_id_column].astype(str))
    metadata.insert(1, "patient_id", metadata[spec.patient_id_column].astype(str))
    metadata.insert(2, "_cohort_order", np.arange(len(metadata), dtype=int))
    return metadata


def _stratum_labels(
    metadata: pd.DataFrame, exact_strata: Sequence[str]
) -> pd.Series:
    values = metadata[list(exact_strata)].astype(str)
    return values.agg("\x1f".join, axis=1)


def _prepare_split_pool(
    metadata: pd.DataFrame,
    spec: RepeatedHoldoutSpec,
) -> SplitPool:
    labels = _stratum_labels(metadata, spec.exact_strata)
    counts = labels.value_counts(sort=False)
    singleton_labels = counts[counts == 1]
    singleton_mapping = {
        str(label): int(count) for label, count in singleton_labels.items()
    }

    if not singleton_labels.empty and spec.singleton_strata_policy == "error":
        raise SplitGenerationError(
            "Every exact stratum needs at least two patients for stratified "
            f"holdout; singleton strata: {singleton_mapping}"
        )

    singleton_mask = labels.isin(singleton_labels.index)
    fixed_indices = np.flatnonzero(singleton_mask.to_numpy())
    eligible_indices = np.flatnonzero((~singleton_mask).to_numpy())

    detected_fixed_ids = tuple(
        metadata.iloc[fixed_indices]["sample_id"].astype(str).tolist()
    )
    if spec.expected_fixed_train_sample_ids:
        if set(detected_fixed_ids) != set(spec.expected_fixed_train_sample_ids):
            raise SplitGenerationError(
                "Detected singleton fixed-train sample IDs differ from "
                "split.expected_fixed_train_sample_ids: "
                f"detected={sorted(detected_fixed_ids)}, "
                f"expected={sorted(spec.expected_fixed_train_sample_ids)}"
            )
    elif fixed_indices.size and spec.singleton_strata_policy == "fixed_train":
        raise SplitGenerationError(
            "singleton_strata_policy=fixed_train requires an explicit "
            "expected_fixed_train_sample_ids audit list"
        )

    effective_train_size = spec.train_size - int(fixed_indices.size)
    if effective_train_size < 1:
        raise SplitGenerationError(
            "Fixed-train singleton count leaves no randomized training patients"
        )
    if effective_train_size + spec.holdout_size != int(eligible_indices.size):
        raise SplitGenerationError(
            "After removing fixed-train singleton strata, effective train + "
            "holdout sizes do not equal the eligible patient count"
        )

    eligible_labels = labels.iloc[eligible_indices].reset_index(drop=True)
    eligible_counts = eligible_labels.value_counts(sort=False)
    remaining_small = eligible_counts[eligible_counts < 2]
    if not remaining_small.empty:
        raise SplitGenerationError(
            "Non-singleton eligible strata unexpectedly contain fewer than two "
            f"patients: {remaining_small.to_dict()}"
        )
    n_strata = int(len(eligible_counts))
    if n_strata > effective_train_size or n_strata > spec.holdout_size:
        raise SplitGenerationError(
            f"Number of eligible exact strata ({n_strata}) exceeds effective "
            "train or holdout size"
        )

    return SplitPool(
        eligible_indices=np.asarray(eligible_indices, dtype=int),
        fixed_train_indices=np.asarray(fixed_indices, dtype=int),
        eligible_strata=eligible_labels,
        singleton_strata=singleton_mapping,
    )


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    union = a | b
    return float(len(a & b) / len(union)) if union else 0.0


@dataclass(frozen=True)
class _BalanceScorer:
    categorical: Tuple[Tuple[np.ndarray, int], ...]
    numeric: Tuple[Tuple[np.ndarray, float], ...]

    @classmethod
    def from_metadata(
        cls,
        metadata: pd.DataFrame,
        spec: RepeatedHoldoutSpec,
    ) -> "_BalanceScorer":
        categorical = []
        for column in spec.balance_categorical:
            codes, levels = pd.factorize(
                metadata[column].astype(str), sort=True
            )
            categorical.append((np.asarray(codes, dtype=int), int(len(levels))))
        numeric = []
        for column in spec.balance_numeric:
            values = metadata[column].to_numpy(dtype=float)
            numeric.append((values, float(np.std(values, ddof=0))))
        return cls(tuple(categorical), tuple(numeric))

    def score(
        self,
        train_indices: np.ndarray,
        holdout_indices: np.ndarray,
    ) -> float:
        score = 0.0
        train_n = float(len(train_indices))
        holdout_n = float(len(holdout_indices))
        for codes, n_levels in self.categorical:
            train_counts = np.bincount(
                codes[train_indices], minlength=n_levels
            ).astype(float)
            holdout_counts = np.bincount(
                codes[holdout_indices], minlength=n_levels
            ).astype(float)
            score += 0.5 * float(
                np.abs(train_counts / train_n - holdout_counts / holdout_n).sum()
            )
        for values, scale in self.numeric:
            if scale > 0:
                score += abs(
                    float(values[train_indices].mean())
                    - float(values[holdout_indices].mean())
                ) / scale
        return float(score)

def generate_repeated_holdout(
    metadata: pd.DataFrame,
    spec: RepeatedHoldoutSpec,
) -> SplitGenerationResult:
    """Generate deterministic distinct repeated holdouts using metadata only."""

    if len(metadata) != spec.expected_patients:
        raise SplitGenerationError(
            "Current implementation expects one metadata row per patient after "
            "validation"
        )

    pool = _prepare_split_pool(metadata, spec)
    effective_train_size = spec.train_size - len(pool.fixed_train_indices)
    patient_ids = metadata["patient_id"].astype(str).to_numpy()

    scorer = _BalanceScorer.from_metadata(metadata, spec)
    chosen: List[ChosenSplit] = []
    previous_holdouts: List[Set[str]] = []
    split_x = np.zeros(len(pool.eligible_indices), dtype=np.uint8)
    for repeat_index in range(1, spec.repeats + 1):
        best: Optional[Tuple[Tuple[float, float, float, int], ChosenSplit]] = None
        eligible_candidate_count = 0
        seed_start = spec.base_seed + (
            (repeat_index - 1) * spec.candidates_per_repeat
        )
        splitter = StratifiedShuffleSplit(
            n_splits=spec.candidates_per_repeat,
            train_size=effective_train_size,
            test_size=spec.holdout_size,
            random_state=seed_start,
        )
        for offset, (train_local, holdout_local) in enumerate(
            splitter.split(split_x, pool.eligible_strata)
        ):
            seed = int(seed_start + offset)
            random_train = pool.eligible_indices[np.asarray(train_local, dtype=int)]
            holdout_indices = pool.eligible_indices[
                np.asarray(holdout_local, dtype=int)
            ]
            train_indices = np.concatenate(
                [pool.fixed_train_indices, random_train]
            ).astype(int)
            train_indices = np.sort(train_indices)
            holdout_indices = np.sort(holdout_indices.astype(int))

            holdout_set = set(patient_ids[holdout_indices].tolist())
            overlaps = [_jaccard(holdout_set, old) for old in previous_holdouts]
            max_overlap = max(overlaps, default=0.0)
            mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0
            if max_overlap > spec.max_pairwise_holdout_jaccard + 1e-15:
                continue
            eligible_candidate_count += 1
            base_score = scorer.score(train_indices, holdout_indices)
            total_score = base_score + spec.diversity_weight * mean_overlap
            split_id = f"split_{repeat_index:02d}"
            candidate = ChosenSplit(
                repeat_index=repeat_index,
                split_id=split_id,
                candidate_seed=seed,
                train_indices=train_indices,
                holdout_indices=holdout_indices,
                fixed_train_indices=pool.fixed_train_indices.copy(),
                base_balance_score=float(base_score),
                mean_previous_jaccard=float(mean_overlap),
                max_previous_jaccard=float(max_overlap),
                total_score=float(total_score),
                eligible_candidates=0,
            )
            rank = (
                float(total_score),
                float(base_score),
                float(max_overlap),
                int(seed),
            )
            if best is None or rank < best[0]:
                best = (rank, candidate)

        if best is None:
            raise SplitGenerationError(
                f"No eligible candidate found for repeat {repeat_index}; "
                "increase max_pairwise_holdout_jaccard or "
                "candidates_per_repeat"
            )
        selected = best[1]
        selected = ChosenSplit(
            **{
                **selected.__dict__,
                "eligible_candidates": int(eligible_candidate_count),
            }
        )
        chosen.append(selected)
        previous_holdouts.append(
            set(patient_ids[selected.holdout_indices].tolist())
        )

    assignment_frames: List[pd.DataFrame] = []
    count_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    balance_rows: List[Dict[str, Any]] = []
    fixed_ids = set(
        metadata.iloc[pool.fixed_train_indices]["sample_id"].astype(str)
    )

    for item in chosen:
        roles = np.full(len(metadata), spec.train_role, dtype=object)
        roles[item.holdout_indices] = spec.holdout_role
        reasons = np.full(
            len(metadata), "stratified_candidate_train", dtype=object
        )
        reasons[item.holdout_indices] = "stratified_candidate_holdout"
        reasons[item.fixed_train_indices] = "fixed_train_singleton_stratum"

        frame = metadata.copy()
        frame.insert(0, "split_set_id", spec.split_set_id)
        frame.insert(1, "split_id", item.split_id)
        frame.insert(2, "repeat_index", item.repeat_index)
        frame.insert(3, "role", roles)
        frame.insert(4, "assignment_reason", reasons)
        frame.insert(5, "candidate_seed", item.candidate_seed)
        frame = frame.sort_values("_cohort_order", kind="stable")
        assignment_frames.append(frame)

        for role_name in (spec.train_role, spec.holdout_role):
            count_rows.append(
                {
                    "split_set_id": spec.split_set_id,
                    "split_id": item.split_id,
                    "repeat_index": item.repeat_index,
                    "role": role_name,
                    "patient_count": int((roles == role_name).sum()),
                    "sample_count": int((roles == role_name).sum()),
                    "fixed_train_count": int(
                        len(item.fixed_train_indices)
                        if role_name == spec.train_role
                        else 0
                    ),
                }
            )
        candidate_rows.append(
            {
                "split_set_id": spec.split_set_id,
                "split_id": item.split_id,
                "repeat_index": item.repeat_index,
                "candidate_seed": item.candidate_seed,
                "candidates_evaluated": spec.candidates_per_repeat,
                "eligible_candidates": item.eligible_candidates,
                "fixed_train_count": len(item.fixed_train_indices),
                "base_balance_score": item.base_balance_score,
                "mean_previous_holdout_jaccard": item.mean_previous_jaccard,
                "max_previous_holdout_jaccard": item.max_previous_jaccard,
                "diversity_weight": spec.diversity_weight,
                "total_score": item.total_score,
            }
        )

        train = metadata.iloc[item.train_indices]
        holdout = metadata.iloc[item.holdout_indices]
        categorical_columns: List[Tuple[str, str]] = []
        for name in spec.exact_strata:
            categorical_columns.append((name, "exact_stratum_component"))
        for name in spec.balance_categorical:
            categorical_columns.append((name, "balance_categorical"))
        for column, variable_type in categorical_columns:
            levels = sorted(metadata[column].astype(str).unique().tolist())
            train_counts = train[column].astype(str).value_counts()
            holdout_counts = holdout[column].astype(str).value_counts()
            for level in levels:
                train_n = int(train_counts.get(level, 0))
                holdout_n = int(holdout_counts.get(level, 0))
                train_prop = train_n / len(train)
                holdout_prop = holdout_n / len(holdout)
                balance_rows.append(
                    {
                        "split_set_id": spec.split_set_id,
                        "split_id": item.split_id,
                        "repeat_index": item.repeat_index,
                        "variable": column,
                        "variable_type": variable_type,
                        "level": level,
                        "train_n": train_n,
                        "holdout_n": holdout_n,
                        "train_value": train_prop,
                        "holdout_value": holdout_prop,
                        "absolute_difference": abs(train_prop - holdout_prop),
                        "standardized_difference": np.nan,
                    }
                )
        for column in spec.balance_numeric:
            full_sd = float(metadata[column].std(ddof=0))
            train_mean = float(train[column].mean())
            holdout_mean = float(holdout[column].mean())
            standardized = (
                abs(train_mean - holdout_mean) / full_sd if full_sd > 0 else 0.0
            )
            balance_rows.append(
                {
                    "split_set_id": spec.split_set_id,
                    "split_id": item.split_id,
                    "repeat_index": item.repeat_index,
                    "variable": column,
                    "variable_type": "balance_numeric",
                    "level": "__mean__",
                    "train_n": len(train),
                    "holdout_n": len(holdout),
                    "train_value": train_mean,
                    "holdout_value": holdout_mean,
                    "absolute_difference": abs(train_mean - holdout_mean),
                    "standardized_difference": standardized,
                }
            )

    assignments = pd.concat(assignment_frames, ignore_index=True, sort=False)
    assignments = assignments.drop(columns=["_cohort_order"])
    split_counts = pd.DataFrame(count_rows)
    selected_candidates = pd.DataFrame(candidate_rows)
    balance_audit = pd.DataFrame(balance_rows)

    overlap_rows: List[Dict[str, Any]] = []
    for left_index, left in enumerate(chosen):
        left_set = set(patient_ids[left.holdout_indices].tolist())
        for right in chosen[left_index + 1 :]:
            right_set = set(patient_ids[right.holdout_indices].tolist())
            overlap_rows.append(
                {
                    "split_set_id": spec.split_set_id,
                    "split_id_a": left.split_id,
                    "split_id_b": right.split_id,
                    "holdout_intersection": int(len(left_set & right_set)),
                    "holdout_union": int(len(left_set | right_set)),
                    "holdout_jaccard": _jaccard(left_set, right_set),
                }
            )
    pairwise_overlap = pd.DataFrame(
        overlap_rows,
        columns=[
            "split_set_id",
            "split_id_a",
            "split_id_b",
            "holdout_intersection",
            "holdout_union",
            "holdout_jaccard",
        ],
    )

    holdout_counts = (
        assignments.loc[assignments["role"] == spec.holdout_role]
        .groupby(["sample_id", "patient_id"], sort=False)
        .size()
        .rename("holdout_count")
        .reset_index()
    )
    all_ids = metadata[["sample_id", "patient_id", "_cohort_order"]].copy()
    sample_holdout_frequency = all_ids.merge(
        holdout_counts,
        on=["sample_id", "patient_id"],
        how="left",
        validate="one_to_one",
    )
    sample_holdout_frequency["holdout_count"] = (
        sample_holdout_frequency["holdout_count"].fillna(0).astype(int)
    )
    sample_holdout_frequency["holdout_fraction"] = (
        sample_holdout_frequency["holdout_count"] / spec.repeats
    )
    sample_holdout_frequency["fixed_train"] = (
        sample_holdout_frequency["sample_id"].astype(str).isin(fixed_ids)
    )
    sample_holdout_frequency = sample_holdout_frequency.sort_values(
        "_cohort_order"
    ).drop(columns=["_cohort_order"])

    fixed_rows: List[Dict[str, Any]] = []
    labels = _stratum_labels(metadata, spec.exact_strata)
    for index in pool.fixed_train_indices:
        row = metadata.iloc[int(index)]
        fixed_rows.append(
            {
                "split_set_id": spec.split_set_id,
                "sample_id": str(row["sample_id"]),
                "patient_id": str(row["patient_id"]),
                "exact_stratum": str(labels.iloc[int(index)]),
                "singleton_strata_policy": spec.singleton_strata_policy,
                "role_in_every_repeat": spec.train_role,
                **{name: row[name] for name in spec.exact_strata},
            }
        )
    fixed_train_audit = pd.DataFrame(fixed_rows)

    return SplitGenerationResult(
        metadata=metadata.drop(columns=["_cohort_order"]),
        assignments=assignments,
        split_counts=split_counts,
        balance_audit=balance_audit,
        selected_candidates=selected_candidates,
        pairwise_overlap=pairwise_overlap,
        sample_holdout_frequency=sample_holdout_frequency,
        fixed_train_audit=fixed_train_audit,
        chosen_splits=tuple(chosen),
    )


def validate_generation_result(
    result: SplitGenerationResult,
    spec: RepeatedHoldoutSpec,
) -> Mapping[str, Any]:
    """Validate generated dimensions, roles, overlap, and singleton policy."""

    expected_rows = spec.expected_patients * spec.repeats
    if len(result.assignments) != expected_rows:
        raise SplitGenerationError(
            f"Assignment rows={len(result.assignments)}, expected={expected_rows}"
        )
    required = {
        "split_set_id",
        "split_id",
        "repeat_index",
        "role",
        "assignment_reason",
        "sample_id",
        "patient_id",
    }
    missing = sorted(required - set(result.assignments.columns))
    if missing:
        raise SplitGenerationError(
            f"Assignments are missing required columns: {missing}"
        )
    duplicate_count = int(
        result.assignments.duplicated(["split_id", "patient_id"]).sum()
    )
    if duplicate_count:
        raise SplitGenerationError(
            f"Assignments contain duplicate split/patient rows: {duplicate_count}"
        )

    fixed_expected = set(spec.expected_fixed_train_sample_ids)
    for repeat_index, frame in result.assignments.groupby(
        "repeat_index", sort=True
    ):
        counts = frame["role"].value_counts().to_dict()
        if counts.get(spec.train_role, 0) != spec.train_size:
            raise SplitGenerationError(
                f"Repeat {repeat_index} train count differs from {spec.train_size}"
            )
        if counts.get(spec.holdout_role, 0) != spec.holdout_size:
            raise SplitGenerationError(
                f"Repeat {repeat_index} holdout count differs from {spec.holdout_size}"
            )
        fixed_observed = set(
            frame.loc[
                frame["assignment_reason"] == "fixed_train_singleton_stratum",
                "sample_id",
            ].astype(str)
        )
        if fixed_observed != fixed_expected:
            raise SplitGenerationError(
                f"Repeat {repeat_index} fixed-train IDs differ from configuration"
            )
        if fixed_observed & set(
            frame.loc[frame["role"] == spec.holdout_role, "sample_id"].astype(str)
        ):
            raise SplitGenerationError(
                f"Repeat {repeat_index} places a fixed-train sample in holdout"
            )

    maximum_jaccard = (
        float(result.pairwise_overlap["holdout_jaccard"].max())
        if not result.pairwise_overlap.empty
        else 0.0
    )
    if maximum_jaccard > spec.max_pairwise_holdout_jaccard + 1e-12:
        raise SplitGenerationError(
            "Generated pairwise holdout overlap exceeds configured limit"
        )
    unique_holdouts = int(
        result.assignments.loc[
            result.assignments["role"] == spec.holdout_role
        ]
        .groupby("split_id")["patient_id"]
        .apply(lambda x: frozenset(x.astype(str)))
        .nunique()
    )
    if unique_holdouts != spec.repeats:
        raise SplitGenerationError("Repeated holdout sets are not all distinct")

    return {
        "status": "PASS",
        "split_set_id": spec.split_set_id,
        "samples": spec.expected_samples,
        "patients": spec.expected_patients,
        "repeats": spec.repeats,
        "train_size": spec.train_size,
        "holdout_size": spec.holdout_size,
        "fixed_train_count": len(fixed_expected),
        "fixed_train_sample_ids": sorted(fixed_expected),
        "maximum_pairwise_holdout_jaccard": maximum_jaccard,
        "metadata_only_candidate_ranking": True,
        "model_predictions_used": False,
    }


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _metadata_input_manifest(spec: RepeatedHoldoutSpec) -> List[Dict[str, Any]]:
    rows = []
    for index, path in enumerate(spec.metadata_files, 1):
        if not path.is_file():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        frame = pd.read_csv(path)
        stat = path.stat()
        rows.append(
            {
                "source_index": index,
                "configured_path": _repo_relative(path, spec.repository_root),
                "resolved_path": str(path),
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "size_bytes": int(stat.st_size),
                "mtime_utc": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
                "sha256": _sha256_file(path),
            }
        )
    return rows


def write_frozen_split_set(
    result: SplitGenerationResult,
    spec: RepeatedHoldoutSpec,
    *,
    replace_incomplete: bool = False,
) -> Mapping[str, Any]:
    """Write a split set once and freeze it with hashes.

    A directory containing ``SPLITS_FROZEN.json`` is immutable. A different
    assignment requires a new split-set ID. ``replace_incomplete`` is accepted
    only for a partial directory without a frozen marker.
    """

    validation = validate_generation_result(result, spec)
    output_root = spec.output_root
    frozen_marker = output_root / "SPLITS_FROZEN.json"
    if frozen_marker.exists():
        raise FileExistsError(
            f"Split set is already frozen and cannot be overwritten: {output_root}. "
            "Create a new split_set.id for a different assignment set."
        )
    if output_root.exists() and any(output_root.iterdir()):
        if not replace_incomplete:
            raise FileExistsError(
                f"Incomplete split output already exists: {output_root}. "
                "Use --replace-incomplete only after confirming it has no "
                "frozen marker."
            )
        allowed_root = resolve_project_path(DEFAULT_SPLIT_ROOT, spec.repository_root)
        if not _is_within(output_root, allowed_root):
            raise SplitGenerationError(
                "Refusing to replace a directory outside split-set root"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    _require_yaml()
    paths = {
        "assignments": output_root / "repeated_holdout_assignments.csv",
        "split_counts": output_root / "split_counts.csv",
        "balance_audit": output_root / "split_balance_audit.csv",
        "selected_candidates": output_root / "selected_split_candidates.csv",
        "pairwise_overlap": output_root / "pairwise_holdout_overlap.csv",
        "sample_holdout_frequency": output_root / "sample_holdout_frequency.csv",
        "fixed_train_audit": output_root / "fixed_train_singleton_audit.csv",
        "metadata_snapshot": output_root / "combined_metadata_snapshot.csv",
        "spec_snapshot": output_root / "split_set_spec_snapshot.yaml",
        "input_manifest": output_root / "metadata_input_manifest.json",
        "validation": output_root / "split_generation_validation.json",
    }
    result.assignments.to_csv(paths["assignments"], index=False)
    result.split_counts.to_csv(paths["split_counts"], index=False)
    result.balance_audit.to_csv(paths["balance_audit"], index=False)
    result.selected_candidates.to_csv(paths["selected_candidates"], index=False)
    result.pairwise_overlap.to_csv(paths["pairwise_overlap"], index=False)
    result.sample_holdout_frequency.to_csv(
        paths["sample_holdout_frequency"], index=False
    )
    result.fixed_train_audit.to_csv(paths["fixed_train_audit"], index=False)
    result.metadata.to_csv(paths["metadata_snapshot"], index=False)
    paths["spec_snapshot"].write_text(
        yaml.safe_dump(spec.raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    _write_json(paths["input_manifest"], _metadata_input_manifest(spec))
    _write_json(paths["validation"], validation)

    file_hashes = {
        name: {
            "path": _repo_relative(path, spec.repository_root),
            "sha256": _sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for name, path in paths.items()
    }
    marker = {
        "status": "FROZEN",
        "split_set_version": SPLIT_SET_VERSION,
        "split_set_id": spec.split_set_id,
        "description": spec.description,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "expected_samples": spec.expected_samples,
        "expected_patients": spec.expected_patients,
        "repeats": spec.repeats,
        "train_size": spec.train_size,
        "holdout_size": spec.holdout_size,
        "exact_strata": list(spec.exact_strata),
        "singleton_strata_policy": spec.singleton_strata_policy,
        "fixed_train_sample_ids": sorted(spec.expected_fixed_train_sample_ids),
        "fixed_train_generalization_limitation": (
            "Exact singleton strata are never represented in holdout; performance "
            "summaries from this split set do not directly validate generalization "
            "to those strata."
        ),
        "balance_categorical": list(spec.balance_categorical),
        "balance_numeric": list(spec.balance_numeric),
        "candidate_seeds": [
            item.candidate_seed for item in result.chosen_splits
        ],
        "maximum_pairwise_holdout_jaccard": validation[
            "maximum_pairwise_holdout_jaccard"
        ],
        "metadata_only_candidate_ranking": True,
        "model_predictions_used": False,
        "independent_test_performance_used": False,
        "files": file_hashes,
    }
    _write_json(frozen_marker, marker)
    return marker


def validate_frozen_split_set(
    spec: RepeatedHoldoutSpec,
) -> Mapping[str, Any]:
    """Validate a frozen split set and all recorded file hashes."""

    marker_path = spec.output_root / "SPLITS_FROZEN.json"
    if not marker_path.is_file():
        raise FileNotFoundError(f"Frozen marker not found: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "FROZEN":
        raise SplitGenerationError("SPLITS_FROZEN.json status is not FROZEN")
    if marker.get("split_set_id") != spec.split_set_id:
        raise SplitGenerationError("Frozen marker split_set_id mismatch")
    if int(marker.get("repeats", -1)) != spec.repeats:
        raise SplitGenerationError("Frozen marker repeat count mismatch")
    if int(marker.get("train_size", -1)) != spec.train_size:
        raise SplitGenerationError("Frozen marker train size mismatch")
    if int(marker.get("holdout_size", -1)) != spec.holdout_size:
        raise SplitGenerationError("Frozen marker holdout size mismatch")
    if set(marker.get("fixed_train_sample_ids", [])) != set(
        spec.expected_fixed_train_sample_ids
    ):
        raise SplitGenerationError("Frozen marker fixed-train IDs mismatch")

    files = marker.get("files")
    if not isinstance(files, Mapping) or not files:
        raise SplitGenerationError("Frozen marker lacks file manifest")
    for name, record in files.items():
        if not isinstance(record, Mapping):
            raise SplitGenerationError(f"Invalid frozen file record: {name}")
        path_value = record.get("path")
        if not isinstance(path_value, str):
            raise SplitGenerationError(f"Frozen file path missing: {name}")
        path = resolve_project_path(path_value, spec.repository_root)
        if not path.is_file():
            raise FileNotFoundError(f"Frozen file missing: {path}")
        observed = _sha256_file(path)
        expected = str(record.get("sha256", ""))
        if observed != expected:
            raise SplitGenerationError(
                f"Frozen file SHA256 mismatch for {name}: "
                f"observed={observed}, expected={expected}"
            )

    assignments_path = resolve_project_path(
        files["assignments"]["path"], spec.repository_root
    )
    assignments = pd.read_csv(assignments_path)
    if len(assignments) != spec.expected_patients * spec.repeats:
        raise SplitGenerationError("Frozen assignment row count mismatch")
    for repeat_index, frame in assignments.groupby("repeat_index", sort=True):
        counts = frame["role"].value_counts().to_dict()
        if counts.get(spec.train_role, 0) != spec.train_size:
            raise SplitGenerationError(
                f"Frozen repeat {repeat_index} train count mismatch"
            )
        if counts.get(spec.holdout_role, 0) != spec.holdout_size:
            raise SplitGenerationError(
                f"Frozen repeat {repeat_index} holdout count mismatch"
            )
        fixed = set(
            frame.loc[
                frame["assignment_reason"] == "fixed_train_singleton_stratum",
                "sample_id",
            ].astype(str)
        )
        if fixed != set(spec.expected_fixed_train_sample_ids):
            raise SplitGenerationError(
                f"Frozen repeat {repeat_index} fixed-train IDs mismatch"
            )

    return {
        "status": "PASS",
        "split_set_id": spec.split_set_id,
        "marker": str(marker_path),
        "assignment_rows": int(len(assignments)),
        "assignment_sha256": str(files["assignments"]["sha256"]),
        "repeats": spec.repeats,
        "train_size": spec.train_size,
        "holdout_size": spec.holdout_size,
        "fixed_train_sample_ids": sorted(spec.expected_fixed_train_sample_ids),
        "files_verified": len(files),
    }
