#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic, metadata-only repeated holdout generation for RA-ILD TRB.

This module creates a frozen set of patient-level train/holdout assignments from
combined cohort metadata. Candidate splits are ranked only by prespecified
metadata-balance criteria; labels are used for stratification, never model
predictions or downstream performance.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .paths import resolve_project_path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


SPLIT_SET_VERSION = "1.0"
SPLIT_SET_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")
DEFAULT_SPLIT_ROOT = Path("TRB/set/split_sets")


class SplitGenerationError(ValueError):
    """Raised when split input, configuration, or generated assignments are unsafe."""


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
class ChosenSplit:
    repeat_index: int
    split_id: str
    candidate_seed: int
    train_indices: np.ndarray
    holdout_indices: np.ndarray
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
    chosen_splits: Tuple[ChosenSplit, ...]


def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: mamba install -c conda-forge pyyaml")


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


def _integer(parent: Mapping[str, Any], key: str, context: str, minimum: int = 0) -> int:
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


def _string_list(value: Any, context: str, *, allow_empty: bool = True) -> Tuple[str, ...]:
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
        raise SplitGenerationError(f"Path must remain inside repository root: {path}") from exc


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
        raise SplitGenerationError(f"split_set_version must be '{SPLIT_SET_VERSION}'")

    split_set = _mapping(raw, "split_set", "root")
    split_set_id = _string(split_set, "id", "split_set")
    if not SPLIT_SET_ID_PATTERN.fullmatch(split_set_id):
        raise SplitGenerationError(
            "split_set.id must be 3-80 lowercase characters using only a-z, 0-9, '.', '_' or '-'"
        )
    description = str(split_set.get("description", "")).strip()
    output_value = split_set.get("output_root", str(DEFAULT_SPLIT_ROOT / split_set_id))
    if not isinstance(output_value, str) or not output_value.strip():
        raise SplitGenerationError("split_set.output_root must be a non-empty path")
    output_path = Path(output_value.strip())
    if output_path.is_absolute():
        raise SplitGenerationError("split_set.output_root must be repository-relative")
    output_root = resolve_project_path(output_path, repository_root)
    allowed_root = resolve_project_path(DEFAULT_SPLIT_ROOT, repository_root)
    if not _is_within(output_root, allowed_root):
        raise SplitGenerationError(
            f"split_set.output_root must be inside {DEFAULT_SPLIT_ROOT.as_posix()}"
        )
    if output_root.name != split_set_id:
        raise SplitGenerationError("split_set.output_root must end with the exact split_set.id")

    source = _mapping(raw, "source", "root")
    metadata_values = source.get("metadata_files")
    if not isinstance(metadata_values, Sequence) or isinstance(metadata_values, (str, bytes)):
        raise SplitGenerationError("source.metadata_files must be a non-empty list")
    metadata_files: List[Path] = []
    for index, value in enumerate(metadata_values):
        if not isinstance(value, str) or not value.strip():
            raise SplitGenerationError(
                f"source.metadata_files[{index}] must be a non-empty repository-relative path"
            )
        candidate = Path(value.strip())
        if candidate.is_absolute():
            raise SplitGenerationError("source.metadata_files entries must be repository-relative")
        metadata_files.append(resolve_project_path(candidate, repository_root))
    if not metadata_files:
        raise SplitGenerationError("source.metadata_files must not be empty")

    sample_id_column = _string(source, "sample_id_column", "source")
    patient_id_column = _string(source, "patient_id_column", "source")
    require_one = source.get("require_one_sample_per_patient", True)
    if not isinstance(require_one, bool):
        raise SplitGenerationError("source.require_one_sample_per_patient must be boolean")
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
            "split.train_size + split.holdout_size must equal source.expected_patients"
        )
    exact_strata = _string_list(split.get("exact_strata", []), "split.exact_strata", allow_empty=False)
    if label_column not in exact_strata:
        raise SplitGenerationError("split.label_column must be included in split.exact_strata")
    balance_categorical = _string_list(
        split.get("balance_categorical", []), "split.balance_categorical"
    )
    balance_numeric = _string_list(split.get("balance_numeric", []), "split.balance_numeric")
    candidates = _integer(split, "candidates_per_repeat", "split", 1)
    base_seed = _integer(split, "base_seed", "split", 0)
    diversity_weight = _number(split, "diversity_weight", "split", 0.0)
    max_jaccard = _number(
        split, "max_pairwise_holdout_jaccard", "split", 0.0, 1.0
    )
    if max_jaccard >= 1.0:
        raise SplitGenerationError("split.max_pairwise_holdout_jaccard must be < 1")
    roles = _mapping(split, "role_names", "split")
    train_role = _string(roles, "train", "split.role_names")
    holdout_role = _string(roles, "holdout", "split.role_names")
    if train_role == holdout_role:
        raise SplitGenerationError("split.role_names.train and holdout must differ")

    all_balance = (*exact_strata, *balance_categorical, *balance_numeric)
    if len(all_balance) != len(set(all_balance)):
        duplicates = sorted({name for name in all_balance if all_balance.count(name) > 1})
        raise SplitGenerationError(
            f"Stratification/balance variables must not be repeated across lists: {duplicates}"
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
        raise SplitGenerationError(f"Metadata file contains duplicated columns: {duplicated}")
    frame = frame.copy()
    frame["_source_metadata_index"] = int(index)
    frame["_source_metadata_path"] = str(path)
    frame["_source_row"] = np.arange(1, len(frame) + 1, dtype=int)
    return frame


def load_combined_metadata(spec: RepeatedHoldoutSpec) -> pd.DataFrame:
    """Combine source metadata and validate the patient-level splitting unit."""

    frames = [_read_metadata_file(path, index) for index, path in enumerate(spec.metadata_files, 1)]
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
            raise SplitGenerationError(f"Combined metadata column {column!r} contains missing values")

    metadata = metadata.copy()
    metadata[spec.sample_id_column] = metadata[spec.sample_id_column].astype(str).str.strip()
    metadata[spec.patient_id_column] = metadata[spec.patient_id_column].astype(str).str.strip()
    if (metadata[spec.sample_id_column] == "").any():
        raise SplitGenerationError("Combined metadata contains blank sample identifiers")
    if (metadata[spec.patient_id_column] == "").any():
        raise SplitGenerationError("Combined metadata contains blank patient identifiers")

    duplicate_samples = metadata.loc[
        metadata[spec.sample_id_column].duplicated(keep=False), spec.sample_id_column
    ].unique().tolist()
    if duplicate_samples:
        raise SplitGenerationError(
            f"Combined metadata contains duplicate sample IDs: {duplicate_samples[:20]}"
        )
    patient_counts = metadata.groupby(spec.patient_id_column, sort=False).size()
    if spec.require_one_sample_per_patient and (patient_counts != 1).any():
        bad = patient_counts[patient_counts != 1].head(20).to_dict()
        raise SplitGenerationError(
            "RA-ILD TRB repeated holdout requires exactly one sample per patient; "
            f"violations: {bad}"
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
            raise SplitGenerationError(f"Metadata column {column!r} contains blank values")
    for column in spec.balance_numeric:
        converted = pd.to_numeric(metadata[column], errors="coerce")
        bad = converted.isna() | ~np.isfinite(converted.to_numpy(float))
        if bad.any():
            examples = metadata.loc[bad, column].astype(str).head(10).tolist()
            raise SplitGenerationError(
                f"Metadata numeric balance column {column!r} contains invalid values: {examples}"
            )
        metadata[column] = converted.astype(float)

    metadata.insert(0, "sample_id", metadata[spec.sample_id_column].astype(str))
    metadata.insert(1, "patient_id", metadata[spec.patient_id_column].astype(str))
    metadata.insert(2, "_cohort_order", np.arange(len(metadata), dtype=int))
    return metadata


def _stratum_labels(metadata: pd.DataFrame, exact_strata: Sequence[str]) -> pd.Series:
    values = metadata[list(exact_strata)].astype(str)
    return values.agg("\x1f".join, axis=1)


def _validate_strata(labels: pd.Series, train_size: int, holdout_size: int) -> None:
    counts = labels.value_counts(sort=False)
    singletons = counts[counts < 2]
    if not singletons.empty:
        raise SplitGenerationError(
            "Every exact stratum needs at least two patients for stratified holdout; "
            f"singleton strata: {singletons.to_dict()}"
        )
    n_strata = int(len(counts))
    if n_strata > train_size or n_strata > holdout_size:
        raise SplitGenerationError(
            f"Number of exact strata ({n_strata}) exceeds train or holdout size"
        )


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    union = a | b
    return float(len(a & b) / len(union)) if union else 0.0


def _categorical_total_variation(
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
    holdout_indices: np.ndarray,
    column: str,
) -> float:
    levels = sorted(metadata[column].astype(str).unique().tolist())
    train = metadata.iloc[train_indices][column].astype(str).value_counts(normalize=True)
    holdout = metadata.iloc[holdout_indices][column].astype(str).value_counts(normalize=True)
    return 0.5 * sum(abs(float(train.get(level, 0.0)) - float(holdout.get(level, 0.0))) for level in levels)


def _numeric_standardized_difference(
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
    holdout_indices: np.ndarray,
    column: str,
) -> float:
    full = metadata[column].to_numpy(float)
    scale = float(np.std(full, ddof=0))
    if scale <= 0:
        return 0.0
    train_mean = float(metadata.iloc[train_indices][column].mean())
    holdout_mean = float(metadata.iloc[holdout_indices][column].mean())
    return abs(train_mean - holdout_mean) / scale


def _metadata_balance_score(
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
    holdout_indices: np.ndarray,
    spec: RepeatedHoldoutSpec,
) -> float:
    score = 0.0
    for column in spec.balance_categorical:
        score += _categorical_total_variation(metadata, train_indices, holdout_indices, column)
    for column in spec.balance_numeric:
        score += _numeric_standardized_difference(metadata, train_indices, holdout_indices, column)
    return float(score)


def generate_repeated_holdout(
    metadata: pd.DataFrame,
    spec: RepeatedHoldoutSpec,
) -> SplitGenerationResult:
    """Generate deterministic, distinct repeated holdouts using metadata-only ranking."""

    if len(metadata) != spec.expected_patients:
        raise SplitGenerationError(
            "Current implementation expects one metadata row per patient after validation"
        )
    strata = _stratum_labels(metadata, spec.exact_strata)
    _validate_strata(strata, spec.train_size, spec.holdout_size)
    patient_ids = metadata["patient_id"].astype(str).to_numpy()

    chosen: List[ChosenSplit] = []
    previous_holdouts: List[set[str]] = []
    for repeat_index in range(1, spec.repeats + 1):
        best: Optional[Tuple[Tuple[float, float, float, int], ChosenSplit]] = None
        eligible = 0
        seed_start = spec.base_seed + (repeat_index - 1) * spec.candidates_per_repeat
        for offset in range(spec.candidates_per_repeat):
            seed = int(seed_start + offset)
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                train_size=spec.train_size,
                test_size=spec.holdout_size,
                random_state=seed,
            )
            train_indices, holdout_indices = next(splitter.split(np.zeros(len(metadata)), strata))
            holdout_set = set(patient_ids[holdout_indices].tolist())
            overlaps = [_jaccard(holdout_set, old) for old in previous_holdouts]
            max_overlap = max(overlaps, default=0.0)
            mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0
            if max_overlap > spec.max_pairwise_holdout_jaccard + 1e-15:
                continue
            eligible += 1
            base_score = _metadata_balance_score(metadata, train_indices, holdout_indices, spec)
            total_score = base_score + spec.diversity_weight * mean_overlap
            split_id = f"split_{repeat_index:02d}"
            candidate = ChosenSplit(
                repeat_index=repeat_index,
                split_id=split_id,
                candidate_seed=seed,
                train_indices=np.asarray(train_indices, dtype=int),
                holdout_indices=np.asarray(holdout_indices, dtype=int),
                base_balance_score=float(base_score),
                mean_previous_jaccard=float(mean_overlap),
                max_previous_jaccard=float(max_overlap),
                total_score=float(total_score),
                eligible_candidates=0,
            )
            rank = (float(total_score), float(base_score), float(max_overlap), int(seed))
            if best is None or rank < best[0]:
                best = (rank, candidate)
        if best is None:
            raise SplitGenerationError(
                f"No eligible candidate found for repeat {repeat_index}; increase "
                "max_pairwise_holdout_jaccard or candidates_per_repeat"
            )
        selected = best[1]
        selected = ChosenSplit(
            **{**selected.__dict__, "eligible_candidates": int(eligible)}
        )
        chosen.append(selected)
        previous_holdouts.append(set(patient_ids[selected.holdout_indices].tolist()))

    assignment_frames: List[pd.DataFrame] = []
    count_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    balance_rows: List[Dict[str, Any]] = []
    for item in chosen:
        roles = np.full(len(metadata), spec.train_role, dtype=object)
        roles[item.holdout_indices] = spec.holdout_role
        frame = metadata.copy()
        frame.insert(0, "split_set_id", spec.split_set_id)
        frame.insert(1, "split_id", item.split_id)
        frame.insert(2, "repeat_index", item.repeat_index)
        frame.insert(3, "role", roles)
        frame.insert(4, "candidate_seed", item.candidate_seed)
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
        for column, role in categorical_columns:
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
                        "variable_type": role,
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
            standardized = abs(train_mean - holdout_mean) / full_sd if full_sd > 0 else 0.0
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
        holdout_counts, on=["sample_id", "patient_id"], how="left", validate="one_to_one"
    )
    sample_holdout_frequency["holdout_count"] = (
        sample_holdout_frequency["holdout_count"].fillna(0).astype(int)
    )
    sample_holdout_frequency["holdout_fraction"] = (
        sample_holdout_frequency["holdout_count"] / spec.repeats
    )
    sample_holdout_frequency = sample_holdout_frequency.sort_values("_cohort_order").drop(
        columns=["_cohort_order"]
    )

    return SplitGenerationResult(
        metadata=metadata.drop(columns=["_cohort_order"]),
        assignments=assignments,
        split_counts=split_counts,
        balance_audit=balance_audit,
        selected_candidates=selected_candidates,
        pairwise_overlap=pairwise_overlap,
        sample_holdout_frequency=sample_holdout_frequency,
        chosen_splits=tuple(chosen),
    )


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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
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

    A directory containing ``SPLITS_FROZEN.json`` is immutable. To change a split,
    create a new split_set.id. ``replace_incomplete`` is accepted only for a partial
    directory that does not contain the frozen marker.
    """

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
                "Use --replace-incomplete only after confirming it has no frozen marker."
            )
        allowed_root = resolve_project_path(DEFAULT_SPLIT_ROOT, spec.repository_root)
        if not _is_within(output_root, allowed_root):
            raise SplitGenerationError("Refusing to replace a directory outside split-set root")
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
        "metadata_snapshot": output_root / "combined_metadata_snapshot.csv",
        "spec_snapshot": output_root / "split_set_spec_snapshot.yaml",
        "input_manifest": output_root / "metadata_input_manifest.json",
    }
    result.assignments.to_csv(paths["assignments"], index=False)
    result.split_counts.to_csv(paths["split_counts"], index=False)
    result.balance_audit.to_csv(paths["balance_audit"], index=False)
    result.selected_candidates.to_csv(paths["selected_candidates"], index=False)
    result.pairwise_overlap.to_csv(paths["pairwise_overlap"], index=False)
    result.sample_holdout_frequency.to_csv(paths["sample_holdout_frequency"], index=False)
    result.metadata.to_csv(paths["metadata_snapshot"], index=False)
    paths["spec_snapshot"].write_text(
        yaml.safe_dump(spec.raw, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    _write_json(paths["input_manifest"], _metadata_input_manifest(spec))

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
        "balance_categorical": list(spec.balance_categorical),
        "balance_numeric": list(spec.balance_numeric),
        "candidate_seeds": [item.candidate_seed for item in result.chosen_splits],
        "metadata_only_candidate_ranking": True,
        "model_predictions_used": False,
        "files": file_hashes,
    }
    _write_json(frozen_marker, marker)
    return marker
