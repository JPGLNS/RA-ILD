#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare full-cohort inputs for leakage-controlled repeated-holdout training.

Batch 04 reuses the existing nested-CV scientific engine.  This module only:

* combines the old 123-sample train matrix and 51-sample test matrix into a
  static-feature-only 174-sample matrix;
* builds a deterministic full-cohort sparse sequence representation;
* converts the three frozen 123/51 assignments into V2-compatible outer and
  inner assignment tables; and
* generates a concrete scheme YAML that points at those prepared inputs.

The full-cohort sequence catalog is an indexing universe only.  Public-reference
masks and all label-dependent public features are still rebuilt strictly from the
current training partition by :mod:`ra_ild_trb.nested_cv`.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold

from .paths import resolve_project_path
from .specifications import select_static_tcr_features

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


BUNDLE_VERSION = "1.0"


class RepeatedHoldoutTrainingError(ValueError):
    """Raised when a repeated-holdout training bundle cannot be prepared safely."""


@dataclass(frozen=True)
class TrainingBundleSpec:
    source_path: Path
    repository_root: Path
    raw: Mapping[str, Any]
    bundle_id: str
    output_root: Path
    split_assignments: Path
    split_frozen_marker: Path
    split_train_role: str
    split_holdout_role: str
    train_base_matrix: Path
    test_final_matrix: Path
    feature_manifest: Path
    train_aa_dir: Path
    test_aa_dir: Path
    sample_id_column: str
    label_column: str
    required_metadata_columns: Tuple[str, ...]
    sequence_column: str
    frequency_candidates: Tuple[str, ...]
    file_globs: Tuple[str, ...]
    inner_folds: int
    inner_strata: Tuple[str, ...]
    balance_categorical: Tuple[str, ...]
    balance_numeric: Tuple[str, ...]
    inner_candidates: int
    inner_base_seed: int
    scheme_template: Path
    generated_scheme: Path
    scheme_id: str
    scheme_output_root: str

    @property
    def full_metadata_path(self) -> Path:
        return self.output_root / "full_cohort_metadata.csv"

    @property
    def full_base_matrix_path(self) -> Path:
        return self.output_root / "full_cohort_static_base_matrix.csv.gz"

    @property
    def catalog_path(self) -> Path:
        return self.output_root / "full_cohort_cdr3_aa_catalog.csv.gz"

    @property
    def presence_path(self) -> Path:
        return self.output_root / "full_cohort_public_presence.npz"

    @property
    def frequency_path(self) -> Path:
        return self.output_root / "full_cohort_public_frequency.npz"

    @property
    def cache_metadata_path(self) -> Path:
        return self.output_root / "full_cohort_public_sparse_cache.json"

    @property
    def aa_manifest_path(self) -> Path:
        return self.output_root / "aa_clone_table_manifest.csv"

    @property
    def outer_assignments_path(self) -> Path:
        return self.output_root / "repeated_holdout_outer_assignments.csv"

    @property
    def inner_assignments_path(self) -> Path:
        return self.output_root / "repeated_holdout_inner_assignments.csv.gz"

    @property
    def inner_audit_path(self) -> Path:
        return self.output_root / "inner_fold_balance_audit.csv"

    @property
    def selected_inner_candidates_path(self) -> Path:
        return self.output_root / "selected_inner_fold_candidates.csv"

    @property
    def marker_path(self) -> Path:
        return self.output_root / "TRAINING_BUNDLE_FROZEN.json"


def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: mamba install -c conda-forge pyyaml")


def _mapping(parent: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise RepeatedHoldoutTrainingError(f"{context}.{key} must be a mapping")
    return value


def _string(parent: Mapping[str, Any], key: str, context: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RepeatedHoldoutTrainingError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _positive_int(parent: Mapping[str, Any], key: str, context: str, minimum: int = 1) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RepeatedHoldoutTrainingError(f"{context}.{key} must be an integer >= {minimum}")
    return int(value)


def _string_list(value: Any, context: str, *, allow_empty: bool = True) -> Tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RepeatedHoldoutTrainingError(f"{context} must be a list of strings")
    result: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise RepeatedHoldoutTrainingError(f"{context} must contain non-empty strings")
        result.append(item.strip())
    if not allow_empty and not result:
        raise RepeatedHoldoutTrainingError(f"{context} must not be empty")
    if len(result) != len(set(result)):
        raise RepeatedHoldoutTrainingError(f"{context} contains duplicates")
    return tuple(result)


def _repo_relative(path: Path, repository_root: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    root = Path(repository_root).expanduser().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise RepeatedHoldoutTrainingError(f"Path is outside repository root: {resolved}") from exc


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_training_bundle_spec(path: Path, *, repository_root: Path) -> TrainingBundleSpec:
    """Load and validate a Batch 04 training-bundle YAML."""

    _require_yaml()
    source_path = Path(path).expanduser().resolve()
    root = Path(repository_root).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Training-bundle spec not found: {source_path}")
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise RepeatedHoldoutTrainingError("Training-bundle YAML root must be a mapping")
    if str(raw.get("training_bundle_version")) != BUNDLE_VERSION:
        raise RepeatedHoldoutTrainingError(
            f"training_bundle_version must be {BUNDLE_VERSION!r}"
        )

    bundle = _mapping(raw, "bundle", "root")
    bundle_id = _string(bundle, "id", "bundle")
    output_root = resolve_project_path(_string(bundle, "output_root", "bundle"), root)
    allowed = resolve_project_path("TRB/set/training_bundles", root)
    try:
        output_root.relative_to(allowed)
    except ValueError as exc:
        raise RepeatedHoldoutTrainingError(
            f"bundle.output_root must be under {allowed}"
        ) from exc
    if output_root.name != bundle_id:
        raise RepeatedHoldoutTrainingError("bundle.output_root must end with bundle.id")

    split = _mapping(raw, "split_set", "root")
    source = _mapping(raw, "source", "root")
    aa = _mapping(raw, "aa_tables", "root")
    inner = _mapping(raw, "inner_cv", "root")
    scheme = _mapping(raw, "scheme", "root")

    spec = TrainingBundleSpec(
        source_path=source_path,
        repository_root=root,
        raw=raw,
        bundle_id=bundle_id,
        output_root=output_root,
        split_assignments=resolve_project_path(_string(split, "assignments", "split_set"), root),
        split_frozen_marker=resolve_project_path(_string(split, "frozen_marker", "split_set"), root),
        split_train_role=_string(split, "train_role", "split_set"),
        split_holdout_role=_string(split, "holdout_role", "split_set"),
        train_base_matrix=resolve_project_path(_string(source, "train_base_matrix", "source"), root),
        test_final_matrix=resolve_project_path(_string(source, "test_final_matrix", "source"), root),
        feature_manifest=resolve_project_path(_string(source, "feature_manifest", "source"), root),
        train_aa_dir=resolve_project_path(_string(source, "train_aa_clone_table_dir", "source"), root),
        test_aa_dir=resolve_project_path(_string(source, "test_aa_clone_table_dir", "source"), root),
        sample_id_column=_string(source, "sample_id_column", "source"),
        label_column=_string(source, "label_column", "source"),
        required_metadata_columns=_string_list(
            source.get("required_metadata_columns", []),
            "source.required_metadata_columns",
            allow_empty=False,
        ),
        sequence_column=_string(aa, "sequence_column", "aa_tables"),
        frequency_candidates=_string_list(
            aa.get("frequency_column_candidates", []),
            "aa_tables.frequency_column_candidates",
            allow_empty=False,
        ),
        file_globs=_string_list(
            aa.get("file_globs", []), "aa_tables.file_globs", allow_empty=False
        ),
        inner_folds=_positive_int(inner, "folds", "inner_cv", 2),
        inner_strata=_string_list(
            inner.get("exact_strata", []), "inner_cv.exact_strata", allow_empty=False
        ),
        balance_categorical=_string_list(
            inner.get("balance_categorical", []), "inner_cv.balance_categorical"
        ),
        balance_numeric=_string_list(
            inner.get("balance_numeric", []), "inner_cv.balance_numeric"
        ),
        inner_candidates=_positive_int(inner, "candidates", "inner_cv", 1),
        inner_base_seed=_positive_int(inner, "base_seed", "inner_cv", 0),
        scheme_template=resolve_project_path(_string(scheme, "template", "scheme"), root),
        generated_scheme=resolve_project_path(_string(scheme, "generated_scheme", "scheme"), root),
        scheme_id=_string(scheme, "id", "scheme"),
        scheme_output_root=_string(scheme, "output_root", "scheme"),
    )
    if spec.split_train_role == spec.split_holdout_role:
        raise RepeatedHoldoutTrainingError("split_set train_role and holdout_role must differ")
    return spec


def _read_csv(path: Path, label: str, **kwargs) -> pd.DataFrame:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    frame = pd.read_csv(path, **kwargs)
    duplicated = frame.columns[frame.columns.duplicated()].astype(str).tolist()
    if duplicated:
        raise RepeatedHoldoutTrainingError(f"{label} contains duplicated columns: {duplicated[:20]}")
    return frame


def load_frozen_assignments(spec: TrainingBundleSpec) -> Tuple[pd.DataFrame, Mapping[str, Any]]:
    """Load the frozen three-way split assignment and verify its marker hash."""

    if not spec.split_frozen_marker.is_file():
        raise FileNotFoundError(f"Frozen split marker not found: {spec.split_frozen_marker}")
    marker = json.loads(spec.split_frozen_marker.read_text(encoding="utf-8"))
    if not isinstance(marker, Mapping) or marker.get("status") != "FROZEN":
        raise RepeatedHoldoutTrainingError("Split marker is not a FROZEN split-set marker")
    assignments = _read_csv(spec.split_assignments, "frozen repeated-holdout assignments")
    expected_hash = (
        marker.get("files", {}).get("assignments", {}).get("sha256")
        if isinstance(marker.get("files"), Mapping)
        else None
    )
    observed_hash = _sha256_file(spec.split_assignments)
    if expected_hash and str(expected_hash) != observed_hash:
        raise RepeatedHoldoutTrainingError(
            "Frozen assignment SHA256 does not match SPLITS_FROZEN.json"
        )
    required = {
        "split_id",
        "repeat_index",
        "role",
        "sample_id",
        "patient_id",
        spec.label_column,
    }
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise RepeatedHoldoutTrainingError(f"Frozen assignments are missing columns: {missing}")
    assignments["sample_id"] = assignments["sample_id"].astype(str)
    assignments["patient_id"] = assignments["patient_id"].astype(str)
    assignments["repeat_index"] = pd.to_numeric(
        assignments["repeat_index"], errors="raise"
    ).astype(int)
    if assignments.duplicated(["split_id", "sample_id"]).any():
        raise RepeatedHoldoutTrainingError("Frozen assignments duplicate split_id/sample_id")
    split_ids = assignments["split_id"].drop_duplicates().astype(str).tolist()
    repeat_ids = sorted(assignments["repeat_index"].unique().tolist())
    if repeat_ids != list(range(1, len(split_ids) + 1)):
        raise RepeatedHoldoutTrainingError(
            f"repeat_index coverage must be 1..N; observed={repeat_ids}"
        )
    for split_id, frame in assignments.groupby("split_id", sort=False):
        counts = frame["role"].astype(str).value_counts().to_dict()
        if counts.get(spec.split_train_role, 0) < spec.inner_folds:
            raise RepeatedHoldoutTrainingError(
                f"{split_id} has too few training samples for {spec.inner_folds} inner folds"
            )
        if counts.get(spec.split_holdout_role, 0) < 1:
            raise RepeatedHoldoutTrainingError(f"{split_id} has no holdout samples")
        if len(frame) != frame["sample_id"].nunique():
            raise RepeatedHoldoutTrainingError(f"{split_id} sample IDs are not unique")
    return assignments, marker


def prepare_full_static_matrix(
    spec: TrainingBundleSpec,
    assignments: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Tuple[str, ...]]:
    """Combine old train/test matrices while retaining only static model inputs."""

    train = _read_csv(spec.train_base_matrix, "old training base matrix")
    test = _read_csv(spec.test_final_matrix, "old test final matrix")
    manifest = _read_csv(spec.feature_manifest, "feature manifest")
    for label, frame in (("train", train), ("test", test)):
        if spec.sample_id_column not in frame.columns:
            raise RepeatedHoldoutTrainingError(
                f"{label} matrix is missing {spec.sample_id_column!r}"
            )
        frame[spec.sample_id_column] = frame[spec.sample_id_column].astype(str)
        if frame[spec.sample_id_column].duplicated().any():
            raise RepeatedHoldoutTrainingError(f"{label} matrix has duplicate sample IDs")

    shared_columns = set(train.columns) & set(test.columns)
    static_features = tuple(
        select_static_tcr_features(
            manifest,
            available_columns=shared_columns,
            require_all_available=True,
        )
    )
    required = [spec.sample_id_column, *spec.required_metadata_columns]
    missing_train = sorted(set(required) - set(train.columns))
    missing_test = sorted(set(required) - set(test.columns))
    if missing_train or missing_test:
        raise RepeatedHoldoutTrainingError(
            f"Required columns missing: train={missing_train}, test={missing_test}"
        )
    output_columns: List[str] = []
    for name in required + list(static_features):
        if name not in output_columns:
            output_columns.append(name)
    combined = pd.concat(
        [train.loc[:, output_columns], test.loc[:, output_columns]],
        ignore_index=True,
        sort=False,
    )
    if combined[spec.sample_id_column].duplicated().any():
        duplicates = combined.loc[
            combined[spec.sample_id_column].duplicated(keep=False), spec.sample_id_column
        ].astype(str).unique().tolist()
        raise RepeatedHoldoutTrainingError(
            f"Train/test matrices overlap in sample IDs: {duplicates[:20]}"
        )

    cohort_order = (
        assignments.loc[assignments["repeat_index"] == 1, "sample_id"].astype(str).tolist()
    )
    if len(cohort_order) != len(set(cohort_order)):
        raise RepeatedHoldoutTrainingError("Split-set repeat 1 does not provide unique cohort order")
    if set(cohort_order) != set(combined[spec.sample_id_column].astype(str)):
        missing_matrix = sorted(set(cohort_order) - set(combined[spec.sample_id_column]))
        extra_matrix = sorted(set(combined[spec.sample_id_column]) - set(cohort_order))
        raise RepeatedHoldoutTrainingError(
            f"Full matrix and split-set sample mismatch: missing={missing_matrix[:20]}, "
            f"extra={extra_matrix[:20]}"
        )
    combined = combined.set_index(spec.sample_id_column).loc[cohort_order].reset_index()
    combined = combined.rename(columns={spec.sample_id_column: "sample_id"})

    split_metadata = (
        assignments.loc[assignments["repeat_index"] == 1]
        .drop(columns=["split_set_id", "split_id", "repeat_index", "role", "candidate_seed"], errors="ignore")
        .drop_duplicates("sample_id")
        .set_index("sample_id")
        .loc[cohort_order]
        .reset_index()
    )
    if spec.label_column in split_metadata.columns:
        observed = combined.set_index("sample_id")[spec.label_column].astype(str)
        expected = split_metadata.set_index("sample_id")[spec.label_column].astype(str)
        mismatched = observed.index[observed != expected].tolist()
        if mismatched:
            raise RepeatedHoldoutTrainingError(
                f"Matrix labels disagree with split metadata: {mismatched[:20]}"
            )

    if combined.isna().any().any():
        bad = combined.isna().sum()
        raise RepeatedHoldoutTrainingError(
            f"Full static matrix contains missing values: {bad[bad > 0].to_dict()}"
        )
    return combined, split_metadata, static_features


def _candidate_files(spec: TrainingBundleSpec) -> Tuple[Path, ...]:
    files: Dict[Path, None] = {}
    for directory in (spec.train_aa_dir, spec.test_aa_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"AA clone-table directory not found: {directory}")
        for pattern in spec.file_globs:
            for path in directory.glob(pattern):
                if path.is_file():
                    files[path.resolve()] = None
    return tuple(sorted(files, key=lambda value: str(value)))


def _filename_sample_id(path: Path, expected_ids: Sequence[str]) -> Optional[str]:
    name = path.name
    for suffix in (".csv.gz", ".csv", ".tsv.gz", ".tsv"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    matches = []
    for sample_id in expected_ids:
        escaped = re.escape(str(sample_id))
        if re.search(rf"(^|[^A-Za-z0-9]){escaped}([^A-Za-z0-9]|$)", name):
            matches.append(str(sample_id))
        elif name == str(sample_id) or name.startswith(str(sample_id) + "_"):
            matches.append(str(sample_id))
    matches = sorted(set(matches), key=len, reverse=True)
    return matches[0] if len(matches) == 1 else None


def discover_aa_clone_tables(
    spec: TrainingBundleSpec,
    expected_sample_ids: Sequence[str],
) -> pd.DataFrame:
    """Find exactly one usable AA clone table for every full-cohort sample."""

    expected = tuple(str(value) for value in expected_sample_ids)
    expected_set = set(expected)
    rows: List[Dict[str, Any]] = []
    assignments: Dict[str, List[Dict[str, Any]]] = {sample_id: [] for sample_id in expected}
    for path in _candidate_files(spec):
        try:
            preview = pd.read_csv(path, nrows=50)
        except Exception:
            continue
        if spec.sequence_column not in preview.columns:
            continue
        frequency_column = next(
            (name for name in spec.frequency_candidates if name in preview.columns), None
        )
        if frequency_column is None:
            continue
        sample_id: Optional[str] = None
        if "sample_id" in preview.columns:
            values = preview["sample_id"].dropna().astype(str).unique().tolist()
            if len(values) == 1 and values[0] in expected_set:
                sample_id = values[0]
        if sample_id is None:
            sample_id = _filename_sample_id(path, expected)
        if sample_id is None or sample_id not in expected_set:
            continue
        item = {
            "sample_id": sample_id,
            "path": str(path),
            "frequency_column": frequency_column,
            "size_bytes": int(path.stat().st_size),
        }
        assignments[sample_id].append(item)

    missing = [sample_id for sample_id, values in assignments.items() if not values]
    duplicate = {sample_id: values for sample_id, values in assignments.items() if len(values) > 1}
    if missing or duplicate:
        details = []
        if missing:
            details.append(f"missing samples={missing[:30]}")
        if duplicate:
            compact = {key: [Path(item["path"]).name for item in value] for key, value in list(duplicate.items())[:10]}
            details.append(f"multiple candidate files={compact}")
        raise RepeatedHoldoutTrainingError(
            "Unable to map one AA clone table per sample: " + "; ".join(details)
        )
    for sample_id in expected:
        rows.append(assignments[sample_id][0])
    manifest = pd.DataFrame(rows)
    if manifest["path"].duplicated().any():
        raise RepeatedHoldoutTrainingError("The same AA clone table was assigned to multiple samples")
    return manifest


def _load_sample_sequences(
    path: Path,
    sequence_column: str,
    frequency_column: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=[sequence_column, frequency_column])
    frame = frame.dropna(subset=[sequence_column, frequency_column]).copy()
    frame[sequence_column] = frame[sequence_column].astype(str).str.strip().str.upper()
    frame = frame.loc[frame[sequence_column].ne("")].copy()
    frame[frequency_column] = pd.to_numeric(frame[frequency_column], errors="coerce")
    frame = frame.dropna(subset=[frequency_column])
    values = frame[frequency_column].to_numpy(float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise RepeatedHoldoutTrainingError(
            f"AA clone table contains invalid frequency values: {path}"
        )
    frame = (
        frame.groupby(sequence_column, sort=False, as_index=False)[frequency_column]
        .sum()
    )
    total = float(frame[frequency_column].sum())
    if total <= 0:
        raise RepeatedHoldoutTrainingError(f"AA clone table has zero total frequency: {path}")
    frame[frequency_column] = frame[frequency_column].astype(float) / total
    return frame


def build_full_public_sparse_cache(
    spec: TrainingBundleSpec,
    sample_ids: Sequence[str],
    aa_manifest: pd.DataFrame,
) -> Mapping[str, Any]:
    """Build deterministic full-cohort presence/frequency sparse matrices."""

    catalog: set[str] = set()
    manifest_lookup = aa_manifest.set_index("sample_id")
    for sample_id in sample_ids:
        row = manifest_lookup.loc[str(sample_id)]
        frame = _load_sample_sequences(
            Path(row["path"]), spec.sequence_column, str(row["frequency_column"])
        )
        catalog.update(frame[spec.sequence_column].astype(str).tolist())
    ordered_catalog = tuple(sorted(catalog))
    if not ordered_catalog:
        raise RepeatedHoldoutTrainingError("Full-cohort AA catalog is empty")
    column_lookup = {sequence: index for index, sequence in enumerate(ordered_catalog)}

    index_chunks: List[np.ndarray] = []
    frequency_chunks: List[np.ndarray] = []
    indptr = [0]
    sample_audit = []
    for sample_id in sample_ids:
        row = manifest_lookup.loc[str(sample_id)]
        frame = _load_sample_sequences(
            Path(row["path"]), spec.sequence_column, str(row["frequency_column"])
        )
        indices = np.fromiter(
            (column_lookup[value] for value in frame[spec.sequence_column].astype(str)),
            dtype=np.int32,
            count=len(frame),
        )
        frequencies = frame[str(row["frequency_column"])].to_numpy(np.float64)
        order = np.argsort(indices, kind="stable")
        indices = indices[order]
        frequencies = frequencies[order]
        index_chunks.append(indices)
        frequency_chunks.append(frequencies)
        indptr.append(indptr[-1] + len(indices))
        sample_audit.append(
            {
                "sample_id": str(sample_id),
                "unique_cdr3_aa": int(len(indices)),
                "frequency_sum": float(frequencies.sum()),
            }
        )

    all_indices = np.concatenate(index_chunks).astype(np.int32, copy=False)
    all_frequency = np.concatenate(frequency_chunks).astype(np.float64, copy=False)
    indptr_array = np.asarray(indptr, dtype=np.int64)
    shape = (len(sample_ids), len(ordered_catalog))
    presence = sparse.csr_matrix(
        (np.ones(len(all_indices), dtype=np.uint8), all_indices, indptr_array),
        shape=shape,
    )
    frequency = sparse.csr_matrix(
        (all_frequency, all_indices, indptr_array),
        shape=shape,
    )
    presence.sort_indices()
    frequency.sort_indices()

    catalog_frame = pd.DataFrame(
        {
            "catalog_index": np.arange(len(ordered_catalog), dtype=np.int64),
            "cdr3_aa": ordered_catalog,
        }
    )
    catalog_frame.to_csv(spec.catalog_path, index=False, compression="gzip")
    sparse.save_npz(spec.presence_path, presence, compressed=True)
    sparse.save_npz(spec.frequency_path, frequency, compressed=True)
    cache_metadata = {
        "bundle_version": BUNDLE_VERSION,
        "bundle_id": spec.bundle_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_ids": [str(value) for value in sample_ids],
        "matrix_shape": [int(shape[0]), int(shape[1])],
        "catalog_size": int(shape[1]),
        "nnz": int(presence.nnz),
        "sequence_column": spec.sequence_column,
        "catalog_is_indexing_universe_only": True,
        "public_reference_masks_training_partition_only": True,
        "holdout_labels_used_for_reference_construction": False,
        "sample_audit": sample_audit,
    }
    spec.cache_metadata_path.write_text(
        json.dumps(cache_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return cache_metadata


def _categorical_score(
    full: pd.DataFrame,
    fold: pd.DataFrame,
    columns: Iterable[str],
) -> float:
    scores = []
    for column in columns:
        levels = sorted(full[column].astype(str).unique().tolist())
        full_prop = full[column].astype(str).value_counts(normalize=True)
        fold_prop = fold[column].astype(str).value_counts(normalize=True)
        for level in levels:
            scores.append(abs(float(full_prop.get(level, 0.0)) - float(fold_prop.get(level, 0.0))))
    return max(scores, default=0.0)


def _numeric_score(
    full: pd.DataFrame,
    fold: pd.DataFrame,
    columns: Iterable[str],
) -> float:
    scores = []
    for column in columns:
        full_values = pd.to_numeric(full[column], errors="raise").to_numpy(float)
        fold_values = pd.to_numeric(fold[column], errors="raise").to_numpy(float)
        sd = float(np.std(full_values, ddof=0))
        scores.append(abs(float(np.mean(fold_values)) - float(np.mean(full_values))) / sd if sd > 0 else 0.0)
    return max(scores, default=0.0)


def generate_v2_assignments(
    spec: TrainingBundleSpec,
    assignments: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert frozen holdouts into V2 outer/inner fixed assignments."""

    outer_rows: List[Dict[str, Any]] = []
    inner_rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []

    for repeat_index, split_frame in assignments.groupby("repeat_index", sort=True):
        split_frame = split_frame.copy()
        split_id = str(split_frame["split_id"].iloc[0])
        if split_frame["split_id"].nunique() != 1:
            raise RepeatedHoldoutTrainingError(
                f"repeat_index {repeat_index} maps to multiple split IDs"
            )
        split_frame["sample_id"] = split_frame["sample_id"].astype(str)
        for row in split_frame.itertuples(index=False):
            role = str(getattr(row, "role"))
            if role == spec.split_holdout_role:
                fold = 1
            elif role == spec.split_train_role:
                fold = 2
            else:
                raise RepeatedHoldoutTrainingError(
                    f"Unexpected split role {role!r} in {split_id}"
                )
            outer_rows.append(
                {
                    "sample_id": str(getattr(row, "sample_id")),
                    "outer_repeat": int(repeat_index),
                    "outer_fold": fold,
                    "split_id": split_id,
                    "frozen_role": role,
                }
            )

        train = split_frame.loc[split_frame["role"].astype(str) == spec.split_train_role].copy()
        strata = train.loc[:, list(spec.inner_strata)].astype(str).agg("||".join, axis=1)
        counts = strata.value_counts()
        if int(counts.min()) < spec.inner_folds:
            raise RepeatedHoldoutTrainingError(
                f"{split_id} inner exact strata have minimum count {int(counts.min())}; "
                f"need >= {spec.inner_folds}. Adjust inner_cv.exact_strata explicitly."
            )

        best: Optional[Tuple[float, int, np.ndarray]] = None
        for candidate_index in range(spec.inner_candidates):
            seed = int(spec.inner_base_seed + int(repeat_index) * 100000 + candidate_index)
            splitter = StratifiedKFold(
                n_splits=spec.inner_folds,
                shuffle=True,
                random_state=seed,
            )
            fold_labels = np.zeros(len(train), dtype=int)
            candidate_scores = []
            for fold_index, (_, valid_idx) in enumerate(
                splitter.split(np.zeros(len(train)), strata.to_numpy()), start=1
            ):
                fold_labels[valid_idx] = fold_index
                valid = train.iloc[valid_idx]
                cat_score = _categorical_score(
                    train, valid, (*spec.inner_strata, *spec.balance_categorical)
                )
                num_score = _numeric_score(train, valid, spec.balance_numeric)
                candidate_scores.append(max(cat_score, num_score))
            score = float(max(candidate_scores))
            key = (score, seed)
            if best is None or key < (best[0], best[1]):
                best = (score, seed, fold_labels.copy())
        assert best is not None
        score, seed, fold_labels = best
        selected_rows.append(
            {
                "split_id": split_id,
                "outer_repeat": int(repeat_index),
                "candidate_seed": int(seed),
                "candidates_evaluated": int(spec.inner_candidates),
                "max_balance_score": float(score),
                "inner_folds": int(spec.inner_folds),
                "exact_strata": "|".join(spec.inner_strata),
            }
        )
        train = train.reset_index(drop=True)
        for index, row in train.iterrows():
            inner_rows.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "outer_repeat": int(repeat_index),
                    "outer_fold": 1,
                    "inner_fold": int(fold_labels[index]),
                    "split_id": split_id,
                }
            )
        for fold_index in range(1, spec.inner_folds + 1):
            valid = train.loc[fold_labels == fold_index]
            label_counts = valid[spec.label_column].astype(str).value_counts().to_dict()
            if len(label_counts) < 2:
                raise RepeatedHoldoutTrainingError(
                    f"{split_id} inner fold {fold_index} does not contain both labels"
                )
            audit_rows.append(
                {
                    "split_id": split_id,
                    "outer_repeat": int(repeat_index),
                    "inner_fold": fold_index,
                    "validation_samples": int(len(valid)),
                    "label_counts": json.dumps(label_counts, sort_keys=True),
                    "categorical_max_abs_difference": _categorical_score(
                        train, valid, (*spec.inner_strata, *spec.balance_categorical)
                    ),
                    "numeric_max_standardized_difference": _numeric_score(
                        train, valid, spec.balance_numeric
                    ),
                    "candidate_seed": int(seed),
                }
            )

    outer = pd.DataFrame(outer_rows).sort_values(
        ["outer_repeat", "sample_id"], kind="stable"
    )
    inner = pd.DataFrame(inner_rows).sort_values(
        ["outer_repeat", "inner_fold", "sample_id"], kind="stable"
    )
    selected = pd.DataFrame(selected_rows)
    audit = pd.DataFrame(audit_rows)
    return outer, inner, selected, audit


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def generate_scheme_yaml(
    spec: TrainingBundleSpec,
    cache_metadata: Mapping[str, Any],
    assignments: pd.DataFrame,
) -> Mapping[str, Any]:
    """Generate a concrete scheme from the Batch 02 no-clinical model template."""

    _require_yaml()
    if not spec.scheme_template.is_file():
        raise FileNotFoundError(f"Scheme template not found: {spec.scheme_template}")
    template = yaml.safe_load(spec.scheme_template.read_text(encoding="utf-8"))
    if not isinstance(template, Mapping):
        raise RepeatedHoldoutTrainingError("Scheme template root must be a mapping")
    result = copy.deepcopy(dict(template))
    scheme = result.get("scheme")
    if not isinstance(scheme, dict):
        raise RepeatedHoldoutTrainingError("Scheme template has no scheme mapping")
    scheme.update(
        {
            "id": spec.scheme_id,
            "description": (
                "Three frozen 123/51 repeated holdouts with automatic split-specific "
                "Elastic Net alpha/lambda and inner-OOF threshold selection; no M0, age, or sex."
            ),
            "status": "development",
            "output_root": spec.scheme_output_root,
        }
    )
    split_count = int(assignments["repeat_index"].nunique())
    first_counts = (
        assignments.groupby(["repeat_index", "role"]).size().unstack(fill_value=0)
    )
    train_size = int(first_counts[spec.split_train_role].iloc[0])
    holdout_size = int(first_counts[spec.split_holdout_role].iloc[0])
    if not (first_counts[spec.split_train_role] == train_size).all() or not (
        first_counts[spec.split_holdout_role] == holdout_size
    ).all():
        raise RepeatedHoldoutTrainingError("Repeated splits do not have constant train/holdout sizes")

    overrides = result.get("overrides", {})
    if not isinstance(overrides, Mapping):
        raise RepeatedHoldoutTrainingError("Scheme template overrides must be a mapping")
    injected = {
        "data": {
            "train": {
                "expected_samples": int(cache_metadata["matrix_shape"][0]),
                "metadata": _repo_relative(spec.full_metadata_path, spec.repository_root),
                "base_matrix": _repo_relative(spec.full_base_matrix_path, spec.repository_root),
                "feature_manifest": _repo_relative(spec.feature_manifest, spec.repository_root),
                "aa_clone_table_dir": _repo_relative(spec.output_root, spec.repository_root),
                "public_catalog": _repo_relative(spec.catalog_path, spec.repository_root),
                "reference_definition": _repo_relative(spec.marker_path, spec.repository_root),
            }
        },
        "public_reference": {
            "expected_sizes": {
                "catalog": int(cache_metadata["catalog_size"]),
                "global": 0,
                "RA_specific": 0,
                "ILD_specific": 0,
                "shared": 0,
            },
            "cache": {
                "presence": _repo_relative(spec.presence_path, spec.repository_root),
                "frequency": _repo_relative(spec.frequency_path, spec.repository_root),
                "metadata": _repo_relative(spec.cache_metadata_path, spec.repository_root),
            },
        },
        "cross_validation": {
            "outer_repeats": split_count,
            "outer_folds": 2,
            "executed_outer_folds": [1],
            "inner_folds": spec.inner_folds,
            "outer_assignments": _repo_relative(spec.outer_assignments_path, spec.repository_root),
            "inner_assignments": _repo_relative(spec.inner_assignments_path, spec.repository_root),
            "balance_columns": [*spec.inner_strata, *spec.balance_categorical],
        },
        "repeated_holdout_training": {
            "mode": "frozen_repeated_holdout",
            "split_set_id": str(assignments["split_set_id"].iloc[0]),
            "assignments": _repo_relative(spec.split_assignments, spec.repository_root),
            "frozen_marker": _repo_relative(spec.split_frozen_marker, spec.repository_root),
            "training_bundle_marker": _repo_relative(spec.marker_path, spec.repository_root),
            "split_count": split_count,
            "train_size": train_size,
            "holdout_size": holdout_size,
            "validation_outer_fold": 1,
            "catalog_role": "indexing_universe_only",
            "public_reference_policy": "current_training_partition_only",
        },
        "outer_tasks": {
            "default_workers": 1,
            "max_workers": 3,
            "minimum_complete_tasks_for_validation": 1,
        },
        "aggregation": {
            "expected_predictions_per_sample": split_count,
        },
        "final_model": {"status": "not_selected"},
    }
    result["overrides"] = _deep_merge(overrides, injected)
    spec.generated_scheme.parent.mkdir(parents=True, exist_ok=True)
    spec.generated_scheme.write_text(
        yaml.safe_dump(result, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return result


def prepare_training_bundle(
    spec: TrainingBundleSpec,
    *,
    dry_run: bool = False,
    replace_incomplete: bool = False,
) -> Mapping[str, Any]:
    """Prepare or inspect the full Batch 04 repeated-holdout training bundle."""

    assignments, split_marker = load_frozen_assignments(spec)
    combined, metadata, static_features = prepare_full_static_matrix(spec, assignments)
    aa_manifest = discover_aa_clone_tables(spec, combined["sample_id"].astype(str).tolist())
    summary = {
        "bundle_id": spec.bundle_id,
        "samples": int(len(combined)),
        "static_features": int(len(static_features)),
        "aa_tables": int(len(aa_manifest)),
        "split_count": int(assignments["repeat_index"].nunique()),
        "train_sizes": assignments.loc[
            assignments["role"].astype(str) == spec.split_train_role
        ].groupby("repeat_index").size().astype(int).tolist(),
        "holdout_sizes": assignments.loc[
            assignments["role"].astype(str) == spec.split_holdout_role
        ].groupby("repeat_index").size().astype(int).tolist(),
        "output_root": str(spec.output_root),
        "dry_run": bool(dry_run),
    }
    if dry_run:
        return summary

    if spec.marker_path.exists():
        raise FileExistsError(
            f"Training bundle is frozen and cannot be overwritten: {spec.output_root}"
        )
    if spec.output_root.exists() and any(spec.output_root.iterdir()):
        if not replace_incomplete:
            raise FileExistsError(
                f"Incomplete training bundle already exists: {spec.output_root}. "
                "Use --replace-incomplete only after confirming no frozen marker exists."
            )
        shutil.rmtree(spec.output_root)
    spec.output_root.mkdir(parents=True, exist_ok=True)

    combined.to_csv(spec.full_base_matrix_path, index=False, compression="gzip")
    metadata.to_csv(spec.full_metadata_path, index=False)
    aa_manifest.to_csv(spec.aa_manifest_path, index=False)
    cache_metadata = build_full_public_sparse_cache(
        spec, combined["sample_id"].astype(str).tolist(), aa_manifest
    )
    outer, inner, selected, audit = generate_v2_assignments(spec, assignments)
    outer.to_csv(spec.outer_assignments_path, index=False)
    inner.to_csv(spec.inner_assignments_path, index=False, compression="gzip")
    selected.to_csv(spec.selected_inner_candidates_path, index=False)
    audit.to_csv(spec.inner_audit_path, index=False)

    files = {
        "full_metadata": spec.full_metadata_path,
        "full_base_matrix": spec.full_base_matrix_path,
        "catalog": spec.catalog_path,
        "presence": spec.presence_path,
        "frequency": spec.frequency_path,
        "cache_metadata": spec.cache_metadata_path,
        "aa_manifest": spec.aa_manifest_path,
        "outer_assignments": spec.outer_assignments_path,
        "inner_assignments": spec.inner_assignments_path,
        "inner_audit": spec.inner_audit_path,
        "selected_inner_candidates": spec.selected_inner_candidates_path,
    }
    marker = {
        "status": "FROZEN",
        "training_bundle_version": BUNDLE_VERSION,
        "bundle_id": spec.bundle_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_spec": _repo_relative(spec.source_path, spec.repository_root),
        "split_set_id": split_marker.get("split_set_id"),
        "split_frozen_marker_sha256": _sha256_file(spec.split_frozen_marker),
        "samples": int(len(combined)),
        "static_feature_count": int(len(static_features)),
        "catalog_size": int(cache_metadata["catalog_size"]),
        "matrix_nnz": int(cache_metadata["nnz"]),
        "split_count": int(assignments["repeat_index"].nunique()),
        "train_sizes": summary["train_sizes"],
        "holdout_sizes": summary["holdout_sizes"],
        "scientific_engine_changed": False,
        "public_reference_training_partition_only": True,
        "automatic_alpha_lambda_selection": True,
        "inner_oof_threshold_selection": True,
        "files": {
            name: {
                "path": _repo_relative(path, spec.repository_root),
                "sha256": _sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
            for name, path in files.items()
        },
    }
    spec.marker_path.write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    generate_scheme_yaml(spec, cache_metadata, assignments)
    marker["generated_scheme"] = _repo_relative(spec.generated_scheme, spec.repository_root)
    marker["generated_scheme_sha256"] = _sha256_file(spec.generated_scheme)
    spec.marker_path.write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return marker
