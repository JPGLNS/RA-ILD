#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare frozen metadata-defined analysis subcohorts for TRB modeling.

Batch 12 is deliberately additive.  It does not modify the existing repeated-
holdout, training-bundle, nested-CV, public-reference, 3-mer, aggregation, or
visualization engines.  Instead, it creates ordinary metadata and matrix inputs
that satisfy the already existing Batch 03/04 contracts.

The primary use case is material-specific analysis (PBMC-only or
buffercoat-only), but the file format is intentionally generic enough to allow
one or more accepted values from a configured metadata column.
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

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


SUBSET_VERSION = "1.0"
SUBSET_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")
DEFAULT_OUTPUT_ROOT = Path("TRB/set/cohort_subsets")


class CohortSubsetError(ValueError):
    """Raised when a subcohort cannot be prepared reproducibly and safely."""


@dataclass(frozen=True)
class CohortSubsetSpec:
    source_path: Path
    repository_root: Path
    raw: Mapping[str, Any]
    subset_id: str
    description: str
    output_root: Path
    metadata_files: Tuple[Path, ...]
    train_base_matrix: Path
    test_final_matrix: Path
    metadata_sample_id_column: str
    matrix_sample_id_column: str
    patient_id_column: str
    label_column: str
    filter_column: str
    include_values: Tuple[str, ...]
    case_insensitive: bool
    strip_whitespace: bool
    require_one_sample_per_patient: bool
    expected_samples: int
    expected_patients: int
    expected_label_counts: Mapping[str, int]
    expected_filter_counts: Mapping[str, int]

    @property
    def metadata_output(self) -> Path:
        return self.output_root / "subset_metadata.csv"

    @property
    def train_matrix_output(self) -> Path:
        return self.output_root / "subset_train_base_matrix.csv.gz"

    @property
    def test_matrix_output(self) -> Path:
        return self.output_root / "subset_test_final_matrix.csv.gz"

    @property
    def sample_audit_output(self) -> Path:
        return self.output_root / "subset_sample_audit.csv"

    @property
    def summary_output(self) -> Path:
        return self.output_root / "subset_summary.json"

    @property
    def resolved_spec_output(self) -> Path:
        return self.output_root / "subset_resolved_spec.yaml"

    @property
    def marker_output(self) -> Path:
        return self.output_root / "SUBSET_FROZEN.json"


@dataclass(frozen=True)
class CohortSubsetResult:
    metadata: pd.DataFrame
    train_matrix: pd.DataFrame
    test_matrix: pd.DataFrame
    sample_audit: pd.DataFrame
    summary: Mapping[str, Any]


def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: mamba install -c conda-forge pyyaml")


def _mapping(parent: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise CohortSubsetError(f"{context}.{key} must be a mapping")
    return value


def _string(parent: Mapping[str, Any], key: str, context: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CohortSubsetError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _bool(parent: Mapping[str, Any], key: str, context: str, default: bool) -> bool:
    value = parent.get(key, default)
    if not isinstance(value, bool):
        raise CohortSubsetError(f"{context}.{key} must be boolean")
    return value


def _positive_int(parent: Mapping[str, Any], key: str, context: str) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CohortSubsetError(f"{context}.{key} must be an integer >= 1")
    return int(value)


def _string_list(value: Any, context: str) -> Tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CohortSubsetError(f"{context} must be a non-empty list of strings")
    out: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise CohortSubsetError(f"{context} must contain non-empty strings")
        out.append(item.strip())
    if not out:
        raise CohortSubsetError(f"{context} must not be empty")
    if len(out) != len(set(out)):
        raise CohortSubsetError(f"{context} contains duplicated values")
    return tuple(out)


def _count_mapping(value: Any, context: str) -> Mapping[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise CohortSubsetError(f"{context} must be a mapping of value -> count")
    out: Dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str) or not key.strip():
            raise CohortSubsetError(f"{context} keys must be non-empty strings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise CohortSubsetError(f"{context}.{key} must be an integer >= 0")
        out[key.strip()] = int(count)
    return out


def _resolve_repo_path(value: str, repository_root: Path, context: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        resolved = path.expanduser().resolve()
    else:
        resolved = (repository_root / path).resolve()
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError as exc:
        raise CohortSubsetError(f"{context} must remain inside repository root: {resolved}") from exc
    return resolved


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, repository_root: Path) -> Mapping[str, Any]:
    resolved = Path(path).resolve()
    try:
        display = resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        display = str(resolved)
    return {
        "path": display,
        "sha256": _sha256_file(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    frame = pd.read_csv(path)
    frame = frame.drop(
        columns=[column for column in frame.columns if str(column).startswith("Unnamed:")],
        errors="ignore",
    )
    duplicated = frame.columns[frame.columns.duplicated()].astype(str).tolist()
    if duplicated:
        raise CohortSubsetError(f"{label} contains duplicated columns: {duplicated[:20]}")
    return frame


def _normalize_value(value: Any, *, strip_whitespace: bool, case_insensitive: bool) -> str:
    if pd.isna(value):
        raise CohortSubsetError("Filter values cannot be missing")
    text = str(value)
    if strip_whitespace:
        text = text.strip()
    if not text:
        raise CohortSubsetError("Filter values cannot be blank")
    return text.casefold() if case_insensitive else text


def _normalize_series(
    series: pd.Series,
    *,
    strip_whitespace: bool,
    case_insensitive: bool,
    label: str,
) -> pd.Series:
    if series.isna().any():
        raise CohortSubsetError(f"{label} contains missing values")
    values = series.astype(str)
    if strip_whitespace:
        values = values.str.strip()
    if values.eq("").any():
        raise CohortSubsetError(f"{label} contains blank values")
    return values.str.casefold() if case_insensitive else values


def load_cohort_subset_spec(
    path: Path,
    *,
    repository_root: Path,
) -> CohortSubsetSpec:
    """Load a Batch 12 material-subset YAML specification."""
    _require_yaml()
    source_path = Path(path).expanduser().resolve()
    root = Path(repository_root).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Cohort-subset specification not found: {source_path}")
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise CohortSubsetError("Subset YAML root must be a mapping")
    if str(raw.get("material_subset_version")) != SUBSET_VERSION:
        raise CohortSubsetError(
            f"material_subset_version must be {SUBSET_VERSION!r}"
        )

    subset = _mapping(raw, "subset", "root")
    subset_id = _string(subset, "id", "subset")
    if not SUBSET_ID_PATTERN.fullmatch(subset_id):
        raise CohortSubsetError(
            "subset.id must be 3-80 lowercase characters using only a-z, 0-9, '.', '_' or '-'"
        )
    description = str(subset.get("description", "")).strip()
    output_value = str(subset.get("output_root", DEFAULT_OUTPUT_ROOT / subset_id))
    output_root = _resolve_repo_path(output_value, root, "subset.output_root")
    allowed_root = (root / DEFAULT_OUTPUT_ROOT).resolve()
    if not _is_within(output_root, allowed_root):
        raise CohortSubsetError(
            f"subset.output_root must be under {DEFAULT_OUTPUT_ROOT.as_posix()}"
        )
    if output_root.name != subset_id:
        raise CohortSubsetError("subset.output_root must end with the exact subset.id")

    source = _mapping(raw, "source", "root")
    metadata_values = source.get("metadata_files")
    if not isinstance(metadata_values, Sequence) or isinstance(metadata_values, (str, bytes)):
        raise CohortSubsetError("source.metadata_files must be a non-empty list")
    metadata_files: List[Path] = []
    for index, value in enumerate(metadata_values):
        if not isinstance(value, str) or not value.strip():
            raise CohortSubsetError(
                f"source.metadata_files[{index}] must be a non-empty path"
            )
        metadata_files.append(
            _resolve_repo_path(value.strip(), root, f"source.metadata_files[{index}]")
        )
    if not metadata_files:
        raise CohortSubsetError("source.metadata_files must not be empty")

    filter_section = _mapping(raw, "filter", "root")
    expected = _mapping(raw, "expected", "root")
    return CohortSubsetSpec(
        source_path=source_path,
        repository_root=root,
        raw=raw,
        subset_id=subset_id,
        description=description,
        output_root=output_root,
        metadata_files=tuple(metadata_files),
        train_base_matrix=_resolve_repo_path(
            _string(source, "train_base_matrix", "source"), root, "source.train_base_matrix"
        ),
        test_final_matrix=_resolve_repo_path(
            _string(source, "test_final_matrix", "source"), root, "source.test_final_matrix"
        ),
        metadata_sample_id_column=_string(
            source, "metadata_sample_id_column", "source"
        ),
        matrix_sample_id_column=_string(source, "matrix_sample_id_column", "source"),
        patient_id_column=_string(source, "patient_id_column", "source"),
        label_column=_string(source, "label_column", "source"),
        filter_column=_string(filter_section, "column", "filter"),
        include_values=_string_list(filter_section.get("include_values"), "filter.include_values"),
        case_insensitive=_bool(filter_section, "case_insensitive", "filter", True),
        strip_whitespace=_bool(filter_section, "strip_whitespace", "filter", True),
        require_one_sample_per_patient=_bool(
            source, "require_one_sample_per_patient", "source", True
        ),
        expected_samples=_positive_int(expected, "samples", "expected"),
        expected_patients=_positive_int(expected, "patients", "expected"),
        expected_label_counts=_count_mapping(
            expected.get("label_counts"), "expected.label_counts"
        ),
        expected_filter_counts=_count_mapping(
            expected.get("filter_counts"), "expected.filter_counts"
        ),
    )


def _validate_identifiers(
    frame: pd.DataFrame,
    *,
    sample_column: str,
    patient_column: Optional[str],
    label: str,
) -> None:
    required = [sample_column] + ([patient_column] if patient_column else [])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise CohortSubsetError(f"{label} is missing identifier columns: {missing}")
    for column in required:
        if frame[column].isna().any():
            raise CohortSubsetError(f"{label}.{column} contains missing values")
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise CohortSubsetError(f"{label}.{column} contains blank values")
    duplicates = frame.loc[
        frame[sample_column].duplicated(keep=False), sample_column
    ].astype(str).unique().tolist()
    if duplicates:
        raise CohortSubsetError(
            f"{label} contains duplicate sample IDs: {duplicates[:20]}"
        )


def _validate_expected_counts(
    metadata: pd.DataFrame,
    spec: CohortSubsetSpec,
) -> None:
    if len(metadata) != spec.expected_samples:
        raise CohortSubsetError(
            f"Expected {spec.expected_samples} selected samples, observed {len(metadata)}"
        )
    patient_count = int(metadata[spec.patient_id_column].nunique())
    if patient_count != spec.expected_patients:
        raise CohortSubsetError(
            f"Expected {spec.expected_patients} selected patients, observed {patient_count}"
        )
    if spec.require_one_sample_per_patient:
        counts = metadata.groupby(spec.patient_id_column, sort=False).size()
        if (counts != 1).any():
            raise CohortSubsetError(
                "Selected cohort requires one sample per patient; violations="
                f"{counts[counts != 1].head(20).to_dict()}"
            )

    observed_labels = metadata[spec.label_column].astype(str).value_counts().to_dict()
    if spec.expected_label_counts and observed_labels != dict(spec.expected_label_counts):
        raise CohortSubsetError(
            "Selected label counts do not match expected.label_counts; "
            f"expected={dict(spec.expected_label_counts)}, observed={observed_labels}"
        )

    observed_filters = metadata[spec.filter_column].astype(str).value_counts().to_dict()
    if spec.expected_filter_counts and observed_filters != dict(spec.expected_filter_counts):
        raise CohortSubsetError(
            "Selected filter counts do not match expected.filter_counts; "
            f"expected={dict(spec.expected_filter_counts)}, observed={observed_filters}"
        )


def _check_matrix_against_metadata(
    matrix: pd.DataFrame,
    metadata_lookup: pd.DataFrame,
    *,
    spec: CohortSubsetSpec,
    label: str,
) -> None:
    required = {
        spec.matrix_sample_id_column,
        spec.filter_column,
        spec.label_column,
    }
    missing = sorted(required - set(matrix.columns))
    if missing:
        raise CohortSubsetError(f"{label} is missing required columns: {missing}")
    _validate_identifiers(
        matrix,
        sample_column=spec.matrix_sample_id_column,
        patient_column=None,
        label=label,
    )
    ids = matrix[spec.matrix_sample_id_column].astype(str)
    if not set(ids).issubset(set(metadata_lookup.index.astype(str))):
        extra = sorted(set(ids) - set(metadata_lookup.index.astype(str)))
        raise CohortSubsetError(f"{label} contains samples absent from metadata: {extra[:20]}")

    aligned = metadata_lookup.loc[ids]
    matrix_filter = _normalize_series(
        matrix[spec.filter_column],
        strip_whitespace=spec.strip_whitespace,
        case_insensitive=spec.case_insensitive,
        label=f"{label}.{spec.filter_column}",
    ).to_numpy()
    metadata_filter = _normalize_series(
        aligned[spec.filter_column],
        strip_whitespace=spec.strip_whitespace,
        case_insensitive=spec.case_insensitive,
        label=f"metadata.{spec.filter_column}",
    ).to_numpy()
    if not np.array_equal(matrix_filter, metadata_filter):
        mismatch = ids[np.asarray(matrix_filter) != np.asarray(metadata_filter)].head(20).tolist()
        raise CohortSubsetError(
            f"{label} filter values disagree with metadata for samples: {mismatch}"
        )
    matrix_labels = matrix[spec.label_column].astype(str).str.strip().to_numpy()
    metadata_labels = aligned[spec.label_column].astype(str).str.strip().to_numpy()
    if not np.array_equal(matrix_labels, metadata_labels):
        mismatch = ids[np.asarray(matrix_labels) != np.asarray(metadata_labels)].head(20).tolist()
        raise CohortSubsetError(
            f"{label} labels disagree with metadata for samples: {mismatch}"
        )


def prepare_cohort_subset(spec: CohortSubsetSpec) -> CohortSubsetResult:
    """Filter metadata and source matrices while preserving source row order."""
    metadata_frames: List[pd.DataFrame] = []
    for index, path in enumerate(spec.metadata_files, start=1):
        frame = _read_csv(path, f"metadata file {index}")
        frame = frame.copy()
        frame["_batch12_source_metadata_index"] = int(index)
        frame["_batch12_source_metadata_path"] = str(path)
        frame["_batch12_source_row"] = np.arange(1, len(frame) + 1, dtype=int)
        metadata_frames.append(frame)
    metadata = pd.concat(metadata_frames, ignore_index=True, sort=False)

    required_metadata = {
        spec.metadata_sample_id_column,
        spec.patient_id_column,
        spec.label_column,
        spec.filter_column,
    }
    missing_metadata = sorted(required_metadata - set(metadata.columns))
    if missing_metadata:
        raise CohortSubsetError(
            f"Combined metadata is missing required columns: {missing_metadata}"
        )
    _validate_identifiers(
        metadata,
        sample_column=spec.metadata_sample_id_column,
        patient_column=spec.patient_id_column,
        label="combined metadata",
    )
    if metadata[spec.label_column].isna().any():
        raise CohortSubsetError(f"metadata.{spec.label_column} contains missing values")
    metadata[spec.label_column] = metadata[spec.label_column].astype(str).str.strip()
    if metadata[spec.label_column].eq("").any():
        raise CohortSubsetError(f"metadata.{spec.label_column} contains blank values")

    normalized = _normalize_series(
        metadata[spec.filter_column],
        strip_whitespace=spec.strip_whitespace,
        case_insensitive=spec.case_insensitive,
        label=f"metadata.{spec.filter_column}",
    )
    accepted = {
        _normalize_value(
            value,
            strip_whitespace=spec.strip_whitespace,
            case_insensitive=spec.case_insensitive,
        )
        for value in spec.include_values
    }
    observed_values = sorted(set(normalized.tolist()))
    unknown_requested = sorted(accepted - set(observed_values))
    if unknown_requested:
        raise CohortSubsetError(
            "Requested filter values were not observed in metadata: "
            f"{unknown_requested}; observed={observed_values}"
        )
    selected = metadata.loc[normalized.isin(accepted)].copy()
    if selected.empty:
        raise CohortSubsetError("The configured filter selected zero samples")
    _validate_expected_counts(selected, spec)

    train = _read_csv(spec.train_base_matrix, "source training base matrix")
    test = _read_csv(spec.test_final_matrix, "source test final matrix")
    metadata_lookup = metadata.set_index(spec.metadata_sample_id_column, drop=False)
    _check_matrix_against_metadata(
        train, metadata_lookup, spec=spec, label="source training base matrix"
    )
    _check_matrix_against_metadata(
        test, metadata_lookup, spec=spec, label="source test final matrix"
    )

    train_ids = set(train[spec.matrix_sample_id_column].astype(str))
    test_ids = set(test[spec.matrix_sample_id_column].astype(str))
    overlap = sorted(train_ids & test_ids)
    if overlap:
        raise CohortSubsetError(
            f"Source train/test matrices overlap in sample IDs: {overlap[:20]}"
        )
    metadata_ids = set(metadata[spec.metadata_sample_id_column].astype(str))
    matrix_ids = train_ids | test_ids
    if matrix_ids != metadata_ids:
        missing = sorted(metadata_ids - matrix_ids)
        extra = sorted(matrix_ids - metadata_ids)
        raise CohortSubsetError(
            "Full source metadata/matrix sample mismatch; "
            f"missing_from_matrices={missing[:20]}, extra_in_matrices={extra[:20]}"
        )

    selected_ids_order = selected[spec.metadata_sample_id_column].astype(str).tolist()
    selected_set = set(selected_ids_order)
    selected_train = train.loc[
        train[spec.matrix_sample_id_column].astype(str).isin(selected_set)
    ].copy()
    selected_test = test.loc[
        test[spec.matrix_sample_id_column].astype(str).isin(selected_set)
    ].copy()
    output_ids = set(selected_train[spec.matrix_sample_id_column].astype(str)) | set(
        selected_test[spec.matrix_sample_id_column].astype(str)
    )
    if output_ids != selected_set:
        missing = sorted(selected_set - output_ids)
        extra = sorted(output_ids - selected_set)
        raise CohortSubsetError(
            "Selected matrix sample mismatch; "
            f"missing={missing[:20]}, extra={extra[:20]}"
        )

    source_partition = {
        sample_id: "old_train"
        for sample_id in selected_train[spec.matrix_sample_id_column].astype(str)
    }
    source_partition.update(
        {
            sample_id: "old_test"
            for sample_id in selected_test[spec.matrix_sample_id_column].astype(str)
        }
    )
    sample_audit = selected[
        [
            spec.metadata_sample_id_column,
            spec.patient_id_column,
            spec.label_column,
            spec.filter_column,
            "_batch12_source_metadata_index",
            "_batch12_source_metadata_path",
            "_batch12_source_row",
        ]
    ].copy()
    sample_audit = sample_audit.rename(
        columns={spec.metadata_sample_id_column: "sample_id"}
    )
    sample_audit.insert(
        1,
        "source_matrix_partition",
        sample_audit["sample_id"].astype(str).map(source_partition),
    )
    if sample_audit["source_matrix_partition"].isna().any():
        raise CohortSubsetError("Unable to assign a source matrix partition to every sample")

    clean_metadata = selected.drop(
        columns=[
            "_batch12_source_metadata_index",
            "_batch12_source_metadata_path",
            "_batch12_source_row",
        ],
        errors="ignore",
    ).reset_index(drop=True)

    summary: Dict[str, Any] = {
        "material_subset_version": SUBSET_VERSION,
        "subset_id": spec.subset_id,
        "description": spec.description,
        "filter": {
            "column": spec.filter_column,
            "include_values": list(spec.include_values),
            "case_insensitive": spec.case_insensitive,
            "strip_whitespace": spec.strip_whitespace,
        },
        "selected_samples": int(len(clean_metadata)),
        "selected_patients": int(clean_metadata[spec.patient_id_column].nunique()),
        "label_counts": {
            str(key): int(value)
            for key, value in clean_metadata[spec.label_column].astype(str).value_counts().items()
        },
        "filter_counts": {
            str(key): int(value)
            for key, value in clean_metadata[spec.filter_column].astype(str).value_counts().items()
        },
        "source_partition_counts": {
            str(key): int(value)
            for key, value in sample_audit["source_matrix_partition"].value_counts().items()
        },
        "train_matrix_shape": [int(selected_train.shape[0]), int(selected_train.shape[1])],
        "test_matrix_shape": [int(selected_test.shape[0]), int(selected_test.shape[1])],
        "sample_order_sha256": hashlib.sha256(
            "\n".join(selected_ids_order).encode("utf-8")
        ).hexdigest(),
        "existing_modeling_code_modified": False,
        "downstream_contract": (
            "Use subset_metadata.csv as the sole metadata input for the existing "
            "repeated-holdout split generator, and use the two subset matrices in "
            "the existing repeated-holdout training-bundle specification."
        ),
    }
    return CohortSubsetResult(
        metadata=clean_metadata,
        train_matrix=selected_train.reset_index(drop=True),
        test_matrix=selected_test.reset_index(drop=True),
        sample_audit=sample_audit.reset_index(drop=True),
        summary=summary,
    )


def _write_csv_atomic(frame: pd.DataFrame, path: Path, *, compression: Optional[str] = None) -> None:
    temp = path.with_name(path.name + ".tmp")
    frame.to_csv(temp, index=False, compression=compression)
    temp.replace(path)


def _write_text_atomic(text: str, path: Path) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def write_frozen_cohort_subset(
    result: CohortSubsetResult,
    spec: CohortSubsetSpec,
    *,
    replace_incomplete: bool = False,
) -> Mapping[str, Any]:
    """Write subset outputs and a final immutable SHA256 marker."""
    output = spec.output_root
    if spec.marker_output.exists():
        raise FileExistsError(
            f"Frozen subset already exists and is immutable: {spec.marker_output}"
        )
    if output.exists():
        existing = list(output.iterdir())
        if existing and not replace_incomplete:
            raise FileExistsError(
                "Subset output directory is non-empty but not frozen. Use "
                "--replace-incomplete only after reviewing the partial directory: "
                f"{output}"
            )
        if existing:
            shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    _write_csv_atomic(result.metadata, spec.metadata_output)
    _write_csv_atomic(result.train_matrix, spec.train_matrix_output, compression="gzip")
    _write_csv_atomic(result.test_matrix, spec.test_matrix_output, compression="gzip")
    _write_csv_atomic(result.sample_audit, spec.sample_audit_output)
    _write_text_atomic(
        json.dumps(result.summary, ensure_ascii=False, indent=2) + "\n",
        spec.summary_output,
    )
    resolved_payload = dict(spec.raw)
    resolved_payload["resolved"] = {
        "source_spec": str(spec.source_path),
        "repository_root": str(spec.repository_root),
        "output_root": str(spec.output_root),
        "selected_samples": int(len(result.metadata)),
        "selected_patients": int(result.metadata[spec.patient_id_column].nunique()),
    }
    _write_text_atomic(
        yaml.safe_dump(resolved_payload, sort_keys=False, allow_unicode=True),
        spec.resolved_spec_output,
    )

    outputs = {
        "metadata": spec.metadata_output,
        "train_matrix": spec.train_matrix_output,
        "test_matrix": spec.test_matrix_output,
        "sample_audit": spec.sample_audit_output,
        "summary": spec.summary_output,
        "resolved_spec": spec.resolved_spec_output,
    }
    sources = {
        "metadata_files": [
            _file_record(path, spec.repository_root) for path in spec.metadata_files
        ],
        "train_base_matrix": _file_record(spec.train_base_matrix, spec.repository_root),
        "test_final_matrix": _file_record(spec.test_final_matrix, spec.repository_root),
        "subset_spec": _file_record(spec.source_path, spec.repository_root),
    }
    marker: Dict[str, Any] = {
        "status": "FROZEN",
        "material_subset_version": SUBSET_VERSION,
        "subset_id": spec.subset_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_samples": int(len(result.metadata)),
        "selected_patients": int(result.metadata[spec.patient_id_column].nunique()),
        "existing_modeling_code_modified": False,
        "sources": sources,
        "files": {
            key: _file_record(path, spec.repository_root)
            for key, path in outputs.items()
        },
    }
    _write_text_atomic(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n",
        spec.marker_output,
    )
    return marker


def verify_frozen_subset(
    marker_path: Path,
    *,
    repository_root: Path,
) -> Mapping[str, Any]:
    """Verify hashes recorded by a Batch 12 frozen marker."""
    path = Path(marker_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Subset marker not found: {path}")
    marker = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(marker, Mapping) or marker.get("status") != "FROZEN":
        raise CohortSubsetError("Subset marker is not FROZEN")
    files = marker.get("files")
    if not isinstance(files, Mapping) or not files:
        raise CohortSubsetError("Subset marker contains no output file records")
    root = Path(repository_root).expanduser().resolve()
    for key, record in files.items():
        if not isinstance(record, Mapping):
            raise CohortSubsetError(f"Invalid marker record for {key}")
        file_path = _resolve_repo_path(str(record.get("path", "")), root, f"marker.files.{key}")
        if not file_path.is_file():
            raise FileNotFoundError(f"Frozen subset output is missing: {file_path}")
        observed = _sha256_file(file_path)
        if observed != str(record.get("sha256")):
            raise CohortSubsetError(
                f"Frozen subset hash mismatch for {key}: expected={record.get('sha256')}, observed={observed}"
            )
    return marker
