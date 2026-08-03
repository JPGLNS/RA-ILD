#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare frozen metadata-defined analysis subcohorts for IGH modeling.

Batch 11B is additive. It filters the complete IGH source cohort before split
creation, public-reference construction, repeat-specific 3-mer selection, or
model fitting. Existing full-cohort inputs and results are never modified.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

SUBSET_VERSION = "1.0"
SUBSET_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")
DEFAULT_OUTPUT_ROOT = Path("IGH/set/cohort_subsets")

class CohortSubsetError(ValueError):
    """Raised when a material subset cannot be prepared safely."""

@dataclass(frozen=True)
class CohortSubsetSpec:
    source_path: Path
    repository_root: Path
    raw: Mapping[str, Any]
    subset_id: str
    description: str
    output_root: Path
    metadata_files: Tuple[Path, Path]
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
        raise RuntimeError("PyYAML is required: mamba install -c conda-forge pyyaml")

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
    out = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise CohortSubsetError(f"{context} contains an invalid value")
        out.append(item.strip())
    if not out or len(out) != len(set(out)):
        raise CohortSubsetError(f"{context} must be non-empty and unique")
    return tuple(out)
def _count_mapping(value: Any, context: str) -> Mapping[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise CohortSubsetError(f"{context} must be a mapping")
    out: Dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str) or not key.strip():
            raise CohortSubsetError(f"{context} keys must be non-empty strings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise CohortSubsetError(f"{context}.{key} must be an integer >= 0")
        out[key.strip()] = int(count)
    return out

def _resolve(value: str, root: Path, context: str) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise CohortSubsetError(f"{context} must remain inside repository root") from exc
    return resolved

def _within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
def _record(path: Path, root: Path) -> Mapping[str, Any]:
    try:
        display = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        display = str(path.resolve())
    return {"path": display, "sha256": _sha256(path), "size_bytes": path.stat().st_size}
def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    frame = pd.read_csv(path)
    frame = frame.drop(columns=[c for c in frame.columns if str(c).startswith('Unnamed:')], errors='ignore')
    if frame.columns.duplicated().any():
        raise CohortSubsetError(f"{label} contains duplicate columns")
    return frame

def _normalize_series(series: pd.Series, *, strip: bool, fold: bool, label: str) -> pd.Series:
    if series.isna().any():
        raise CohortSubsetError(f"{label} contains missing values")
    out = series.astype(str)
    if strip:
        out = out.str.strip()
    if out.eq('').any():
        raise CohortSubsetError(f"{label} contains blank values")
    return out.str.casefold() if fold else out

def load_cohort_subset_spec(path: Path, *, repository_root: Path) -> CohortSubsetSpec:
    _require_yaml()
    source_path = Path(path).expanduser().resolve()
    root = Path(repository_root).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    raw = yaml.safe_load(source_path.read_text(encoding='utf-8'))
    if not isinstance(raw, Mapping):
        raise CohortSubsetError("Subset YAML root must be a mapping")
    if str(raw.get('material_subset_version')) != SUBSET_VERSION:
        raise CohortSubsetError(f"material_subset_version must be {SUBSET_VERSION!r}")
    subset = _mapping(raw, 'subset', 'root')
    subset_id = _string(subset, 'id', 'subset')
    if not SUBSET_ID_PATTERN.fullmatch(subset_id):
        raise CohortSubsetError("subset.id has invalid characters")
    output_root = _resolve(str(subset.get('output_root', DEFAULT_OUTPUT_ROOT / subset_id)), root, 'subset.output_root')
    allowed = (root / DEFAULT_OUTPUT_ROOT).resolve()
    if not _within(output_root, allowed) or output_root.name != subset_id:
        raise CohortSubsetError(f"subset.output_root must be under {DEFAULT_OUTPUT_ROOT} and end with subset.id")
    source = _mapping(raw, 'source', 'root')
    metadata_values = source.get('metadata_files')
    if not isinstance(metadata_values, Sequence) or isinstance(metadata_values, (str, bytes)) or len(metadata_values) != 2:
        raise CohortSubsetError("source.metadata_files must contain exactly train and test metadata paths")
    metadata_files = tuple(_resolve(str(v), root, f'source.metadata_files[{i}]') for i, v in enumerate(metadata_values))
    filt = _mapping(raw, 'filter', 'root')
    expected = _mapping(raw, 'expected', 'root')
    return CohortSubsetSpec(
        source_path=source_path,
        repository_root=root,
        raw=raw,
        subset_id=subset_id,
        description=str(subset.get('description', '')).strip(),
        output_root=output_root,
        metadata_files=(metadata_files[0], metadata_files[1]),
        train_base_matrix=_resolve(_string(source, 'train_base_matrix', 'source'), root, 'source.train_base_matrix'),
        test_final_matrix=_resolve(_string(source, 'test_final_matrix', 'source'), root, 'source.test_final_matrix'),
        metadata_sample_id_column=_string(source, 'metadata_sample_id_column', 'source'),
        matrix_sample_id_column=_string(source, 'matrix_sample_id_column', 'source'),
        patient_id_column=_string(source, 'patient_id_column', 'source'),
        label_column=_string(source, 'label_column', 'source'),
        filter_column=_string(filt, 'column', 'filter'),
        include_values=_string_list(filt.get('include_values'), 'filter.include_values'),
        case_insensitive=_bool(filt, 'case_insensitive', 'filter', True),
        strip_whitespace=_bool(filt, 'strip_whitespace', 'filter', True),
        require_one_sample_per_patient=_bool(source, 'require_one_sample_per_patient', 'source', True),
        expected_samples=_positive_int(expected, 'samples', 'expected'),
        expected_patients=_positive_int(expected, 'patients', 'expected'),
        expected_label_counts=_count_mapping(expected.get('label_counts'), 'expected.label_counts'),
        expected_filter_counts=_count_mapping(expected.get('filter_counts'), 'expected.filter_counts'),
    )

def _validate_ids(frame: pd.DataFrame, sample: str, patient: Optional[str], label: str) -> None:
    cols = [sample] + ([patient] if patient else [])
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise CohortSubsetError(f"{label} missing identifiers: {missing}")
    for c in cols:
        if frame[c].isna().any():
            raise CohortSubsetError(f"{label}.{c} contains missing values")
        frame[c] = frame[c].astype(str).str.strip()
        if frame[c].eq('').any():
            raise CohortSubsetError(f"{label}.{c} contains blank values")
    if frame[sample].duplicated().any():
        raise CohortSubsetError(f"{label} contains duplicate sample IDs")

def _validate_pair(metadata: pd.DataFrame, matrix: pd.DataFrame, spec: CohortSubsetSpec, label: str) -> None:
    _validate_ids(metadata, spec.metadata_sample_id_column, spec.patient_id_column, f'{label} metadata')
    _validate_ids(matrix, spec.matrix_sample_id_column, None, f'{label} matrix')
    m_ids = set(metadata[spec.metadata_sample_id_column])
    x_ids = set(matrix[spec.matrix_sample_id_column])
    if m_ids != x_ids:
        raise CohortSubsetError(f"{label} metadata/matrix sample mismatch: metadata_only={sorted(m_ids-x_ids)[:10]}, matrix_only={sorted(x_ids-m_ids)[:10]}")
    lookup = metadata.set_index(spec.metadata_sample_id_column)
    indexed = matrix.set_index(spec.matrix_sample_id_column)
    for column in (spec.label_column, spec.filter_column):
        if column in indexed.columns:
            left = lookup.loc[indexed.index, column].astype(str).str.strip()
            right = indexed[column].astype(str).str.strip()
            bad = left.ne(right)
            if bad.any():
                raise CohortSubsetError(f"{label} metadata/matrix disagree for {column}: {bad[bad].index.tolist()[:10]}")

def prepare_cohort_subset(spec: CohortSubsetSpec) -> CohortSubsetResult:
    train_meta = _read_csv(spec.metadata_files[0], 'train metadata')
    test_meta = _read_csv(spec.metadata_files[1], 'test metadata')
    train_matrix = _read_csv(spec.train_base_matrix, 'train base matrix')
    test_matrix = _read_csv(spec.test_final_matrix, 'test final matrix')
    _validate_pair(train_meta, train_matrix, spec, 'train')
    _validate_pair(test_meta, test_matrix, spec, 'test')
    combined = pd.concat([
        train_meta.assign(_source_partition='train'),
        test_meta.assign(_source_partition='test'),
    ], ignore_index=True)
    _validate_ids(combined, spec.metadata_sample_id_column, spec.patient_id_column, 'combined metadata')
    if spec.require_one_sample_per_patient and combined[spec.patient_id_column].duplicated().any():
        raise CohortSubsetError("Complete metadata violates one-sample-per-patient contract")
    for required in (spec.label_column, spec.filter_column):
        if required not in combined.columns:
            raise CohortSubsetError(f"metadata missing required column: {required}")
    normalized = _normalize_series(combined[spec.filter_column], strip=spec.strip_whitespace, fold=spec.case_insensitive, label=spec.filter_column)
    accepted = {v.strip().casefold() if spec.case_insensitive else v.strip() for v in spec.include_values}
    selected_mask = normalized.isin(accepted)
    selected = combined.loc[selected_mask].copy()
    if len(selected) != spec.expected_samples:
        raise CohortSubsetError(f"Expected {spec.expected_samples} selected samples, observed {len(selected)}")
    patients = selected[spec.patient_id_column].nunique()
    if patients != spec.expected_patients:
        raise CohortSubsetError(f"Expected {spec.expected_patients} selected patients, observed {patients}")
    if spec.require_one_sample_per_patient and selected[spec.patient_id_column].duplicated().any():
        raise CohortSubsetError("Selected cohort has duplicate patients")
    labels = selected[spec.label_column].astype(str).value_counts().to_dict()
    if spec.expected_label_counts and labels != dict(spec.expected_label_counts):
        raise CohortSubsetError(f"Label count mismatch: expected={dict(spec.expected_label_counts)}, observed={labels}")
    filters = selected[spec.filter_column].astype(str).value_counts().to_dict()
    if spec.expected_filter_counts and filters != dict(spec.expected_filter_counts):
        raise CohortSubsetError(f"Filter count mismatch: expected={dict(spec.expected_filter_counts)}, observed={filters}")
    train_ids = set(selected.loc[selected['_source_partition'].eq('train'), spec.metadata_sample_id_column])
    test_ids = set(selected.loc[selected['_source_partition'].eq('test'), spec.metadata_sample_id_column])
    train_out = train_matrix.loc[train_matrix[spec.matrix_sample_id_column].astype(str).isin(train_ids)].copy()
    test_out = test_matrix.loc[test_matrix[spec.matrix_sample_id_column].astype(str).isin(test_ids)].copy()
    if len(train_out) + len(test_out) != len(selected):
        raise CohortSubsetError("Selected matrix row count does not match selected metadata")
    audit = combined[[spec.metadata_sample_id_column, spec.patient_id_column, spec.label_column, spec.filter_column, '_source_partition']].copy()
    audit['selected'] = selected_mask.to_numpy(bool)
    summary = {
        'subset_id': spec.subset_id,
        'samples': int(len(selected)),
        'patients': int(patients),
        'train_partition_samples': int(len(train_out)),
        'test_partition_samples': int(len(test_out)),
        'label_counts': {str(k): int(v) for k, v in labels.items()},
        'filter_counts': {str(k): int(v) for k, v in filters.items()},
        'filter_column': spec.filter_column,
        'include_values': list(spec.include_values),
        'source_data_modified': False,
    }
    return CohortSubsetResult(
        metadata=selected.drop(columns=['_source_partition']).reset_index(drop=True),
        train_matrix=train_out.reset_index(drop=True),
        test_matrix=test_out.reset_index(drop=True),
        sample_audit=audit.reset_index(drop=True),
        summary=summary,
    )

def write_cohort_subset(spec: CohortSubsetSpec, result: CohortSubsetResult, *, replace_incomplete: bool = False) -> Path:
    output = spec.output_root
    if output.exists():
        if (output / 'SUBSET_FROZEN.json').is_file():
            raise CohortSubsetError(f"Frozen subset already exists and is immutable: {output}")
        if not replace_incomplete:
            raise CohortSubsetError("Incomplete output exists; review it and use --replace-incomplete")
        shutil.rmtree(output)
    temp = output.with_name(output.name + '.tmp')
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    try:
        result.metadata.to_csv(temp/'subset_metadata.csv', index=False)
        result.train_matrix.to_csv(temp/'subset_train_base_matrix.csv.gz', index=False, compression='gzip')
        result.test_matrix.to_csv(temp/'subset_test_final_matrix.csv.gz', index=False, compression='gzip')
        result.sample_audit.to_csv(temp/'subset_sample_audit.csv', index=False)
        (temp/'subset_summary.json').write_text(json.dumps(result.summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        (temp/'subset_resolved_spec.yaml').write_text(yaml.safe_dump(dict(spec.raw), sort_keys=False, allow_unicode=True), encoding='utf-8')
        generated = [temp/name for name in (
            'subset_metadata.csv','subset_train_base_matrix.csv.gz','subset_test_final_matrix.csv.gz',
            'subset_sample_audit.csv','subset_summary.json','subset_resolved_spec.yaml')]
        marker = {
            'material_subset_version': SUBSET_VERSION,
            'subset_id': spec.subset_id,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'summary': dict(result.summary),
            'source_files': [_record(p, spec.repository_root) for p in (*spec.metadata_files, spec.train_base_matrix, spec.test_final_matrix)],
            'generated_files': [
                {'path': p.name, 'sha256': _sha256(p), 'size_bytes': p.stat().st_size}
                for p in generated
            ],
        }
        (temp/'SUBSET_FROZEN.json').write_text(json.dumps(marker, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        output.parent.mkdir(parents=True, exist_ok=True)
        temp.rename(output)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise
    return output

def verify_cohort_subset_marker(marker_path: Path, *, repository_root: Path) -> Mapping[str, Any]:
    marker_path = Path(marker_path).expanduser().resolve()
    if not marker_path.is_file():
        raise FileNotFoundError(marker_path)
    marker = json.loads(marker_path.read_text(encoding='utf-8'))
    if not isinstance(marker, Mapping) or str(marker.get('material_subset_version')) != SUBSET_VERSION:
        raise CohortSubsetError("Invalid subset marker")
    base = marker_path.parent
    records = marker.get('generated_files')
    if not isinstance(records, list) or not records:
        raise CohortSubsetError("Marker has no generated_files")
    for record in records:
        path = base / str(record['path'])
        if not path.is_file():
            raise CohortSubsetError(f"Frozen subset file missing: {path}")
        if _sha256(path) != str(record['sha256']):
            raise CohortSubsetError(f"Frozen subset file hash mismatch: {path}")
    return marker
