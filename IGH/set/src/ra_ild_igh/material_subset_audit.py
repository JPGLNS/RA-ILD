#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only material-subset audit for the IGH repeated-holdout framework.

Batch 11A inventories the full IGH cohort before any PBMC-only or buffercoat-
only source cohort is frozen. It validates metadata/matrix identity, reports
material/cohort/batch/sex composition, evaluates candidate exact-stratification
policies, and writes a hash-audited report. It never modifies source data,
existing split sets, training bundles, experiments, model outputs, or the
independent test result directory.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


AUDIT_VERSION = "1.0"
AUDIT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")
DEFAULT_OUTPUT_ROOT = Path("IGH/set/material_subset_audits")


class MaterialSubsetAuditError(ValueError):
    """Raised when the material-subset audit contract is violated."""


@dataclass(frozen=True)
class MaterialSubsetAuditConfig:
    source_path: Path
    repository_root: Path
    raw: Mapping[str, Any]
    audit_id: str
    description: str
    output_dir: Path
    metadata_files: Tuple[Path, ...]
    matrix_files: Tuple[Path, ...]
    metadata_sample_id_column: str
    matrix_sample_id_column: str
    patient_id_column: str
    label_column: str
    material_column: str
    batch_column: str
    sex_column: str
    age_column: str
    require_one_sample_per_patient: bool
    expected_samples: Optional[int]
    expected_patients: Optional[int]
    expected_labels: Tuple[str, ...]
    candidate_exact_strata: Tuple[Tuple[str, ...], ...]
    balance_categorical: Tuple[str, ...]
    balance_numeric: Tuple[str, ...]
    minimum_exact_stratum_count: int
    train_fraction: float


@dataclass(frozen=True)
class MaterialSubsetAuditResult:
    metadata: pd.DataFrame
    sample_audit: pd.DataFrame
    material_summary: pd.DataFrame
    material_cohort_counts: pd.DataFrame
    material_cohort_batch_counts: pd.DataFrame
    material_sex_counts: pd.DataFrame
    strata_audit: pd.DataFrame
    recommendations: pd.DataFrame
    summary: Mapping[str, Any]
    input_paths: Tuple[Path, ...]


def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required: mamba install -c conda-forge pyyaml")


def _mapping(parent: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise MaterialSubsetAuditError(f"{context}.{key} must be a mapping")
    return value


def _string(parent: Mapping[str, Any], key: str, context: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MaterialSubsetAuditError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _bool(parent: Mapping[str, Any], key: str, context: str, default: bool) -> bool:
    value = parent.get(key, default)
    if not isinstance(value, bool):
        raise MaterialSubsetAuditError(f"{context}.{key} must be boolean")
    return value


def _optional_positive_int(parent: Mapping[str, Any], key: str, context: str) -> Optional[int]:
    value = parent.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MaterialSubsetAuditError(f"{context}.{key} must be null or an integer >= 1")
    return int(value)


def _positive_int(parent: Mapping[str, Any], key: str, context: str, default: int) -> int:
    value = parent.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MaterialSubsetAuditError(f"{context}.{key} must be an integer >= 1")
    return int(value)


def _fraction(parent: Mapping[str, Any], key: str, context: str, default: float) -> float:
    value = parent.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MaterialSubsetAuditError(f"{context}.{key} must be numeric")
    value = float(value)
    if not math.isfinite(value) or not (0.5 <= value < 1.0):
        raise MaterialSubsetAuditError(f"{context}.{key} must be in [0.5, 1.0)")
    return value


def _string_list(value: Any, context: str, *, allow_empty: bool = False) -> Tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise MaterialSubsetAuditError(f"{context} must be a list of strings")
    out: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise MaterialSubsetAuditError(f"{context} contains an invalid value")
        out.append(item.strip())
    if not out and not allow_empty:
        raise MaterialSubsetAuditError(f"{context} must not be empty")
    if len(out) != len(set(out)):
        raise MaterialSubsetAuditError(f"{context} contains duplicates")
    return tuple(out)


def _nested_string_lists(value: Any, context: str) -> Tuple[Tuple[str, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise MaterialSubsetAuditError(f"{context} must be a non-empty list of lists")
    out = tuple(_string_list(item, f"{context}[{index}]") for index, item in enumerate(value))
    if not out:
        raise MaterialSubsetAuditError(f"{context} must not be empty")
    if len(out) != len(set(out)):
        raise MaterialSubsetAuditError(f"{context} contains duplicated policies")
    return out


def _resolve_repo_path(value: str, root: Path, context: str) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise MaterialSubsetAuditError(f"{context} must stay inside repository root: {resolved}") from exc
    return resolved


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, root: Path) -> Mapping[str, Any]:
    resolved = Path(path).resolve()
    try:
        display = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        display = str(resolved)
    return {
        "path": display,
        "sha256": _sha256(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    frame = pd.read_csv(path)
    frame = frame.drop(
        columns=[c for c in frame.columns if str(c).startswith("Unnamed:")],
        errors="ignore",
    )
    duplicates = frame.columns[frame.columns.duplicated()].astype(str).tolist()
    if duplicates:
        raise MaterialSubsetAuditError(f"{label} has duplicate columns: {duplicates[:20]}")
    return frame


def load_material_subset_audit_config(
    path: Path,
    *,
    repository_root: Path,
) -> MaterialSubsetAuditConfig:
    """Load and validate one Batch 11A audit YAML."""
    _require_yaml()
    source_path = Path(path).expanduser().resolve()
    root = Path(repository_root).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Audit config not found: {source_path}")
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise MaterialSubsetAuditError("Audit YAML root must be a mapping")
    if str(raw.get("material_subset_audit_version")) != AUDIT_VERSION:
        raise MaterialSubsetAuditError(
            f"material_subset_audit_version must be {AUDIT_VERSION!r}"
        )

    audit = _mapping(raw, "audit", "root")
    audit_id = _string(audit, "id", "audit")
    if not AUDIT_ID_PATTERN.fullmatch(audit_id):
        raise MaterialSubsetAuditError(
            "audit.id must be 3-80 lowercase characters using a-z, 0-9, '.', '_' or '-'"
        )
    output_value = str(audit.get("output_dir", DEFAULT_OUTPUT_ROOT / audit_id))
    output_dir = _resolve_repo_path(output_value, root, "audit.output_dir")
    allowed_root = (root / DEFAULT_OUTPUT_ROOT).resolve()
    if not _is_within(output_dir, allowed_root) or output_dir.name != audit_id:
        raise MaterialSubsetAuditError(
            f"audit.output_dir must be under {DEFAULT_OUTPUT_ROOT} and end with audit.id"
        )

    source = _mapping(raw, "source", "root")
    metadata_values = _string_list(source.get("metadata_files"), "source.metadata_files")
    matrix_values = _string_list(source.get("matrix_files"), "source.matrix_files")
    columns = _mapping(raw, "columns", "root")
    expected = raw.get("expected", {})
    if expected is None:
        expected = {}
    if not isinstance(expected, Mapping):
        raise MaterialSubsetAuditError("root.expected must be a mapping")
    split = _mapping(raw, "split_recommendation", "root")

    labels_value = expected.get("labels", ["RA", "ILD"])
    return MaterialSubsetAuditConfig(
        source_path=source_path,
        repository_root=root,
        raw=raw,
        audit_id=audit_id,
        description=str(audit.get("description", "")).strip(),
        output_dir=output_dir,
        metadata_files=tuple(
            _resolve_repo_path(value, root, f"source.metadata_files[{index}]")
            for index, value in enumerate(metadata_values)
        ),
        matrix_files=tuple(
            _resolve_repo_path(value, root, f"source.matrix_files[{index}]")
            for index, value in enumerate(matrix_values)
        ),
        metadata_sample_id_column=_string(columns, "metadata_sample_id", "columns"),
        matrix_sample_id_column=_string(columns, "matrix_sample_id", "columns"),
        patient_id_column=_string(columns, "patient_id", "columns"),
        label_column=_string(columns, "label", "columns"),
        material_column=_string(columns, "material", "columns"),
        batch_column=_string(columns, "batch", "columns"),
        sex_column=_string(columns, "sex", "columns"),
        age_column=_string(columns, "age", "columns"),
        require_one_sample_per_patient=_bool(
            source, "require_one_sample_per_patient", "source", True
        ),
        expected_samples=_optional_positive_int(expected, "samples", "expected"),
        expected_patients=_optional_positive_int(expected, "patients", "expected"),
        expected_labels=_string_list(labels_value, "expected.labels"),
        candidate_exact_strata=_nested_string_lists(
            split.get("candidate_exact_strata"),
            "split_recommendation.candidate_exact_strata",
        ),
        balance_categorical=_string_list(
            split.get("balance_categorical", []),
            "split_recommendation.balance_categorical",
            allow_empty=True,
        ),
        balance_numeric=_string_list(
            split.get("balance_numeric", []),
            "split_recommendation.balance_numeric",
            allow_empty=True,
        ),
        minimum_exact_stratum_count=_positive_int(
            split,
            "minimum_exact_stratum_count",
            "split_recommendation",
            2,
        ),
        train_fraction=_fraction(
            split, "train_fraction", "split_recommendation", 0.70
        ),
    )


def _clean_identifier(series: pd.Series, label: str) -> pd.Series:
    if series.isna().any():
        raise MaterialSubsetAuditError(f"{label} contains missing values")
    values = series.astype(str).str.strip()
    if values.eq("").any():
        raise MaterialSubsetAuditError(f"{label} contains blank values")
    return values


def _clean_categorical(series: pd.Series, label: str) -> pd.Series:
    return _clean_identifier(series, label)


def _load_metadata(config: MaterialSubsetAuditConfig) -> pd.DataFrame:
    required = [
        config.metadata_sample_id_column,
        config.patient_id_column,
        config.label_column,
        config.material_column,
        config.batch_column,
        config.sex_column,
        config.age_column,
    ]
    parts: List[pd.DataFrame] = []
    for path in config.metadata_files:
        frame = _read_csv(path, "metadata")
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise MaterialSubsetAuditError(f"Metadata {path} is missing columns: {missing}")
        frame = frame.copy()
        frame["__metadata_source__"] = path.relative_to(config.repository_root).as_posix()
        parts.append(frame)
    metadata = pd.concat(parts, ignore_index=True, sort=False)
    for column in required[:-1]:
        metadata[column] = _clean_categorical(metadata[column], f"metadata.{column}")
    metadata[config.age_column] = pd.to_numeric(metadata[config.age_column], errors="raise")
    if not np.isfinite(metadata[config.age_column].to_numpy(float)).all():
        raise MaterialSubsetAuditError(f"metadata.{config.age_column} contains non-finite values")

    sample = config.metadata_sample_id_column
    patient = config.patient_id_column
    duplicated_samples = metadata.loc[metadata[sample].duplicated(keep=False), sample].unique().tolist()
    if duplicated_samples:
        raise MaterialSubsetAuditError(f"Metadata contains duplicate sample IDs: {duplicated_samples[:20]}")
    if config.require_one_sample_per_patient:
        counts = metadata.groupby(patient, sort=False).size()
        bad = counts[counts != 1]
        if not bad.empty:
            raise MaterialSubsetAuditError(
                "One sample per patient is required; violations=" + str(bad.head(20).to_dict())
            )

    if config.expected_samples is not None and len(metadata) != config.expected_samples:
        raise MaterialSubsetAuditError(
            f"Expected {config.expected_samples} samples, observed {len(metadata)}"
        )
    n_patients = int(metadata[patient].nunique())
    if config.expected_patients is not None and n_patients != config.expected_patients:
        raise MaterialSubsetAuditError(
            f"Expected {config.expected_patients} patients, observed {n_patients}"
        )
    observed_labels = set(metadata[config.label_column].astype(str))
    expected_labels = set(config.expected_labels)
    if observed_labels != expected_labels:
        raise MaterialSubsetAuditError(
            f"Label set mismatch: expected={sorted(expected_labels)}, observed={sorted(observed_labels)}"
        )
    return metadata


def _load_matrix_sample_audit(
    config: MaterialSubsetAuditConfig,
    metadata: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sample_col = config.matrix_sample_id_column
    metadata_sample = config.metadata_sample_id_column
    matrix_parts: List[pd.DataFrame] = []
    comparisons = [
        config.label_column,
        config.material_column,
        config.batch_column,
        config.sex_column,
        config.age_column,
    ]
    for path in config.matrix_files:
        frame = _read_csv(path, "matrix")
        if sample_col not in frame.columns:
            raise MaterialSubsetAuditError(f"Matrix {path} lacks {sample_col!r}")
        keep = [sample_col] + [column for column in comparisons if column in frame.columns]
        subset = frame[keep].copy()
        subset[sample_col] = _clean_identifier(subset[sample_col], f"matrix.{sample_col}")
        subset["__matrix_source__"] = path.relative_to(config.repository_root).as_posix()
        matrix_parts.append(subset)
    matrix = pd.concat(matrix_parts, ignore_index=True, sort=False)
    duplicates = matrix.loc[matrix[sample_col].duplicated(keep=False), sample_col].unique().tolist()
    if duplicates:
        raise MaterialSubsetAuditError(f"Matrix files contain duplicate sample IDs: {duplicates[:20]}")

    metadata_ids = set(metadata[metadata_sample].astype(str))
    matrix_ids = set(matrix[sample_col].astype(str))
    missing_matrix = sorted(metadata_ids - matrix_ids)
    extra_matrix = sorted(matrix_ids - metadata_ids)
    if missing_matrix or extra_matrix:
        raise MaterialSubsetAuditError(
            "Metadata/matrix sample-set mismatch; "
            f"missing_matrix={missing_matrix[:20]}, extra_matrix={extra_matrix[:20]}"
        )

    lookup = metadata.set_index(metadata_sample, drop=False)
    mismatch_rows: List[Dict[str, Any]] = []
    for _, row in matrix.iterrows():
        sample_id = str(row[sample_col])
        meta = lookup.loc[sample_id]
        for column in comparisons:
            if column not in matrix.columns or pd.isna(row.get(column)):
                continue
            if column == config.age_column:
                left = float(pd.to_numeric(pd.Series([row[column]]), errors="raise").iloc[0])
                right = float(meta[column])
                equal = math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
            else:
                left = str(row[column]).strip()
                right = str(meta[column]).strip()
                equal = left == right
            if not equal:
                mismatch_rows.append(
                    {
                        "sample_id": sample_id,
                        "column": column,
                        "metadata_value": right,
                        "matrix_value": left,
                    }
                )
    mismatch = pd.DataFrame(mismatch_rows)
    if not mismatch.empty:
        raise MaterialSubsetAuditError(
            "Metadata/matrix value mismatch; examples=" + mismatch.head(20).to_json(orient="records")
        )

    audit = metadata[[
        metadata_sample,
        config.patient_id_column,
        config.label_column,
        config.material_column,
        config.batch_column,
        config.sex_column,
        config.age_column,
        "__metadata_source__",
    ]].copy()
    source_lookup = matrix.set_index(sample_col)["__matrix_source__"]
    audit["matrix_source"] = audit[metadata_sample].map(source_lookup)
    audit["metadata_matrix_match"] = True
    return matrix, audit


def _count_table(frame: pd.DataFrame, columns: Sequence[str], count_name: str) -> pd.DataFrame:
    return (
        frame.groupby(list(columns), dropna=False, sort=True)
        .size()
        .reset_index(name=count_name)
        .sort_values(list(columns), kind="mergesort")
        .reset_index(drop=True)
    )


def _policy_name(columns: Sequence[str]) -> str:
    return "+".join(columns)


def audit_material_subsets(config: MaterialSubsetAuditConfig) -> MaterialSubsetAuditResult:
    """Run a read-only full-cohort material composition audit."""
    metadata = _load_metadata(config)
    _, sample_audit = _load_matrix_sample_audit(config, metadata)
    material = config.material_column
    cohort = config.label_column
    batch = config.batch_column
    sex = config.sex_column
    patient = config.patient_id_column

    material_cohort = _count_table(metadata, [material, cohort], "sample_count")
    material_cohort_batch = _count_table(
        metadata, [material, cohort, batch], "sample_count"
    )
    material_sex = _count_table(metadata, [material, sex], "sample_count")

    summary_rows: List[Dict[str, Any]] = []
    strata_rows: List[Dict[str, Any]] = []
    recommendation_rows: List[Dict[str, Any]] = []
    expected_labels = list(config.expected_labels)

    for material_value, subset in metadata.groupby(material, sort=True):
        label_counts = subset[cohort].value_counts().to_dict()
        summary_row: Dict[str, Any] = {
            "material": material_value,
            "samples": int(len(subset)),
            "patients": int(subset[patient].nunique()),
            "batches": int(subset[batch].nunique()),
            "female": int((subset[sex].astype(str).str.casefold() == "female").sum()),
            "male": int((subset[sex].astype(str).str.casefold() == "male").sum()),
            "age_mean": float(subset[config.age_column].mean()),
            "age_sd": float(subset[config.age_column].std(ddof=1)) if len(subset) > 1 else 0.0,
        }
        for label in expected_labels:
            summary_row[f"{label}_count"] = int(label_counts.get(label, 0))
        material_summary_label_feasible = all(label_counts.get(label, 0) >= 2 for label in expected_labels)
        summary_row["both_labels_with_at_least_2"] = bool(material_summary_label_feasible)
        summary_rows.append(summary_row)

        selected_policy: Optional[Tuple[str, ...]] = None
        selected_reason = ""
        for order, policy in enumerate(config.candidate_exact_strata, start=1):
            missing_columns = [column for column in policy if column not in subset.columns]
            if missing_columns:
                raise MaterialSubsetAuditError(
                    f"Candidate exact strata reference missing columns: {missing_columns}"
                )
            counts = subset.groupby(list(policy), dropna=False).size()
            minimum = int(counts.min()) if len(counts) else 0
            singleton_count = int((counts == 1).sum())
            below_minimum = int((counts < config.minimum_exact_stratum_count).sum())
            feasible = bool(
                material_summary_label_feasible
                and len(counts) > 0
                and below_minimum == 0
            )
            strata_rows.append(
                {
                    "material": material_value,
                    "policy_order": order,
                    "exact_strata": _policy_name(policy),
                    "stratum_count": int(len(counts)),
                    "minimum_stratum_count": minimum,
                    "singleton_stratum_count": singleton_count,
                    "strata_below_minimum": below_minimum,
                    "minimum_required": config.minimum_exact_stratum_count,
                    "label_feasible": bool(material_summary_label_feasible),
                    "exact_stratification_feasible": feasible,
                }
            )
            if selected_policy is None and feasible:
                selected_policy = policy
                selected_reason = (
                    f"first candidate with all strata >= {config.minimum_exact_stratum_count}"
                )

        n_samples = int(len(subset))
        train_size = int(math.floor(n_samples * config.train_fraction + 0.5))
        train_size = min(max(train_size, 1), n_samples - 1)
        holdout_size = n_samples - train_size
        if selected_policy is None:
            selected_exact = "REVIEW_REQUIRED"
            selected_reason = (
                "no candidate exact-stratification policy passed the minimum-count rule"
            )
        else:
            selected_exact = _policy_name(selected_policy)
        recommendation_rows.append(
            {
                "material": material_value,
                "samples": n_samples,
                "train_size": train_size,
                "holdout_size": holdout_size,
                "train_fraction_requested": config.train_fraction,
                "recommended_exact_strata": selected_exact,
                "balance_categorical": "+".join(config.balance_categorical),
                "balance_numeric": "+".join(config.balance_numeric),
                "recommendation_status": (
                    "candidate_available" if selected_policy is not None else "review_required"
                ),
                "reason": selected_reason,
            }
        )

    material_summary = pd.DataFrame(summary_rows).sort_values("material").reset_index(drop=True)
    strata_audit = pd.DataFrame(strata_rows).sort_values(
        ["material", "policy_order"]
    ).reset_index(drop=True)
    recommendations = pd.DataFrame(recommendation_rows).sort_values("material").reset_index(drop=True)
    summary = {
        "audit_version": AUDIT_VERSION,
        "audit_id": config.audit_id,
        "receptor": "IGH",
        "samples": int(len(metadata)),
        "patients": int(metadata[patient].nunique()),
        "materials": sorted(metadata[material].astype(str).unique().tolist()),
        "labels": sorted(metadata[cohort].astype(str).unique().tolist()),
        "batches": sorted(metadata[batch].astype(str).unique().tolist()),
        "matrix_sample_set_matches_metadata": True,
        "one_sample_per_patient": bool(
            metadata.groupby(patient, sort=False).size().eq(1).all()
        ),
        "recommendations_are_preliminary": True,
        "scientific_note": (
            "Recommendations describe feasible metadata stratification only. "
            "They do not generate splits, fit models, select features, or read independent-test results."
        ),
    }
    return MaterialSubsetAuditResult(
        metadata=metadata,
        sample_audit=sample_audit,
        material_summary=material_summary,
        material_cohort_counts=material_cohort,
        material_cohort_batch_counts=material_cohort_batch,
        material_sex_counts=material_sex,
        strata_audit=strata_audit,
        recommendations=recommendations,
        summary=summary,
        input_paths=tuple(config.metadata_files + config.matrix_files),
    )


def _output_records(
    staging_dir: Path,
    final_dir: Path,
    root: Path,
) -> List[Mapping[str, Any]]:
    records = []
    for path in sorted(staging_dir.iterdir()):
        if path.name == "AUDIT_COMPLETE.json" or not path.is_file():
            continue
        final_path = final_dir / path.name
        try:
            display = final_path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            display = str(final_path.resolve())
        records.append({
            "path": display,
            "sha256": _sha256(path),
            "size_bytes": int(path.stat().st_size),
        })
    return records


def write_material_subset_audit(
    config: MaterialSubsetAuditConfig,
    result: MaterialSubsetAuditResult,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write audit tables and a final hash marker."""
    target = config.output_dir
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"Audit output already exists: {target}; use --overwrite only after review"
        )
    temp = target.with_name(target.name + ".tmp")
    if temp.exists():
        shutil.rmtree(temp)
    temp.parent.mkdir(parents=True, exist_ok=True)
    temp.mkdir()
    try:
        result.sample_audit.to_csv(temp / "sample_identity_audit.csv", index=False)
        result.material_summary.to_csv(temp / "material_summary.csv", index=False)
        result.material_cohort_counts.to_csv(temp / "material_cohort_counts.csv", index=False)
        result.material_cohort_batch_counts.to_csv(
            temp / "material_cohort_batch_counts.csv", index=False
        )
        result.material_sex_counts.to_csv(temp / "material_sex_counts.csv", index=False)
        result.strata_audit.to_csv(temp / "candidate_exact_strata_audit.csv", index=False)
        result.recommendations.to_csv(temp / "material_split_recommendations.csv", index=False)
        (temp / "audit_summary.json").write_text(
            json.dumps(result.summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _require_yaml()
        resolved = dict(config.raw)
        resolved.setdefault("audit", {})["output_dir"] = target.relative_to(
            config.repository_root
        ).as_posix()
        (temp / "audit_resolved_config.yaml").write_text(
            yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        marker = {
            "marker_version": "1.0",
            "audit_id": config.audit_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "read_only_source_audit": True,
            "inputs": [
                _file_record(path, config.repository_root) for path in result.input_paths
            ],
            "outputs": _output_records(temp, target, config.repository_root),
        }
        (temp / "AUDIT_COMPLETE.json").write_text(
            json.dumps(marker, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if target.exists():
            shutil.rmtree(target)
        temp.replace(target)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise
    return target


def verify_material_subset_audit_marker(
    marker_path: Path,
    *,
    repository_root: Path,
) -> Mapping[str, Any]:
    """Verify every input and output hash recorded in a COMPLETE marker."""
    marker_path = Path(marker_path).resolve()
    root = Path(repository_root).resolve()
    if not marker_path.is_file():
        raise FileNotFoundError(marker_path)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if not isinstance(marker, Mapping) or marker.get("status") != "complete":
        raise MaterialSubsetAuditError("Invalid or incomplete audit marker")
    for section in ("inputs", "outputs"):
        records = marker.get(section)
        if not isinstance(records, list):
            raise MaterialSubsetAuditError(f"Marker {section} must be a list")
        for record in records:
            path = Path(str(record["path"]))
            resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            observed = _sha256(resolved)
            if observed != record.get("sha256"):
                raise MaterialSubsetAuditError(
                    f"SHA256 mismatch for {resolved}: expected={record.get('sha256')}, observed={observed}"
                )
    return marker
