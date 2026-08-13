#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase-4 config-driven TRB feature-matrix assembly.

This module does not calculate repertoire features, select a training vocabulary,
encode categoricals, standardize predictors, or fit a model. It consumes the
Phase-3 feature-module tables plus metadata and materializes an auditable matrix
according to a resolved Feature Framework plan.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .feature_framework import FeatureGroup, FeatureRegistry, ResolvedFeaturePlan
from .feature_modules import MANIFEST_FILENAME as MODULE_MANIFEST_FILENAME


MATRIX_FILENAME = "04_feature_matrix.csv"
CONTEXT_FILENAME = "04_sample_context.csv"
MATRIX_MANIFEST_FILENAME = "04_feature_matrix_manifest.csv"
SELECTION_AUDIT_FILENAME = "04_feature_selection_audit.csv"
RESOLVED_PLAN_FILENAME = "04_resolved_matrix_plan.json"
SUMMARY_FILENAME = "04_feature_matrix_build_summary.md"


class FeatureMatrixError(ValueError):
    """Raised when feature modules/metadata violate the Phase-4 assembly contract."""


@dataclass(frozen=True)
class FeatureMatrixResult:
    plan_id: str
    repertoire_source: str
    selected_feature_groups: Tuple[str, ...]
    matrix: pd.DataFrame
    context: pd.DataFrame
    manifest: pd.DataFrame
    selection_audit: pd.DataFrame
    dropped_reference_columns: Tuple[str, ...]
    module_dir: Path
    metadata_path: Path
    reference_manifest_path: Optional[Path]

    @property
    def sample_count(self) -> int:
        return int(len(self.matrix))

    @property
    def feature_count(self) -> int:
        return int(self.matrix.shape[1] - 1)


def expected_output_paths(output_dir: Path) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    return {
        "matrix": output_dir / MATRIX_FILENAME,
        "context": output_dir / CONTEXT_FILENAME,
        "manifest": output_dir / MATRIX_MANIFEST_FILENAME,
        "selection_audit": output_dir / SELECTION_AUDIT_FILENAME,
        "resolved_plan": output_dir / RESOLVED_PLAN_FILENAME,
        "summary": output_dir / SUMMARY_FILENAME,
    }


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    frame = pd.read_csv(path)
    unnamed = [c for c in frame.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        frame = frame.drop(columns=unnamed)
    duplicated = frame.columns[frame.columns.duplicated()].astype(str).tolist()
    if duplicated:
        raise FeatureMatrixError(f"{label} contains duplicate columns: {duplicated[:20]}")
    return frame


def _normalize_ids(frame: pd.DataFrame, key: str, label: str) -> List[str]:
    if key not in frame.columns:
        raise FeatureMatrixError(f"{label} is missing ID column {key!r}")
    if frame[key].isna().any():
        raise FeatureMatrixError(f"{label} contains missing {key!r} values")
    ids = frame[key].astype(str).str.strip()
    if ids.eq("").any():
        raise FeatureMatrixError(f"{label} contains empty {key!r} values")
    if ids.duplicated().any():
        dup = ids[ids.duplicated(keep=False)].unique().tolist()[:20]
        raise FeatureMatrixError(f"{label} contains duplicate IDs: {dup}")
    frame[key] = ids
    return ids.tolist()


def _bool_series(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    valid = normalized.isin({"true", "false", "1", "0", "yes", "no"})
    if not valid.all():
        bad = normalized[~valid].unique().tolist()[:10]
        raise FeatureMatrixError(f"{label} contains invalid booleans: {bad}")
    return normalized.isin({"true", "1", "yes"})


def _validate_numeric(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    if not columns:
        return
    try:
        values = frame[list(columns)].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise FeatureMatrixError(f"{label} contains non-numeric values") from exc
    if not np.isfinite(values).all():
        raise FeatureMatrixError(f"{label} contains missing/NaN/infinite numeric values")


def _infer_metadata_type(series: pd.Series, column: str) -> Tuple[pd.Series, str]:
    if series.isna().any():
        raise FeatureMatrixError(f"metadata feature {column!r} contains missing values")
    if pd.api.types.is_numeric_dtype(series):
        converted = pd.to_numeric(series, errors="raise").astype(float)
        if not np.isfinite(converted.to_numpy(float)).all():
            raise FeatureMatrixError(f"metadata feature {column!r} contains non-finite values")
        return converted, "numeric"

    as_text = series.astype(str).str.strip()
    if as_text.eq("").any():
        raise FeatureMatrixError(f"metadata feature {column!r} contains empty values")
    converted = pd.to_numeric(as_text, errors="coerce")
    if converted.notna().all():
        converted = converted.astype(float)
        if not np.isfinite(converted.to_numpy(float)).all():
            raise FeatureMatrixError(f"metadata feature {column!r} contains non-finite values")
        return converted, "numeric"
    return as_text, "categorical"


def load_metadata(
    path: Path,
    id_col: str,
    context_columns: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    metadata = _read_csv(path, "metadata")
    ids = _normalize_ids(metadata, id_col, "metadata")

    requested_context: List[str] = []
    for column in context_columns:
        column = str(column).strip()
        if column and column != id_col and column not in requested_context:
            requested_context.append(column)
    missing_context = [c for c in requested_context if c not in metadata.columns]
    if missing_context:
        raise FeatureMatrixError(f"metadata is missing context columns: {missing_context}")

    context = metadata[[id_col, *requested_context]].copy().rename(columns={id_col: "sample_id"})
    if context.isna().any().any():
        bad = context.columns[context.isna().any()].tolist()
        raise FeatureMatrixError(f"sample context contains missing values: {bad}")
    return metadata, context, ids


def load_module_manifest(module_dir: Path, repertoire_source: str) -> pd.DataFrame:
    module_dir = Path(module_dir).expanduser().resolve()
    manifest = _read_csv(module_dir / MODULE_MANIFEST_FILENAME, "Phase-3 module manifest")
    required = {
        "repertoire_source",
        "feature_group",
        "module_file",
        "feature_order",
        "feature_name",
        "module_type",
        "fit_scope",
        "model_role",
        "reference_drop",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise FeatureMatrixError(f"Phase-3 module manifest is missing columns: {missing}")
    if manifest["feature_name"].astype(str).duplicated().any():
        dup = manifest.loc[
            manifest["feature_name"].astype(str).duplicated(keep=False), "feature_name"
        ].astype(str).tolist()[:20]
        raise FeatureMatrixError(f"Phase-3 manifest contains duplicate feature names: {dup}")
    sources = set(manifest["repertoire_source"].astype(str))
    if sources != {repertoire_source}:
        raise FeatureMatrixError(
            f"module manifest repertoire_source mismatch: expected={repertoire_source!r}, observed={sorted(sources)}"
        )
    manifest = manifest.copy()
    manifest["reference_drop"] = _bool_series(
        manifest["reference_drop"], "module manifest.reference_drop"
    )
    manifest["feature_order"] = pd.to_numeric(
        manifest["feature_order"], errors="raise"
    ).astype(int)
    return manifest


def _metadata_group_columns(group: FeatureGroup, metadata: pd.DataFrame) -> List[str]:
    columns: List[str] = []
    for name in group.selector.exact:
        if name not in metadata.columns:
            raise FeatureMatrixError(
                f"metadata feature group {group.id!r} requires missing column {name!r}"
            )
        columns.append(name)
    for prefix in group.selector.prefixes:
        matched = [c for c in metadata.columns if str(c).startswith(prefix)]
        if not matched:
            raise FeatureMatrixError(
                f"metadata feature group {group.id!r} prefix {prefix!r} matched no columns"
            )
        columns.extend(matched)
    if not columns:
        raise FeatureMatrixError(f"metadata feature group {group.id!r} has no selected columns")
    if len(columns) != len(set(columns)):
        raise FeatureMatrixError(f"metadata feature group {group.id!r} resolves duplicate columns")
    return columns


def _load_module_group(
    module_dir: Path,
    manifest: pd.DataFrame,
    group: FeatureGroup,
    expected_ids: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subset = manifest.loc[manifest["feature_group"].astype(str) == group.id].copy()
    if subset.empty:
        raise FeatureMatrixError(
            f"selected feature group {group.id!r} is not materialized in the Phase-3 module manifest"
        )
    module_files = subset["module_file"].astype(str).unique().tolist()
    if len(module_files) != 1:
        raise FeatureMatrixError(
            f"feature group {group.id!r} must map to exactly one module_file, got {module_files}"
        )
    subset = subset.sort_values("feature_order", kind="stable")
    feature_names = subset["feature_name"].astype(str).tolist()
    path = module_dir / module_files[0]
    frame = _read_csv(path, f"module {group.id}")
    ids = _normalize_ids(frame, "sample_id", f"module {group.id}")
    expected_set = set(expected_ids)
    observed_set = set(ids)
    missing = sorted(expected_set - observed_set)
    extra = sorted(observed_set - expected_set)
    if missing or extra:
        raise FeatureMatrixError(
            f"module {group.id!r} sample coverage mismatch; missing={missing[:20]}, extra={extra[:20]}"
        )
    actual_features = [c for c in frame.columns if c != "sample_id"]
    if actual_features != feature_names:
        raise FeatureMatrixError(
            f"module {group.id!r} columns do not match manifest order/content"
        )
    _validate_numeric(frame, feature_names, f"module {group.id}")
    ordered = frame.set_index("sample_id").loc[list(expected_ids)].reset_index()
    if ordered["sample_id"].tolist() != list(expected_ids):
        raise FeatureMatrixError(f"module {group.id!r} could not be reordered to metadata order")
    return ordered, subset


def validate_reference_manifest(
    manifest: pd.DataFrame,
    reference_path: Path,
) -> None:
    reference = _read_csv(reference_path, "reference matrix manifest")
    required = ["matrix_order", "feature_name", "feature_group", "data_type"]
    missing = [c for c in required if c not in reference.columns]
    if missing:
        raise FeatureMatrixError(f"reference matrix manifest is missing columns: {missing}")
    current = manifest[required].copy().reset_index(drop=True)
    expected = reference[required].copy().reset_index(drop=True)
    if len(current) != len(expected):
        raise FeatureMatrixError(
            f"reference feature schema size mismatch: current={len(current)}, reference={len(expected)}"
        )
    for column in required:
        if current[column].astype(str).tolist() != expected[column].astype(str).tolist():
            for index, (a, b) in enumerate(
                zip(current[column].astype(str), expected[column].astype(str)), start=1
            ):
                if a != b:
                    raise FeatureMatrixError(
                        f"reference feature schema mismatch in {column!r} at row {index}: current={a!r}, reference={b!r}"
                    )
            raise FeatureMatrixError(f"reference feature schema mismatch in {column!r}")


def assemble_feature_matrix(
    *,
    metadata_path: Path,
    module_dir: Path,
    registry: FeatureRegistry,
    resolved_plan: ResolvedFeaturePlan,
    id_col: str = "libraryid",
    context_columns: Sequence[str] = ("cohort", "patient", "age", "sex", "material", "batch"),
    keep_reference_columns: bool = False,
    reference_manifest_path: Optional[Path] = None,
) -> FeatureMatrixResult:
    metadata_path = Path(metadata_path).expanduser().resolve()
    module_dir = Path(module_dir).expanduser().resolve()
    metadata, context, sample_ids = load_metadata(metadata_path, id_col, context_columns)
    module_manifest = load_module_manifest(
        module_dir, resolved_plan.repertoire_source.id
    )

    matrix_values: Dict[str, object] = {"sample_id": list(sample_ids)}
    matrix_rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    dropped: List[str] = []
    seen_features = set()
    matrix_order = 0

    for group_order, group in enumerate(resolved_plan.feature_groups, start=1):
        if group.status != "defined":
            raise FeatureMatrixError(
                f"Phase-4 assembler requires materialized status='defined' groups; {group.id!r} has status={group.status!r}"
            )

        if group.module_type == "metadata_static":
            names = _metadata_group_columns(group, metadata)
            source_type = "metadata"
            source_file = str(metadata_path)
            source_frame = metadata.set_index(id_col).loc[sample_ids]
            source_manifest = None
        else:
            source_frame, source_manifest = _load_module_group(
                module_dir, module_manifest, group, sample_ids
            )
            source_type = "feature_module"
            source_file = str((module_dir / source_manifest["module_file"].iloc[0]).resolve())
            names = source_manifest["feature_name"].astype(str).tolist()

        reference_set = set(group.reference_drop_columns)
        for feature_order, feature_name in enumerate(names, start=1):
            if feature_name in seen_features:
                raise FeatureMatrixError(
                    f"feature name collision while assembling matrix: {feature_name!r}"
                )
            seen_features.add(feature_name)

            is_reference_drop = feature_name in reference_set
            if source_manifest is not None:
                row = source_manifest.loc[
                    source_manifest["feature_name"].astype(str) == feature_name
                ]
                if len(row) != 1:
                    raise FeatureMatrixError(
                        f"manifest lookup for {feature_name!r} in {group.id!r} is not one-to-one"
                    )
                manifest_reference = bool(row.iloc[0]["reference_drop"])
                if manifest_reference != is_reference_drop:
                    raise FeatureMatrixError(
                        f"reference-drop contract mismatch for {feature_name!r}: registry={is_reference_drop}, module_manifest={manifest_reference}"
                    )

            included = keep_reference_columns or not is_reference_drop
            exclusion_reason = "" if included else "reference_drop"
            dtype = "numeric"
            if source_type == "metadata":
                converted, dtype = _infer_metadata_type(
                    source_frame[feature_name], feature_name
                )
            else:
                converted = pd.to_numeric(source_frame[feature_name], errors="raise").astype(float)
                if not np.isfinite(converted.to_numpy(float)).all():
                    raise FeatureMatrixError(
                        f"module feature {feature_name!r} contains non-finite values"
                    )

            if included:
                matrix_order += 1
                matrix_values[feature_name] = list(converted)
                matrix_rows.append(
                    {
                        "manifest_version": "1.0",
                        "matrix_order": matrix_order,
                        "feature_name": feature_name,
                        "feature_group": group.id,
                        "group_order": group_order,
                        "feature_order_in_group": feature_order,
                        "source_type": source_type,
                        "source_file": source_file,
                        "data_type": dtype,
                        "family": group.family,
                        "module_type": group.module_type,
                        "fit_scope": group.fit_scope,
                        "leakage_policy": group.leakage_policy,
                        "model_role": group.model_role,
                        "repertoire_source": resolved_plan.repertoire_source.id,
                        "plan_id": resolved_plan.plan_id,
                    }
                )
            else:
                dropped.append(feature_name)

            audit_rows.append(
                {
                    "plan_id": resolved_plan.plan_id,
                    "repertoire_source": resolved_plan.repertoire_source.id,
                    "feature_group": group.id,
                    "group_order": group_order,
                    "feature_order_in_group": feature_order,
                    "feature_name": feature_name,
                    "source_type": source_type,
                    "data_type": dtype,
                    "model_role": group.model_role,
                    "reference_drop": is_reference_drop,
                    "included": included,
                    "exclusion_reason": exclusion_reason,
                    "matrix_order": matrix_order if included else "",
                }
            )

    matrix = pd.DataFrame(matrix_values)
    manifest = pd.DataFrame(matrix_rows)
    audit = pd.DataFrame(audit_rows)
    if matrix.columns.duplicated().any():
        dup = matrix.columns[matrix.columns.duplicated()].tolist()
        raise FeatureMatrixError(f"assembled matrix has duplicate columns: {dup[:20]}")
    if matrix["sample_id"].tolist() != sample_ids:
        raise FeatureMatrixError("assembled matrix changed metadata sample order")
    if matrix.isna().any().any():
        bad = matrix.columns[matrix.isna().any()].tolist()
        raise FeatureMatrixError(f"assembled matrix contains missing values: {bad[:20]}")

    if manifest["feature_name"].tolist() != list(matrix.columns[1:]):
        raise FeatureMatrixError("matrix manifest order does not match matrix columns")

    reference_path = None
    if reference_manifest_path is not None:
        reference_path = Path(reference_manifest_path).expanduser().resolve()
        validate_reference_manifest(manifest, reference_path)

    return FeatureMatrixResult(
        plan_id=resolved_plan.plan_id,
        repertoire_source=resolved_plan.repertoire_source.id,
        selected_feature_groups=tuple(group.id for group in resolved_plan.feature_groups),
        matrix=matrix,
        context=context,
        manifest=manifest,
        selection_audit=audit,
        dropped_reference_columns=tuple(dropped),
        module_dir=module_dir,
        metadata_path=metadata_path,
        reference_manifest_path=reference_path,
    )


def make_resolved_payload(
    result: FeatureMatrixResult,
    resolved_plan: ResolvedFeaturePlan,
    *,
    id_col: str,
    context_columns: Sequence[str],
    keep_reference_columns: bool,
) -> Dict[str, object]:
    return {
        "schema_version": "1.0",
        "phase": "feature_matrix_assembly",
        "plan": resolved_plan.as_dict(),
        "metadata": str(result.metadata_path),
        "metadata_id_column": id_col,
        "module_directory": str(result.module_dir),
        "sample_context_columns": list(context_columns),
        "keep_reference_columns": bool(keep_reference_columns),
        "reference_manifest": (
            str(result.reference_manifest_path) if result.reference_manifest_path else None
        ),
        "sample_count": result.sample_count,
        "feature_count": result.feature_count,
        "dropped_reference_columns": list(result.dropped_reference_columns),
        "feature_groups": list(result.selected_feature_groups),
    }


def make_markdown_summary(
    result: FeatureMatrixResult,
    *,
    output_dir: Path,
    keep_reference_columns: bool,
) -> str:
    audit = result.selection_audit
    lines = [
        "# TRB Feature Matrix — Phase 4",
        "",
        "## Contract",
        "",
        "- Matrix assembly is config-driven by the Feature Framework plan.",
        "- Phase 4 does not recalculate repertoire features or refit the 3-mer vocabulary.",
        "- Phase 4 does not standardize numeric predictors or encode categorical predictors.",
        "- `cohort` and other trace/context fields are kept in a separate sample-context table, not mixed into the predictor matrix.",
        "- Reference columns for compositional blocks are dropped by default and can be retained only with an explicit override.",
        "- TEST can be checked against a TRAIN matrix manifest with `--reference-manifest`.",
        "",
        "## Provenance",
        "",
        f"- Plan: `{result.plan_id}`",
        f"- Repertoire source: `{result.repertoire_source}`",
        f"- Metadata: `{result.metadata_path}`",
        f"- Module directory: `{result.module_dir}`",
        f"- Output directory: `{Path(output_dir).resolve()}`",
        f"- Reference manifest: `{result.reference_manifest_path or 'none'}`",
        "",
        "## Matrix result",
        "",
        f"- Samples: **{result.sample_count}**",
        f"- Predictor features: **{result.feature_count}**",
        f"- Matrix columns including sample_id: **{result.matrix.shape[1]}**",
        f"- Context columns including sample_id: **{result.context.shape[1]}**",
        f"- Reference columns retained: **{keep_reference_columns}**",
        f"- Reference columns dropped: **{len(result.dropped_reference_columns)}**",
        "",
        "## Selected feature groups",
        "",
        "| feature_group | available_before_reference_drop | included | excluded_reference | data_types | fit_scope |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for group_id in result.selected_feature_groups:
        sub = audit[audit["feature_group"] == group_id]
        dtypes = ",".join(sorted(set(sub["data_type"].astype(str))))
        included = int(_bool_series(sub["included"], "selection_audit.included").sum())
        excluded = int(_bool_series(sub["reference_drop"], "selection_audit.reference_drop").sum())
        fit_scope = str(result.manifest.loc[
            result.manifest["feature_group"] == group_id, "fit_scope"
        ].iloc[0]) if included else "n/a"
        lines.append(
            f"| {group_id} | {len(sub)} | {included} | {excluded if not keep_reference_columns else 0} | {dtypes} | {fit_scope} |"
        )
    if result.dropped_reference_columns:
        lines.extend([
            "",
            "## Reference columns dropped",
            "",
            *[f"- `{name}`" for name in result.dropped_reference_columns],
        ])
    lines.extend([
        "",
        "## Downstream modeling",
        "",
        "- Use `04_feature_matrix.csv` as predictors only.",
        "- Obtain the outcome label (for example `cohort`) from `04_sample_context.csv` by one-to-one `sample_id` join.",
        "- Fit standardization/encoding inside training folds; never fit preprocessing on the independent test set.",
        "",
    ])
    return "\n".join(lines)


def _write_dataframe_atomic(frame: pd.DataFrame, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def write_feature_matrix_result(
    result: FeatureMatrixResult,
    resolved_plan: ResolvedFeaturePlan,
    output_dir: Path,
    *,
    id_col: str,
    context_columns: Sequence[str],
    keep_reference_columns: bool,
    dry_run: bool = False,
    overwrite: bool = False,
) -> Dict[str, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    outputs = expected_output_paths(output_dir)
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite and not dry_run:
        raise FileExistsError(
            "Phase-4 outputs already exist; add --overwrite after review:\n  - "
            + "\n  - ".join(str(p) for p in existing)
        )
    if dry_run:
        return outputs

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_dataframe_atomic(result.matrix, outputs["matrix"])
    _write_dataframe_atomic(result.context, outputs["context"])
    _write_dataframe_atomic(result.manifest, outputs["manifest"])
    _write_dataframe_atomic(result.selection_audit, outputs["selection_audit"])

    resolved_payload = make_resolved_payload(
        result,
        resolved_plan,
        id_col=id_col,
        context_columns=context_columns,
        keep_reference_columns=keep_reference_columns,
    )
    temp_json = outputs["resolved_plan"].with_suffix(".json.tmp")
    temp_json.write_text(
        json.dumps(resolved_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temp_json.replace(outputs["resolved_plan"])

    summary = make_markdown_summary(
        result,
        output_dir=output_dir,
        keep_reference_columns=keep_reference_columns,
    )
    temp_summary = outputs["summary"].with_suffix(".md.tmp")
    temp_summary.write_text(summary + "\n", encoding="utf-8")
    temp_summary.replace(outputs["summary"])
    return outputs
