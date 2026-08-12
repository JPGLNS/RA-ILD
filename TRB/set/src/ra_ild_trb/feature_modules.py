#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase-3 modular output helpers for TRB Step02 features.

This module never recalculates biological features.  It splits the validated
Phase-2 Step02 artifacts into registry-defined feature-group tables, validates
that the split is lossless/reversible, and builds a feature-level manifest for
later matrix assembly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .feature_framework import FeatureGroup, FeatureRegistry


STATIC_GROUP_IDS: Tuple[str, ...] = (
    "tcr_qc_depth",
    "tcr_diversity",
    "tcr_clonal_expansion",
    "tcr_aa_length",
    "tcr_aa_composition_unweighted",
    "tcr_aa_composition_weighted",
    "tcr_physicochemical_unweighted",
    "tcr_physicochemical_weighted",
)

KMER_GROUP_IDS: Tuple[str, ...] = (
    "tcr_3mer_unweighted",
    "tcr_3mer_weighted",
)

ALL_MODULE_IDS: Tuple[str, ...] = STATIC_GROUP_IDS + KMER_GROUP_IDS

CORE_INPUT_FILENAME = "02_core_sample_features.csv"
UNWEIGHTED_INPUT_FILENAME = "02_3mer_unweighted_features.csv"
WEIGHTED_INPUT_FILENAME = "02_3mer_weighted_features.csv"
PROVENANCE_INPUT_FILENAME = "02_resolved_feature_source.json"

MANIFEST_FILENAME = "02_feature_module_manifest.csv"
SUMMARY_FILENAME = "02_feature_module_build_summary.md"


class FeatureModuleError(ValueError):
    """Raised when Step02 artifacts violate the modular-output contract."""


@dataclass(frozen=True)
class ModuleBuildResult:
    repertoire_source: str
    plan_id: str
    selected_feature_groups: Tuple[str, ...]
    modules: Mapping[str, pd.DataFrame]
    manifest: pd.DataFrame
    core_feature_count: int
    static_candidate_count: int
    sample_count: int


def module_filename(group_id: str) -> str:
    if group_id not in ALL_MODULE_IDS:
        raise FeatureModuleError(f"unknown Phase-3 module id: {group_id}")
    return f"{group_id}.csv"


def expected_output_paths(output_dir: Path) -> Dict[str, Path]:
    paths = {gid: output_dir / module_filename(gid) for gid in ALL_MODULE_IDS}
    paths["manifest"] = output_dir / MANIFEST_FILENAME
    paths["summary"] = output_dir / SUMMARY_FILENAME
    return paths


def _validate_sample_table(df: pd.DataFrame, label: str) -> List[str]:
    if "sample_id" not in df.columns:
        raise FeatureModuleError(f"{label} is missing required column 'sample_id'")
    if df["sample_id"].isna().any():
        raise FeatureModuleError(f"{label} contains empty sample_id values")
    ids = df["sample_id"].astype(str).str.strip()
    if ids.eq("").any():
        raise FeatureModuleError(f"{label} contains empty sample_id values")
    if ids.duplicated().any():
        dup = ids[ids.duplicated(keep=False)].unique().tolist()[:10]
        raise FeatureModuleError(f"{label} contains duplicate sample_id values: {dup}")
    if df.columns.duplicated().any():
        dup_cols = df.columns[df.columns.duplicated()].tolist()
        raise FeatureModuleError(f"{label} contains duplicate columns: {dup_cols}")
    return ids.tolist()


def _validate_numeric_features(df: pd.DataFrame, label: str) -> None:
    feature_cols = [c for c in df.columns if c != "sample_id"]
    if not feature_cols:
        raise FeatureModuleError(f"{label} contains no feature columns")
    try:
        values = df[feature_cols].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise FeatureModuleError(f"{label} contains non-numeric feature values") from exc
    if not np.isfinite(values).all():
        raise FeatureModuleError(f"{label} contains NaN or infinite feature values")


def _require_group(registry: FeatureRegistry, group_id: str) -> FeatureGroup:
    group = registry.feature_groups.get(group_id)
    if group is None:
        raise FeatureModuleError(f"registry is missing required feature group {group_id!r}")
    if group.status != "defined":
        raise FeatureModuleError(
            f"Phase-3 module {group_id!r} must be status='defined', got {group.status!r}"
        )
    return group


def static_group_columns(registry: FeatureRegistry) -> Dict[str, Tuple[str, ...]]:
    """Return the exact core-column contract for the eight static modules."""
    result: Dict[str, Tuple[str, ...]] = {}
    seen: Dict[str, str] = {}
    for group_id in STATIC_GROUP_IDS:
        group = _require_group(registry, group_id)
        if group.module_type != "sample_static":
            raise FeatureModuleError(
                f"{group_id} must be module_type='sample_static', got {group.module_type!r}"
            )
        if group.selector.prefixes:
            raise FeatureModuleError(
                f"{group_id} static selector must use exact columns only; prefixes={group.selector.prefixes}"
            )
        columns = tuple(group.selector.exact)
        if not columns:
            raise FeatureModuleError(f"{group_id} static selector has no exact columns")
        for column in columns:
            previous = seen.get(column)
            if previous is not None:
                raise FeatureModuleError(
                    f"core column {column!r} belongs to multiple modules: {previous}, {group_id}"
                )
            seen[column] = group_id
        result[group_id] = columns
    return result


def split_core_dataframe(
    core_df: pd.DataFrame,
    registry: FeatureRegistry,
) -> Dict[str, pd.DataFrame]:
    """Split the legacy 96-feature Step02 core into eight exact registry modules."""
    sample_ids = _validate_sample_table(core_df, "02 core")
    _validate_numeric_features(core_df, "02 core")

    groups = static_group_columns(registry)
    expected = [column for gid in STATIC_GROUP_IDS for column in groups[gid]]
    actual = [column for column in core_df.columns if column != "sample_id"]

    missing = [column for column in expected if column not in actual]
    extra = [column for column in actual if column not in set(expected)]
    if missing or extra:
        raise FeatureModuleError(
            "02 core columns do not match the registry static-module contract; "
            f"missing={missing}, extra={extra}"
        )
    if len(actual) != len(expected):
        raise FeatureModuleError(
            f"02 core feature-count mismatch: actual={len(actual)}, expected={len(expected)}"
        )

    modules: Dict[str, pd.DataFrame] = {}
    for group_id in STATIC_GROUP_IDS:
        columns = list(groups[group_id])
        module = core_df[["sample_id", *columns]].copy()
        if module["sample_id"].astype(str).str.strip().tolist() != sample_ids:
            raise FeatureModuleError(f"sample order changed while building {group_id}")
        modules[group_id] = module

    # Lossless/reversible assertion: rebuild in the exact original core order.
    by_feature = {
        column: modules[group_id][column]
        for group_id in STATIC_GROUP_IDS
        for column in groups[group_id]
    }
    rebuilt = pd.DataFrame({"sample_id": core_df["sample_id"]})
    for column in actual:
        rebuilt[column] = by_feature[column]
    if list(rebuilt.columns) != list(core_df.columns) or not rebuilt.equals(core_df):
        raise FeatureModuleError("core module split is not exactly reversible")
    return modules


def validate_kmer_dataframe(
    df: pd.DataFrame,
    registry: FeatureRegistry,
    group_id: str,
    expected_sample_ids: Sequence[str],
) -> pd.DataFrame:
    group = _require_group(registry, group_id)
    if group.module_type != "training_vocabulary":
        raise FeatureModuleError(
            f"{group_id} must be module_type='training_vocabulary', got {group.module_type!r}"
        )
    if group.selector.exact or len(group.selector.prefixes) != 1:
        raise FeatureModuleError(
            f"{group_id} must define exactly one prefix selector and no exact columns"
        )

    sample_ids = _validate_sample_table(df, group_id)
    if list(sample_ids) != list(expected_sample_ids):
        raise FeatureModuleError(
            f"{group_id} sample order does not match 02 core sample order"
        )
    _validate_numeric_features(df, group_id)

    prefix = group.selector.prefixes[0]
    feature_cols = [column for column in df.columns if column != "sample_id"]
    invalid = [column for column in feature_cols if not column.startswith(prefix)]
    if invalid:
        raise FeatureModuleError(
            f"{group_id} contains columns outside prefix {prefix!r}: {invalid[:10]}"
        )
    if len(feature_cols) != len(set(feature_cols)):
        raise FeatureModuleError(f"{group_id} contains duplicate feature names")
    return df.copy()


def read_phase2_provenance(path: Path) -> Tuple[str, str, Tuple[str, ...]]:
    if not path.is_file():
        raise FileNotFoundError(f"Phase-2 provenance file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FeatureModuleError("Phase-2 provenance JSON must be an object")
    plan = payload.get("plan")
    if not isinstance(plan, dict):
        raise FeatureModuleError("Phase-2 provenance is missing plan object")
    source = plan.get("repertoire_source")
    if not isinstance(source, dict) or not source.get("id"):
        raise FeatureModuleError("Phase-2 provenance is missing plan.repertoire_source.id")
    source_id = str(source["id"])
    plan_id = str(plan.get("plan_id") or "").strip()
    groups_raw = plan.get("feature_groups")
    if not isinstance(groups_raw, list):
        raise FeatureModuleError("Phase-2 provenance plan.feature_groups must be a list")
    selected: List[str] = []
    for item in groups_raw:
        if not isinstance(item, dict) or not item.get("id"):
            raise FeatureModuleError("invalid feature-group entry in Phase-2 provenance")
        selected.append(str(item["id"]))
    if len(selected) != len(set(selected)):
        raise FeatureModuleError("Phase-2 provenance contains duplicate selected feature groups")
    return source_id, plan_id, tuple(selected)


def _candidate_default(group: FeatureGroup, feature_name: str) -> bool:
    return (
        group.model_role == "candidate_predictor"
        and feature_name not in set(group.reference_drop_columns)
    )


def build_feature_manifest(
    registry: FeatureRegistry,
    repertoire_source: str,
    plan_id: str,
    selected_feature_groups: Sequence[str],
    modules: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    if repertoire_source not in registry.repertoire_sources:
        raise FeatureModuleError(f"unknown repertoire source in provenance: {repertoire_source!r}")
    selected = set(selected_feature_groups)
    rows: List[Dict[str, object]] = []
    module_order = 0
    for group_id in ALL_MODULE_IDS:
        module_order += 1
        group = _require_group(registry, group_id)
        module = modules.get(group_id)
        if module is None:
            raise FeatureModuleError(f"missing built module {group_id}")
        features = [column for column in module.columns if column != "sample_id"]
        for feature_order, feature_name in enumerate(features, start=1):
            rows.append(
                {
                    "manifest_version": "1.0",
                    "repertoire_source": repertoire_source,
                    "plan_id": plan_id,
                    "feature_group": group_id,
                    "module_order": module_order,
                    "module_file": module_filename(group_id),
                    "feature_order": feature_order,
                    "feature_name": feature_name,
                    "family": group.family,
                    "module_type": group.module_type,
                    "fit_scope": group.fit_scope,
                    "leakage_policy": group.leakage_policy,
                    "model_role": group.model_role,
                    "producer": group.producer,
                    "selected_in_phase2_plan": group_id in selected,
                    "reference_drop": feature_name in set(group.reference_drop_columns),
                    "candidate_default": _candidate_default(group, feature_name),
                }
            )
    manifest = pd.DataFrame(rows)
    if manifest["feature_name"].duplicated().any():
        dup = manifest.loc[manifest["feature_name"].duplicated(keep=False), "feature_name"].tolist()[:10]
        raise FeatureModuleError(f"feature names collide across modules: {dup}")
    return manifest


def build_modules_from_step02(
    step02_dir: Path,
    registry: FeatureRegistry,
) -> ModuleBuildResult:
    step02_dir = Path(step02_dir).expanduser().resolve()
    required = {
        "core": step02_dir / CORE_INPUT_FILENAME,
        "unweighted": step02_dir / UNWEIGHTED_INPUT_FILENAME,
        "weighted": step02_dir / WEIGHTED_INPUT_FILENAME,
        "provenance": step02_dir / PROVENANCE_INPUT_FILENAME,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Phase-2 Step02 inputs:\n  - " + "\n  - ".join(missing))

    repertoire_source, plan_id, selected = read_phase2_provenance(required["provenance"])
    if repertoire_source not in registry.repertoire_sources:
        raise FeatureModuleError(
            f"Phase-2 provenance source {repertoire_source!r} is not present in registry"
        )

    core_df = pd.read_csv(required["core"])
    static_modules = split_core_dataframe(core_df, registry)
    sample_ids = core_df["sample_id"].astype(str).str.strip().tolist()

    unweighted_df = pd.read_csv(required["unweighted"])
    weighted_df = pd.read_csv(required["weighted"])
    modules: Dict[str, pd.DataFrame] = dict(static_modules)
    modules["tcr_3mer_unweighted"] = validate_kmer_dataframe(
        unweighted_df, registry, "tcr_3mer_unweighted", sample_ids
    )
    modules["tcr_3mer_weighted"] = validate_kmer_dataframe(
        weighted_df, registry, "tcr_3mer_weighted", sample_ids
    )

    manifest = build_feature_manifest(
        registry=registry,
        repertoire_source=repertoire_source,
        plan_id=plan_id,
        selected_feature_groups=selected,
        modules=modules,
    )
    static_manifest = manifest[manifest["feature_group"].isin(STATIC_GROUP_IDS)]
    static_candidate_count = int(static_manifest["candidate_default"].astype(bool).sum())

    return ModuleBuildResult(
        repertoire_source=repertoire_source,
        plan_id=plan_id,
        selected_feature_groups=selected,
        modules=modules,
        manifest=manifest,
        core_feature_count=int(len([c for c in core_df.columns if c != "sample_id"])),
        static_candidate_count=static_candidate_count,
        sample_count=len(sample_ids),
    )


def module_summary_table(result: ModuleBuildResult) -> pd.DataFrame:
    rows = []
    for group_id in ALL_MODULE_IDS:
        module = result.modules[group_id]
        subset = result.manifest[result.manifest["feature_group"] == group_id]
        first = subset.iloc[0]
        rows.append(
            {
                "feature_group": group_id,
                "module_file": module_filename(group_id),
                "sample_count": len(module),
                "feature_count": module.shape[1] - 1,
                "module_type": first["module_type"],
                "fit_scope": first["fit_scope"],
                "model_role": first["model_role"],
                "selected_in_phase2_plan": bool(first["selected_in_phase2_plan"]),
                "reference_drop_count": int(subset["reference_drop"].astype(bool).sum()),
                "candidate_default_count": int(subset["candidate_default"].astype(bool).sum()),
            }
        )
    return pd.DataFrame(rows)


def make_markdown_summary(
    result: ModuleBuildResult,
    step02_dir: Path,
    output_dir: Path,
) -> str:
    table = module_summary_table(result)
    lines = [
        "# TRB Step02 Feature Modules — Phase 3",
        "",
        "## Contract",
        "",
        "- This phase does **not** recalculate biological features.",
        "- The 96 legacy core features are split by registry selectors into eight static modules.",
        "- The unweighted and weighted 3-mer matrices are retained as two training-vocabulary modules.",
        "- The eight static modules are recombined in-memory and must exactly reproduce the original core table.",
        "- Feature generation and feature selection remain separate: all ten modules are materialized even if a group was not selected in the Phase-2 plan.",
        "",
        "## Provenance",
        "",
        f"- Repertoire source: `{result.repertoire_source}`",
        f"- Phase-2 plan: `{result.plan_id}`",
        f"- Phase-2 Step02 directory: `{Path(step02_dir).resolve()}`",
        f"- Module output directory: `{Path(output_dir).resolve()}`",
        f"- Samples: **{result.sample_count}**",
        f"- Core features excluding sample_id: **{result.core_feature_count}**",
        f"- Static default candidate predictors after QC/reference exclusions: **{result.static_candidate_count}**",
        "",
        "## Module inventory",
        "",
        "| feature_group | feature_count | module_type | fit_scope | selected_in_phase2_plan | reference_drop_count | candidate_default_count |",
        "| --- | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {row['feature_group']} | {int(row['feature_count'])} | {row['module_type']} | "
            f"{row['fit_scope']} | {bool(row['selected_in_phase2_plan'])} | "
            f"{int(row['reference_drop_count'])} | {int(row['candidate_default_count'])} |"
        )
    lines.extend(
        [
            "",
            "## Compatibility checks",
            "",
            f"- Core reversible split: **PASS** ({result.core_feature_count} features).",
            f"- Static 96→83 candidate contract: **{'PASS' if result.static_candidate_count == 83 else 'FAIL'}**.",
            "- Sample order is identical across core, unweighted 3-mer, weighted 3-mer, and every module.",
            "- Feature values are copied from Phase-2 artifacts without scaling, filtering, or recomputation.",
            "",
        ]
    )
    return "\n".join(lines)


def write_dataframe_atomic(df: pd.DataFrame, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp, index=False)
    temp.replace(path)


def write_modules(
    result: ModuleBuildResult,
    step02_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Dict[str, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    outputs = expected_output_paths(output_dir)
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite and not dry_run:
        preview = "\n".join(f"  - {path}" for path in existing[:20])
        raise FileExistsError(
            "Phase-3 module outputs already exist; use --overwrite after review:\n" + preview
        )
    if dry_run:
        return outputs

    output_dir.mkdir(parents=True, exist_ok=True)
    for group_id in ALL_MODULE_IDS:
        write_dataframe_atomic(result.modules[group_id], outputs[group_id])
    write_dataframe_atomic(result.manifest, outputs["manifest"])
    summary = make_markdown_summary(result, step02_dir, output_dir)
    temp = outputs["summary"].with_suffix(".md.tmp")
    temp.write_text(summary + "\n", encoding="utf-8")
    temp.replace(outputs["summary"])
    return outputs
