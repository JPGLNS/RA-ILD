#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a plan-selected TRB predictor matrix from Phase-3 feature modules."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
SET_DIR = SCRIPT_DIR.parent
SRC_DIR = SET_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.feature_framework import load_feature_plan, load_feature_registry  # noqa: E402
from ra_ild_trb.feature_matrix import (  # noqa: E402
    assemble_feature_matrix,
    write_feature_matrix_result,
)
from ra_ild_trb.paths import find_repository_root, resolve_project_path  # noqa: E402

SCRIPT_VERSION = "1.0.0-phase4"
DEFAULT_REGISTRY = "TRB/set/configs/feature_framework/trb_feature_registry_v1.yaml"
DEFAULT_PLAN = "TRB/set/configs/feature_framework/trb_feature_plan_top10000_primary_weighted_v1.yaml"
DEFAULT_CONTEXT = "cohort,patient,age,sex,material,batch"


def _csv_list(value: str) -> List[str]:
    result: List[str] = []
    for item in value.split(","):
        item = item.strip()
        if item and item not in result:
            result.append(item)
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble a config-driven TRB predictor matrix from Phase-3 modules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--module-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--id-col", default="libraryid")
    parser.add_argument("--context-cols", default=DEFAULT_CONTEXT)
    parser.add_argument(
        "--reference-manifest",
        default=None,
        help="Optional TRAIN 04_feature_matrix_manifest.csv used to enforce identical TEST feature schema.",
    )
    parser.add_argument(
        "--keep-reference-columns",
        action="store_true",
        help="Keep compositional reference columns. Default is to drop registry-defined reference columns.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = (
        Path(args.repository_root).expanduser().resolve()
        if args.repository_root
        else find_repository_root(SCRIPT_DIR)
    )
    registry_path = resolve_project_path(args.registry, repo_root, must_exist=True, expect="file")
    plan_path = resolve_project_path(args.plan, repo_root, must_exist=True, expect="file")
    metadata = resolve_project_path(args.metadata, repo_root, must_exist=True, expect="file")
    module_dir = resolve_project_path(args.module_dir, repo_root, must_exist=True, expect="dir")
    output_dir = resolve_project_path(args.output_dir, repo_root, must_exist=False)
    reference_manifest = (
        resolve_project_path(args.reference_manifest, repo_root, must_exist=True, expect="file")
        if args.reference_manifest
        else None
    )
    context_cols = _csv_list(args.context_cols)

    registry = load_feature_registry(registry_path)
    plan = load_feature_plan(plan_path)
    resolved_plan = registry.resolve_plan(plan)

    result = assemble_feature_matrix(
        metadata_path=metadata,
        module_dir=module_dir,
        registry=registry,
        resolved_plan=resolved_plan,
        id_col=args.id_col,
        context_columns=context_cols,
        keep_reference_columns=args.keep_reference_columns,
        reference_manifest_path=reference_manifest,
    )

    print("")
    print("=" * 78)
    print("TRB Feature Framework Phase 4：matrix assembly")
    print("=" * 78)
    print(f"Script version:      {SCRIPT_VERSION}")
    print(f"Plan:                {result.plan_id}")
    print(f"Repertoire source:   {result.repertoire_source}")
    print(f"Samples:             {result.sample_count}")
    print(f"Predictor features:  {result.feature_count:,}")
    print(f"Matrix columns:      {result.matrix.shape[1]:,} (including sample_id)")
    print(f"Context columns:     {result.context.shape[1]} (including sample_id)")
    print(f"Reference drops:     {len(result.dropped_reference_columns)}")
    print(f"Reference manifest:  {result.reference_manifest_path or 'none'}")
    print("")
    print("[Selected feature groups]")
    for group_id in result.selected_feature_groups:
        sub = result.selection_audit[result.selection_audit["feature_group"] == group_id]
        included = int(sub["included"].astype(bool).sum())
        excluded = int((~sub["included"].astype(bool)).sum())
        print(f"- {group_id}: included={included:,}; excluded={excluded:,}")

    outputs = write_feature_matrix_result(
        result,
        resolved_plan,
        output_dir,
        id_col=args.id_col,
        context_columns=context_cols,
        keep_reference_columns=args.keep_reference_columns,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )

    print("")
    print("[Contracts]")
    print("- PASS: metadata is authoritative sample order; every selected module has exact sample coverage.")
    print("- PASS: no duplicate predictor names and no missing predictor values.")
    print("- PASS: cohort/context fields are separate from the predictor matrix.")
    if reference_manifest:
        print("- PASS: feature schema is identical to the supplied reference manifest.")
    if args.dry_run:
        print("- DRY-RUN PASS: complete assembly/validation finished; no Phase-4 files written.")
    else:
        print(f"- WRITE PASS: {len(outputs)} Phase-4 outputs written.")
        for name, path in outputs.items():
            print(f"  - {name}: {path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
