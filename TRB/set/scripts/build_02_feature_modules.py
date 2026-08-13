#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build registry-defined TRB Step02 feature-module tables (Phase 3)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
SET_DIR = SCRIPT_DIR.parent
SRC_DIR = SET_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.feature_framework import load_feature_registry  # noqa: E402
from ra_ild_trb.feature_modules import (  # noqa: E402
    ALL_MODULE_IDS,
    build_modules_from_step02,
    module_summary_table,
    write_modules,
)
from ra_ild_trb.paths import find_repository_root, resolve_project_path  # noqa: E402

SCRIPT_VERSION = "1.0.0-phase3"
DEFAULT_REGISTRY = "TRB/set/configs/feature_framework/trb_feature_registry_v1.yaml"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split validated Phase-2 TRB Step02 outputs into registry-defined feature modules "
            "without recalculating biological feature values."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--step02-dir",
        required=True,
        help="Directory containing Phase-2 source-aware Step02 outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Module output directory. Default: <step02-dir>/02_feature_modules.",
    )
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--repository-root", default=None)
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
    registry_path = resolve_project_path(
        args.registry, repo_root, must_exist=True, expect="file"
    )
    step02_dir = resolve_project_path(
        args.step02_dir, repo_root, must_exist=True, expect="dir"
    )
    output_dir = (
        resolve_project_path(args.output_dir, repo_root, must_exist=False)
        if args.output_dir
        else step02_dir / "02_feature_modules"
    )

    registry = load_feature_registry(registry_path)
    result = build_modules_from_step02(step02_dir, registry)
    outputs = write_modules(
        result,
        step02_dir,
        output_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    summary = module_summary_table(result)
    print("")
    print("=" * 86)
    print("TRB Step02 Feature Modules — Phase 3")
    print("=" * 86)
    print(f"Script version:     {SCRIPT_VERSION}")
    print(f"Repertoire source:  {result.repertoire_source}")
    print(f"Phase-2 plan:       {result.plan_id}")
    print(f"Samples:            {result.sample_count}")
    print(f"Core feature count: {result.core_feature_count} (excluding sample_id)")
    print(f"Static candidates:  {result.static_candidate_count}")
    print(f"Step02 input:        {step02_dir}")
    print(f"Module output:       {output_dir}")
    print("")
    print("[Module inventory]")
    for _, row in summary.iterrows():
        print(
            f"- {row['feature_group']}: features={int(row['feature_count']):,}; "
            f"fit_scope={row['fit_scope']}; selected_in_plan={bool(row['selected_in_phase2_plan'])}; "
            f"reference_drop={int(row['reference_drop_count'])}; "
            f"candidate_default={int(row['candidate_default_count'])}"
        )

    print("")
    print("[Compatibility contract]")
    print("- PASS: eight static modules exactly reconstruct the original 96-feature core.")
    print(
        f"- {'PASS' if result.static_candidate_count == 83 else 'FAIL'}: "
        f"static 96→83 default candidate contract ({result.static_candidate_count})."
    )
    print("- PASS: both 3-mer modules preserve Phase-2 sample order and values.")

    if args.dry_run:
        print("- DRY-RUN PASS: complete validation performed; no module files written.")
    else:
        print("- WRITE PASS: 10 module tables + manifest + summary written.")
        for key in [*ALL_MODULE_IDS, "manifest", "summary"]:
            print(f"  - {key}: {outputs[key]}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nERROR: user interrupted run", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
