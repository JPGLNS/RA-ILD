#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the IGH Batch 11A material-subset composition audit."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.material_subset_audit import (
    audit_material_subsets,
    load_material_subset_audit_config,
    verify_material_subset_audit_marker,
    write_material_subset_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit IGH material composition without creating subsets or fitting models."
    )
    parser.add_argument("--config", help="Batch 11A audit YAML")
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-marker", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.repository_root).resolve() if args.repository_root else Path.cwd().resolve()
    try:
        if args.verify_marker:
            marker = verify_material_subset_audit_marker(
                Path(args.verify_marker), repository_root=root
            )
            print(f"Verified audit marker: {marker['audit_id']}")
            print("IGH_BATCH11A_MARKER_VERIFICATION_PASS")
            return 0
        if not args.config:
            raise ValueError("--config is required unless --verify-marker is used")
        config = load_material_subset_audit_config(
            Path(args.config), repository_root=root
        )
        result = audit_material_subsets(config)
        print("IGH Batch 11A material-subset audit")
        print(f"Audit ID: {config.audit_id}")
        print(f"Samples: {result.summary['samples']}")
        print(f"Patients: {result.summary['patients']}")
        print("\nMaterial summary")
        print(result.material_summary.round(3).to_string(index=False))
        print("\nCandidate exact-strata audit")
        print(result.strata_audit.to_string(index=False))
        print("\nPreliminary split recommendations")
        print(result.recommendations.to_string(index=False))
        if args.dry_run:
            print("IGH_BATCH11A_REAL_DATA_DRY_RUN_PASS")
            return 0
        output = write_material_subset_audit(
            config, result, overwrite=bool(args.overwrite)
        )
        print(f"Output directory: {output}")
        print("IGH_BATCH11A_AUDIT_COMPLETE")
        return 0
    except Exception as exc:
        print(f"IGH Batch 11A audit: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
