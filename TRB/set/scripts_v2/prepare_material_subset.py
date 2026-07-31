#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare and freeze a material-defined TRB analysis subcohort."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.cohort_subset import (  # noqa: E402
    load_cohort_subset_spec,
    prepare_cohort_subset,
    verify_frozen_subset,
    write_frozen_cohort_subset,
)


def _find_repository_root(start: Path) -> Path:
    current = start.expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() or (
            (candidate / "TRB").is_dir() and (candidate / "TRB/set").is_dir()
        ):
            return candidate
    raise FileNotFoundError(
        f"Unable to find repository root from {start}; use --repository-root"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare immutable PBMC-only or buffercoat-only TRB source inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--spec", help="Batch 12 material-subset YAML")
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--replace-incomplete",
        action="store_true",
        help="Replace a partial output directory only when no SUBSET_FROZEN.json exists",
    )
    parser.add_argument(
        "--verify-marker",
        default=None,
        help="Verify an existing SUBSET_FROZEN.json instead of preparing a subset",
    )
    args = parser.parse_args()
    if bool(args.spec) == bool(args.verify_marker):
        parser.error("Provide exactly one of --spec or --verify-marker")
    return args


def main() -> int:
    args = parse_args()
    try:
        anchor = Path(args.spec or args.verify_marker).expanduser().resolve()
        root = (
            Path(args.repository_root).expanduser().resolve()
            if args.repository_root
            else _find_repository_root(anchor)
        )
        if args.verify_marker:
            marker = verify_frozen_subset(
                Path(args.verify_marker), repository_root=root
            )
            print("TRB Batch 12 material subset verification: PASS")
            print(f"Subset ID:        {marker['subset_id']}")
            print(f"Selected samples: {marker['selected_samples']}")
            print(f"Marker:           {Path(args.verify_marker).resolve()}")
            return 0

        spec = load_cohort_subset_spec(Path(args.spec), repository_root=root)
        result = prepare_cohort_subset(spec)
        print("RA-ILD TRB Batch 12 material-subset preparation")
        print(f"Subset ID:        {spec.subset_id}")
        print(f"Filter:           {spec.filter_column} in {list(spec.include_values)}")
        print(f"Selected samples: {len(result.metadata)}")
        print(f"Selected patients:{result.metadata[spec.patient_id_column].nunique()}")
        print(f"Label counts:     {dict(result.summary['label_counts'])}")
        print(f"Source partitions:{dict(result.summary['source_partition_counts'])}")
        print(f"Train matrix:     {result.train_matrix.shape}")
        print(f"Test matrix:      {result.test_matrix.shape}")
        if args.dry_run:
            print("Dry run only; no subset files were written.")
            return 0

        marker = write_frozen_cohort_subset(
            result,
            spec,
            replace_incomplete=args.replace_incomplete,
        )
        print("Material subset:  COMPLETE AND FROZEN")
        print(f"Output:           {spec.output_root}")
        print(f"Metadata SHA256:  {marker['files']['metadata']['sha256']}")
        print(json.dumps(result.summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"RA-ILD TRB Batch 12 material subset: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
