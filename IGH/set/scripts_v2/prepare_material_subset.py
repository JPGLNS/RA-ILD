#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare or verify one frozen IGH material-specific source cohort."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from ra_ild_igh.cohort_subset import (
    load_cohort_subset_spec, prepare_cohort_subset,
    verify_cohort_subset_marker, write_cohort_subset,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Prepare a frozen IGH material subset before any split or model fitting.')
    p.add_argument('--spec')
    p.add_argument('--repository-root', default=None)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--replace-incomplete', action='store_true')
    p.add_argument('--verify-marker', default=None)
    return p.parse_args()

def main() -> int:
    args = parse_args()
    root = Path(args.repository_root).resolve() if args.repository_root else Path.cwd().resolve()
    try:
        if args.verify_marker:
            marker = verify_cohort_subset_marker(Path(args.verify_marker), repository_root=root)
            print(f"Verified frozen subset: {marker['subset_id']}")
            print('IGH_BATCH11B_SUBSET_MARKER_VERIFICATION_PASS')
            return 0
        if not args.spec:
            raise ValueError('--spec is required unless --verify-marker is used')
        spec = load_cohort_subset_spec(Path(args.spec), repository_root=root)
        result = prepare_cohort_subset(spec)
        print('IGH Batch 11B material subset')
        print(f'Subset ID: {spec.subset_id}')
        print(f"Selected samples: {result.summary['samples']}")
        print(f"Selected patients: {result.summary['patients']}")
        print(f"Label counts: {result.summary['label_counts']}")
        print(f"Source partitions: train={result.summary['train_partition_samples']}, test={result.summary['test_partition_samples']}")
        if args.dry_run:
            print('IGH_BATCH11B_REAL_DATA_DRY_RUN_PASS')
            return 0
        output = write_cohort_subset(spec, result, replace_incomplete=bool(args.replace_incomplete))
        print(f'Frozen subset: {output}')
        print('IGH_BATCH11B_SUBSET_FROZEN')
        return 0
    except Exception as exc:
        print(f'IGH Batch 11B material subset: FAIL\n{exc}', file=sys.stderr)
        return 1
if __name__ == '__main__':
    raise SystemExit(main())
