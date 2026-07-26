#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare the Batch 04 full-cohort repeated-holdout training bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.paths import find_repository_root  # noqa: E402
from ra_ild_igh.repeated_holdout_training import (  # noqa: E402
    load_training_bundle_spec,
    prepare_training_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare full-cohort static features, sparse public cache, and fixed "
            "outer/inner assignments for the three frozen repeated holdouts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--spec", required=True)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replace-incomplete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        spec_path = Path(args.spec).expanduser().resolve()
        root = (
            Path(args.repository_root).expanduser().resolve()
            if args.repository_root
            else find_repository_root(spec_path)
        )
        spec = load_training_bundle_spec(spec_path, repository_root=root)
        result = prepare_training_bundle(
            spec,
            dry_run=args.dry_run,
            replace_incomplete=args.replace_incomplete,
        )
        print("RA-ILD IGH Batch 04 repeated-holdout training preparation")
        print(f"Bundle ID:        {spec.bundle_id}")
        print(f"Output root:      {spec.output_root}")
        if args.dry_run:
            print(f"Samples:          {result['samples']}")
            print(f"Static features:  {result['static_features']}")
            print(f"AA tables:        {result['aa_tables']}")
            print(f"Splits:           {result['split_count']}")
            print(f"Train sizes:      {result['train_sizes']}")
            print(f"Holdout sizes:    {result['holdout_sizes']}")
            print("Dry run only; full sequence files were not loaded and no output was written.")
        else:
            print("Training bundle:  COMPLETE AND FROZEN")
            print(f"Samples:          {result['samples']}")
            print(f"Static features:  {result['static_feature_count']}")
            print(f"Catalog size:     {result['catalog_size']}")
            print(f"Matrix nnz:       {result['matrix_nnz']}")
            print(f"Generated scheme: {result['generated_scheme']}")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"RA-ILD IGH Batch 04 preparation: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
