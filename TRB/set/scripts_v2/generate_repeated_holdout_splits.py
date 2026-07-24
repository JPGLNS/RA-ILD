#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate and freeze repeated RA-ILD train/holdout assignments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.paths import find_repository_root  # noqa: E402
from ra_ild_trb.repeated_holdout import (  # noqa: E402
    generate_repeated_holdout,
    load_combined_metadata,
    load_repeated_holdout_spec,
    write_frozen_split_set,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic metadata-balanced repeated holdout splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--spec", required=True, help="Repeated-holdout split-set YAML")
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Generate in memory only")
    parser.add_argument(
        "--replace-incomplete",
        action="store_true",
        help="Replace a partial output directory only when no SPLITS_FROZEN.json exists",
    )
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
        spec = load_repeated_holdout_spec(spec_path, repository_root=root)
        metadata = load_combined_metadata(spec)
        result = generate_repeated_holdout(metadata, spec)

        print("RA-ILD TRB repeated holdout generation")
        print(f"Split set:        {spec.split_set_id}")
        print(f"Samples:          {len(metadata)}")
        print(f"Patients:         {metadata['patient_id'].nunique()}")
        print(f"Repeats:          {spec.repeats}")
        print(f"Per split:        train={spec.train_size}, holdout={spec.holdout_size}")
        print(f"Exact strata:     {', '.join(spec.exact_strata)}")
        print(f"Candidates/repeat:{spec.candidates_per_repeat}")
        print("Selected candidates:")
        for row in result.selected_candidates.itertuples(index=False):
            print(
                f"  {row.split_id}: seed={row.candidate_seed}, "
                f"balance={row.base_balance_score:.8f}, "
                f"max_previous_jaccard={row.max_previous_holdout_jaccard:.6f}"
            )
        if args.dry_run:
            print("Dry run only; no split files were written.")
            return 0

        marker = write_frozen_split_set(
            result,
            spec,
            replace_incomplete=args.replace_incomplete,
        )
        print("Split generation: COMPLETE AND FROZEN")
        print(f"Output:           {spec.output_root}")
        print(f"Assignments:      {marker['files']['assignments']['path']}")
        print(f"Assignment SHA256:{marker['files']['assignments']['sha256']}")
        return 0
    except Exception as exc:
        print(f"RA-ILD TRB repeated holdout generation: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
