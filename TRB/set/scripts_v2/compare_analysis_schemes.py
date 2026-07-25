#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare completed schemes only when they share the exact frozen holdout set."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.repeated_holdout_summary import compare_aggregated_schemes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare completed repeated-holdout scheme summaries.")
    parser.add_argument("--aggregation-dir", action="append", required=True, help="Repeat for every scheme")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        output = Path(args.output_dir).expanduser().resolve()
        files = {
            "summary": output / "09_scheme_metric_summary.csv",
            "paired": output / "09_scheme_pairwise_metric_differences.csv",
            "audit": output / "09_scheme_comparison_audit.json",
            "complete": output / "09_SCHEME_COMPARISON_COMPLETE.json",
        }
        existing = [path for path in files.values() if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError("Scheme comparison outputs already exist; add --overwrite for a documented rerun")
        summary, paired, audit = compare_aggregated_schemes([Path(x) for x in args.aggregation_dir])
        output.mkdir(parents=True, exist_ok=True)
        summary.to_csv(files["summary"], index=False)
        paired.to_csv(files["paired"], index=False)
        files["audit"].write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        marker = {**audit, "status": "COMPLETE", "output_dir": str(output)}
        files["complete"].write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("TRB scheme comparison: COMPLETE")
        print(f"Schemes:          {audit['scheme_count']}")
        print(f"Same assignments: {audit['same_frozen_assignments']}")
        print(f"Summary rows:     {len(summary)}")
        print(f"Paired rows:      {len(paired)}")
        print(f"Output:           {output}")
        return 0
    except Exception as exc:
        print(f"TRB scheme comparison: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
