#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate completed TRB Batch 18 XGBoost repeated-holdout tasks."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.xgboost_config import load_xgboost_config  # noqa: E402
from ra_ild_trb.xgboost_summary import discover_completed_tasks, write_aggregation  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate TRB XGBoost repeated-holdout results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--task-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_xgboost_config(
            Path(args.config),
            repository_root=Path(args.repository_root) if args.repository_root else None,
        )
        task_root = (
            Path(args.task_root).expanduser().resolve()
            if args.task_root
            else config.path("outer_tasks.output_root", must_exist=True, expect="dir")
        )
        completed = discover_completed_tasks(task_root)
        expected = int(config.section("outer_tasks")["expected_tasks"])
        if len(completed) != expected:
            raise RuntimeError(
                f"Refusing partial aggregation: completed_tasks={len(completed)}, expected_tasks={expected}. "
                "Finish/repair the configured outer tasks before aggregation."
            )
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else config.path("outer_tasks.output_root").parent / "03_summary"
        )
        paths = write_aggregation(task_root, output_dir, overwrite=args.overwrite)
        print("TRB XGBoost repeated-holdout aggregation: COMPLETE")
        print(f"Task root: {task_root}")
        print(f"Completed tasks: {len(completed)}/{expected}")
        print(f"Output: {output_dir}")
        for key, path in paths.items():
            print(f"  {key:24s} {path}")
        return 0
    except Exception as exc:
        print(f"TRB XGBoost repeated-holdout aggregation: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
