#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create, inspect, and safely execute all TRB V2 outer nested-CV tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.config import load_experiment_config  # noqa: E402
from ra_ild_trb.outer_cv import (  # noqa: E402
    OuterCVError,
    build_outer_tasks,
    execute_outer_tasks,
    manifest_frame,
    scan_outer_tasks,
)


def parse_task_filter(text: str) -> Tuple[Tuple[int, int], ...]:
    if not text.strip():
        return tuple()
    values = []
    for item in text.split(","):
        parts = item.strip().split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                "--tasks must use repeat:fold pairs, e.g. 1:1,1:2"
            )
        try:
            repeat, fold = (int(parts[0]), int(parts[1]))
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Task repeat/fold values must be integers") from exc
        if repeat < 1 or fold < 1:
            raise argparse.ArgumentTypeError("Task repeat/fold values must be >= 1")
        values.append((repeat, fold))
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("--tasks contains duplicate repeat:fold pairs")
    return tuple(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect or execute the full TRB V2 outer-task manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--tasks", type=parse_task_filter, default=tuple())
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rerun-incomplete", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    return parser.parse_args()


def _subset_tasks(tasks: Sequence, requested):
    selected = list(tasks)
    if requested:
        requested_set = set(requested)
        selected = [
            task for task in selected
            if (task.outer_repeat, task.outer_fold) in requested_set
        ]
        observed = {(task.outer_repeat, task.outer_fold) for task in selected}
        missing = sorted(requested_set - observed)
        if missing:
            raise OuterCVError(f"Requested tasks outside configured range: {missing}")
    return tuple(selected)


def main() -> int:
    args = parse_args()
    try:
        config = load_experiment_config(
            Path(args.config),
            repository_root=Path(args.repository_root) if args.repository_root else None,
        )
        cv = config.section("cross_validation")
        engine = config.section("model_engine")
        models = tuple(config.section("models"))
        selection = config.section("model_selection")
        outer = config.section("outer_tasks")

        output_root = config.path("outer_tasks.output_root")
        runner_script = config.path("outer_tasks.runner_script", must_exist=True, expect="file")
        manifest_path = config.path("outer_tasks.manifest")
        status_path = config.path("outer_tasks.status")
        log_dir = config.path("outer_tasks.logs_dir")
        all_tasks = build_outer_tasks(
            int(cv["outer_repeats"]), int(cv["outer_folds"]), output_root
        )
        tasks = _subset_tasks(all_tasks, args.tasks)
        manifest = manifest_frame(
            all_tasks,
            python_executable=args.python_executable,
            runner_script=runner_script,
            config_path=config.source_path,
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(manifest_path, index=False)

        status = scan_outer_tasks(
            all_tasks,
            expected_models=models,
            expected_samples=int(config.raw["data"]["train"]["expected_samples"]),
            expected_candidates_per_model=len(engine["alpha_grid"]) * len(engine["lambda_grid"]),
            expected_static_features=int(selection["expected_static_feature_count"]),
        )
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status.to_csv(status_path, index=False)
        counts = status["status"].value_counts().to_dict()
        print("TRB V2 outer-task status")
        print(f"Manifest: {manifest_path}")
        print(f"Status:   {status_path}")
        print(
            "Counts: "
            + ", ".join(
                f"{name}={int(counts.get(name, 0))}"
                for name in ("complete", "missing", "incomplete", "invalid")
            )
        )

        if args.status_only or not args.execute:
            if not args.execute:
                print("No tasks executed. Add --execute to start selected pending tasks.")
            return 0

        selected_ids = {task.task_id for task in tasks}
        selected_status = status.loc[status["task_id"].isin(selected_ids)].copy()
        if args.limit is not None:
            if args.limit < 1:
                raise OuterCVError("--limit must be >= 1")
            pending_ids = selected_status.loc[
                selected_status["status"] != "complete", "task_id"
            ].astype(str).tolist()[: args.limit]
            tasks = tuple(task for task in tasks if task.task_id in set(pending_ids))
            selected_status = selected_status.loc[
                selected_status["task_id"].isin(pending_ids)
            ].copy()
        workers = int(args.workers or outer["default_workers"])
        max_workers = int(outer["max_workers"])
        if workers > max_workers:
            raise OuterCVError(
                f"Requested workers={workers} exceeds configured max_workers={max_workers}."
            )
        execution = execute_outer_tasks(
            tasks,
            selected_status,
            python_executable=args.python_executable,
            runner_script=runner_script,
            config_path=config.source_path,
            log_dir=log_dir,
            max_workers=workers,
            rerun_incomplete=args.rerun_incomplete,
        )
        execution_path = output_root / "07_outer_task_execution_log.csv"
        execution.to_csv(execution_path, index=False)
        print(f"Execution log: {execution_path}")
        if not execution.empty:
            print("Actions:")
            for name, count in execution["action"].value_counts().items():
                print(f"  {name}: {int(count)}")
        failed = execution["action"].isin(["failed", "scheduler_exception"]).any()
        blocked = execution["action"].eq("blocked_incomplete_or_invalid").any()
        return 1 if failed or blocked else 0
    except Exception as exc:
        print(f"TRB V2 outer-task manager: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
