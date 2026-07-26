#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the Batch 07 outer-task manifest and every currently discovered task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.config import load_experiment_config  # noqa: E402
from ra_ild_igh.outer_cv import build_outer_tasks, manifest_frame, scan_outer_tasks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check IGH V2 outer-task orchestration and current task outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--require-all", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks = []

    def record(name, passed, detail):
        checks.append((name, bool(passed), str(detail)))

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
        tasks = build_outer_tasks(int(cv["outer_repeats"]), int(cv["outer_folds"]), output_root)
        expected_tasks = int(config.raw["nested_cv"]["expected_outer_tasks"])
        record("Outer task count", len(tasks) == expected_tasks, f"observed={len(tasks)}, expected={expected_tasks}")
        record("Unique task IDs", len({task.task_id for task in tasks}) == len(tasks), f"unique={len({task.task_id for task in tasks})}")
        record("Repeat coverage", {task.outer_repeat for task in tasks} == set(range(1, int(cv["outer_repeats"]) + 1)), f"repeats=1..{cv['outer_repeats']}")
        record("Fold coverage", {task.outer_fold for task in tasks} == set(range(1, int(cv["outer_folds"]) + 1)), f"folds=1..{cv['outer_folds']}")
        manifest = manifest_frame(
            tasks,
            python_executable=sys.executable,
            runner_script=config.path("outer_tasks.runner_script", must_exist=True, expect="file"),
            config_path=config.source_path,
        )
        record("Manifest row count", len(manifest) == len(tasks), f"rows={len(manifest)}")
        command_ok = manifest.apply(
            lambda row: (
                f"--outer-repeat {int(row['outer_repeat'])}" in row["command"]
                and f"--outer-fold {int(row['outer_fold'])}" in row["command"]
                and "--output-dir" in row["command"]
            ),
            axis=1,
        ).all()
        record("Manifest command mapping", command_ok, "repeat/fold/output-dir encoded")
        v1_root = (config.repository_root / "IGH/set/train/result/05_modeling").resolve()
        record("V2 output isolation", v1_root not in output_root.parents and output_root != v1_root, f"output_root={output_root}")

        status = scan_outer_tasks(
            tasks,
            expected_models=models,
            expected_samples=int(config.raw["data"]["train"]["expected_samples"]),
            expected_candidates_per_model=len(engine["alpha_grid"]) * len(engine["lambda_grid"]),
            expected_static_features=int(selection["expected_static_feature_count"]),
        )
        record("Status row count", len(status) == len(tasks), f"rows={len(status)}")
        allowed = {"complete", "missing", "incomplete", "invalid"}
        record("Status values", set(status["status"]).issubset(allowed), f"observed={sorted(set(status['status']))}")
        invalid_count = int(status["status"].eq("invalid").sum())
        incomplete_count = int(status["status"].eq("incomplete").sum())
        complete_count = int(status["status"].eq("complete").sum())
        record("No invalid discovered tasks", invalid_count == 0, f"invalid={invalid_count}")
        record("No incomplete discovered tasks", incomplete_count == 0, f"incomplete={incomplete_count}")
        minimum_complete = int(outer["minimum_complete_tasks_for_validation"])
        record("Minimum complete tasks", complete_count >= minimum_complete, f"complete={complete_count}, minimum={minimum_complete}")
        if args.require_all:
            record("All outer tasks complete", complete_count == len(tasks), f"complete={complete_count}, expected={len(tasks)}")

        passed = sum(item[1] for item in checks)
        print(f"Checks={len(checks)} PASS={passed} FAIL={len(checks) - passed}")
        for name, ok, detail in checks:
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        return 0 if passed == len(checks) else 1
    except Exception as exc:
        print(f"Outer orchestration check: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
