#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect and execute all IGH Linear SVM repeated-holdout tasks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.linear_svm_config import (  # noqa: E402
    expand_linear_svm_candidates,
    load_linear_svm_config,
)

DEFAULT_WORKERS = 1
MAX_WORKERS = 8


@dataclass(frozen=True)
class Task:
    repeat: int
    fold: int
    output_dir: Path

    @property
    def key(self) -> str:
        return f"{self.repeat}:{self.fold}"


def parse_task_filter(text: str) -> Tuple[Tuple[int, int], ...]:
    if not text.strip():
        return tuple()
    result = []
    for item in text.split(","):
        parts = item.strip().split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                "--tasks must use repeat:fold pairs, e.g. 1:1,2:1"
            )
        repeat, fold = int(parts[0]), int(parts[1])
        if repeat < 1 or fold < 1:
            raise argparse.ArgumentTypeError("repeat/fold values must be >= 1")
        result.append((repeat, fold))
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("--tasks contains duplicates")
    return tuple(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect or execute IGH Linear SVM outer tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--tasks", type=parse_task_filter, default=tuple())
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rerun-incomplete", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    return parser.parse_args()


def build_tasks(config) -> Tuple[Task, ...]:
    cv = config.section("cross_validation")
    outer_root = config.path("outer_tasks.output_root")
    folds = cv.get("executed_outer_folds")
    if folds is None:
        folds = list(range(1, int(cv["outer_folds"]) + 1))
    return tuple(
        Task(
            repeat=repeat,
            fold=int(fold),
            output_dir=outer_root / f"repeat_{repeat:02d}_fold_{int(fold):02d}",
        )
        for repeat in range(1, int(cv["outer_repeats"]) + 1)
        for fold in folds
    )


def task_status(task: Task, *, expected_models: Sequence[str], candidate_count: int) -> dict:
    marker = task.output_dir / "06_TASK_COMPLETE.json"
    metrics_path = task.output_dir / "06_outer_validation_metrics.csv"
    tuning_path = task.output_dir / "06_inner_tuning_results.csv"
    if not task.output_dir.exists():
        status = "missing"
        detail = "task directory missing"
    elif not marker.is_file():
        status = "incomplete"
        detail = "completion marker missing"
    else:
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if payload.get("status") != "COMPLETE" or payload.get("engine") != "linear_svc":
                raise ValueError("invalid completion marker")
            metrics = pd.read_csv(metrics_path)
            tuning = pd.read_csv(tuning_path)
            observed_models = set(metrics["model"].astype(str))
            if observed_models != set(map(str, expected_models)):
                raise ValueError("metric model set mismatch")
            counts = tuning.groupby(tuning["model"].astype(str)).size().to_dict()
            if any(int(counts.get(name, 0)) != int(candidate_count) for name in expected_models):
                raise ValueError("candidate count mismatch")
            status = "complete"
            detail = ""
        except Exception as exc:
            status = "invalid"
            detail = str(exc)
    return {
        "outer_repeat": task.repeat,
        "outer_fold": task.fold,
        "task_key": task.key,
        "output_dir": str(task.output_dir),
        "status": status,
        "detail": detail,
    }


def write_status(config, tasks: Sequence[Task], candidate_count: int) -> pd.DataFrame:
    models = tuple(config.section("models"))
    frame = pd.DataFrame(
        [
            task_status(
                task, expected_models=models, candidate_count=candidate_count
            )
            for task in tasks
        ]
    )
    status_path = config.path("outer_tasks.status")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(status_path, index=False)
    return frame


def write_manifest(config, tasks: Sequence[Task], python_executable: str) -> None:
    runner = config.path("outer_tasks.runner_script", must_exist=True, expect="file")
    manifest = pd.DataFrame(
        [
            {
                "outer_repeat": task.repeat,
                "outer_fold": task.fold,
                "output_dir": str(task.output_dir),
                "command": " ".join(
                    [
                        python_executable,
                        str(runner),
                        "--config",
                        str(config.source_path),
                        "--repository-root",
                        str(config.repository_root),
                        "--outer-repeat",
                        str(task.repeat),
                        "--outer-fold",
                        str(task.fold),
                        "--output-dir",
                        str(task.output_dir),
                    ]
                ),
            }
            for task in tasks
        ]
    )
    path = config.path("outer_tasks.manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)


def select_tasks(
    all_tasks: Sequence[Task],
    status: pd.DataFrame,
    requested: Sequence[Tuple[int, int]],
    limit: int,
    rerun_incomplete: bool,
) -> Tuple[Task, ...]:
    selected = list(all_tasks)
    if requested:
        requested_set = set(requested)
        selected = [task for task in selected if (task.repeat, task.fold) in requested_set]
        observed = {(task.repeat, task.fold) for task in selected}
        missing = sorted(requested_set - observed)
        if missing:
            raise ValueError(f"Requested tasks outside configured range: {missing}")
    status_lookup = {
        (int(row.outer_repeat), int(row.outer_fold)): str(row.status)
        for row in status.itertuples(index=False)
    }
    allowed = {"missing"}
    if rerun_incomplete:
        allowed.update({"incomplete", "invalid"})
    selected = [
        task
        for task in selected
        if status_lookup.get((task.repeat, task.fold), "missing") in allowed
    ]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be >= 1")
        selected = selected[:limit]
    return tuple(selected)


def run_one(config, task: Task, python_executable: str, overwrite: bool) -> Tuple[Task, int]:
    runner = config.path("outer_tasks.runner_script", must_exist=True, expect="file")
    log_dir = config.path("outer_tasks.logs_dir")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"repeat_{task.repeat:02d}_fold_{task.fold:02d}.log"
    command = [
        python_executable,
        str(runner),
        "--config",
        str(config.source_path),
        "--repository-root",
        str(config.repository_root),
        "--outer-repeat",
        str(task.repeat),
        "--outer-fold",
        str(task.fold),
        "--output-dir",
        str(task.output_dir),
    ]
    if overwrite:
        command.append("--overwrite")
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=config.repository_root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return task, int(completed.returncode)


def main() -> int:
    args = parse_args()
    try:
        if not 1 <= int(args.workers) <= MAX_WORKERS:
            raise ValueError(f"--workers must be between 1 and {MAX_WORKERS}")
        config = load_linear_svm_config(
            Path(args.config),
            repository_root=Path(args.repository_root)
            if args.repository_root
            else None,
        )
        candidates = expand_linear_svm_candidates(config.section("model_engine"))
        tasks = build_tasks(config)
        if len(tasks) != int(config.section("outer_tasks")["expected_tasks"]):
            raise ValueError("Configured task count does not match outer_tasks.expected_tasks")
        write_manifest(config, tasks, args.python_executable)
        status = write_status(config, tasks, len(candidates))
        counts = status["status"].value_counts().to_dict()
        print("IGH Linear SVM outer-task status")
        print(f"Manifest: {config.path('outer_tasks.manifest')}")
        print(f"Status:   {config.path('outer_tasks.status')}")
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

        selected = select_tasks(
            tasks,
            status,
            args.tasks,
            args.limit,
            args.rerun_incomplete,
        )
        if not selected:
            print("No eligible tasks selected for execution.")
            return 0
        print(f"Executing {len(selected)} task(s) with workers={args.workers}")
        failed = []
        with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = {
                executor.submit(
                    run_one,
                    config,
                    task,
                    args.python_executable,
                    args.rerun_incomplete,
                ): task
                for task in selected
            }
            for future in as_completed(futures):
                task, returncode = future.result()
                label = "PASS" if returncode == 0 else "FAIL"
                print(f"[{label}] repeat={task.repeat}, fold={task.fold}")
                if returncode != 0:
                    failed.append(task.key)
        final_status = write_status(config, tasks, len(candidates))
        if failed:
            raise RuntimeError(f"Linear SVM tasks failed: {failed}")
        print("IGH Linear SVM task execution: COMPLETE")
        return 0
    except Exception as exc:
        print(f"IGH Linear SVM task orchestration: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
