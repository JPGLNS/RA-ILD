#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Globally memory-guarded scheduler for the TRB V3 five-algorithm task matrix.

Runs the existing one-task entry points without changing their scientific code.
It schedules all 15 cohort/algorithm schemes through one process pool, so the
50-worker and 120-GiB limits apply globally rather than separately per engine.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = Path("TRB/set/configs/schemes/generated/hierarchical_igh_aligned_five_algorithm/FIVE_ALGORITHM_RUN_MANIFEST.json")


@dataclass(frozen=True)
class Task:
    cohort: str
    algorithm: str
    repeat: int
    fold: int
    config_path: Path
    runner_path: Path
    output_dir: Path
    expected_models: tuple[str, ...]
    candidate_count: int
    memory_gb: float

    @property
    def key(self) -> str:
        return f"{self.cohort}:{self.algorithm}:{self.repeat:03d}:{self.fold:02d}"


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Resolved config root is not a mapping: {path}")
    return dict(value)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _available_gb() -> float:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        raise RuntimeError("Memory-guarded runner requires Linux /proc/meminfo")
    values = {}
    for line in meminfo.read_text(encoding="ascii").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        parts = raw.strip().split()
        if parts and parts[0].isdigit():
            values[key] = int(parts[0]) * 1024
    available = values.get("MemAvailable")
    if available is None:
        raise RuntimeError("MemAvailable is missing from /proc/meminfo")
    return available / (1024 ** 3)


def _task_complete(task: Task) -> tuple[bool, str]:
    marker = task.output_dir / "06_TASK_COMPLETE.json"
    metrics_path = task.output_dir / "06_outer_validation_metrics.csv"
    predictions_path = task.output_dir / "06_outer_validation_predictions.csv"
    tuning_path = task.output_dir / "06_inner_tuning_results.csv"
    required = [marker, metrics_path, predictions_path, tuning_path]
    missing = [str(p.name) for p in required if not p.is_file()]
    if missing:
        return False, "missing files: " + ",".join(missing)
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE":
            return False, f"marker status={payload.get('status')!r}"
        expected_engine = {
            "linear_svm": "linear_svc",
            "xgboost": "xgboost",
        }.get(task.algorithm)
        if expected_engine and payload.get("engine") != expected_engine:
            return False, f"engine marker mismatch: {payload.get('engine')!r}"
        metrics = pd.read_csv(metrics_path)
        observed = tuple(metrics["model"].astype(str).drop_duplicates())
        if set(observed) != set(task.expected_models) or len(metrics) != len(task.expected_models):
            return False, f"model metrics mismatch: {observed}"
        tuning = pd.read_csv(tuning_path)
        counts = tuning.groupby(tuning["model"].astype(str)).size().to_dict()
        bad = {
            model: int(counts.get(model, 0))
            for model in task.expected_models
            if int(counts.get(model, 0)) != task.candidate_count
        }
        if bad:
            return False, f"candidate rows mismatch: {bad}"
        return True, ""
    except Exception as exc:
        return False, f"completion audit error: {exc}"


def _build_tasks(root: Path, manifest: Mapping[str, Any], overrides: Mapping[str, float]) -> list[Task]:
    entries = list(manifest.get("entries", []))
    if len(entries) != 15:
        raise ValueError(f"Expected 15 cohort/algorithm entries, observed {len(entries)}")
    by_entry: list[list[Task]] = []
    for entry in entries:
        config_path = _resolve(root, str(entry["config"]))
        if not config_path.is_file():
            raise FileNotFoundError(f"Resolved experiment config missing: {config_path}")
        config = _load_yaml(config_path)
        cv = config["cross_validation"]
        if int(cv["outer_repeats"]) != 100:
            raise ValueError(f"{entry['cohort']}/{entry['algorithm']}: outer_repeats != 100")
        folds = tuple(int(x) for x in cv.get("executed_outer_folds", [1]))
        if folds != (1,):
            raise ValueError(f"{entry['cohort']}/{entry['algorithm']}: expected executed fold [1]")
        if int(config["data"]["train"]["expected_samples"]) != int(entry["expected_samples"]):
            raise ValueError(f"{entry['cohort']}/{entry['algorithm']}: sample count mismatch")
        models = tuple(config["models"].keys())
        if models != tuple(entry["expected_models"]):
            raise ValueError(f"{entry['cohort']}/{entry['algorithm']}: model contract mismatch")
        runner = _resolve(root, str(config["outer_tasks"]["runner_script"]))
        if not runner.is_file():
            raise FileNotFoundError(f"Task runner missing: {runner}")
        out_root = _resolve(root, str(config["outer_tasks"]["output_root"]))
        algorithm = str(entry["algorithm"])
        engine = config["model_engine"]
        if algorithm in {"elastic_net", "ridge", "lasso"}:
            candidate_count = len(engine["alpha_grid"]) * len(engine["lambda_grid"])
            expected_alpha = {"ridge": [0.0], "lasso": [1.0]}
            if algorithm in expected_alpha and list(map(float, engine["alpha_grid"])) != expected_alpha[algorithm]:
                raise ValueError(f"{algorithm}: alpha endpoint mismatch")
        elif algorithm == "linear_svm":
            from importlib.util import spec_from_file_location
            # Candidate count is frozen in the prepared SVM marker/config.
            marker = config_path.parent / "LINEAR_SVM_SCHEME_PREPARED.json"
            if not marker.is_file():
                raise FileNotFoundError(f"Linear SVM prepared marker missing: {marker}")
            candidate_count = int(json.loads(marker.read_text())["candidate_count"])
        elif algorithm == "xgboost":
            marker = config_path.parent / "XGBOOST_SCHEME_PREPARED.json"
            if not marker.is_file():
                raise FileNotFoundError(f"XGBoost prepared marker missing: {marker}")
            candidate_count = int(json.loads(marker.read_text())["candidate_count"])
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        memory = float(overrides.get(algorithm, {
            "elastic_net": 3.0, "ridge": 3.0, "lasso": 3.0,
            "linear_svm": 3.0, "xgboost": 5.0,
        }[algorithm]))
        if memory <= 0:
            raise ValueError(f"Invalid memory reservation for {algorithm}: {memory}")
        tasks = []
        for repeat in range(1, 101):
            tasks.append(Task(
                cohort=str(entry["cohort"]),
                algorithm=algorithm,
                repeat=repeat,
                fold=1,
                config_path=config_path,
                runner_path=runner,
                output_dir=out_root / f"repeat_{repeat:02d}_fold_01",
                expected_models=models,
                candidate_count=int(candidate_count),
                memory_gb=memory,
            ))
        by_entry.append(tasks)
    # Round-robin cohort/algorithm schemes by repeat to avoid starving one engine.
    ordered: list[Task] = []
    for repeat_index in range(100):
        for group in by_entry:
            ordered.append(group[repeat_index])
    return ordered


class MemoryAdmission:
    def __init__(self, max_workers: int, budget_gb: float, min_free_gb: float):
        self.max_workers = max_workers
        self.budget_gb = budget_gb
        self.min_free_gb = min_free_gb
        self.active = 0
        self.reserved = 0.0
        self.condition = threading.Condition()

    def acquire(self, amount: float) -> None:
        with self.condition:
            while True:
                available = _available_gb()
                if (
                    self.active < self.max_workers
                    and self.reserved + amount <= self.budget_gb
                    and available - amount >= self.min_free_gb
                ):
                    self.active += 1
                    self.reserved += amount
                    return
                self.condition.wait(timeout=10.0)

    def release(self, amount: float) -> None:
        with self.condition:
            self.active -= 1
            self.reserved -= amount
            self.condition.notify_all()


def _run_one(task: Task, root: Path, python: str, overwrite: bool, admission: MemoryAdmission) -> dict[str, Any]:
    admission.acquire(task.memory_gb)
    started = time.time()
    log_dir = task.output_dir.parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task.output_dir.name}.log"
    command = [
        python, str(task.runner_path),
        "--config", str(task.config_path),
        "--repository-root", str(root),
        "--outer-repeat", str(task.repeat),
        "--outer-fold", str(task.fold),
        "--output-dir", str(task.output_dir),
    ]
    if overwrite:
        command.append("--overwrite")
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command, cwd=root, stdout=handle, stderr=subprocess.STDOUT,
                text=True, check=False,
            )
        return {
            "task": task.key, "cohort": task.cohort, "algorithm": task.algorithm,
            "repeat": task.repeat, "fold": task.fold,
            "status": "complete" if completed.returncode == 0 else "failed",
            "returncode": int(completed.returncode),
            "runtime_seconds": round(time.time() - started, 3),
            "memory_reservation_gb": task.memory_gb,
            "log": str(log_path),
        }
    finally:
        admission.release(task.memory_gb)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--max-workers", type=int, default=50)
    parser.add_argument("--memory-budget-gb", type=float, default=120.0)
    parser.add_argument("--min-free-gb", type=float, default=12.0)
    parser.add_argument("--reserve-elastic-net-gb", type=float, default=3.0)
    parser.add_argument("--reserve-ridge-gb", type=float, default=3.0)
    parser.add_argument("--reserve-lasso-gb", type=float, default=3.0)
    parser.add_argument("--reserve-linear-svm-gb", type=float, default=3.0)
    parser.add_argument("--reserve-xgboost-gb", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rerun-incomplete", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = Path(args.repository_root).expanduser().resolve()
    manifest_path = _resolve(root, args.manifest)
    try:
        if not 1 <= args.max_workers <= 50:
            raise ValueError("--max-workers must be between 1 and 50")
        if args.memory_budget_gb <= 0 or args.min_free_gb < 0:
            raise ValueError("Invalid memory budget/minimum free memory")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        reservations = {
            "elastic_net": args.reserve_elastic_net_gb,
            "ridge": args.reserve_ridge_gb,
            "lasso": args.reserve_lasso_gb,
            "linear_svm": args.reserve_linear_svm_gb,
            "xgboost": args.reserve_xgboost_gb,
        }
        tasks = _build_tasks(root, manifest, reservations)
        if len(tasks) != 1500:
            raise ValueError(f"Expected 1500 task slots, observed {len(tasks)}")
        rows = []
        pending = []
        blocked = []
        for task in tasks:
            complete, detail = _task_complete(task)
            if complete:
                rows.append({
                    "task": task.key, "cohort": task.cohort, "algorithm": task.algorithm,
                    "repeat": task.repeat, "fold": task.fold, "status": "already_complete",
                    "detail": "", "output_dir": str(task.output_dir),
                })
            elif task.output_dir.exists() and not args.rerun_incomplete:
                blocked.append((task, detail))
            else:
                pending.append(task)
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("--limit must be >= 1")
            pending = pending[:args.limit]
        print("TRB V3 five-algorithm global task scheduler")
        print(f"Task slots: {len(tasks)}")
        print(f"Already complete: {len(rows)}")
        print(f"Pending: {len(pending)}")
        print(f"Blocked incomplete/invalid: {len(blocked)}")
        print(f"Max workers: {args.max_workers}")
        print(f"Memory reservation budget: {args.memory_budget_gb:.1f} GiB")
        print(f"Current MemAvailable: {_available_gb():.1f} GiB")
        print(f"Minimum free-memory guard: {args.min_free_gb:.1f} GiB")
        print("Reservations (GiB/task): " + json.dumps(reservations, sort_keys=True))
        for task, detail in blocked[:20]:
            print(f"[BLOCKED] {task.key}: {detail}")
        if not args.execute:
            print("DRY-RUN: no task processes launched; add --execute to start pending tasks.")
            return 0
        if blocked:
            raise RuntimeError(
                "Incomplete/invalid task directories exist. Inspect them and use "
                "--rerun-incomplete only when an intentional overwrite is desired."
            )
        admission = MemoryAdmission(args.max_workers, args.memory_budget_gb, args.min_free_gb)
        failures = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {
                pool.submit(_run_one, task, root, args.python_executable,
                            args.rerun_incomplete, admission): task
                for task in pending
            }
            for future in as_completed(futures):
                result = future.result()
                rows.append(result)
                print(f"[{result['status'].upper()}] {result['task']} "
                      f"runtime={result['runtime_seconds']}s")
                if result["status"] != "complete":
                    failures.append(result["task"])
        status_path = root / "TRB/set/experiments/trb_v3_179_igh_aligned_five_algorithm_global/07_GLOBAL_TASK_STATUS.csv"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(status_path, index=False)
        print(f"Status report: {status_path}")
        if failures:
            raise RuntimeError(f"Failed tasks: {failures[:30]}")
        print("GLOBAL FIVE-ALGORITHM TASK EXECUTION: COMPLETE")
        return 0
    except Exception as exc:
        print(f"GLOBAL FIVE-ALGORITHM SCHEDULER: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
