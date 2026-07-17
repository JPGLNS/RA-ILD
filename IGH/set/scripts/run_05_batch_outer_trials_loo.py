#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch-run all repeated nested-CV outer tasks for the RA / RA-ILD IGH project.

Default workload:
    20 repeats × 5 outer folds = 100 tasks

Each task calls:
    set/scripts/run_05_single_outer_trial_loo.py

Design goals:
- sequential execution by default (safer for memory-heavy sparse matrices)
- nohup-friendly unbuffered logging
- per-task stdout/stderr logs
- resume support: completed tasks are skipped
- failure isolation: one failed task does not stop the remaining tasks
- batch-level status CSV and summary Markdown
- lock file to avoid accidentally launching two batch runs at once
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


SCRIPT_VERSION = "1.1.0-IGH"
EXPECTED_RECEPTOR = "IGH"

ROOT = Path("/data/users/chenhaisheng/RA-ILD/IGH")
DEFAULT_WORKER = ROOT / "set/scripts/run_05_single_outer_trial_loo.py"
DEFAULT_RESULT_ROOT = (
    ROOT / "set/train/result/05_modeling/single_outer_trial_loo"
)
DEFAULT_BATCH_DIR = ROOT / "set/train/result/05_modeling/batch_run_loo"

EXPECTED_OUTER_TRAIN = 96
EXPECTED_OUTER_VALIDATION = 24

EXPECTED_TASK_FILES = [
    "05_trial_configuration.json",
    "05_trial_sample_roles.csv",
    "05_dynamic_public_features_outer_train_loo.csv",
    "05_dynamic_public_features_outer_validation.csv",
    "05_public_reference_build_summary.csv",
    "05_public_loo_assignments.csv",
    "05_inner_tuning_results.csv",
    "05_inner_selected_oof_predictions.csv",
    "05_outer_validation_predictions.csv",
    "05_outer_validation_metrics.csv",
    "05_final_model_coefficients.csv",
    "05_preprocessing_summary.csv",
    "05_static_igh_feature_list.csv",
    "05_single_outer_trial_summary.md",
]

EXPECTED_MODELS = {
    "M0_clinical",
    "M1_static_igh",
    "M2_static_igh_public",
    "M3_static_igh_public_material",
}


def parse_int_list(text: str) -> List[int]:
    text = text.strip()
    if not text:
        raise argparse.ArgumentTypeError("Empty integer list.")
    values: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            left, right = part.split("-", 1)
            start, end = int(left), int(right)
            if end < start:
                raise argparse.ArgumentTypeError(
                    f"Invalid range: {part}"
                )
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    return sorted(set(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run all IGH LOO outer-fold modeling tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--worker-script", default=str(DEFAULT_WORKER))
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--batch-dir", default=str(DEFAULT_BATCH_DIR))
    parser.add_argument(
        "--repeats",
        type=parse_int_list,
        default=parse_int_list("1-20"),
        help="Repeat IDs, e.g. 1-20 or 1,3,5.",
    )
    parser.add_argument(
        "--folds",
        type=parse_int_list,
        default=parse_int_list("1-5"),
        help="Outer fold IDs, e.g. 1-5.",
    )
    parser.add_argument(
        "--public-feature-set",
        default="raw_bilateral",
        choices=["raw_bilateral", "log_ratio", "all18"],
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to run the worker.",
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Run tasks even when their outputs pass completion checks.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately after the first failed task.",
    )
    parser.add_argument(
        "--ignore-lock",
        action="store_true",
        help="Ignore an existing batch lock file. Use cautiously.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned tasks without running them.",
    )
    parser.add_argument(
        "--extra-worker-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments passed to every worker task.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def task_name(repeat: int, fold: int) -> str:
    return f"repeat_{repeat:02d}_fold_{fold:02d}"


def task_dir(result_root: Path, repeat: int, fold: int) -> Path:
    return result_root / task_name(repeat, fold)


def task_log_paths(log_dir: Path, repeat: int, fold: int) -> Tuple[Path, Path]:
    stem = task_name(repeat, fold)
    return log_dir / f"{stem}.stdout.log", log_dir / f"{stem}.stderr.log"


def validate_completed_task(
    result_root: Path,
    repeat: int,
    fold: int,
    public_feature_set: str,
) -> Tuple[bool, str]:
    """Validate that an IGH outer-trial directory is safe to resume-skip."""
    out_dir = task_dir(result_root, repeat, fold)
    missing = [
        name for name in EXPECTED_TASK_FILES
        if not (out_dir / name).is_file()
    ]
    if missing:
        return False, f"missing files: {missing}"

    try:
        metrics = pd.read_csv(out_dir / "05_outer_validation_metrics.csv")
    except Exception as exc:
        return False, f"cannot read metrics: {exc}"

    required_cols = {
        "model", "fit_converged", "roc_auc", "pr_auc",
        "n_outer_train", "n_outer_validation",
    }
    if not required_cols.issubset(metrics.columns):
        return False, (
            "metrics missing columns: "
            f"{sorted(required_cols - set(metrics.columns))}"
        )

    if len(metrics) != len(EXPECTED_MODELS):
        return False, (
            f"expected {len(EXPECTED_MODELS)} metric rows, "
            f"observed {len(metrics)}"
        )

    models = set(metrics["model"].astype(str))
    if models != EXPECTED_MODELS:
        return False, f"unexpected IGH model set: {sorted(models)}"

    converged = (
        metrics["fit_converged"].astype(str).str.lower()
        .isin({"true", "1", "yes"})
    )
    if not converged.all():
        return False, "one or more final fits did not converge"

    for metric_name in ("roc_auc", "pr_auc"):
        values = pd.to_numeric(metrics[metric_name], errors="coerce")
        if values.isna().any() or ((values < 0) | (values > 1)).any():
            return False, f"invalid {metric_name} values"

    train_sizes = set(pd.to_numeric(metrics["n_outer_train"], errors="coerce"))
    valid_sizes = set(pd.to_numeric(metrics["n_outer_validation"], errors="coerce"))
    if train_sizes != {EXPECTED_OUTER_TRAIN}:
        return False, f"unexpected outer-training sizes: {sorted(train_sizes)}"
    if valid_sizes != {EXPECTED_OUTER_VALIDATION}:
        return False, f"unexpected outer-validation sizes: {sorted(valid_sizes)}"

    try:
        predictions = pd.read_csv(
            out_dir / "05_outer_validation_predictions.csv"
        )
    except Exception as exc:
        return False, f"cannot read predictions: {exc}"

    prediction_required = {
        "model", "sample_id", "outer_repeat", "outer_fold",
        "true_label", "probability_ILD",
    }
    if not prediction_required.issubset(predictions.columns):
        return False, (
            "predictions missing columns: "
            f"{sorted(prediction_required - set(predictions.columns))}"
        )
    if set(predictions["model"].astype(str)) != EXPECTED_MODELS:
        return False, "prediction model set mismatch"
    expected_prediction_rows = EXPECTED_OUTER_VALIDATION * len(EXPECTED_MODELS)
    if len(predictions) != expected_prediction_rows:
        return False, (
            f"expected {expected_prediction_rows} prediction rows, "
            f"observed {len(predictions)}"
        )
    per_model = predictions.groupby("model").size().to_dict()
    if any(per_model.get(model, 0) != EXPECTED_OUTER_VALIDATION for model in EXPECTED_MODELS):
        return False, f"unexpected prediction rows per model: {per_model}"
    if predictions.duplicated(["model", "sample_id"]).any():
        return False, "duplicated model/sample_id predictions"
    probabilities = pd.to_numeric(
        predictions["probability_ILD"], errors="coerce"
    )
    if probabilities.isna().any() or ((probabilities < 0) | (probabilities > 1)).any():
        return False, "invalid probability_ILD values"

    try:
        config = json.loads(
            (out_dir / "05_trial_configuration.json")
            .read_text(encoding="utf-8")
        )
    except Exception as exc:
        return False, f"cannot read configuration: {exc}"

    if str(config.get("receptor", "")).upper() != EXPECTED_RECEPTOR:
        return False, f"configuration receptor is not {EXPECTED_RECEPTOR}"
    if int(config.get("outer_repeat", -1)) != repeat:
        return False, "configuration repeat mismatch"
    if int(config.get("outer_fold", -1)) != fold:
        return False, "configuration fold mismatch"
    if config.get("public_feature_set") != public_feature_set:
        return False, (
            "configuration public-feature-set mismatch: "
            f"expected {public_feature_set}, "
            f"observed {config.get('public_feature_set')}"
        )
    if int(config.get("n_outer_train", -1)) != EXPECTED_OUTER_TRAIN:
        return False, "configuration outer-training size mismatch"
    if int(config.get("n_outer_validation", -1)) != EXPECTED_OUTER_VALIDATION:
        return False, "configuration outer-validation size mismatch"

    return True, "all IGH expected files and checks passed"

def write_status_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    columns = [
        "task_order",
        "outer_repeat",
        "outer_fold",
        "task_name",
        "status",
        "start_time",
        "end_time",
        "runtime_seconds",
        "return_code",
        "completion_check",
        "stdout_log",
        "stderr_log",
        "result_dir",
        "message",
    ]
    df = pd.DataFrame(rows)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    df[columns].to_csv(path, index=False)


def write_summary(
    path: Path,
    rows: Sequence[Dict[str, object]],
    args: argparse.Namespace,
    started: str,
    ended: str,
    total_runtime: float,
) -> None:
    df = pd.DataFrame(rows)
    counts = (
        df["status"].value_counts().to_dict()
        if not df.empty and "status" in df.columns else {}
    )
    completed = int(counts.get("completed", 0))
    skipped = int(counts.get("skipped_completed", 0))
    failed = int(counts.get("failed", 0))
    dry = int(counts.get("dry_run", 0))

    lines = [
        "# 05 IGH LOO Batch Run Summary",
        "",
        f"- Script version: `{SCRIPT_VERSION}`",
        f"- Receptor: `{EXPECTED_RECEPTOR}`",
        f"- Host: `{socket.gethostname()}`",
        f"- Started: `{started}`",
        f"- Ended: `{ended}`",
        f"- Total runtime: **{total_runtime:.2f} seconds**",
        f"- Worker: `{args.worker_script}`",
        f"- Public feature set: `{args.public_feature_set}`",
        f"- Requested repeats: `{args.repeats}`",
        f"- Requested folds: `{args.folds}`",
        "",
        "## Status counts",
        "",
        f"- Newly completed: **{completed}**",
        f"- Skipped because already complete: **{skipped}**",
        f"- Failed: **{failed}**",
        f"- Dry-run tasks: **{dry}**",
        "",
        "## Failed tasks",
        "",
    ]
    if failed == 0:
        lines.append("- None")
    else:
        failed_rows = df[df["status"] == "failed"]
        for _, row in failed_rows.iterrows():
            lines.append(
                f"- `{row['task_name']}`: return code "
                f"`{row['return_code']}`; {row['message']}"
            )

    lines.extend([
        "",
        "## Resume behavior",
        "",
        "- Re-running the same command skips tasks whose expected outputs are complete and whose complete IGH outputs pass integrity checks and whose four final models converged.",
        "- Failed or incomplete tasks are run again automatically.",
        "- Use `--rerun-completed` only when intentionally replacing all completed task results.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def acquire_lock(lock_path: Path, ignore_lock: bool) -> None:
    if lock_path.exists() and not ignore_lock:
        content = lock_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(
            f"Batch lock already exists: {lock_path}\n{content}\n"
            "Another batch may be running. Remove the lock only after "
            "confirming that no batch process is active, or use --ignore-lock."
        )

    payload = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "created": now_iso(),
        "command": sys.argv,
    }
    lock_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    started_iso = now_iso()
    started_time = time.time()

    worker = Path(args.worker_script).expanduser().resolve()
    result_root = Path(args.result_root).expanduser().resolve()
    batch_dir = Path(args.batch_dir).expanduser().resolve()
    log_dir = batch_dir / "task_logs"
    status_path = batch_dir / "05_batch_run_status.csv"
    summary_path = batch_dir / "05_batch_run_summary.md"
    lock_path = batch_dir / "05_batch_run.lock"

    if not worker.is_file():
        raise FileNotFoundError(f"Worker script not found: {worker}")

    result_root.mkdir(parents=True, exist_ok=True)
    batch_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    acquire_lock(lock_path, args.ignore_lock)

    rows: List[Dict[str, object]] = []
    tasks = [
        (repeat, fold)
        for repeat in args.repeats
        for fold in args.folds
    ]

    print("=" * 80, flush=True)
    print("05 IGH LOO Batch Outer-Trial Runner", flush=True)
    print("=" * 80, flush=True)
    print(f"Receptor: {EXPECTED_RECEPTOR}", flush=True)
    print(f"Worker: {worker}", flush=True)
    print(f"Tasks: {len(tasks)}", flush=True)
    print(f"Result root: {result_root}", flush=True)
    print(f"Batch logs: {batch_dir}", flush=True)
    print(f"Public feature set: {args.public_feature_set}", flush=True)
    print(f"Started: {started_iso}", flush=True)
    print("", flush=True)

    try:
        for order, (repeat, fold) in enumerate(tasks, start=1):
            name = task_name(repeat, fold)
            out_dir = task_dir(result_root, repeat, fold)
            stdout_log, stderr_log = task_log_paths(
                log_dir, repeat, fold
            )
            complete, completion_reason = validate_completed_task(
                result_root, repeat, fold, args.public_feature_set
            )

            row: Dict[str, object] = {
                "task_order": order,
                "outer_repeat": repeat,
                "outer_fold": fold,
                "task_name": name,
                "status": "",
                "start_time": "",
                "end_time": "",
                "runtime_seconds": "",
                "return_code": "",
                "completion_check": completion_reason,
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
                "result_dir": str(out_dir),
                "message": "",
            }

            if complete and not args.rerun_completed:
                row.update({
                    "status": "skipped_completed",
                    "message": "existing task outputs passed completion checks",
                })
                rows.append(row)
                write_status_csv(status_path, rows)
                print(
                    f"[{order:03d}/{len(tasks):03d}] {name}: "
                    "SKIP (already complete)",
                    flush=True,
                )
                continue

            command = [
                args.python,
                "-u",
                str(worker),
                "--outer-repeat", str(repeat),
                "--outer-fold", str(fold),
                "--public-feature-set", args.public_feature_set,
                "--overwrite",
                *args.extra_worker_args,
            ]

            if args.dry_run:
                row.update({
                    "status": "dry_run",
                    "message": " ".join(command),
                })
                rows.append(row)
                write_status_csv(status_path, rows)
                print(
                    f"[{order:03d}/{len(tasks):03d}] {name}: "
                    f"DRY RUN\n  {' '.join(command)}",
                    flush=True,
                )
                continue

            task_started_iso = now_iso()
            task_started = time.time()
            row["start_time"] = task_started_iso

            print(
                f"[{order:03d}/{len(tasks):03d}] {name}: START "
                f"at {task_started_iso}",
                flush=True,
            )
            print(f"  stdout: {stdout_log}", flush=True)
            print(f"  stderr: {stderr_log}", flush=True)

            with stdout_log.open("w", encoding="utf-8") as stdout_handle, \
                 stderr_log.open("w", encoding="utf-8") as stderr_handle:
                process = subprocess.run(
                    command,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                    check=False,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )

            task_runtime = time.time() - task_started
            task_ended_iso = now_iso()
            complete_after, reason_after = validate_completed_task(
                result_root, repeat, fold, args.public_feature_set
            )

            row.update({
                "end_time": task_ended_iso,
                "runtime_seconds": round(task_runtime, 3),
                "return_code": process.returncode,
                "completion_check": reason_after,
            })

            if process.returncode == 0 and complete_after:
                row.update({
                    "status": "completed",
                    "message": "worker returned 0 and outputs passed checks",
                })
                print(
                    f"[{order:03d}/{len(tasks):03d}] {name}: COMPLETE "
                    f"({task_runtime:.1f}s)",
                    flush=True,
                )
            else:
                row.update({
                    "status": "failed",
                    "message": (
                        f"worker return code={process.returncode}; "
                        f"completion check={reason_after}"
                    ),
                })
                print(
                    f"[{order:03d}/{len(tasks):03d}] {name}: FAILED "
                    f"return_code={process.returncode}; {reason_after}",
                    flush=True,
                )

            rows.append(row)
            write_status_csv(status_path, rows)

            ended_iso = now_iso()
            write_summary(
                summary_path,
                rows,
                args,
                started_iso,
                ended_iso,
                time.time() - started_time,
            )

            if row["status"] == "failed" and args.stop_on_error:
                print("Stopping because --stop-on-error was set.", flush=True)
                return 1

    except KeyboardInterrupt:
        print("\nInterrupted by user.", flush=True)
        raise
    finally:
        ended_iso = now_iso()
        write_status_csv(status_path, rows)
        write_summary(
            summary_path,
            rows,
            args,
            started_iso,
            ended_iso,
            time.time() - started_time,
        )
        if lock_path.exists():
            lock_path.unlink()

    failed_count = sum(row["status"] == "failed" for row in rows)
    completed_count = sum(row["status"] == "completed" for row in rows)
    skipped_count = sum(
        row["status"] == "skipped_completed" for row in rows
    )

    print("", flush=True)
    print("[Batch completed]", flush=True)
    print(f"Newly completed: {completed_count}", flush=True)
    print(f"Skipped complete: {skipped_count}", flush=True)
    print(f"Failed: {failed_count}", flush=True)
    print(f"Status CSV: {status_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    print(f"Ended: {now_iso()}", flush=True)

    return 1 if failed_count else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
