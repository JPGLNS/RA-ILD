#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Batch-run all paired TRB+IGH LOO outer-fold modeling tasks.

Default workload
----------------
20 repeats × 5 outer folds = 100 tasks.

Each task calls ``run_05_single_outer_trial_loo_TRB_IGH_v1_0_0.py``.
The frozen patient-level outer assignments are the source of truth for each
outer-training/validation size, so folds of 23 and 24 validation patients are
both handled correctly for the 118-patient training cohort.

Design goals
------------
* Sequential execution by default, avoiding memory contention between two
  receptor-specific sparse caches and high-dimensional Elastic Net fits.
* Nohup-friendly, unbuffered logging with per-task stdout/stderr logs.
* Heartbeat messages while a long worker task is running.
* Resume support: only tasks passing strict joint-output integrity checks are
  skipped.
* Failure isolation: one failed task does not stop later tasks unless
  ``--stop-on-error`` is requested.
* Batch-level status CSV, summary Markdown and lock file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.0-PAIRED-TRB-IGH"
ANALYSIS_VIEW = "TRB_IGH"
CV_UNIT = "patient"

ROOT = Path("/data/users/chenhaisheng/RA-ILD/TRB_IGH")
DEFAULT_WORKER = (
    ROOT / "set/scripts/run_05_single_outer_trial_loo_TRB_IGH_v1_0_0.py"
)
DEFAULT_OUTER_ASSIGNMENTS = (
    ROOT / "set/train/result/05_modeling/cv_splits/05_outer_fold_assignments.csv"
)
DEFAULT_RESULT_ROOT = (
    ROOT / "set/train/result/05_modeling/single_outer_trial_loo_TRB_IGH"
)
DEFAULT_BATCH_DIR = (
    ROOT / "set/train/result/05_modeling/batch_run_loo_TRB_IGH"
)

EXPECTED_TASK_FILES = [
    "05_trial_configuration.json",
    "05_trial_sample_roles.csv",
    "05_dynamic_public_features_outer_train_loo_TRB_IGH.csv",
    "05_dynamic_public_features_outer_validation_TRB_IGH.csv",
    "05_public_reference_build_summary_TRB_IGH.csv",
    "05_public_loo_assignments_TRB_IGH.csv",
    "05_inner_tuning_results.csv",
    "05_inner_selected_oof_predictions.csv",
    "05_outer_validation_predictions.csv",
    "05_outer_validation_metrics.csv",
    "05_final_model_coefficients.csv",
    "05_preprocessing_summary.csv",
    "05_static_TRB_IGH_feature_list.csv",
    "05_single_outer_trial_summary.md",
]

EXPECTED_MODELS = {
    "M0_clinical",
    "M1_static_trb_igh",
    "M2_static_trb_igh_public",
    "M3_static_trb_igh_public_material",
}


class TaskValidationError(RuntimeError):
    """Raised internally when a completed task fails integrity validation."""


def parse_int_list(text: str) -> List[int]:
    text = text.strip()
    if not text:
        raise argparse.ArgumentTypeError("Empty integer list.")
    values: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start, end = int(left), int(right)
            if end < start:
                raise argparse.ArgumentTypeError(f"Invalid range: {part}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    if not values:
        raise argparse.ArgumentTypeError("No integer IDs were parsed.")
    return sorted(set(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run paired TRB+IGH leakage-controlled LOO outer-fold trials."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--worker-script", default=str(DEFAULT_WORKER))
    parser.add_argument(
        "--outer-assignments", default=str(DEFAULT_OUTER_ASSIGNMENTS)
    )
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--batch-dir", default=str(DEFAULT_BATCH_DIR))
    parser.add_argument(
        "--repeats",
        type=parse_int_list,
        default=parse_int_list("1-20"),
        help="Repeat IDs, for example 1-20 or 1,3,5.",
    )
    parser.add_argument(
        "--folds",
        type=parse_int_list,
        default=parse_int_list("1-5"),
        help="Outer-fold IDs, for example 1-5.",
    )
    parser.add_argument(
        "--public-feature-set",
        default="raw_bilateral",
        choices=["raw_bilateral", "log_ratio", "all18"],
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to invoke the single-task worker.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=300,
        help="Print a batch-log heartbeat this often while a worker is active; 0 disables it.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=5.0,
        help="Worker-process polling interval.",
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Run tasks even if existing outputs pass all completion checks.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately after the first failed task.",
    )
    parser.add_argument(
        "--ignore-lock",
        action="store_true",
        help="Ignore an existing batch lock file. Use only after checking processes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print commands without executing workers.",
    )
    parser.add_argument(
        "--extra-worker-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments after this flag are passed verbatim to every worker task.",
    )
    args = parser.parse_args()
    if args.heartbeat_seconds < 0:
        parser.error("--heartbeat-seconds must be >= 0.")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be > 0.")
    return args


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_name(repeat: int, fold: int) -> str:
    return f"repeat_{repeat:02d}_fold_{fold:02d}"


def public_result_dir(result_root: Path, public_feature_set: str) -> Path:
    return result_root / f"public_{public_feature_set}"


def task_dir(
    result_root: Path,
    public_feature_set: str,
    repeat: int,
    fold: int,
) -> Path:
    return public_result_dir(result_root, public_feature_set) / task_name(
        repeat, fold
    )


def task_log_paths(
    log_dir: Path,
    repeat: int,
    fold: int,
) -> Tuple[Path, Path]:
    stem = task_name(repeat, fold)
    return log_dir / f"{stem}.stdout.log", log_dir / f"{stem}.stderr.log"


def ensure_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )


def load_outer_assignments(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Outer assignments not found: {path}")
    df = pd.read_csv(path, dtype=str)
    df = df.drop(
        columns=[c for c in df.columns if str(c).startswith("Unnamed:")],
        errors="ignore",
    )
    required = {
        "outer_repeat",
        "outer_fold",
        "sample_id",
        "cohort",
        "trb_libraryid",
        "igh_libraryid",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Outer assignments missing columns: {missing}")
    for col in ["sample_id", "cohort", "trb_libraryid", "igh_libraryid"]:
        df[col] = df[col].astype(str).str.strip()
        if df[col].eq("").any() or df[col].isna().any():
            raise ValueError(f"Outer assignments contain empty {col} values.")
    df["cohort"] = df["cohort"].str.upper()
    if set(df["cohort"]) != {"RA", "ILD"}:
        raise ValueError(
            f"Outer assignments cohort must be RA/ILD; observed {sorted(set(df['cohort']))}."
        )
    for col in ["outer_repeat", "outer_fold"]:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
    if df.duplicated(["outer_repeat", "sample_id"]).any():
        raise ValueError(
            "Each patient must occur exactly once per outer repeat in assignments."
        )
    if df.duplicated(["outer_repeat", "trb_libraryid"]).any():
        raise ValueError("Duplicated TRB library within an outer repeat.")
    if df.duplicated(["outer_repeat", "igh_libraryid"]).any():
        raise ValueError("Duplicated IGH library within an outer repeat.")

    reference_mapping = None
    reference_ids = None
    for repeat, part in df.groupby("outer_repeat", sort=True):
        if set(part["outer_fold"]) != set(range(1, int(part["outer_fold"].max()) + 1)):
            raise ValueError(f"Outer repeat {repeat} has non-contiguous fold IDs.")
        if part["sample_id"].duplicated().any():
            raise ValueError(f"Outer repeat {repeat} contains duplicated patients.")
        current_ids = set(part["sample_id"])
        current_mapping = {
            row.sample_id: (row.trb_libraryid, row.igh_libraryid, row.cohort)
            for row in part.itertuples(index=False)
        }
        if reference_ids is None:
            reference_ids = current_ids
            reference_mapping = current_mapping
        else:
            if current_ids != reference_ids:
                raise ValueError(
                    f"Patient set differs in outer repeat {repeat}."
                )
            if current_mapping != reference_mapping:
                raise ValueError(
                    f"Patient/TRB/IGH/cohort mapping differs in outer repeat {repeat}."
                )
    return df


def validate_requested_tasks(
    outer: pd.DataFrame,
    repeats: Sequence[int],
    folds: Sequence[int],
) -> None:
    available = set(
        zip(outer["outer_repeat"].astype(int), outer["outer_fold"].astype(int))
    )
    missing = [
        (repeat, fold)
        for repeat in repeats
        for fold in folds
        if (repeat, fold) not in available
    ]
    if missing:
        raise ValueError(
            "Requested outer tasks are absent from frozen assignments: "
            f"{missing[:20]}"
        )


def expected_task_info(
    outer: pd.DataFrame,
    repeat: int,
    fold: int,
) -> Dict[str, object]:
    repeat_df = outer[outer["outer_repeat"] == repeat].copy()
    if repeat_df.empty:
        raise ValueError(f"Outer repeat {repeat} not found.")
    valid = repeat_df[repeat_df["outer_fold"] == fold].copy()
    if valid.empty:
        raise ValueError(f"Outer task repeat={repeat}, fold={fold} not found.")
    train = repeat_df[repeat_df["outer_fold"] != fold].copy()
    valid_ids = set(valid["sample_id"].astype(str))
    train_ids = set(train["sample_id"].astype(str))
    if train_ids & valid_ids:
        raise RuntimeError("Frozen outer training and validation IDs overlap.")
    if train_ids | valid_ids != set(repeat_df["sample_id"].astype(str)):
        raise RuntimeError("Frozen outer task does not cover the complete repeat.")
    return {
        "n_total": len(repeat_df),
        "n_train": len(train),
        "n_valid": len(valid),
        "valid_ids": valid_ids,
        "train_ids": train_ids,
        "valid_cohort_counts": valid["cohort"].value_counts().to_dict(),
    }


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise TaskValidationError(message)


def validate_completed_task(
    result_root: Path,
    repeat: int,
    fold: int,
    public_feature_set: str,
    expected: Mapping[str, object],
) -> Tuple[bool, str]:
    """Return whether an existing joint task is safe to resume-skip."""
    out_dir = task_dir(
        result_root, public_feature_set, repeat, fold
    )
    try:
        missing = [
            name
            for name in EXPECTED_TASK_FILES
            if not (out_dir / name).is_file()
        ]
        _assert(not missing, f"missing files: {missing}")

        config = json.loads(
            (out_dir / "05_trial_configuration.json").read_text(
                encoding="utf-8"
            )
        )
        _assert(
            str(config.get("analysis_view", "")).upper() == ANALYSIS_VIEW,
            "configuration analysis_view is not TRB_IGH",
        )
        _assert(
            str(config.get("cv_unit", "")).lower() == CV_UNIT,
            "configuration cv_unit is not patient",
        )
        _assert(
            int(config.get("outer_repeat", -1)) == repeat,
            "configuration repeat mismatch",
        )
        _assert(
            int(config.get("outer_fold", -1)) == fold,
            "configuration fold mismatch",
        )
        _assert(
            config.get("public_feature_set_per_receptor")
            == public_feature_set,
            "configuration public-feature-set mismatch",
        )
        _assert(
            int(config.get("n_outer_train", -1)) == int(expected["n_train"]),
            "configuration outer-training size mismatch",
        )
        _assert(
            int(config.get("n_outer_validation", -1))
            == int(expected["n_valid"]),
            "configuration outer-validation size mismatch",
        )
        receptor_inputs = config.get("receptor_inputs", {})
        _assert(
            set(receptor_inputs) == {"TRB", "IGH"},
            "configuration does not contain exactly TRB and IGH inputs",
        )

        metrics = pd.read_csv(out_dir / "05_outer_validation_metrics.csv")
        required_metrics = {
            "model",
            "fit_converged",
            "roc_auc",
            "pr_auc",
            "n_outer_train",
            "n_outer_validation",
            "outer_repeat",
            "outer_fold",
        }
        _assert(
            required_metrics.issubset(metrics.columns),
            "metrics missing columns: "
            f"{sorted(required_metrics - set(metrics.columns))}",
        )
        _assert(
            len(metrics) == len(EXPECTED_MODELS),
            f"expected {len(EXPECTED_MODELS)} metric rows; observed {len(metrics)}",
        )
        _assert(
            set(metrics["model"].astype(str)) == EXPECTED_MODELS,
            "unexpected joint model set",
        )
        _assert(
            ensure_bool(metrics["fit_converged"]).all(),
            "one or more final fits did not converge",
        )
        for metric_name in ("roc_auc", "pr_auc"):
            values = pd.to_numeric(metrics[metric_name], errors="coerce")
            _assert(
                values.notna().all()
                and ((values >= 0) & (values <= 1)).all(),
                f"invalid {metric_name} values",
            )
        _assert(
            set(pd.to_numeric(metrics["n_outer_train"], errors="raise"))
            == {int(expected["n_train"])},
            "metrics outer-training size mismatch",
        )
        _assert(
            set(pd.to_numeric(metrics["n_outer_validation"], errors="raise"))
            == {int(expected["n_valid"])},
            "metrics outer-validation size mismatch",
        )
        _assert(
            set(pd.to_numeric(metrics["outer_repeat"], errors="raise"))
            == {repeat},
            "metrics repeat mismatch",
        )
        _assert(
            set(pd.to_numeric(metrics["outer_fold"], errors="raise"))
            == {fold},
            "metrics fold mismatch",
        )

        predictions = pd.read_csv(
            out_dir / "05_outer_validation_predictions.csv", dtype=str
        )
        required_predictions = {
            "model",
            "sample_id",
            "trb_libraryid",
            "igh_libraryid",
            "outer_repeat",
            "outer_fold",
            "true_label",
            "probability_ILD",
        }
        _assert(
            required_predictions.issubset(predictions.columns),
            "predictions missing columns: "
            f"{sorted(required_predictions - set(predictions.columns))}",
        )
        _assert(
            set(predictions["model"]) == EXPECTED_MODELS,
            "prediction model set mismatch",
        )
        expected_rows = int(expected["n_valid"]) * len(EXPECTED_MODELS)
        _assert(
            len(predictions) == expected_rows,
            f"expected {expected_rows} prediction rows; observed {len(predictions)}",
        )
        _assert(
            not predictions.duplicated(["model", "sample_id"]).any(),
            "duplicated model/patient predictions",
        )
        expected_valid_ids = set(expected["valid_ids"])
        for model_name, part in predictions.groupby("model"):
            _assert(
                set(part["sample_id"].astype(str)) == expected_valid_ids,
                f"validation patient set mismatch for {model_name}",
            )
        probabilities = pd.to_numeric(
            predictions["probability_ILD"], errors="coerce"
        )
        _assert(
            probabilities.notna().all()
            and ((probabilities >= 0) & (probabilities <= 1)).all(),
            "invalid probability_ILD values",
        )

        roles = pd.read_csv(out_dir / "05_trial_sample_roles.csv", dtype=str)
        required_roles = {
            "sample_id",
            "trb_libraryid",
            "igh_libraryid",
            "selected_outer_task",
        }
        _assert(
            required_roles.issubset(roles.columns),
            "sample roles missing required joint columns",
        )
        _assert(
            len(roles) == int(expected["n_total"]),
            "sample-role row count mismatch",
        )
        _assert(
            not roles["sample_id"].duplicated().any(),
            "duplicated patients in sample roles",
        )
        role_map = roles.set_index("sample_id")["selected_outer_task"]
        observed_valid = set(
            role_map[role_map.astype(str).str.lower() == "validation"].index
        )
        _assert(
            observed_valid == expected_valid_ids,
            "sample-role validation patient set mismatch",
        )

        public_summary = pd.read_csv(
            out_dir / "05_public_reference_build_summary_TRB_IGH.csv"
        )
        _assert(
            "receptor" in public_summary.columns,
            "public-reference summary missing receptor column",
        )
        _assert(
            set(public_summary["receptor"].astype(str).str.upper())
            == {"TRB", "IGH"},
            "public-reference summary does not contain both receptors",
        )

        static_features = pd.read_csv(
            out_dir / "05_static_TRB_IGH_feature_list.csv"
        )
        _assert(
            {"feature_name", "receptor"}.issubset(static_features.columns),
            "static feature list missing required columns",
        )
        _assert(
            set(static_features["receptor"].astype(str).str.upper())
            == {"TRB", "IGH"},
            "static feature list does not contain both receptors",
        )

        return True, "all paired TRB+IGH expected files and checks passed"
    except Exception as exc:
        return False, str(exc)


def write_status_csv(
    path: Path,
    rows: Sequence[Dict[str, object]],
) -> None:
    columns = [
        "task_order",
        "outer_repeat",
        "outer_fold",
        "task_name",
        "expected_outer_train",
        "expected_outer_validation",
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
    frame = pd.DataFrame(rows)
    for col in columns:
        if col not in frame.columns:
            frame[col] = ""
    frame[columns].to_csv(path, index=False)


def write_summary(
    path: Path,
    rows: Sequence[Dict[str, object]],
    args: argparse.Namespace,
    outer_path: Path,
    outer_hash: str,
    started: str,
    ended: str,
    total_runtime: float,
) -> None:
    frame = pd.DataFrame(rows)
    counts = (
        frame["status"].value_counts().to_dict()
        if not frame.empty and "status" in frame.columns
        else {}
    )
    completed = int(counts.get("completed", 0))
    skipped = int(counts.get("skipped_completed", 0))
    failed = int(counts.get("failed", 0))
    dry = int(counts.get("dry_run", 0))
    pending = len(args.repeats) * len(args.folds) - len(frame)

    lines = [
        "# 05 Paired TRB+IGH LOO Batch Run Summary",
        "",
        f"- Script version: `{SCRIPT_VERSION}`",
        f"- Analysis view: `{ANALYSIS_VIEW}`",
        f"- CV unit: `{CV_UNIT}`",
        f"- Host: `{socket.gethostname()}`",
        f"- Started: `{started}`",
        f"- Ended/status update: `{ended}`",
        f"- Total elapsed runtime: **{total_runtime:.2f} seconds**",
        f"- Worker: `{args.worker_script}`",
        f"- Frozen outer assignments: `{outer_path}`",
        f"- Frozen outer SHA256: `{outer_hash}`",
        f"- Public feature set per receptor: `{args.public_feature_set}`",
        f"- Requested repeats: `{args.repeats}`",
        f"- Requested folds: `{args.folds}`",
        "",
        "## Status counts",
        "",
        f"- Newly completed: **{completed}**",
        f"- Skipped because already complete: **{skipped}**",
        f"- Failed: **{failed}**",
        f"- Dry-run tasks: **{dry}**",
        f"- Not yet processed in this run: **{pending}**",
        "",
        "## Failed tasks",
        "",
    ]
    if failed == 0:
        lines.append("- None")
    else:
        for _, row in frame[frame["status"] == "failed"].iterrows():
            lines.append(
                f"- `{row['task_name']}`: return code "
                f"`{row['return_code']}`; {row['message']}"
            )
    lines.extend(
        [
            "",
            "## Resume behavior",
            "",
            (
                "- Re-running the same command skips only tasks whose joint "
                "TRB+IGH outputs pass all integrity checks."
            ),
            (
                "- The expected 23/24-patient validation size is read from "
                "the frozen patient-level assignments for each task."
            ),
            "- Failed or incomplete tasks are run again automatically.",
            (
                "- Use `--rerun-completed` only when intentionally replacing "
                "validated task results."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def acquire_lock(lock_path: Path, ignore_lock: bool) -> None:
    if lock_path.exists() and not ignore_lock:
        content = lock_path.read_text(
            encoding="utf-8", errors="replace"
        )
        raise RuntimeError(
            f"Batch lock already exists: {lock_path}\n{content}\n"
            "Confirm that no batch process is active before removing it."
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


def normalize_extra_args(values: Sequence[str]) -> List[str]:
    values = list(values)
    if values and values[0] == "--":
        return values[1:]
    return values


def run_worker_with_heartbeat(
    command: Sequence[str],
    stdout_log: Path,
    stderr_log: Path,
    heartbeat_seconds: int,
    poll_seconds: float,
    task_label: str,
) -> int:
    started = time.time()
    next_heartbeat = (
        started + heartbeat_seconds if heartbeat_seconds else float("inf")
    )
    with stdout_log.open("w", encoding="utf-8") as stdout_handle, \
            stderr_log.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            list(command),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        while True:
            return_code = process.poll()
            if return_code is not None:
                return int(return_code)
            now = time.time()
            if now >= next_heartbeat:
                print(
                    f"[heartbeat] {task_label} still running | "
                    f"worker_pid={process.pid} | elapsed={now - started:.1f}s | "
                    f"stdout={stdout_log}",
                    flush=True,
                )
                next_heartbeat = now + heartbeat_seconds
            time.sleep(poll_seconds)


def main() -> int:
    args = parse_args()
    started_iso = now_iso()
    started_time = time.time()

    worker = Path(args.worker_script).expanduser().resolve()
    outer_path = Path(args.outer_assignments).expanduser().resolve()
    result_root = Path(args.result_root).expanduser().resolve()
    batch_dir = Path(args.batch_dir).expanduser().resolve()
    log_dir = batch_dir / "task_logs" / f"public_{args.public_feature_set}"
    status_path = batch_dir / "05_batch_run_status.csv"
    summary_path = batch_dir / "05_batch_run_summary.md"
    lock_path = batch_dir / "05_batch_run.lock"
    active_path = batch_dir / "05_active_task.json"

    if not worker.is_file():
        raise FileNotFoundError(f"Worker script not found: {worker}")
    outer = load_outer_assignments(outer_path)
    validate_requested_tasks(outer, args.repeats, args.folds)
    outer_hash = file_sha256(outer_path)

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
    extra_worker_args = normalize_extra_args(args.extra_worker_args)

    print("=" * 88, flush=True)
    print("05 Paired TRB+IGH LOO Batch Outer-Trial Runner", flush=True)
    print("=" * 88, flush=True)
    print(f"Script version: {SCRIPT_VERSION}", flush=True)
    print(f"Analysis view: {ANALYSIS_VIEW}", flush=True)
    print(f"CV unit: {CV_UNIT}", flush=True)
    print(f"Worker: {worker}", flush=True)
    print(f"Frozen outer assignments: {outer_path}", flush=True)
    print(f"Frozen outer SHA256: {outer_hash}", flush=True)
    print(f"Tasks requested: {len(tasks)}", flush=True)
    print(f"Result root: {public_result_dir(result_root, args.public_feature_set)}", flush=True)
    print(f"Batch logs: {batch_dir}", flush=True)
    print(f"Public feature set per receptor: {args.public_feature_set}", flush=True)
    print("Execution mode: sequential", flush=True)
    print(f"Heartbeat: every {args.heartbeat_seconds}s", flush=True)
    print(f"Started: {started_iso}", flush=True)
    print("", flush=True)

    try:
        for order, (repeat, fold) in enumerate(tasks, start=1):
            name = task_name(repeat, fold)
            expected = expected_task_info(outer, repeat, fold)
            out_dir = task_dir(
                result_root, args.public_feature_set, repeat, fold
            )
            stdout_log, stderr_log = task_log_paths(
                log_dir, repeat, fold
            )
            complete, completion_reason = validate_completed_task(
                result_root=result_root,
                repeat=repeat,
                fold=fold,
                public_feature_set=args.public_feature_set,
                expected=expected,
            )

            row: Dict[str, object] = {
                "task_order": order,
                "outer_repeat": repeat,
                "outer_fold": fold,
                "task_name": name,
                "expected_outer_train": expected["n_train"],
                "expected_outer_validation": expected["n_valid"],
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
                row.update(
                    {
                        "status": "skipped_completed",
                        "message": (
                            "existing paired task outputs passed all checks"
                        ),
                    }
                )
                rows.append(row)
                write_status_csv(status_path, rows)
                write_summary(
                    summary_path,
                    rows,
                    args,
                    outer_path,
                    outer_hash,
                    started_iso,
                    now_iso(),
                    time.time() - started_time,
                )
                print(
                    f"[{order:03d}/{len(tasks):03d}] {name}: SKIP "
                    f"(validated complete; train={expected['n_train']}, "
                    f"validation={expected['n_valid']})",
                    flush=True,
                )
                continue

            command = [
                args.python,
                "-u",
                str(worker),
                "--outer-repeat",
                str(repeat),
                "--outer-fold",
                str(fold),
                "--public-feature-set",
                args.public_feature_set,
                "--overwrite",
                *extra_worker_args,
            ]

            if args.dry_run:
                row.update(
                    {
                        "status": "dry_run",
                        "message": " ".join(command),
                    }
                )
                rows.append(row)
                write_status_csv(status_path, rows)
                write_summary(
                    summary_path,
                    rows,
                    args,
                    outer_path,
                    outer_hash,
                    started_iso,
                    now_iso(),
                    time.time() - started_time,
                )
                print(
                    f"[{order:03d}/{len(tasks):03d}] {name}: DRY RUN "
                    f"(train={expected['n_train']}, validation={expected['n_valid']})\n"
                    f"  {' '.join(command)}",
                    flush=True,
                )
                continue

            task_started_iso = now_iso()
            task_started = time.time()
            row["start_time"] = task_started_iso
            active_payload = {
                "batch_pid": os.getpid(),
                "task_order": order,
                "task_count": len(tasks),
                "task_name": name,
                "outer_repeat": repeat,
                "outer_fold": fold,
                "expected_outer_train": expected["n_train"],
                "expected_outer_validation": expected["n_valid"],
                "started": task_started_iso,
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
                "result_dir": str(out_dir),
                "command": command,
            }
            active_path.write_text(
                json.dumps(active_payload, ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )

            print(
                f"[{order:03d}/{len(tasks):03d}] {name}: START at "
                f"{task_started_iso} | train={expected['n_train']} | "
                f"validation={expected['n_valid']}",
                flush=True,
            )
            print(f"  stdout: {stdout_log}", flush=True)
            print(f"  stderr: {stderr_log}", flush=True)

            return_code = run_worker_with_heartbeat(
                command=command,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                heartbeat_seconds=args.heartbeat_seconds,
                poll_seconds=args.poll_seconds,
                task_label=name,
            )

            task_runtime = time.time() - task_started
            task_ended_iso = now_iso()
            complete_after, reason_after = validate_completed_task(
                result_root=result_root,
                repeat=repeat,
                fold=fold,
                public_feature_set=args.public_feature_set,
                expected=expected,
            )
            row.update(
                {
                    "end_time": task_ended_iso,
                    "runtime_seconds": round(task_runtime, 3),
                    "return_code": return_code,
                    "completion_check": reason_after,
                }
            )

            if return_code == 0 and complete_after:
                row.update(
                    {
                        "status": "completed",
                        "message": (
                            "worker returned 0 and paired outputs passed checks"
                        ),
                    }
                )
                print(
                    f"[{order:03d}/{len(tasks):03d}] {name}: COMPLETE "
                    f"({task_runtime:.1f}s)",
                    flush=True,
                )
            else:
                row.update(
                    {
                        "status": "failed",
                        "message": (
                            f"worker return code={return_code}; "
                            f"completion check={reason_after}"
                        ),
                    }
                )
                print(
                    f"[{order:03d}/{len(tasks):03d}] {name}: FAILED "
                    f"return_code={return_code}; {reason_after}",
                    flush=True,
                )

            rows.append(row)
            write_status_csv(status_path, rows)
            write_summary(
                summary_path,
                rows,
                args,
                outer_path,
                outer_hash,
                started_iso,
                now_iso(),
                time.time() - started_time,
            )
            if active_path.exists():
                active_path.unlink()

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
            outer_path,
            outer_hash,
            started_iso,
            ended_iso,
            time.time() - started_time,
        )
        if active_path.exists():
            active_path.unlink()
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
