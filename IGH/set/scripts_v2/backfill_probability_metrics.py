#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backfill IGH log-loss and Brier score from frozen holdout predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.config import load_experiment_config  # noqa: E402
from ra_ild_igh.metrics import classification_metrics  # noqa: E402
from ra_ild_igh.outer_cv import build_outer_tasks  # noqa: E402

PREDICTION_FILE = "06_outer_validation_predictions.csv"
METRIC_FILE = "06_outer_validation_metrics.csv"
COMPLETE_FILE = "06_TASK_COMPLETE.json"
BACKUP_FILE = "06_outer_validation_metrics.before_batch12.csv"
AUDIT_FILE = "09_probability_metric_backfill_audit.csv"
EXISTING_METRICS = (
    "roc_auc", "pr_auc", "accuracy", "sensitivity_recall",
    "specificity", "precision", "f1",
)
NEW_METRICS = ("log_loss", "brier_score")


class BackfillError(ValueError):
    pass


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BackfillError(f"Unable to read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise BackfillError(f"JSON must contain an object: {path}")
    return value


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise BackfillError(f"{label} is missing columns: {missing}")


def _reordered_columns(frame: pd.DataFrame) -> List[str]:
    columns = [c for c in frame.columns if c not in NEW_METRICS]
    insertion = columns.index("pr_auc") + 1 if "pr_auc" in columns else len(columns)
    return columns[:insertion] + list(NEW_METRICS) + columns[insertion:]


def _atomic_csv_write(frame: pd.DataFrame, path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    temporary.replace(path)


def _recompute_task(task, *, tolerance: float, dry_run: bool) -> Dict[str, object]:
    task_dir = Path(task.output_dir)
    complete_path = task_dir / COMPLETE_FILE
    prediction_path = task_dir / PREDICTION_FILE
    metric_path = task_dir / METRIC_FILE
    backup_path = task_dir / BACKUP_FILE

    for path in (complete_path, prediction_path, metric_path):
        if not path.is_file():
            raise BackfillError(f"Task {task.task_id} is missing {path.name}")
    marker = _read_json(complete_path)
    if str(marker.get("status")) != "COMPLETE":
        raise BackfillError(f"Task {task.task_id} completion marker is not COMPLETE")

    prediction_sha = sha256_file(prediction_path)
    complete_sha = sha256_file(complete_path)
    metric_sha_before = sha256_file(metric_path)
    predictions = pd.read_csv(prediction_path)
    metrics = pd.read_csv(metric_path)
    _require_columns(
        predictions,
        ("model", "sample_id", "true_label", "probability_ILD", "threshold"),
        f"{task.task_id} predictions",
    )
    _require_columns(
        metrics, ("model", "threshold", *EXISTING_METRICS),
        f"{task.task_id} metrics",
    )
    if metrics["model"].astype(str).duplicated().any():
        raise BackfillError(f"Task {task.task_id} has duplicate metric model rows")
    if predictions.duplicated(["model", "sample_id"]).any():
        raise BackfillError(f"Task {task.task_id} has duplicate prediction rows")

    updated = metrics.copy()
    max_existing_difference = 0.0
    model_order = metrics["model"].astype(str).tolist()
    if set(predictions["model"].astype(str)) != set(model_order):
        raise BackfillError(f"Task {task.task_id} model sets differ between files")

    for row_index, model in enumerate(model_order):
        group = predictions.loc[predictions["model"].astype(str) == model].copy()
        y = pd.to_numeric(group["true_label"], errors="raise").to_numpy(int)
        probability = pd.to_numeric(
            group["probability_ILD"], errors="raise"
        ).to_numpy(float)
        thresholds = pd.to_numeric(group["threshold"], errors="raise").unique()
        if len(thresholds) != 1:
            raise BackfillError(f"Task {task.task_id}/{model} has multiple thresholds")
        saved_threshold = float(metrics.iloc[row_index]["threshold"])
        if not np.isclose(
            float(thresholds[0]), saved_threshold, rtol=0.0, atol=tolerance
        ):
            raise BackfillError(f"Task {task.task_id}/{model} threshold mismatch")
        recomputed = classification_metrics(
            y, probability, saved_threshold,
            include_brier=True, include_log_loss=True,
        )
        for metric_name in EXISTING_METRICS:
            saved = float(metrics.iloc[row_index][metric_name])
            observed = float(recomputed[metric_name])
            difference = abs(saved - observed)
            max_existing_difference = max(max_existing_difference, difference)
            if difference > tolerance:
                raise BackfillError(
                    f"Task {task.task_id}/{model}/{metric_name} mismatch: "
                    f"saved={saved}, recomputed={observed}, difference={difference}"
                )
        updated.loc[updated.index[row_index], "log_loss"] = float(
            recomputed["log_loss"]
        )
        updated.loc[updated.index[row_index], "brier_score"] = float(
            recomputed["brier_score"]
        )

    updated = updated.loc[:, _reordered_columns(updated)]
    already_current = all(metric in metrics.columns for metric in NEW_METRICS)
    backup_sha = ""
    metric_sha_after = metric_sha_before
    if already_current:
        for metric_name in NEW_METRICS:
            old = pd.to_numeric(metrics[metric_name], errors="raise").to_numpy(float)
            new = pd.to_numeric(updated[metric_name], errors="raise").to_numpy(float)
            if not np.allclose(old, new, rtol=0.0, atol=tolerance):
                raise BackfillError(
                    f"Task {task.task_id} existing {metric_name} values do not reproduce"
                )
        action = "already_current"
        if backup_path.is_file():
            backup_sha = sha256_file(backup_path)
    else:
        action = "would_update" if dry_run else "updated"
        if not dry_run:
            if backup_path.exists():
                raise BackfillError(
                    f"Backup already exists for an un-upgraded task: {backup_path}"
                )
            shutil.copy2(metric_path, backup_path)
            backup_sha = sha256_file(backup_path)
            if backup_sha != metric_sha_before:
                raise BackfillError(f"Task {task.task_id} backup SHA256 mismatch")
            _atomic_csv_write(updated, metric_path)
            metric_sha_after = sha256_file(metric_path)

    return {
        "task_index": int(task.task_index),
        "task_id": str(task.task_id),
        "outer_repeat": int(task.outer_repeat),
        "outer_fold": int(task.outer_fold),
        "action": action,
        "model_count": int(len(model_order)),
        "prediction_rows": int(len(predictions)),
        "max_existing_metric_abs_difference": float(max_existing_difference),
        "prediction_sha256": prediction_sha,
        "complete_marker_sha256": complete_sha,
        "metric_sha256_before": metric_sha_before,
        "backup_sha256": backup_sha,
        "metric_sha256_after": metric_sha_after,
        "metric_path": str(metric_path),
        "backup_path": str(backup_path) if backup_path.exists() else "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if not np.isfinite(args.tolerance) or args.tolerance <= 0:
            raise BackfillError("--tolerance must be finite and > 0")
        config = load_experiment_config(
            Path(args.config),
            repository_root=Path(args.repository_root)
            if args.repository_root else None,
        )
        repeated = config.section("repeated_holdout_training")
        if str(repeated.get("mode")) != "frozen_repeated_holdout":
            raise BackfillError(
                "This tool requires repeated_holdout_training.mode=frozen_repeated_holdout"
            )
        cv = config.section("cross_validation")
        configured = build_outer_tasks(
            int(cv["outer_repeats"]),
            int(cv["outer_folds"]),
            config.path("outer_tasks.output_root"),
        )
        executed_folds = tuple(
            int(value) for value in cv.get("executed_outer_folds", (1,))
        )
        tasks = tuple(
            task for task in configured if task.outer_fold in executed_folds
        )
        expected = int(repeated["split_count"])
        if len(tasks) != expected:
            raise BackfillError(
                f"Configured executed task count={len(tasks)}, "
                f"expected split_count={expected}"
            )
        rows = [
            _recompute_task(
                task, tolerance=float(args.tolerance), dry_run=bool(args.dry_run)
            )
            for task in tasks
        ]
        audit = pd.DataFrame(rows).sort_values("task_index", kind="stable")
        audit_path = config.path("outer_tasks.output_root") / AUDIT_FILE
        if not args.dry_run:
            audit.to_csv(audit_path, index=False)

        counts = audit["action"].value_counts().to_dict()
        print("IGH probability-metric backfill: COMPLETE")
        print(f"Experiment: {config.experiment_id}")
        print(f"Tasks:      {len(tasks)}/{expected}")
        print(f"Actions:    {counts}")
        print(
            "Max old-metric difference: "
            f"{audit['max_existing_metric_abs_difference'].max():.3e}"
        )
        if args.dry_run:
            print("Dry run only; no task metrics or backups were written.")
            print("IGH_BATCH12_BACKFILL_DRY_RUN_PASS")
        else:
            print(f"Audit:      {audit_path}")
            print("Next: rerun aggregate_repeated_holdout_results.py --overwrite")
            print("IGH_BATCH12_BACKFILL_PASS")
        return 0
    except Exception as exc:
        print(f"IGH probability-metric backfill: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
