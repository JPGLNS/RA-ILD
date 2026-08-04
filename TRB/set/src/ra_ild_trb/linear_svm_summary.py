#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repeated-holdout aggregation for TRB Linear SVM task outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


class LinearSVMSummaryError(ValueError):
    """Raised when completed Linear SVM task outputs cannot be aggregated."""


TASK_FILES: Mapping[str, str] = {
    "metrics": "06_outer_validation_metrics.csv",
    "predictions": "06_outer_validation_predictions.csv",
    "coefficients": "06_final_model_coefficients.csv",
    "configuration": "06_task_configuration.json",
    "complete": "06_TASK_COMPLETE.json",
}


@dataclass(frozen=True)
class LinearSVMTaskFiles:
    task_dir: Path
    outer_repeat: int
    outer_fold: int
    paths: Mapping[str, Path]


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        raise LinearSVMSummaryError(f"Invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise LinearSVMSummaryError(f"JSON root must be a mapping: {path}")
    return value


def discover_completed_tasks(task_root: Path) -> Tuple[LinearSVMTaskFiles, ...]:
    root = Path(task_root).expanduser().resolve()
    if not root.is_dir():
        raise LinearSVMSummaryError(f"Task root not found: {root}")
    tasks: List[LinearSVMTaskFiles] = []
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        paths = {key: task_dir / name for key, name in TASK_FILES.items()}
        if not paths["complete"].is_file():
            continue
        missing = [str(path) for key, path in paths.items() if key != "complete" and not path.is_file()]
        if missing:
            raise LinearSVMSummaryError(
                f"Completed task is missing required files: {task_dir}: {missing}"
            )
        marker = _read_json(paths["complete"])
        if str(marker.get("status", "")).upper() != "COMPLETE":
            raise LinearSVMSummaryError(f"Task marker is not COMPLETE: {paths['complete']}")
        outer_repeat = int(marker.get("outer_repeat", 0))
        outer_fold = int(marker.get("outer_fold", 0))
        if outer_repeat < 1 or outer_fold < 1:
            raise LinearSVMSummaryError(f"Invalid task coordinates: {paths['complete']}")
        tasks.append(
            LinearSVMTaskFiles(
                task_dir=task_dir,
                outer_repeat=outer_repeat,
                outer_fold=outer_fold,
                paths=paths,
            )
        )
    if not tasks:
        raise LinearSVMSummaryError(f"No completed Linear SVM tasks found under {root}")
    coordinates = [(task.outer_repeat, task.outer_fold) for task in tasks]
    if len(coordinates) != len(set(coordinates)):
        raise LinearSVMSummaryError("Duplicate outer repeat/fold task coordinates")
    return tuple(tasks)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise LinearSVMSummaryError(f"{label} missing columns: {missing}")


def _read_task_frame(task: LinearSVMTaskFiles, key: str) -> pd.DataFrame:
    frame = pd.read_csv(task.paths[key])
    if frame.empty:
        raise LinearSVMSummaryError(f"Empty {key} file: {task.paths[key]}")
    frame["source_task_dir"] = str(task.task_dir)
    return frame


def aggregate_completed_tasks(
    tasks: Sequence[LinearSVMTaskFiles],
) -> Dict[str, pd.DataFrame]:
    metrics_parts: List[pd.DataFrame] = []
    prediction_parts: List[pd.DataFrame] = []
    coefficient_parts: List[pd.DataFrame] = []
    for task in tasks:
        metrics = _read_task_frame(task, "metrics")
        predictions = _read_task_frame(task, "predictions")
        coefficients = _read_task_frame(task, "coefficients")
        _require_columns(
            metrics,
            (
                "model",
                "outer_repeat",
                "outer_fold",
                "roc_auc",
                "average_precision",
                "accuracy",
                "sensitivity_recall",
                "specificity",
                "precision",
                "f1",
                "selected_threshold",
                "selected_C",
                "selected_penalty",
                "selected_loss",
                "selected_class_weight",
                "probability_metrics_available",
            ),
            "task metrics",
        )
        _require_columns(
            predictions,
            (
                "model",
                "outer_repeat",
                "outer_fold",
                "sample_id",
                "true_label",
                "prediction_score",
                "threshold",
            ),
            "task predictions",
        )
        _require_columns(
            coefficients,
            ("model", "feature_name", "coefficient", "absolute_coefficient"),
            "task coefficients",
        )
        if metrics["probability_metrics_available"].astype(str).str.lower().isin(
            {"true", "1"}
        ).any():
            raise LinearSVMSummaryError(
                f"Uncalibrated Linear SVM task unexpectedly advertises probabilities: {task.task_dir}"
            )
        metrics_parts.append(metrics)
        prediction_parts.append(predictions)
        coefficient_parts.append(coefficients)

    metrics = pd.concat(metrics_parts, ignore_index=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    coefficients = pd.concat(coefficient_parts, ignore_index=True)

    key_columns = ["model", "outer_repeat", "outer_fold"]
    if metrics.duplicated(key_columns).any():
        raise LinearSVMSummaryError("Duplicate model/repeat/fold rows in aggregated metrics")
    expected_pairs = set(
        map(tuple, metrics[["outer_repeat", "outer_fold"]].astype(int).to_numpy())
    )
    for model, subset in metrics.groupby("model", sort=False):
        observed = set(
            map(tuple, subset[["outer_repeat", "outer_fold"]].astype(int).to_numpy())
        )
        if observed != expected_pairs:
            raise LinearSVMSummaryError(
                f"Model {model} does not cover all completed repeat/fold tasks"
            )

    metric_names = (
        "roc_auc",
        "average_precision",
        "accuracy",
        "sensitivity_recall",
        "specificity",
        "precision",
        "f1",
        "zero_threshold_accuracy",
        "zero_threshold_sensitivity_recall",
        "zero_threshold_specificity",
        "zero_threshold_precision",
        "zero_threshold_f1",
    )
    summary_rows: List[Dict[str, object]] = []
    for model, subset in metrics.groupby("model", sort=False):
        for metric in metric_names:
            if metric not in subset.columns:
                continue
            values = pd.to_numeric(subset[metric], errors="coerce").dropna().to_numpy(float)
            if not len(values):
                continue
            summary_rows.append(
                {
                    "model": str(model),
                    "metric": metric,
                    "n_tasks": int(len(values)),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    metric_summary = pd.DataFrame(summary_rows)

    parameter_columns = [
        "selected_candidate_id",
        "selected_candidate_block",
        "selected_C",
        "selected_penalty",
        "selected_loss",
        "selected_class_weight",
        "selected_dual_requested",
        "selected_dual_resolved",
    ]
    available_parameters = [column for column in parameter_columns if column in metrics.columns]
    parameter_frequency = (
        metrics.groupby(["model", *available_parameters], dropna=False, sort=False)
        .size()
        .rename("selected_task_count")
        .reset_index()
    )
    total_by_model = metrics.groupby("model", sort=False).size().rename("model_task_count")
    parameter_frequency = parameter_frequency.merge(total_by_model, on="model", how="left")
    parameter_frequency["selection_frequency"] = (
        parameter_frequency["selected_task_count"]
        / parameter_frequency["model_task_count"]
    )

    coefficient_numeric = coefficients.copy()
    coefficient_numeric["coefficient"] = pd.to_numeric(
        coefficient_numeric["coefficient"], errors="raise"
    )
    coefficient_numeric["absolute_coefficient"] = pd.to_numeric(
        coefficient_numeric["absolute_coefficient"], errors="raise"
    )
    coefficient_numeric["positive"] = coefficient_numeric["coefficient"] > 0
    coefficient_numeric["negative"] = coefficient_numeric["coefficient"] < 0
    if "nonzero" in coefficient_numeric.columns:
        nonzero = coefficient_numeric["nonzero"].astype(str).str.lower().isin(
            {"true", "1"}
        )
    else:
        nonzero = coefficient_numeric["absolute_coefficient"] > 1.0e-12
    coefficient_numeric["nonzero_bool"] = nonzero
    coefficient_summary = (
        coefficient_numeric.groupby(["model", "feature_name"], sort=False)
        .agg(
            n_fits=("coefficient", "size"),
            mean_coefficient=("coefficient", "mean"),
            sd_coefficient=("coefficient", "std"),
            median_coefficient=("coefficient", "median"),
            mean_absolute_coefficient=("absolute_coefficient", "mean"),
            positive_frequency=("positive", "mean"),
            negative_frequency=("negative", "mean"),
            nonzero_frequency=("nonzero_bool", "mean"),
        )
        .reset_index()
    )
    coefficient_summary["sd_coefficient"] = coefficient_summary[
        "sd_coefficient"
    ].fillna(0.0)
    coefficient_summary["sign_consistency"] = coefficient_summary[
        ["positive_frequency", "negative_frequency"]
    ].max(axis=1)

    return {
        "metrics": metrics,
        "predictions": predictions,
        "coefficients": coefficients,
        "metric_summary": metric_summary,
        "parameter_frequency": parameter_frequency,
        "coefficient_summary": coefficient_summary,
    }


def write_aggregation(
    task_root: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> Mapping[str, Path]:
    tasks = discover_completed_tasks(task_root)
    aggregated = aggregate_completed_tasks(tasks)
    output = Path(output_dir).expanduser().resolve()
    paths: Dict[str, Path] = {
        "metrics": output / "08_holdout_task_metrics.csv",
        "predictions": output / "08_holdout_predictions.csv.gz",
        "coefficients": output / "08_final_model_coefficients.csv.gz",
        "metric_summary": output / "08_metric_summary.csv",
        "parameter_frequency": output / "08_selected_parameter_frequency.csv",
        "coefficient_summary": output / "08_coefficient_stability_summary.csv",
        "summary": output / "08_aggregation_summary.json",
        "complete": output / "08_REPEATED_HOLDOUT_AGGREGATION_COMPLETE.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Aggregation outputs already exist; use --overwrite for a documented rerun:\n"
            + "\n".join(f"  - {path}" for path in existing)
        )
    output.mkdir(parents=True, exist_ok=True)
    aggregated["metrics"].to_csv(paths["metrics"], index=False)
    aggregated["predictions"].to_csv(paths["predictions"], index=False, compression="gzip")
    aggregated["coefficients"].to_csv(paths["coefficients"], index=False, compression="gzip")
    aggregated["metric_summary"].to_csv(paths["metric_summary"], index=False)
    aggregated["parameter_frequency"].to_csv(paths["parameter_frequency"], index=False)
    aggregated["coefficient_summary"].to_csv(paths["coefficient_summary"], index=False)

    models = aggregated["metrics"]["model"].astype(str).drop_duplicates().tolist()
    summary = {
        "aggregation_type": "linear_svm_repeated_holdout",
        "model_engine": "linear_svc",
        "score_type": "decision_function",
        "probability_metrics_available": False,
        "task_root": str(Path(task_root).expanduser().resolve()),
        "completed_task_count": int(len(tasks)),
        "models": models,
        "model_count": int(len(models)),
        "metric_rows": int(len(aggregated["metrics"])),
        "prediction_rows": int(len(aggregated["predictions"])),
        "coefficient_rows": int(len(aggregated["coefficients"])),
        "log_loss_and_brier_excluded": True,
    }
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    completion = {
        "status": "COMPLETE",
        **summary,
        "output_dir": str(output),
    }
    paths["complete"].write_text(
        json.dumps(completion, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths
