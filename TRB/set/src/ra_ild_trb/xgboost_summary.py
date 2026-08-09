#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repeated-holdout aggregation for TRB Batch 18 XGBoost outputs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


class XGBoostSummaryError(ValueError):
    pass


TASK_FILES: Mapping[str, str] = {
    "metrics": "06_outer_validation_metrics.csv",
    "predictions": "06_outer_validation_predictions.csv",
    "importance": "06_final_model_feature_importance.csv",
    "configuration": "06_task_configuration.json",
    "complete": "06_TASK_COMPLETE.json",
}


@dataclass(frozen=True)
class XGBoostTaskFiles:
    task_dir: Path
    outer_repeat: int
    outer_fold: int
    paths: Mapping[str, Path]


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise XGBoostSummaryError(f"Invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise XGBoostSummaryError(f"JSON root must be a mapping: {path}")
    return value


def discover_completed_tasks(task_root: Path) -> Tuple[XGBoostTaskFiles, ...]:
    root = Path(task_root).expanduser().resolve()
    if not root.is_dir():
        raise XGBoostSummaryError(f"Task root not found: {root}")
    tasks: List[XGBoostTaskFiles] = []
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        paths = {key: task_dir / name for key, name in TASK_FILES.items()}
        if not paths["complete"].is_file():
            continue
        missing = [str(path) for key, path in paths.items() if key != "complete" and not path.is_file()]
        if missing:
            raise XGBoostSummaryError(f"Completed task missing required files: {task_dir}: {missing}")
        marker = _read_json(paths["complete"])
        if str(marker.get("status", "")).upper() != "COMPLETE" or str(marker.get("engine", "")) != "xgboost":
            raise XGBoostSummaryError(f"Invalid XGBoost completion marker: {paths['complete']}")
        repeat = int(marker.get("outer_repeat", 0))
        fold = int(marker.get("outer_fold", 0))
        if repeat < 1 or fold < 1:
            raise XGBoostSummaryError(f"Invalid task coordinates: {paths['complete']}")
        tasks.append(XGBoostTaskFiles(task_dir, repeat, fold, paths))
    if not tasks:
        raise XGBoostSummaryError(f"No completed XGBoost tasks found under {root}")
    coords = [(t.outer_repeat, t.outer_fold) for t in tasks]
    if len(coords) != len(set(coords)):
        raise XGBoostSummaryError("Duplicate repeat/fold task coordinates")
    return tuple(tasks)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise XGBoostSummaryError(f"{label} missing columns: {missing}")


def _read_task_frame(task: XGBoostTaskFiles, key: str) -> pd.DataFrame:
    frame = pd.read_csv(task.paths[key])
    if frame.empty:
        raise XGBoostSummaryError(f"Empty {key} file: {task.paths[key]}")
    frame["source_task_dir"] = str(task.task_dir)
    return frame


def aggregate_completed_tasks(tasks: Sequence[XGBoostTaskFiles]) -> Dict[str, pd.DataFrame]:
    metrics_parts: List[pd.DataFrame] = []
    prediction_parts: List[pd.DataFrame] = []
    importance_parts: List[pd.DataFrame] = []
    for task in tasks:
        metrics = _read_task_frame(task, "metrics")
        predictions = _read_task_frame(task, "predictions")
        importance = _read_task_frame(task, "importance")
        _require_columns(metrics, (
            "model", "split_id", "outer_repeat", "outer_fold", "roc_auc", "pr_auc",
            "log_loss", "brier_score", "accuracy", "sensitivity_recall", "specificity",
            "precision", "f1", "threshold", "selected_candidate_id", "selected_parameters_json",
            "probability_metrics_available",
        ), "task metrics")
        _require_columns(predictions, (
            "model", "split_id", "outer_repeat", "outer_fold", "sample_id", "true_label",
            "true_cohort", "probability_ILD", "threshold", "predicted_label",
        ), "task predictions")
        _require_columns(importance, (
            "model", "split_id", "outer_repeat", "outer_fold", "feature_name",
            "importance_gain", "importance_weight", "importance_cover",
        ), "task feature importance")
        if not metrics["probability_metrics_available"].astype(str).str.lower().isin({"true", "1"}).all():
            raise XGBoostSummaryError(f"XGBoost task does not advertise probabilities: {task.task_dir}")
        metrics_parts.append(metrics)
        prediction_parts.append(predictions)
        importance_parts.append(importance)
    metrics = pd.concat(metrics_parts, ignore_index=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    importance = pd.concat(importance_parts, ignore_index=True)
    key = ["model", "outer_repeat", "outer_fold"]
    if metrics.duplicated(key).any():
        raise XGBoostSummaryError("Duplicate model/repeat/fold rows in aggregated metrics")
    expected_pairs = set(map(tuple, metrics[["outer_repeat", "outer_fold"]].astype(int).to_numpy()))
    for model, subset in metrics.groupby("model", sort=False):
        observed = set(map(tuple, subset[["outer_repeat", "outer_fold"]].astype(int).to_numpy()))
        if observed != expected_pairs:
            raise XGBoostSummaryError(f"Model {model} does not cover all completed tasks")

    metric_names = (
        "roc_auc", "pr_auc", "log_loss", "brier_score", "accuracy",
        "sensitivity_recall", "specificity", "precision", "f1",
    )
    summary_rows: List[Dict[str, object]] = []
    for model, subset in metrics.groupby("model", sort=False):
        for metric in metric_names:
            values = pd.to_numeric(subset[metric], errors="coerce").dropna().to_numpy(float)
            if not len(values):
                continue
            summary_rows.append({
                "model": str(model), "metric": metric, "n_tasks": int(len(values)),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "median": float(np.median(values)), "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)), "min": float(np.min(values)),
                "max": float(np.max(values)),
            })
    metric_summary = pd.DataFrame(summary_rows)

    selected_columns = [
        c for c in metrics.columns if c.startswith("selected_xgb_")
    ]
    parameter_columns = ["selected_candidate_id", "selected_parameters_json", *selected_columns]
    parameter_frequency = (
        metrics.groupby(["model", *parameter_columns], dropna=False, sort=False)
        .size().rename("selected_task_count").reset_index()
    )
    total = metrics.groupby("model", sort=False).size().rename("model_task_count")
    parameter_frequency = parameter_frequency.merge(total, on="model", how="left")
    parameter_frequency["selection_frequency"] = parameter_frequency["selected_task_count"] / parameter_frequency["model_task_count"]

    numeric = importance.copy()
    importance_columns = [
        "importance_gain", "importance_weight", "importance_cover", "importance_total_gain", "importance_total_cover"
    ]
    for column in importance_columns:
        if column in numeric.columns:
            numeric[column] = pd.to_numeric(numeric[column], errors="raise")
    numeric["gain_nonzero"] = numeric["importance_gain"] > 0
    aggregations = {
        "retained_task_count": ("importance_gain", "size"),
        "mean_importance_gain_retained": ("importance_gain", "mean"),
        "sd_importance_gain_retained": ("importance_gain", "std"),
        "median_importance_gain_retained": ("importance_gain", "median"),
        "mean_importance_weight_retained": ("importance_weight", "mean"),
        "mean_importance_cover_retained": ("importance_cover", "mean"),
        "gain_nonzero_task_count": ("gain_nonzero", "sum"),
        "gain_nonzero_frequency_among_retained": ("gain_nonzero", "mean"),
    }
    if "importance_total_gain" in numeric.columns:
        aggregations["mean_importance_total_gain_retained"] = ("importance_total_gain", "mean")
    if "importance_total_cover" in numeric.columns:
        aggregations["mean_importance_total_cover_retained"] = ("importance_total_cover", "mean")
    importance_summary = numeric.groupby(["model", "feature_name"], sort=False).agg(**aggregations).reset_index()
    importance_summary["sd_importance_gain_retained"] = importance_summary["sd_importance_gain_retained"].fillna(0.0)
    importance_summary = importance_summary.merge(
        total.rename("model_task_count_for_importance"),
        on="model",
        how="left",
        validate="many_to_one",
    )
    importance_summary["retained_frequency"] = (
        importance_summary["retained_task_count"]
        / importance_summary["model_task_count_for_importance"]
    )
    importance_summary["gain_nonzero_frequency_all_tasks"] = (
        importance_summary["gain_nonzero_task_count"]
        / importance_summary["model_task_count_for_importance"]
    )
    # Missing rows correspond to predictors removed by fold/task-local preprocessing;
    # treating them as zero avoids overstating cross-task tree importance stability.
    importance_summary["mean_importance_gain_all_tasks"] = (
        importance_summary["mean_importance_gain_retained"]
        * importance_summary["retained_frequency"]
    )

    return {
        "metrics": metrics,
        "predictions": predictions,
        "importance": importance,
        "metric_summary": metric_summary,
        "parameter_frequency": parameter_frequency,
        "importance_summary": importance_summary,
    }


def write_aggregation(task_root: Path, output_dir: Path, *, overwrite: bool = False) -> Mapping[str, Path]:
    tasks = discover_completed_tasks(task_root)
    aggregated = aggregate_completed_tasks(tasks)
    output = Path(output_dir).expanduser().resolve()
    paths: Dict[str, Path] = {
        "metrics": output / "08_holdout_task_metrics.csv",
        "predictions": output / "08_holdout_predictions.csv.gz",
        "importance": output / "08_final_model_feature_importance.csv.gz",
        "metric_summary": output / "08_metric_summary.csv",
        "parameter_frequency": output / "08_selected_parameter_frequency.csv",
        "importance_summary": output / "08_feature_importance_stability_summary.csv",
        "summary": output / "08_aggregation_summary.json",
        "complete": output / "08_REPEATED_HOLDOUT_AGGREGATION_COMPLETE.json",
    }
    existing = [p for p in paths.values() if p.exists()]
    if existing and not overwrite:
        raise FileExistsError("Aggregation outputs already exist; use --overwrite for a documented rerun:\n" + "\n".join(f"  - {p}" for p in existing))
    output.mkdir(parents=True, exist_ok=True)
    aggregated["metrics"].to_csv(paths["metrics"], index=False)
    aggregated["predictions"].to_csv(paths["predictions"], index=False, compression="gzip")
    aggregated["importance"].to_csv(paths["importance"], index=False, compression="gzip")
    aggregated["metric_summary"].to_csv(paths["metric_summary"], index=False)
    aggregated["parameter_frequency"].to_csv(paths["parameter_frequency"], index=False)
    aggregated["importance_summary"].to_csv(paths["importance_summary"], index=False)
    models = aggregated["metrics"]["model"].astype(str).drop_duplicates().tolist()
    summary = {
        "aggregation_type": "xgboost_repeated_holdout",
        "model_engine": "xgboost",
        "score_type": "probability",
        "probability_metrics_available": True,
        "task_root": str(Path(task_root).expanduser().resolve()),
        "completed_task_count": int(len(tasks)),
        "models": models,
        "model_count": int(len(models)),
        "metric_rows": int(len(aggregated["metrics"])),
        "prediction_rows": int(len(aggregated["predictions"])),
        "feature_importance_rows": int(len(aggregated["importance"])),
        "log_loss_and_brier_included": True,
    }
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paths["complete"].write_text(json.dumps({"status": "COMPLETE", **summary, "output_dir": str(output)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return paths
