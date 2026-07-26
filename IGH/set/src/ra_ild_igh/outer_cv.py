#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Outer-task orchestration, aggregation, and stability analysis for IGH V2.

The module operates only on the independent V2 nested-CV output tree. It does
not read the independent test set and never changes the frozen V1 results.
"""

from __future__ import annotations

import json
import math
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class OuterCVError(ValueError):
    """Raised when an outer-task tree or aggregate result is invalid."""


TASK_FILE_NAMES: Mapping[str, str] = MappingProxyType(
    {
        "configuration": "06_task_configuration.json",
        "sample_roles": "06_task_sample_roles.csv",
        "outer_train_public": "06_dynamic_public_features_outer_train_loo.csv",
        "outer_validation_public": "06_dynamic_public_features_outer_validation.csv",
        "public_reference_summary": "06_public_reference_build_summary.csv",
        "public_loo_assignments": "06_public_loo_assignments.csv",
        "inner_tuning": "06_inner_tuning_results.csv",
        "inner_oof": "06_inner_selected_oof_predictions.csv",
        "outer_predictions": "06_outer_validation_predictions.csv",
        "outer_metrics": "06_outer_validation_metrics.csv",
        "coefficients": "06_final_model_coefficients.csv",
        "preprocessing": "06_preprocessing_summary.csv",
        "static_feature_list": "06_static_igh_feature_list.csv",
        "complete": "06_TASK_COMPLETE.json",
    }
)

AGGREGATE_FILE_NAMES: Mapping[str, str] = MappingProxyType(
    {
        "manifest": "07_outer_task_manifest.csv",
        "status": "07_outer_task_status.csv",
        "all_metrics": "07_all_outer_metrics.csv",
        "all_predictions": "07_all_outer_predictions.csv.gz",
        "all_hyperparameters": "07_all_selected_hyperparameters.csv",
        "all_coefficients": "07_all_coefficients.csv.gz",
        "task_metric_summary": "07_outer_task_metric_summary.csv",
        "repeat_metrics": "07_repeat_pooled_metrics.csv",
        "hyperparameter_frequency": "07_hyperparameter_selection_frequency.csv",
        "feature_stability": "07_feature_stability.csv.gz",
        "stable_features": "07_stable_features.csv",
        "intercept_stability": "07_intercept_stability.csv",
        "sample_prediction_stability": "07_sample_prediction_stability.csv",
        "pairwise_model_differences": "07_pairwise_model_metric_differences.csv",
        "summary": "07_aggregation_summary.json",
        "complete": "07_AGGREGATION_COMPLETE.json",
    }
)


@dataclass(frozen=True)
class OuterTask:
    task_index: int
    outer_repeat: int
    outer_fold: int
    task_id: str
    output_dir: Path

    @property
    def complete_marker(self) -> Path:
        return self.output_dir / TASK_FILE_NAMES["complete"]


@dataclass(frozen=True)
class TaskInspection:
    task: OuterTask
    status: str
    reason: str
    n_outer_train: Optional[int] = None
    n_outer_validation: Optional[int] = None
    runtime_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "task_index": self.task.task_index,
            "task_id": self.task.task_id,
            "outer_repeat": self.task.outer_repeat,
            "outer_fold": self.task.outer_fold,
            "output_dir": str(self.task.output_dir),
            "complete_marker": str(self.task.complete_marker),
            "status": self.status,
            "reason": self.reason,
            "n_outer_train": self.n_outer_train if self.n_outer_train is not None else "",
            "n_outer_validation": (
                self.n_outer_validation if self.n_outer_validation is not None else ""
            ),
            "runtime_seconds": self.runtime_seconds if self.runtime_seconds is not None else "",
        }


@dataclass(frozen=True)
class OuterAggregate:
    manifest: pd.DataFrame
    status: pd.DataFrame
    all_metrics: pd.DataFrame
    all_predictions: pd.DataFrame
    all_hyperparameters: pd.DataFrame
    all_coefficients: pd.DataFrame
    task_metric_summary: pd.DataFrame
    repeat_metrics: pd.DataFrame
    hyperparameter_frequency: pd.DataFrame
    feature_stability: pd.DataFrame
    stable_features: pd.DataFrame
    intercept_stability: pd.DataFrame
    sample_prediction_stability: pd.DataFrame
    pairwise_model_differences: pd.DataFrame


def task_output_paths(output_dir: Path) -> Dict[str, Path]:
    directory = Path(output_dir)
    return {key: directory / name for key, name in TASK_FILE_NAMES.items()}


def aggregate_output_paths(output_dir: Path) -> Dict[str, Path]:
    directory = Path(output_dir)
    return {key: directory / name for key, name in AGGREGATE_FILE_NAMES.items()}


def build_outer_tasks(
    outer_repeats: int,
    outer_folds: int,
    output_root: Path,
) -> Tuple[OuterTask, ...]:
    if isinstance(outer_repeats, bool) or int(outer_repeats) < 1:
        raise OuterCVError("outer_repeats must be an integer >= 1.")
    if isinstance(outer_folds, bool) or int(outer_folds) < 2:
        raise OuterCVError("outer_folds must be an integer >= 2.")
    root = Path(output_root).expanduser().resolve()
    tasks: List[OuterTask] = []
    index = 0
    for repeat in range(1, int(outer_repeats) + 1):
        for fold in range(1, int(outer_folds) + 1):
            index += 1
            task_id = f"repeat_{repeat:02d}_fold_{fold:02d}"
            tasks.append(
                OuterTask(
                    task_index=index,
                    outer_repeat=repeat,
                    outer_fold=fold,
                    task_id=task_id,
                    output_dir=root / task_id,
                )
            )
    return tuple(tasks)


def task_command(
    task: OuterTask,
    *,
    python_executable: str,
    runner_script: Path,
    config_path: Path,
    overwrite: bool = False,
) -> Tuple[str, ...]:
    executable = str(python_executable).strip()
    if not executable:
        raise OuterCVError("python_executable must be non-empty.")
    command = [
        executable,
        str(Path(runner_script).resolve()),
        "--config",
        str(Path(config_path).resolve()),
        "--outer-repeat",
        str(task.outer_repeat),
        "--outer-fold",
        str(task.outer_fold),
        "--output-dir",
        str(task.output_dir),
    ]
    if overwrite:
        command.append("--overwrite")
    return tuple(command)


def manifest_frame(
    tasks: Sequence[OuterTask],
    *,
    python_executable: str,
    runner_script: Path,
    config_path: Path,
) -> pd.DataFrame:
    rows = []
    for task in tasks:
        command = task_command(
            task,
            python_executable=python_executable,
            runner_script=runner_script,
            config_path=config_path,
        )
        rows.append(
            {
                "task_index": task.task_index,
                "task_id": task.task_id,
                "outer_repeat": task.outer_repeat,
                "outer_fold": task.outer_fold,
                "output_dir": str(task.output_dir),
                "complete_marker": str(task.complete_marker),
                "command": shlex.join(command),
            }
        )
    frame = pd.DataFrame(rows)
    if frame["task_id"].duplicated().any():
        raise OuterCVError("Generated outer-task manifest contains duplicate task IDs.")
    return frame


def _load_json(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OuterCVError(f"Unable to read {label}: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise OuterCVError(f"{label} must contain a JSON object: {path}")
    return value


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise OuterCVError(f"{label} is missing columns: {missing}")


def inspect_outer_task(
    task: OuterTask,
    *,
    expected_models: Sequence[str],
    expected_samples: int,
    expected_candidates_per_model: int,
    expected_static_features: int,
) -> TaskInspection:
    models = tuple(str(value) for value in expected_models)
    if not task.output_dir.exists():
        return TaskInspection(task, "missing", "output directory does not exist")
    if not task.output_dir.is_dir():
        return TaskInspection(task, "invalid", "output path is not a directory")
    paths = task_output_paths(task.output_dir)
    missing = [key for key, path in paths.items() if not path.is_file()]
    if missing:
        return TaskInspection(
            task,
            "incomplete",
            "missing required files: " + ",".join(missing),
        )
    try:
        marker = _load_json(paths["complete"], "task completion marker")
        configuration = _load_json(paths["configuration"], "task configuration")
        if marker.get("status") != "COMPLETE":
            raise OuterCVError(f"completion status={marker.get('status')!r}")
        for source, label in ((marker, "marker"), (configuration, "configuration")):
            if int(source.get("outer_repeat", -1)) != task.outer_repeat:
                raise OuterCVError(f"{label} outer_repeat mismatch")
            if int(source.get("outer_fold", -1)) != task.outer_fold:
                raise OuterCVError(f"{label} outer_fold mismatch")
        n_train = int(configuration["outer_train_samples"])
        n_valid = int(configuration["outer_validation_samples"])
        if n_train + n_valid != int(expected_samples):
            raise OuterCVError(
                f"task sample count={n_train + n_valid}, expected={expected_samples}"
            )
        configured_models = tuple(configuration.get("models", []))
        if configured_models != models:
            raise OuterCVError(
                f"model order mismatch: observed={configured_models}, expected={models}"
            )
        if int(configuration.get("candidate_count_per_model", -1)) != int(
            expected_candidates_per_model
        ):
            raise OuterCVError("candidate count mismatch")

        roles = pd.read_csv(paths["sample_roles"])
        _require_columns(roles, ("sample_id", "outer_role"), "sample roles")
        if len(roles) != expected_samples or roles["sample_id"].duplicated().any():
            raise OuterCVError("sample-role rows or uniqueness are invalid")
        role_counts = roles["outer_role"].value_counts().to_dict()
        if role_counts.get("training", 0) != n_train or role_counts.get("validation", 0) != n_valid:
            raise OuterCVError("sample-role train/validation counts are invalid")

        metrics = pd.read_csv(paths["outer_metrics"])
        _require_columns(
            metrics,
            (
                "model",
                "outer_repeat",
                "outer_fold",
                "selected_l1_ratio_alpha",
                "selected_lambda",
            ),
            "outer metrics",
        )
        if len(metrics) != len(models) or tuple(metrics["model"].astype(str)) != models:
            raise OuterCVError("outer metric model rows/order are invalid")

        prediction = pd.read_csv(paths["outer_predictions"])
        _require_columns(
            prediction,
            ("model", "sample_id", "true_label", "probability_ILD", "predicted_label"),
            "outer predictions",
        )
        if len(prediction) != len(models) * n_valid:
            raise OuterCVError("outer prediction row count is invalid")
        if prediction.duplicated(["model", "sample_id"]).any():
            raise OuterCVError("outer predictions contain duplicate model/sample rows")

        tuning = pd.read_csv(paths["inner_tuning"])
        _require_columns(tuning, ("model", "l1_ratio_alpha", "lambda"), "inner tuning")
        if len(tuning) != len(models) * expected_candidates_per_model:
            raise OuterCVError("inner tuning row count is invalid")
        if tuning.duplicated(["model", "l1_ratio_alpha", "lambda"]).any():
            raise OuterCVError("inner tuning contains duplicate candidates")

        oof = pd.read_csv(paths["inner_oof"])
        _require_columns(oof, ("model", "sample_id", "inner_fold"), "selected inner OOF")
        if len(oof) != len(models) * n_train:
            raise OuterCVError("selected inner OOF row count is invalid")
        if oof.duplicated(["model", "sample_id"]).any():
            raise OuterCVError("selected inner OOF contains duplicate model/sample rows")

        coefficients = pd.read_csv(paths["coefficients"])
        _require_columns(
            coefficients,
            ("model", "feature_name", "coefficient", "nonzero"),
            "coefficients",
        )
        if coefficients.duplicated(["model", "feature_name"]).any():
            raise OuterCVError("coefficient table contains duplicate model/features")
        if set(coefficients["model"].astype(str)) != set(models):
            raise OuterCVError("coefficient table does not contain every model")

        static_features = pd.read_csv(paths["static_feature_list"])
        _require_columns(static_features, ("feature_order", "feature_name"), "static features")
        if len(static_features) != expected_static_features:
            raise OuterCVError("static feature count is invalid")
        if static_features["feature_name"].duplicated().any():
            raise OuterCVError("static feature list contains duplicates")

        runtime = float(marker.get("runtime_seconds", configuration.get("runtime_seconds", np.nan)))
        return TaskInspection(
            task,
            "complete",
            "all required outputs passed structural validation",
            n_outer_train=n_train,
            n_outer_validation=n_valid,
            runtime_seconds=runtime if np.isfinite(runtime) else None,
        )
    except Exception as exc:
        return TaskInspection(task, "invalid", str(exc))


def scan_outer_tasks(
    tasks: Sequence[OuterTask],
    *,
    expected_models: Sequence[str],
    expected_samples: int,
    expected_candidates_per_model: int,
    expected_static_features: int,
) -> pd.DataFrame:
    inspections = [
        inspect_outer_task(
            task,
            expected_models=expected_models,
            expected_samples=expected_samples,
            expected_candidates_per_model=expected_candidates_per_model,
            expected_static_features=expected_static_features,
        )
        for task in tasks
    ]
    return pd.DataFrame([item.to_dict() for item in inspections])


def _run_one_command(command: Sequence[str], log_path: Path) -> Dict[str, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND: " + shlex.join(command) + "\n\n")
        handle.flush()
        completed = subprocess.run(
            list(command),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return {
        "returncode": int(completed.returncode),
        "runtime_seconds": float(time.time() - started),
        "log_path": str(log_path),
    }


def execute_outer_tasks(
    tasks: Sequence[OuterTask],
    current_status: pd.DataFrame,
    *,
    python_executable: str,
    runner_script: Path,
    config_path: Path,
    log_dir: Path,
    max_workers: int = 1,
    rerun_incomplete: bool = False,
) -> pd.DataFrame:
    if isinstance(max_workers, bool) or int(max_workers) < 1:
        raise OuterCVError("max_workers must be an integer >= 1.")
    _require_columns(current_status, ("task_id", "status"), "current status")
    status_lookup = current_status.set_index("task_id")["status"].astype(str).to_dict()
    runnable: List[Tuple[OuterTask, Tuple[str, ...], Path]] = []
    rows: List[Dict[str, object]] = []
    for task in tasks:
        status = status_lookup.get(task.task_id, "missing")
        if status == "complete":
            rows.append(
                {
                    "task_id": task.task_id,
                    "outer_repeat": task.outer_repeat,
                    "outer_fold": task.outer_fold,
                    "action": "skipped_complete",
                    "returncode": 0,
                    "runtime_seconds": 0.0,
                    "log_path": "",
                }
            )
            continue
        if status in {"incomplete", "invalid"} and not rerun_incomplete:
            rows.append(
                {
                    "task_id": task.task_id,
                    "outer_repeat": task.outer_repeat,
                    "outer_fold": task.outer_fold,
                    "action": "blocked_incomplete_or_invalid",
                    "returncode": "",
                    "runtime_seconds": 0.0,
                    "log_path": "",
                }
            )
            continue
        overwrite = status in {"incomplete", "invalid"}
        command = task_command(
            task,
            python_executable=python_executable,
            runner_script=runner_script,
            config_path=config_path,
            overwrite=overwrite,
        )
        runnable.append((task, command, Path(log_dir) / f"{task.task_id}.log"))

    with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
        futures = {
            executor.submit(_run_one_command, command, log_path): task
            for task, command, log_path in runnable
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                rows.append(
                    {
                        "task_id": task.task_id,
                        "outer_repeat": task.outer_repeat,
                        "outer_fold": task.outer_fold,
                        "action": "completed" if result["returncode"] == 0 else "failed",
                        **result,
                    }
                )
            except Exception as exc:  # defensive scheduler boundary
                rows.append(
                    {
                        "task_id": task.task_id,
                        "outer_repeat": task.outer_repeat,
                        "outer_fold": task.outer_fold,
                        "action": "scheduler_exception",
                        "returncode": -1,
                        "runtime_seconds": 0.0,
                        "log_path": "",
                        "error": str(exc),
                    }
                )
    output = pd.DataFrame(rows)
    if not output.empty:
        output = output.sort_values(["outer_repeat", "outer_fold"]).reset_index(drop=True)
    return output


def _read_task_frame(task: OuterTask, key: str) -> pd.DataFrame:
    path = task_output_paths(task.output_dir)[key]
    frame = pd.read_csv(path)
    if "outer_repeat" not in frame.columns:
        frame["outer_repeat"] = task.outer_repeat
    if "outer_fold" not in frame.columns:
        frame["outer_fold"] = task.outer_fold
    frame["task_id"] = task.task_id
    frame["task_index"] = task.task_index
    return frame


def _metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_names = [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "sensitivity_recall",
        "specificity",
        "precision",
        "f1",
    ]
    rows = []
    for model, group in metrics.groupby("model", sort=False):
        for metric in metric_names:
            values = pd.to_numeric(group[metric], errors="raise").to_numpy(float)
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "n_outer_tasks": len(values),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(np.median(values)),
                    "q025": float(np.quantile(values, 0.025)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "q975": float(np.quantile(values, 0.975)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def _confusion_metrics(y: np.ndarray, probability: np.ndarray, prediction: np.ndarray) -> Dict[str, object]:
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = np.asarray(prediction, dtype=int)
    if set(np.unique(y).tolist()) != {0, 1}:
        raise OuterCVError("Repeat-level predictions must contain both classes.")
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {
        "n_samples": int(len(y)),
        "n_negative": int(np.sum(y == 0)),
        "n_positive": int(np.sum(y == 1)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "accuracy": float(accuracy_score(y, prediction)),
        "sensitivity_recall": float(recall_score(y, prediction, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def _repeat_metrics(predictions: pd.DataFrame, expected_samples: int) -> pd.DataFrame:
    rows = []
    for (model, repeat), group in predictions.groupby(["model", "outer_repeat"], sort=False):
        if len(group) != expected_samples:
            raise OuterCVError(
                f"{model} repeat {repeat} has {len(group)} OOF predictions; "
                f"expected {expected_samples}."
            )
        if group["sample_id"].duplicated().any():
            raise OuterCVError(f"{model} repeat {repeat} has duplicate sample predictions.")
        metrics = _confusion_metrics(
            group["true_label"].to_numpy(int),
            group["probability_ILD"].to_numpy(float),
            group["predicted_label"].to_numpy(int),
        )
        rows.append({"model": model, "outer_repeat": int(repeat), **metrics})
    return pd.DataFrame(rows)


def _hyperparameter_frequency(metrics: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    selected = metrics.loc[
        :,
        [
            "task_id",
            "task_index",
            "outer_repeat",
            "outer_fold",
            "model",
            "selected_l1_ratio_alpha",
            "selected_lambda",
            "inner_selected_roc_auc",
            "inner_selected_pr_auc",
        ],
    ].copy()
    selected = selected.rename(columns={"selected_l1_ratio_alpha": "alpha", "selected_lambda": "lambda"})
    rows = []
    for model, model_rows in selected.groupby("model", sort=False):
        denominator = len(model_rows)
        for (alpha, lambda_value), group in model_rows.groupby(["alpha", "lambda"], sort=True):
            rows.append(
                {
                    "model": model,
                    "alpha": float(alpha),
                    "lambda": float(lambda_value),
                    "selection_count": int(len(group)),
                    "selection_frequency": float(len(group) / denominator),
                    "n_model_tasks": int(denominator),
                    "mean_inner_roc_auc": float(group["inner_selected_roc_auc"].mean()),
                    "mean_inner_pr_auc": float(group["inner_selected_pr_auc"].mean()),
                }
            )
    frequency = pd.DataFrame(rows).sort_values(
        ["model", "selection_count", "lambda", "alpha"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    return selected, frequency


def _coefficient_stability(
    coefficients: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    coefficient_tolerance: float,
    minimum_selection_frequency: float,
    minimum_sign_consistency: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require_columns(coefficients, ("model", "feature_name", "coefficient"), "coefficients")
    task_counts = metrics.groupby("model")["task_id"].nunique().to_dict()
    feature_rows = []
    intercept_rows = []
    for (model, feature), group in coefficients.groupby(["model", "feature_name"], sort=False):
        values = pd.to_numeric(group["coefficient"], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise OuterCVError(f"Non-finite coefficients for {model}/{feature}.")
        n_total = int(task_counts[model])
        n_available = int(group["task_id"].nunique())
        selected_mask = np.abs(values) > float(coefficient_tolerance)
        selected_values = values[selected_mask]
        selection_count = int(np.sum(selected_mask))
        positive_count = int(np.sum(selected_values > 0))
        negative_count = int(np.sum(selected_values < 0))
        sign_consistency = (
            float(max(positive_count, negative_count) / selection_count)
            if selection_count
            else 0.0
        )
        dominant_sign = (
            "positive" if positive_count > negative_count else
            "negative" if negative_count > positive_count else
            "tie_or_none"
        )
        row = {
            "model": model,
            "feature_name": feature,
            "n_model_tasks": n_total,
            "tasks_available": n_available,
            "availability_frequency": float(n_available / n_total),
            "selection_count": selection_count,
            "selection_frequency": float(selection_count / n_total),
            "selection_frequency_when_available": (
                float(selection_count / n_available) if n_available else 0.0
            ),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "sign_consistency": sign_consistency,
            "dominant_sign": dominant_sign,
            "coefficient_sum": float(np.sum(values)),
            "mean_coefficient_all_tasks": float(np.sum(values) / n_total),
            "mean_abs_coefficient_all_tasks": float(np.sum(np.abs(values)) / n_total),
            "mean_selected_coefficient": (
                float(np.mean(selected_values)) if selection_count else 0.0
            ),
            "median_selected_coefficient": (
                float(np.median(selected_values)) if selection_count else 0.0
            ),
            "mean_abs_selected_coefficient": (
                float(np.mean(np.abs(selected_values))) if selection_count else 0.0
            ),
            "stable_selection": bool(
                feature != "__INTERCEPT__"
                and selection_count > 0
                and selection_count / n_total >= float(minimum_selection_frequency)
                and sign_consistency >= float(minimum_sign_consistency)
            ),
        }
        if feature == "__INTERCEPT__":
            intercept_rows.append(row)
        else:
            feature_rows.append(row)
    feature_frame = pd.DataFrame(feature_rows)
    if not feature_frame.empty:
        feature_frame = feature_frame.sort_values(
            ["model", "stable_selection", "selection_frequency", "sign_consistency", "mean_abs_coefficient_all_tasks"],
            ascending=[True, False, False, False, False],
        ).reset_index(drop=True)
    stable = feature_frame.loc[feature_frame["stable_selection"]].copy()
    intercept = pd.DataFrame(intercept_rows)
    return feature_frame, stable, intercept


def _sample_prediction_stability(
    predictions: pd.DataFrame,
    *,
    expected_repeats: int,
    require_all_tasks: bool,
) -> pd.DataFrame:
    rows = []
    for (model, sample_id), group in predictions.groupby(["model", "sample_id"], sort=False):
        truth = group["true_label"].astype(int)
        if truth.nunique() != 1:
            raise OuterCVError(f"Inconsistent true labels for {model}/{sample_id}.")
        n = int(len(group))
        if require_all_tasks and n != expected_repeats:
            raise OuterCVError(
                f"{model}/{sample_id} has {n} repeated OOF predictions; "
                f"expected {expected_repeats}."
            )
        probability = group["probability_ILD"].to_numpy(float)
        predicted = group["predicted_label"].to_numpy(int)
        true_value = int(truth.iloc[0])
        rows.append(
            {
                "model": model,
                "sample_id": sample_id,
                "true_label": true_value,
                "true_cohort": str(group["true_cohort"].iloc[0]),
                "n_predictions": n,
                "expected_predictions": int(expected_repeats),
                "coverage_frequency": float(n / expected_repeats),
                "mean_probability_ILD": float(np.mean(probability)),
                "sd_probability_ILD": float(np.std(probability, ddof=1)) if n > 1 else 0.0,
                "median_probability_ILD": float(np.median(probability)),
                "min_probability_ILD": float(np.min(probability)),
                "max_probability_ILD": float(np.max(probability)),
                "positive_classification_frequency": float(np.mean(predicted)),
                "correct_classification_frequency": float(np.mean(predicted == true_value)),
                "mean_threshold": float(group["threshold"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "sample_id"]).reset_index(drop=True)


def _pairwise_model_differences(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_names = ["roc_auc", "pr_auc", "sensitivity_recall", "specificity", "f1"]
    models = list(dict.fromkeys(metrics["model"].astype(str)))
    indexed = metrics.set_index(["task_id", "model"])
    rows = []
    for index_a in range(len(models)):
        for index_b in range(index_a + 1, len(models)):
            model_a = models[index_a]
            model_b = models[index_b]
            common = sorted(
                set(metrics.loc[metrics["model"] == model_a, "task_id"])
                & set(metrics.loc[metrics["model"] == model_b, "task_id"])
            )
            for metric in metric_names:
                a = indexed.loc[(common, model_a), metric].to_numpy(float)
                b = indexed.loc[(common, model_b), metric].to_numpy(float)
                difference = b - a
                rows.append(
                    {
                        "reference_model": model_a,
                        "comparison_model": model_b,
                        "metric": metric,
                        "n_paired_tasks": len(common),
                        "mean_difference_comparison_minus_reference": float(np.mean(difference)),
                        "sd_difference": float(np.std(difference, ddof=1)) if len(difference) > 1 else 0.0,
                        "median_difference": float(np.median(difference)),
                        "comparison_win_frequency": float(np.mean(difference > 0)),
                        "tie_frequency": float(np.mean(np.isclose(difference, 0.0, atol=1e-15, rtol=0.0))),
                        "comparison_loss_frequency": float(np.mean(difference < 0)),
                    }
                )
    return pd.DataFrame(rows)


def aggregate_outer_results(
    tasks: Sequence[OuterTask],
    status: pd.DataFrame,
    *,
    expected_models: Sequence[str],
    expected_samples: int,
    expected_repeats: int,
    require_all_tasks: bool,
    coefficient_tolerance: float,
    minimum_selection_frequency: float,
    minimum_sign_consistency: float,
    manifest: Optional[pd.DataFrame] = None,
) -> OuterAggregate:
    _require_columns(status, ("task_id", "status"), "task status")
    task_lookup = {task.task_id: task for task in tasks}
    complete_ids = status.loc[status["status"] == "complete", "task_id"].astype(str).tolist()
    bad = status.loc[status["status"].isin(["invalid", "incomplete"])]
    if not bad.empty:
        raise OuterCVError(
            "Cannot aggregate while invalid/incomplete tasks exist: "
            + ", ".join(bad["task_id"].astype(str).head(10))
        )
    if require_all_tasks and len(complete_ids) != len(tasks):
        raise OuterCVError(
            f"Complete outer tasks={len(complete_ids)}, expected={len(tasks)}."
        )
    if not complete_ids:
        raise OuterCVError("No complete outer tasks are available for aggregation.")

    metric_parts = []
    prediction_parts = []
    coefficient_parts = []
    for task_id in complete_ids:
        task = task_lookup[task_id]
        metric_parts.append(_read_task_frame(task, "outer_metrics"))
        prediction_parts.append(_read_task_frame(task, "outer_predictions"))
        coefficient_parts.append(_read_task_frame(task, "coefficients"))
    metrics = pd.concat(metric_parts, ignore_index=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    coefficients = pd.concat(coefficient_parts, ignore_index=True)

    if metrics.duplicated(["task_id", "model"]).any():
        raise OuterCVError("Aggregated metrics contain duplicate task/model rows.")
    if predictions.duplicated(["task_id", "model", "sample_id"]).any():
        raise OuterCVError("Aggregated predictions contain duplicate task/model/sample rows.")
    observed_models = tuple(dict.fromkeys(metrics["model"].astype(str)))
    if observed_models != tuple(expected_models):
        raise OuterCVError(
            f"Aggregated model order={observed_models}, expected={tuple(expected_models)}."
        )

    selected, hyper_frequency = _hyperparameter_frequency(metrics)
    task_summary = _metric_summary(metrics)
    repeat_metrics = (
        _repeat_metrics(predictions, expected_samples)
        if require_all_tasks
        else pd.DataFrame()
    )
    feature_stability, stable_features, intercept = _coefficient_stability(
        coefficients,
        metrics,
        coefficient_tolerance=coefficient_tolerance,
        minimum_selection_frequency=minimum_selection_frequency,
        minimum_sign_consistency=minimum_sign_consistency,
    )
    sample_stability = _sample_prediction_stability(
        predictions,
        expected_repeats=expected_repeats,
        require_all_tasks=require_all_tasks,
    )
    pairwise = _pairwise_model_differences(metrics)
    manifest_frame_value = manifest.copy() if manifest is not None else pd.DataFrame()
    return OuterAggregate(
        manifest=manifest_frame_value,
        status=status.copy(),
        all_metrics=metrics,
        all_predictions=predictions,
        all_hyperparameters=selected,
        all_coefficients=coefficients,
        task_metric_summary=task_summary,
        repeat_metrics=repeat_metrics,
        hyperparameter_frequency=hyper_frequency,
        feature_stability=feature_stability,
        stable_features=stable_features,
        intercept_stability=intercept,
        sample_prediction_stability=sample_stability,
        pairwise_model_differences=pairwise,
    )


def write_outer_aggregate(
    aggregate: OuterAggregate,
    output_dir: Path,
    *,
    experiment_id: str,
    expected_tasks: int,
    require_all_tasks: bool,
    overwrite: bool = False,
) -> Dict[str, Path]:
    paths = aggregate_output_paths(output_dir)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Aggregation outputs already exist; add --overwrite only for a documented rerun:\n"
            + "\n".join(f"  - {path}" for path in existing)
        )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    aggregate.manifest.to_csv(paths["manifest"], index=False)
    aggregate.status.to_csv(paths["status"], index=False)
    aggregate.all_metrics.to_csv(paths["all_metrics"], index=False)
    aggregate.all_predictions.to_csv(paths["all_predictions"], index=False, compression="gzip")
    aggregate.all_hyperparameters.to_csv(paths["all_hyperparameters"], index=False)
    aggregate.all_coefficients.to_csv(paths["all_coefficients"], index=False, compression="gzip")
    aggregate.task_metric_summary.to_csv(paths["task_metric_summary"], index=False)
    aggregate.repeat_metrics.to_csv(paths["repeat_metrics"], index=False)
    aggregate.hyperparameter_frequency.to_csv(paths["hyperparameter_frequency"], index=False)
    aggregate.feature_stability.to_csv(paths["feature_stability"], index=False, compression="gzip")
    aggregate.stable_features.to_csv(paths["stable_features"], index=False)
    aggregate.intercept_stability.to_csv(paths["intercept_stability"], index=False)
    aggregate.sample_prediction_stability.to_csv(paths["sample_prediction_stability"], index=False)
    aggregate.pairwise_model_differences.to_csv(paths["pairwise_model_differences"], index=False)

    status_counts = aggregate.status["status"].value_counts().to_dict()
    summary = {
        "experiment_id": experiment_id,
        "expected_outer_tasks": int(expected_tasks),
        "require_all_tasks": bool(require_all_tasks),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "completed_outer_tasks": int(status_counts.get("complete", 0)),
        "models": list(dict.fromkeys(aggregate.all_metrics["model"].astype(str))),
        "outer_metric_rows": int(len(aggregate.all_metrics)),
        "outer_prediction_rows": int(len(aggregate.all_predictions)),
        "coefficient_rows": int(len(aggregate.all_coefficients)),
        "stable_feature_rows": int(len(aggregate.stable_features)),
        "independent_test_read": False,
    }
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    marker = {
        "status": "COMPLETE" if require_all_tasks else "PARTIAL",
        "experiment_id": experiment_id,
        "completed_outer_tasks": int(status_counts.get("complete", 0)),
        "expected_outer_tasks": int(expected_tasks),
        "output_dir": str(Path(output_dir).resolve()),
        "independent_test_read": False,
    }
    paths["complete"].write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths
