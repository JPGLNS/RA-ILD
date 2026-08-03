#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate and compare frozen repeated-holdout RA-ILD IGH experiments.

This module is deliberately separate from the classical repeated outer-CV
aggregator.  In a repeated holdout, every repeat evaluates only the configured
holdout patients; a patient may therefore have zero, one, or multiple holdout
predictions across repeats.  Treating these outputs as complete OOF predictions
for the whole cohort would be scientifically incorrect.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class RepeatedHoldoutSummaryError(ValueError):
    """Raised when repeated-holdout outputs violate their frozen contract."""


METRIC_NAMES: Tuple[str, ...] = (
    "roc_auc",
    "pr_auc",
    "log_loss",
    "brier_score",
    "accuracy",
    "sensitivity_recall",
    "specificity",
    "precision",
    "f1",
)

LOWER_IS_BETTER_METRICS = frozenset({"log_loss", "brier_score"})

OUTPUT_FILE_NAMES: Mapping[str, str] = {
    "manifest": "08_task_manifest.csv",
    "status": "08_task_status.csv",
    "metrics": "08_holdout_task_metrics.csv",
    "predictions": "08_holdout_predictions.csv.gz",
    "hyperparameters": "08_selected_hyperparameters.csv",
    "hyperparameter_frequency": "08_hyperparameter_frequency.csv",
    "threshold_summary": "08_threshold_summary.csv",
    "metric_summary": "08_model_metric_summary.csv",
    "model_ranking": "08_model_ranking.csv",
    "coefficients": "08_all_coefficients.csv.gz",
    "feature_stability": "08_feature_stability.csv.gz",
    "stable_features": "08_stable_features.csv",
    "pairwise_model_differences": "08_pairwise_model_metric_differences.csv",
    "sample_prediction_summary": "08_sample_holdout_prediction_summary.csv",
    "membership_audit": "08_split_membership_audit.csv",
    "summary": "08_aggregation_summary.json",
    "complete": "08_REPEATED_HOLDOUT_AGGREGATION_COMPLETE.json",
}

TASK_FILE_NAMES: Mapping[str, str] = {
    "configuration": "06_task_configuration.json",
    "sample_roles": "06_task_sample_roles.csv",
    "outer_predictions": "06_outer_validation_predictions.csv",
    "outer_metrics": "06_outer_validation_metrics.csv",
    "coefficients": "06_final_model_coefficients.csv",
    "complete": "06_TASK_COMPLETE.json",
}


@dataclass(frozen=True)
class RepeatedHoldoutAggregate:
    manifest: pd.DataFrame
    status: pd.DataFrame
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    selected_hyperparameters: pd.DataFrame
    hyperparameter_frequency: pd.DataFrame
    threshold_summary: pd.DataFrame
    metric_summary: pd.DataFrame
    model_ranking: pd.DataFrame
    coefficients: pd.DataFrame
    feature_stability: pd.DataFrame
    stable_features: pd.DataFrame
    pairwise_model_differences: pd.DataFrame
    sample_prediction_summary: pd.DataFrame
    membership_audit: pd.DataFrame


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise RepeatedHoldoutSummaryError(f"Unable to read {label}: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RepeatedHoldoutSummaryError(f"{label} must contain a JSON object: {path}")
    return value


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RepeatedHoldoutSummaryError(f"{label} is missing columns: {missing}")


def load_frozen_assignments(
    assignments_path: Path,
    frozen_marker_path: Path,
    *,
    expected_split_set_id: str,
    expected_splits: int,
    expected_train_size: int,
    expected_holdout_size: int,
) -> Tuple[pd.DataFrame, Mapping[str, Any], str, str]:
    assignments_path = Path(assignments_path)
    frozen_marker_path = Path(frozen_marker_path)
    if not assignments_path.is_file() or not frozen_marker_path.is_file():
        raise FileNotFoundError("Frozen split assignments or marker is missing")
    marker = _json(frozen_marker_path, "frozen split marker")
    if marker.get("status") != "FROZEN":
        raise RepeatedHoldoutSummaryError("Split marker status must be FROZEN")
    if str(marker.get("split_set_id")) != str(expected_split_set_id):
        raise RepeatedHoldoutSummaryError("Split-set ID does not match the experiment contract")
    actual_assignment_sha = sha256_file(assignments_path)
    marker_files = marker.get("files", {})
    expected_sha = (
        marker_files.get("assignments", {}).get("sha256")
        if isinstance(marker_files, Mapping)
        else None
    )
    if expected_sha and str(expected_sha) != actual_assignment_sha:
        raise RepeatedHoldoutSummaryError("Frozen assignment SHA256 does not match SPLITS_FROZEN.json")

    assignments = pd.read_csv(assignments_path)
    _require_columns(
        assignments,
        ("split_set_id", "split_id", "repeat_index", "role", "sample_id", "patient_id", "cohort"),
        "repeated holdout assignments",
    )
    assignments["sample_id"] = assignments["sample_id"].astype(str)
    assignments["patient_id"] = assignments["patient_id"].astype(str)
    assignments["repeat_index"] = pd.to_numeric(assignments["repeat_index"], errors="raise").astype(int)
    if set(assignments["split_set_id"].astype(str)) != {str(expected_split_set_id)}:
        raise RepeatedHoldoutSummaryError("Assignment rows contain an unexpected split_set_id")
    repeats = sorted(assignments["repeat_index"].unique().tolist())
    if repeats != list(range(1, int(expected_splits) + 1)):
        raise RepeatedHoldoutSummaryError(f"Observed repeats={repeats}, expected 1..{expected_splits}")
    if assignments.duplicated(["repeat_index", "sample_id"]).any():
        raise RepeatedHoldoutSummaryError("A sample has more than one role within a split")
    for repeat, group in assignments.groupby("repeat_index", sort=True):
        counts = group["role"].astype(str).value_counts().to_dict()
        if counts.get("train", 0) != int(expected_train_size):
            raise RepeatedHoldoutSummaryError(f"Repeat {repeat} train size is invalid: {counts}")
        if counts.get("holdout", 0) != int(expected_holdout_size):
            raise RepeatedHoldoutSummaryError(f"Repeat {repeat} holdout size is invalid: {counts}")
        if len(group) != int(expected_train_size) + int(expected_holdout_size):
            raise RepeatedHoldoutSummaryError(f"Repeat {repeat} total sample count is invalid")
    return assignments, marker, actual_assignment_sha, sha256_file(frozen_marker_path)


def _task_paths(task: Any) -> Dict[str, Path]:
    root = Path(task.output_dir)
    return {key: root / name for key, name in TASK_FILE_NAMES.items()}


def _metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model, group in metrics.groupby("model", sort=False):
        for metric in METRIC_NAMES:
            values = pd.to_numeric(group[metric], errors="raise").to_numpy(float)
            rows.append(
                {
                    "model": str(model),
                    "metric": metric,
                    "n_splits": int(len(values)),
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


def _model_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    wide = summary.pivot(index="model", columns="metric", values="mean").reset_index()
    for metric in METRIC_NAMES:
        if metric not in wide.columns:
            wide[metric] = np.nan
    wide["rank_mean_roc_auc"] = wide["roc_auc"].rank(method="min", ascending=False).astype(int)
    wide["rank_mean_pr_auc"] = wide["pr_auc"].rank(method="min", ascending=False).astype(int)
    wide["rank_mean_log_loss"] = wide["log_loss"].rank(method="min", ascending=True).astype(int)
    wide["rank_mean_brier_score"] = wide["brier_score"].rank(method="min", ascending=True).astype(int)
    wide = wide.sort_values(
        ["roc_auc", "pr_auc", "f1", "specificity", "sensitivity_recall", "model"],
        ascending=[False, False, False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    wide.insert(0, "descriptive_rank", np.arange(1, len(wide) + 1))
    wide["ranking_rule"] = "mean_roc_auc_then_mean_pr_auc_then_mean_f1_descriptive_only"
    wide["automatic_final_model_selection"] = False
    return wide


def _hyperparameter_tables(
    metrics: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normalized = metrics.copy()
    legacy_defaults = {
        "inner_selected_log_loss": np.nan,
        "inner_selected_brier_score": np.nan,
        "tuning_primary_metric": "roc_auc",
        "candidate_selection_policy": "pooled_roc_pr_lambda_alpha",
    }
    for column, default in legacy_defaults.items():
        if column not in normalized.columns:
            normalized[column] = default

    selected = normalized.loc[:, [
        "split_set_id", "split_id", "task_id",
        "outer_repeat", "outer_fold", "model",
        "selected_l1_ratio_alpha", "selected_lambda", "threshold",
        "inner_selected_roc_auc", "inner_selected_pr_auc",
        "inner_selected_log_loss", "inner_selected_brier_score",
        "tuning_primary_metric", "candidate_selection_policy",
    ]].copy()
    selected = selected.rename(
        columns={
            "selected_l1_ratio_alpha": "alpha",
            "selected_lambda": "lambda",
        }
    )

    frequency_rows: List[Dict[str, Any]] = []
    for model, model_rows in selected.groupby(
        "model", sort=False
    ):
        primary_metrics = model_rows[
            "tuning_primary_metric"
        ].astype(str).unique().tolist()
        selection_policies = model_rows[
            "candidate_selection_policy"
        ].astype(str).unique().tolist()
        if len(primary_metrics) != 1 or len(selection_policies) != 1:
            raise RepeatedHoldoutSummaryError(
                f"Model {model} mixes tuning selection policies "
                "within one experiment"
            )

        denominator = len(model_rows)
        for (alpha, lambda_value), group in model_rows.groupby(
            ["alpha", "lambda"],
            sort=True,
        ):
            frequency_rows.append(
                {
                    "model": model,
                    "tuning_primary_metric": primary_metrics[0],
                    "candidate_selection_policy": selection_policies[0],
                    "alpha": float(alpha),
                    "lambda": float(lambda_value),
                    "selection_count": int(len(group)),
                    "selection_frequency": float(
                        len(group) / denominator
                    ),
                    "n_splits": int(denominator),
                    "mean_inner_roc_auc": float(
                        group["inner_selected_roc_auc"].mean()
                    ),
                    "mean_inner_pr_auc": float(
                        group["inner_selected_pr_auc"].mean()
                    ),
                    "mean_inner_log_loss": float(
                        group["inner_selected_log_loss"].mean()
                    ),
                    "mean_inner_brier_score": float(
                        group["inner_selected_brier_score"].mean()
                    ),
                }
            )
    frequency = pd.DataFrame(frequency_rows)
    if not frequency.empty:
        frequency = frequency.sort_values(
            ["model", "selection_count", "lambda", "alpha"],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)
    threshold = selected.groupby(
        "model", sort=False
    )["threshold"].agg(
        n_splits="count",
        mean="mean",
        sd="std",
        median="median",
        minimum="min",
        maximum="max",
    ).reset_index()
    threshold["sd"] = threshold["sd"].fillna(0.0)
    return selected, frequency, threshold

def _coefficient_stability(
    coefficients: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    coefficient_tolerance: float,
    minimum_selection_frequency: float,
    minimum_sign_consistency: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(coefficients, ("model", "feature_name", "coefficient", "task_id"), "coefficients")
    task_counts = metrics.groupby("model")["task_id"].nunique().to_dict()
    rows: List[Dict[str, Any]] = []
    for (model, feature), group in coefficients.groupby(["model", "feature_name"], sort=False):
        values = pd.to_numeric(group["coefficient"], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise RepeatedHoldoutSummaryError(f"Non-finite coefficients for {model}/{feature}")
        n_total = int(task_counts[str(model)])
        n_available = int(group["task_id"].nunique())
        selected_values = values[np.abs(values) > float(coefficient_tolerance)]
        selection_count = int(len(selected_values))
        positive_count = int(np.sum(selected_values > 0))
        negative_count = int(np.sum(selected_values < 0))
        sign_consistency = float(max(positive_count, negative_count) / selection_count) if selection_count else 0.0
        dominant_sign = "positive" if positive_count > negative_count else "negative" if negative_count > positive_count else "tie_or_none"
        selection_frequency = float(selection_count / n_total)
        rows.append(
            {
                "model": str(model),
                "feature_name": str(feature),
                "n_model_splits": n_total,
                "splits_available": n_available,
                "availability_frequency": float(n_available / n_total),
                "selection_count": selection_count,
                "selection_frequency": selection_frequency,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "sign_consistency": sign_consistency,
                "dominant_sign": dominant_sign,
                "mean_coefficient_all_splits": float(np.sum(values) / n_total),
                "mean_abs_coefficient_all_splits": float(np.sum(np.abs(values)) / n_total),
                "mean_selected_coefficient": float(np.mean(selected_values)) if selection_count else 0.0,
                "stable_selection": bool(
                    feature != "__INTERCEPT__"
                    and selection_frequency >= float(minimum_selection_frequency)
                    and sign_consistency >= float(minimum_sign_consistency)
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["model", "stable_selection", "selection_frequency", "sign_consistency", "mean_abs_coefficient_all_splits"],
            ascending=[True, False, False, False, False],
        ).reset_index(drop=True)
    stable = frame.loc[frame["stable_selection"]].copy() if not frame.empty else frame.copy()
    return frame, stable


def _pairwise_model_differences(metrics: pd.DataFrame) -> pd.DataFrame:
    models = list(dict.fromkeys(metrics["model"].astype(str)))
    indexed = metrics.set_index(["split_id", "model"])
    rows: List[Dict[str, Any]] = []
    for i, model_a in enumerate(models):
        for model_b in models[i + 1 :]:
            common = sorted(
                set(metrics.loc[metrics["model"] == model_a, "split_id"].astype(str))
                & set(metrics.loc[metrics["model"] == model_b, "split_id"].astype(str))
            )
            for metric in METRIC_NAMES:
                a = np.asarray([indexed.loc[(split_id, model_a), metric] for split_id in common], dtype=float)
                b = np.asarray([indexed.loc[(split_id, model_b), metric] for split_id in common], dtype=float)
                difference = b - a
                lower_is_better = metric in LOWER_IS_BETTER_METRICS
                comparison_wins = difference < 0 if lower_is_better else difference > 0
                comparison_losses = difference > 0 if lower_is_better else difference < 0
                rows.append(
                    {
                        "reference_model": model_a,
                        "comparison_model": model_b,
                        "metric": metric,
                        "metric_direction": ("lower_is_better" if lower_is_better else "higher_is_better"),
                        "favorable_difference_sign": ("negative" if lower_is_better else "positive"),
                        "n_paired_splits": int(len(common)),
                        "mean_difference_comparison_minus_reference": float(np.mean(difference)),
                        "sd_difference": float(np.std(difference, ddof=1)) if len(difference) > 1 else 0.0,
                        "median_difference": float(np.median(difference)),
                        "comparison_win_frequency": float(np.mean(comparison_wins)),
                        "tie_frequency": float(np.mean(np.isclose(difference, 0.0, atol=1e-15, rtol=0.0))),
                        "comparison_loss_frequency": float(np.mean(comparison_losses)),
                    }
                )
    return pd.DataFrame(rows)


def _sample_prediction_summary(
    predictions: pd.DataFrame,
    assignments: pd.DataFrame,
    models: Sequence[str],
) -> pd.DataFrame:
    sample_base = assignments.drop_duplicates("sample_id").loc[:, ["sample_id", "patient_id", "cohort"]].copy()
    expected = (
        assignments.loc[assignments["role"].astype(str) == "holdout"]
        .groupby("sample_id").size().rename("expected_holdout_count")
    )
    sample_base = sample_base.merge(expected, on="sample_id", how="left", validate="one_to_one")
    sample_base["expected_holdout_count"] = sample_base["expected_holdout_count"].fillna(0).astype(int)
    rows: List[Dict[str, Any]] = []
    grouped = {(str(model), str(sample)): group for (model, sample), group in predictions.groupby(["model", "sample_id"], sort=False)}
    for model in models:
        for item in sample_base.itertuples(index=False):
            group = grouped.get((str(model), str(item.sample_id)))
            n = 0 if group is None else int(len(group))
            if n != int(item.expected_holdout_count):
                raise RepeatedHoldoutSummaryError(
                    f"Prediction coverage mismatch for {model}/{item.sample_id}: observed={n}, expected={item.expected_holdout_count}"
                )
            probability = np.asarray([], dtype=float) if group is None else group["probability_ILD"].to_numpy(float)
            predicted = np.asarray([], dtype=int) if group is None else group["predicted_label"].to_numpy(int)
            true_value = 1 if str(item.cohort).upper() == "ILD" else 0
            rows.append(
                {
                    "model": str(model),
                    "sample_id": str(item.sample_id),
                    "patient_id": str(item.patient_id),
                    "true_cohort": str(item.cohort),
                    "true_label": true_value,
                    "expected_holdout_count": int(item.expected_holdout_count),
                    "n_predictions": n,
                    "mean_probability_ILD": float(np.mean(probability)) if n else np.nan,
                    "sd_probability_ILD": float(np.std(probability, ddof=1)) if n > 1 else (0.0 if n == 1 else np.nan),
                    "min_probability_ILD": float(np.min(probability)) if n else np.nan,
                    "max_probability_ILD": float(np.max(probability)) if n else np.nan,
                    "positive_classification_frequency": float(np.mean(predicted)) if n else np.nan,
                    "correct_classification_frequency": float(np.mean(predicted == true_value)) if n else np.nan,
                    "not_evaluated_in_any_holdout": bool(n == 0),
                }
            )
    return pd.DataFrame(rows)


def aggregate_repeated_holdout_results(
    tasks: Sequence[Any],
    status: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    expected_models: Sequence[str],
    expected_split_set_id: str,
    expected_split_count: int,
    expected_holdout_size: int,
    coefficient_tolerance: float,
    minimum_selection_frequency: float,
    minimum_sign_consistency: float,
    manifest: Optional[pd.DataFrame] = None,
) -> RepeatedHoldoutAggregate:
    _require_columns(status, ("task_id", "status"), "task status")
    bad = status.loc[status["status"] != "complete"]
    if not bad.empty:
        raise RepeatedHoldoutSummaryError(
            "All repeated-holdout tasks must be complete before aggregation: "
            + ", ".join(bad["task_id"].astype(str).tolist())
        )
    if len(tasks) != int(expected_split_count):
        raise RepeatedHoldoutSummaryError("Executed task count does not match split count")
    task_lookup = {str(task.task_id): task for task in tasks}
    complete_ids = status["task_id"].astype(str).tolist()
    if set(complete_ids) != set(task_lookup):
        raise RepeatedHoldoutSummaryError("Task status IDs do not match executed tasks")

    metric_parts: List[pd.DataFrame] = []
    prediction_parts: List[pd.DataFrame] = []
    coefficient_parts: List[pd.DataFrame] = []
    membership_rows: List[Dict[str, Any]] = []
    models = tuple(map(str, expected_models))

    for task_id in complete_ids:
        task = task_lookup[task_id]
        repeat = int(task.outer_repeat)
        paths = _task_paths(task)
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise RepeatedHoldoutSummaryError(f"Task {task_id} is missing files: {missing}")
        marker = _json(paths["complete"], f"{task_id} completion marker")
        configuration = _json(paths["configuration"], f"{task_id} configuration")
        if marker.get("status") != "COMPLETE":
            raise RepeatedHoldoutSummaryError(f"Task {task_id} completion marker is not COMPLETE")
        configured_models = tuple(map(str, configuration.get("models", [])))
        if configured_models != models:
            raise RepeatedHoldoutSummaryError(f"Task {task_id} model order mismatch")

        assignment = assignments.loc[assignments["repeat_index"] == repeat].copy()
        if assignment.empty:
            raise RepeatedHoldoutSummaryError(f"No frozen assignment for repeat {repeat}")
        split_ids = assignment["split_id"].astype(str).unique().tolist()
        if len(split_ids) != 1:
            raise RepeatedHoldoutSummaryError(f"Repeat {repeat} has multiple split IDs")
        split_id = split_ids[0]
        expected_holdout = set(assignment.loc[assignment["role"].astype(str) == "holdout", "sample_id"].astype(str))
        if len(expected_holdout) != int(expected_holdout_size):
            raise RepeatedHoldoutSummaryError(f"Split {split_id} holdout size is invalid")

        roles = pd.read_csv(paths["sample_roles"])
        _require_columns(roles, ("sample_id", "outer_role"), f"{task_id} sample roles")
        observed_role_holdout = set(roles.loc[roles["outer_role"].astype(str) == "validation", "sample_id"].astype(str))
        if observed_role_holdout != expected_holdout:
            raise RepeatedHoldoutSummaryError(f"Task {task_id} validation roles do not match frozen holdout")

        metrics = pd.read_csv(paths["outer_metrics"])
        _require_columns(metrics, ("model", *METRIC_NAMES, "threshold", "selected_l1_ratio_alpha", "selected_lambda", "inner_selected_roc_auc", "inner_selected_pr_auc"), f"{task_id} metrics")
        if "outer_repeat" not in metrics.columns:
            metrics["outer_repeat"] = int(task.outer_repeat)
        if "outer_fold" not in metrics.columns:
            metrics["outer_fold"] = int(task.outer_fold)
        if tuple(metrics["model"].astype(str)) != models or len(metrics) != len(models):
            raise RepeatedHoldoutSummaryError(f"Task {task_id} metric model rows/order are invalid")
        metrics.insert(0, "split_set_id", str(expected_split_set_id))
        metrics.insert(1, "split_id", split_id)
        metrics["task_id"] = task_id
        metrics["task_index"] = int(task.task_index)
        metric_parts.append(metrics)

        predictions = pd.read_csv(paths["outer_predictions"])
        _require_columns(predictions, ("model", "sample_id", "true_label", "true_cohort", "probability_ILD", "threshold", "predicted_label"), f"{task_id} predictions")
        predictions["sample_id"] = predictions["sample_id"].astype(str)
        if predictions.duplicated(["model", "sample_id"]).any():
            raise RepeatedHoldoutSummaryError(f"Task {task_id} has duplicate model/sample predictions")
        for model in models:
            model_ids = set(predictions.loc[predictions["model"].astype(str) == model, "sample_id"])
            if model_ids != expected_holdout or len(model_ids) != int(expected_holdout_size):
                raise RepeatedHoldoutSummaryError(f"Task {task_id}/{model} prediction membership mismatch")
            membership_rows.append(
                {
                    "split_set_id": str(expected_split_set_id),
                    "split_id": split_id,
                    "task_id": task_id,
                    "model": model,
                    "expected_holdout_samples": int(expected_holdout_size),
                    "observed_prediction_samples": int(len(model_ids)),
                    "membership_exact_match": True,
                    "holdout_sample_id_sha256": hashlib.sha256(("\n".join(sorted(model_ids)) + "\n").encode("utf-8")).hexdigest(),
                }
            )
        predictions.insert(0, "split_set_id", str(expected_split_set_id))
        predictions.insert(1, "split_id", split_id)
        predictions["task_id"] = task_id
        predictions["task_index"] = int(task.task_index)
        prediction_parts.append(predictions)

        coefficients = pd.read_csv(paths["coefficients"])
        _require_columns(coefficients, ("model", "feature_name", "coefficient"), f"{task_id} coefficients")
        coefficients.insert(0, "split_set_id", str(expected_split_set_id))
        coefficients.insert(1, "split_id", split_id)
        coefficients["task_id"] = task_id
        coefficients["task_index"] = int(task.task_index)
        coefficient_parts.append(coefficients)

    metrics = pd.concat(metric_parts, ignore_index=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    coefficients = pd.concat(coefficient_parts, ignore_index=True)
    if metrics.duplicated(["split_id", "model"]).any():
        raise RepeatedHoldoutSummaryError("Aggregated metrics contain duplicate split/model rows")
    expected_metric_rows = int(expected_split_count) * len(models)
    expected_prediction_rows = int(expected_split_count) * len(models) * int(expected_holdout_size)
    if len(metrics) != expected_metric_rows or len(predictions) != expected_prediction_rows:
        raise RepeatedHoldoutSummaryError(
            f"Aggregate row counts are invalid: metrics={len(metrics)}/{expected_metric_rows}, predictions={len(predictions)}/{expected_prediction_rows}"
        )

    metric_summary = _metric_summary(metrics)
    ranking = _model_ranking(metric_summary)
    selected, hyper_frequency, threshold_summary = _hyperparameter_tables(metrics)
    feature_stability, stable_features = _coefficient_stability(
        coefficients,
        metrics,
        coefficient_tolerance=coefficient_tolerance,
        minimum_selection_frequency=minimum_selection_frequency,
        minimum_sign_consistency=minimum_sign_consistency,
    )
    pairwise = _pairwise_model_differences(metrics)
    sample_summary = _sample_prediction_summary(predictions, assignments, models)
    return RepeatedHoldoutAggregate(
        manifest=manifest.copy() if manifest is not None else pd.DataFrame(),
        status=status.copy(),
        metrics=metrics,
        predictions=predictions,
        selected_hyperparameters=selected,
        hyperparameter_frequency=hyper_frequency,
        threshold_summary=threshold_summary,
        metric_summary=metric_summary,
        model_ranking=ranking,
        coefficients=coefficients,
        feature_stability=feature_stability,
        stable_features=stable_features,
        pairwise_model_differences=pairwise,
        sample_prediction_summary=sample_summary,
        membership_audit=pd.DataFrame(membership_rows),
    )


def write_repeated_holdout_aggregate(
    aggregate: RepeatedHoldoutAggregate,
    output_dir: Path,
    *,
    experiment_id: str,
    split_set_id: str,
    assignment_sha256: str,
    split_marker_sha256: str,
    expected_split_count: int,
    expected_holdout_size: int,
    overwrite: bool = False,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    paths = {key: output_dir / name for key, name in OUTPUT_FILE_NAMES.items()}
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Repeated-holdout aggregation outputs already exist; use --overwrite only for a documented rerun:\n"
            + "\n".join(f"  - {path}" for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate.manifest.to_csv(paths["manifest"], index=False)
    aggregate.status.to_csv(paths["status"], index=False)
    aggregate.metrics.to_csv(paths["metrics"], index=False)
    aggregate.predictions.to_csv(paths["predictions"], index=False, compression="gzip")
    aggregate.selected_hyperparameters.to_csv(paths["hyperparameters"], index=False)
    aggregate.hyperparameter_frequency.to_csv(paths["hyperparameter_frequency"], index=False)
    aggregate.threshold_summary.to_csv(paths["threshold_summary"], index=False)
    aggregate.metric_summary.to_csv(paths["metric_summary"], index=False)
    aggregate.model_ranking.to_csv(paths["model_ranking"], index=False)
    aggregate.coefficients.to_csv(paths["coefficients"], index=False, compression="gzip")
    aggregate.feature_stability.to_csv(paths["feature_stability"], index=False, compression="gzip")
    aggregate.stable_features.to_csv(paths["stable_features"], index=False)
    aggregate.pairwise_model_differences.to_csv(paths["pairwise_model_differences"], index=False)
    aggregate.sample_prediction_summary.to_csv(paths["sample_prediction_summary"], index=False)
    aggregate.membership_audit.to_csv(paths["membership_audit"], index=False)

    models = list(dict.fromkeys(aggregate.metrics["model"].astype(str)))
    summary = {
        "experiment_id": str(experiment_id),
        "analysis_mode": "frozen_repeated_holdout",
        "split_set_id": str(split_set_id),
        "assignment_sha256": str(assignment_sha256),
        "split_marker_sha256": str(split_marker_sha256),
        "expected_split_count": int(expected_split_count),
        "completed_split_count": int(aggregate.status["status"].eq("complete").sum()),
        "holdout_size_per_split": int(expected_holdout_size),
        "models": models,
        "metric_rows": int(len(aggregate.metrics)),
        "prediction_rows": int(len(aggregate.predictions)),
        "stable_feature_rows": int(len(aggregate.stable_features)),
        "ranking_is_descriptive_only": True,
        "automatic_final_model_selection": False,
        "independent_test_read": False,
    }
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_hashes = {
        key: {"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)}
        for key, path in paths.items() if key not in {"complete"}
    }
    marker = {"status": "COMPLETE", **summary, "files": output_hashes}
    paths["complete"].write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return paths


def compare_aggregated_schemes(
    aggregation_dirs: Sequence[Path],
) -> Tuple[pd.DataFrame, pd.DataFrame, Mapping[str, Any]]:
    if not aggregation_dirs:
        raise RepeatedHoldoutSummaryError("At least one aggregation directory is required")
    records: List[Dict[str, Any]] = []
    metric_frames: List[pd.DataFrame] = []
    assignment_hash: Optional[str] = None
    split_set_id: Optional[str] = None
    for directory in aggregation_dirs:
        directory = Path(directory).resolve()
        marker_path = directory / OUTPUT_FILE_NAMES["complete"]
        metrics_path = directory / OUTPUT_FILE_NAMES["metrics"]
        marker = _json(marker_path, f"aggregation marker {directory}")
        if marker.get("status") != "COMPLETE":
            raise RepeatedHoldoutSummaryError(f"Aggregation is not COMPLETE: {directory}")
        current_hash = str(marker.get("assignment_sha256", ""))
        current_split_set = str(marker.get("split_set_id", ""))
        if assignment_hash is None:
            assignment_hash = current_hash
            split_set_id = current_split_set
        elif current_hash != assignment_hash or current_split_set != split_set_id:
            raise RepeatedHoldoutSummaryError("Schemes do not use the same frozen split assignments")
        scheme_id = str(marker.get("experiment_id"))
        metrics = pd.read_csv(metrics_path)
        metrics.insert(0, "scheme_id", scheme_id)
        metric_frames.append(metrics)
        records.append(
            {
                "scheme_id": scheme_id,
                "aggregation_dir": str(directory),
                "split_set_id": current_split_set,
                "assignment_sha256": current_hash,
                "models": ";".join(map(str, marker.get("models", []))),
                "completed_split_count": int(marker.get("completed_split_count", 0)),
                "metric_rows": int(marker.get("metric_rows", 0)),
                "prediction_rows": int(marker.get("prediction_rows", 0)),
            }
        )
    all_metrics = pd.concat(metric_frames, ignore_index=True)
    summary_rows: List[Dict[str, Any]] = []
    for (scheme, model), group in all_metrics.groupby(["scheme_id", "model"], sort=False):
        for metric in METRIC_NAMES:
            values = group[metric].to_numpy(float)
            summary_rows.append(
                {
                    "scheme_id": scheme,
                    "model": model,
                    "metric": metric,
                    "n_splits": int(len(values)),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    summary = pd.DataFrame(summary_rows)

    pair_rows: List[Dict[str, Any]] = []
    schemes = list(dict.fromkeys(all_metrics["scheme_id"].astype(str)))
    for i, scheme_a in enumerate(schemes):
        for scheme_b in schemes[i + 1 :]:
            a_rows = all_metrics.loc[all_metrics["scheme_id"] == scheme_a]
            b_rows = all_metrics.loc[all_metrics["scheme_id"] == scheme_b]
            common_models = sorted(set(a_rows["model"].astype(str)) & set(b_rows["model"].astype(str)))
            for model in common_models:
                a_model = a_rows.loc[a_rows["model"].astype(str) == model].set_index("split_id")
                b_model = b_rows.loc[b_rows["model"].astype(str) == model].set_index("split_id")
                common_splits = sorted(set(a_model.index.astype(str)) & set(b_model.index.astype(str)))
                for metric in METRIC_NAMES:
                    a = np.asarray([a_model.loc[x, metric] for x in common_splits], dtype=float)
                    b = np.asarray([b_model.loc[x, metric] for x in common_splits], dtype=float)
                    difference = b - a
                    lower_is_better = metric in LOWER_IS_BETTER_METRICS
                    comparison_wins = difference < 0 if lower_is_better else difference > 0
                    comparison_losses = difference > 0 if lower_is_better else difference < 0
                    pair_rows.append(
                        {
                            "reference_scheme": scheme_a,
                            "comparison_scheme": scheme_b,
                            "model": model,
                            "metric": metric,
                            "metric_direction": ("lower_is_better" if lower_is_better else "higher_is_better"),
                            "favorable_difference_sign": ("negative" if lower_is_better else "positive"),
                            "n_paired_splits": int(len(common_splits)),
                            "mean_difference_comparison_minus_reference": float(np.mean(difference)),
                            "sd_difference": float(np.std(difference, ddof=1)) if len(difference) > 1 else 0.0,
                            "median_difference": float(np.median(difference)),
                            "comparison_win_frequency": float(np.mean(comparison_wins)),
                            "tie_frequency": float(np.mean(np.isclose(difference, 0.0, atol=1e-15, rtol=0.0))),
                            "comparison_loss_frequency": float(np.mean(comparison_losses)),
                        }
                    )
    audit = {
        "status": "PASS",
        "scheme_count": int(len(records)),
        "split_set_id": split_set_id,
        "assignment_sha256": assignment_hash,
        "same_frozen_assignments": True,
        "scheme_registry": records,
        "paired_comparisons_are_descriptive_only": True,
        "automatic_winner_selection": False,
    }
    return summary, pd.DataFrame(pair_rows), audit
