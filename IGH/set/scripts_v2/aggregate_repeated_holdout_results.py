#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate IGH repeated-holdout nested-CV outputs.

This aggregator is intentionally separate from the legacy repeated-K-fold
aggregator. In a repeated-holdout design, every outer repeat contains one
executed validation fold rather than pooled out-of-fold predictions for every
sample. Therefore:

* one outer-task metric row is already one repeat-level metric row;
* each patient is predicted only when assigned to holdout;
* prediction coverage is checked against the observed holdout assignments,
  not against the total repeat count;
* paired model differences are descriptive across repeated resamples and no
  independence-based p-values are calculated;
* no final model is selected automatically.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "PyYAML is required: mamba install -c conda-forge pyyaml"
    ) from exc


TASK_FILES: Mapping[str, str] = {
    "configuration": "06_task_configuration.json",
    "sample_roles": "06_task_sample_roles.csv",
    "inner_tuning": "06_inner_tuning_results.csv",
    "inner_oof": "06_inner_selected_oof_predictions.csv",
    "outer_predictions": "06_outer_validation_predictions.csv",
    "outer_metrics": "06_outer_validation_metrics.csv",
    "coefficients": "06_final_model_coefficients.csv",
    "repeat_3mer_vocabulary": "06_repeat_3mer_vocabulary.csv",
    "complete": "06_TASK_COMPLETE.json",
}

OUTPUT_FILES: Mapping[str, str] = {
    "task_status": "07_repeated_holdout_task_status.csv",
    "all_metrics": "07_all_outer_metrics.csv",
    "all_predictions": "07_all_outer_predictions.csv.gz",
    "all_tuning": "07_all_inner_tuning_results.csv.gz",
    "all_oof": "07_all_inner_selected_oof_predictions.csv.gz",
    "selected_hyperparameters": "07_all_selected_hyperparameters.csv",
    "all_coefficients": "07_all_coefficients.csv.gz",
    "all_vocabularies": "07_all_repeat_3mer_vocabulary.csv.gz",
    "model_catalog": "07_model_catalog.csv",
    "metric_summary_long": "07_model_metric_summary_long.csv",
    "metric_summary_wide": "07_model_metric_summary_wide.csv",
    "repeat_metrics": "07_repeat_holdout_metrics.csv",
    "baseline_differences": "07_baseline_paired_differences.csv.gz",
    "baseline_difference_summary": "07_baseline_paired_difference_summary.csv",
    "pairwise_differences": "07_pairwise_model_metric_differences.csv",
    "rank_detail": "07_model_rank_by_repeat.csv.gz",
    "rank_summary": "07_model_rank_summary.csv",
    "hyperparameter_frequency": "07_hyperparameter_selection_frequency.csv",
    "alpha_frequency": "07_alpha_selection_frequency.csv",
    "lambda_frequency": "07_lambda_selection_frequency.csv",
    "regularization_family_frequency": "07_regularization_family_frequency.csv",
    "inner_convergence": "07_inner_tuning_convergence_summary.csv",
    "outer_convergence": "07_outer_refit_convergence_summary.csv",
    "feature_stability": "07_feature_stability.csv.gz",
    "stable_features": "07_stable_features.csv",
    "intercept_stability": "07_intercept_stability.csv",
    "sample_holdout_counts": "07_sample_holdout_count_audit.csv",
    "sample_prediction_stability": "07_sample_holdout_prediction_stability.csv",
    "report": "07_summary_report.md",
    "summary": "07_aggregation_summary.json",
    "complete": "07_REPEATED_HOLDOUT_AGGREGATION_COMPLETE.json",
}

PERFORMANCE_METRICS: Tuple[str, ...] = (
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

PAIRED_METRICS: Tuple[str, ...] = (
    "roc_auc",
    "pr_auc",
    "log_loss",
    "brier_score",
    "sensitivity_recall",
    "specificity",
    "f1",
)

LOWER_IS_BETTER_METRICS = frozenset({"log_loss", "brier_score"})


class AggregationError(ValueError):
    """Raised when repeated-holdout outputs violate the expected contract."""


@dataclass(frozen=True)
class TaskSpec:
    task_index: int
    outer_repeat: int
    outer_fold: int
    task_id: str
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate completed IGH repeated-holdout nested-CV tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--baseline-model", default="A1_core83")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> MutableMapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, MutableMapping):
        raise AggregationError(f"YAML root must be a mapping: {path}")
    return value


def resolve_path(value: object, repository_root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AggregationError(f"{label} must be a non-empty path string.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AggregationError(f"{label} must be a mapping.")
    return value


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise AggregationError(f"{label} is missing columns: {missing}")


def read_json(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AggregationError(f"Unable to read {label}: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise AggregationError(f"{label} must contain a JSON object: {path}")
    return value


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    frame = pd.read_csv(path)
    duplicated = frame.columns[frame.columns.duplicated()].tolist()
    if duplicated:
        raise AggregationError(f"{label} has duplicate columns: {duplicated}")
    return frame


def to_bool(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0"}
    bad = sorted(set(normalized) - allowed)
    if bad:
        raise AggregationError(f"{label} contains non-boolean values: {bad[:10]}")
    return normalized.isin({"true", "1"})


def build_tasks(config: Mapping[str, object], output_root: Path) -> Tuple[TaskSpec, ...]:
    cv = require_mapping(config.get("cross_validation"), "cross_validation")
    outer = require_mapping(config.get("outer_tasks"), "outer_tasks")
    repeats = int(cv["outer_repeats"])
    outer_folds = int(cv["outer_folds"])
    executed = tuple(int(x) for x in cv.get("executed_outer_folds", range(1, outer_folds + 1)))
    if repeats < 1 or outer_folds < 2:
        raise AggregationError("outer_repeats must be >=1 and outer_folds must be >=2.")
    if len(executed) != 1:
        raise AggregationError(
            "This script is for repeated holdout and requires exactly one "
            f"executed outer fold; observed={executed}."
        )
    if not set(executed).issubset(set(range(1, outer_folds + 1))):
        raise AggregationError("executed_outer_folds is outside the configured fold range.")
    expected_tasks = int(outer["expected_tasks"])
    tasks: List[TaskSpec] = []
    index = 0
    for repeat in range(1, repeats + 1):
        for fold in executed:
            index += 1
            task_id = f"repeat_{repeat:02d}_fold_{fold:02d}"
            tasks.append(
                TaskSpec(
                    task_index=index,
                    outer_repeat=repeat,
                    outer_fold=fold,
                    task_id=task_id,
                    output_dir=output_root / task_id,
                )
            )
    if len(tasks) != expected_tasks:
        raise AggregationError(
            f"Generated repeated-holdout tasks={len(tasks)}, expected={expected_tasks}."
        )
    return tuple(tasks)


def task_paths(task: TaskSpec) -> Dict[str, Path]:
    return {name: task.output_dir / filename for name, filename in TASK_FILES.items()}


def inspect_and_load_task(
    task: TaskSpec,
    *,
    expected_models: Sequence[str],
    expected_samples: int,
    expected_train: int,
    expected_holdout: int,
    expected_candidates: int,
    inner_folds: int,
) -> Tuple[Dict[str, object], Dict[str, pd.DataFrame]]:
    paths = task_paths(task)
    required = [
        "configuration",
        "sample_roles",
        "inner_tuning",
        "inner_oof",
        "outer_predictions",
        "outer_metrics",
        "coefficients",
        "complete",
    ]
    missing = [name for name in required if not paths[name].is_file()]
    if missing:
        raise AggregationError(f"{task.task_id} missing required files: {missing}")

    marker = read_json(paths["complete"], f"{task.task_id} completion marker")
    configuration = read_json(paths["configuration"], f"{task.task_id} configuration")
    if marker.get("status") != "COMPLETE":
        raise AggregationError(f"{task.task_id} completion status is not COMPLETE.")
    for source, label in ((marker, "marker"), (configuration, "configuration")):
        if int(source.get("outer_repeat", -1)) != task.outer_repeat:
            raise AggregationError(f"{task.task_id} {label} outer_repeat mismatch.")
        if int(source.get("outer_fold", -1)) != task.outer_fold:
            raise AggregationError(f"{task.task_id} {label} outer_fold mismatch.")

    n_train = int(configuration["outer_train_samples"])
    n_holdout = int(configuration["outer_validation_samples"])
    if (n_train, n_holdout) != (expected_train, expected_holdout):
        raise AggregationError(
            f"{task.task_id} train/holdout={n_train}/{n_holdout}, "
            f"expected={expected_train}/{expected_holdout}."
        )
    if n_train + n_holdout != expected_samples:
        raise AggregationError(f"{task.task_id} total sample count mismatch.")
    configured_models = tuple(str(x) for x in configuration.get("models", []))
    if configured_models != tuple(expected_models):
        raise AggregationError(
            f"{task.task_id} model order={configured_models}, expected={tuple(expected_models)}."
        )
    if int(configuration.get("candidate_count_per_model", -1)) != expected_candidates:
        raise AggregationError(f"{task.task_id} candidate count mismatch.")

    roles = read_csv(paths["sample_roles"], f"{task.task_id} sample roles")
    require_columns(roles, ("sample_id", "cohort", "outer_role"), "sample roles")
    if len(roles) != expected_samples or roles["sample_id"].astype(str).duplicated().any():
        raise AggregationError(f"{task.task_id} sample-role rows/uniqueness are invalid.")
    role_counts = roles["outer_role"].astype(str).value_counts().to_dict()
    if role_counts.get("training", 0) != expected_train:
        raise AggregationError(f"{task.task_id} training role count mismatch.")
    if role_counts.get("validation", 0) != expected_holdout:
        raise AggregationError(f"{task.task_id} validation role count mismatch.")

    metrics = read_csv(paths["outer_metrics"], f"{task.task_id} outer metrics")
    require_columns(
        metrics,
        (
            "model",
            "outer_repeat",
            "outer_fold",
            "selected_l1_ratio_alpha",
            "selected_lambda",
            "inner_selected_roc_auc",
            "inner_selected_pr_auc",
            "fit_converged",
            "iterations_used",
            *PERFORMANCE_METRICS,
        ),
        "outer metrics",
    )
    if len(metrics) != len(expected_models):
        raise AggregationError(f"{task.task_id} outer metric row count mismatch.")
    if tuple(metrics["model"].astype(str)) != tuple(expected_models):
        raise AggregationError(f"{task.task_id} outer metric model order mismatch.")
    if not to_bool(metrics["fit_converged"], "fit_converged").all():
        raise AggregationError(f"{task.task_id} contains a non-converged outer refit.")

    predictions = read_csv(
        paths["outer_predictions"], f"{task.task_id} outer predictions"
    )
    require_columns(
        predictions,
        (
            "model",
            "sample_id",
            "true_label",
            "true_cohort",
            "probability_ILD",
            "threshold",
            "predicted_label",
        ),
        "outer predictions",
    )
    if len(predictions) != len(expected_models) * expected_holdout:
        raise AggregationError(f"{task.task_id} outer prediction row count mismatch.")
    if predictions.duplicated(["model", "sample_id"]).any():
        raise AggregationError(f"{task.task_id} duplicate model/sample predictions.")

    tuning = read_csv(paths["inner_tuning"], f"{task.task_id} inner tuning")
    require_columns(
        tuning,
        (
            "model",
            "l1_ratio_alpha",
            "lambda",
            "all_fits_converged",
            "max_iterations_used",
        ),
        "inner tuning",
    )
    if len(tuning) != len(expected_models) * expected_candidates:
        raise AggregationError(f"{task.task_id} inner tuning row count mismatch.")
    if tuning.duplicated(["model", "l1_ratio_alpha", "lambda"]).any():
        raise AggregationError(f"{task.task_id} duplicate tuning candidates.")
    if not to_bool(tuning["all_fits_converged"], "all_fits_converged").all():
        raise AggregationError(f"{task.task_id} contains non-converged inner fits.")

    oof = read_csv(paths["inner_oof"], f"{task.task_id} selected inner OOF")
    require_columns(oof, ("model", "sample_id", "inner_fold"), "selected inner OOF")
    if len(oof) != len(expected_models) * expected_train:
        raise AggregationError(f"{task.task_id} selected inner OOF row count mismatch.")
    if oof.duplicated(["model", "sample_id"]).any():
        raise AggregationError(f"{task.task_id} duplicate selected OOF predictions.")
    if oof["inner_fold"].nunique() != inner_folds:
        raise AggregationError(f"{task.task_id} inner-fold coverage mismatch.")

    coefficients = read_csv(paths["coefficients"], f"{task.task_id} coefficients")
    require_columns(
        coefficients,
        ("model", "feature_name", "coefficient", "nonzero"),
        "coefficients",
    )
    if coefficients.duplicated(["model", "feature_name"]).any():
        raise AggregationError(f"{task.task_id} duplicate model/feature coefficients.")
    if set(coefficients["model"].astype(str)) != set(expected_models):
        raise AggregationError(f"{task.task_id} coefficient models are incomplete.")

    frames = {
        "roles": roles,
        "metrics": metrics,
        "predictions": predictions,
        "tuning": tuning,
        "oof": oof,
        "coefficients": coefficients,
    }
    if paths["repeat_3mer_vocabulary"].is_file():
        frames["vocabulary"] = read_csv(
            paths["repeat_3mer_vocabulary"], f"{task.task_id} 3-mer vocabulary"
        )

    meta = {
        "task_index": task.task_index,
        "task_id": task.task_id,
        "outer_repeat": task.outer_repeat,
        "outer_fold": task.outer_fold,
        "output_dir": str(task.output_dir),
        "status": "complete",
        "n_outer_train": n_train,
        "n_outer_validation": n_holdout,
        "runtime_seconds": float(
            marker.get("runtime_seconds", configuration.get("runtime_seconds", np.nan))
        ),
    }
    return meta, frames


def annotate(frame: pd.DataFrame, task: TaskSpec) -> pd.DataFrame:
    output = frame.copy()
    output["outer_repeat"] = task.outer_repeat
    output["outer_fold"] = task.outer_fold
    output["task_id"] = task.task_id
    output["task_index"] = task.task_index
    return output


def numeric_summary(values: np.ndarray) -> Dict[str, float]:
    x = np.asarray(values, dtype=float)
    if x.size == 0 or not np.isfinite(x).all():
        raise AggregationError("Summary values must be non-empty and finite.")
    return {
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "median": float(np.median(x)),
        "q025": float(np.quantile(x, 0.025)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "q975": float(np.quantile(x, 0.975)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def metric_summaries(
    metrics: pd.DataFrame, model_order: Sequence[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    for model in model_order:
        group = metrics.loc[metrics["model"].astype(str) == model]
        for metric in PERFORMANCE_METRICS:
            values = pd.to_numeric(group[metric], errors="raise").to_numpy(float)
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "n_repeats": int(len(values)),
                    **numeric_summary(values),
                }
            )
    long = pd.DataFrame(rows)
    wide = long.pivot(index="model", columns="metric")
    wide.columns = [f"{metric}_{stat}" for stat, metric in wide.columns]
    wide = wide.reset_index()
    order_lookup = {model: i for i, model in enumerate(model_order)}
    wide["_order"] = wide["model"].map(order_lookup)
    wide = wide.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return long, wide


def selected_hyperparameters(
    metrics: pd.DataFrame,
    model_order: Sequence[str],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
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

    selected = normalized.loc[
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
            "inner_selected_log_loss",
            "inner_selected_brier_score",
            "tuning_primary_metric",
            "candidate_selection_policy",
        ],
    ].copy()
    selected = selected.rename(
        columns={
            "selected_l1_ratio_alpha": "alpha",
            "selected_lambda": "lambda",
        }
    )

    def family(alpha: float) -> str:
        if np.isclose(alpha, 0.0, atol=1e-15, rtol=0.0):
            return "Ridge_alpha0"
        if np.isclose(alpha, 1.0, atol=1e-15, rtol=0.0):
            return "Lasso_alpha1"
        return "ElasticNet_interior"

    selected["regularization_family"] = (
        selected["alpha"].astype(float).map(family)
    )
    model_order_lookup = {
        model: i for i, model in enumerate(model_order)
    }

    pair_rows: List[Dict[str, object]] = []
    alpha_rows: List[Dict[str, object]] = []
    lambda_rows: List[Dict[str, object]] = []
    family_rows: List[Dict[str, object]] = []
    for model in model_order:
        group = selected.loc[
            selected["model"].astype(str) == model
        ]
        primary_metrics = group[
            "tuning_primary_metric"
        ].astype(str).unique().tolist()
        selection_policies = group[
            "candidate_selection_policy"
        ].astype(str).unique().tolist()
        if len(primary_metrics) != 1 or len(selection_policies) != 1:
            raise AggregationError(
                f"Model {model} mixes tuning selection policies "
                "within one experiment."
            )
        denominator = len(group)
        for (alpha, lambda_value), item in group.groupby(
            ["alpha", "lambda"], sort=True
        ):
            pair_rows.append(
                {
                    "model": model,
                    "tuning_primary_metric": primary_metrics[0],
                    "candidate_selection_policy": selection_policies[0],
                    "alpha": float(alpha),
                    "lambda": float(lambda_value),
                    "selection_count": int(len(item)),
                    "selection_frequency": float(
                        len(item) / denominator
                    ),
                    "n_model_repeats": int(denominator),
                    "mean_inner_roc_auc": float(
                        item["inner_selected_roc_auc"].mean()
                    ),
                    "mean_inner_pr_auc": float(
                        item["inner_selected_pr_auc"].mean()
                    ),
                    "mean_inner_log_loss": float(
                        item["inner_selected_log_loss"].mean()
                    ),
                    "mean_inner_brier_score": float(
                        item["inner_selected_brier_score"].mean()
                    ),
                }
            )
        for alpha, item in group.groupby("alpha", sort=True):
            alpha_rows.append(
                {
                    "model": model,
                    "alpha": float(alpha),
                    "selection_count": int(len(item)),
                    "selection_frequency": float(
                        len(item) / denominator
                    ),
                    "n_model_repeats": int(denominator),
                }
            )
        for lambda_value, item in group.groupby(
            "lambda", sort=True
        ):
            lambda_rows.append(
                {
                    "model": model,
                    "lambda": float(lambda_value),
                    "selection_count": int(len(item)),
                    "selection_frequency": float(
                        len(item) / denominator
                    ),
                    "n_model_repeats": int(denominator),
                }
            )
        for family_name, item in group.groupby(
            "regularization_family", sort=False
        ):
            family_rows.append(
                {
                    "model": model,
                    "regularization_family": family_name,
                    "selection_count": int(len(item)),
                    "selection_frequency": float(
                        len(item) / denominator
                    ),
                    "n_model_repeats": int(denominator),
                }
            )

    def ordered(
        frame: pd.DataFrame,
        extra: Sequence[str],
    ) -> pd.DataFrame:
        frame["_model_order"] = frame["model"].map(
            model_order_lookup
        )
        return (
            frame.sort_values(["_model_order", *extra])
            .drop(columns="_model_order")
            .reset_index(drop=True)
        )

    return (
        selected,
        ordered(
            pd.DataFrame(pair_rows),
            ["selection_count", "lambda", "alpha"],
        ).sort_values(
            ["model", "selection_count", "lambda", "alpha"],
            ascending=[True, False, False, False],
        ).reset_index(drop=True),
        ordered(pd.DataFrame(alpha_rows), ["alpha"]),
        ordered(pd.DataFrame(lambda_rows), ["lambda"]),
        ordered(
            pd.DataFrame(family_rows),
            ["regularization_family"],
        ),
    )

def paired_differences(
    metrics: pd.DataFrame,
    *,
    baseline_model: str,
    model_order: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if baseline_model not in model_order:
        raise AggregationError(f"Baseline model not found: {baseline_model}")
    indexed = metrics.set_index(["outer_repeat", "model"])
    detail_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    repeats = sorted(metrics["outer_repeat"].astype(int).unique())
    for comparison in model_order:
        if comparison == baseline_model:
            continue
        for metric in PAIRED_METRICS:
            lower_is_better = metric in LOWER_IS_BETTER_METRICS
            differences = []
            for repeat in repeats:
                baseline = float(indexed.loc[(repeat, baseline_model), metric])
                value = float(indexed.loc[(repeat, comparison), metric])
                difference = value - baseline
                differences.append(difference)
                detail_rows.append(
                    {
                        "outer_repeat": int(repeat),
                        "baseline_model": baseline_model,
                        "comparison_model": comparison,
                        "metric": metric,
                        "metric_direction": ("lower_is_better" if lower_is_better else "higher_is_better"),
                        "favorable_difference_sign": ("negative" if lower_is_better else "positive"),
                        "baseline_value": baseline,
                        "comparison_value": value,
                        "difference_comparison_minus_baseline": difference,
                    }
                )
            x = np.asarray(differences, dtype=float)
            ties = np.isclose(x, 0.0, atol=1e-15, rtol=0.0)
            comparison_wins = x < 0 if lower_is_better else x > 0
            comparison_losses = x > 0 if lower_is_better else x < 0
            summary_rows.append(
                {
                    "baseline_model": baseline_model,
                    "comparison_model": comparison,
                    "metric": metric,
                    "metric_direction": ("lower_is_better" if lower_is_better else "higher_is_better"),
                    "favorable_difference_sign": ("negative" if lower_is_better else "positive"),
                    "n_paired_repeats": int(len(x)),
                    **numeric_summary(x),
                    "comparison_win_count": int(np.sum(comparison_wins)),
                    "tie_count": int(np.sum(ties)),
                    "comparison_loss_count": int(np.sum(comparison_losses)),
                    "comparison_win_frequency": float(np.mean(comparison_wins)),
                    "tie_frequency": float(np.mean(ties)),
                    "comparison_loss_frequency": float(np.mean(comparison_losses)),
                }
            )
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def all_pairwise_differences(
    metrics: pd.DataFrame, model_order: Sequence[str]
) -> pd.DataFrame:
    indexed = metrics.set_index(["outer_repeat", "model"])
    repeats = sorted(metrics["outer_repeat"].astype(int).unique())
    rows: List[Dict[str, object]] = []
    for i, reference in enumerate(model_order):
        for comparison in model_order[i + 1 :]:
            for metric in PAIRED_METRICS:
                differences = np.asarray(
                    [
                        float(indexed.loc[(repeat, comparison), metric])
                        - float(indexed.loc[(repeat, reference), metric])
                        for repeat in repeats
                    ],
                    dtype=float,
                )
                ties = np.isclose(differences, 0.0, atol=1e-15, rtol=0.0)
                lower_is_better = metric in LOWER_IS_BETTER_METRICS
                comparison_wins = differences < 0 if lower_is_better else differences > 0
                comparison_losses = differences > 0 if lower_is_better else differences < 0
                rows.append(
                    {
                        "reference_model": reference,
                        "comparison_model": comparison,
                        "metric": metric,
                        "metric_direction": ("lower_is_better" if lower_is_better else "higher_is_better"),
                        "favorable_difference_sign": ("negative" if lower_is_better else "positive"),
                        "n_paired_repeats": int(len(differences)),
                        **numeric_summary(differences),
                        "comparison_win_frequency": float(np.mean(comparison_wins)),
                        "tie_frequency": float(np.mean(ties)),
                        "comparison_loss_frequency": float(np.mean(comparison_losses)),
                    }
                )
    return pd.DataFrame(rows)


def model_ranks(
    metrics: pd.DataFrame, model_order: Sequence[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    detail_parts: List[pd.DataFrame] = []
    for metric in PAIRED_METRICS:
        pivot = metrics.pivot(index="outer_repeat", columns="model", values=metric)
        pivot = pivot.loc[:, list(model_order)]
        ranks = pivot.rank(
            axis=1, method="average",
            ascending=metric in LOWER_IS_BETTER_METRICS,
        )
        long = ranks.stack().rename("rank").reset_index()
        long["metric"] = metric
        detail_parts.append(long)
    detail = pd.concat(detail_parts, ignore_index=True)
    rows: List[Dict[str, object]] = []
    for (model, metric), group in detail.groupby(["model", "metric"], sort=False):
        rank = group["rank"].to_numpy(float)
        rows.append(
            {
                "model": model,
                "metric": metric,
                "n_repeats": int(len(rank)),
                "mean_rank": float(np.mean(rank)),
                "median_rank": float(np.median(rank)),
                "q25_rank": float(np.quantile(rank, 0.25)),
                "q75_rank": float(np.quantile(rank, 0.75)),
                "first_place_frequency": float(np.mean(np.isclose(rank, 1.0))),
                "top3_frequency": float(np.mean(rank <= 3.0)),
            }
        )
    summary = pd.DataFrame(rows)
    order_lookup = {model: i for i, model in enumerate(model_order)}
    summary["_order"] = summary["model"].map(order_lookup)
    summary = summary.sort_values(["metric", "_order"]).drop(columns="_order")
    return detail, summary.reset_index(drop=True)


def convergence_summaries(
    tuning: pd.DataFrame, metrics: pd.DataFrame, model_order: Sequence[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tuning = tuning.copy()
    tuning["all_fits_converged"] = to_bool(
        tuning["all_fits_converged"], "all_fits_converged"
    )
    metrics = metrics.copy()
    metrics["fit_converged"] = to_bool(metrics["fit_converged"], "fit_converged")
    inner_rows: List[Dict[str, object]] = []
    outer_rows: List[Dict[str, object]] = []
    for model in model_order:
        inner = tuning.loc[tuning["model"].astype(str) == model]
        outer = metrics.loc[metrics["model"].astype(str) == model]
        inner_rows.append(
            {
                "model": model,
                "candidate_rows": int(len(inner)),
                "converged_candidate_rows": int(inner["all_fits_converged"].sum()),
                "convergence_frequency": float(inner["all_fits_converged"].mean()),
                "max_iterations_used": int(inner["max_iterations_used"].max()),
            }
        )
        outer_rows.append(
            {
                "model": model,
                "outer_refits": int(len(outer)),
                "converged_outer_refits": int(outer["fit_converged"].sum()),
                "convergence_frequency": float(outer["fit_converged"].mean()),
                "max_iterations_used": int(outer["iterations_used"].max()),
            }
        )
    return pd.DataFrame(inner_rows), pd.DataFrame(outer_rows)


def coefficient_stability(
    coefficients: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    coefficient_tolerance: float,
    minimum_selection_frequency: float,
    minimum_sign_consistency: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require_columns(
        coefficients,
        ("model", "feature_name", "coefficient", "task_id"),
        "all coefficients",
    )
    task_counts = metrics.groupby("model")["task_id"].nunique().to_dict()
    feature_rows: List[Dict[str, object]] = []
    intercept_rows: List[Dict[str, object]] = []
    for (model, feature), group in coefficients.groupby(
        ["model", "feature_name"], sort=False
    ):
        values = pd.to_numeric(group["coefficient"], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise AggregationError(f"Non-finite coefficients for {model}/{feature}.")
        n_total = int(task_counts[model])
        n_available = int(group["task_id"].nunique())
        selected_values = values[np.abs(values) > coefficient_tolerance]
        selection_count = int(len(selected_values))
        positive_count = int(np.sum(selected_values > 0))
        negative_count = int(np.sum(selected_values < 0))
        sign_consistency = (
            float(max(positive_count, negative_count) / selection_count)
            if selection_count
            else 0.0
        )
        dominant_sign = (
            "positive"
            if positive_count > negative_count
            else "negative"
            if negative_count > positive_count
            else "tie_or_none"
        )
        row = {
            "model": str(model),
            "feature_name": str(feature),
            "n_model_repeats": n_total,
            "repeats_available": n_available,
            "availability_frequency": float(n_available / n_total),
            "selection_count": selection_count,
            "selection_frequency_all_repeats": float(selection_count / n_total),
            "selection_frequency_when_available": (
                float(selection_count / n_available) if n_available else 0.0
            ),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "sign_consistency": sign_consistency,
            "dominant_sign": dominant_sign,
            "coefficient_sum": float(np.sum(values)),
            "mean_coefficient_all_repeats": float(np.sum(values) / n_total),
            "mean_abs_coefficient_all_repeats": float(np.sum(np.abs(values)) / n_total),
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
                and selection_count / n_total >= minimum_selection_frequency
                and sign_consistency >= minimum_sign_consistency
            ),
        }
        if feature == "__INTERCEPT__":
            intercept_rows.append(row)
        else:
            feature_rows.append(row)
    features = pd.DataFrame(feature_rows)
    if not features.empty:
        features = features.sort_values(
            [
                "model",
                "stable_selection",
                "selection_frequency_all_repeats",
                "sign_consistency",
                "mean_abs_coefficient_all_repeats",
            ],
            ascending=[True, False, False, False, False],
        ).reset_index(drop=True)
    stable = features.loc[features["stable_selection"]].copy()
    intercepts = pd.DataFrame(intercept_rows)
    return features, stable, intercepts


def sample_holdout_stability(
    roles: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    model_order: Sequence[str],
    expected_repeats: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(
        roles,
        ("sample_id", "cohort", "outer_role", "outer_repeat", "task_id"),
        "all sample roles",
    )
    roles = roles.copy()
    roles["sample_id"] = roles["sample_id"].astype(str)
    sample_truth = roles.groupby("sample_id")["cohort"].agg(
        lambda x: tuple(dict.fromkeys(x.astype(str)))
    )
    inconsistent = sample_truth.loc[sample_truth.map(len) != 1]
    if not inconsistent.empty:
        raise AggregationError(
            f"Inconsistent cohort labels for samples: {inconsistent.index[:10].tolist()}"
        )
    audit = (
        roles.assign(
            is_holdout=roles["outer_role"].astype(str).eq("validation").astype(int),
            is_training=roles["outer_role"].astype(str).eq("training").astype(int),
        )
        .groupby("sample_id", as_index=False)
        .agg(
            cohort=("cohort", "first"),
            repeats_observed=("outer_repeat", "nunique"),
            holdout_appearances=("is_holdout", "sum"),
            training_appearances=("is_training", "sum"),
        )
    )
    audit["expected_repeats"] = expected_repeats
    audit["holdout_frequency"] = audit["holdout_appearances"] / expected_repeats
    if not (audit["repeats_observed"] == expected_repeats).all():
        raise AggregationError("Not every sample has a role in every repeat.")
    if not (
        audit["holdout_appearances"] + audit["training_appearances"]
        == expected_repeats
    ).all():
        raise AggregationError("Sample holdout/training role totals are inconsistent.")

    predictions = predictions.copy()
    predictions["sample_id"] = predictions["sample_id"].astype(str)
    rows: List[Dict[str, object]] = []
    for model in model_order:
        model_predictions = predictions.loc[predictions["model"].astype(str) == model]
        for item in audit.itertuples(index=False):
            group = model_predictions.loc[
                model_predictions["sample_id"] == item.sample_id
            ]
            n = int(len(group))
            expected = int(item.holdout_appearances)
            if n != expected:
                raise AggregationError(
                    f"{model}/{item.sample_id} has {n} predictions, "
                    f"expected {expected} from holdout assignments."
                )
            row: Dict[str, object] = {
                "model": model,
                "sample_id": item.sample_id,
                "true_cohort": item.cohort,
                "n_predictions": n,
                "expected_holdout_predictions": expected,
                "coverage_frequency": 1.0 if expected > 0 else np.nan,
                "holdout_frequency": float(item.holdout_frequency),
            }
            if n == 0:
                row.update(
                    {
                        "true_label": np.nan,
                        "mean_probability_ILD": np.nan,
                        "sd_probability_ILD": np.nan,
                        "median_probability_ILD": np.nan,
                        "min_probability_ILD": np.nan,
                        "max_probability_ILD": np.nan,
                        "positive_classification_frequency": np.nan,
                        "correct_classification_frequency": np.nan,
                        "mean_threshold": np.nan,
                    }
                )
            else:
                truth = group["true_label"].astype(int)
                if truth.nunique() != 1:
                    raise AggregationError(
                        f"Inconsistent true labels for {model}/{item.sample_id}."
                    )
                probability = group["probability_ILD"].to_numpy(float)
                predicted = group["predicted_label"].to_numpy(int)
                true_value = int(truth.iloc[0])
                row.update(
                    {
                        "true_label": true_value,
                        "mean_probability_ILD": float(np.mean(probability)),
                        "sd_probability_ILD": (
                            float(np.std(probability, ddof=1)) if n > 1 else 0.0
                        ),
                        "median_probability_ILD": float(np.median(probability)),
                        "min_probability_ILD": float(np.min(probability)),
                        "max_probability_ILD": float(np.max(probability)),
                        "positive_classification_frequency": float(np.mean(predicted)),
                        "correct_classification_frequency": float(
                            np.mean(predicted == true_value)
                        ),
                        "mean_threshold": float(group["threshold"].mean()),
                    }
                )
            rows.append(row)
    return audit.sort_values("sample_id").reset_index(drop=True), pd.DataFrame(rows)


def model_catalog(
    metrics: pd.DataFrame, model_order: Sequence[str]
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for order, model in enumerate(model_order, start=1):
        group = metrics.loc[metrics["model"].astype(str) == model]
        descriptions = tuple(dict.fromkeys(group["model_definition"].astype(str)))
        if len(descriptions) != 1:
            raise AggregationError(f"Inconsistent model definitions for {model}.")
        rows.append(
            {
                "model_order": order,
                "model": model,
                "model_definition": descriptions[0],
                "n_repeats": int(len(group)),
                "median_final_predictors": float(
                    pd.to_numeric(group["n_final_predictors"], errors="raise").median()
                ),
                "min_final_predictors": int(group["n_final_predictors"].min()),
                "max_final_predictors": int(group["n_final_predictors"].max()),
                "median_nonzero_coefficients": float(
                    pd.to_numeric(
                        group["n_nonzero_coefficients"], errors="raise"
                    ).median()
                ),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def format_value(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if math.isnan(value):
                return ""
            return f"{value:.4f}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(format_value(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def build_report(
    *,
    experiment_id: str,
    baseline_model: str,
    expected_tasks: int,
    expected_samples: int,
    expected_train: int,
    expected_holdout: int,
    model_order: Sequence[str],
    metric_summary: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    regularization_frequency: pd.DataFrame,
    inner_convergence: pd.DataFrame,
    outer_convergence: pd.DataFrame,
    stable_features: pd.DataFrame,
    sample_audit: pd.DataFrame,
) -> str:
    lookup = metric_summary.set_index(["model", "metric"])
    performance_rows = []
    for model in model_order:
        roc = lookup.loc[(model, "roc_auc")]
        pr = lookup.loc[(model, "pr_auc")]
        if model == baseline_model:
            delta = 0.0
            win = np.nan
        else:
            baseline_roc = baseline_summary.loc[
                (baseline_summary["comparison_model"] == model)
                & (baseline_summary["metric"] == "roc_auc")
            ]
            delta = float(baseline_roc["median"].iloc[0])
            win = float(baseline_roc["comparison_win_frequency"].iloc[0])
        performance_rows.append(
            (
                model,
                roc["median"],
                roc["q25"],
                roc["q75"],
                pr["median"],
                pr["q25"],
                pr["q75"],
                delta,
                win,
            )
        )

    family_pivot = regularization_frequency.pivot(
        index="model",
        columns="regularization_family",
        values="selection_frequency",
    ).fillna(0.0)
    family_rows = []
    for model in model_order:
        family_rows.append(
            (
                model,
                float(family_pivot.loc[model].get("Ridge_alpha0", 0.0)),
                float(family_pivot.loc[model].get("ElasticNet_interior", 0.0)),
                float(family_pivot.loc[model].get("Lasso_alpha1", 0.0)),
            )
        )

    holdout_values = sample_audit["holdout_appearances"].to_numpy(float)
    text = [
        f"# IGH repeated-holdout aggregation: {experiment_id}",
        "",
        "## Completion and design",
        "",
        f"- Completed outer tasks: {expected_tasks}/{expected_tasks}.",
        f"- Patients: {expected_samples}; each repeat uses {expected_train} training and "
        f"{expected_holdout} holdout patients.",
        f"- Models: {len(model_order)}; baseline for descriptive paired differences: "
        f"`{baseline_model}`.",
        "- Repeated holdouts are overlapping resamples, not independent clinical cohorts. "
        "Paired differences and win frequencies are descriptive; no independence-based "
        "p-values are reported.",
        "- No final model is automatically selected.",
        "",
        "## Convergence",
        "",
        f"- Inner candidate rows: {int(inner_convergence['candidate_rows'].sum())}; "
        f"all converged: {bool((inner_convergence['convergence_frequency'] == 1.0).all())}.",
        f"- Final outer refits: {int(outer_convergence['outer_refits'].sum())}; "
        f"all converged: {bool((outer_convergence['convergence_frequency'] == 1.0).all())}.",
        f"- Maximum inner iterations used: "
        f"{int(inner_convergence['max_iterations_used'].max())}.",
        f"- Maximum outer iterations used: "
        f"{int(outer_convergence['max_iterations_used'].max())}.",
        "",
        "## Model performance across repeated holdouts",
        "",
        markdown_table(
            [
                "Model",
                "ROC median",
                "ROC Q1",
                "ROC Q3",
                "PR median",
                "PR Q1",
                "PR Q3",
                "Median ΔROC vs A1",
                "ROC win frequency vs A1",
            ],
            performance_rows,
        ),
        "",
        "## Selected regularization family",
        "",
        markdown_table(
            ["Model", "Ridge α=0", "Elastic Net 0<α<1", "Lasso α=1"],
            family_rows,
        ),
        "",
        "## Feature stability",
        "",
        f"- Stable feature rows under configured frequency/sign thresholds: "
        f"{len(stable_features)}.",
        "- Repeat-specific 3-mer availability can vary by outer-training partition. "
        "The output therefore reports both availability frequency and selection "
        "frequency when available.",
        "",
        "## Holdout coverage",
        "",
        f"- Mean holdout appearances per patient: {np.mean(holdout_values):.3f}.",
        f"- SD holdout appearances: {np.std(holdout_values, ddof=1):.3f}.",
        f"- Range: {int(np.min(holdout_values))}–{int(np.max(holdout_values))}.",
        "- Patients never placed in holdout are retained in the sample audit with zero "
        "predictions rather than being silently omitted.",
        "",
        "## Output interpretation",
        "",
        "- `07_model_metric_summary_long.csv` and `_wide.csv` contain model-level "
        "performance distributions.",
        "- `07_baseline_paired_difference_summary.csv` summarizes same-repeat changes "
        "relative to A1.",
        "- Hyperparameter files explicitly show how often α=0 (Ridge), interior α "
        "(Elastic Net), and α=1 (Lasso) were selected.",
        "- Feature-stability files summarize coefficient availability, non-zero "
        "frequency, direction consistency, and magnitude.",
        "- Sample prediction stability is normalized to each patient's actual holdout "
        "appearances, not to the total repeat count.",
        "",
    ]
    return "\n".join(text)


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    compression = "gzip" if path.suffix == ".gz" else None
    frame.to_csv(path, index=False, compression=compression)


def ensure_output_policy(paths: Mapping[str, Path], overwrite: bool) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Aggregation outputs already exist; use --overwrite for a documented rerun:\n"
            + "\n".join(f"  - {path}" for path in existing)
        )


def main() -> int:
    args = parse_args()
    try:
        repository_root = (
            Path(args.repository_root).expanduser().resolve()
            if args.repository_root
            else Path.cwd().resolve()
        )
        config_path = Path(args.config).expanduser()
        if not config_path.is_absolute():
            config_path = (repository_root / config_path).resolve()
        config = load_yaml(config_path)

        experiment = require_mapping(config.get("experiment"), "experiment")
        data = require_mapping(config.get("data"), "data")
        train_data = require_mapping(data.get("train"), "data.train")
        cv = require_mapping(config.get("cross_validation"), "cross_validation")
        outer = require_mapping(config.get("outer_tasks"), "outer_tasks")
        aggregation = require_mapping(config.get("aggregation"), "aggregation")
        stability = require_mapping(config.get("stability", {}), "stability")
        models_mapping = require_mapping(config.get("models"), "models")
        model_order = tuple(str(x) for x in models_mapping.keys())

        experiment_id = str(experiment["id"])
        expected_samples = int(train_data["expected_samples"])
        expected_repeats = int(cv["outer_repeats"])
        inner_folds = int(cv["inner_folds"])
        engine = require_mapping(config.get("model_engine"), "model_engine")
        expected_candidates = len(engine["alpha_grid"]) * len(engine["lambda_grid"])

        repeated = config.get("repeated_holdout_training")
        if isinstance(repeated, Mapping):
            expected_train = int(repeated["train_size"])
            expected_holdout = int(repeated["holdout_size"])
            split_count = int(repeated["split_count"])
            if split_count != expected_repeats:
                raise AggregationError(
                    f"split_count={split_count}, outer_repeats={expected_repeats}."
                )
        else:
            expected_holdout = int(
                int(aggregation["expected_prediction_rows"])
                / (expected_repeats * len(model_order))
            )
            expected_train = expected_samples - expected_holdout

        output_root = resolve_path(
            outer["output_root"], repository_root, "outer_tasks.output_root"
        )
        tasks = build_tasks(config, output_root)
        expected_tasks = int(outer["expected_tasks"])

        if args.output_dir:
            output_dir = Path(args.output_dir).expanduser().resolve()
        else:
            output_dir = resolve_path(
                aggregation["output_dir"], repository_root, "aggregation.output_dir"
            )
        output_paths = {
            key: output_dir / filename for key, filename in OUTPUT_FILES.items()
        }
        ensure_output_policy(output_paths, args.overwrite)
        output_dir.mkdir(parents=True, exist_ok=True)

        status_rows = []
        role_parts = []
        metric_parts = []
        prediction_parts = []
        tuning_parts = []
        oof_parts = []
        coefficient_parts = []
        vocabulary_parts = []

        for task in tasks:
            meta, frames = inspect_and_load_task(
                task,
                expected_models=model_order,
                expected_samples=expected_samples,
                expected_train=expected_train,
                expected_holdout=expected_holdout,
                expected_candidates=expected_candidates,
                inner_folds=inner_folds,
            )
            status_rows.append(meta)
            role_parts.append(annotate(frames["roles"], task))
            metric_parts.append(annotate(frames["metrics"], task))
            prediction_parts.append(annotate(frames["predictions"], task))
            tuning_parts.append(annotate(frames["tuning"], task))
            oof_parts.append(annotate(frames["oof"], task))
            coefficient_parts.append(annotate(frames["coefficients"], task))
            if "vocabulary" in frames:
                vocabulary_parts.append(annotate(frames["vocabulary"], task))

        status = pd.DataFrame(status_rows).sort_values("task_index").reset_index(drop=True)
        roles = pd.concat(role_parts, ignore_index=True)
        metrics = pd.concat(metric_parts, ignore_index=True)
        predictions = pd.concat(prediction_parts, ignore_index=True)
        tuning = pd.concat(tuning_parts, ignore_index=True)
        oof = pd.concat(oof_parts, ignore_index=True)
        coefficients = pd.concat(coefficient_parts, ignore_index=True)
        vocabularies = (
            pd.concat(vocabulary_parts, ignore_index=True)
            if vocabulary_parts
            else pd.DataFrame()
        )

        if len(status) != expected_tasks:
            raise AggregationError("Completed task count mismatch.")
        if metrics.duplicated(["task_id", "model"]).any():
            raise AggregationError("Aggregated metrics contain duplicate task/model rows.")
        if predictions.duplicated(["task_id", "model", "sample_id"]).any():
            raise AggregationError(
                "Aggregated predictions contain duplicate task/model/sample rows."
            )
        if metrics["outer_repeat"].nunique() != expected_repeats:
            raise AggregationError("Aggregated repeat coverage mismatch.")
        if tuple(dict.fromkeys(metrics["model"].astype(str))) != model_order:
            raise AggregationError("Aggregated model order mismatch.")

        expected_metric_rows = expected_tasks * len(model_order)
        expected_prediction_rows = expected_tasks * expected_holdout * len(model_order)
        expected_tuning_rows = expected_tasks * len(model_order) * expected_candidates
        expected_oof_rows = expected_tasks * expected_train * len(model_order)
        observed = {
            "metric_rows": len(metrics),
            "prediction_rows": len(predictions),
            "tuning_rows": len(tuning),
            "oof_rows": len(oof),
        }
        expected = {
            "metric_rows": expected_metric_rows,
            "prediction_rows": expected_prediction_rows,
            "tuning_rows": expected_tuning_rows,
            "oof_rows": expected_oof_rows,
        }
        if observed != expected:
            raise AggregationError(
                f"Aggregated row counts={observed}, expected={expected}."
            )

        repeat_metrics = metrics.copy()
        metric_long, metric_wide = metric_summaries(metrics, model_order)
        (
            selected,
            hyper_frequency,
            alpha_frequency,
            lambda_frequency,
            regularization_frequency,
        ) = selected_hyperparameters(metrics, model_order)
        baseline_detail, baseline_summary = paired_differences(
            metrics,
            baseline_model=args.baseline_model,
            model_order=model_order,
        )
        pairwise = all_pairwise_differences(metrics, model_order)
        rank_detail, rank_summary = model_ranks(metrics, model_order)
        inner_convergence, outer_convergence = convergence_summaries(
            tuning, metrics, model_order
        )
        coefficient_tolerance = float(
            stability.get("coefficient_tolerance", 1.0e-12)
        )
        minimum_selection_frequency = float(
            stability.get("minimum_selection_frequency", 0.5)
        )
        minimum_sign_consistency = float(
            stability.get("minimum_sign_consistency", 0.8)
        )
        feature_stability, stable_features, intercept_stability = coefficient_stability(
            coefficients,
            metrics,
            coefficient_tolerance=coefficient_tolerance,
            minimum_selection_frequency=minimum_selection_frequency,
            minimum_sign_consistency=minimum_sign_consistency,
        )
        sample_audit, sample_stability = sample_holdout_stability(
            roles,
            predictions,
            model_order=model_order,
            expected_repeats=expected_repeats,
        )
        catalog = model_catalog(metrics, model_order)

        report = build_report(
            experiment_id=experiment_id,
            baseline_model=args.baseline_model,
            expected_tasks=expected_tasks,
            expected_samples=expected_samples,
            expected_train=expected_train,
            expected_holdout=expected_holdout,
            model_order=model_order,
            metric_summary=metric_long,
            baseline_summary=baseline_summary,
            regularization_frequency=regularization_frequency,
            inner_convergence=inner_convergence,
            outer_convergence=outer_convergence,
            stable_features=stable_features,
            sample_audit=sample_audit,
        )

        frame_map: Mapping[str, pd.DataFrame] = {
            "task_status": status,
            "all_metrics": metrics,
            "all_predictions": predictions,
            "all_tuning": tuning,
            "all_oof": oof,
            "selected_hyperparameters": selected,
            "all_coefficients": coefficients,
            "all_vocabularies": vocabularies,
            "model_catalog": catalog,
            "metric_summary_long": metric_long,
            "metric_summary_wide": metric_wide,
            "repeat_metrics": repeat_metrics,
            "baseline_differences": baseline_detail,
            "baseline_difference_summary": baseline_summary,
            "pairwise_differences": pairwise,
            "rank_detail": rank_detail,
            "rank_summary": rank_summary,
            "hyperparameter_frequency": hyper_frequency,
            "alpha_frequency": alpha_frequency,
            "lambda_frequency": lambda_frequency,
            "regularization_family_frequency": regularization_frequency,
            "inner_convergence": inner_convergence,
            "outer_convergence": outer_convergence,
            "feature_stability": feature_stability,
            "stable_features": stable_features,
            "intercept_stability": intercept_stability,
            "sample_holdout_counts": sample_audit,
            "sample_prediction_stability": sample_stability,
        }
        for key, frame in frame_map.items():
            write_frame(frame, output_paths[key])
        output_paths["report"].write_text(report, encoding="utf-8")

        holdout_counts = sample_audit["holdout_appearances"].to_numpy(int)
        summary = {
            "status": "COMPLETE",
            "aggregation_mode": "repeated_holdout",
            "experiment_id": experiment_id,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "config": str(config_path),
            "output_dir": str(output_dir),
            "baseline_model": args.baseline_model,
            "expected_outer_tasks": expected_tasks,
            "completed_outer_tasks": int(len(status)),
            "models": list(model_order),
            "model_count": len(model_order),
            "patients": expected_samples,
            "outer_repeats": expected_repeats,
            "train_size_per_repeat": expected_train,
            "holdout_size_per_repeat": expected_holdout,
            "candidate_count_per_model": expected_candidates,
            "inner_folds": inner_folds,
            "expected_inner_fits": expected_tuning_rows * inner_folds,
            "outer_refits": expected_metric_rows,
            "row_counts": {
                "outer_metrics": int(len(metrics)),
                "outer_predictions": int(len(predictions)),
                "inner_tuning": int(len(tuning)),
                "inner_selected_oof": int(len(oof)),
                "coefficients": int(len(coefficients)),
                "repeat_3mer_vocabulary": int(len(vocabularies)),
                "stable_features": int(len(stable_features)),
            },
            "convergence": {
                "all_inner_candidates_converged": bool(
                    (inner_convergence["convergence_frequency"] == 1.0).all()
                ),
                "max_inner_iterations_used": int(
                    inner_convergence["max_iterations_used"].max()
                ),
                "all_outer_refits_converged": bool(
                    (outer_convergence["convergence_frequency"] == 1.0).all()
                ),
                "max_outer_iterations_used": int(
                    outer_convergence["max_iterations_used"].max()
                ),
            },
            "holdout_appearances": {
                "mean": float(np.mean(holdout_counts)),
                "sd": float(np.std(holdout_counts, ddof=1)),
                "min": int(np.min(holdout_counts)),
                "median": float(np.median(holdout_counts)),
                "max": int(np.max(holdout_counts)),
                "zero_holdout_patients": int(np.sum(holdout_counts == 0)),
            },
            "stable_feature_thresholds": {
                "coefficient_tolerance": coefficient_tolerance,
                "minimum_selection_frequency": minimum_selection_frequency,
                "minimum_sign_consistency": minimum_sign_consistency,
            },
            "scientific_contract": {
                "repeats_treated_as_independent_cohorts": False,
                "paired_differences_are_descriptive": True,
                "p_values_calculated": False,
                "final_model_selected": False,
                "independent_test_read": False,
            },
            "output_files": {key: str(path) for key, path in output_paths.items()},
        }
        output_paths["summary"].write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        marker = {
            "status": "COMPLETE",
            "aggregation_mode": "repeated_holdout",
            "experiment_id": experiment_id,
            "completed_outer_tasks": expected_tasks,
            "expected_outer_tasks": expected_tasks,
            "model_count": len(model_order),
            "output_dir": str(output_dir),
            "summary": str(output_paths["summary"]),
            "final_model_selected": False,
            "independent_test_read": False,
        }
        output_paths["complete"].write_text(
            json.dumps(marker, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        print("IGH repeated-holdout aggregation: COMPLETE")
        print(f"Experiment:          {experiment_id}")
        print(f"Tasks:               {len(status)}/{expected_tasks}")
        print(f"Models:              {len(model_order)}")
        print(f"Outer metric rows:   {len(metrics)}")
        print(f"Prediction rows:     {len(predictions)}")
        print(f"Inner tuning rows:   {len(tuning)}")
        print(f"Inner OOF rows:      {len(oof)}")
        print(f"Stable features:     {len(stable_features)}")
        print(f"Output:              {output_dir}")
        print(f"Summary report:      {output_paths['report']}")
        print(f"Completion marker:   {output_paths['complete']}")
        print("IGH_REPEATED_HOLDOUT_AGGREGATION_PASS")
        return 0
    except Exception as exc:
        print(f"IGH repeated-holdout aggregation: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
