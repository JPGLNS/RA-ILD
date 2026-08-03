#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only visualization of frozen IGH repeated-holdout results.

The module consumes the dedicated IGH repeated-holdout aggregation outputs
(``07_*`` files). It never refits a model, retunes hyperparameters, rebuilds
public references, changes thresholds, or reads the independent test set.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class VisualizationError(ValueError):
    """Raised when visualization inputs violate the frozen-result contract."""


VISUALIZATION_VERSION = "1.0.1"
AGGREGATION_FILES: Mapping[str, str] = {
    "metrics": "07_all_outer_metrics.csv",
    "predictions": "07_all_outer_predictions.csv.gz",
    "summary": "07_aggregation_summary.json",
    "complete": "07_REPEATED_HOLDOUT_AGGREGATION_COMPLETE.json",
}
OUTPUT_FILES: Mapping[str, str] = {
    "figure1": "Figure1_model_performance_boxplots.pdf",
    "figure1_data": "Figure1_plot_data.csv",
    "figure2": "Figure2_mean_ROC_curves.pdf",
    "figure2_coordinates": "Figure2_mean_ROC_coordinates.csv.gz",
    "figure2_repeat_auc": "Figure2_repeat_ROC_AUC.csv",
    "figure3": "Figure3_best_repeat_ROC_curves.pdf",
    "figure3_selection": "Figure3_best_repeat_selection.csv",
    "figure3_coordinates": "Figure3_best_repeat_ROC_coordinates.csv.gz",
    "figure4": "Figure4_best_repeat_prediction_boxplots.pdf",
    "figure4_data": "Figure4_plot_data.csv",
    "input_audit": "visualization_input_audit.csv",
    "resolved_config": "visualization_resolved_config.yaml",
    "manifest": "visualization_manifest.json",
    "complete": "VISUALIZATION_COMPLETE.json",
}
METRIC_LABELS: Mapping[str, str] = {
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC (Average Precision)",
    "accuracy": "Accuracy",
    "sensitivity_recall": "Sensitivity",
    "specificity": "Specificity",
    "precision": "Precision",
    "f1": "F1 score",
    "log_loss": "Log loss",
    "brier_score": "Brier score",
}
HIGHER_IS_BETTER = {
    "roc_auc",
    "pr_auc",
    "accuracy",
    "sensitivity_recall",
    "specificity",
    "precision",
    "f1",
}
LOWER_IS_BETTER = {"log_loss", "brier_score"}
IDENTIFIER_COLUMNS = ("task_id", "task_index", "outer_repeat", "outer_fold")


@dataclass(frozen=True)
class VisualizationConfig:
    source_path: Path
    repository_root: Path
    visualization_id: str
    aggregation_dir: Path
    output_dir: Path
    require_complete_marker: bool
    overwrite: bool
    metric_tolerance: float
    model_labels: Mapping[str, str]
    figures: Mapping[str, Mapping[str, Any]]
    style: Mapping[str, Any]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class AggregatedResults:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    summary: Mapping[str, Any]
    complete_marker: Mapping[str, Any]
    input_paths: Mapping[str, Path]
    input_sha256: Mapping[str, str]


@dataclass(frozen=True)
class VisualizationResult:
    output_dir: Path
    generated_files: Mapping[str, Path]
    selected_best_repeats: pd.DataFrame


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _trapezoid_area(y: np.ndarray, x: np.ndarray) -> float:
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = getattr(np, "trapz", None)
    if integrate is None:  # pragma: no cover
        raise VisualizationError("NumPy provides neither trapezoid nor trapz.")
    return float(integrate(np.asarray(y, float), np.asarray(x, float)))


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise VisualizationError(f"Unable to read {label}: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise VisualizationError(f"{label} must contain a JSON object: {path}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VisualizationError(f"{label} must be a mapping.")
    return value


def _resolve_path(value: Any, repository_root: Path, label: str) -> Path:
    if value is None or not str(value).strip():
        raise VisualizationError(f"{label} is required.")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = repository_root / path
    return path.resolve()


def _unique_strings(value: Any, label: str) -> Tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise VisualizationError(f"{label} must be a non-empty list.")
    items = tuple(str(item).strip() for item in value)
    if not items or any(not item for item in items) or len(items) != len(set(items)):
        raise VisualizationError(f"{label} must contain unique non-empty strings.")
    return items


def load_visualization_config(
    path: Path,
    *,
    repository_root: Optional[Path] = None,
) -> VisualizationConfig:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Visualization config not found: {source_path}")
    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise VisualizationError(f"Unable to parse YAML: {source_path}: {exc}") from exc
    raw = _mapping(raw, "visualization config")
    if str(raw.get("visualization_version")) != "1.0":
        raise VisualizationError("visualization_version must be '1.0'.")
    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root is not None
        else Path.cwd().resolve()
    )
    section = _mapping(raw.get("visualization"), "visualization")
    visualization_id = str(section.get("id", "")).strip()
    if not visualization_id:
        raise VisualizationError("visualization.id is required.")
    if str(section.get("source_mode", "native_repeated_holdout")) != "native_repeated_holdout":
        raise VisualizationError("visualization.source_mode must be native_repeated_holdout.")
    aggregation_dir = _resolve_path(section.get("aggregation_dir"), root, "visualization.aggregation_dir")
    output_dir = _resolve_path(section.get("output_dir"), root, "visualization.output_dir")
    metric_tolerance = float(section.get("metric_tolerance", 1.0e-6))
    if not np.isfinite(metric_tolerance) or metric_tolerance <= 0:
        raise VisualizationError("visualization.metric_tolerance must be finite and > 0.")
    labels = {
        str(key): str(value)
        for key, value in _mapping(raw.get("model_labels", {}), "model_labels").items()
    }
    figures_raw = _mapping(raw.get("figures"), "figures")
    unknown = sorted(set(figures_raw) - {"figure1", "figure2", "figure3", "figure4"})
    if unknown:
        raise VisualizationError(f"Unknown figure sections: {unknown}")
    figures: Dict[str, Mapping[str, Any]] = {}
    enabled = 0
    for name in ("figure1", "figure2", "figure3", "figure4"):
        item = dict(_mapping(figures_raw.get(name, {"enabled": False}), name))
        figures[name] = item
        if bool(item.get("enabled", False)):
            enabled += 1
            selector = item.get("models", "all")
            if isinstance(selector, str):
                if selector.strip().lower() != "all":
                    raise VisualizationError(f"figures.{name}.models must be 'all' or a list.")
            else:
                _unique_strings(selector, f"figures.{name}.models")
            if name == "figure1":
                metrics = _unique_strings(item.get("metrics", ()), "figures.figure1.metrics")
                unsupported = sorted(set(metrics) - set(METRIC_LABELS))
                if unsupported:
                    raise VisualizationError(f"Unsupported Figure 1 metrics: {unsupported}")
    if enabled == 0:
        raise VisualizationError("At least one figure must be enabled.")
    style = dict(_mapping(raw.get("style", {}), "style"))
    return VisualizationConfig(
        source_path=source_path,
        repository_root=root,
        visualization_id=visualization_id,
        aggregation_dir=aggregation_dir,
        output_dir=output_dir,
        require_complete_marker=bool(section.get("require_complete_marker", True)),
        overwrite=bool(section.get("overwrite", False)),
        metric_tolerance=metric_tolerance,
        model_labels=labels,
        figures=figures,
        style=style,
        raw=raw,
    )


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise VisualizationError(f"{label} is missing columns: {missing}")


def load_aggregated_results(config: VisualizationConfig) -> AggregatedResults:
    paths = {key: config.aggregation_dir / value for key, value in AGGREGATION_FILES.items()}
    missing = [key for key, path in paths.items() if not path.is_file()]
    if missing:
        raise VisualizationError(
            "Aggregation directory is missing required files: "
            + ", ".join(f"{key}={paths[key]}" for key in missing)
        )
    summary = _read_json(paths["summary"], "aggregation summary")
    complete = _read_json(paths["complete"], "aggregation completion marker")
    if config.require_complete_marker:
        if complete.get("status") != "COMPLETE":
            raise VisualizationError("Aggregation completion marker is not COMPLETE.")
        if complete.get("aggregation_mode") != "repeated_holdout":
            raise VisualizationError("Aggregation marker mode is not repeated_holdout.")
    if summary.get("aggregation_mode") != "repeated_holdout":
        raise VisualizationError("Aggregation summary mode is not repeated_holdout.")
    contract = summary.get("scientific_contract", {})
    if isinstance(contract, Mapping):
        if bool(contract.get("repeats_treated_as_independent_cohorts", False)):
            raise VisualizationError("Aggregation incorrectly treats repeats as independent cohorts.")
        if bool(contract.get("final_model_selected", False)):
            raise VisualizationError("Visualization input must not claim automatic final-model selection.")
    metrics = pd.read_csv(paths["metrics"])
    predictions = pd.read_csv(paths["predictions"])
    _require_columns(
        metrics,
        ("model", "outer_repeat", "outer_fold", "roc_auc", "pr_auc", "f1"),
        "aggregated metrics",
    )
    _require_columns(
        predictions,
        (
            "model",
            "outer_repeat",
            "outer_fold",
            "sample_id",
            "true_label",
            "true_cohort",
            "probability_ILD",
            "threshold",
            "predicted_label",
        ),
        "aggregated predictions",
    )
    if metrics.duplicated(["model", "outer_repeat", "outer_fold"]).any():
        raise VisualizationError("Aggregated metrics contain duplicate model/repeat/fold rows.")
    if predictions.duplicated(["model", "outer_repeat", "outer_fold", "sample_id"]).any():
        raise VisualizationError("Aggregated predictions contain duplicate model/repeat/fold/sample rows.")
    if not np.isfinite(pd.to_numeric(predictions["probability_ILD"], errors="raise")).all():
        raise VisualizationError("Prediction probabilities contain non-finite values.")
    p = predictions["probability_ILD"].to_numpy(float)
    if np.any((p < 0) | (p > 1)):
        raise VisualizationError("Prediction probabilities are outside [0, 1].")
    thresholds = pd.to_numeric(predictions["threshold"], errors="raise").to_numpy(float)
    if not np.isfinite(thresholds).all() or np.any((thresholds < 0) | (thresholds > 1)):
        raise VisualizationError("Saved thresholds are invalid.")
    observed_models = tuple(dict.fromkeys(metrics["model"].astype(str)))
    summary_models = tuple(str(value) for value in summary.get("models", observed_models))
    if summary_models != observed_models:
        raise VisualizationError(
            f"Model order mismatch between summary and metrics: {summary_models} vs {observed_models}."
        )
    if set(predictions["model"].astype(str)) != set(observed_models):
        raise VisualizationError("Prediction model set does not match metrics.")
    return AggregatedResults(
        metrics=metrics,
        predictions=predictions,
        summary=summary,
        complete_marker=complete,
        input_paths=paths,
        input_sha256={key: sha256_file(path) for key, path in paths.items()},
    )


def _model_selector(selector: Any, available: Sequence[str], label: str) -> Tuple[str, ...]:
    if isinstance(selector, str) and selector.strip().lower() == "all":
        return tuple(available)
    selected = _unique_strings(selector, label)
    unknown = [name for name in selected if name not in available]
    if unknown:
        raise VisualizationError(f"Unknown models in {label}: {unknown}")
    return selected


def _metric_values(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    predicted = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, predicted)),
        "sensitivity_recall": float(recall_score(y, predicted, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
    }


def reproduce_metric_audit(results: AggregatedResults, tolerance: float) -> pd.DataFrame:
    indexed = results.metrics.set_index(["model", "outer_repeat", "outer_fold"])
    rows: List[Dict[str, Any]] = []
    for key, group in results.predictions.groupby(
        ["model", "outer_repeat", "outer_fold"], sort=False
    ):
        model, repeat, fold = key
        if key not in indexed.index:
            raise VisualizationError(f"Predictions have no matching metric row: {key}")
        y = pd.to_numeric(group["true_label"], errors="raise").to_numpy(int)
        if set(np.unique(y).tolist()) != {0, 1}:
            raise VisualizationError(f"{key} does not contain both classes.")
        p = pd.to_numeric(group["probability_ILD"], errors="raise").to_numpy(float)
        threshold_values = pd.to_numeric(group["threshold"], errors="raise").unique()
        if len(threshold_values) != 1:
            raise VisualizationError(f"{key} contains multiple saved thresholds.")
        threshold = float(threshold_values[0])
        recomputed_prediction = (p >= threshold).astype(int)
        saved_prediction = pd.to_numeric(group["predicted_label"], errors="raise").to_numpy(int)
        if not np.array_equal(recomputed_prediction, saved_prediction):
            raise VisualizationError(f"Saved predicted labels do not match threshold for {key}.")
        calculated = _metric_values(y, p, threshold)
        saved = indexed.loc[key]
        for metric, value in calculated.items():
            if metric not in results.metrics.columns:
                continue
            saved_value = float(saved[metric])
            difference = value - saved_value
            passed = bool(np.isclose(value, saved_value, atol=tolerance, rtol=0.0))
            rows.append(
                {
                    "model": str(model),
                    "outer_repeat": int(repeat),
                    "outer_fold": int(fold),
                    "metric": metric,
                    "saved_value": saved_value,
                    "recalculated_value": value,
                    "difference_recalculated_minus_saved": difference,
                    "absolute_difference": abs(difference),
                    "tolerance": tolerance,
                    "passed": passed,
                    "n_holdout": int(len(group)),
                    "n_RA": int(np.sum(y == 0)),
                    "n_ILD": int(np.sum(y == 1)),
                }
            )
    audit = pd.DataFrame(rows)
    if audit.empty or not audit["passed"].all():
        failed = audit.loc[~audit["passed"]].head(20) if not audit.empty else audit
        raise VisualizationError(
            "Saved metrics do not reproduce from frozen predictions:\n"
            + failed.to_string(index=False)
        )
    expected_groups = len(results.metrics)
    observed_groups = results.predictions.groupby(
        ["model", "outer_repeat", "outer_fold"]
    ).ngroups
    if observed_groups != expected_groups:
        raise VisualizationError(
            f"Prediction groups={observed_groups}, metric rows={expected_groups}."
        )
    return audit


def _panel_layout(n: int, max_columns: int = 3) -> Tuple[int, int]:
    if n < 1:
        raise VisualizationError("At least one panel is required.")
    columns = min(max(1, int(max_columns)), n)
    rows = int(math.ceil(n / columns))
    return rows, columns


def _axes_grid(n: int, style: Mapping[str, Any], max_columns: int = 3):
    rows, columns = _panel_layout(n, max_columns=max_columns)
    width = float(style.get("panel_width", 5.0))
    height = float(style.get("panel_height", 4.2))
    fig, axes = plt.subplots(rows, columns, figsize=(width * columns, height * rows), squeeze=False)
    flat = list(axes.ravel())
    for axis in flat[n:]:
        axis.set_visible(False)
    return fig, flat[:n]


def _display_label(model: str, config: VisualizationConfig) -> str:
    return config.model_labels.get(model, model)


def _save_pdf(fig, path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def _figure1(
    config: VisualizationConfig,
    results: AggregatedResults,
    output_path: Path,
) -> pd.DataFrame:
    section = config.figures["figure1"]
    models = _model_selector(section.get("models", "all"), tuple(dict.fromkeys(results.metrics["model"].astype(str))), "figure1.models")
    metrics = _unique_strings(section.get("metrics", ()), "figure1.metrics")
    missing = sorted(set(metrics) - set(results.metrics.columns))
    if missing:
        raise VisualizationError(f"Figure 1 metrics are absent from aggregation: {missing}")
    fig, axes = _axes_grid(len(metrics), config.style, int(section.get("max_columns", 3)))
    show_points = bool(section.get("show_points", True))
    show_mean = bool(section.get("show_mean", True))
    rng = np.random.default_rng(int(section.get("jitter_seed", 20260803)))
    rows: List[pd.DataFrame] = []
    for axis, metric in zip(axes, metrics):
        values = []
        for model in models:
            subset = results.metrics.loc[results.metrics["model"].astype(str) == model]
            series = pd.to_numeric(subset[metric], errors="raise").to_numpy(float)
            if not np.isfinite(series).all():
                raise VisualizationError(f"Non-finite Figure 1 values for {model}/{metric}.")
            values.append(series)
            part = subset[[column for column in IDENTIFIER_COLUMNS if column in subset.columns]].copy()
            part.insert(0, "model", model)
            part["metric"] = metric
            part["value"] = series
            rows.append(part)
        _boxplot_with_tick_labels(
            axis,
            values,
            [_display_label(model, config) for model in models],
            showmeans=show_mean,
        )
        if show_points:
            for index, series in enumerate(values, start=1):
                jitter = rng.normal(0.0, float(section.get("jitter_width", 0.05)), size=len(series))
                axis.scatter(np.full(len(series), index) + jitter, series, s=float(section.get("point_size", 10)), alpha=float(section.get("point_alpha", 0.30)))
        axis.set_title(METRIC_LABELS[metric])
        axis.tick_params(axis="x", rotation=float(section.get("x_label_rotation", 45)))
        axis.grid(axis="y", alpha=0.25)
        if metric in HIGHER_IS_BETTER:
            axis.set_ylabel("Higher is better")
        elif metric in LOWER_IS_BETTER:
            axis.set_ylabel("Lower is better")
    _save_pdf(fig, output_path, int(config.style.get("dpi", 150)))
    return pd.concat(rows, ignore_index=True)


def _boxplot_with_tick_labels(
    axis: Any,
    values: Sequence[np.ndarray],
    tick_labels: Sequence[str],
    **kwargs: Any,
) -> Mapping[str, Any]:
    """Draw a boxplot across Matplotlib API generations.

    Matplotlib 3.9 renamed ``labels`` to ``tick_labels`` and Matplotlib 3.11
    removed the old keyword. Older supported environments may still expose
    only ``labels``. Inspect the bound method signature so the same frozen
    visualization package works in both API families without suppressing
    unrelated TypeError exceptions.
    """

    parameters = inspect.signature(axis.boxplot).parameters
    label_keyword = "tick_labels" if "tick_labels" in parameters else "labels"
    options = dict(kwargs)
    options[label_keyword] = list(tick_labels)
    return axis.boxplot(values, **options)


def _roc_group(group: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float]:
    y = pd.to_numeric(group["true_label"], errors="raise").to_numpy(int)
    p = pd.to_numeric(group["probability_ILD"], errors="raise").to_numpy(float)
    if set(np.unique(y).tolist()) != {0, 1}:
        raise VisualizationError("ROC group does not contain both classes.")
    fpr, tpr, _ = roc_curve(y, p)
    return fpr, tpr, float(roc_auc_score(y, p))


def _figure2(
    config: VisualizationConfig,
    results: AggregatedResults,
    output_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    section = config.figures["figure2"]
    available = tuple(dict.fromkeys(results.metrics["model"].astype(str)))
    models = _model_selector(section.get("models", "all"), available, "figure2.models")
    grid_points = int(section.get("grid_points", 201))
    if grid_points < 20:
        raise VisualizationError("figure2.grid_points must be >= 20.")
    lower = float(section.get("variability_lower_quantile", 0.025))
    upper = float(section.get("variability_upper_quantile", 0.975))
    if not 0 <= lower < upper <= 1:
        raise VisualizationError("Figure 2 variability quantiles are invalid.")
    common_fpr = np.linspace(0.0, 1.0, grid_points)
    fig, axes = _axes_grid(len(models), config.style, int(section.get("max_columns", 3)))
    coordinate_rows: List[Dict[str, Any]] = []
    auc_rows: List[Dict[str, Any]] = []
    for axis, model in zip(axes, models):
        curves = []
        model_predictions = results.predictions.loc[results.predictions["model"].astype(str) == model]
        for (repeat, fold), group in model_predictions.groupby(["outer_repeat", "outer_fold"], sort=True):
            fpr, tpr, auc = _roc_group(group)
            interpolated = np.interp(common_fpr, fpr, tpr)
            interpolated[0] = 0.0
            interpolated[-1] = 1.0
            curves.append(interpolated)
            auc_rows.append({"model": model, "outer_repeat": int(repeat), "outer_fold": int(fold), "roc_auc": auc})
        matrix = np.vstack(curves)
        mean_tpr = matrix.mean(axis=0)
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0
        low_tpr = np.quantile(matrix, lower, axis=0)
        high_tpr = np.quantile(matrix, upper, axis=0)
        repeat_aucs = np.asarray([row["roc_auc"] for row in auc_rows if row["model"] == model], float)
        mean_auc = float(np.mean(repeat_aucs))
        sd_auc = float(np.std(repeat_aucs, ddof=1)) if len(repeat_aucs) > 1 else 0.0
        auc_of_mean_curve = _trapezoid_area(mean_tpr, common_fpr)
        axis.plot(common_fpr, mean_tpr, linewidth=2.0, label=f"Mean repeat AUC={mean_auc:.3f} ± {sd_auc:.3f}")
        if bool(section.get("show_variability_band", True)):
            axis.fill_between(common_fpr, low_tpr, high_tpr, alpha=float(section.get("band_alpha", 0.20)), label=f"Across-repeat q{lower:.3f}–q{upper:.3f}")
        axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
        axis.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1), title=_display_label(model, config))
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right", fontsize=8)
        for index, fpr_value in enumerate(common_fpr):
            coordinate_rows.append(
                {
                    "model": model,
                    "fpr": float(fpr_value),
                    "mean_tpr": float(mean_tpr[index]),
                    "lower_tpr": float(low_tpr[index]),
                    "upper_tpr": float(high_tpr[index]),
                    "repeat_count": int(matrix.shape[0]),
                    "mean_repeat_roc_auc": mean_auc,
                    "sd_repeat_roc_auc": sd_auc,
                    "auc_of_mean_curve": auc_of_mean_curve,
                    "variability_lower_quantile": lower,
                    "variability_upper_quantile": upper,
                }
            )
    _save_pdf(fig, output_path, int(config.style.get("dpi", 150)))
    return pd.DataFrame(coordinate_rows), pd.DataFrame(auc_rows)


def _best_repeat_selection(
    config: VisualizationConfig,
    results: AggregatedResults,
    figure_name: str,
    models: Sequence[str],
) -> pd.DataFrame:
    section = config.figures[figure_name]
    order = tuple(section.get("selection_metrics", ["roc_auc", "pr_auc", "f1"]))
    missing = sorted(set(order) - set(results.metrics.columns))
    if missing:
        raise VisualizationError(f"Best-repeat selection metrics are missing: {missing}")
    rows = []
    for model in models:
        subset = results.metrics.loc[results.metrics["model"].astype(str) == model].copy()
        sort_columns = list(order) + ["outer_repeat", "outer_fold"]
        ascending = [metric in LOWER_IS_BETTER for metric in order] + [True, True]
        selected = subset.sort_values(sort_columns, ascending=ascending, kind="mergesort").iloc[0]
        row = {
            "model": model,
            "outer_repeat": int(selected["outer_repeat"]),
            "outer_fold": int(selected["outer_fold"]),
            "selection_order": " > ".join(
                f"{metric} {'ascending' if metric in LOWER_IS_BETTER else 'descending'}"
                for metric in order
            ) + " > outer_repeat ascending > outer_fold ascending",
        }
        for metric in METRIC_LABELS:
            if metric in selected.index:
                row[metric] = float(selected[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def _figure3(
    config: VisualizationConfig,
    results: AggregatedResults,
    output_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    section = config.figures["figure3"]
    available = tuple(dict.fromkeys(results.metrics["model"].astype(str)))
    models = _model_selector(section.get("models", "all"), available, "figure3.models")
    selection = _best_repeat_selection(config, results, "figure3", models)
    fig, axes = _axes_grid(len(models), config.style, int(section.get("max_columns", 3)))
    coordinate_rows: List[Dict[str, Any]] = []
    for axis, item in zip(axes, selection.itertuples(index=False)):
        group = results.predictions.loc[
            (results.predictions["model"].astype(str) == item.model)
            & (results.predictions["outer_repeat"].astype(int) == item.outer_repeat)
            & (results.predictions["outer_fold"].astype(int) == item.outer_fold)
        ].copy()
        fpr, tpr, auc = _roc_group(group)
        axis.plot(fpr, tpr, linewidth=2.0, label=f"ROC-AUC={auc:.3f}")
        axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
        if bool(section.get("show_operating_point", True)):
            y = group["true_label"].to_numpy(int)
            pred = group["predicted_label"].to_numpy(int)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            operating_fpr = fp / (fp + tn) if fp + tn else np.nan
            operating_tpr = tp / (tp + fn) if tp + fn else np.nan
            axis.scatter([operating_fpr], [operating_tpr], s=45, label="Saved threshold")
        axis.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1), title=f"{_display_label(item.model, config)}\nrepeat {item.outer_repeat}")
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right", fontsize=8)
        for fpr_value, tpr_value in zip(fpr, tpr):
            coordinate_rows.append(
                {
                    "model": item.model,
                    "outer_repeat": item.outer_repeat,
                    "outer_fold": item.outer_fold,
                    "fpr": float(fpr_value),
                    "tpr": float(tpr_value),
                    "roc_auc": auc,
                }
            )
    _save_pdf(fig, output_path, int(config.style.get("dpi", 150)))
    return selection, pd.DataFrame(coordinate_rows)


def _figure4(
    config: VisualizationConfig,
    results: AggregatedResults,
    output_path: Path,
    shared_selection: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    section = config.figures["figure4"]
    available = tuple(dict.fromkeys(results.metrics["model"].astype(str)))
    models = _model_selector(section.get("models", "all"), available, "figure4.models")
    reuse = bool(section.get("reuse_figure3_selection", True))
    if reuse:
        if shared_selection is None:
            raise VisualizationError("Figure 4 requests Figure 3 selection reuse, but Figure 3 is disabled.")
        selected_models = tuple(shared_selection["model"].astype(str))
        if tuple(models) != selected_models:
            raise VisualizationError("Figure 4 models must exactly match Figure 3 when reuse is enabled.")
        selection = shared_selection.copy()
    else:
        selection = _best_repeat_selection(config, results, "figure4", models)
    fig, axes = _axes_grid(len(models), config.style, int(section.get("max_columns", 3)))
    output_parts: List[pd.DataFrame] = []
    rng = np.random.default_rng(int(section.get("jitter_seed", 20260803)))
    for axis, item in zip(axes, selection.itertuples(index=False)):
        group = results.predictions.loc[
            (results.predictions["model"].astype(str) == item.model)
            & (results.predictions["outer_repeat"].astype(int) == item.outer_repeat)
            & (results.predictions["outer_fold"].astype(int) == item.outer_fold)
        ].copy()
        ra = group.loc[group["true_label"].astype(int) == 0, "probability_ILD"].to_numpy(float)
        ild = group.loc[group["true_label"].astype(int) == 1, "probability_ILD"].to_numpy(float)
        _boxplot_with_tick_labels(
            axis,
            [ra, ild],
            ["RA", "RA-ILD"],
            showmeans=True,
        )
        for index, values in enumerate((ra, ild), start=1):
            jitter = rng.normal(0.0, float(section.get("jitter_width", 0.05)), size=len(values))
            axis.scatter(np.full(len(values), index) + jitter, values, s=float(section.get("point_size", 14)), alpha=float(section.get("point_alpha", 0.45)))
        threshold_values = group["threshold"].astype(float).unique()
        if len(threshold_values) != 1:
            raise VisualizationError("Figure 4 group contains multiple thresholds.")
        threshold = float(threshold_values[0])
        if bool(section.get("show_threshold", True)):
            axis.axhline(threshold, linestyle="--", linewidth=1.2, label=f"Saved threshold={threshold:.3f}")
            axis.legend(fontsize=8)
        axis.set(ylabel="Predicted probability of RA-ILD", ylim=(-0.02, 1.02), title=f"{_display_label(item.model, config)}\nrepeat {item.outer_repeat}")
        axis.grid(axis="y", alpha=0.25)
        part = group.copy()
        part["selection_source"] = "figure3" if reuse else "figure4"
        output_parts.append(part)
    _save_pdf(fig, output_path, int(config.style.get("dpi", 150)))
    return selection, pd.concat(output_parts, ignore_index=True)


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, compression="gzip" if path.suffix == ".gz" else None)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _existing_output_guard(output_dir: Path, overwrite: bool) -> None:
    existing = [output_dir / name for name in OUTPUT_FILES.values() if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Visualization outputs already exist; use --overwrite only for a documented rerun:\n"
            + "\n".join(f"  - {path}" for path in existing)
        )


def generate_visualizations(
    config: VisualizationConfig,
    *,
    overwrite_override: Optional[bool] = None,
) -> VisualizationResult:
    results = load_aggregated_results(config)
    overwrite = config.overwrite if overwrite_override is None else bool(overwrite_override)
    _existing_output_guard(config.output_dir, overwrite)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: config.output_dir / name for key, name in OUTPUT_FILES.items()}

    audit = reproduce_metric_audit(results, config.metric_tolerance)
    _write_frame(audit, paths["input_audit"])
    _atomic_write_text(
        paths["resolved_config"],
        yaml.safe_dump(dict(config.raw), allow_unicode=True, sort_keys=False),
    )
    generated: Dict[str, Path] = {
        "input_audit": paths["input_audit"],
        "resolved_config": paths["resolved_config"],
    }

    figure3_selection: Optional[pd.DataFrame] = None
    if bool(config.figures["figure1"].get("enabled", False)):
        data = _figure1(config, results, paths["figure1"])
        _write_frame(data, paths["figure1_data"])
        generated.update(figure1=paths["figure1"], figure1_data=paths["figure1_data"])
    if bool(config.figures["figure2"].get("enabled", False)):
        coordinates, repeat_auc = _figure2(config, results, paths["figure2"])
        _write_frame(coordinates, paths["figure2_coordinates"])
        _write_frame(repeat_auc, paths["figure2_repeat_auc"])
        generated.update(
            figure2=paths["figure2"],
            figure2_coordinates=paths["figure2_coordinates"],
            figure2_repeat_auc=paths["figure2_repeat_auc"],
        )
    if bool(config.figures["figure3"].get("enabled", False)):
        figure3_selection, coordinates = _figure3(config, results, paths["figure3"])
        _write_frame(figure3_selection, paths["figure3_selection"])
        _write_frame(coordinates, paths["figure3_coordinates"])
        generated.update(
            figure3=paths["figure3"],
            figure3_selection=paths["figure3_selection"],
            figure3_coordinates=paths["figure3_coordinates"],
        )
    selected_best = figure3_selection if figure3_selection is not None else pd.DataFrame()
    if bool(config.figures["figure4"].get("enabled", False)):
        figure4_selection, data = _figure4(
            config,
            results,
            paths["figure4"],
            figure3_selection,
        )
        _write_frame(data, paths["figure4_data"])
        generated.update(figure4=paths["figure4"], figure4_data=paths["figure4_data"])
        if selected_best.empty:
            selected_best = figure4_selection

    manifest_outputs = {
        key: {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for key, path in generated.items()
    }
    manifest = {
        "visualization_version": VISUALIZATION_VERSION,
        "visualization_id": config.visualization_id,
        "source_mode": "native_repeated_holdout",
        "aggregation_dir": str(config.aggregation_dir),
        "output_dir": str(config.output_dir),
        "input_files": {
            key: {"path": str(path), "sha256": results.input_sha256[key]}
            for key, path in results.input_paths.items()
        },
        "generated_outputs": manifest_outputs,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "scikit_learn": sklearn.__version__,
            "pyyaml": yaml.__version__,
        },
        "scientific_contract": {
            "model_refit_performed": False,
            "hyperparameter_retuning_performed": False,
            "holdout_threshold_optimization_performed": False,
            "cross_repeat_n_by_n_read": False,
            "independent_test_read": False,
            "best_repeat_figures_are_descriptive": True,
            "variability_band_is_not_independent_sample_ci": True,
        },
    }
    _atomic_write_text(paths["manifest"], json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    generated["manifest"] = paths["manifest"]
    complete = {
        "status": "COMPLETE",
        "visualization_version": VISUALIZATION_VERSION,
        "visualization_id": config.visualization_id,
        "source_mode": "native_repeated_holdout",
        "output_dir": str(config.output_dir),
        "manifest": str(paths["manifest"]),
        "model_refit_performed": False,
        "independent_test_read": False,
    }
    _atomic_write_text(paths["complete"], json.dumps(complete, ensure_ascii=False, indent=2) + "\n")
    generated["complete"] = paths["complete"]
    return VisualizationResult(
        output_dir=config.output_dir,
        generated_files=dict(generated),
        selected_best_repeats=selected_best,
    )
