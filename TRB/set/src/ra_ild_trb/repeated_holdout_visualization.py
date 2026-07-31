#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configurable visualization of frozen repeated-holdout TRB results.

This module is deliberately read-only with respect to model training outputs.
It consumes the native repeated-holdout aggregation files written by
``repeated_holdout_summary.py`` and produces publication-ready PDF figures plus
plot-data/audit files. It never refits a model, recalculates a threshold for
selection, or reads the cross-repeat n x n evaluation by default.
"""
from __future__ import annotations

import hashlib
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
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class VisualizationError(ValueError):
    """Raised when visualization inputs violate the frozen-result contract."""


VISUALIZATION_VERSION = "1.0.1"

AGGREGATION_FILES: Mapping[str, str] = {
    "metrics": "08_holdout_task_metrics.csv",
    "predictions": "08_holdout_predictions.csv.gz",
    "summary": "08_aggregation_summary.json",
    "complete": "08_REPEATED_HOLDOUT_AGGREGATION_COMPLETE.json",
}

METRIC_LABELS: Mapping[str, str] = {
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC (Average Precision)",
    "f1": "F1 score",
    "sensitivity_recall": "Sensitivity",
    "specificity": "Specificity",
    "accuracy": "Accuracy",
    "precision": "Precision",
}

ALLOWED_METRICS: Tuple[str, ...] = tuple(METRIC_LABELS)
IDENTIFIER_SORT_ASCENDING = {
    "outer_repeat",
    "outer_fold",
    "task_index",
    "task_id",
    "split_id",
}

DEFAULT_FIGURE_FILES: Mapping[str, str] = {
    "figure1": "Figure1_model_performance_boxplots.pdf",
    "figure2": "Figure2_mean_ROC_curves.pdf",
    "figure3": "Figure3_best_repeat_ROC_curves.pdf",
    "figure4": "Figure4_best_repeat_prediction_boxplots.pdf",
}

OUTPUT_AUDIT_FILES: Mapping[str, str] = {
    "figure1_data": "Figure1_plot_data.csv",
    "figure2_coordinates": "Figure2_mean_ROC_coordinates.csv.gz",
    "figure2_repeat_auc": "Figure2_repeat_ROC_AUC.csv",
    "figure3_selection": "Figure3_best_repeat_selection.csv",
    "figure3_coordinates": "Figure3_best_repeat_ROC_coordinates.csv.gz",
    "figure4_data": "Figure4_plot_data.csv",
    "input_audit": "visualization_input_audit.csv",
    "resolved_config": "visualization_resolved_config.yaml",
    "manifest": "visualization_manifest.json",
    "complete": "VISUALIZATION_COMPLETE.json",
}


@dataclass(frozen=True)
class VisualizationConfig:
    source_path: Path
    repository_root: Path
    visualization_id: str
    aggregation_dir: Path
    output_dir: Path
    require_complete_marker: bool
    overwrite: bool
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
    """Integrate a curve across supported NumPy versions.

    NumPy 2.4+ removed the long-deprecated ``np.trapz`` alias.  New NumPy
    versions expose ``np.trapezoid`` while older validated environments may
    still expose only ``np.trapz``.  Resolve the available implementation at
    runtime so the visualization layer remains compatible with both families.
    """

    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # NumPy < 2.0 compatibility
        integrate = getattr(np, "trapz", None)
    if integrate is None:  # pragma: no cover - defensive future-proofing
        raise VisualizationError(
            "This NumPy version provides neither numpy.trapezoid nor numpy.trapz."
        )
    return float(integrate(np.asarray(y, dtype=float), np.asarray(x, dtype=float)))


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - error text is the contract
        raise VisualizationError(f"Unable to read {label}: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise VisualizationError(f"{label} must contain a JSON object: {path}")
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VisualizationError(f"{label} must be a mapping.")
    return value


def _require_sequence(value: Any, label: str) -> Tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise VisualizationError(f"{label} must be a non-empty list.")
    items = tuple(str(item) for item in value)
    if not items or len(items) != len(set(items)):
        raise VisualizationError(f"{label} must be non-empty and contain unique values.")
    return items


def _resolve_path(value: Any, repository_root: Path, label: str) -> Path:
    if value is None or not str(value).strip():
        raise VisualizationError(f"{label} is required.")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = repository_root / path
    return path.resolve()


def load_visualization_config(
    path: Path,
    *,
    repository_root: Optional[Path] = None,
) -> VisualizationConfig:
    """Load and validate one visualization YAML configuration."""
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Visualization config not found: {source_path}")
    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise VisualizationError(f"Unable to parse YAML config: {source_path}: {exc}") from exc
    raw = _require_mapping(raw, "visualization config")
    if str(raw.get("visualization_version")) != "1.0":
        raise VisualizationError("visualization_version must be '1.0'.")

    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root is not None
        else Path.cwd().resolve()
    )
    visual = _require_mapping(raw.get("visualization"), "visualization")
    visualization_id = str(visual.get("id", "")).strip()
    if not visualization_id:
        raise VisualizationError("visualization.id is required.")
    aggregation_dir = _resolve_path(
        visual.get("aggregation_dir"), root, "visualization.aggregation_dir"
    )
    output_dir = _resolve_path(
        visual.get("output_dir"), root, "visualization.output_dir"
    )
    require_complete_marker = bool(visual.get("require_complete_marker", True))
    overwrite = bool(visual.get("overwrite", False))
    source_mode = str(visual.get("source_mode", "native_repeated_holdout"))
    if source_mode != "native_repeated_holdout":
        raise VisualizationError(
            "visualization.source_mode must be native_repeated_holdout; "
            "cross-repeat n x n results are intentionally excluded from this module."
        )

    labels_raw = raw.get("model_labels", {})
    model_labels = {
        str(key): str(value) for key, value in _require_mapping(labels_raw, "model_labels").items()
    }
    figures = _require_mapping(raw.get("figures"), "figures")
    known = {"figure1", "figure2", "figure3", "figure4"}
    unknown = sorted(set(figures) - known)
    if unknown:
        raise VisualizationError(f"Unknown figure sections: {unknown}")
    enabled = []
    normalized_figures: Dict[str, Mapping[str, Any]] = {}
    for name in ("figure1", "figure2", "figure3", "figure4"):
        section = _require_mapping(figures.get(name, {"enabled": False}), name)
        normalized_figures[name] = dict(section)
        if bool(section.get("enabled", False)):
            enabled.append(name)
            model_selector = section.get("models")
            if isinstance(model_selector, str):
                if model_selector.strip().lower() != "all":
                    raise VisualizationError(
                        f"figures.{name}.models must be 'all' or a non-empty unique list."
                    )
            else:
                _require_sequence(model_selector, f"figures.{name}.models")
    if not enabled:
        raise VisualizationError("At least one figure must be enabled.")

    style_raw = raw.get("style", {})
    style = dict(_require_mapping(style_raw, "style"))
    return VisualizationConfig(
        source_path=source_path,
        repository_root=root,
        visualization_id=visualization_id,
        aggregation_dir=aggregation_dir,
        output_dir=output_dir,
        require_complete_marker=require_complete_marker,
        overwrite=overwrite,
        model_labels=model_labels,
        figures=normalized_figures,
        style=style,
        raw=raw,
    )


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise VisualizationError(f"{label} is missing columns: {missing}")


def _numeric(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    for column in columns:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except Exception as exc:
            raise VisualizationError(f"{label}.{column} must be numeric: {exc}") from exc


def _validate_binary_rows(predictions: pd.DataFrame) -> None:
    for (model, split_id), group in predictions.groupby(["model", "split_id"], sort=False):
        labels = set(group["true_label"].astype(int).unique().tolist())
        if labels != {0, 1}:
            raise VisualizationError(
                f"Predictions for {model}/{split_id} must contain both true classes; observed={sorted(labels)}"
            )
        if group["sample_id"].astype(str).duplicated().any():
            raise VisualizationError(f"Duplicate sample prediction for {model}/{split_id}")
        thresholds = np.unique(group["threshold"].to_numpy(float))
        if len(thresholds) != 1:
            raise VisualizationError(f"Predictions for {model}/{split_id} contain multiple thresholds")


def _specificity(y_true: np.ndarray, predicted: np.ndarray) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) else float("nan")


def _prediction_metric_row(group: pd.DataFrame) -> Dict[str, float]:
    y = group["true_label"].to_numpy(int)
    probability = group["probability_ILD"].to_numpy(float)
    threshold = float(group["threshold"].iloc[0])
    predicted = (probability >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "sensitivity_recall": float(recall_score(y, predicted, zero_division=0)),
        "specificity": _specificity(y, predicted),
    }


def _audit_metrics_against_predictions(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    tolerance: float = 1.0e-6,
) -> pd.DataFrame:
    metric_index = metrics.set_index(["model", "split_id"], verify_integrity=True)
    rows: List[Dict[str, Any]] = []
    for (model, split_id), group in predictions.groupby(["model", "split_id"], sort=False):
        key = (str(model), str(split_id))
        if key not in metric_index.index:
            raise VisualizationError(f"Prediction pair missing from metrics: {key}")
        observed = _prediction_metric_row(group)
        metric_row = metric_index.loc[key]
        threshold_metric = float(metric_row["threshold"])
        threshold_prediction = float(group["threshold"].iloc[0])
        threshold_difference = abs(threshold_metric - threshold_prediction)
        maximum_difference = threshold_difference
        row: Dict[str, Any] = {
            "model": str(model),
            "split_id": str(split_id),
            "n_predictions": int(len(group)),
            "threshold_metric": threshold_metric,
            "threshold_prediction": threshold_prediction,
            "threshold_abs_difference": threshold_difference,
        }
        for metric_name, recalculated in observed.items():
            saved = float(metric_row[metric_name])
            difference = abs(saved - recalculated)
            maximum_difference = max(maximum_difference, difference)
            row[f"saved_{metric_name}"] = saved
            row[f"recalculated_{metric_name}"] = recalculated
            row[f"abs_difference_{metric_name}"] = difference
        row["maximum_abs_difference"] = maximum_difference
        row["audit_pass"] = bool(maximum_difference <= tolerance)
        rows.append(row)
    audit = pd.DataFrame(rows)
    if len(audit) != len(metrics):
        raise VisualizationError(
            f"Metrics/predictions pair count mismatch: metrics={len(metrics)}, audited={len(audit)}"
        )
    failed = audit.loc[~audit["audit_pass"]]
    if not failed.empty:
        example = failed.iloc[0]
        raise VisualizationError(
            "Saved metrics do not reproduce from saved predictions within "
            f"tolerance={tolerance}: {example['model']}/{example['split_id']} "
            f"maximum_abs_difference={example['maximum_abs_difference']}"
        )
    return audit.sort_values(["model", "split_id"], kind="stable").reset_index(drop=True)


def load_aggregated_results(config: VisualizationConfig) -> Tuple[AggregatedResults, pd.DataFrame]:
    """Load, validate, and audit native repeated-holdout aggregate files."""
    paths = {
        key: config.aggregation_dir / filename
        for key, filename in AGGREGATION_FILES.items()
    }
    mandatory = [paths["metrics"], paths["predictions"]]
    if config.require_complete_marker:
        mandatory.extend([paths["summary"], paths["complete"]])
    missing = [str(path) for path in mandatory if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required aggregation files are missing:\n  - " + "\n  - ".join(missing))

    summary: Mapping[str, Any] = {}
    marker: Mapping[str, Any] = {}
    if paths["summary"].is_file():
        summary = _read_json(paths["summary"], "aggregation summary")
    if paths["complete"].is_file():
        marker = _read_json(paths["complete"], "aggregation completion marker")
    if config.require_complete_marker:
        status_candidates = [marker.get("status"), marker.get("aggregation_status")]
        if not any(str(value).upper() == "COMPLETE" for value in status_candidates):
            raise VisualizationError(
                f"Aggregation completion marker is not COMPLETE: {paths['complete']}"
            )

    metrics = pd.read_csv(paths["metrics"])
    predictions = pd.read_csv(paths["predictions"])
    _require_columns(
        metrics,
        (
            "model", "split_id", "outer_repeat", "roc_auc", "pr_auc", "f1",
            "sensitivity_recall", "specificity", "threshold",
        ),
        "holdout metrics",
    )
    _require_columns(
        predictions,
        (
            "model", "split_id", "outer_repeat", "sample_id", "true_label",
            "true_cohort", "probability_ILD", "threshold", "predicted_label",
        ),
        "holdout predictions",
    )
    metrics = metrics.copy()
    predictions = predictions.copy()
    for frame in (metrics, predictions):
        frame["model"] = frame["model"].astype(str)
        frame["split_id"] = frame["split_id"].astype(str)
    predictions["sample_id"] = predictions["sample_id"].astype(str)
    _numeric(metrics, (*ALLOWED_METRICS, "threshold", "outer_repeat"), "holdout metrics")
    _numeric(
        predictions,
        ("outer_repeat", "true_label", "probability_ILD", "threshold", "predicted_label"),
        "holdout predictions",
    )

    if metrics.duplicated(["model", "split_id"]).any():
        raise VisualizationError("Holdout metrics contain duplicate model/split_id rows.")
    if not np.isfinite(metrics[list(ALLOWED_METRICS) + ["threshold"]].to_numpy(float)).all():
        raise VisualizationError("Holdout metrics contain non-finite values.")
    for column in (*ALLOWED_METRICS, "threshold"):
        values = metrics[column].to_numpy(float)
        if np.any((values < 0) | (values > 1)):
            raise VisualizationError(f"Holdout metrics.{column} contains values outside [0, 1].")
    probability = predictions["probability_ILD"].to_numpy(float)
    threshold = predictions["threshold"].to_numpy(float)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise VisualizationError("probability_ILD must contain finite values in [0, 1].")
    if not np.isfinite(threshold).all() or np.any((threshold < 0) | (threshold > 1)):
        raise VisualizationError("Prediction thresholds must contain finite values in [0, 1].")
    if set(predictions["true_label"].astype(int).unique()) - {0, 1}:
        raise VisualizationError("true_label must contain only 0 and 1.")
    if set(predictions["predicted_label"].astype(int).unique()) - {0, 1}:
        raise VisualizationError("predicted_label must contain only 0 and 1.")

    metric_pairs = set(zip(metrics["model"], metrics["split_id"]))
    prediction_pairs = set(zip(predictions["model"], predictions["split_id"]))
    if metric_pairs != prediction_pairs:
        missing_metrics = sorted(prediction_pairs - metric_pairs)[:5]
        missing_predictions = sorted(metric_pairs - prediction_pairs)[:5]
        raise VisualizationError(
            "Metrics/predictions model-split pairs differ. "
            f"Missing metrics examples={missing_metrics}; missing predictions examples={missing_predictions}"
        )
    _validate_binary_rows(predictions)

    # Outer-repeat identity must agree between metric and prediction files.
    repeat_metric = metrics.set_index(["model", "split_id"])["outer_repeat"].astype(int)
    repeat_prediction = predictions.groupby(["model", "split_id"])["outer_repeat"].nunique()
    if (repeat_prediction != 1).any():
        raise VisualizationError("A model/split prediction group contains multiple outer_repeat values.")
    observed_repeat = predictions.groupby(["model", "split_id"])["outer_repeat"].first().astype(int)
    if not repeat_metric.sort_index().equals(observed_repeat.sort_index()):
        raise VisualizationError("outer_repeat does not agree between metrics and predictions.")

    audit = _audit_metrics_against_predictions(metrics, predictions)
    input_sha = {
        key: sha256_file(path)
        for key, path in paths.items()
        if path.is_file()
    }
    return (
        AggregatedResults(
            metrics=metrics.sort_values(["model", "outer_repeat", "split_id"], kind="stable").reset_index(drop=True),
            predictions=predictions.sort_values(["model", "outer_repeat", "split_id", "sample_id"], kind="stable").reset_index(drop=True),
            summary=summary,
            complete_marker=marker,
            input_paths=paths,
            input_sha256=input_sha,
        ),
        audit,
    )


def _enabled(config: VisualizationConfig, name: str) -> bool:
    return bool(config.figures[name].get("enabled", False))


def _available_models(results: AggregatedResults) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(results.metrics["model"].astype(str).tolist()))


def _requested_models(
    config: VisualizationConfig,
    name: str,
    results: AggregatedResults,
) -> Tuple[str, ...]:
    selector = config.figures[name].get("models")
    if isinstance(selector, str) and selector.strip().lower() == "all":
        return _available_models(results)
    return _require_sequence(selector, f"figures.{name}.models")


def _all_requested_models(
    config: VisualizationConfig,
    results: AggregatedResults,
) -> Tuple[str, ...]:
    ordered: List[str] = []
    for name in ("figure1", "figure2", "figure3", "figure4"):
        if _enabled(config, name):
            for model in _requested_models(config, name, results):
                if model not in ordered:
                    ordered.append(model)
    return tuple(ordered)


def validate_requested_models(config: VisualizationConfig, results: AggregatedResults) -> None:
    available = set(_available_models(results))
    requested = set(_all_requested_models(config, results))
    missing = sorted(requested - available)
    if missing:
        raise VisualizationError(
            f"Requested models are absent from aggregation results: {missing}. "
            f"Available models: {sorted(available)}"
        )


def _model_label(config: VisualizationConfig, model: str) -> str:
    return config.model_labels.get(str(model), str(model))


def _max_columns(section: Mapping[str, Any]) -> int:
    value = int(section.get("max_columns", 3))
    if value < 1 or value > 3:
        raise VisualizationError("max_columns must be between 1 and 3.")
    return value


def _subplot_geometry(n: int, max_columns: int) -> Tuple[Figure, List[Axes]]:
    if n < 1:
        raise VisualizationError("Cannot create a figure with zero panels.")
    columns = min(max_columns, n)
    rows = int(math.ceil(n / max_columns))
    width = {1: 7.0, 2: 11.0, 3: 15.0}[columns]
    height = max(4.8, 4.3 * rows)
    figure = plt.figure(figsize=(width, height), constrained_layout=True)
    units = max_columns * 2
    grid = figure.add_gridspec(rows, units)
    axes: List[Axes] = []
    full_rows = n // max_columns
    remainder = n % max_columns
    index = 0
    for row in range(rows):
        count = max_columns if row < full_rows else (remainder or max_columns)
        if count == 1 and n == 1:
            spans = [(0, units)]
        elif count == 1:
            spans = [((units - 2) // 2, (units - 2) // 2 + 2)]
        elif count == 2:
            spans = [(0, units // 2), (units // 2, units)]
        else:
            width_units = units // count
            spans = [
                (column * width_units, (column + 1) * width_units)
                for column in range(count)
            ]
        for start, stop in spans:
            if index >= n:
                break
            axes.append(figure.add_subplot(grid[row, start:stop]))
            index += 1
    return figure, axes


def _panel_label(index: int) -> str:
    # A..Z, AA..AZ is more than enough for these figures.
    result = ""
    value = int(index)
    while True:
        result = chr(ord("A") + (value % 26)) + result
        value = value // 26 - 1
        if value < 0:
            return result


def _apply_axis_style(axis: Axes) -> None:
    axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _save_pdf(figure: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)


def _deterministic_jitter(n: int, width: float = 0.16) -> np.ndarray:
    if n <= 1:
        return np.zeros(n, dtype=float)
    # Symmetric deterministic positions avoid random visual drift between reruns.
    return np.linspace(-width, width, n)


def _figure_output_path(config: VisualizationConfig, name: str) -> Path:
    section = config.figures[name]
    file_name = str(section.get("output_file", DEFAULT_FIGURE_FILES[name])).strip()
    if not file_name.lower().endswith(".pdf"):
        raise VisualizationError(f"figures.{name}.output_file must end with .pdf")
    if Path(file_name).name != file_name:
        raise VisualizationError(f"figures.{name}.output_file must be a file name, not a path")
    return config.output_dir / file_name


def plot_figure1(
    config: VisualizationConfig,
    results: AggregatedResults,
    output_path: Path,
) -> pd.DataFrame:
    """Plot one metric-comparison panel per selected metric in one PDF."""
    section = config.figures["figure1"]
    models = _requested_models(config, "figure1", results)
    metrics_requested = _require_sequence(section.get("metrics"), "figures.figure1.metrics")
    unknown = sorted(set(metrics_requested) - set(ALLOWED_METRICS))
    if unknown:
        raise VisualizationError(f"Unsupported Figure 1 metrics: {unknown}")
    show_points = bool(section.get("show_repeat_points", True))
    show_mean = bool(section.get("show_mean", True))
    y_limits = section.get("y_limits", [0.0, 1.0])
    if not isinstance(y_limits, Sequence) or len(y_limits) != 2:
        raise VisualizationError("figures.figure1.y_limits must contain [lower, upper].")
    y_lower, y_upper = map(float, y_limits)
    if not (y_lower < y_upper):
        raise VisualizationError("figures.figure1.y_limits must be increasing.")

    plot_data = results.metrics.loc[
        results.metrics["model"].isin(models),
        ["model", "split_id", "outer_repeat", *metrics_requested],
    ].copy()
    long = plot_data.melt(
        id_vars=["model", "split_id", "outer_repeat"],
        value_vars=list(metrics_requested),
        var_name="metric",
        value_name="value",
    )
    long["model_label"] = long["model"].map(lambda value: _model_label(config, value))
    long["metric_label"] = long["metric"].map(METRIC_LABELS)

    figure, axes = _subplot_geometry(len(metrics_requested), _max_columns(section))
    figure.suptitle("Repeated-holdout model performance", fontsize=15, fontweight="bold")
    palette = plt.get_cmap("tab10")
    for panel_index, (axis, metric_name) in enumerate(zip(axes, metrics_requested)):
        values_by_model = [
            long.loc[(long["metric"] == metric_name) & (long["model"] == model), "value"].to_numpy(float)
            for model in models
        ]
        box = axis.boxplot(
            values_by_model,
            positions=np.arange(1, len(models) + 1),
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
        )
        for model_index, patch in enumerate(box["boxes"]):
            patch.set_facecolor(palette(model_index % 10))
            patch.set_alpha(0.55)
        if show_points:
            for model_index, values in enumerate(values_by_model, start=1):
                jitter = _deterministic_jitter(len(values))
                axis.scatter(
                    np.full(len(values), model_index, dtype=float) + jitter,
                    values,
                    s=13,
                    alpha=0.45,
                    color=palette((model_index - 1) % 10),
                    edgecolors="none",
                    zorder=3,
                )
        if show_mean:
            means = [float(np.mean(values)) for values in values_by_model]
            axis.scatter(
                np.arange(1, len(models) + 1),
                means,
                marker="D",
                s=32,
                color="black",
                label="Mean",
                zorder=4,
            )
            if panel_index == 0:
                axis.legend(frameon=False, loc="lower right")
        display_labels = [_model_label(config, model) for model in models]
        rotate_labels = len(models) > 3 or max(map(len, display_labels), default=0) > 14
        axis.set_xticks(
            np.arange(1, len(models) + 1),
            display_labels,
            rotation=25 if rotate_labels else 0,
            ha="right" if rotate_labels else "center",
        )
        axis.set_ylabel(METRIC_LABELS[metric_name])
        axis.set_ylim(y_lower, y_upper)
        axis.set_title(f"{_panel_label(panel_index)}. {METRIC_LABELS[metric_name]}", loc="left", fontweight="bold")
        _apply_axis_style(axis)
    _save_pdf(figure, output_path)
    return long.sort_values(["metric", "model", "outer_repeat"], kind="stable").reset_index(drop=True)


def _roc_groups(predictions: pd.DataFrame, model: str) -> Iterable[Tuple[str, pd.DataFrame]]:
    model_rows = predictions.loc[predictions["model"] == model]
    for split_id, group in model_rows.groupby("split_id", sort=False):
        yield str(split_id), group.sort_values("sample_id", kind="stable")


def plot_figure2(
    config: VisualizationConfig,
    results: AggregatedResults,
    output_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Plot macro-averaged ROC curves across repeats with variability bands."""
    section = config.figures["figure2"]
    models = _requested_models(config, "figure2", results)
    points = int(section.get("fpr_grid_points", 201))
    if points < 21 or points > 5001:
        raise VisualizationError("figures.figure2.fpr_grid_points must be between 21 and 5001.")
    quantiles = section.get("variability_quantiles", [0.025, 0.975])
    if not isinstance(quantiles, Sequence) or len(quantiles) != 2:
        raise VisualizationError("figure2.variability_quantiles must contain two values.")
    q_lower, q_upper = map(float, quantiles)
    if not (0.0 <= q_lower < q_upper <= 1.0):
        raise VisualizationError("figure2.variability_quantiles must satisfy 0 <= lower < upper <= 1.")
    show_band = bool(section.get("show_repeat_variability_band", True))
    fpr_grid = np.linspace(0.0, 1.0, points)

    figure, axes = _subplot_geometry(len(models), _max_columns(section))
    figure.suptitle("Mean ROC curves across repeated holdouts", fontsize=15, fontweight="bold")
    coordinate_rows: List[Dict[str, Any]] = []
    repeat_rows: List[Dict[str, Any]] = []
    palette = plt.get_cmap("tab10")
    metric_index = results.metrics.set_index(["model", "split_id"])
    for panel_index, (axis, model) in enumerate(zip(axes, models)):
        interpolated: List[np.ndarray] = []
        aucs: List[float] = []
        for split_id, group in _roc_groups(results.predictions, model):
            y = group["true_label"].to_numpy(int)
            probability = group["probability_ILD"].to_numpy(float)
            fpr, tpr, _ = roc_curve(y, probability)
            interp = np.interp(fpr_grid, fpr, tpr)
            interp[0] = 0.0
            interp[-1] = 1.0
            auc_value = float(roc_auc_score(y, probability))
            saved_auc = float(metric_index.loc[(model, split_id), "roc_auc"])
            repeat = int(metric_index.loc[(model, split_id), "outer_repeat"])
            interpolated.append(interp)
            aucs.append(auc_value)
            repeat_rows.append(
                {
                    "model": model,
                    "model_label": _model_label(config, model),
                    "split_id": split_id,
                    "outer_repeat": repeat,
                    "roc_auc": auc_value,
                    "saved_roc_auc": saved_auc,
                    "n_test": int(len(group)),
                }
            )
        matrix = np.vstack(interpolated)
        mean_tpr = np.mean(matrix, axis=0)
        lower_tpr = np.quantile(matrix, q_lower, axis=0)
        upper_tpr = np.quantile(matrix, q_upper, axis=0)
        mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
        lower_tpr[0], lower_tpr[-1] = 0.0, 1.0
        upper_tpr[0], upper_tpr[-1] = 0.0, 1.0
        mean_auc = float(np.mean(aucs))
        sd_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
        auc_of_mean_curve = _trapezoid_area(mean_tpr, fpr_grid)
        for fpr_value, mean_value, lower_value, upper_value in zip(
            fpr_grid, mean_tpr, lower_tpr, upper_tpr
        ):
            coordinate_rows.append(
                {
                    "model": model,
                    "model_label": _model_label(config, model),
                    "fpr": float(fpr_value),
                    "mean_tpr": float(mean_value),
                    "lower_tpr": float(lower_value),
                    "upper_tpr": float(upper_value),
                    "lower_quantile": q_lower,
                    "upper_quantile": q_upper,
                    "n_repeats": int(len(aucs)),
                    "mean_repeat_roc_auc": mean_auc,
                    "sd_repeat_roc_auc": sd_auc,
                    "auc_of_mean_curve": auc_of_mean_curve,
                }
            )
        color = palette(panel_index % 10)
        if show_band:
            axis.fill_between(
                fpr_grid,
                lower_tpr,
                upper_tpr,
                color=color,
                alpha=0.18,
                label=f"{q_lower:.1%}-{q_upper:.1%} repeat band",
            )
        axis.plot(
            fpr_grid,
            mean_tpr,
            color=color,
            linewidth=2.2,
            label=f"Mean ROC-AUC = {mean_auc:.3f} ± {sd_auc:.3f}",
        )
        axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="gray")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.02)
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.set_title(f"{_panel_label(panel_index)}. {_model_label(config, model)}", loc="left", fontweight="bold")
        axis.legend(frameon=False, loc="lower right", fontsize=9)
        _apply_axis_style(axis)
    _save_pdf(figure, output_path)
    return pd.DataFrame(coordinate_rows), pd.DataFrame(repeat_rows)


def select_best_repeats(
    config: VisualizationConfig,
    results: AggregatedResults,
    models: Sequence[str],
    selection: Mapping[str, Any],
) -> pd.DataFrame:
    """Select one best observed native holdout per model deterministically."""
    mode = str(selection.get("mode", "best"))
    if mode != "best":
        raise VisualizationError("Only repeat_selection.mode=best is supported.")
    primary = str(selection.get("primary_metric", "roc_auc"))
    if primary not in ALLOWED_METRICS:
        raise VisualizationError(f"Unsupported best-repeat primary metric: {primary}")
    tie_breakers_raw = selection.get("tie_breakers", ["pr_auc", "f1", "outer_repeat"])
    tie_breakers = _require_sequence(tie_breakers_raw, "repeat_selection.tie_breakers")
    unknown = sorted(set(tie_breakers) - set(results.metrics.columns))
    if unknown:
        raise VisualizationError(f"Unknown best-repeat tie-breaker columns: {unknown}")
    sort_columns = [primary] + [name for name in tie_breakers if name != primary]
    ascending = [column in IDENTIFIER_SORT_ASCENDING for column in sort_columns]
    rows: List[pd.Series] = []
    for model in models:
        candidates = results.metrics.loc[results.metrics["model"] == model].copy()
        candidates = candidates.sort_values(sort_columns, ascending=ascending, kind="stable")
        best = candidates.iloc[0].copy()
        best["selection_primary_metric"] = primary
        best["selection_rule"] = " > ".join(
            f"{column} {'ascending' if direction else 'descending'}"
            for column, direction in zip(sort_columns, ascending)
        )
        rows.append(best)
    selected = pd.DataFrame(rows).reset_index(drop=True)
    columns = [
        "model", "split_id", "outer_repeat", "outer_fold", "task_id", "task_index",
        "roc_auc", "pr_auc", "f1", "sensitivity_recall", "specificity",
        "threshold", "selection_primary_metric", "selection_rule",
    ]
    available = [column for column in columns if column in selected.columns]
    selected = selected.loc[:, available]
    selected.insert(1, "model_label", selected["model"].map(lambda value: _model_label(config, value)))
    selected["best_repeat_is_descriptive_only"] = True
    return selected


def _selection_for_figure3(
    config: VisualizationConfig,
    results: AggregatedResults,
) -> pd.DataFrame:
    section = config.figures["figure3"]
    selection = _require_mapping(section.get("repeat_selection", {}), "figure3.repeat_selection")
    return select_best_repeats(
        config,
        results,
        _requested_models(config, "figure3", results),
        selection,
    )


def plot_figure3(
    config: VisualizationConfig,
    results: AggregatedResults,
    output_path: Path,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    """Plot ROC curves from the best observed native holdout for each model."""
    section = config.figures["figure3"]
    models = _requested_models(config, "figure3", results)
    mark_threshold = bool(section.get("mark_saved_threshold_operating_point", True))
    selected_index = selected.set_index("model")
    figure, axes = _subplot_geometry(len(models), _max_columns(section))
    figure.suptitle("Best observed holdout ROC curves", fontsize=15, fontweight="bold")
    rows: List[Dict[str, Any]] = []
    palette = plt.get_cmap("tab10")
    for panel_index, (axis, model) in enumerate(zip(axes, models)):
        row = selected_index.loc[model]
        split_id = str(row["split_id"])
        group = results.predictions.loc[
            (results.predictions["model"] == model)
            & (results.predictions["split_id"] == split_id)
        ].copy()
        y = group["true_label"].to_numpy(int)
        probability = group["probability_ILD"].to_numpy(float)
        fpr, tpr, thresholds = roc_curve(y, probability)
        auc_value = float(roc_auc_score(y, probability))
        ap_value = float(average_precision_score(y, probability))
        saved_threshold = float(row["threshold"])
        predicted = (probability >= saved_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        operating_fpr = float(fp / (fp + tn)) if (fp + tn) else float("nan")
        operating_tpr = float(tp / (tp + fn)) if (tp + fn) else float("nan")
        for fpr_value, tpr_value, roc_threshold in zip(fpr, tpr, thresholds):
            rows.append(
                {
                    "model": model,
                    "model_label": _model_label(config, model),
                    "split_id": split_id,
                    "outer_repeat": int(row["outer_repeat"]),
                    "fpr": float(fpr_value),
                    "tpr": float(tpr_value),
                    "roc_threshold": float(roc_threshold) if np.isfinite(roc_threshold) else np.inf,
                    "saved_classification_threshold": saved_threshold,
                    "saved_threshold_fpr": operating_fpr,
                    "saved_threshold_tpr": operating_tpr,
                    "roc_auc": auc_value,
                    "pr_auc": ap_value,
                    "n_test": int(len(group)),
                    "n_RA": int(np.sum(y == 0)),
                    "n_ILD": int(np.sum(y == 1)),
                }
            )
        color = palette(panel_index % 10)
        axis.plot(fpr, tpr, color=color, linewidth=2.2, label=f"ROC-AUC = {auc_value:.3f}")
        axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="gray")
        if mark_threshold:
            axis.scatter(
                [operating_fpr],
                [operating_tpr],
                color="black",
                s=38,
                marker="o",
                zorder=4,
                label=f"Saved threshold = {saved_threshold:.3f}",
                clip_on=False,
            )
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.02)
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.set_title(f"{_panel_label(panel_index)}. {_model_label(config, model)}", loc="left", fontweight="bold")
        axis.text(
            0.04,
            0.96,
            f"Repeat {int(row['outer_repeat'])}\nPR-AUC = {ap_value:.3f}\nn = {len(group)}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=9,
        )
        axis.legend(frameon=False, loc="lower right", fontsize=9)
        _apply_axis_style(axis)
    _save_pdf(figure, output_path)
    return pd.DataFrame(rows)


def _figure4_selection(
    config: VisualizationConfig,
    results: AggregatedResults,
    figure3_selection: Optional[pd.DataFrame],
) -> pd.DataFrame:
    section = config.figures["figure4"]
    selection = _require_mapping(section.get("repeat_selection", {}), "figure4.repeat_selection")
    reuse = selection.get("reuse_selection_from")
    models = _requested_models(config, "figure4", results)
    if reuse is not None:
        if str(reuse) != "figure3":
            raise VisualizationError("figure4.repeat_selection.reuse_selection_from must be figure3.")
        if figure3_selection is None:
            raise VisualizationError("Figure 4 requested reuse from Figure 3, but Figure 3 is not enabled.")
        missing = sorted(set(models) - set(figure3_selection["model"].astype(str)))
        if missing:
            raise VisualizationError(
                "Figure 4 models are not all present in Figure 3 selection: " + str(missing)
            )
        ordered = figure3_selection.set_index("model").loc[list(models)].reset_index()
        return ordered
    return select_best_repeats(config, results, models, selection)


def plot_figure4(
    config: VisualizationConfig,
    results: AggregatedResults,
    output_path: Path,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    """Plot RA/ILD predicted-probability boxplots for selected best repeats."""
    section = config.figures["figure4"]
    models = _requested_models(config, "figure4", results)
    show_points = bool(section.get("show_patient_points", True))
    show_threshold = bool(section.get("show_saved_threshold", True))
    selected_index = selected.set_index("model")
    figure, axes = _subplot_geometry(len(models), _max_columns(section))
    figure.suptitle("Predicted ILD probabilities in best observed holdouts", fontsize=15, fontweight="bold")
    plot_parts: List[pd.DataFrame] = []
    palette = plt.get_cmap("tab10")
    for panel_index, (axis, model) in enumerate(zip(axes, models)):
        selected_row = selected_index.loc[model]
        split_id = str(selected_row["split_id"])
        group = results.predictions.loc[
            (results.predictions["model"] == model)
            & (results.predictions["split_id"] == split_id)
        ].copy()
        group["display_group"] = pd.Categorical(
            group["true_cohort"].astype(str), categories=["RA", "ILD"], ordered=True
        )
        values_by_group = [
            group.loc[group["true_cohort"].astype(str) == label, "probability_ILD"].to_numpy(float)
            for label in ("RA", "ILD")
        ]
        if any(len(values) == 0 for values in values_by_group):
            raise VisualizationError(f"Figure 4 requires both RA and ILD for {model}/{split_id}")
        box = axis.boxplot(
            values_by_group,
            positions=[1, 2],
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for group_index, patch in enumerate(box["boxes"]):
            patch.set_facecolor(palette(group_index))
            patch.set_alpha(0.55)
        if show_points:
            for group_index, values in enumerate(values_by_group, start=1):
                axis.scatter(
                    np.full(len(values), group_index, dtype=float) + _deterministic_jitter(len(values), width=0.12),
                    values,
                    color=palette(group_index - 1),
                    edgecolors="none",
                    alpha=0.65,
                    s=22,
                    zorder=3,
                )
        threshold = float(selected_row["threshold"])
        if show_threshold:
            axis.axhline(
                threshold,
                linestyle="--",
                linewidth=1.5,
                color="black",
                label=f"Saved threshold = {threshold:.3f}",
            )
            axis.legend(frameon=False, loc="upper left", fontsize=9)
        axis.set_xticks([1, 2], ["RA", "ILD"])
        axis.set_ylabel("Predicted probability of ILD")
        axis.set_ylim(0.0, 1.02)
        axis.set_title(f"{_panel_label(panel_index)}. {_model_label(config, model)}", loc="left", fontweight="bold")
        axis.text(
            0.98,
            0.04,
            f"Repeat {int(selected_row['outer_repeat'])}\nROC-AUC = {float(selected_row['roc_auc']):.3f}",
            transform=axis.transAxes,
            va="bottom",
            ha="right",
            fontsize=9,
        )
        _apply_axis_style(axis)
        part = group.copy()
        part.insert(0, "selected_model_label", _model_label(config, model))
        part.insert(0, "selected_model", model)
        part["selected_split_id"] = split_id
        part["selected_outer_repeat"] = int(selected_row["outer_repeat"])
        part["selection_roc_auc"] = float(selected_row["roc_auc"])
        plot_parts.append(part)
    _save_pdf(figure, output_path)
    return pd.concat(plot_parts, ignore_index=True)


def _resolved_config_payload(config: VisualizationConfig) -> Dict[str, Any]:
    payload = dict(config.raw)
    visual = dict(_require_mapping(payload.get("visualization"), "visualization"))
    visual["aggregation_dir"] = str(config.aggregation_dir)
    visual["output_dir"] = str(config.output_dir)
    visual["repository_root"] = str(config.repository_root)
    payload["visualization"] = visual
    payload["resolved_by"] = {
        "module": "ra_ild_trb.repeated_holdout_visualization",
        "module_version": VISUALIZATION_VERSION,
    }
    return payload


def _planned_outputs(config: VisualizationConfig) -> Dict[str, Path]:
    base_keys = ("input_audit", "resolved_config", "manifest", "complete")
    outputs: Dict[str, Path] = {
        key: config.output_dir / OUTPUT_AUDIT_FILES[key] for key in base_keys
    }
    if _enabled(config, "figure1"):
        outputs["figure1"] = _figure_output_path(config, "figure1")
        outputs["figure1_data"] = config.output_dir / OUTPUT_AUDIT_FILES["figure1_data"]
    if _enabled(config, "figure2"):
        outputs["figure2"] = _figure_output_path(config, "figure2")
        outputs["figure2_coordinates"] = config.output_dir / OUTPUT_AUDIT_FILES["figure2_coordinates"]
        outputs["figure2_repeat_auc"] = config.output_dir / OUTPUT_AUDIT_FILES["figure2_repeat_auc"]
    if _enabled(config, "figure3"):
        outputs["figure3"] = _figure_output_path(config, "figure3")
        outputs["figure3_coordinates"] = config.output_dir / OUTPUT_AUDIT_FILES["figure3_coordinates"]
    if _enabled(config, "figure3") or _enabled(config, "figure4"):
        outputs["figure3_selection"] = config.output_dir / OUTPUT_AUDIT_FILES["figure3_selection"]
    if _enabled(config, "figure4"):
        outputs["figure4"] = _figure_output_path(config, "figure4")
        outputs["figure4_data"] = config.output_dir / OUTPUT_AUDIT_FILES["figure4_data"]
    return outputs


def _check_output_collision(outputs: Mapping[str, Path], overwrite: bool) -> None:
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Visualization outputs already exist; use --overwrite only for a documented rerun:\n  - "
            + "\n  - ".join(map(str, existing))
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def generate_visualizations(
    config: VisualizationConfig,
    *,
    overwrite_override: Optional[bool] = None,
) -> VisualizationResult:
    """Generate all enabled figures and audit artifacts."""
    results, input_audit = load_aggregated_results(config)
    validate_requested_models(config, results)
    outputs = _planned_outputs(config)
    overwrite = config.overwrite if overwrite_override is None else bool(overwrite_override)
    _check_output_collision(outputs, overwrite)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    generated: MutableMapping[str, Path] = {}
    selected_figure3: Optional[pd.DataFrame] = None

    input_audit.to_csv(outputs["input_audit"], index=False)
    generated["input_audit"] = outputs["input_audit"]
    outputs["resolved_config"].write_text(
        yaml.safe_dump(_resolved_config_payload(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    generated["resolved_config"] = outputs["resolved_config"]

    if _enabled(config, "figure1"):
        data = plot_figure1(config, results, outputs["figure1"])
        data.to_csv(outputs["figure1_data"], index=False)
        generated["figure1"] = outputs["figure1"]
        generated["figure1_data"] = outputs["figure1_data"]

    if _enabled(config, "figure2"):
        coordinates, repeat_auc = plot_figure2(config, results, outputs["figure2"])
        coordinates.to_csv(outputs["figure2_coordinates"], index=False, compression="gzip")
        repeat_auc.to_csv(outputs["figure2_repeat_auc"], index=False)
        generated["figure2"] = outputs["figure2"]
        generated["figure2_coordinates"] = outputs["figure2_coordinates"]
        generated["figure2_repeat_auc"] = outputs["figure2_repeat_auc"]

    if _enabled(config, "figure3"):
        selected_figure3 = _selection_for_figure3(config, results)
        coordinates = plot_figure3(
            config, results, outputs["figure3"], selected_figure3
        )
        selected_figure3.to_csv(outputs["figure3_selection"], index=False)
        coordinates.to_csv(outputs["figure3_coordinates"], index=False, compression="gzip")
        generated["figure3"] = outputs["figure3"]
        generated["figure3_selection"] = outputs["figure3_selection"]
        generated["figure3_coordinates"] = outputs["figure3_coordinates"]

    selected_figure4: Optional[pd.DataFrame] = None
    if _enabled(config, "figure4"):
        selected_figure4 = _figure4_selection(config, results, selected_figure3)
        data = plot_figure4(config, results, outputs["figure4"], selected_figure4)
        data.to_csv(outputs["figure4_data"], index=False)
        generated["figure4"] = outputs["figure4"]
        generated["figure4_data"] = outputs["figure4_data"]
        if selected_figure3 is None:
            # Figure 4 can select independently. Preserve the selection audit under
            # the common filename so every best-repeat plot remains traceable.
            selected_figure4.to_csv(outputs["figure3_selection"], index=False)
            generated["figure3_selection"] = outputs["figure3_selection"]

    manifest_payload = {
        "visualization_id": config.visualization_id,
        "visualization_version": VISUALIZATION_VERSION,
        "source_mode": "native_repeated_holdout",
        "aggregation_dir": str(config.aggregation_dir),
        "output_dir": str(config.output_dir),
        "requested_models": list(_all_requested_models(config, results)),
        "enabled_figures": [
            name for name in ("figure1", "figure2", "figure3", "figure4")
            if _enabled(config, name)
        ],
        "input_files": {
            key: {
                "path": str(results.input_paths[key]),
                "sha256": results.input_sha256.get(key),
            }
            for key in results.input_paths
            if results.input_paths[key].is_file()
        },
        "generated_files": {
            key: str(path) for key, path in generated.items()
        },
        "scientific_notes": {
            "mean_roc_method": "per-repeat ROC interpolation followed by macro-average TPR",
            "mean_roc_band": "across-repeat percentile variability; not an independent-sample CI",
            "best_repeat": "descriptive best observed native holdout; not representative overall performance",
            "thresholds": "saved inner-training-derived thresholds; never reselected from holdout labels",
            "cross_repeat_results_used": False,
        },
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "scikit_learn": sklearn.__version__,
            "pyyaml": yaml.__version__,
        },
    }
    _write_json(outputs["manifest"], manifest_payload)
    generated["manifest"] = outputs["manifest"]

    # Completion marker is written last and includes hashes for every prior output.
    output_hashes = {
        key: {"path": str(path), "sha256": sha256_file(path)}
        for key, path in generated.items()
    }
    completion_payload = {
        "status": "COMPLETE",
        "visualization_id": config.visualization_id,
        "visualization_version": VISUALIZATION_VERSION,
        "source_mode": "native_repeated_holdout",
        "input_metric_rows": int(len(results.metrics)),
        "input_prediction_rows": int(len(results.predictions)),
        "input_audit_rows": int(len(input_audit)),
        "input_audit_all_pass": bool(input_audit["audit_pass"].all()),
        "generated_file_count_before_marker": int(len(generated)),
        "files": output_hashes,
        "pid": int(os.getpid()),
    }
    _write_json(outputs["complete"], completion_payload)
    generated["complete"] = outputs["complete"]

    selected = (
        selected_figure3
        if selected_figure3 is not None
        else selected_figure4
        if selected_figure4 is not None
        else pd.DataFrame()
    )
    return VisualizationResult(
        output_dir=config.output_dir,
        generated_files=dict(generated),
        selected_best_repeats=selected.copy(),
    )
