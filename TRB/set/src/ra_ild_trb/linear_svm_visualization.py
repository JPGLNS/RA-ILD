#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration-driven visualization for aggregated TRB Linear SVM results."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


class LinearSVMVisualizationError(ValueError):
    """Raised when Linear SVM visualization inputs/configuration are invalid."""


def _load_yaml(path: Path) -> Mapping[str, object]:
    if yaml is None:
        raise LinearSVMVisualizationError("PyYAML is required")
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise LinearSVMVisualizationError("Visualization YAML root must be a mapping")
    return value


def _resolve_path(value: object, repository_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def _select_models(value: object, available: Sequence[str]) -> List[str]:
    ordered = list(dict.fromkeys(map(str, available)))
    if value is None or value == "all":
        return ordered
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise LinearSVMVisualizationError("models must be 'all' or a list")
    selected = [str(item) for item in value]
    unknown = sorted(set(selected) - set(ordered))
    if unknown:
        raise LinearSVMVisualizationError(f"Unknown models: {unknown}")
    if not selected:
        raise LinearSVMVisualizationError("At least one model must be selected")
    return selected


def _axes_grid(n_panels: int, max_columns: int = 3):
    columns = min(max_columns, max(1, n_panels))
    rows = int(math.ceil(n_panels / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 4.3 * rows))
    axes_array = np.asarray(axes, dtype=object).reshape(-1)
    for axis in axes_array[n_panels:]:
        axis.set_visible(False)
    return figure, axes_array[:n_panels]


def _boxplot_with_tick_labels(
    axis,
    values: Sequence[np.ndarray],
    tick_labels: Sequence[str],
    **kwargs,
):
    """Draw boxplots across Matplotlib's labels/tick_labels API rename.

    Matplotlib 3.9 renamed ``labels`` to ``tick_labels`` and Matplotlib 3.11
    removed the old keyword. Older supported environments still require
    ``labels``. Try the current API first and fall back only when that keyword
    is unavailable.
    """
    normalized_labels = [str(value) for value in tick_labels]
    try:
        return axis.boxplot(
            values,
            tick_labels=normalized_labels,
            **kwargs,
        )
    except TypeError as exc:
        message = str(exc)
        if "tick_labels" not in message:
            raise
        return axis.boxplot(
            values,
            labels=normalized_labels,
            **kwargs,
        )


def _load_aggregation(aggregation_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = aggregation_dir / "08_holdout_task_metrics.csv"
    predictions_path = aggregation_dir / "08_holdout_predictions.csv.gz"
    marker_path = aggregation_dir / "08_REPEATED_HOLDOUT_AGGREGATION_COMPLETE.json"
    for path in (metrics_path, predictions_path, marker_path):
        if not path.is_file():
            raise LinearSVMVisualizationError(f"Missing aggregation input: {path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if str(marker.get("status", "")).upper() != "COMPLETE":
        raise LinearSVMVisualizationError("Aggregation completion marker is invalid")
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)
    required_metrics = {
        "model",
        "outer_repeat",
        "outer_fold",
        "roc_auc",
        "average_precision",
        "f1",
        "sensitivity_recall",
        "specificity",
    }
    required_predictions = {
        "model",
        "outer_repeat",
        "outer_fold",
        "true_label",
        "true_cohort",
        "prediction_score",
        "threshold",
    }
    if required_metrics - set(metrics.columns):
        raise LinearSVMVisualizationError(
            f"Metrics missing columns: {sorted(required_metrics - set(metrics.columns))}"
        )
    if required_predictions - set(predictions.columns):
        raise LinearSVMVisualizationError(
            "Predictions missing columns: "
            f"{sorted(required_predictions - set(predictions.columns))}"
        )
    return metrics, predictions


def _best_repeats(metrics: pd.DataFrame, models: Sequence[str]) -> pd.DataFrame:
    rows = []
    for model in models:
        subset = metrics.loc[metrics["model"].astype(str) == model].copy()
        if subset.empty:
            raise LinearSVMVisualizationError(f"No metric rows for model {model}")
        subset = subset.sort_values(
            ["roc_auc", "average_precision", "f1", "outer_repeat", "outer_fold"],
            ascending=[False, False, False, True, True],
            kind="mergesort",
        )
        rows.append(subset.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def create_visualizations(config_path: Path, *, overwrite: bool = False) -> Mapping[str, Path]:
    raw = _load_yaml(config_path)
    root = raw.get("visualization", raw)
    if not isinstance(root, Mapping):
        raise LinearSVMVisualizationError("visualization must be a mapping")
    repository_root = _resolve_path(root.get("repository_root", "."), Path.cwd())
    aggregation_dir = _resolve_path(root.get("aggregation_dir"), repository_root)
    output_dir = _resolve_path(root.get("output_dir"), repository_root)
    metrics, predictions = _load_aggregation(aggregation_dir)
    available_models = metrics["model"].astype(str).drop_duplicates().tolist()
    models = _select_models(root.get("models", "all"), available_models)
    metrics = metrics.loc[metrics["model"].astype(str).isin(models)].copy()
    predictions = predictions.loc[predictions["model"].astype(str).isin(models)].copy()

    configured_metrics = root.get(
        "performance_metrics",
        [
            "roc_auc",
            "average_precision",
            "f1",
            "sensitivity_recall",
            "specificity",
        ],
    )
    if not isinstance(configured_metrics, Sequence) or isinstance(
        configured_metrics, (str, bytes)
    ):
        raise LinearSVMVisualizationError("performance_metrics must be a list")
    performance_metrics = [str(value) for value in configured_metrics]
    unknown_metrics = sorted(set(performance_metrics) - set(metrics.columns))
    if unknown_metrics:
        raise LinearSVMVisualizationError(f"Unknown metrics: {unknown_metrics}")

    paths: Dict[str, Path] = {
        "figure1": output_dir / "Figure1_model_performance_boxplots.pdf",
        "figure1_data": output_dir / "Figure1_plot_data.csv",
        "figure2": output_dir / "Figure2_mean_ROC_curves.pdf",
        "figure2_coordinates": output_dir / "Figure2_mean_ROC_coordinates.csv.gz",
        "figure2_auc": output_dir / "Figure2_repeat_ROC_AUC.csv",
        "figure3": output_dir / "Figure3_best_repeat_ROC_curves.pdf",
        "figure3_selection": output_dir / "Figure3_best_repeat_selection.csv",
        "figure3_coordinates": output_dir / "Figure3_best_repeat_ROC_coordinates.csv.gz",
        "figure4": output_dir / "Figure4_best_repeat_decision_score_boxplots.pdf",
        "figure4_data": output_dir / "Figure4_plot_data.csv",
        "audit": output_dir / "visualization_input_audit.csv",
        "resolved": output_dir / "visualization_resolved_config.yaml",
        "manifest": output_dir / "visualization_manifest.json",
        "complete": output_dir / "VISUALIZATION_COMPLETE.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Visualization outputs already exist; use --overwrite for a documented rerun:\n"
            + "\n".join(f"  - {path}" for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: performance distributions.
    fig, axes = _axes_grid(len(performance_metrics))
    long_parts = []
    for axis, metric in zip(axes, performance_metrics):
        values = []
        for model in models:
            subset = pd.to_numeric(
                metrics.loc[metrics["model"].astype(str) == model, metric],
                errors="raise",
            ).to_numpy(float)
            values.append(subset)
            for value in subset:
                long_parts.append({"model": model, "metric": metric, "value": value})
        _boxplot_with_tick_labels(axis, values, models, showmeans=True)
        axis.set_title(metric)
        axis.set_ylabel(metric)
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths["figure1"], bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(long_parts).to_csv(paths["figure1_data"], index=False)

    # Figure 2: repeat-level ROC curves and macro mean.
    common_fpr = np.linspace(0.0, 1.0, 201)
    mean_coordinate_rows = []
    repeat_auc_rows = []
    fig, axes = _axes_grid(len(models))
    for axis, model in zip(axes, models):
        model_predictions = predictions.loc[
            predictions["model"].astype(str) == model
        ].copy()
        interpolated = []
        for (repeat, fold), subset in model_predictions.groupby(
            ["outer_repeat", "outer_fold"], sort=True
        ):
            truth = pd.to_numeric(subset["true_label"], errors="raise").to_numpy(int)
            score = pd.to_numeric(
                subset["prediction_score"], errors="raise"
            ).to_numpy(float)
            fpr, tpr, _ = roc_curve(truth, score)
            curve_auc = float(auc(fpr, tpr))
            repeat_auc_rows.append(
                {
                    "model": model,
                    "outer_repeat": int(repeat),
                    "outer_fold": int(fold),
                    "roc_auc": curve_auc,
                }
            )
            interp = np.interp(common_fpr, fpr, tpr)
            interp[0] = 0.0
            interp[-1] = 1.0
            interpolated.append(interp)
            axis.plot(fpr, tpr, alpha=0.12, linewidth=0.8)
        matrix = np.vstack(interpolated)
        mean_tpr = matrix.mean(axis=0)
        lower = np.quantile(matrix, 0.10, axis=0)
        upper = np.quantile(matrix, 0.90, axis=0)
        mean_auc = float(np.mean([row["roc_auc"] for row in repeat_auc_rows if row["model"] == model]))
        sd_auc = float(np.std([row["roc_auc"] for row in repeat_auc_rows if row["model"] == model], ddof=1)) if len(interpolated) > 1 else 0.0
        axis.plot(common_fpr, mean_tpr, linewidth=2.0, label=f"Mean AUC={mean_auc:.3f} ± {sd_auc:.3f}")
        axis.fill_between(common_fpr, lower, upper, alpha=0.18)
        axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
        axis.set_title(model)
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.legend(loc="lower right")
        axis.grid(alpha=0.2)
        for fpr_value, mean_value, low_value, high_value in zip(
            common_fpr, mean_tpr, lower, upper
        ):
            mean_coordinate_rows.append(
                {
                    "model": model,
                    "fpr": float(fpr_value),
                    "mean_tpr": float(mean_value),
                    "p10_tpr": float(low_value),
                    "p90_tpr": float(high_value),
                }
            )
    fig.tight_layout()
    fig.savefig(paths["figure2"], bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(mean_coordinate_rows).to_csv(
        paths["figure2_coordinates"], index=False, compression="gzip"
    )
    pd.DataFrame(repeat_auc_rows).to_csv(paths["figure2_auc"], index=False)

    best = _best_repeats(metrics, models)
    best.to_csv(paths["figure3_selection"], index=False)

    # Figure 3: deterministically selected best repeat ROC.
    best_coordinate_rows = []
    fig, axes = _axes_grid(len(models))
    for axis, (_, selection) in zip(axes, best.iterrows()):
        model = str(selection["model"])
        repeat = int(selection["outer_repeat"])
        fold = int(selection["outer_fold"])
        subset = predictions.loc[
            (predictions["model"].astype(str) == model)
            & (pd.to_numeric(predictions["outer_repeat"]) == repeat)
            & (pd.to_numeric(predictions["outer_fold"]) == fold)
        ].copy()
        truth = pd.to_numeric(subset["true_label"], errors="raise").to_numpy(int)
        score = pd.to_numeric(subset["prediction_score"], errors="raise").to_numpy(float)
        fpr, tpr, thresholds = roc_curve(truth, score)
        curve_auc = float(auc(fpr, tpr))
        axis.plot(fpr, tpr, linewidth=2.0, label=f"AUC={curve_auc:.3f}")
        axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
        selected_threshold = float(pd.to_numeric(subset["threshold"], errors="raise").iloc[0])
        finite = np.isfinite(thresholds)
        if finite.any():
            index = int(np.argmin(np.abs(thresholds[finite] - selected_threshold)))
            finite_indices = np.flatnonzero(finite)
            real_index = int(finite_indices[index])
            axis.scatter([fpr[real_index]], [tpr[real_index]], s=35, label="Saved Youden point")
        axis.set_title(f"{model}\nrepeat {repeat}, fold {fold}")
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.legend(loc="lower right")
        axis.grid(alpha=0.2)
        for fpr_value, tpr_value, threshold_value in zip(fpr, tpr, thresholds):
            best_coordinate_rows.append(
                {
                    "model": model,
                    "outer_repeat": repeat,
                    "outer_fold": fold,
                    "fpr": float(fpr_value),
                    "tpr": float(tpr_value),
                    "threshold": float(threshold_value),
                }
            )
    fig.tight_layout()
    fig.savefig(paths["figure3"], bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(best_coordinate_rows).to_csv(
        paths["figure3_coordinates"], index=False, compression="gzip"
    )

    # Figure 4: class-stratified decision scores for the same best repeats.
    score_rows = []
    fig, axes = _axes_grid(len(models))
    for axis, (_, selection) in zip(axes, best.iterrows()):
        model = str(selection["model"])
        repeat = int(selection["outer_repeat"])
        fold = int(selection["outer_fold"])
        subset = predictions.loc[
            (predictions["model"].astype(str) == model)
            & (pd.to_numeric(predictions["outer_repeat"]) == repeat)
            & (pd.to_numeric(predictions["outer_fold"]) == fold)
        ].copy()
        cohorts = ["RA", "ILD"]
        values = []
        for cohort in cohorts:
            cohort_values = pd.to_numeric(
                subset.loc[subset["true_cohort"].astype(str) == cohort, "prediction_score"],
                errors="raise",
            ).to_numpy(float)
            values.append(cohort_values)
            for value in cohort_values:
                score_rows.append(
                    {
                        "model": model,
                        "outer_repeat": repeat,
                        "outer_fold": fold,
                        "true_cohort": cohort,
                        "prediction_score": float(value),
                    }
                )
        _boxplot_with_tick_labels(axis, values, cohorts, showmeans=True)
        for index, cohort_values in enumerate(values, start=1):
            if len(cohort_values):
                x = np.full(len(cohort_values), index, dtype=float)
                axis.scatter(x, cohort_values, alpha=0.55, s=16)
        saved_threshold = float(pd.to_numeric(subset["threshold"], errors="raise").iloc[0])
        axis.axhline(saved_threshold, linestyle="--", linewidth=1.5, label="Inner-OOF Youden")
        axis.axhline(0.0, linestyle=":", linewidth=1.5, label="Natural zero")
        axis.set_title(f"{model}\nrepeat {repeat}, fold {fold}")
        axis.set_ylabel("Linear SVM decision score for ILD")
        axis.legend(loc="best")
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(paths["figure4"], bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(score_rows).to_csv(paths["figure4_data"], index=False)

    audit = pd.DataFrame(
        [
            {
                "aggregation_dir": str(aggregation_dir),
                "metrics_rows": int(len(metrics)),
                "prediction_rows": int(len(predictions)),
                "selected_models": ",".join(models),
                "score_type": "decision_function",
                "probability_axis_used": False,
            }
        ]
    )
    audit.to_csv(paths["audit"], index=False)
    resolved = {
        "visualization": {
            "repository_root": str(repository_root),
            "aggregation_dir": str(aggregation_dir),
            "output_dir": str(output_dir),
            "models": models,
            "performance_metrics": performance_metrics,
            "score_type": "decision_function",
        }
    }
    paths["resolved"].write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    manifest = {
        "visualization_type": "linear_svm_repeated_holdout",
        "model_engine": "linear_svc",
        "score_type": "decision_function",
        "models": models,
        "figures": [str(paths[key]) for key in ("figure1", "figure2", "figure3", "figure4")],
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paths["complete"].write_text(
        json.dumps({"status": "COMPLETE", **manifest}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return paths
