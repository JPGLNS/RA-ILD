#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused tests for configurable repeated-holdout visualization Batch 11."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, recall_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.repeated_holdout_visualization import (
    VisualizationError,
    generate_visualizations,
    load_visualization_config,
    select_best_repeats,
    _subplot_geometry,
    _trapezoid_area,
)


def _metric_row(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    predicted = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "accuracy": float(np.mean(predicted == y)),
        "sensitivity_recall": float(recall_score(y, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "f1": float(f1_score(y, predicted, zero_division=0)),
    }


def _synthetic_files(root: Path, *, marker_status: str = "COMPLETE") -> tuple[Path, Path]:
    aggregation = root / "aggregation"
    output = root / "visualization"
    aggregation.mkdir(parents=True)
    models = ("A1_core83", "A3_core83_unweighted500", "A5_core83_public")
    quality = {
        "A1_core83": [0.07, 0.10, 0.14, 0.18],
        "A3_core83_unweighted500": [0.14, 0.20, 0.30, 0.11],
        "A5_core83_public": [0.20, 0.16, 0.24, 0.36],
    }
    metric_rows = []
    prediction_rows = []
    y = np.asarray([0] * 6 + [1] * 6, dtype=int)
    for repeat in range(1, 5):
        split_id = f"split_{repeat:03d}"
        for model_index, model in enumerate(models):
            q = quality[model][repeat - 1]
            # Strictly deterministic and non-tied probabilities.
            negative = np.linspace(0.08, 0.42, 6) + 0.01 * repeat + 0.005 * model_index
            positive = np.linspace(0.58, 0.92, 6) - q + 0.006 * np.arange(6)
            p = np.clip(np.concatenate([negative, positive]), 0.001, 0.999)
            threshold = 0.50 + 0.01 * (repeat - 2)
            metrics = _metric_row(y, p, threshold)
            metric_rows.append(
                {
                    "split_set_id": "synthetic_repeat4",
                    "split_id": split_id,
                    "task_id": f"repeat_{repeat:03d}_fold_01",
                    "task_index": repeat,
                    "outer_repeat": repeat,
                    "outer_fold": 1,
                    "model": model,
                    "threshold": threshold,
                    "selected_l1_ratio_alpha": 0.5,
                    "selected_lambda": 1.0,
                    "inner_selected_roc_auc": 0.7,
                    "inner_selected_pr_auc": 0.7,
                    **metrics,
                }
            )
            predicted = (p >= threshold).astype(int)
            for sample_index, (truth, probability, pred) in enumerate(zip(y, p, predicted), start=1):
                prediction_rows.append(
                    {
                        "split_set_id": "synthetic_repeat4",
                        "split_id": split_id,
                        "task_id": f"repeat_{repeat:03d}_fold_01",
                        "task_index": repeat,
                        "model": model,
                        "outer_repeat": repeat,
                        "outer_fold": 1,
                        "sample_id": f"R{repeat:02d}_S{sample_index:02d}",
                        "true_label": int(truth),
                        "true_cohort": "ILD" if truth else "RA",
                        "probability_ILD": float(probability),
                        "threshold": threshold,
                        "predicted_label": int(pred),
                        "predicted_cohort": "ILD" if pred else "RA",
                    }
                )
    metrics_frame = pd.DataFrame(metric_rows)
    predictions_frame = pd.DataFrame(prediction_rows)
    metrics_frame.to_csv(aggregation / "08_holdout_task_metrics.csv", index=False)
    predictions_frame.to_csv(
        aggregation / "08_holdout_predictions.csv.gz", index=False, compression="gzip"
    )
    (aggregation / "08_aggregation_summary.json").write_text(
        json.dumps(
            {
                "experiment_id": "synthetic",
                "analysis_mode": "frozen_repeated_holdout",
                "split_set_id": "synthetic_repeat4",
                "completed_split_count": 4,
                "models": list(models),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (aggregation / "08_REPEATED_HOLDOUT_AGGREGATION_COMPLETE.json").write_text(
        json.dumps({"status": marker_status}, indent=2) + "\n",
        encoding="utf-8",
    )
    return aggregation, output


def _config(root: Path, aggregation: Path, output: Path, *, unknown_model: bool = False) -> Path:
    models = ["A1_core83", "A3_core83_unweighted500", "A5_core83_public"]
    if unknown_model:
        models[-1] = "MISSING_MODEL"
    payload = {
        "visualization_version": "1.0",
        "visualization": {
            "id": "synthetic_visualization",
            "source_mode": "native_repeated_holdout",
            "aggregation_dir": str(aggregation),
            "output_dir": str(output),
            "require_complete_marker": True,
            "overwrite": False,
        },
        "model_labels": {
            "A1_core83": "Core83",
            "A3_core83_unweighted500": "Core83 + Unweighted Top500",
            "A5_core83_public": "Core83 + Public",
        },
        "figures": {
            "figure1": {
                "enabled": True,
                "models": models,
                "metrics": ["roc_auc", "pr_auc", "f1", "sensitivity_recall", "specificity"],
                "max_columns": 3,
                "show_repeat_points": True,
                "show_mean": True,
            },
            "figure2": {
                "enabled": True,
                "models": models,
                "max_columns": 3,
                "fpr_grid_points": 101,
                "show_repeat_variability_band": True,
                "variability_quantiles": [0.025, 0.975],
            },
            "figure3": {
                "enabled": True,
                "models": models[1:],
                "max_columns": 3,
                "repeat_selection": {
                    "mode": "best",
                    "primary_metric": "roc_auc",
                    "tie_breakers": ["pr_auc", "f1", "outer_repeat"],
                },
                "mark_saved_threshold_operating_point": True,
            },
            "figure4": {
                "enabled": True,
                "models": models[1:],
                "max_columns": 3,
                "repeat_selection": {"reuse_selection_from": "figure3"},
                "show_patient_points": True,
                "show_saved_threshold": True,
            },
        },
    }
    config_path = root / "visualization.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path



def test_numpy_trapezoid_compatibility() -> None:
    x = np.asarray([0.0, 0.5, 1.0])
    y = np.asarray([0.0, 0.5, 1.0])
    assert np.isclose(_trapezoid_area(y, x), 0.5)


def test_subplot_layout_policy() -> None:
    one_figure, one_axes = _subplot_geometry(1, 3)
    one_figure.canvas.draw()
    assert len(one_axes) == 1
    assert one_axes[0].get_position().width > 0.80

    two_figure, two_axes = _subplot_geometry(2, 3)
    two_figure.canvas.draw()
    assert len(two_axes) == 2
    assert np.isclose(two_axes[0].get_position().width, two_axes[1].get_position().width)

    five_figure, five_axes = _subplot_geometry(5, 3)
    five_figure.canvas.draw()
    assert len(five_axes) == 5
    assert abs(five_axes[3].get_position().width - five_axes[4].get_position().width) < 0.02

    import matplotlib.pyplot as plt
    plt.close(one_figure)
    plt.close(two_figure)
    plt.close(five_figure)

def test_all_four_figures_and_audits() -> None:
    with tempfile.TemporaryDirectory(prefix="trb_visual_test_") as temp:
        root = Path(temp)
        aggregation, output = _synthetic_files(root)
        config = load_visualization_config(_config(root, aggregation, output), repository_root=root)
        result = generate_visualizations(config)
        expected = [
            "Figure1_model_performance_boxplots.pdf",
            "Figure2_mean_ROC_curves.pdf",
            "Figure3_best_repeat_ROC_curves.pdf",
            "Figure4_best_repeat_prediction_boxplots.pdf",
            "Figure1_plot_data.csv",
            "Figure2_mean_ROC_coordinates.csv.gz",
            "Figure2_repeat_ROC_AUC.csv",
            "Figure3_best_repeat_selection.csv",
            "Figure3_best_repeat_ROC_coordinates.csv.gz",
            "Figure4_plot_data.csv",
            "visualization_input_audit.csv",
            "visualization_resolved_config.yaml",
            "visualization_manifest.json",
            "VISUALIZATION_COMPLETE.json",
        ]
        for name in expected:
            path = output / name
            assert path.is_file(), name
            assert path.stat().st_size > 0, name
        for name in expected[:4]:
            assert (output / name).read_bytes().startswith(b"%PDF")
        marker = json.loads((output / "VISUALIZATION_COMPLETE.json").read_text())
        assert marker["status"] == "COMPLETE"
        assert marker["input_audit_all_pass"] is True
        assert len(result.generated_files) == len(expected)

        figure1 = pd.read_csv(output / "Figure1_plot_data.csv")
        assert set(figure1["metric"]) == {
            "roc_auc", "pr_auc", "f1", "sensitivity_recall", "specificity"
        }
        assert figure1.groupby(["metric", "model"]).size().eq(4).all()

        coordinates = pd.read_csv(output / "Figure2_mean_ROC_coordinates.csv.gz")
        assert coordinates.groupby("model").size().eq(101).all()
        endpoints = coordinates.groupby("model").agg(
            first_fpr=("fpr", "min"), last_fpr=("fpr", "max")
        )
        assert np.allclose(endpoints["first_fpr"], 0.0)
        assert np.allclose(endpoints["last_fpr"], 1.0)

        selected = pd.read_csv(output / "Figure3_best_repeat_selection.csv")
        assert set(selected["model"]) == {
            "A3_core83_unweighted500", "A5_core83_public"
        }
        assert selected["best_repeat_is_descriptive_only"].all()
        figure4 = pd.read_csv(output / "Figure4_plot_data.csv")
        assert set(figure4["selected_model"]) == set(selected["model"])
        selected_pairs = set(zip(selected["model"], selected["split_id"]))
        plotted_pairs = set(zip(figure4["selected_model"], figure4["selected_split_id"]))
        assert selected_pairs == plotted_pairs


def test_best_repeat_tie_breaker_is_deterministic() -> None:
    with tempfile.TemporaryDirectory(prefix="trb_visual_tie_") as temp:
        root = Path(temp)
        aggregation, output = _synthetic_files(root)
        config = load_visualization_config(_config(root, aggregation, output), repository_root=root)
        metrics = pd.read_csv(aggregation / "08_holdout_task_metrics.csv")
        # Force equal primary/secondary metrics for one model and verify smaller repeat wins.
        mask = metrics["model"] == "A3_core83_unweighted500"
        metrics.loc[mask, ["roc_auc", "pr_auc", "f1"]] = 0.75
        from types import SimpleNamespace
        results = SimpleNamespace(metrics=metrics)
        selected = select_best_repeats(
            config,
            results,
            ["A3_core83_unweighted500"],
            {
                "mode": "best",
                "primary_metric": "roc_auc",
                "tie_breakers": ["pr_auc", "f1", "outer_repeat"],
            },
        )
        assert int(selected.loc[0, "outer_repeat"]) == 1



def test_models_all_selector() -> None:
    with tempfile.TemporaryDirectory(prefix="trb_visual_all_") as temp:
        root = Path(temp)
        aggregation, output = _synthetic_files(root)
        config_path = _config(root, aggregation, output)
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        payload["figures"] = {
            "figure1": {
                "enabled": True,
                "models": "all",
                "metrics": ["roc_auc"],
                "max_columns": 3,
            },
            "figure2": {"enabled": False},
            "figure3": {"enabled": False},
            "figure4": {"enabled": False},
        }
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        config = load_visualization_config(config_path, repository_root=root)
        generate_visualizations(config)
        data = pd.read_csv(output / "Figure1_plot_data.csv")
        assert set(data["model"]) == {
            "A1_core83", "A3_core83_unweighted500", "A5_core83_public"
        }
        assert (output / "Figure1_model_performance_boxplots.pdf").is_file()

def test_unknown_model_is_rejected() -> None:
    with tempfile.TemporaryDirectory(prefix="trb_visual_missing_") as temp:
        root = Path(temp)
        aggregation, output = _synthetic_files(root)
        config = load_visualization_config(
            _config(root, aggregation, output, unknown_model=True), repository_root=root
        )
        try:
            generate_visualizations(config)
        except VisualizationError as exc:
            assert "Requested models are absent" in str(exc)
        else:
            raise AssertionError("Unknown model should be rejected")


def test_incomplete_aggregation_is_rejected() -> None:
    with tempfile.TemporaryDirectory(prefix="trb_visual_incomplete_") as temp:
        root = Path(temp)
        aggregation, output = _synthetic_files(root, marker_status="INCOMPLETE")
        config = load_visualization_config(_config(root, aggregation, output), repository_root=root)
        try:
            generate_visualizations(config)
        except VisualizationError as exc:
            assert "not COMPLETE" in str(exc)
        else:
            raise AssertionError("Incomplete aggregation should be rejected")


def test_existing_output_requires_explicit_overwrite() -> None:
    with tempfile.TemporaryDirectory(prefix="trb_visual_overwrite_") as temp:
        root = Path(temp)
        aggregation, output = _synthetic_files(root)
        config = load_visualization_config(_config(root, aggregation, output), repository_root=root)
        generate_visualizations(config)
        try:
            generate_visualizations(config)
        except FileExistsError as exc:
            assert "already exist" in str(exc)
        else:
            raise AssertionError("Existing output should require explicit overwrite")
        generate_visualizations(config, overwrite_override=True)


def main() -> int:
    tests = [
        test_numpy_trapezoid_compatibility,
        test_subplot_layout_policy,
        test_all_four_figures_and_audits,
        test_best_repeat_tie_breaker_is_deterministic,
        test_models_all_selector,
        test_unknown_model_is_rejected,
        test_incomplete_aggregation_is_rejected,
        test_existing_output_requires_explicit_overwrite,
    ]
    for function in tests:
        function()
        print(f"PASS {function.__name__}")
    print(f"Batch 11 focused tests: {len(tests)}/{len(tests)} PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
