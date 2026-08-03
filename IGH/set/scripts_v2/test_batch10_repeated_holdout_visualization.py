#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused acceptance tests for IGH Batch 10 visualization."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.repeated_holdout_visualization import (  # noqa: E402
    VisualizationError,
    generate_visualizations,
    load_visualization_config,
)


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "sensitivity_recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def create_fixture(root: Path) -> tuple[Path, Path]:
    aggregation = root / "experiment" / "03_summary"
    output = root / "experiment" / "06_visualization" / "test_v1"
    aggregation.mkdir(parents=True)
    rng = np.random.default_rng(20260803)
    models = ["A1_core83", "A5_core83_public"]
    metric_rows = []
    prediction_rows = []
    task_index = 0
    for repeat in range(1, 5):
        task_index += 1
        y = np.asarray([0] * 6 + [1] * 6, dtype=int)
        for model_index, model in enumerate(models):
            base = np.linspace(0.12, 0.88, len(y))
            noise = rng.normal(0, 0.07 + model_index * 0.01, len(y))
            p = np.clip(base + noise + 0.01 * repeat - 0.015 * model_index, 0.001, 0.999)
            threshold = 0.5 + 0.01 * (repeat % 2)
            row = metrics(y, p, threshold)
            row.update(
                {
                    "task_id": f"repeat_{repeat:02d}_fold_01",
                    "task_index": task_index,
                    "model": model,
                    "model_definition": model,
                    "outer_repeat": repeat,
                    "outer_fold": 1,
                    "selected_l1_ratio_alpha": 0.5,
                    "selected_lambda": 3.0,
                    "inner_selected_roc_auc": 0.7,
                    "inner_selected_pr_auc": 0.65,
                    "fit_converged": True,
                    "iterations_used": 200,
                    "n_final_predictors": 83,
                    "n_nonzero_coefficients": 40,
                }
            )
            metric_rows.append(row)
            pred = (p >= threshold).astype(int)
            for index, (truth, probability, predicted) in enumerate(zip(y, p, pred)):
                prediction_rows.append(
                    {
                        "task_id": f"repeat_{repeat:02d}_fold_01",
                        "task_index": task_index,
                        "model": model,
                        "outer_repeat": repeat,
                        "outer_fold": 1,
                        "sample_id": f"R{repeat:02d}_S{index + 1:02d}",
                        "true_label": int(truth),
                        "true_cohort": "ILD" if truth else "RA",
                        "probability_ILD": float(probability),
                        "threshold": threshold,
                        "predicted_label": int(predicted),
                        "predicted_cohort": "ILD" if predicted else "RA",
                    }
                )
    pd.DataFrame(metric_rows).to_csv(aggregation / "07_all_outer_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        aggregation / "07_all_outer_predictions.csv.gz", index=False, compression="gzip"
    )
    summary = {
        "status": "COMPLETE",
        "aggregation_mode": "repeated_holdout",
        "experiment_id": "synthetic_igh_repeat4",
        "models": models,
        "scientific_contract": {
            "repeats_treated_as_independent_cohorts": False,
            "final_model_selected": False,
        },
    }
    (aggregation / "07_aggregation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    marker = {
        "status": "COMPLETE",
        "aggregation_mode": "repeated_holdout",
        "experiment_id": "synthetic_igh_repeat4",
        "completed_outer_tasks": 4,
        "expected_outer_tasks": 4,
        "final_model_selected": False,
    }
    (aggregation / "07_REPEATED_HOLDOUT_AGGREGATION_COMPLETE.json").write_text(
        json.dumps(marker, indent=2) + "\n", encoding="utf-8"
    )
    config = {
        "visualization_version": "1.0",
        "visualization": {
            "id": "synthetic_visualization_v1",
            "source_mode": "native_repeated_holdout",
            "aggregation_dir": str(aggregation),
            "output_dir": str(output),
            "require_complete_marker": True,
            "overwrite": False,
            "metric_tolerance": 1e-6,
        },
        "model_labels": {"A1_core83": "A1", "A5_core83_public": "A5"},
        "figures": {
            "figure1": {
                "enabled": True,
                "models": "all",
                "metrics": ["roc_auc", "pr_auc", "f1"],
                "show_points": True,
            },
            "figure2": {
                "enabled": True,
                "models": "all",
                "grid_points": 51,
            },
            "figure3": {
                "enabled": True,
                "models": "all",
                "selection_metrics": ["roc_auc", "pr_auc", "f1"],
            },
            "figure4": {
                "enabled": True,
                "models": "all",
                "reuse_figure3_selection": True,
            },
        },
        "style": {"panel_width": 3.8, "panel_height": 3.2, "dpi": 72},
    }
    config_path = root / "visualization.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path, output


def assert_pdf(path: Path) -> None:
    assert path.is_file(), path
    assert path.read_bytes()[:4] == b"%PDF", path


def test_all_figures_and_audits() -> None:
    with tempfile.TemporaryDirectory(prefix="igh_batch10_") as directory:
        config_path, output = create_fixture(Path(directory))
        config = load_visualization_config(config_path)
        result = generate_visualizations(config)
        for filename in (
            "Figure1_model_performance_boxplots.pdf",
            "Figure2_mean_ROC_curves.pdf",
            "Figure3_best_repeat_ROC_curves.pdf",
            "Figure4_best_repeat_prediction_boxplots.pdf",
        ):
            assert_pdf(output / filename)
        assert len(pd.read_csv(output / "Figure1_plot_data.csv")) == 2 * 4 * 3
        coordinates = pd.read_csv(output / "Figure2_mean_ROC_coordinates.csv.gz")
        assert len(coordinates) == 2 * 51
        selection = pd.read_csv(output / "Figure3_best_repeat_selection.csv")
        assert len(selection) == 2
        figure4 = pd.read_csv(output / "Figure4_plot_data.csv")
        assert set(figure4["selection_source"]) == {"figure3"}
        audit = pd.read_csv(output / "visualization_input_audit.csv")
        assert audit["passed"].all()
        marker = json.loads((output / "VISUALIZATION_COMPLETE.json").read_text())
        assert marker["status"] == "COMPLETE"
        assert marker["model_refit_performed"] is False
        assert result.selected_best_repeats.shape[0] == 2


def test_unknown_model_is_rejected() -> None:
    with tempfile.TemporaryDirectory(prefix="igh_batch10_") as directory:
        config_path, _ = create_fixture(Path(directory))
        raw = yaml.safe_load(config_path.read_text())
        raw["figures"]["figure2"]["models"] = ["UNKNOWN_MODEL"]
        config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
        config = load_visualization_config(config_path)
        try:
            generate_visualizations(config)
        except VisualizationError as exc:
            assert "Unknown models" in str(exc)
        else:
            raise AssertionError("Unknown model was not rejected.")


def test_existing_output_requires_overwrite() -> None:
    with tempfile.TemporaryDirectory(prefix="igh_batch10_") as directory:
        config_path, _ = create_fixture(Path(directory))
        config = load_visualization_config(config_path)
        generate_visualizations(config)
        try:
            generate_visualizations(config)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing output was overwritten without permission.")
        generate_visualizations(config, overwrite_override=True)


def test_metric_tampering_is_rejected() -> None:
    with tempfile.TemporaryDirectory(prefix="igh_batch10_") as directory:
        config_path, _ = create_fixture(Path(directory))
        config = load_visualization_config(config_path)
        metrics_path = config.aggregation_dir / "07_all_outer_metrics.csv"
        frame = pd.read_csv(metrics_path)
        frame.loc[0, "roc_auc"] += 0.05
        frame.to_csv(metrics_path, index=False)
        try:
            generate_visualizations(config)
        except VisualizationError as exc:
            assert "do not reproduce" in str(exc)
        else:
            raise AssertionError("Metric tampering was not rejected.")


def main() -> int:
    tests = [
        test_all_figures_and_audits,
        test_unknown_model_is_rejected,
        test_existing_output_requires_overwrite,
        test_metric_tampering_is_rejected,
    ]
    passed = 0
    for test in tests:
        test()
        passed += 1
        print(f"PASS {test.__name__}")
    print(f"IGH Batch 10 focused tests: {passed}/{len(tests)} PASS")
    print("IGH_BATCH10_ACCEPTANCE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
