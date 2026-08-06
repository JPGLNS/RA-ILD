#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused synthetic acceptance test for IGH Batch 14 Linear SVM."""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "IGH/set/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ra_ild_igh.linear_svm import (  # noqa: E402
    LinearSVMCandidate,
    choose_youden_threshold,
    fit_linear_svm,
    score_classification_metrics,
)
from ra_ild_igh.linear_svm_config import (  # noqa: E402
    LinearSVMConfigError,
    expand_linear_svm_candidates,
)
from ra_ild_igh.linear_svm_summary import write_aggregation  # noqa: E402
from ra_ild_igh.linear_svm_visualization import create_visualizations  # noqa: E402


def _assert_baseline_import_compatibility() -> None:
    baseline = SRC / "ra_ild_igh/nested_cv.py"
    if baseline.is_file():
        module = importlib.import_module("ra_ild_igh.linear_svm_nested_cv")
        assert hasattr(module, "run_linear_svm_outer_task")
        runner = ROOT / "IGH/set/scripts_v2/run_linear_svm_nested_cv_task.py"
        assert runner.is_file()
    print("PASS baseline IGH V2 import compatibility")


def _test_candidate_and_fit() -> None:
    engine = {
        "engine": "linear_svc",
        "candidate_blocks": [
            {
                "id": "main",
                "priority": 1,
                "penalty": "l2",
                "loss": "squared_hinge",
                "class_weight_grid": ["balanced"],
                "C_grid": [0.001, 0.01, 0.1, 1.0],
                "dual": "auto",
            }
        ],
    }
    candidates = expand_linear_svm_candidates(engine)
    assert len(candidates) == 4
    try:
        expand_linear_svm_candidates(
            {
                "engine": "linear_svc",
                "candidate_blocks": [
                    {
                        "id": "invalid",
                        "penalty": "l1",
                        "loss": "hinge",
                        "class_weight_grid": ["balanced"],
                        "C_grid": [1.0],
                        "dual": "auto",
                    }
                ],
            }
        )
    except LinearSVMConfigError:
        pass
    else:  # pragma: no cover
        raise AssertionError("Illegal l1+hinge combination was accepted")

    rng = np.random.default_rng(20260804)
    X = np.vstack(
        [
            rng.normal(-0.8, 0.6, size=(40, 8)),
            rng.normal(0.8, 0.6, size=(40, 8)),
        ]
    )
    y = np.asarray([0] * 40 + [1] * 40, dtype=int)
    fit = fit_linear_svm(
        X,
        y,
        candidate=candidates[2],
        max_iter=10000,
        random_state=2026,
    )
    assert fit.converged
    score = fit.decision_score(X)
    threshold, youden = choose_youden_threshold(y, score)
    metrics = score_classification_metrics(y, score, threshold)
    assert metrics["roc_auc"] > 0.95
    assert metrics["average_precision"] > 0.95
    assert youden > 0.80
    print("PASS candidate expansion, legal-combination guard and LinearSVC fit")


def _write_task(task_dir: Path, repeat: int, model_offsets: dict[str, float]) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    prediction_rows = []
    coefficient_rows = []
    for model_index, (model, offset) in enumerate(model_offsets.items()):
        truth = np.asarray([0] * 10 + [1] * 10, dtype=int)
        base = np.linspace(-2.0, 2.0, len(truth))
        score = base + offset + (repeat - 2) * 0.03
        threshold, youden = choose_youden_threshold(truth, score)
        metrics = score_classification_metrics(truth, score, threshold)
        metrics.update(
            {
                "model": model,
                "model_definition": f"Synthetic {model}",
                "model_engine": "linear_svc",
                "score_type": "decision_function",
                "threshold_source": "inner_oof_youden",
                "selected_threshold": threshold,
                "selected_youden_j": youden,
                "outer_repeat": repeat,
                "outer_fold": 1,
                "selected_candidate_id": f"main__C_{0.1 + model_index}",
                "selected_candidate_block": "main",
                "selected_C": 0.1 + model_index,
                "selected_penalty": "l2",
                "selected_loss": "squared_hinge",
                "selected_class_weight": "balanced",
                "selected_dual_requested": "auto",
                "selected_dual_resolved": False,
                "probability_metrics_available": False,
                "log_loss": np.nan,
                "brier_score": np.nan,
            }
        )
        metric_rows.append(metrics)
        for index, (label, value) in enumerate(zip(truth, score), start=1):
            prediction_rows.append(
                {
                    "model": model,
                    "model_engine": "linear_svc",
                    "outer_repeat": repeat,
                    "outer_fold": 1,
                    "sample_id": f"R{repeat:02d}_{index:03d}",
                    "true_label": int(label),
                    "true_cohort": "ILD" if label else "RA",
                    "prediction_score": float(value),
                    "decision_score_ILD": float(value),
                    "score_type": "decision_function",
                    "threshold": float(threshold),
                    "threshold_source": "inner_oof_youden",
                }
            )
        for feature_index in range(4):
            value = (feature_index + 1) * (1 if model_index == 0 else -1) * 0.1
            coefficient_rows.append(
                {
                    "model": model,
                    "feature_name": f"feature_{feature_index}",
                    "coefficient": value,
                    "absolute_coefficient": abs(value),
                    "nonzero": True,
                }
            )
    pd.DataFrame(metric_rows).to_csv(
        task_dir / "06_outer_validation_metrics.csv", index=False
    )
    pd.DataFrame(prediction_rows).to_csv(
        task_dir / "06_outer_validation_predictions.csv", index=False
    )
    pd.DataFrame(coefficient_rows).to_csv(
        task_dir / "06_final_model_coefficients.csv", index=False
    )
    (task_dir / "06_task_configuration.json").write_text(
        json.dumps({"outer_repeat": repeat, "outer_fold": 1}) + "\n",
        encoding="utf-8",
    )
    (task_dir / "06_TASK_COMPLETE.json").write_text(
        json.dumps(
            {"status": "COMPLETE", "outer_repeat": repeat, "outer_fold": 1}
        )
        + "\n",
        encoding="utf-8",
    )



def _test_scheme_preparation_and_status(temp_root: Path) -> None:
    temp_root.mkdir(parents=True, exist_ok=True)
    base_path = temp_root / "base_resolved.yaml"
    scheme_path = temp_root / "svm_scheme.yaml"
    output_root = temp_root / "svm_experiment"
    base = {
        "schema_version": "1.0",
        "experiment": {
            "id": "base",
            "description": "Synthetic base",
            "receptor": "IGH",
            "random_seed": 20260804,
        },
        "data": {"train": {}, "test": {}},
        "public_reference": {},
        "cross_validation": {
            "outer_repeats": 3,
            "outer_folds": 2,
            "inner_folds": 5,
            "executed_outer_folds": [1],
        },
        "models": {
            "M0_clinical": {
                "description": "age and sex",
                "numeric": ["age"],
                "categorical": ["sex"],
                "static_feature_groups": [],
                "dynamic_public": [],
            },
            "M2_static_igh_public": {
                "description": "static and public",
                "numeric": [],
                "categorical": [],
                "static_feature_groups": ["core83"],
                "dynamic_public": ["all_ref_public_clone_ratio"],
            },
        },
        "preprocessing": {},
        "modeling": {
            "positive_label": "ILD",
            "negative_label": "RA",
            "coefficient_nonzero_tolerance": 1.0e-12,
        },
        "model_selection": {"static_feature_group": "static_igh_candidate_predictors"},
        "nested_cv": {"regression_task": {}},
        "outer_tasks": {
            "output_root": "unused",
            "manifest": "unused",
            "status": "unused",
            "logs_dir": "unused",
            "runner_script": "unused",
            "expected_tasks": 0,
        },
    }
    base_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    scheme = {
        "linear_svm_scheme_version": "1.0",
        "scheme": {
            "id": "synthetic_linear_svm",
            "description": "Synthetic scheme-preparation acceptance",
            "base_resolved_config": str(base_path),
            "output_root": str(output_root),
        },
        "model_engine": {
            "engine": "linear_svc",
            "candidate_blocks": [
                {
                    "id": "main",
                    "priority": 1,
                    "penalty": "l2",
                    "loss": "squared_hinge",
                    "class_weight_grid": ["balanced"],
                    "C_grid": [0.01, 0.1, 1.0],
                    "dual": "auto",
                }
            ],
            "max_iter": 10000,
            "tolerance": 1.0e-4,
            "zero_sd_tolerance": 1.0e-12,
            "fit_intercept": True,
            "intercept_scaling": 1.0,
        },
    }
    scheme_path.write_text(yaml.safe_dump(scheme, sort_keys=False), encoding="utf-8")
    preparer = ROOT / "IGH/set/scripts_v2/prepare_linear_svm_scheme.py"
    subprocess.run(
        [
            sys.executable,
            str(preparer),
            "--scheme",
            str(scheme_path),
            "--repository-root",
            str(ROOT),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    resolved_path = output_root / "00_config/resolved_config.yaml"
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    assert resolved["model_engine"]["engine"] == "linear_svc"
    assert resolved["model_selection"]["expected_candidate_count"] == 3
    assert resolved["outer_tasks"]["expected_tasks"] == 3
    assert set(resolved["models"]) == {"M0_clinical", "M2_static_igh_public"}
    orchestrator = ROOT / "IGH/set/scripts_v2/run_all_linear_svm_tasks.py"
    subprocess.run(
        [
            sys.executable,
            str(orchestrator),
            "--config",
            str(resolved_path),
            "--repository-root",
            str(ROOT),
            "--status-only",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    status = pd.read_csv(output_root / "00_config/outer_task_status.csv")
    assert len(status) == 3
    assert set(status["status"]) == {"missing"}
    print("PASS scheme inheritance, candidate counts, repeat counts and worker manifest")


def _test_aggregation_and_visualization(temp_root: Path) -> None:
    task_root = temp_root / "01_outer_tasks"
    for repeat in range(1, 4):
        _write_task(
            task_root / f"repeat_{repeat:02d}_fold_01",
            repeat,
            {"M0_clinical": -0.05, "M2_static_igh_public": 0.05},
        )
    summary_dir = temp_root / "03_summary"
    paths = write_aggregation(task_root, summary_dir)
    assert paths["metrics"].is_file()
    metrics = pd.read_csv(paths["metrics"])
    assert len(metrics) == 6
    assert set(metrics["model"]) == {"M0_clinical", "M2_static_igh_public"}

    visual_dir = temp_root / "06_visualization/main"
    visual_config = temp_root / "visualization.yaml"
    visual_config.write_text(
        yaml.safe_dump(
            {
                "visualization": {
                    "repository_root": str(temp_root),
                    "aggregation_dir": str(summary_dir),
                    "output_dir": str(visual_dir),
                    "models": "all",
                    "performance_metrics": [
                        "roc_auc",
                        "average_precision",
                        "f1",
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    figures = create_visualizations(visual_config)
    for key in ("figure1", "figure2", "figure3", "figure4"):
        assert figures[key].is_file()
        assert figures[key].read_bytes().startswith(b"%PDF")
    print("PASS repeated-holdout aggregation and four score-based figures")


def _test_probability_metrics_are_absent() -> None:
    source = (SRC / "ra_ild_igh/linear_svm_nested_cv.py").read_text(encoding="utf-8")
    assert "predict_proba" not in source
    assert '"probability_metrics_available": False' in source
    assert '"log_loss": np.nan' in source
    assert '"brier_score": np.nan' in source
    print("PASS uncalibrated SVM excludes probability-only metrics")


def main() -> int:
    _assert_baseline_import_compatibility()
    _test_candidate_and_fit()
    with tempfile.TemporaryDirectory(prefix="igh_batch15_acceptance_") as temp:
        temp_root = Path(temp)
        _test_scheme_preparation_and_status(temp_root / "scheme")
        _test_aggregation_and_visualization(temp_root / "results")
    _test_probability_metrics_are_absent()
    print("IGH_BATCH14_LINEAR_SVM_ACCEPTANCE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
