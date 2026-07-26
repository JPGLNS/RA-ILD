#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for outer-task scheduling, aggregation, and stability analysis."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

TEST_DIR = Path(__file__).resolve().parent
SET_DIR = TEST_DIR.parent
SRC_DIR = SET_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.outer_cv import (  # noqa: E402
    OuterCVError,
    aggregate_outer_results,
    build_outer_tasks,
    inspect_outer_task,
    manifest_frame,
    scan_outer_tasks,
    task_command,
    task_output_paths,
    write_outer_aggregate,
)


MODELS = ("M0_clinical", "M1_static_igh")
SAMPLES = ("S1", "S2", "S3", "S4")
TRUTH = {"S1": 0, "S2": 1, "S3": 0, "S4": 1}


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_complete_task(task, *, coefficient_variant: int = 0) -> None:
    paths = task_output_paths(task.output_dir)
    task.output_dir.mkdir(parents=True, exist_ok=True)
    valid_ids = SAMPLES[:2] if task.outer_fold == 1 else SAMPLES[2:]
    train_ids = tuple(sample for sample in SAMPLES if sample not in valid_ids)
    config = {
        "outer_repeat": task.outer_repeat,
        "outer_fold": task.outer_fold,
        "outer_train_samples": len(train_ids),
        "outer_validation_samples": len(valid_ids),
        "models": list(MODELS),
        "candidate_count_per_model": 2,
        "runtime_seconds": 10.0 + task.task_index,
    }
    paths["configuration"].write_text(json.dumps(config), encoding="utf-8")
    marker = {
        "status": "COMPLETE",
        "outer_repeat": task.outer_repeat,
        "outer_fold": task.outer_fold,
        "runtime_seconds": 10.0 + task.task_index,
    }
    paths["complete"].write_text(json.dumps(marker), encoding="utf-8")

    roles = pd.DataFrame(
        {
            "sample_id": SAMPLES,
            "cohort": ["RA" if TRUTH[s] == 0 else "ILD" for s in SAMPLES],
            "outer_repeat": task.outer_repeat,
            "outer_fold": task.outer_fold,
            "outer_role": ["validation" if s in valid_ids else "training" for s in SAMPLES],
        }
    )
    _write_csv(paths["sample_roles"], roles)

    metric_rows = []
    prediction_rows = []
    for model_index, model in enumerate(MODELS):
        alpha = 0.1 if (task.task_index + model_index) % 2 else 0.5
        lambda_value = 1.0 if alpha == 0.1 else 10.0
        metric_rows.append(
            {
                "model": model,
                "outer_repeat": task.outer_repeat,
                "outer_fold": task.outer_fold,
                "roc_auc": 0.60 + 0.02 * task.task_index + 0.03 * model_index,
                "pr_auc": 0.55 + 0.02 * task.task_index + 0.03 * model_index,
                "accuracy": 0.5 + 0.1 * model_index,
                "sensitivity_recall": 0.5 + 0.1 * model_index,
                "specificity": 0.5 + 0.1 * model_index,
                "precision": 0.5 + 0.1 * model_index,
                "f1": 0.5 + 0.1 * model_index,
                "TN": 1,
                "FP": 0,
                "FN": 1,
                "TP": 0,
                "selected_l1_ratio_alpha": alpha,
                "selected_lambda": lambda_value,
                "inner_selected_roc_auc": 0.7 + 0.01 * task.task_index,
                "inner_selected_pr_auc": 0.65 + 0.01 * task.task_index,
            }
        )
        for sample in valid_ids:
            base_probability = {
                "S1": 0.2,
                "S2": 0.8,
                "S3": 0.35,
                "S4": 0.65,
            }[sample]
            probability = min(0.99, max(0.01, base_probability + 0.02 * model_index + 0.005 * task.outer_repeat))
            predicted = int(probability >= 0.5)
            prediction_rows.append(
                {
                    "model": model,
                    "outer_repeat": task.outer_repeat,
                    "outer_fold": task.outer_fold,
                    "sample_id": sample,
                    "true_label": TRUTH[sample],
                    "true_cohort": "ILD" if TRUTH[sample] else "RA",
                    "probability_ILD": probability,
                    "threshold": 0.5,
                    "predicted_label": predicted,
                    "predicted_cohort": "ILD" if predicted else "RA",
                }
            )
    _write_csv(paths["outer_metrics"], pd.DataFrame(metric_rows))
    _write_csv(paths["outer_predictions"], pd.DataFrame(prediction_rows))

    tuning_rows = []
    for model in MODELS:
        for alpha, lambda_value in ((0.1, 1.0), (0.5, 10.0)):
            tuning_rows.append(
                {
                    "model": model,
                    "l1_ratio_alpha": alpha,
                    "lambda": lambda_value,
                    "C_inverse_lambda": 1.0 / lambda_value,
                }
            )
    _write_csv(paths["inner_tuning"], pd.DataFrame(tuning_rows))

    oof_rows = []
    for model in MODELS:
        for index, sample in enumerate(train_ids):
            oof_rows.append(
                {
                    "model": model,
                    "inner_fold": index + 1,
                    "sample_id": sample,
                    "true_label": TRUTH[sample],
                    "probability": 0.75 if TRUTH[sample] else 0.25,
                    "selected_threshold": 0.5,
                }
            )
    _write_csv(paths["inner_oof"], pd.DataFrame(oof_rows))

    f1_coefficient = 0.4 if coefficient_variant < 3 else 0.0
    coefficients = pd.DataFrame(
        [
            {"model": "M0_clinical", "feature_name": "__INTERCEPT__", "coefficient": -0.1, "absolute_coefficient": 0.1, "nonzero": True},
            {"model": "M0_clinical", "feature_name": "age", "coefficient": 0.2, "absolute_coefficient": 0.2, "nonzero": True},
            {"model": "M1_static_igh", "feature_name": "__INTERCEPT__", "coefficient": -0.2, "absolute_coefficient": 0.2, "nonzero": True},
            {"model": "M1_static_igh", "feature_name": "age", "coefficient": 0.1, "absolute_coefficient": 0.1, "nonzero": True},
            {"model": "M1_static_igh", "feature_name": "f1", "coefficient": f1_coefficient, "absolute_coefficient": abs(f1_coefficient), "nonzero": bool(f1_coefficient)},
        ]
    )
    _write_csv(paths["coefficients"], coefficients)
    _write_csv(
        paths["static_feature_list"],
        pd.DataFrame({"feature_order": [1, 2, 3], "feature_name": ["f1", "f2", "f3"]}),
    )

    # Files required for task completeness but not read by the aggregate layer.
    _write_csv(paths["outer_train_public"], pd.DataFrame({"sample_id": train_ids}))
    _write_csv(paths["outer_validation_public"], pd.DataFrame({"sample_id": valid_ids}))
    _write_csv(paths["public_reference_summary"], pd.DataFrame({"context": ["outer"]}))
    _write_csv(paths["public_loo_assignments"], pd.DataFrame({"context": ["outer"]}))
    _write_csv(paths["preprocessing"], pd.DataFrame({"feature_name": ["age"]}))


class TestOuterCV(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tasks = build_outer_tasks(2, 2, self.root / "tasks")
        self.runner = self.root / "run_nested_cv_task.py"
        self.runner.write_text("print('stub')\n", encoding="utf-8")
        self.config = self.root / "config.yaml"
        self.config.write_text("stub: true\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_task_manifest_order_and_commands(self) -> None:
        manifest = manifest_frame(
            self.tasks,
            python_executable="python3",
            runner_script=self.runner,
            config_path=self.config,
        )
        self.assertEqual(manifest["task_id"].tolist(), [
            "repeat_01_fold_01", "repeat_01_fold_02",
            "repeat_02_fold_01", "repeat_02_fold_02",
        ])
        self.assertIn("--outer-repeat 2", manifest.iloc[-1]["command"])
        self.assertIn("--outer-fold 2", manifest.iloc[-1]["command"])

    def test_task_command_overwrite(self) -> None:
        command = task_command(
            self.tasks[0],
            python_executable="python3",
            runner_script=self.runner,
            config_path=self.config,
            overwrite=True,
        )
        self.assertEqual(command[-1], "--overwrite")

    def test_missing_task_inspection(self) -> None:
        result = inspect_outer_task(
            self.tasks[0],
            expected_models=MODELS,
            expected_samples=4,
            expected_candidates_per_model=2,
            expected_static_features=3,
        )
        self.assertEqual(result.status, "missing")

    def test_complete_task_inspection(self) -> None:
        write_complete_task(self.tasks[0])
        result = inspect_outer_task(
            self.tasks[0],
            expected_models=MODELS,
            expected_samples=4,
            expected_candidates_per_model=2,
            expected_static_features=3,
        )
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.n_outer_train, 2)
        self.assertEqual(result.n_outer_validation, 2)

    def test_incomplete_task_inspection(self) -> None:
        self.tasks[0].output_dir.mkdir(parents=True)
        result = inspect_outer_task(
            self.tasks[0],
            expected_models=MODELS,
            expected_samples=4,
            expected_candidates_per_model=2,
            expected_static_features=3,
        )
        self.assertEqual(result.status, "incomplete")

    def _complete_all(self):
        for index, task in enumerate(self.tasks):
            write_complete_task(task, coefficient_variant=index)
        return scan_outer_tasks(
            self.tasks,
            expected_models=MODELS,
            expected_samples=4,
            expected_candidates_per_model=2,
            expected_static_features=3,
        )

    def test_scan_status_counts(self) -> None:
        write_complete_task(self.tasks[0])
        status = scan_outer_tasks(
            self.tasks,
            expected_models=MODELS,
            expected_samples=4,
            expected_candidates_per_model=2,
            expected_static_features=3,
        )
        self.assertEqual(int((status["status"] == "complete").sum()), 1)
        self.assertEqual(int((status["status"] == "missing").sum()), 3)

    def test_full_aggregation_dimensions(self) -> None:
        status = self._complete_all()
        manifest = manifest_frame(
            self.tasks,
            python_executable="python3",
            runner_script=self.runner,
            config_path=self.config,
        )
        result = aggregate_outer_results(
            self.tasks,
            status,
            expected_models=MODELS,
            expected_samples=4,
            expected_repeats=2,
            require_all_tasks=True,
            coefficient_tolerance=1e-12,
            minimum_selection_frequency=0.5,
            minimum_sign_consistency=0.8,
            manifest=manifest,
        )
        self.assertEqual(len(result.all_metrics), 8)
        self.assertEqual(len(result.all_predictions), 16)
        self.assertEqual(len(result.repeat_metrics), 4)
        self.assertTrue((result.sample_prediction_stability["n_predictions"] == 2).all())
        self.assertEqual(len(result.all_hyperparameters), 8)

    def test_hyperparameter_frequencies_sum_to_one(self) -> None:
        status = self._complete_all()
        result = aggregate_outer_results(
            self.tasks,
            status,
            expected_models=MODELS,
            expected_samples=4,
            expected_repeats=2,
            require_all_tasks=True,
            coefficient_tolerance=1e-12,
            minimum_selection_frequency=0.5,
            minimum_sign_consistency=0.8,
        )
        sums = result.hyperparameter_frequency.groupby("model")["selection_frequency"].sum()
        np.testing.assert_allclose(sums.to_numpy(), np.ones(len(MODELS)))

    def test_feature_stability_thresholds(self) -> None:
        status = self._complete_all()
        result = aggregate_outer_results(
            self.tasks,
            status,
            expected_models=MODELS,
            expected_samples=4,
            expected_repeats=2,
            require_all_tasks=True,
            coefficient_tolerance=1e-12,
            minimum_selection_frequency=0.5,
            minimum_sign_consistency=0.8,
        )
        row = result.feature_stability.set_index(["model", "feature_name"]).loc[("M1_static_igh", "f1")]
        self.assertAlmostEqual(float(row["selection_frequency"]), 0.75)
        self.assertAlmostEqual(float(row["sign_consistency"]), 1.0)
        self.assertTrue(bool(row["stable_selection"]))

    def test_partial_aggregation_allowed(self) -> None:
        write_complete_task(self.tasks[0])
        status = scan_outer_tasks(
            self.tasks,
            expected_models=MODELS,
            expected_samples=4,
            expected_candidates_per_model=2,
            expected_static_features=3,
        )
        result = aggregate_outer_results(
            self.tasks,
            status,
            expected_models=MODELS,
            expected_samples=4,
            expected_repeats=2,
            require_all_tasks=False,
            coefficient_tolerance=1e-12,
            minimum_selection_frequency=0.5,
            minimum_sign_consistency=0.8,
        )
        self.assertTrue(result.repeat_metrics.empty)
        self.assertEqual(len(result.all_metrics), 2)

    def test_partial_aggregation_rejected_when_required(self) -> None:
        write_complete_task(self.tasks[0])
        status = scan_outer_tasks(
            self.tasks,
            expected_models=MODELS,
            expected_samples=4,
            expected_candidates_per_model=2,
            expected_static_features=3,
        )
        with self.assertRaises(OuterCVError):
            aggregate_outer_results(
                self.tasks,
                status,
                expected_models=MODELS,
                expected_samples=4,
                expected_repeats=2,
                require_all_tasks=True,
                coefficient_tolerance=1e-12,
                minimum_selection_frequency=0.5,
                minimum_sign_consistency=0.8,
            )

    def test_invalid_task_blocks_aggregation(self) -> None:
        status = self._complete_all()
        status.loc[0, "status"] = "invalid"
        with self.assertRaises(OuterCVError):
            aggregate_outer_results(
                self.tasks,
                status,
                expected_models=MODELS,
                expected_samples=4,
                expected_repeats=2,
                require_all_tasks=False,
                coefficient_tolerance=1e-12,
                minimum_selection_frequency=0.5,
                minimum_sign_consistency=0.8,
            )

    def test_write_aggregate_and_overwrite_guard(self) -> None:
        status = self._complete_all()
        manifest = manifest_frame(
            self.tasks,
            python_executable="python3",
            runner_script=self.runner,
            config_path=self.config,
        )
        result = aggregate_outer_results(
            self.tasks,
            status,
            expected_models=MODELS,
            expected_samples=4,
            expected_repeats=2,
            require_all_tasks=True,
            coefficient_tolerance=1e-12,
            minimum_selection_frequency=0.5,
            minimum_sign_consistency=0.8,
            manifest=manifest,
        )
        output = self.root / "summary"
        paths = write_outer_aggregate(
            result,
            output,
            experiment_id="test",
            expected_tasks=4,
            require_all_tasks=True,
        )
        self.assertTrue(paths["complete"].is_file())
        self.assertTrue(paths["all_predictions"].is_file())
        with self.assertRaises(FileExistsError):
            write_outer_aggregate(
                result,
                output,
                experiment_id="test",
                expected_tasks=4,
                require_all_tasks=True,
            )

    def test_invalid_task_dimensions_detected(self) -> None:
        write_complete_task(self.tasks[0])
        paths = task_output_paths(self.tasks[0].output_dir)
        metrics = pd.read_csv(paths["outer_metrics"]).iloc[:1]
        metrics.to_csv(paths["outer_metrics"], index=False)
        result = inspect_outer_task(
            self.tasks[0],
            expected_models=MODELS,
            expected_samples=4,
            expected_candidates_per_model=2,
            expected_static_features=3,
        )
        self.assertEqual(result.status, "invalid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
