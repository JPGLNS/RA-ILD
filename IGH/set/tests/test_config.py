#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the IGH V2 configuration layer."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
SET_DIR = TEST_DIR.parent
SRC_DIR = SET_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.config import ConfigError, load_experiment_config, validate_experiment_mapping  # noqa: E402

CONFIG = SET_DIR / "configs" / "igh_baseline_m2_v1.yaml"


class TestBaselineConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_experiment_config(CONFIG)

    def test_identity(self) -> None:
        self.assertEqual(self.config.experiment_id, "igh_baseline_m2_v1")
        self.assertEqual(self.config.raw["experiment"]["receptor"], "IGH")

    def test_sample_counts(self) -> None:
        self.assertEqual(self.config.raw["data"]["train"]["expected_samples"], 120)
        self.assertEqual(self.config.raw["data"]["test"]["expected_samples"], 49)

    def test_reference_sizes(self) -> None:
        sizes = self.config.raw["public_reference"]["expected_sizes"]
        self.assertEqual(sizes["catalog"], 41052)
        self.assertEqual(sizes["global"], 6387)
        self.assertEqual(sizes["RA_specific"], 210)
        self.assertEqual(sizes["ILD_specific"], 504)
        self.assertEqual(sizes["shared"], 263)

    def test_public_cache_paths(self) -> None:
        cache = self.config.raw["public_reference"]["cache"]
        self.assertTrue(cache["presence"].endswith("05_train_public_presence.npz"))
        self.assertTrue(cache["frequency"].endswith("05_train_public_frequency.npz"))
        self.assertTrue(cache["metadata"].endswith("05_train_public_sparse_cache.json"))

    def test_public_regression_task(self) -> None:
        task = self.config.raw["public_reference"]["regression_task"]
        self.assertEqual(task["outer_repeat"], 1)
        self.assertEqual(task["outer_fold"], 1)
        self.assertTrue(task["task_dir"].endswith("repeat_01_fold_01"))

    def test_preprocessing_definition(self) -> None:
        preprocessing = self.config.raw["preprocessing"]
        self.assertEqual(
            preprocessing["numeric_standardization"],
            "zscore_population_ddof0",
        )
        self.assertEqual(
            preprocessing["categorical_encoding"],
            "sorted_reference_dummy",
        )
        self.assertTrue(preprocessing["zero_variance_filter"])
        self.assertEqual(
            preprocessing["unseen_category_policy"],
            "reference_all_zero_with_audit",
        )

    def test_preprocessing_regression_paths(self) -> None:
        task = self.config.raw["preprocessing"]["regression_task"]
        self.assertEqual(task["outer_repeat"], 1)
        self.assertEqual(task["outer_fold"], 1)
        self.assertTrue(task["preprocessing_summary"].endswith("05_preprocessing_summary.csv"))
        self.assertTrue(task["static_feature_list"].endswith("05_static_igh_feature_list.csv"))
        self.assertTrue(task["coefficients"].endswith("05_final_model_coefficients.csv"))

    def test_modeling_definition(self) -> None:
        modeling = self.config.raw["modeling"]
        self.assertEqual(modeling["positive_label"], "ILD")
        self.assertEqual(modeling["negative_label"], "RA")
        self.assertEqual(
            modeling["threshold_selection"],
            "youden_closest_to_0.5",
        )
        self.assertAlmostEqual(
            float(modeling["coefficient_nonzero_tolerance"]),
            1e-12,
        )

    def test_modeling_regression_paths(self) -> None:
        task = self.config.raw["modeling"]["regression_task"]
        self.assertEqual(task["outer_repeat"], 1)
        self.assertEqual(task["outer_fold"], 1)
        self.assertTrue(task["inner_selected_oof"].endswith(
            "05_inner_selected_oof_predictions.csv"
        ))
        self.assertTrue(task["outer_predictions"].endswith(
            "05_outer_validation_predictions.csv"
        ))
        self.assertTrue(task["independent_metrics"].endswith(
            "08_independent_test_metrics.csv"
        ))

    def test_model_selection_definition(self) -> None:
        selection = self.config.raw["model_selection"]
        self.assertEqual(
            selection["static_feature_group"],
            "static_igh_candidate_predictors",
        )
        self.assertEqual(selection["expected_static_feature_count"], 1083)
        self.assertEqual(selection["hyperparameter_grid_order"], "alpha_then_lambda")
        self.assertEqual(
            [item["field"] for item in selection["candidate_sort"]],
            [
                "pooled_inner_roc_auc",
                "pooled_inner_pr_auc",
                "lambda",
                "l1_ratio_alpha",
            ],
        )

    def test_model_selection_regression_path(self) -> None:
        path = self.config.raw["model_selection"]["regression_task"]["inner_tuning_results"]
        self.assertTrue(path.endswith("05_inner_tuning_results.csv"))

    def test_invalid_candidate_sort_rejected(self) -> None:
        broken = copy.deepcopy(self.config.raw)
        broken["model_selection"]["candidate_sort"][0]["ascending"] = True
        with self.assertRaises(ConfigError):
            validate_experiment_mapping(broken)

    def test_selected_pair_must_be_final_candidate(self) -> None:
        broken = copy.deepcopy(self.config.raw)
        broken["final_model"]["selected_lambda"] = 3
        with self.assertRaises(ConfigError):
            validate_experiment_mapping(broken)


    def test_nested_cv_definition(self) -> None:
        nested = self.config.raw["nested_cv"]
        self.assertEqual(nested["assignment_policy"], "fixed_precomputed")
        self.assertEqual(nested["training_public_policy"], "exact_leave_one_out")
        self.assertEqual(nested["validation_public_policy"], "training_reference_only")
        self.assertEqual(nested["expected_outer_tasks"], 100)
        self.assertEqual(nested["expected_inner_fits_per_task"], 480)

    def test_nested_cv_regression_paths(self) -> None:
        task = self.config.raw["nested_cv"]["regression_task"]
        self.assertEqual(task["outer_repeat"], 1)
        self.assertEqual(task["outer_fold"], 1)
        self.assertTrue(task["configuration"].endswith("05_trial_configuration.json"))
        self.assertTrue(task["sample_roles"].endswith("05_trial_sample_roles.csv"))

    def test_invalid_nested_fit_count_rejected(self) -> None:
        broken = copy.deepcopy(self.config.raw)
        broken["nested_cv"]["expected_inner_fits_per_task"] = 479
        with self.assertRaises(ConfigError):
            validate_experiment_mapping(broken)

    def test_outer_task_orchestration_definition(self) -> None:
        outer = self.config.raw["outer_tasks"]
        self.assertEqual(outer["expected_tasks"], 100)
        self.assertEqual(outer["default_workers"], 1)
        self.assertEqual(outer["max_workers"], 4)
        self.assertTrue(outer["runner_script"].endswith("run_nested_cv_task.py"))

    def test_aggregation_definition(self) -> None:
        aggregation = self.config.raw["aggregation"]
        self.assertTrue(aggregation["require_all_tasks"])
        self.assertEqual(aggregation["expected_metric_rows"], 400)
        self.assertEqual(aggregation["expected_predictions_per_sample"], 20)
        self.assertEqual(aggregation["expected_prediction_rows"], 9600)

    def test_invalid_outer_worker_bounds_rejected(self) -> None:
        broken = copy.deepcopy(self.config.raw)
        broken["outer_tasks"]["default_workers"] = 5
        broken["outer_tasks"]["max_workers"] = 4
        with self.assertRaises(ConfigError):
            validate_experiment_mapping(broken)

    def test_invalid_aggregation_prediction_count_rejected(self) -> None:
        broken = copy.deepcopy(self.config.raw)
        broken["aggregation"]["expected_prediction_rows"] = 9599
        with self.assertRaises(ConfigError):
            validate_experiment_mapping(broken)


    def test_locked_model(self) -> None:
        final = self.config.raw["final_model"]
        self.assertEqual(final["selected_model"], "M2_static_igh_public")
        self.assertAlmostEqual(float(final["selected_alpha"]), 0.9)
        self.assertAlmostEqual(float(final["selected_lambda"]), 10.0)
        self.assertAlmostEqual(float(final["locked_threshold"]), 0.4568009623694209)

    def test_model_set(self) -> None:
        self.assertEqual(
            set(self.config.raw["models"]),
            {
                "M0_clinical",
                "M1_static_igh",
                "M2_static_igh_public",
                "M3_static_igh_public_material",
            },
        )

    def test_relative_path_resolution(self) -> None:
        observed = self.config.path("data.train.base_matrix")
        expected = (
            self.config.repository_root
            / "IGH/set/train/result/04_final_feature_matrix/04_train_base_feature_matrix.csv"
        ).resolve()
        self.assertEqual(observed, expected)

    def test_invalid_alpha_rejected(self) -> None:
        broken = copy.deepcopy(self.config.raw)
        broken["model_engine"]["alpha_grid"] = [0.1, 1.2]
        with self.assertRaises(ConfigError):
            validate_experiment_mapping(broken)

    def test_unknown_final_model_rejected(self) -> None:
        broken = copy.deepcopy(self.config.raw)
        broken["final_model"]["selected_model"] = "M_NOT_DEFINED"
        with self.assertRaises(ConfigError):
            validate_experiment_mapping(broken)


if __name__ == "__main__":
    unittest.main(verbosity=2)
