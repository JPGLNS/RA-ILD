#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the TRB V2 configuration layer."""

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

from ra_ild_trb.config import ConfigError, load_experiment_config, validate_experiment_mapping  # noqa: E402

CONFIG = SET_DIR / "configs" / "trb_baseline_m2_v1.yaml"


class TestBaselineConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_experiment_config(CONFIG)

    def test_identity(self) -> None:
        self.assertEqual(self.config.experiment_id, "trb_baseline_m2_v1")
        self.assertEqual(self.config.raw["experiment"]["receptor"], "TRB")

    def test_sample_counts(self) -> None:
        self.assertEqual(self.config.raw["data"]["train"]["expected_samples"], 123)
        self.assertEqual(self.config.raw["data"]["test"]["expected_samples"], 51)

    def test_reference_sizes(self) -> None:
        sizes = self.config.raw["public_reference"]["expected_sizes"]
        self.assertEqual(sizes["catalog"], 1000139)
        self.assertEqual(sizes["global"], 394235)
        self.assertEqual(sizes["RA_specific"], 41649)
        self.assertEqual(sizes["ILD_specific"], 28947)
        self.assertEqual(sizes["shared"], 38585)

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
        self.assertTrue(task["static_feature_list"].endswith("05_static_tcr_feature_list.csv"))
        self.assertTrue(task["coefficients"].endswith("05_final_model_coefficients.csv"))

    def test_locked_model(self) -> None:
        final = self.config.raw["final_model"]
        self.assertEqual(final["selected_model"], "M2_static_tcr_public")
        self.assertAlmostEqual(float(final["selected_alpha"]), 0.5)
        self.assertAlmostEqual(float(final["selected_lambda"]), 30.0)
        self.assertAlmostEqual(float(final["locked_threshold"]), 0.509337)

    def test_model_set(self) -> None:
        self.assertEqual(
            set(self.config.raw["models"]),
            {
                "M0_clinical",
                "M1_static_tcr",
                "M2_static_tcr_public",
                "M3_static_tcr_public_material",
            },
        )

    def test_relative_path_resolution(self) -> None:
        observed = self.config.path("data.train.base_matrix")
        expected = (
            self.config.repository_root
            / "TRB/set/train/result/04_final_feature_matrix/04_train_base_feature_matrix.csv"
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
