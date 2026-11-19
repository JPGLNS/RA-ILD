#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for the TRB V3 Repeat100 terminal ROC-AUC summary."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts_v2"
    / "summarize_trb_v3_179_repeat100_performance.py"
)
SPEC = importlib.util.spec_from_file_location("trb_v3_performance_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class TestTRBV3PerformanceSummary(unittest.TestCase):
    def test_summary_uses_repeat_level_mean_and_sd(self):
        frame = pd.DataFrame({
            "model": ["M0"] * 3,
            "outer_repeat": [1, 2, 3],
            "outer_fold": [1, 1, 1],
            "roc_auc": [0.6, 0.7, 0.8],
        })
        summary = MODULE.summarize_model_metrics(
            frame,
            cohort="total",
            algorithm="elastic_net",
            expected_models=("M0",),
            expected_repeats=3,
        )
        self.assertEqual(len(summary), 1)
        self.assertEqual(int(summary.loc[0, "n_repeats"]), 3)
        self.assertAlmostEqual(float(summary.loc[0, "roc_auc_mean"]), 0.7)
        self.assertAlmostEqual(
            float(summary.loc[0, "roc_auc_sd"]),
            float(np.std([0.6, 0.7, 0.8], ddof=1)),
        )

    def test_rejects_missing_repeat(self):
        frame = pd.DataFrame({
            "model": ["M0", "M0"],
            "outer_repeat": [1, 3],
            "outer_fold": [1, 1],
            "roc_auc": [0.6, 0.8],
        })
        with self.assertRaises(MODULE.SummaryError):
            MODULE.summarize_model_metrics(
                frame,
                cohort="total",
                algorithm="elastic_net",
                expected_models=("M0",),
                expected_repeats=3,
            )

    def test_rejects_duplicate_repeat(self):
        frame = pd.DataFrame({
            "model": ["M0", "M0"],
            "outer_repeat": [1, 1],
            "outer_fold": [1, 1],
            "roc_auc": [0.6, 0.7],
        })
        with self.assertRaises(MODULE.SummaryError):
            MODULE.summarize_model_metrics(
                frame,
                cohort="total",
                algorithm="elastic_net",
                expected_models=("M0",),
                expected_repeats=2,
            )

    def test_checks_executed_outer_folds_not_outer_folds(self):
        # This design can retain an original outer_folds setting > 1 while
        # explicitly executing only fold 1 for each repeated holdout.
        MODULE.validate_repeat_holdout_config(
            {"outer_repeats": 100, "outer_folds": 5, "executed_outer_folds": [1]},
            cohort="total",
            algorithm="elastic_net",
            expected_repeats=100,
        )

    def test_rejects_unexpected_executed_fold_contract(self):
        with self.assertRaises(MODULE.SummaryError):
            MODULE.validate_repeat_holdout_config(
                {"outer_repeats": 100, "executed_outer_folds": [1, 2]},
                cohort="total",
                algorithm="elastic_net",
                expected_repeats=100,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
