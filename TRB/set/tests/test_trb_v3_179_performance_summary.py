#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for TRB V3 Repeat100 performance aggregation."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts_v2" / "summarize_trb_v3_179_repeat100_performance.py"
SPEC = importlib.util.spec_from_file_location("trb_v3_performance_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class TestTRBV3PerformanceSummary(unittest.TestCase):
    def test_interpolated_roc_has_expected_endpoints(self):
        y = np.array([0, 0, 1, 1])
        score = np.array([0.1, 0.4, 0.35, 0.8])
        curve = MODULE.interpolate_roc_curve(y, score)
        self.assertEqual(len(curve), len(MODULE.ROC_GRID))
        self.assertAlmostEqual(float(curve[0]), 0.0)
        self.assertAlmostEqual(float(curve[-1]), 1.0)
        self.assertTrue(np.all(np.diff(curve) >= -1e-12))

    def test_summary_uses_repeat_level_mean_and_sd(self):
        frame = pd.DataFrame({
            "cohort": ["total"] * 3,
            "algorithm": ["elastic_net"] * 3,
            "model": ["M0"] * 3,
            "outer_repeat": [1, 2, 3],
            "roc_auc": [0.6, 0.7, 0.8],
            "average_precision": [0.5, 0.6, 0.7],
        })
        summary = MODULE.summarize_repeat_metrics(frame)
        self.assertEqual(len(summary), 1)
        self.assertEqual(int(summary.loc[0, "n_repeats"]), 3)
        self.assertAlmostEqual(float(summary.loc[0, "roc_auc_mean"]), 0.7)
        self.assertAlmostEqual(float(summary.loc[0, "average_precision_mean"]), 0.6)
        self.assertAlmostEqual(
            float(summary.loc[0, "roc_auc_sd"]),
            float(np.std([0.6, 0.7, 0.8], ddof=1)),
        )

    def test_ap_column_aliases_are_supported(self):
        self.assertEqual(
            MODULE._ap_column(pd.DataFrame({"pr_auc": [0.5]}), Path("x.csv")),
            "pr_auc",
        )
        self.assertEqual(
            MODULE._ap_column(pd.DataFrame({"average_precision": [0.5]}), Path("x.csv")),
            "average_precision",
        )

    def test_rejects_single_class_roc_input(self):
        with self.assertRaises(MODULE.SummaryError):
            MODULE.interpolate_roc_curve([0, 0, 0], [0.1, 0.2, 0.3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
