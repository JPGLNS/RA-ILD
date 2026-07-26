#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for IGH V2 binary metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.metrics import (  # noqa: E402
    BOOTSTRAP_METRICS,
    MetricError,
    classification_metrics,
    safe_pr_auc,
    safe_roc_auc,
    stratified_bootstrap_ci,
)


class TestMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
        self.p = np.array([0.05, 0.30, 0.55, 0.60, 0.40, 0.65, 0.80, 0.95])
        self.threshold = 0.60

    def test_metrics_match_v1_sklearn_calculation(self) -> None:
        observed = classification_metrics(self.y, self.p, self.threshold)
        prediction = (self.p >= self.threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            self.y, prediction, labels=[0, 1]
        ).ravel()
        self.assertAlmostEqual(observed["roc_auc"], roc_auc_score(self.y, self.p))
        self.assertAlmostEqual(
            observed["pr_auc"], average_precision_score(self.y, self.p)
        )
        self.assertAlmostEqual(
            observed["accuracy"], accuracy_score(self.y, prediction)
        )
        self.assertAlmostEqual(
            observed["sensitivity_recall"],
            recall_score(self.y, prediction, zero_division=0),
        )
        self.assertAlmostEqual(observed["specificity"], tn / (tn + fp))
        self.assertAlmostEqual(
            observed["precision"],
            precision_score(self.y, prediction, zero_division=0),
        )
        self.assertAlmostEqual(
            observed["f1"], f1_score(self.y, prediction, zero_division=0)
        )
        self.assertEqual(
            (observed["TN"], observed["FP"], observed["FN"], observed["TP"]),
            (tn, fp, fn, tp),
        )

    def test_brier_and_sample_summary_are_optional(self) -> None:
        basic = classification_metrics(self.y, self.p, self.threshold)
        self.assertNotIn("brier_score", basic)
        observed = classification_metrics(
            self.y,
            self.p,
            self.threshold,
            include_brier=True,
            include_sample_summary=True,
        )
        self.assertAlmostEqual(observed["brier_score"], brier_score_loss(self.y, self.p))
        self.assertEqual(observed["n_test"], 8)
        self.assertEqual(observed["n_RA"], 4)
        self.assertEqual(observed["n_ILD"], 4)
        self.assertAlmostEqual(observed["positive_prevalence"], 0.5)

    def test_auc_helpers(self) -> None:
        self.assertAlmostEqual(safe_roc_auc(self.y, self.p), roc_auc_score(self.y, self.p))
        self.assertAlmostEqual(
            safe_pr_auc(self.y, self.p), average_precision_score(self.y, self.p)
        )

    def test_bootstrap_is_reproducible(self) -> None:
        first = stratified_bootstrap_ci(
            self.y,
            self.p,
            self.threshold,
            reps=100,
            seed=20260713,
        )
        second = stratified_bootstrap_ci(
            self.y,
            self.p,
            self.threshold,
            reps=100,
            seed=20260713,
        )
        self.assertEqual(first["metric"].tolist(), list(BOOTSTRAP_METRICS))
        np.testing.assert_allclose(
            first.select_dtypes(include=["number"]).to_numpy(),
            second.select_dtypes(include=["number"]).to_numpy(),
            rtol=0,
            atol=0,
        )

    def test_bootstrap_preserves_observed_estimates(self) -> None:
        ci = stratified_bootstrap_ci(
            self.y,
            self.p,
            self.threshold,
            reps=100,
            seed=5,
        ).set_index("metric")
        observed = classification_metrics(
            self.y, self.p, self.threshold, include_brier=True
        )
        for metric in BOOTSTRAP_METRICS:
            self.assertAlmostEqual(ci.loc[metric, "estimate"], observed[metric])
            self.assertLessEqual(ci.loc[metric, "ci95_lower"], ci.loc[metric, "ci95_upper"])

    def test_invalid_labels_are_rejected(self) -> None:
        with self.assertRaises(MetricError):
            classification_metrics(np.zeros(4, dtype=int), np.linspace(0.1, 0.4, 4), 0.5)

    def test_invalid_probabilities_are_rejected(self) -> None:
        with self.assertRaises(MetricError):
            classification_metrics(self.y, np.r_[self.p[:-1], np.nan], 0.5)

    def test_invalid_bootstrap_arguments_are_rejected(self) -> None:
        with self.assertRaises(MetricError):
            stratified_bootstrap_ci(
                self.y,
                self.p,
                self.threshold,
                reps=99,
                seed=1,
            )
        with self.assertRaises(MetricError):
            stratified_bootstrap_ci(
                self.y,
                self.p,
                self.threshold,
                reps=100,
                seed=1,
                metric_names=["unknown"],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
