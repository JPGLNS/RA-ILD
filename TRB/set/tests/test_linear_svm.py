#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest

import numpy as np

from ra_ild_trb.linear_svm import (
    LinearSVMCandidate,
    LinearSVMError,
    add_natural_zero_metrics,
    choose_youden_threshold,
    fit_linear_svm,
    score_classification_metrics,
)


class LinearSVMTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(20260804)
        negative = rng.normal(loc=-1.0, scale=0.5, size=(30, 5))
        positive = rng.normal(loc=1.0, scale=0.5, size=(30, 5))
        self.X = np.vstack([negative, positive])
        self.y = np.asarray([0] * 30 + [1] * 30, dtype=int)

    def candidate(self, **overrides):
        values = {
            "candidate_id": "main__C_0p1",
            "block_id": "main",
            "priority": 1,
            "C": 0.1,
            "penalty": "l2",
            "loss": "squared_hinge",
            "class_weight": "balanced",
            "dual": "auto",
        }
        values.update(overrides)
        return LinearSVMCandidate(**values)

    def test_linear_svm_fit_and_decision_score(self) -> None:
        fit = fit_linear_svm(
            self.X,
            self.y,
            candidate=self.candidate(),
            random_state=17,
            max_iter=10000,
        )
        score = fit.decision_score(self.X)
        self.assertEqual(score.shape, (60,))
        self.assertTrue(np.isfinite(score).all())
        self.assertTrue(fit.converged)
        threshold, youden = choose_youden_threshold(self.y, score)
        metrics = score_classification_metrics(self.y, score, threshold)
        metrics = add_natural_zero_metrics(metrics, self.y, score)
        self.assertGreaterEqual(metrics["roc_auc"], 0.95)
        self.assertGreaterEqual(metrics["average_precision"], 0.95)
        self.assertGreaterEqual(youden, 0.8)
        self.assertIn("zero_threshold_f1", metrics)

    def test_l1_hinge_is_rejected(self) -> None:
        with self.assertRaises(LinearSVMError):
            self.candidate(penalty="l1", loss="hinge", dual="auto")

    def test_coefficient_metadata(self) -> None:
        fit = fit_linear_svm(
            self.X,
            self.y,
            candidate=self.candidate(),
            random_state=23,
        )
        frame = fit.coefficient_frame([f"f{i}" for i in range(5)], model_name="M1")
        self.assertEqual(len(frame), 6)
        self.assertEqual(set(frame["model_engine"]), {"linear_svc"})
        self.assertFalse(frame["coefficient_sparsity_interpretable"].iloc[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
