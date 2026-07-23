#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for TRB V2 threshold rules."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.thresholds import (  # noqa: E402
    ThresholdError,
    apply_threshold,
    choose_threshold_for_sensitivity,
    choose_threshold_max_f1,
    choose_threshold_youden,
)


class TestThresholds(unittest.TestCase):
    def setUp(self) -> None:
        self.y = np.array([0, 0, 0, 1, 1, 1], dtype=int)
        self.p = np.array([0.10, 0.35, 0.40, 0.45, 0.70, 0.90])

    def test_apply_threshold_uses_greater_than_or_equal(self) -> None:
        observed = apply_threshold(np.array([0.49, 0.50, 0.51]), 0.50)
        np.testing.assert_array_equal(observed, np.array([0, 1, 1]))

    def test_youden_matches_v1_reference_implementation(self) -> None:
        fpr, tpr, thresholds = roc_curve(self.y, self.p)
        finite = np.isfinite(thresholds)
        j = tpr[finite] - fpr[finite]
        candidates = thresholds[finite][j == np.max(j)]
        expected = float(candidates[np.argmin(np.abs(candidates - 0.5))])
        self.assertAlmostEqual(choose_threshold_youden(self.y, self.p), expected)

    def test_youden_tie_break_is_closest_to_half(self) -> None:
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.4, 0.6, 0.9])
        observed = choose_threshold_youden(y, p)
        fpr, tpr, thresholds = roc_curve(y, p)
        finite = np.isfinite(thresholds)
        j = tpr[finite] - fpr[finite]
        candidates = thresholds[finite][j == np.max(j)]
        expected = candidates[np.argmin(np.abs(candidates - 0.5))]
        self.assertEqual(observed, expected)

    def test_max_f1_returns_observed_probability_threshold(self) -> None:
        threshold = choose_threshold_max_f1(self.y, self.p)
        self.assertIn(threshold, self.p.tolist())
        self.assertGreaterEqual(threshold, 0)
        self.assertLessEqual(threshold, 1)

    def test_sensitivity_target_uses_highest_eligible_threshold(self) -> None:
        threshold = choose_threshold_for_sensitivity(self.y, self.p, 2 / 3)
        prediction = apply_threshold(self.p, threshold)
        sensitivity = prediction[self.y == 1].mean()
        self.assertGreaterEqual(sensitivity, 2 / 3)

    def test_invalid_probability_is_rejected(self) -> None:
        with self.assertRaises(ThresholdError):
            choose_threshold_youden(self.y, np.array([0, 0, 0, 1, 1, 1.2]))
        with self.assertRaises(ThresholdError):
            apply_threshold(np.array([0.2, np.nan]), 0.5)

    def test_single_class_labels_are_rejected(self) -> None:
        with self.assertRaises(ThresholdError):
            choose_threshold_youden(np.zeros(3, dtype=int), np.array([0.1, 0.2, 0.3]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
