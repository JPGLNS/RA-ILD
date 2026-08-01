#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from ra_ild_trb.metrics import classification_metrics
from ra_ild_trb.repeated_holdout_summary import (
    _metric_summary,
    _model_ranking,
    _pairwise_model_differences,
)


class TestProbabilityMetrics(unittest.TestCase):
    def test_classification_metrics_include_probability_metrics(self) -> None:
        y = np.array([0, 0, 1, 1], dtype=int)
        p = np.array([0.1, 0.3, 0.7, 0.9], dtype=float)
        result = classification_metrics(
            y,
            p,
            0.5,
            include_brier=True,
            include_log_loss=True,
        )
        self.assertAlmostEqual(
            result["log_loss"], log_loss(y, p, labels=[0, 1])
        )
        self.assertAlmostEqual(result["brier_score"], brier_score_loss(y, p))

    def test_loss_metrics_rank_ascending_and_pairwise_win_when_lower(self) -> None:
        rows = []
        for split_id in ("s1", "s2"):
            for model, roc, loss, brier in (
                ("A", 0.60, 0.70, 0.25),
                ("B", 0.70, 0.55, 0.20),
            ):
                rows.append(
                    {
                        "split_id": split_id,
                        "model": model,
                        "roc_auc": roc,
                        "pr_auc": roc - 0.05,
                        "log_loss": loss,
                        "brier_score": brier,
                        "accuracy": 0.60,
                        "sensitivity_recall": 0.60,
                        "specificity": 0.60,
                        "precision": 0.55,
                        "f1": 0.57,
                    }
                )
        metrics = pd.DataFrame(rows)
        ranking = _model_ranking(_metric_summary(metrics)).set_index("model")
        self.assertEqual(int(ranking.loc["B", "rank_mean_log_loss"]), 1)
        self.assertEqual(int(ranking.loc["B", "rank_mean_brier_score"]), 1)

        pairwise = _pairwise_model_differences(metrics)
        loss = pairwise.loc[pairwise["metric"] == "log_loss"].iloc[0]
        self.assertEqual(loss["metric_direction"], "lower_is_better")
        self.assertLess(
            float(loss["mean_difference_comparison_minus_reference"]), 0
        )
        self.assertEqual(float(loss["comparison_win_frequency"]), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
