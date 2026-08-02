#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for configurable inner-CV tuning metric selection (Batch 14)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.nested_cv import NestedCVError, NestedCVOptions  # noqa: E402
from ra_ild_trb.repeated_holdout_summary import _hyperparameter_tables  # noqa: E402
from ra_ild_trb.scheme_management import _normalize_tuning_selection  # noqa: E402
from ra_ild_trb.specifications import (  # noqa: E402
    SpecificationError,
    candidate_selection_policy_name,
    rank_tuning_candidates,
    select_best_tuning_candidate,
    tuning_sort_policy,
)


class TestTuningMetricSelection(unittest.TestCase):
    def setUp(self) -> None:
        # Candidate A has better discrimination but catastrophically worse
        # probability quality. Candidate B should therefore win only under
        # log-loss tuning.
        self.frame = pd.DataFrame(
            [
                {
                    "l1_ratio_alpha": 0.50,
                    "lambda": 1.0,
                    "pooled_inner_roc_auc": 0.700,
                    "pooled_inner_pr_auc": 0.610,
                    "pooled_inner_log_loss": 1.800,
                    "pooled_inner_brier_score": 0.330,
                },
                {
                    "l1_ratio_alpha": 0.90,
                    "lambda": 3.0,
                    "pooled_inner_roc_auc": 0.690,
                    "pooled_inner_pr_auc": 0.600,
                    "pooled_inner_log_loss": 0.610,
                    "pooled_inner_brier_score": 0.220,
                },
            ]
        )

    def test_default_behavior_remains_roc_auc(self) -> None:
        default = select_best_tuning_candidate(self.frame)
        explicit = select_best_tuning_candidate(
            self.frame, primary_metric="roc_auc"
        )
        self.assertEqual(
            (default.alpha, default.lambda_value),
            (0.50, 1.0),
        )
        self.assertEqual(default, explicit)

    def test_log_loss_mode_selects_probability_quality(self) -> None:
        selected = select_best_tuning_candidate(
            self.frame, primary_metric="log_loss"
        )
        self.assertEqual(
            (selected.alpha, selected.lambda_value),
            (0.90, 3.0),
        )

    def test_sort_policies_have_correct_directions(self) -> None:
        roc_columns, roc_ascending = tuning_sort_policy("roc_auc")
        loss_columns, loss_ascending = tuning_sort_policy("log_loss")
        self.assertEqual(roc_columns[0], "pooled_inner_roc_auc")
        self.assertFalse(roc_ascending[0])
        self.assertEqual(loss_columns[0], "pooled_inner_log_loss")
        self.assertTrue(loss_ascending[0])
        self.assertEqual(loss_columns[1], "pooled_inner_brier_score")
        self.assertTrue(loss_ascending[1])

    def test_policy_names_are_auditable(self) -> None:
        self.assertEqual(
            candidate_selection_policy_name("roc_auc"),
            "pooled_roc_pr_lambda_alpha",
        )
        self.assertEqual(
            candidate_selection_policy_name("log_loss"),
            "pooled_log_loss_brier_roc_lambda_alpha",
        )

    def test_invalid_metric_is_rejected(self) -> None:
        with self.assertRaises(SpecificationError):
            rank_tuning_candidates(self.frame, primary_metric="brier_score")
        with self.assertRaises(NestedCVError):
            NestedCVOptions(base_seed=1, tuning_primary_metric="brier_score")

    def test_nested_cv_options_default_and_explicit_modes(self) -> None:
        self.assertEqual(
            NestedCVOptions(base_seed=1).tuning_primary_metric,
            "roc_auc",
        )
        self.assertEqual(
            NestedCVOptions(
                base_seed=1,
                tuning_primary_metric="log_loss",
            ).tuning_primary_metric,
            "log_loss",
        )

    def test_scheme_normalization_updates_low_level_policy(self) -> None:
        resolved = {
            "model_selection": {
                "tuning_primary_metric": "log_loss",
                "candidate_sort": [],
            },
            "nested_cv": {
                "candidate_selection_policy": "obsolete",
            },
        }
        _normalize_tuning_selection(resolved)
        selection = resolved["model_selection"]
        nested = resolved["nested_cv"]
        self.assertEqual(
            selection["candidate_sort"][0],
            {
                "field": "pooled_inner_log_loss",
                "ascending": True,
            },
        )
        self.assertEqual(
            nested["candidate_selection_policy"],
            "pooled_log_loss_brier_roc_lambda_alpha",
        )

    def test_scheme_normalization_defaults_to_roc(self) -> None:
        resolved = {
            "model_selection": {
                "candidate_sort": [],
            },
            "nested_cv": {
                "candidate_selection_policy": "obsolete",
            },
        }
        _normalize_tuning_selection(resolved)
        self.assertEqual(
            resolved["model_selection"]["tuning_primary_metric"],
            "roc_auc",
        )
        self.assertEqual(
            resolved["model_selection"]["candidate_sort"][0]["field"],
            "pooled_inner_roc_auc",
        )

    def test_legacy_aggregate_metrics_default_to_roc(self) -> None:
        legacy = pd.DataFrame(
            [
                {
                    "split_set_id": "split_v1",
                    "split_id": "split_01",
                    "task_id": "repeat_01_fold_01",
                    "outer_repeat": 1,
                    "outer_fold": 1,
                    "model": "M1",
                    "selected_l1_ratio_alpha": 0.5,
                    "selected_lambda": 10.0,
                    "threshold": 0.5,
                    "inner_selected_roc_auc": 0.65,
                    "inner_selected_pr_auc": 0.55,
                }
            ]
        )
        selected, frequency, _ = _hyperparameter_tables(legacy)
        self.assertEqual(selected.loc[0, "tuning_primary_metric"], "roc_auc")
        self.assertEqual(
            selected.loc[0, "candidate_selection_policy"],
            "pooled_roc_pr_lambda_alpha",
        )
        self.assertTrue(np.isnan(selected.loc[0, "inner_selected_log_loss"]))
        self.assertEqual(frequency.loc[0, "tuning_primary_metric"], "roc_auc")


if __name__ == "__main__":
    unittest.main(verbosity=2)
