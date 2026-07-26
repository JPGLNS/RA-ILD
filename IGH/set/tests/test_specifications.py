#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for model specifications and candidate management."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

TEST_DIR = Path(__file__).resolve().parent
SET_DIR = TEST_DIR.parent
SRC_DIR = SET_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.specifications import (  # noqa: E402
    HyperparameterCandidate,
    SpecificationError,
    build_hyperparameter_grid,
    parse_candidate_pair_mapping,
    parse_model_specifications,
    rank_tuning_candidates,
    resolve_all_model_specifications,
    select_best_tuning_candidate,
    select_static_igh_features,
    selected_pairs_from_oof,
    validate_locked_candidate_selection,
    validate_tuning_grid_frame,
)


class TestSpecifications(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = pd.DataFrame(
            {
                "feature_name": ["a", "b", "c", "d", "e", "f"],
                "feature_group": ["igh_div", "igh_kmer", "clinical", "igh_div", "igh_div", "igh_div"],
                "feature_role": ["candidate_predictor", "candidate_predictor", "candidate_predictor", "identifier", "candidate_predictor", "candidate_predictor"],
                "present_in_train_base": [True, "true", True, True, False, 1],
                "reference_drop": [False, "0", False, False, False, "yes"],
            }
        )
        self.models = {
            "M0": {
                "description": "clinical",
                "numeric": ["age"],
                "categorical": ["sex"],
                "static_feature_groups": [],
                "dynamic_public": [],
            },
            "M2": {
                "description": "full",
                "numeric": ["age"],
                "categorical": ["sex"],
                "static_feature_groups": ["static_igh_candidate_predictors"],
                "dynamic_public": ["pub1", "pub2"],
            },
        }

    def test_manifest_selection_rule_and_order(self) -> None:
        self.assertEqual(select_static_igh_features(self.manifest), ["a", "b"])

    def test_manifest_available_columns_reproduces_v1_filter(self) -> None:
        self.assertEqual(
            select_static_igh_features(self.manifest, available_columns=["b"]),
            ["b"],
        )

    def test_manifest_strict_missing_feature_is_rejected(self) -> None:
        with self.assertRaises(SpecificationError):
            select_static_igh_features(
                self.manifest,
                available_columns=["a"],
                require_all_available=True,
            )

    def test_manifest_duplicate_name_is_rejected(self) -> None:
        broken = pd.concat([self.manifest, self.manifest.iloc[[0]]], ignore_index=True)
        with self.assertRaises(SpecificationError):
            select_static_igh_features(broken)

    def test_manifest_invalid_boolean_is_rejected(self) -> None:
        broken = self.manifest.copy()
        broken.loc[0, "reference_drop"] = "maybe"
        with self.assertRaises(SpecificationError):
            select_static_igh_features(broken)

    def test_model_resolution_preserves_order(self) -> None:
        resolved = resolve_all_model_specifications(
            self.models,
            static_features=["s1", "s2"],
            allowed_dynamic_public=["pub1", "pub2"],
        )
        self.assertEqual(resolved["M0"].numeric, ("age",))
        self.assertEqual(
            resolved["M2"].numeric,
            ("age", "s1", "s2", "pub1", "pub2"),
        )
        self.assertEqual(resolved["M2"].categorical, ("sex",))

    def test_unknown_static_group_is_rejected(self) -> None:
        broken = {"M": dict(self.models["M2"])}
        broken["M"]["static_feature_groups"] = ["unknown"]
        with self.assertRaises(SpecificationError):
            resolve_all_model_specifications(
                broken,
                static_features=["s1"],
                allowed_dynamic_public=["pub1", "pub2"],
            )

    def test_unknown_dynamic_public_is_rejected(self) -> None:
        with self.assertRaises(SpecificationError):
            resolve_all_model_specifications(
                self.models,
                static_features=["s1"],
                allowed_dynamic_public=["pub1"],
            )

    def test_numeric_categorical_overlap_is_rejected(self) -> None:
        broken = {"M": dict(self.models["M0"])}
        broken["M"]["categorical"] = ["age"]
        with self.assertRaises(SpecificationError):
            resolve_all_model_specifications(
                broken,
                static_features=["s1"],
                allowed_dynamic_public=[],
            )

    def test_grid_is_alpha_then_lambda(self) -> None:
        grid = build_hyperparameter_grid([0.1, 0.5], [1, 3])
        self.assertEqual(
            [(x.alpha, x.lambda_value) for x in grid],
            [(0.1, 1.0), (0.1, 3.0), (0.5, 1.0), (0.5, 3.0)],
        )
        self.assertAlmostEqual(grid[-1].inverse_lambda_c, 1 / 3)

    def test_invalid_or_duplicate_grid_is_rejected(self) -> None:
        for alphas, lambdas in [([0.1, 0.1], [1]), ([1.2], [1]), ([0.1], [0])]:
            with self.subTest(alphas=alphas, lambdas=lambdas):
                with self.assertRaises(SpecificationError):
                    build_hyperparameter_grid(alphas, lambdas)

    def test_candidate_ranking_matches_v1_tie_breaks(self) -> None:
        frame = pd.DataFrame(
            {
                "l1_ratio_alpha": [0.1, 0.5, 0.9, 0.5],
                "lambda": [1, 1, 3, 3],
                "pooled_inner_roc_auc": [0.8, 0.8, 0.8, 0.8],
                "pooled_inner_pr_auc": [0.7, 0.8, 0.8, 0.8],
            }
        )
        ranked = rank_tuning_candidates(frame)
        self.assertEqual(
            (ranked.iloc[0]["l1_ratio_alpha"], ranked.iloc[0]["lambda"]),
            (0.9, 3.0),
        )
        selected = select_best_tuning_candidate(frame)
        self.assertEqual((selected.alpha, selected.lambda_value), (0.9, 3.0))

    def test_tuning_grid_frame_validation(self) -> None:
        candidates = build_hyperparameter_grid([0.1, 0.5], [1, 2])
        rows = []
        for model in ["M0", "M1"]:
            for candidate in candidates:
                rows.append({"model": model, **candidate.as_dict()})
        validate_tuning_grid_frame(
            pd.DataFrame(rows),
            model_names=["M0", "M1"],
            candidates=candidates,
        )

    def test_bad_inverse_lambda_is_rejected(self) -> None:
        candidates = build_hyperparameter_grid([0.1], [2])
        frame = pd.DataFrame(
            [{"model": "M0", "l1_ratio_alpha": 0.1, "lambda": 2, "C_inverse_lambda": 2}]
        )
        with self.assertRaises(SpecificationError):
            validate_tuning_grid_frame(frame, model_names=["M0"], candidates=candidates)

    def test_selected_pairs_from_oof_requires_one_pair(self) -> None:
        frame = pd.DataFrame(
            {
                "model": ["M0", "M0", "M1"],
                "l1_ratio_alpha": [0.1, 0.1, 0.5],
                "lambda": [1, 1, 3],
            }
        )
        selected = selected_pairs_from_oof(frame)
        self.assertEqual((selected["M1"].alpha, selected["M1"].lambda_value), (0.5, 3.0))
        broken = frame.copy()
        broken.loc[1, "lambda"] = 2
        with self.assertRaises(SpecificationError):
            selected_pairs_from_oof(broken)

    def test_locked_candidate_subset_validation(self) -> None:
        full = build_hyperparameter_grid([0.5, 0.9], [10, 30])
        subset = parse_candidate_pair_mapping(
            [{"alpha": 0.9, "lambda": 10}, {"alpha": 0.5, "lambda": 30}],
            label="candidate_pairs",
        )
        validate_locked_candidate_selection(
            full_grid=full,
            candidate_subset=subset,
            selected=HyperparameterCandidate(0.5, 30),
        )
        with self.assertRaises(SpecificationError):
            validate_locked_candidate_selection(
                full_grid=full,
                candidate_subset=subset,
                selected=HyperparameterCandidate(0.5, 10),
            )

    def test_model_parser_requires_description(self) -> None:
        broken = {"M": {"numeric": ["age"]}}
        with self.assertRaises(SpecificationError):
            parse_model_specifications(broken)


if __name__ == "__main__":
    unittest.main(verbosity=2)
