#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for leakage-controlled TRB V2 preprocessing."""

from __future__ import annotations

import json
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

from ra_ild_trb.preprocessing import (  # noqa: E402
    PreprocessingError,
    fit_preprocessor,
    prepare_design_matrices,
)


class TestPreprocessing(unittest.TestCase):
    def setUp(self) -> None:
        self.train = pd.DataFrame(
            {
                "age": [20.0, 30.0, 40.0, 50.0],
                "signal": [1.0, 2.0, 4.0, 8.0],
                "constant": [7.0, 7.0, 7.0, 7.0],
                "sex": ["female", "male", "female", "male"],
                "material": ["PBMC", "buffercoat", "PBMC", "buffercoat"],
            },
            index=["S1", "S2", "S3", "S4"],
        )
        self.valid = pd.DataFrame(
            {
                "age": [60.0, 70.0],
                "signal": [16.0, 32.0],
                "constant": [7.0, 7.0],
                "sex": ["female", "male"],
                "material": ["PBMC", "buffercoat"],
            },
            index=["V1", "V2"],
        )

    def test_population_standardization(self) -> None:
        result = prepare_design_matrices(
            self.train,
            self.valid,
            ["age", "signal"],
            [],
        )
        np.testing.assert_allclose(result.X_train.mean(axis=0), 0.0, atol=1e-15)
        np.testing.assert_allclose(result.X_train.std(axis=0, ddof=0), 1.0)
        self.assertEqual(result.feature_names, ("age", "signal"))

    def test_validation_does_not_affect_fit_statistics(self) -> None:
        original = fit_preprocessor(self.train, ["age"], [])
        extreme_valid = self.valid.copy()
        extreme_valid["age"] = [1e9, -1e9]
        transformed = original.transform(extreme_valid)
        self.assertAlmostEqual(original.training_means[0], 35.0)
        self.assertAlmostEqual(
            original.training_sds[0],
            np.std([20.0, 30.0, 40.0, 50.0], ddof=0),
        )
        self.assertTrue(np.isfinite(transformed.matrix).all())

    def test_categorical_reference_is_lexicographically_first(self) -> None:
        fitted = fit_preprocessor(self.train, [], ["sex", "material"])
        sex = fitted.category_encodings["sex"]
        material = fitted.category_encodings["material"]
        self.assertEqual(sex.reference, "female")
        self.assertEqual(sex.dummy_feature_names, ("sex__male_vs_female",))
        self.assertEqual(material.reference, "PBMC")
        self.assertEqual(
            material.dummy_feature_names,
            ("material__buffercoat_vs_PBMC",),
        )

    def test_feature_order_matches_v1(self) -> None:
        fitted = fit_preprocessor(
            self.train,
            ["age", "signal"],
            ["sex", "material"],
        )
        self.assertEqual(
            fitted.raw_feature_names,
            (
                "age",
                "signal",
                "sex__male_vs_female",
                "material__buffercoat_vs_PBMC",
            ),
        )
        self.assertEqual(
            fitted.source_types,
            ("numeric", "numeric", "categorical_dummy", "categorical_dummy"),
        )

    def test_zero_variance_feature_is_dropped(self) -> None:
        fitted = fit_preprocessor(
            self.train,
            ["age", "constant"],
            ["sex"],
        )
        self.assertIn("constant", fitted.dropped_feature_names)
        self.assertNotIn("constant", fitted.kept_feature_names)
        audit = fitted.audit_frame().set_index("feature_name")
        self.assertFalse(bool(audit.loc["constant", "kept_after_zero_variance_filter"]))
        self.assertEqual(float(audit.loc["constant", "training_sd"]), 0.0)

    def test_unseen_category_reproduces_v1_reference_code(self) -> None:
        fitted = fit_preprocessor(self.train, ["age"], ["sex"])
        new = pd.DataFrame({"age": [35.0], "sex": ["unknown"]})
        raw = fitted.transform_raw(new)
        dummy_index = raw.feature_names.index("sex__male_vs_female")
        self.assertEqual(raw.matrix[0, dummy_index], 0.0)
        self.assertEqual(raw.unseen_category_counts["sex"], 1)

    def test_strict_unseen_category_is_rejected(self) -> None:
        fitted = fit_preprocessor(self.train, ["age"], ["sex"])
        new = pd.DataFrame({"age": [35.0], "sex": ["unknown"]})
        with self.assertRaises(PreprocessingError):
            fitted.transform(new, strict_unseen_categories=True)

    def test_missing_predictor_column_is_rejected(self) -> None:
        with self.assertRaises(PreprocessingError):
            fit_preprocessor(self.train, ["not_present"], [])

    def test_missing_values_are_rejected(self) -> None:
        broken = self.train.copy()
        broken.loc["S1", "age"] = np.nan
        with self.assertRaises(PreprocessingError):
            fit_preprocessor(broken, ["age"], ["sex"])

    def test_numeric_categorical_overlap_is_rejected(self) -> None:
        with self.assertRaises(PreprocessingError):
            fit_preprocessor(self.train, ["age"], ["age"])

    def test_all_zero_variance_is_rejected(self) -> None:
        only_constant = pd.DataFrame({"x": [1.0, 1.0, 1.0]})
        with self.assertRaises(PreprocessingError):
            fit_preprocessor(only_constant, ["x"], [])

    def test_serializable_metadata(self) -> None:
        fitted = fit_preprocessor(self.train, ["age"], ["sex"])
        encoded = json.dumps(fitted.to_dict())
        self.assertIn("sex__male_vs_female", encoded)
        self.assertEqual(fitted.to_dict()["ddof"], 0)

    def test_matches_v1_reference_implementation(self) -> None:
        train = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0],
                "z": [2.0, 2.5, 4.0, 8.0, 16.0],
                "group": ["B", "A", "C", "A", "B"],
            }
        )
        valid = pd.DataFrame(
            {
                "x": [6.0, 7.0],
                "z": [32.0, 64.0],
                "group": ["C", "A"],
            }
        )

        train_parts = [
            train["x"].to_numpy(float)[:, None],
            train["z"].to_numpy(float)[:, None],
        ]
        valid_parts = [
            valid["x"].to_numpy(float)[:, None],
            valid["z"].to_numpy(float)[:, None],
        ]
        names = ["x", "z"]
        categories = sorted(train["group"].astype(str).unique().tolist())
        reference = categories[0]
        for category in categories[1:]:
            train_parts.append(
                (train["group"].astype(str) == category).to_numpy(float)[:, None]
            )
            valid_parts.append(
                (valid["group"].astype(str) == category).to_numpy(float)[:, None]
            )
            names.append(f"group__{category}_vs_{reference}")
        train_raw = np.hstack(train_parts)
        valid_raw = np.hstack(valid_parts)
        means = train_raw.mean(axis=0)
        sds = train_raw.std(axis=0, ddof=0)
        keep = np.isfinite(sds) & (sds > 1e-12)
        expected_train = (train_raw[:, keep] - means[keep]) / sds[keep]
        expected_valid = (valid_raw[:, keep] - means[keep]) / sds[keep]
        expected_names = tuple(name for name, flag in zip(names, keep) if flag)

        observed = prepare_design_matrices(
            train, valid, ["x", "z"], ["group"]
        )
        np.testing.assert_allclose(observed.X_train, expected_train)
        np.testing.assert_allclose(observed.X_valid, expected_valid)
        self.assertEqual(observed.feature_names, expected_names)

    def test_convenience_result_matches_direct_transform(self) -> None:
        prepared = prepare_design_matrices(
            self.train,
            self.valid,
            ["age", "signal"],
            ["sex"],
        )
        direct_train = prepared.preprocessor.transform(self.train)
        direct_valid = prepared.preprocessor.transform(self.valid)
        np.testing.assert_allclose(prepared.X_train, direct_train.matrix)
        np.testing.assert_allclose(prepared.X_valid, direct_valid.matrix)
        self.assertEqual(prepared.feature_names, direct_train.feature_names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
