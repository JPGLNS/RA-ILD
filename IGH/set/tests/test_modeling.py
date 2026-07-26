#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the IGH V2 Elastic Net backend."""

from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.modeling import (  # noqa: E402
    ElasticNetLogisticConfig,
    ModelingError,
    coefficient_frame,
    count_nonzero_coefficients,
    derive_seed,
    fit_elastic_net,
)


class TestModeling(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(20260711)
        cls.X = rng.normal(size=(80, 6))
        linear = 0.8 * cls.X[:, 0] - 0.5 * cls.X[:, 1] + 0.2 * cls.X[:, 2]
        cls.y = (linear + rng.normal(scale=0.8, size=80) > 0).astype(int)
        cls.feature_names = [f"x{i}" for i in range(cls.X.shape[1])]

    def test_config_uses_inverse_lambda(self) -> None:
        config = ElasticNetLogisticConfig(l1_ratio=0.5, lambda_value=4.0)
        self.assertAlmostEqual(config.C, 0.25)
        self.assertEqual(config.solver, "saga")
        self.assertEqual(config.penalty, "elasticnet")

    def test_invalid_hyperparameters_are_rejected(self) -> None:
        with self.assertRaises(ModelingError):
            ElasticNetLogisticConfig(l1_ratio=0.0, lambda_value=1.0)
        with self.assertRaises(ModelingError):
            ElasticNetLogisticConfig(l1_ratio=0.5, lambda_value=0.0)
        with self.assertRaises(ModelingError):
            ElasticNetLogisticConfig(
                l1_ratio=0.5,
                lambda_value=1.0,
                class_weight="invalid",
            )

    def test_derive_seed_is_deterministic_and_order_sensitive(self) -> None:
        first = derive_seed(20260711, 400, 2)
        second = derive_seed(20260711, 400, 2)
        changed = derive_seed(20260711, 2, 400)
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2**32 - 1)

    def test_fit_is_reproducible(self) -> None:
        kwargs = dict(
            l1_ratio=0.5,
            lambda_value=0.3,
            class_weight="balanced",
            max_iter=5000,
            tolerance=1e-6,
            random_state=derive_seed(20260711, 400, 0),
        )
        first = fit_elastic_net(self.X, self.y, **kwargs)
        second = fit_elastic_net(self.X, self.y, **kwargs)
        np.testing.assert_allclose(first.model.coef_, second.model.coef_, rtol=0, atol=0)
        np.testing.assert_allclose(
            first.predict_probability(self.X),
            second.predict_probability(self.X),
            rtol=0,
            atol=0,
        )
        self.assertEqual(first.n_iter, second.n_iter)
        self.assertEqual(first.converged, second.converged)

    def test_matches_v1_reference_fit(self) -> None:
        seed = derive_seed(20260711, 400, 1)
        observed = fit_elastic_net(
            self.X,
            self.y,
            l1_ratio=0.9,
            lambda_value=1.0,
            class_weight="balanced",
            max_iter=5000,
            tolerance=1e-6,
            random_state=seed,
        )
        expected = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.9,
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            tol=1e-6,
            random_state=seed,
            fit_intercept=True,
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", ConvergenceWarning)
            expected.fit(self.X, self.y)
        np.testing.assert_allclose(observed.model.coef_, expected.coef_, rtol=0, atol=0)
        np.testing.assert_allclose(observed.model.intercept_, expected.intercept_, rtol=0, atol=0)
        np.testing.assert_allclose(
            observed.predict_probability(self.X),
            expected.predict_proba(self.X)[:, 1],
            rtol=0,
            atol=0,
        )

    def test_coefficient_frame_matches_v1_layout(self) -> None:
        fit = fit_elastic_net(
            self.X,
            self.y,
            l1_ratio=0.5,
            lambda_value=3.0,
            max_iter=5000,
            tolerance=1e-6,
            random_state=1,
        )
        frame = coefficient_frame(
            fit.model,
            self.feature_names,
            model_name="M_TEST",
        )
        self.assertEqual(frame.iloc[0]["feature_name"], "__INTERCEPT__")
        self.assertTrue(bool(frame.iloc[0]["nonzero"]))
        self.assertEqual(frame["feature_name"].iloc[1:].tolist(), self.feature_names)
        self.assertEqual(len(frame), len(self.feature_names) + 1)
        self.assertEqual(set(frame["model"]), {"M_TEST"})

    def test_nonzero_count_excludes_intercept(self) -> None:
        fit = fit_elastic_net(
            self.X,
            self.y,
            l1_ratio=1.0,
            lambda_value=30.0,
            max_iter=5000,
            tolerance=1e-6,
            random_state=2,
        )
        observed = count_nonzero_coefficients(fit.model, tolerance=1e-12)
        expected = int(np.sum(np.abs(fit.model.coef_.ravel()) > 1e-12))
        self.assertEqual(observed, expected)

    def test_invalid_design_and_labels_are_rejected(self) -> None:
        with self.assertRaises(ModelingError):
            fit_elastic_net(
                self.X[:, 0],
                self.y,
                l1_ratio=0.5,
                lambda_value=1.0,
            )
        with self.assertRaises(ModelingError):
            fit_elastic_net(
                self.X,
                np.zeros(len(self.X), dtype=int),
                l1_ratio=0.5,
                lambda_value=1.0,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
