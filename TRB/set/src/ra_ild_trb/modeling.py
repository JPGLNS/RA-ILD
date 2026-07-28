#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Elastic Net logistic-regression backend for the TRB V2 framework.

The implementation intentionally mirrors the fitted-model behavior used by the
V1 ``run_05_single_outer_trial_loo.py`` worker:

* scikit-learn ``LogisticRegression``;
* ``penalty='elasticnet'`` and ``solver='saga'``;
* ``C = 1 / lambda``;
* optional ``class_weight='balanced'``;
* deterministic ``random_state`` derived from the experiment seed;
* convergence recorded from ``ConvergenceWarning``;
* intercept fitted and reported separately;
* coefficients whose absolute value exceeds the configured tolerance are
  considered non-zero.

This module does not perform feature selection, preprocessing, threshold
selection, or metric calculation. Those concerns remain separate so that each
stage can be audited independently.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


class ModelingError(ValueError):
    """Raised when model inputs or hyperparameters are invalid."""


ClassWeight = Union[str, None]


def derive_seed(base_seed: int, *parts: int) -> int:
    """Reproduce the deterministic seed derivation used by the V1 worker."""
    values = (base_seed, *parts)
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in values):
        raise ModelingError("Seed components must be integers.")
    text = ":".join(str(int(value)) for value in values)
    return int.from_bytes(
        hashlib.sha256(text.encode("utf-8")).digest()[:8],
        "little",
    ) % (2**32 - 1)


def _validate_design_matrix(X: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(X, dtype=float)
    if array.ndim != 2:
        raise ModelingError(f"{label} must be a two-dimensional matrix.")
    if array.shape[0] < 2:
        raise ModelingError(f"{label} must contain at least two samples.")
    if array.shape[1] < 1:
        raise ModelingError(f"{label} must contain at least one predictor.")
    if not np.isfinite(array).all():
        raise ModelingError(f"{label} contains non-finite values.")
    return array


def _validate_binary_labels(y: np.ndarray, n_samples: int) -> np.ndarray:
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ModelingError("y must be a one-dimensional array.")
    if len(labels) != n_samples:
        raise ModelingError(
            f"X/y sample mismatch: X has {n_samples}, y has {len(labels)}."
        )
    if pd.isna(labels).any():
        raise ModelingError("y contains missing values.")
    try:
        labels = labels.astype(int)
    except (TypeError, ValueError) as exc:
        raise ModelingError("y must contain binary integer labels 0 and 1.") from exc
    unique = set(np.unique(labels).tolist())
    if unique != {0, 1}:
        raise ModelingError(
            f"y must contain both binary classes {{0, 1}}; observed={sorted(unique)}."
        )
    return labels


def _normalize_class_weight(value: ClassWeight) -> ClassWeight:
    if value is None or value == "none":
        return None
    if value == "balanced":
        return "balanced"
    raise ModelingError("class_weight must be one of: None, 'none', 'balanced'.")


@dataclass(frozen=True)
class ElasticNetLogisticConfig:
    """Hyperparameters for Ridge/Elastic Net/Lasso logistic regression."""

    l1_ratio: float
    lambda_value: float
    class_weight: ClassWeight = "balanced"
    max_iter: int = 10000
    tolerance: float = 1.0e-4
    fit_intercept: bool = True
    solver: str = "saga"
    penalty: str = "elasticnet"

    def __post_init__(self) -> None:
        if not 0 <= float(self.l1_ratio) <= 1:
            raise ModelingError("l1_ratio must be in [0, 1].")
        if float(self.lambda_value) <= 0:
            raise ModelingError("lambda_value must be > 0.")
        if isinstance(self.max_iter, bool) or int(self.max_iter) < 1:
            raise ModelingError("max_iter must be an integer >= 1.")
        if float(self.tolerance) <= 0:
            raise ModelingError("tolerance must be > 0.")
        if self.solver != "saga":
            raise ModelingError("The V1-compatible solver must be 'saga'.")
        if self.penalty != "elasticnet":
            raise ModelingError("The V1-compatible penalty must be 'elasticnet'.")
        _normalize_class_weight(self.class_weight)

    @property
    def C(self) -> float:
        """Return scikit-learn inverse regularization strength, ``1/lambda``."""
        return 1.0 / float(self.lambda_value)

    def sklearn_kwargs(self, random_state: int) -> Dict[str, object]:
        return {
            "penalty": self.penalty,
            "solver": self.solver,
            "l1_ratio": float(self.l1_ratio),
            "C": self.C,
            "class_weight": _normalize_class_weight(self.class_weight),
            "max_iter": int(self.max_iter),
            "tol": float(self.tolerance),
            "random_state": int(random_state),
            "fit_intercept": bool(self.fit_intercept),
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "l1_ratio": float(self.l1_ratio),
            "lambda_value": float(self.lambda_value),
            "C_inverse_lambda": self.C,
            "class_weight": (
                "none" if _normalize_class_weight(self.class_weight) is None
                else "balanced"
            ),
            "max_iter": int(self.max_iter),
            "tolerance": float(self.tolerance),
            "fit_intercept": bool(self.fit_intercept),
            "solver": self.solver,
            "penalty": self.penalty,
        }


@dataclass(frozen=True)
class ElasticNetFit:
    """A fitted probability model plus convergence audit information."""

    model: LogisticRegression
    config: ElasticNetLogisticConfig
    random_state: int
    converged: bool
    n_iter: int

    @property
    def classes(self) -> Tuple[int, int]:
        classes = tuple(int(value) for value in np.asarray(self.model.classes_))
        if classes != (0, 1):
            raise ModelingError(f"Unexpected fitted class order: {classes}.")
        return classes

    @property
    def n_features(self) -> int:
        return int(getattr(self.model, "n_features_in_", self.model.coef_.shape[1]))

    def predict_probability(self, X: np.ndarray) -> np.ndarray:
        return predict_positive_probability(self.model, X)

    def coefficient_frame(
        self,
        feature_names: Sequence[str],
        *,
        nonzero_tolerance: float = 1.0e-12,
        model_name: Optional[str] = None,
    ) -> pd.DataFrame:
        return coefficient_frame(
            self.model,
            feature_names,
            nonzero_tolerance=nonzero_tolerance,
            model_name=model_name,
        )


def fit_elastic_net(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l1_ratio: float,
    lambda_value: float,
    class_weight: ClassWeight = "balanced",
    max_iter: int = 10000,
    tolerance: float = 1.0e-4,
    random_state: int = 0,
) -> ElasticNetFit:
    """Fit the exact Elastic Net logistic backend used by the V1 worker."""
    design = _validate_design_matrix(X, "X")
    labels = _validate_binary_labels(y, design.shape[0])
    if isinstance(random_state, bool) or not isinstance(
        random_state, (int, np.integer)
    ):
        raise ModelingError("random_state must be an integer.")

    config = ElasticNetLogisticConfig(
        l1_ratio=float(l1_ratio),
        lambda_value=float(lambda_value),
        class_weight=class_weight,
        max_iter=int(max_iter),
        tolerance=float(tolerance),
    )
    model = LogisticRegression(**config.sklearn_kwargs(int(random_state)))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(design, labels)
        converged = not any(
            issubclass(warning.category, ConvergenceWarning)
            for warning in caught
        )

    classes = tuple(int(value) for value in np.asarray(model.classes_))
    if classes != (0, 1):
        raise ModelingError(f"Unexpected fitted class order: {classes}.")
    n_iter = int(np.max(model.n_iter_))
    return ElasticNetFit(
        model=model,
        config=config,
        random_state=int(random_state),
        converged=bool(converged),
        n_iter=n_iter,
    )


def predict_positive_probability(
    model: LogisticRegression,
    X: np.ndarray,
) -> np.ndarray:
    """Return the probability for class 1 after validating model and matrix."""
    design = np.asarray(X, dtype=float)
    if design.ndim != 2 or design.shape[1] < 1:
        raise ModelingError("Prediction X must be a non-empty two-dimensional matrix.")
    if not np.isfinite(design).all():
        raise ModelingError("Prediction X contains non-finite values.")
    classes = tuple(int(value) for value in np.asarray(getattr(model, "classes_", [])))
    if classes != (0, 1):
        raise ModelingError(f"Model classes must be (0, 1); observed={classes}.")
    expected = int(getattr(model, "n_features_in_", design.shape[1]))
    if design.shape[1] != expected:
        raise ModelingError(
            f"Prediction feature mismatch: X has {design.shape[1]}, model expects {expected}."
        )
    probability = np.asarray(model.predict_proba(design)[:, 1], dtype=float)
    if probability.shape != (design.shape[0],):
        raise ModelingError("Unexpected probability output shape.")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ModelingError("Model produced invalid probabilities.")
    return probability


def coefficient_frame(
    model: LogisticRegression,
    feature_names: Sequence[str],
    *,
    nonzero_tolerance: float = 1.0e-12,
    model_name: Optional[str] = None,
) -> pd.DataFrame:
    """Return the V1-compatible intercept and coefficient table."""
    if float(nonzero_tolerance) < 0:
        raise ModelingError("nonzero_tolerance must be >= 0.")
    names = tuple(str(value) for value in feature_names)
    if len(names) != len(set(names)):
        raise ModelingError("feature_names contains duplicates.")
    coefficients = np.asarray(model.coef_, dtype=float)
    intercept = np.asarray(model.intercept_, dtype=float)
    if coefficients.shape != (1, len(names)):
        raise ModelingError(
            f"Coefficient shape={coefficients.shape}, expected=(1, {len(names)})."
        )
    if intercept.shape != (1,):
        raise ModelingError(f"Unexpected intercept shape: {intercept.shape}.")

    rows = [
        {
            "feature_name": "__INTERCEPT__",
            "coefficient": float(intercept[0]),
            "absolute_coefficient": abs(float(intercept[0])),
            "nonzero": True,
        }
    ]
    for name, coefficient in zip(names, coefficients.ravel()):
        value = float(coefficient)
        rows.append(
            {
                "feature_name": name,
                "coefficient": value,
                "absolute_coefficient": abs(value),
                "nonzero": bool(abs(value) > float(nonzero_tolerance)),
            }
        )
    frame = pd.DataFrame(rows)
    if model_name is not None:
        frame.insert(0, "model", str(model_name))
    return frame


def count_nonzero_coefficients(
    model: LogisticRegression,
    *,
    tolerance: float = 1.0e-12,
) -> int:
    """Count non-intercept coefficients above the V1 absolute tolerance."""
    if float(tolerance) < 0:
        raise ModelingError("tolerance must be >= 0.")
    coefficients = np.asarray(model.coef_, dtype=float)
    if coefficients.ndim != 2 or coefficients.shape[0] != 1:
        raise ModelingError(f"Unexpected coefficient shape: {coefficients.shape}.")
    return int(np.sum(np.abs(coefficients.ravel()) > float(tolerance)))
