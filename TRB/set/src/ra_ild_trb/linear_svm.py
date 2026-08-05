#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LinearSVC backend and score-based evaluation utilities for TRB V2.

This module is intentionally independent from the historical Elastic Net backend.
It consumes already-preprocessed design matrices and returns continuous
``decision_function`` scores. Probability-only metrics are deliberately absent.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.svm import LinearSVC


class LinearSVMError(ValueError):
    """Raised when Linear SVM inputs or configuration are invalid."""


ClassWeight = Union[str, None]
DualValue = Union[str, bool]

VALID_PENALTIES = ("l1", "l2")
VALID_LOSSES = ("hinge", "squared_hinge")
VALID_CLASS_WEIGHTS = ("none", "balanced")
VALID_DUAL_VALUES = ("auto", True, False)


def derive_svm_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable 32-bit seed from integer/string candidate identifiers."""
    if isinstance(base_seed, bool) or not isinstance(base_seed, (int, np.integer)):
        raise LinearSVMError("base_seed must be an integer.")
    text = ":".join([str(int(base_seed))] + [str(value) for value in parts])
    return int.from_bytes(
        hashlib.sha256(text.encode("utf-8")).digest()[:8], "little"
    ) % (2**32 - 1)


def normalize_class_weight(value: ClassWeight) -> ClassWeight:
    if value is None or str(value).strip().lower() == "none":
        return None
    if str(value).strip().lower() == "balanced":
        return "balanced"
    raise LinearSVMError("class_weight must be 'none', None, or 'balanced'.")


def normalize_dual(value: object) -> DualValue:
    if value == "auto":
        return "auto"
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise LinearSVMError("dual must be 'auto', true, or false.")


def validate_linear_svc_combination(
    *, penalty: str, loss: str, dual: DualValue
) -> None:
    penalty = str(penalty).strip().lower()
    loss = str(loss).strip().lower()
    dual = normalize_dual(dual)
    if penalty not in VALID_PENALTIES:
        raise LinearSVMError(f"Unsupported penalty: {penalty!r}.")
    if loss not in VALID_LOSSES:
        raise LinearSVMError(f"Unsupported loss: {loss!r}.")
    if penalty == "l1" and loss == "hinge":
        raise LinearSVMError("LinearSVC does not support penalty='l1', loss='hinge'.")
    if dual is True and penalty == "l1":
        raise LinearSVMError("penalty='l1' requires dual=false or dual='auto'.")
    if dual is False and loss == "hinge":
        raise LinearSVMError("loss='hinge' requires dual=true or dual='auto'.")


def resolve_dual_for_audit(
    requested: DualValue,
    *,
    penalty: str,
    loss: str,
    n_samples: int,
    n_features: int,
) -> bool:
    """Mirror the documented LinearSVC ``dual='auto'`` intent for auditing.

    The estimator remains the source of truth. This function records the expected
    primal/dual path from dimensionality and legal solver combinations.
    """
    requested = normalize_dual(requested)
    validate_linear_svc_combination(penalty=penalty, loss=loss, dual=requested)
    if isinstance(requested, bool):
        return requested
    if penalty == "l1":
        return False
    if loss == "hinge":
        return True
    return bool(int(n_samples) < int(n_features))


@dataclass(frozen=True)
class LinearSVMCandidate:
    candidate_id: str
    block_id: str
    priority: int
    C: float
    penalty: str
    loss: str
    class_weight: str
    dual: DualValue = "auto"

    def __post_init__(self) -> None:
        if not str(self.candidate_id).strip():
            raise LinearSVMError("candidate_id must be non-empty.")
        if not str(self.block_id).strip():
            raise LinearSVMError("block_id must be non-empty.")
        if isinstance(self.priority, bool) or int(self.priority) < 1:
            raise LinearSVMError("priority must be an integer >= 1.")
        if not np.isfinite(float(self.C)) or float(self.C) <= 0:
            raise LinearSVMError("C must be finite and > 0.")
        object.__setattr__(self, "penalty", str(self.penalty).strip().lower())
        object.__setattr__(self, "loss", str(self.loss).strip().lower())
        normalized_weight = (
            "none" if normalize_class_weight(self.class_weight) is None else "balanced"
        )
        object.__setattr__(self, "class_weight", normalized_weight)
        object.__setattr__(self, "dual", normalize_dual(self.dual))
        validate_linear_svc_combination(
            penalty=self.penalty, loss=self.loss, dual=self.dual
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_block": self.block_id,
            "candidate_priority": int(self.priority),
            "C": float(self.C),
            "penalty": self.penalty,
            "loss": self.loss,
            "class_weight": self.class_weight,
            "dual_requested": self.dual,
        }


@dataclass(frozen=True)
class LinearSVMFit:
    model: LinearSVC
    candidate: LinearSVMCandidate
    random_state: int
    converged: bool
    n_iter: int
    dual_resolved: bool

    @property
    def classes(self) -> Tuple[int, int]:
        classes = tuple(int(value) for value in np.asarray(self.model.classes_))
        if classes != (0, 1):
            raise LinearSVMError(f"Unexpected fitted class order: {classes}.")
        return classes

    def decision_score(self, X: np.ndarray) -> np.ndarray:
        return decision_score(self.model, X)

    def coefficient_frame(
        self,
        feature_names: Sequence[str],
        *,
        nonzero_tolerance: float = 1.0e-12,
        model_name: Optional[str] = None,
    ) -> pd.DataFrame:
        frame = linear_svm_coefficient_frame(
            self.model,
            feature_names,
            nonzero_tolerance=nonzero_tolerance,
            model_name=model_name,
        )
        frame["model_engine"] = "linear_svc"
        frame["selected_C"] = float(self.candidate.C)
        frame["selected_penalty"] = self.candidate.penalty
        frame["selected_loss"] = self.candidate.loss
        frame["selected_class_weight"] = self.candidate.class_weight
        frame["coefficient_sparsity_interpretable"] = bool(
            self.candidate.penalty == "l1"
        )
        return frame


def _validate_design_matrix(X: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(X, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise LinearSVMError(
            f"{label} must be a 2D matrix with >=2 rows and >=1 column."
        )
    if not np.isfinite(array).all():
        raise LinearSVMError(f"{label} contains non-finite values.")
    return array


def _validate_binary_labels(y: np.ndarray, n_samples: int) -> np.ndarray:
    labels = np.asarray(y)
    if labels.ndim != 1 or len(labels) != n_samples:
        raise LinearSVMError("y must be one-dimensional and match X rows.")
    if pd.isna(labels).any():
        raise LinearSVMError("y contains missing values.")
    labels = labels.astype(int)
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise LinearSVMError("y must contain both binary classes 0 and 1.")
    return labels


def fit_linear_svm(
    X: np.ndarray,
    y: np.ndarray,
    *,
    candidate: LinearSVMCandidate,
    max_iter: int = 10000,
    tolerance: float = 1.0e-4,
    fit_intercept: bool = True,
    intercept_scaling: float = 1.0,
    random_state: int = 0,
) -> LinearSVMFit:
    design = _validate_design_matrix(X, "X")
    labels = _validate_binary_labels(y, design.shape[0])
    if isinstance(max_iter, bool) or int(max_iter) < 1:
        raise LinearSVMError("max_iter must be an integer >= 1.")
    if not np.isfinite(float(tolerance)) or float(tolerance) <= 0:
        raise LinearSVMError("tolerance must be finite and > 0.")
    if not np.isfinite(float(intercept_scaling)) or float(intercept_scaling) <= 0:
        raise LinearSVMError("intercept_scaling must be finite and > 0.")
    if isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer)):
        raise LinearSVMError("random_state must be an integer.")

    dual_requested = normalize_dual(candidate.dual)
    dual_resolved = resolve_dual_for_audit(
        dual_requested,
        penalty=candidate.penalty,
        loss=candidate.loss,
        n_samples=design.shape[0],
        n_features=design.shape[1],
    )
    model = LinearSVC(
        penalty=candidate.penalty,
        loss=candidate.loss,
        dual=dual_requested,
        C=float(candidate.C),
        class_weight=normalize_class_weight(candidate.class_weight),
        fit_intercept=bool(fit_intercept),
        intercept_scaling=float(intercept_scaling),
        tol=float(tolerance),
        max_iter=int(max_iter),
        random_state=int(random_state),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(design, labels)
        converged = not any(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )
    classes = tuple(int(value) for value in np.asarray(model.classes_))
    if classes != (0, 1):
        raise LinearSVMError(f"Unexpected fitted class order: {classes}.")
    return LinearSVMFit(
        model=model,
        candidate=candidate,
        random_state=int(random_state),
        converged=bool(converged),
        n_iter=int(model.n_iter_),
        dual_resolved=bool(dual_resolved),
    )


def decision_score(model: LinearSVC, X: np.ndarray) -> np.ndarray:
    design = np.asarray(X, dtype=float)
    if design.ndim != 2 or design.shape[1] < 1:
        raise LinearSVMError("Prediction X must be a non-empty 2D matrix.")
    if not np.isfinite(design).all():
        raise LinearSVMError("Prediction X contains non-finite values.")
    classes = tuple(int(value) for value in np.asarray(getattr(model, "classes_", [])))
    if classes != (0, 1):
        raise LinearSVMError(f"Model classes must be (0, 1); observed={classes}.")
    expected = int(getattr(model, "n_features_in_", design.shape[1]))
    if design.shape[1] != expected:
        raise LinearSVMError(
            f"Prediction feature mismatch: X has {design.shape[1]}, model expects {expected}."
        )
    score = np.asarray(model.decision_function(design), dtype=float)
    if score.shape != (design.shape[0],) or not np.isfinite(score).all():
        raise LinearSVMError("LinearSVC produced invalid decision scores.")
    return score


def linear_svm_coefficient_frame(
    model: LinearSVC,
    feature_names: Sequence[str],
    *,
    nonzero_tolerance: float = 1.0e-12,
    model_name: Optional[str] = None,
) -> pd.DataFrame:
    if float(nonzero_tolerance) < 0:
        raise LinearSVMError("nonzero_tolerance must be >= 0.")
    names = tuple(str(value) for value in feature_names)
    if len(names) != len(set(names)):
        raise LinearSVMError("feature_names contains duplicates.")
    coefficients = np.asarray(model.coef_, dtype=float)
    intercept = np.asarray(model.intercept_, dtype=float)
    if coefficients.shape != (1, len(names)):
        raise LinearSVMError(
            f"Coefficient shape={coefficients.shape}, expected=(1, {len(names)})."
        )
    if intercept.shape != (1,):
        raise LinearSVMError(f"Unexpected intercept shape: {intercept.shape}.")
    rows = [
        {
            "feature_name": "__INTERCEPT__",
            "coefficient": float(intercept[0]),
            "absolute_coefficient": abs(float(intercept[0])),
            "coefficient_sign": int(np.sign(intercept[0])),
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
                "coefficient_sign": int(np.sign(value)),
                "nonzero": bool(abs(value) > float(nonzero_tolerance)),
            }
        )
    frame = pd.DataFrame(rows)
    if model_name is not None:
        frame.insert(0, "model", str(model_name))
    return frame


def count_nonzero_coefficients(model: LinearSVC, tolerance: float = 1.0e-12) -> int:
    if float(tolerance) < 0:
        raise LinearSVMError("tolerance must be >= 0.")
    coefficients = np.asarray(model.coef_, dtype=float)
    if coefficients.ndim != 2 or coefficients.shape[0] != 1:
        raise LinearSVMError(f"Unexpected coefficient shape: {coefficients.shape}.")
    return int(np.sum(np.abs(coefficients.ravel()) > float(tolerance)))


def safe_roc_auc(y_true: Sequence[int], score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    if y.shape != s.shape or y.ndim != 1 or not np.isfinite(s).all():
        raise LinearSVMError("Invalid arrays for ROC-AUC.")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise LinearSVMError("ROC-AUC requires both classes.")
    return float(roc_auc_score(y, s))


def safe_average_precision(y_true: Sequence[int], score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    if y.shape != s.shape or y.ndim != 1 or not np.isfinite(s).all():
        raise LinearSVMError("Invalid arrays for Average Precision.")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise LinearSVMError("Average Precision requires both classes.")
    return float(average_precision_score(y, s))


def choose_youden_threshold(
    y_true: Sequence[int], score: Sequence[float]
) -> Tuple[float, float]:
    """Choose maximum Youden J; ties use closest-to-zero then larger threshold."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    if y.shape != s.shape or y.ndim != 1 or not np.isfinite(s).all():
        raise LinearSVMError("Invalid arrays for Youden threshold selection.")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise LinearSVMError("Youden threshold selection requires both classes.")
    fpr, tpr, thresholds = roc_curve(y, s, drop_intermediate=False)
    finite = np.isfinite(thresholds)
    if not finite.any():
        raise LinearSVMError("No finite ROC thresholds were produced.")
    fpr = fpr[finite]
    tpr = tpr[finite]
    thresholds = thresholds[finite]
    youden = tpr - fpr
    maximum = float(np.max(youden))
    candidates = np.flatnonzero(np.isclose(youden, maximum, rtol=0.0, atol=1e-12))
    chosen = sorted(
        candidates.tolist(),
        key=lambda index: (abs(float(thresholds[index])), -float(thresholds[index])),
    )[0]
    return float(thresholds[chosen]), maximum


def apply_score_threshold(score: Sequence[float], threshold: float) -> np.ndarray:
    s = np.asarray(score, dtype=float)
    if s.ndim != 1 or not np.isfinite(s).all() or not np.isfinite(float(threshold)):
        raise LinearSVMError("Invalid score or threshold.")
    return (s >= float(threshold)).astype(int)


def score_classification_metrics(
    y_true: Sequence[int], score: Sequence[float], threshold: float
) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    if y.shape != s.shape or y.ndim != 1:
        raise LinearSVMError("y_true and score must be matching 1D arrays.")
    predicted = apply_score_threshold(s, threshold)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in matrix.ravel()]
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    return {
        "roc_auc": safe_roc_auc(y, s),
        "pr_auc": safe_average_precision(y, s),
        "average_precision": safe_average_precision(y, s),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, predicted)),
        "sensitivity_recall": float(recall_score(y, predicted, zero_division=0)),
        "specificity": specificity,
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def add_natural_zero_metrics(
    metrics: Mapping[str, object], y_true: Sequence[int], score: Sequence[float]
) -> Dict[str, object]:
    result: Dict[str, object] = dict(metrics)
    zero = score_classification_metrics(y_true, score, 0.0)
    for key in (
        "accuracy",
        "sensitivity_recall",
        "specificity",
        "precision",
        "f1",
        "TN",
        "FP",
        "FN",
        "TP",
    ):
        result[f"zero_threshold_{key}"] = zero[key]
    return result
