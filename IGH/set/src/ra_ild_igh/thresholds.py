#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classification-threshold utilities for the IGH V2 framework."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


class ThresholdError(ValueError):
    """Raised when a classification threshold cannot be derived safely."""


def _validated_binary_inputs(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.ndim != 1:
        raise ThresholdError("y_true and probability must be one-dimensional.")
    if len(y) != len(p) or len(y) < 2:
        raise ThresholdError("y_true and probability must have equal length >= 2.")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ThresholdError("probability must contain finite values in [0, 1].")
    try:
        y = y.astype(int)
    except (TypeError, ValueError) as exc:
        raise ThresholdError("y_true must contain binary labels 0 and 1.") from exc
    unique = set(np.unique(y).tolist())
    if unique != {0, 1}:
        raise ThresholdError(
            f"y_true must contain both classes {{0, 1}}; observed={sorted(unique)}."
        )
    return y, p


def apply_threshold(probability: np.ndarray, threshold: float) -> np.ndarray:
    """Classify probabilities with the V1 rule ``probability >= threshold``."""
    p = np.asarray(probability, dtype=float)
    if p.ndim != 1:
        raise ThresholdError("probability must be one-dimensional.")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ThresholdError("probability must contain finite values in [0, 1].")
    value = float(threshold)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ThresholdError("threshold must be finite and in [0, 1].")
    return (p >= value).astype(int)


def choose_threshold_youden(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> float:
    """Reproduce the V1 Youden-J threshold with closest-to-0.5 tie breaking."""
    y, p = _validated_binary_inputs(y_true, probability)
    fpr, tpr, thresholds = roc_curve(y, p)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    youden = tpr[finite] - fpr[finite]
    candidates = thresholds[finite][youden == np.max(youden)]
    return float(candidates[np.argmin(np.abs(candidates - 0.5))])


def choose_threshold_max_f1(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> float:
    """Choose the PR-curve threshold with maximum F1 and closest-to-0.5 ties."""
    y, p = _validated_binary_inputs(y_true, probability)
    precision, recall, thresholds = precision_recall_curve(y, p)
    if len(thresholds) == 0:
        return 0.5
    precision = precision[:-1]
    recall = recall[:-1]
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )
    candidates = thresholds[f1 == np.max(f1)]
    return float(candidates[np.argmin(np.abs(candidates - 0.5))])


def choose_threshold_for_sensitivity(
    y_true: np.ndarray,
    probability: np.ndarray,
    target_sensitivity: float,
) -> float:
    """Choose the highest threshold achieving at least the requested sensitivity."""
    y, p = _validated_binary_inputs(y_true, probability)
    target = float(target_sensitivity)
    if not 0 < target <= 1:
        raise ThresholdError("target_sensitivity must be in (0, 1].")
    fpr, tpr, thresholds = roc_curve(y, p)
    finite = np.isfinite(thresholds)
    eligible = finite & (tpr >= target)
    if not eligible.any():
        raise ThresholdError("No finite threshold reaches target_sensitivity.")
    return float(np.max(thresholds[eligible]))
