#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Binary-classification metrics and bootstrap intervals for IGH V2."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .thresholds import ThresholdError, apply_threshold


class MetricError(ValueError):
    """Raised when binary metrics cannot be calculated safely."""


BOOTSTRAP_METRICS: Tuple[str, ...] = (
    "roc_auc",
    "pr_auc",
    "accuracy",
    "sensitivity_recall",
    "specificity",
    "precision",
    "f1",
    "log_loss",
    "brier_score",
)


def _validated_inputs(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.ndim != 1:
        raise MetricError("y_true and probability must be one-dimensional.")
    if len(y) != len(p) or len(y) < 2:
        raise MetricError("y_true and probability must have equal length >= 2.")
    if pd.isna(y).any():
        raise MetricError("y_true contains missing values.")
    try:
        y = y.astype(int)
    except (TypeError, ValueError) as exc:
        raise MetricError("y_true must contain binary labels 0 and 1.") from exc
    unique = set(np.unique(y).tolist())
    if unique != {0, 1}:
        raise MetricError(
            f"y_true must contain both classes {{0, 1}}; observed={sorted(unique)}."
        )
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise MetricError("probability must contain finite values in [0, 1].")
    return y, p


def safe_roc_auc(y_true: np.ndarray, probability: np.ndarray) -> float:
    y, p = _validated_inputs(y_true, probability)
    return float(roc_auc_score(y, p))


def safe_pr_auc(y_true: np.ndarray, probability: np.ndarray) -> float:
    y, p = _validated_inputs(y_true, probability)
    return float(average_precision_score(y, p))


def classification_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    *,
    include_brier: bool = False,
    include_log_loss: bool = False,
    include_sample_summary: bool = False,
) -> Dict[str, float]:
    """Calculate the V1 metric names and confusion-matrix counts."""
    y, p = _validated_inputs(y_true, probability)
    try:
        prediction = apply_threshold(p, threshold)
    except ThresholdError as exc:
        raise MetricError(str(exc)) from exc

    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    result: Dict[str, float] = {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, prediction)),
        "sensitivity_recall": float(
            recall_score(y, prediction, zero_division=0)
        ),
        "specificity": float(specificity),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
    if include_log_loss:
        result["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
    if include_brier:
        result["brier_score"] = float(brier_score_loss(y, p))
    if include_sample_summary:
        result.update(
            {
                "n_test": int(len(y)),
                "n_RA": int(np.sum(y == 0)),
                "n_ILD": int(np.sum(y == 1)),
                "positive_prevalence": float(np.mean(y)),
            }
        )
    return result


def stratified_bootstrap_ci(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    *,
    reps: int,
    seed: int,
    metric_names: Sequence[str] = BOOTSTRAP_METRICS,
) -> pd.DataFrame:
    """Reproduce the V1 stratified percentile bootstrap implementation."""
    y, p = _validated_inputs(y_true, probability)
    if isinstance(reps, bool) or int(reps) < 100:
        raise MetricError("reps must be an integer >= 100.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise MetricError("seed must be an integer.")

    names = tuple(str(name) for name in metric_names)
    unknown = sorted(set(names) - set(BOOTSTRAP_METRICS))
    if unknown:
        raise MetricError(f"Unsupported bootstrap metrics: {unknown}")
    if not names or len(names) != len(set(names)):
        raise MetricError("metric_names must be non-empty and unique.")

    rng = np.random.default_rng(int(seed))
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    values: Dict[str, List[float]] = {name: [] for name in names}

    for _ in range(int(reps)):
        sampled_negative = rng.choice(negative, size=len(negative), replace=True)
        sampled_positive = rng.choice(positive, size=len(positive), replace=True)
        index = np.concatenate([sampled_negative, sampled_positive])
        rng.shuffle(index)
        metrics = classification_metrics(
            y[index],
            p[index],
            threshold,
            include_brier=True,
            include_log_loss=True,
        )
        for name in names:
            values[name].append(float(metrics[name]))

    observed = classification_metrics(
        y, p, threshold, include_brier=True, include_log_loss=True
    )
    rows = []
    for name in names:
        distribution = np.asarray(values[name], dtype=float)
        rows.append(
            {
                "metric": name,
                "estimate": float(observed[name]),
                "bootstrap_reps": int(reps),
                "ci95_lower": float(np.quantile(distribution, 0.025)),
                "ci95_upper": float(np.quantile(distribution, 0.975)),
                "bootstrap_mean": float(np.mean(distribution)),
                "bootstrap_sd": float(np.std(distribution, ddof=1)),
                "bootstrap_method": "stratified percentile",
            }
        )
    return pd.DataFrame(rows)
