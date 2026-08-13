#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified model backends and fold-local preprocessing for Phase 6.

The numerical choices mirror the established repository backends while exposing
one small common interface for Elastic Net/Ridge/Lasso, LinearSVC and XGBoost.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
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
    roc_curve,
)
from sklearn.svm import LinearSVC

from .unified_experiment import UnifiedExperimentSpec

# XGBoost is imported lazily inside a worker only when the selected algorithm
# actually needs it.  This avoids initializing XGBoost/OpenMP threads in the
# parent process before Linux fork-based repeat parallelism.
XGBClassifier = None

def _get_xgb_classifier():
    global XGBClassifier
    if XGBClassifier is None:
        try:
            from xgboost import XGBClassifier as _XGBClassifier
        except Exception as exc:  # pragma: no cover
            raise UnifiedModelError("xgboost is not importable in the active environment") from exc
        XGBClassifier = _XGBClassifier
    return XGBClassifier


class UnifiedModelError(ValueError):
    pass


def derive_seed(base_seed: int, *parts: object) -> int:
    text = ":".join([str(int(base_seed))] + [str(x) for x in parts])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little") % (2**31 - 1)


@dataclass(frozen=True)
class FittedPreprocessor:
    numeric_columns: Tuple[str, ...]
    categorical_columns: Tuple[str, ...]
    category_levels: Mapping[str, Tuple[str, ...]]
    reference_levels: Mapping[str, str]
    raw_feature_names: Tuple[str, ...]
    means: np.ndarray
    sds: np.ndarray
    keep: np.ndarray
    zero_sd_tolerance: float

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return tuple(name for name, flag in zip(self.raw_feature_names, self.keep) if bool(flag))

    def _raw_matrix(self, frame: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, int]]:
        parts: List[np.ndarray] = []
        for col in self.numeric_columns:
            values = pd.to_numeric(frame[col], errors="raise").to_numpy(float)
            if not np.isfinite(values).all():
                raise UnifiedModelError(f"non-finite numeric predictor {col}")
            parts.append(values[:, None])
        unseen: Dict[str, int] = {}
        for col in self.categorical_columns:
            values = frame[col].astype(str)
            levels = self.category_levels[col]
            unseen[col] = int((~values.isin(levels)).sum())
            reference = self.reference_levels[col]
            for level in levels:
                if level == reference:
                    continue
                parts.append((values == level).to_numpy(float)[:, None])
        if not parts:
            raise UnifiedModelError("no predictors assembled")
        return np.hstack(parts), unseen

    def transform(self, frame: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, int]]:
        raw, unseen = self._raw_matrix(frame)
        keep = np.asarray(self.keep, dtype=bool)
        X = (raw[:, keep] - self.means[keep]) / self.sds[keep]
        if not np.isfinite(X).all():
            raise UnifiedModelError("preprocessing produced non-finite values")
        return X, unseen

    def audit_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature_name": self.raw_feature_names,
            "training_mean": self.means,
            "training_sd": self.sds,
            "kept_after_zero_variance_filter": self.keep,
        })


def fit_preprocessor(
    train: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    *,
    zero_sd_tolerance: float,
) -> FittedPreprocessor:
    numeric = tuple(map(str, numeric_columns))
    categorical = tuple(map(str, categorical_columns))
    if len((*numeric, *categorical)) != len(set((*numeric, *categorical))):
        raise UnifiedModelError("predictor columns overlap or contain duplicates")
    missing = [c for c in (*numeric, *categorical) if c not in train.columns]
    if missing:
        raise UnifiedModelError(f"training frame missing predictors: {missing[:20]}")
    parts: List[np.ndarray] = []
    names: List[str] = []
    levels_map: Dict[str, Tuple[str, ...]] = {}
    ref_map: Dict[str, str] = {}
    for col in numeric:
        values = pd.to_numeric(train[col], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise UnifiedModelError(f"non-finite numeric predictor {col}")
        parts.append(values[:, None])
        names.append(col)
    for col in categorical:
        values = train[col].astype(str)
        levels = tuple(sorted(values.unique().tolist()))
        if not levels:
            raise UnifiedModelError(f"no categories in {col}")
        reference = levels[0]
        levels_map[col] = levels
        ref_map[col] = reference
        for level in levels[1:]:
            parts.append((values == level).to_numpy(float)[:, None])
            names.append(f"{col}__{level}_vs_{reference}")
    if not parts:
        raise UnifiedModelError("no predictors assembled")
    raw = np.hstack(parts)
    means = raw.mean(axis=0)
    sds = raw.std(axis=0, ddof=0)
    keep = np.isfinite(sds) & (sds > float(zero_sd_tolerance))
    if not keep.any():
        raise UnifiedModelError("all predictors were zero variance")
    return FittedPreprocessor(
        numeric_columns=numeric,
        categorical_columns=categorical,
        category_levels=levels_map,
        reference_levels=ref_map,
        raw_feature_names=tuple(names),
        means=means,
        sds=sds,
        keep=keep,
        zero_sd_tolerance=float(zero_sd_tolerance),
    )


@dataclass(frozen=True)
class PreparedFold:
    X_train: np.ndarray
    X_valid: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    train_ids: Tuple[str, ...]
    valid_ids: Tuple[str, ...]
    feature_names: Tuple[str, ...]
    preprocessing_audit: pd.DataFrame
    unseen_categories: Mapping[str, int]


def prepare_fold(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    *,
    positive_label: str,
    zero_sd_tolerance: float,
) -> PreparedFold:
    pre = fit_preprocessor(
        train,
        numeric_columns,
        categorical_columns,
        zero_sd_tolerance=zero_sd_tolerance,
    )
    X_train, _ = pre.transform(train)
    X_valid, unseen = pre.transform(valid)
    y_train = (train["cohort"].astype(str).str.upper() == positive_label.upper()).astype(int).to_numpy()
    y_valid = (valid["cohort"].astype(str).str.upper() == positive_label.upper()).astype(int).to_numpy()
    if set(np.unique(y_train)) != {0, 1} or set(np.unique(y_valid)) != {0, 1}:
        raise UnifiedModelError("every training/validation fold must contain both classes")
    return PreparedFold(
        X_train=X_train,
        X_valid=X_valid,
        y_train=y_train,
        y_valid=y_valid,
        train_ids=tuple(train["sample_id"].astype(str)),
        valid_ids=tuple(valid["sample_id"].astype(str)),
        feature_names=pre.feature_names,
        preprocessing_audit=pre.audit_frame(),
        unseen_categories=unseen,
    )


@dataclass(frozen=True)
class ModelCandidate:
    candidate_id: str
    algorithm: str
    parameters: Mapping[str, Any]
    sample_order: int = 0

    def as_dict(self) -> Dict[str, Any]:
        row = {
            "candidate_id": self.candidate_id,
            "algorithm": self.algorithm,
            "candidate_sample_order": self.sample_order,
            "parameters_json": json.dumps(dict(self.parameters), sort_keys=True, separators=(",", ":")),
        }
        row.update({f"parameter_{k}": v for k, v in sorted(self.parameters.items())})
        return row


def _xgb_candidate_bank(spec: UnifiedExperimentSpec) -> List[ModelCandidate]:
    settings = spec.model.xgboost
    names = list(settings.hyperparameters)
    value_lists = [list(settings.hyperparameters[name]) for name in names]
    pool = [dict(zip(names, values)) for values in itertools.product(*value_lists)]
    if settings.search_n_iter > len(pool):
        raise UnifiedModelError(
            f"xgboost search_n_iter={settings.search_n_iter} exceeds candidate pool={len(pool)}"
        )
    rng = np.random.default_rng(settings.search_random_state)
    selected = rng.choice(len(pool), size=settings.search_n_iter, replace=False)
    result = []
    for order, index in enumerate(selected.tolist(), start=1):
        parameters = pool[index]
        digest = hashlib.sha256(
            json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:10]
        result.append(ModelCandidate(
            candidate_id=f"xgb_{order:03d}_{digest}",
            algorithm="xgboost",
            parameters=parameters,
            sample_order=order,
        ))
    return result


def build_candidates(spec: UnifiedExperimentSpec) -> List[ModelCandidate]:
    algorithm = spec.algorithm
    if algorithm in {"elastic_net", "ridge", "lasso"}:
        grid = getattr(spec.model, algorithm)
        result = []
        order = 0
        for alpha in grid.alpha_grid:
            for lam in grid.lambda_grid:
                order += 1
                result.append(ModelCandidate(
                    candidate_id=f"{algorithm}_a{alpha:g}_l{lam:g}",
                    algorithm=algorithm,
                    parameters={"alpha": float(alpha), "lambda": float(lam)},
                    sample_order=order,
                ))
        return result
    if algorithm == "linear_svm":
        return [
            ModelCandidate(
                candidate_id=f"linear_svm_C{C:g}",
                algorithm=algorithm,
                parameters={
                    "C": float(C),
                    "penalty": spec.model.linear_svm.penalty,
                    "loss": spec.model.linear_svm.loss,
                    "class_weight": spec.model.linear_svm.class_weight,
                    "dual": spec.model.linear_svm.dual,
                },
                sample_order=i,
            )
            for i, C in enumerate(spec.model.linear_svm.C_grid, start=1)
        ]
    if algorithm == "xgboost":
        return _xgb_candidate_bank(spec)
    raise UnifiedModelError(f"unsupported algorithm: {algorithm}")


def _class_weight(value: str):
    return None if str(value).lower() == "none" else "balanced"


@dataclass(frozen=True)
class FittedUnifiedModel:
    algorithm: str
    candidate: ModelCandidate
    model: Any
    converged: bool
    iterations: Optional[int]

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.algorithm == "linear_svm":
            values = np.asarray(self.model.decision_function(X), dtype=float)
        else:
            values = np.asarray(self.model.predict_proba(X)[:, 1], dtype=float)
        if values.shape != (X.shape[0],) or not np.isfinite(values).all():
            raise UnifiedModelError("model produced invalid scores")
        return values

    @property
    def score_type(self) -> str:
        return "decision_function" if self.algorithm == "linear_svm" else "probability"

    def explanation_frame(self, feature_names: Sequence[str]) -> pd.DataFrame:
        names = tuple(map(str, feature_names))
        if self.algorithm in {"elastic_net", "ridge", "lasso", "linear_svm"}:
            coef = np.asarray(self.model.coef_, dtype=float).ravel()
            rows = [{
                "feature_name": "__INTERCEPT__",
                "coefficient": float(np.asarray(self.model.intercept_, dtype=float).ravel()[0]),
                "absolute_coefficient": abs(float(np.asarray(self.model.intercept_, dtype=float).ravel()[0])),
                "nonzero": True,
            }]
            rows.extend({
                "feature_name": name,
                "coefficient": float(value),
                "absolute_coefficient": abs(float(value)),
                "nonzero": bool(abs(float(value)) > 1e-12),
            } for name, value in zip(names, coef))
            frame = pd.DataFrame(rows)
        else:
            gain = np.asarray(getattr(self.model, "feature_importances_", np.zeros(len(names))), dtype=float)
            frame = pd.DataFrame({"feature_name": names, "importance_gain": gain})
        frame.insert(0, "algorithm", self.algorithm)
        frame.insert(1, "candidate_id", self.candidate.candidate_id)
        return frame


def fit_candidate(
    candidate: ModelCandidate,
    X: np.ndarray,
    y: np.ndarray,
    spec: UnifiedExperimentSpec,
    *,
    random_state: int,
) -> FittedUnifiedModel:
    algorithm = candidate.algorithm
    if algorithm in {"elastic_net", "ridge", "lasso"}:
        grid = getattr(spec.model, algorithm)
        alpha = float(candidate.parameters["alpha"])
        lam = float(candidate.parameters["lambda"])
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=alpha,
            C=1.0 / lam,
            class_weight=_class_weight(grid.class_weight),
            max_iter=grid.max_iter,
            tol=grid.tolerance,
            random_state=int(random_state),
            fit_intercept=True,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X, y)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        iterations = int(np.max(model.n_iter_))
        return FittedUnifiedModel(algorithm, candidate, model, converged, iterations)
    if algorithm == "linear_svm":
        s = spec.model.linear_svm
        model = LinearSVC(
            C=float(candidate.parameters["C"]),
            penalty=s.penalty,
            loss=s.loss,
            class_weight=_class_weight(s.class_weight),
            dual="auto",
            max_iter=s.max_iter,
            tol=s.tolerance,
            random_state=int(random_state),
            fit_intercept=True,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X, y)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        iterations = int(model.n_iter_)
        return FittedUnifiedModel(algorithm, candidate, model, converged, iterations)
    if algorithm == "xgboost":
        classifier = _get_xgb_classifier()
        settings = spec.model.xgboost
        parameters = dict(settings.fixed_parameters)
        parameters.update(candidate.parameters)
        model = classifier(
            **parameters,
            random_state=int(random_state),
            n_jobs=int(settings.n_jobs_per_fit),
        )
        model.fit(X, y)
        return FittedUnifiedModel(algorithm, candidate, model, True, None)
    raise UnifiedModelError(f"unsupported algorithm {algorithm}")


def safe_roc_auc(y: Sequence[int], score: Sequence[float]) -> float:
    return float(roc_auc_score(np.asarray(y, int), np.asarray(score, float)))


def safe_pr_auc(y: Sequence[int], score: Sequence[float]) -> float:
    return float(average_precision_score(np.asarray(y, int), np.asarray(score, float)))


def choose_youden_threshold(y: Sequence[int], score: Sequence[float], *, score_type: str) -> float:
    y = np.asarray(y, dtype=int)
    s = np.asarray(score, dtype=float)
    fpr, tpr, thresholds = roc_curve(y, s, drop_intermediate=False)
    finite = np.isfinite(thresholds)
    thresholds = thresholds[finite]
    youden = (tpr - fpr)[finite]
    if not len(thresholds):
        return 0.5 if score_type == "probability" else 0.0
    maximum = np.max(youden)
    candidates = thresholds[np.isclose(youden, maximum, rtol=0.0, atol=1e-12)]
    anchor = 0.5 if score_type == "probability" else 0.0
    # old probability behavior: closest 0.5; old SVM behavior: closest zero, then larger
    distance = np.abs(candidates - anchor)
    best_distance = np.min(distance)
    tied = candidates[np.isclose(distance, best_distance, rtol=0.0, atol=1e-12)]
    return float(np.max(tied))


def classification_metrics(
    y: Sequence[int],
    score: Sequence[float],
    threshold: float,
    *,
    score_type: str,
) -> Dict[str, Any]:
    y = np.asarray(y, dtype=int)
    s = np.asarray(score, dtype=float)
    pred = (s >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel().astype(int)
    result: Dict[str, Any] = {
        "roc_auc": safe_roc_auc(y, s),
        "pr_auc": safe_pr_auc(y, s),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "sensitivity_recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    if score_type == "probability":
        result["log_loss"] = float(log_loss(y, s, labels=[0, 1]))
        result["brier_score"] = float(brier_score_loss(y, s))
        result["probability_metrics_available"] = True
    else:
        result["log_loss"] = float("nan")
        result["brier_score"] = float("nan")
        result["probability_metrics_available"] = False
    return result


def candidate_tuning_row(
    candidate: ModelCandidate,
    y: Sequence[int],
    score: Sequence[float],
    *,
    score_type: str,
    all_converged: bool,
    min_features: int,
    max_features: int,
) -> Dict[str, Any]:
    y = np.asarray(y, int)
    s = np.asarray(score, float)
    row = candidate.as_dict()
    row.update({
        "pooled_inner_roc_auc": safe_roc_auc(y, s),
        "pooled_inner_pr_auc": safe_pr_auc(y, s),
        "all_fits_converged": bool(all_converged),
        "eligible_for_selection": bool(all_converged),
        "min_final_predictors": int(min_features),
        "max_final_predictors": int(max_features),
    })
    if score_type == "probability":
        row["pooled_inner_log_loss"] = float(log_loss(y, s, labels=[0, 1]))
        row["pooled_inner_brier_score"] = float(brier_score_loss(y, s))
    else:
        row["pooled_inner_log_loss"] = float("nan")
        row["pooled_inner_brier_score"] = float("nan")
    return row


def select_candidate(tuning: pd.DataFrame, spec: UnifiedExperimentSpec) -> pd.Series:
    eligible = tuning.loc[tuning["eligible_for_selection"].astype(bool)].copy()
    if eligible.empty:
        raise UnifiedModelError("no tuning candidate converged in all inner folds")
    algorithm = spec.algorithm
    metric = spec.model.tuning_primary_metric
    if algorithm in {"elastic_net", "ridge", "lasso"}:
        if metric == "roc_auc":
            columns = ["pooled_inner_roc_auc", "pooled_inner_pr_auc", "parameter_lambda", "parameter_alpha", "candidate_id"]
            ascending = [False, False, False, False, True]
        else:
            columns = ["pooled_inner_log_loss", "pooled_inner_brier_score", "pooled_inner_roc_auc", "parameter_lambda", "parameter_alpha", "candidate_id"]
            ascending = [True, True, False, False, False, True]
    elif algorithm == "linear_svm":
        columns = ["pooled_inner_roc_auc", "pooled_inner_pr_auc", "parameter_C", "candidate_id"]
        ascending = [False, False, True, True]
    else:
        if metric == "roc_auc":
            columns = ["pooled_inner_roc_auc", "pooled_inner_pr_auc", "pooled_inner_log_loss", "candidate_sample_order", "candidate_id"]
            ascending = [False, False, True, True, True]
        else:
            columns = ["pooled_inner_log_loss", "pooled_inner_roc_auc", "pooled_inner_pr_auc", "candidate_sample_order", "candidate_id"]
            ascending = [True, False, False, True, True]
    return eligible.sort_values(columns, ascending=ascending, kind="mergesort").iloc[0]
