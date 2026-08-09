#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""XGBoost backend utilities for the TRB V2 framework.

The module is additive and consumes already-preprocessed numeric design matrices.
All estimator hyperparameters are supplied by configuration/candidate objects;
no study-specific tuning values are hard-coded here.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import xgboost
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover - exercised by environment preflight
    xgboost = None
    XGBClassifier = None
    _XGBOOST_IMPORT_ERROR = exc
else:
    _XGBOOST_IMPORT_ERROR = None


class XGBoostError(ValueError):
    """Raised when XGBoost inputs/configuration violate the Batch 18 contract."""


def require_xgboost() -> None:
    if XGBClassifier is None:
        raise XGBoostError(
            "xgboost is not importable in the active Python environment: "
            f"{_XGBOOST_IMPORT_ERROR}"
        )


def xgboost_version() -> str:
    require_xgboost()
    return str(getattr(xgboost, "__version__", "unknown"))


def derive_xgb_seed(base_seed: int, *parts: object) -> int:
    if isinstance(base_seed, bool) or not isinstance(base_seed, (int, np.integer)):
        raise XGBoostError("base_seed must be an integer")
    text = ":".join([str(int(base_seed))] + [str(value) for value in parts])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little") % (
        2**31 - 1
    )


def _jsonable_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise XGBoostError(f"Unsupported candidate parameter value: {value!r}")


def canonical_parameter_json(parameters: Mapping[str, Any]) -> str:
    normalized = {str(k): _jsonable_scalar(v) for k, v in sorted(parameters.items())}
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class XGBoostCandidate:
    candidate_id: str
    pool_index: int
    sample_order: int
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not str(self.candidate_id).strip():
            raise XGBoostError("candidate_id must be non-empty")
        if isinstance(self.pool_index, bool) or int(self.pool_index) < 0:
            raise XGBoostError("pool_index must be an integer >= 0")
        if isinstance(self.sample_order, bool) or int(self.sample_order) < 1:
            raise XGBoostError("sample_order must be an integer >= 1")
        if not isinstance(self.parameters, Mapping) or not self.parameters:
            raise XGBoostError("candidate parameters must be a non-empty mapping")
        normalized = {str(k): _jsonable_scalar(v) for k, v in self.parameters.items()}
        object.__setattr__(self, "parameters", normalized)

    @property
    def parameters_json(self) -> str:
        return canonical_parameter_json(self.parameters)

    def as_dict(self, *, prefix: str = "") -> Dict[str, Any]:
        result: Dict[str, Any] = {
            f"{prefix}candidate_id": self.candidate_id,
            f"{prefix}candidate_pool_index": int(self.pool_index),
            f"{prefix}candidate_sample_order": int(self.sample_order),
            f"{prefix}parameters_json": self.parameters_json,
        }
        for key, value in sorted(self.parameters.items()):
            result[f"{prefix}xgb_{key}"] = value
        return result


@dataclass(frozen=True)
class XGBoostFit:
    model: Any
    candidate: XGBoostCandidate
    random_state: int
    n_features: int

    @property
    def classes(self) -> Tuple[int, int]:
        classes = tuple(int(v) for v in np.asarray(self.model.classes_))
        if classes != (0, 1):
            raise XGBoostError(f"Unexpected fitted class order: {classes}")
        return classes

    def predict_probability(self, X: np.ndarray) -> np.ndarray:
        return predict_probability(self.model, X)

    def importance_frame(self, feature_names: Sequence[str], *, model_name: str) -> pd.DataFrame:
        return feature_importance_frame(self.model, feature_names, model_name=model_name)


def _validate_design_matrix(X: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(X, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise XGBoostError(f"{label} must be 2D with >=2 rows and >=1 column")
    if not np.isfinite(array).all():
        raise XGBoostError(f"{label} contains non-finite values")
    return array


def _validate_binary_labels(y: np.ndarray, n_samples: int) -> np.ndarray:
    labels = np.asarray(y)
    if labels.ndim != 1 or len(labels) != n_samples:
        raise XGBoostError("y must be one-dimensional and match X rows")
    if pd.isna(labels).any():
        raise XGBoostError("y contains missing values")
    labels = labels.astype(int)
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise XGBoostError("y must contain both binary classes 0 and 1")
    return labels


def fit_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    *,
    candidate: XGBoostCandidate,
    fixed_parameters: Mapping[str, Any],
    random_state: int,
    n_jobs: int,
) -> XGBoostFit:
    """Fit one XGBClassifier with deterministic per-fit random state.

    ``candidate.parameters`` wins over ``fixed_parameters`` only for tuning keys.
    Runtime-owned ``random_state`` and ``n_jobs`` cannot be injected through YAML
    hyperparameter grids, preventing accidental nested parallelism or seed drift.
    """
    require_xgboost()
    design = _validate_design_matrix(X, "X")
    labels = _validate_binary_labels(y, design.shape[0])
    if isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer)):
        raise XGBoostError("random_state must be an integer")
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1:
        raise XGBoostError("n_jobs must be an integer >= 1")

    parameters = dict(fixed_parameters)
    overlap = set(parameters) & set(candidate.parameters)
    if overlap:
        raise XGBoostError(f"Candidate/fixed parameter overlap: {sorted(overlap)}")
    parameters.update(candidate.parameters)
    reserved = {"random_state", "seed", "n_jobs", "nthread", "early_stopping_rounds", "callbacks"}
    illegal = sorted(reserved & set(parameters))
    if illegal:
        raise XGBoostError(f"Runtime-owned/unsupported XGBoost parameters: {illegal}")
    if str(parameters.get("objective", "binary:logistic")) != "binary:logistic":
        raise XGBoostError("Batch 18 requires objective='binary:logistic'")

    model = XGBClassifier(
        **parameters,
        random_state=int(random_state),
        n_jobs=int(n_jobs),
    )
    model.fit(design, labels)
    classes = tuple(int(v) for v in np.asarray(model.classes_))
    if classes != (0, 1):
        raise XGBoostError(f"Unexpected fitted class order: {classes}")
    return XGBoostFit(
        model=model,
        candidate=candidate,
        random_state=int(random_state),
        n_features=int(design.shape[1]),
    )


def predict_probability(model: Any, X: np.ndarray) -> np.ndarray:
    design = np.asarray(X, dtype=float)
    if design.ndim != 2 or design.shape[1] < 1:
        raise XGBoostError("Prediction X must be a non-empty 2D matrix")
    if not np.isfinite(design).all():
        raise XGBoostError("Prediction X contains non-finite values")
    classes = tuple(int(v) for v in np.asarray(getattr(model, "classes_", [])))
    if classes != (0, 1):
        raise XGBoostError(f"Model classes must be (0, 1); observed={classes}")
    probability = np.asarray(model.predict_proba(design), dtype=float)
    if probability.shape != (design.shape[0], 2):
        raise XGBoostError(f"Unexpected predict_proba shape: {probability.shape}")
    values = probability[:, 1]
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise XGBoostError("XGBoost produced invalid class-1 probabilities")
    return values


def _importance_key_to_index(key: str, feature_names: Sequence[str]) -> int:
    if key.startswith("f") and key[1:].isdigit():
        index = int(key[1:])
        if 0 <= index < len(feature_names):
            return index
    try:
        return list(feature_names).index(key)
    except ValueError as exc:
        raise XGBoostError(f"Cannot map booster feature key {key!r}") from exc


def feature_importance_frame(
    model: Any,
    feature_names: Sequence[str],
    *,
    model_name: str,
) -> pd.DataFrame:
    names = tuple(str(v) for v in feature_names)
    if not names or len(names) != len(set(names)):
        raise XGBoostError("feature_names must be non-empty and unique")
    booster = model.get_booster()
    importance_types = ("gain", "weight", "cover", "total_gain", "total_cover")
    values: Dict[str, np.ndarray] = {
        name: np.zeros(len(names), dtype=float) for name in importance_types
    }
    for importance_type in importance_types:
        raw = booster.get_score(importance_type=importance_type)
        for key, score in raw.items():
            index = _importance_key_to_index(str(key), names)
            values[importance_type][index] = float(score)
    frame = pd.DataFrame({"feature_name": names})
    frame.insert(0, "model", str(model_name))
    for importance_type in importance_types:
        frame[f"importance_{importance_type}"] = values[importance_type]
    frame["importance_gain_nonzero"] = frame["importance_gain"] > 0
    frame["model_engine"] = "xgboost"
    return frame
