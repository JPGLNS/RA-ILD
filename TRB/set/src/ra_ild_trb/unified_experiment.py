#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase-6 unified ML/CV experiment contract.

This layer is additive on top of Phase-5 ``experiment_spec``.  It keeps the
validated repertoire / feature / repeats / workers contract and adds the
model-tuning and repeated-holdout settings needed by the unified CV runner.

The defaults intentionally reproduce the existing repository's established
modeling choices:

* Elastic Net: alpha=[0.1,0.5,0.9], lambda=[0.01,0.03,0.1,0.3,1,3,10,30]
* Ridge/Lasso: same lambda grid with alpha fixed to 0 / 1
* LinearSVC: L2 squared-hinge balanced, C=1e-4..1e3
* XGBoost: deterministic 100-candidate random search from the existing grid
* positive class for ML: ILD (RA is the negative class)
* repeated holdout outside, optimized 5-fold tuning inside
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional, Tuple, Union

import yaml

from .experiment_spec import AdvancedExperimentSpec, load_advanced_experiment

PathLike = Union[str, Path]


class UnifiedExperimentError(ValueError):
    pass


ELASTIC_ALPHA_DEFAULT = (0.1, 0.5, 0.9)
LAMBDA_DEFAULT = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
SVM_C_DEFAULT = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
XGB_FIXED_DEFAULT = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "verbosity": 0,
}
XGB_HYPER_DEFAULT = {
    "max_depth": (1, 2, 3),
    "min_child_weight": (3, 5, 10),
    "learning_rate": (0.03, 0.05),
    "subsample": (0.70, 0.85),
    "colsample_bytree": (0.30, 0.50, 0.70, 1.00),
    "reg_alpha": (0.1, 0.5, 1.0),
    "reg_lambda": (3.0, 5.0, 10.0),
    "gamma": (0.0,),
    "scale_pos_weight": (1.0,),
    "n_estimators": (200, 300, 500),
}


def _mapping(value: Any, context: str, *, default_empty: bool = False) -> Mapping[str, Any]:
    if value is None and default_empty:
        return {}
    if not isinstance(value, Mapping):
        raise UnifiedExperimentError(f"{context} must be a mapping")
    return value


def _number_tuple(
    value: Any,
    context: str,
    *,
    default: Sequence[float],
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    strict_minimum: bool = False,
) -> Tuple[float, ...]:
    if value is None:
        values = tuple(float(x) for x in default)
    else:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
            raise UnifiedExperimentError(f"{context} must be a non-empty list")
        values = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise UnifiedExperimentError(f"{context} must contain numeric values")
            number = float(item)
            if minimum is not None:
                bad = number <= minimum if strict_minimum else number < minimum
                if bad:
                    op = ">" if strict_minimum else ">="
                    raise UnifiedExperimentError(f"{context} values must be {op} {minimum}")
            if maximum is not None and number > maximum:
                raise UnifiedExperimentError(f"{context} values must be <= {maximum}")
            values.append(number)
        values = tuple(values)
    if len(values) != len(set(values)):
        raise UnifiedExperimentError(f"{context} contains duplicates")
    return tuple(values)


def _positive_int(value: Any, context: str, default: int) -> int:
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise UnifiedExperimentError(f"{context} must be an integer >= 1")
    return int(value)


def _nonnegative_int(value: Any, context: str, default: int) -> int:
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UnifiedExperimentError(f"{context} must be an integer >= 0")
    return int(value)


def _positive_float(value: Any, context: str, default: float) -> float:
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise UnifiedExperimentError(f"{context} must be numeric > 0")
    return float(value)


def _nonnegative_float(value: Any, context: str, default: float) -> float:
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0:
        raise UnifiedExperimentError(f"{context} must be numeric >= 0")
    return float(value)


def _scalar_or_tuple(value: Any) -> Tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            raise UnifiedExperimentError("XGBoost hyperparameter lists must not be empty")
        return tuple(value)
    return (value,)


@dataclass(frozen=True)
class LinearModelGrid:
    alpha_grid: Tuple[float, ...]
    lambda_grid: Tuple[float, ...]
    class_weight: str = "balanced"
    max_iter: int = 10000
    tolerance: float = 1.0e-4


@dataclass(frozen=True)
class LinearSVMGrid:
    C_grid: Tuple[float, ...]
    penalty: str = "l2"
    loss: str = "squared_hinge"
    class_weight: str = "balanced"
    dual: str = "auto"
    max_iter: int = 10000
    tolerance: float = 1.0e-4


@dataclass(frozen=True)
class XGBoostGrid:
    fixed_parameters: Mapping[str, Any]
    hyperparameters: Mapping[str, Tuple[Any, ...]]
    search_n_iter: int = 100
    search_random_state: int = 20260807
    n_jobs_per_fit: int = 1


@dataclass(frozen=True)
class UnifiedModelSpec:
    algorithm: str
    positive_label: str
    negative_label: str
    tuning_primary_metric: str
    zero_sd_tolerance: float
    elastic_net: LinearModelGrid
    ridge: LinearModelGrid
    lasso: LinearModelGrid
    linear_svm: LinearSVMGrid
    xgboost: XGBoostGrid


@dataclass(frozen=True)
class UnifiedCVSpec:
    repeats: int
    inner_folds: int
    inner_split_candidates: int
    inner_split_seed: int
    split_assignments: Optional[str]


@dataclass(frozen=True)
class UnifiedExperimentSpec:
    base: AdvancedExperimentSpec
    model: UnifiedModelSpec
    cv: UnifiedCVSpec
    source_path: Path

    @property
    def algorithm(self) -> str:
        return self.model.algorithm

    @property
    def workers(self) -> int:
        return self.base.compute.workers

    @property
    def repertoire_source_id(self) -> str:
        return self.base.repertoire_source_id

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phase5": self.base.as_dict(),
            "unified_ml": {
                "algorithm": self.model.algorithm,
                "positive_label": self.model.positive_label,
                "negative_label": self.model.negative_label,
                "tuning_primary_metric": self.model.tuning_primary_metric,
                "zero_sd_tolerance": self.model.zero_sd_tolerance,
                "elastic_net": {
                    "alpha_grid": list(self.model.elastic_net.alpha_grid),
                    "lambda_grid": list(self.model.elastic_net.lambda_grid),
                },
                "ridge": {
                    "alpha_grid": list(self.model.ridge.alpha_grid),
                    "lambda_grid": list(self.model.ridge.lambda_grid),
                },
                "lasso": {
                    "alpha_grid": list(self.model.lasso.alpha_grid),
                    "lambda_grid": list(self.model.lasso.lambda_grid),
                },
                "linear_svm": {
                    "C_grid": list(self.model.linear_svm.C_grid),
                    "penalty": self.model.linear_svm.penalty,
                    "loss": self.model.linear_svm.loss,
                    "class_weight": self.model.linear_svm.class_weight,
                },
                "xgboost": {
                    "fixed_parameters": dict(self.model.xgboost.fixed_parameters),
                    "hyperparameters": {
                        k: list(v) for k, v in self.model.xgboost.hyperparameters.items()
                    },
                    "search_n_iter": self.model.xgboost.search_n_iter,
                    "search_random_state": self.model.xgboost.search_random_state,
                    "n_jobs_per_fit": self.model.xgboost.n_jobs_per_fit,
                },
            },
            "cv": {
                "strategy": "frozen_repeated_holdout_with_inner_cv",
                "repeats": self.cv.repeats,
                "inner_folds": self.cv.inner_folds,
                "inner_split_candidates": self.cv.inner_split_candidates,
                "inner_split_seed": self.cv.inner_split_seed,
                "split_assignments": self.cv.split_assignments or "auto",
            },
            "workers": self.workers,
        }


def _linear_grid(raw: Mapping[str, Any], *, alpha_default: Sequence[float]) -> LinearModelGrid:
    alpha = _number_tuple(
        raw.get("alpha_grid"),
        "model.parameters.alpha_grid",
        default=alpha_default,
        minimum=0.0,
        maximum=1.0,
    )
    lambdas = _number_tuple(
        raw.get("lambda_grid"),
        "model.parameters.lambda_grid",
        default=LAMBDA_DEFAULT,
        minimum=0.0,
        strict_minimum=True,
    )
    class_weight = str(raw.get("class_weight", "balanced")).strip().lower()
    if class_weight not in {"balanced", "none"}:
        raise UnifiedExperimentError("linear class_weight must be balanced or none")
    return LinearModelGrid(
        alpha_grid=alpha,
        lambda_grid=lambdas,
        class_weight=class_weight,
        max_iter=_positive_int(raw.get("max_iter"), "linear max_iter", 10000),
        tolerance=_positive_float(raw.get("tolerance"), "linear tolerance", 1.0e-4),
    )


def load_unified_experiment(path: PathLike) -> UnifiedExperimentSpec:
    source = Path(path).expanduser().resolve()
    base = load_advanced_experiment(source)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw = _mapping(raw, "experiment YAML")
    model_raw = _mapping(raw.get("model", {}), "model")
    algorithm = base.model.algorithm
    positive = str(model_raw.get("positive_label", "ILD")).strip().upper()
    negative = str(model_raw.get("negative_label", "RA")).strip().upper()
    if {positive, negative} != {"RA", "ILD"} or positive == negative:
        raise UnifiedExperimentError("model positive/negative labels must be RA and ILD")
    tuning = str(model_raw.get("tuning_primary_metric", "roc_auc")).strip().lower()
    if tuning not in {"roc_auc", "log_loss"}:
        raise UnifiedExperimentError("model.tuning_primary_metric must be roc_auc or log_loss")
    if algorithm == "linear_svm" and tuning != "roc_auc":
        raise UnifiedExperimentError("linear_svm supports tuning_primary_metric=roc_auc only")

    parameters = _mapping(model_raw.get("parameters", {}), "model.parameters", default_empty=True)
    en_raw = _mapping(parameters.get("elastic_net", {}), "model.parameters.elastic_net", default_empty=True)
    ridge_raw = _mapping(parameters.get("ridge", {}), "model.parameters.ridge", default_empty=True)
    lasso_raw = _mapping(parameters.get("lasso", {}), "model.parameters.lasso", default_empty=True)
    svm_raw = _mapping(parameters.get("linear_svm", {}), "model.parameters.linear_svm", default_empty=True)
    xgb_raw = _mapping(parameters.get("xgboost", {}), "model.parameters.xgboost", default_empty=True)

    elastic = _linear_grid(en_raw, alpha_default=ELASTIC_ALPHA_DEFAULT)
    ridge = _linear_grid(ridge_raw, alpha_default=(0.0,))
    lasso = _linear_grid(lasso_raw, alpha_default=(1.0,))
    if ridge.alpha_grid != (0.0,):
        raise UnifiedExperimentError("ridge alpha_grid is fixed to [0.0]")
    if lasso.alpha_grid != (1.0,):
        raise UnifiedExperimentError("lasso alpha_grid is fixed to [1.0]")

    C_grid = _number_tuple(
        svm_raw.get("C_grid"),
        "model.parameters.linear_svm.C_grid",
        default=SVM_C_DEFAULT,
        minimum=0.0,
        strict_minimum=True,
    )
    penalty = str(svm_raw.get("penalty", "l2")).strip().lower()
    loss = str(svm_raw.get("loss", "squared_hinge")).strip().lower()
    cw = str(svm_raw.get("class_weight", "balanced")).strip().lower()
    dual = str(svm_raw.get("dual", "auto")).strip().lower()
    if penalty != "l2" or loss != "squared_hinge" or cw not in {"balanced", "none"} or dual != "auto":
        raise UnifiedExperimentError(
            "Phase-6 default Linear SVM contract is l2 + squared_hinge + dual=auto; class_weight balanced/none"
        )
    svm = LinearSVMGrid(
        C_grid=C_grid,
        penalty=penalty,
        loss=loss,
        class_weight=cw,
        dual=dual,
        max_iter=_positive_int(svm_raw.get("max_iter"), "linear_svm max_iter", 10000),
        tolerance=_positive_float(svm_raw.get("tolerance"), "linear_svm tolerance", 1.0e-4),
    )

    fixed = dict(XGB_FIXED_DEFAULT)
    fixed_override = xgb_raw.get("fixed_parameters")
    if fixed_override is not None:
        fixed_override = _mapping(fixed_override, "model.parameters.xgboost.fixed_parameters")
        fixed.update(dict(fixed_override))
    if str(fixed.get("objective", "")) != "binary:logistic":
        raise UnifiedExperimentError("xgboost objective must be binary:logistic")
    illegal_runtime = {"random_state", "seed", "n_jobs", "nthread", "early_stopping_rounds", "callbacks"}
    if illegal_runtime & set(fixed):
        raise UnifiedExperimentError("xgboost fixed_parameters contains runtime-owned keys")

    hyper_raw = xgb_raw.get("hyperparameters")
    if hyper_raw is None:
        hyper = {k: tuple(v) for k, v in XGB_HYPER_DEFAULT.items()}
    else:
        hyper_raw = _mapping(hyper_raw, "model.parameters.xgboost.hyperparameters")
        hyper = {str(k): _scalar_or_tuple(v) for k, v in hyper_raw.items()}
        if not hyper:
            raise UnifiedExperimentError("xgboost hyperparameters must not be empty")
    if illegal_runtime & set(hyper):
        raise UnifiedExperimentError("xgboost hyperparameters contains runtime-owned keys")
    if set(fixed) & set(hyper):
        raise UnifiedExperimentError("xgboost fixed_parameters and hyperparameters overlap")
    search_n = _positive_int(xgb_raw.get("search_n_iter"), "xgboost search_n_iter", 100)
    search_seed = _nonnegative_int(
        xgb_raw.get("search_random_state"), "xgboost search_random_state", 20260807
    )
    n_jobs_fit = _positive_int(xgb_raw.get("n_jobs_per_fit"), "xgboost n_jobs_per_fit", 1)
    xgb = XGBoostGrid(
        fixed_parameters=fixed,
        hyperparameters=hyper,
        search_n_iter=search_n,
        search_random_state=search_seed,
        n_jobs_per_fit=n_jobs_fit,
    )

    validation_raw = _mapping(raw.get("validation", {}), "validation")
    split_assignments = validation_raw.get("split_assignments")
    if split_assignments in (None, "", "auto"):
        split_assignments = None
    elif not isinstance(split_assignments, str):
        raise UnifiedExperimentError("validation.split_assignments must be a path string or auto")
    cv = UnifiedCVSpec(
        repeats=base.validation.repeats,
        inner_folds=base.validation.folds,
        inner_split_candidates=_positive_int(
            validation_raw.get("inner_split_candidates"),
            "validation.inner_split_candidates",
            500,
        ),
        inner_split_seed=_nonnegative_int(
            validation_raw.get("inner_split_seed"),
            "validation.inner_split_seed",
            20260724,
        ),
        split_assignments=split_assignments,
    )

    model = UnifiedModelSpec(
        algorithm=algorithm,
        positive_label=positive,
        negative_label=negative,
        tuning_primary_metric=tuning,
        zero_sd_tolerance=_nonnegative_float(
            model_raw.get("zero_sd_tolerance"), "model.zero_sd_tolerance", 1.0e-12
        ),
        elastic_net=elastic,
        ridge=ridge,
        lasso=lasso,
        linear_svm=svm,
        xgboost=xgb,
    )
    return UnifiedExperimentSpec(base=base, model=model, cv=cv, source_path=source)


def auto_split_assignment_path(spec: UnifiedExperimentSpec, repository_root: PathLike) -> Path:
    root = Path(repository_root).expanduser().resolve()
    if spec.cv.split_assignments:
        candidate = Path(spec.cv.split_assignments).expanduser()
        return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    mapping = {
        "total": "TRB/set/split_sets/ra_ild_repeat100_v1/repeated_holdout_assignments.csv",
        "pbmc": "TRB/set/split_sets/ra_ild_pbmc_repeat100_v1/repeated_holdout_assignments.csv",
        "buffycoat": "TRB/set/split_sets/ra_ild_buffercoat_repeat100_v1/repeated_holdout_assignments.csv",
    }
    return (root / mapping[spec.base.cohort.scope]).resolve()
