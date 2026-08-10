#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pytest

from ra_ild_igh.xgboost_model import (
    XGBoostCandidate,
    XGBoostError,
    derive_xgb_seed,
    fit_xgboost,
)


def _candidate():
    return XGBoostCandidate(
        candidate_id="xgb_test",
        pool_index=0,
        sample_order=1,
        parameters={
            "n_estimators": 12,
            "max_depth": 2,
            "min_child_weight": 1,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 3.0,
            "gamma": 0.0,
            "scale_pos_weight": 1.0,
        },
    )


def test_seed_is_stable_and_context_specific():
    assert derive_xgb_seed(123, "a", 1) == derive_xgb_seed(123, "a", 1)
    assert derive_xgb_seed(123, "a", 1) != derive_xgb_seed(123, "a", 2)


def test_fit_probability_and_importance_contract():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(48, 6))
    y = np.array([0] * 24 + [1] * 24)
    X[24:, 0] += 1.4
    fit = fit_xgboost(
        X,
        y,
        candidate=_candidate(),
        fixed_parameters={
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "verbosity": 0,
        },
        random_state=7,
        n_jobs=1,
    )
    probability = fit.predict_probability(X)
    assert probability.shape == (48,)
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    importance = fit.importance_frame([f"feature_{i}" for i in range(6)], model_name="M0")
    assert len(importance) == 6
    assert set(["importance_gain", "importance_weight", "importance_cover"]).issubset(importance.columns)
    assert (importance["importance_gain"] >= 0).all()


def test_runtime_parameters_cannot_leak_into_candidate():
    candidate = XGBoostCandidate(
        candidate_id="bad",
        pool_index=0,
        sample_order=1,
        parameters={"n_estimators": 5, "random_state": 99},
    )
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.array([0] * 10 + [1] * 10)
    with pytest.raises(XGBoostError, match="Runtime-owned"):
        fit_xgboost(
            X,
            y,
            candidate=candidate,
            fixed_parameters={"objective": "binary:logistic"},
            random_state=1,
            n_jobs=1,
        )
