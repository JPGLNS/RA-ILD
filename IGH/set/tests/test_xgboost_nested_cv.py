#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ra_ild_igh.nested_cv import PreparedInnerFold
from ra_ild_igh.xgboost_model import XGBoostCandidate
from ra_ild_igh.xgboost_nested_cv import XGBoostNestedCVOptions, tune_xgboost_models


def _fold(seed: int, fold: int) -> PreparedInnerFold:
    rng = np.random.default_rng(seed)
    X_train = rng.normal(size=(40, 5))
    y_train = np.array([0] * 20 + [1] * 20)
    X_train[20:, 0] += 1.2
    X_valid = rng.normal(size=(16, 5))
    y_valid = np.array([0] * 8 + [1] * 8)
    X_valid[8:, 0] += 1.2
    return PreparedInnerFold(
        X_train=X_train,
        X_valid=X_valid,
        y_train=y_train,
        y_valid=y_valid,
        train_sample_ids=tuple(f"tr{fold}_{i}" for i in range(40)),
        valid_sample_ids=tuple(f"va{fold}_{i}" for i in range(16)),
        feature_names=tuple(f"feature_{i}" for i in range(5)),
        preprocessing_audit=pd.DataFrame(),
    )


def _candidate(candidate_id: str, order: int, depth: int) -> XGBoostCandidate:
    return XGBoostCandidate(
        candidate_id=candidate_id,
        pool_index=order - 1,
        sample_order=order,
        parameters={
            "n_estimators": 12,
            "max_depth": depth,
            "min_child_weight": 1,
            "learning_rate": 0.1,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 3.0,
            "gamma": 0.0,
            "scale_pos_weight": 1.0,
        },
    )


def test_inner_tuning_runs_candidate_by_fold_and_selects_from_pooled_oof():
    prepared = {"M0": {1: _fold(101, 1), 2: _fold(202, 2)}}
    candidates = (_candidate("xgb_001", 1, 1), _candidate("xgb_002", 2, 2))
    options = XGBoostNestedCVOptions(
        base_seed=20260807,
        tuning_primary_metric="roc_auc",
        n_jobs_per_fit=1,
        fixed_parameters={
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "verbosity": 0,
        },
    )
    result = tune_xgboost_models(prepared, candidates, options, outer_repeat=3, outer_fold=1)
    assert len(result.tuning) == 2
    assert set(result.tuning["candidate_id"].astype(str)) == {"xgb_001", "xgb_002"}
    assert len(result.selected_oof_predictions) == 32
    probability = result.selected_oof_predictions["probability_ILD"].to_numpy(float)
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    selected = result.selected["M0"]
    assert selected.candidate.candidate_id in {"xgb_001", "xgb_002"}
    assert 0.0 <= float(selected.threshold) <= 1.0
    assert selected.threshold_source == "inner_oof_youden"
