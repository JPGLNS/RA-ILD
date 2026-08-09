#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused post-install acceptance checks for TRB Batch 18 XGBoost."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.xgboost_config import (  # noqa: E402
    build_xgboost_candidate_bank,
    candidate_bank_frame,
    load_candidate_bank,
    sha256_file,
)
from ra_ild_trb.xgboost_model import XGBoostCandidate, fit_xgboost, xgboost_version  # noqa: E402
from ra_ild_trb.xgboost_nested_cv import XGBoostNestedCVOptions  # noqa: E402
from ra_ild_trb.xgboost_summary import XGBoostSummaryError  # noqa: E402,F401

FOCUSED_PASS = "TRB_BATCH18_XGBOOST_FOCUSED_TEST_PASS"


def main() -> int:
    engine = {
        "engine": "xgboost",
        "search": {"strategy": "random", "n_iter": 100, "random_state": 20260807, "replacement": False},
        "hyperparameters": {
            "max_depth": [1, 2, 3],
            "min_child_weight": [3, 5, 10],
            "learning_rate": [0.03, 0.05],
            "subsample": [0.70, 0.85],
            "colsample_bytree": [0.30, 0.50, 0.70],
            "reg_alpha": [0.1, 0.5, 1.0],
            "reg_lambda": [3.0, 5.0, 10.0],
            "gamma": 0.0,
            "scale_pos_weight": 1.0,
            "n_estimators": 300,
        },
    }
    first, pool_size = build_xgboost_candidate_bank(engine)
    second, pool_size_2 = build_xgboost_candidate_bank(engine)
    assert pool_size == pool_size_2 == 972
    assert len(first) == len(second) == 100
    assert len({item.pool_index for item in first}) == 100
    assert [(x.candidate_id, x.pool_index) for x in first] == [(x.candidate_id, x.pool_index) for x in second]

    with tempfile.TemporaryDirectory(prefix="trb_batch18_") as tmp:
        bank = Path(tmp) / "candidate_bank.csv"
        candidate_bank_frame(first, pool_size=pool_size).to_csv(bank, index=False, lineterminator="\n")
        loaded = load_candidate_bank(bank, expected_sha256=sha256_file(bank))
        assert [(x.candidate_id, x.parameters_json) for x in loaded] == [(x.candidate_id, x.parameters_json) for x in first]

    smoke_candidate = XGBoostCandidate(
        candidate_id="acceptance_smoke", pool_index=0, sample_order=1,
        parameters={
            "n_estimators": 12, "max_depth": 2, "min_child_weight": 1,
            "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.1, "reg_lambda": 3.0, "gamma": 0.0,
            "scale_pos_weight": 1.0,
        },
    )
    rng = np.random.default_rng(20260807)
    X = rng.normal(size=(48, 6))
    y = np.array([0] * 24 + [1] * 24)
    X[24:, 0] += 1.25
    fit = fit_xgboost(
        X, y, candidate=smoke_candidate,
        fixed_parameters={"objective":"binary:logistic","eval_metric":"logloss","tree_method":"hist","verbosity":0},
        random_state=17, n_jobs=1,
    )
    probability = fit.predict_probability(X)
    assert probability.shape == (48,)
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    importance = fit.importance_frame([f"f_{i}" for i in range(6)], model_name="acceptance")
    assert len(importance) == 6
    assert {"importance_gain", "importance_weight", "importance_cover"}.issubset(importance.columns)

    options = XGBoostNestedCVOptions(
        base_seed=1, tuning_primary_metric="roc_auc", n_jobs_per_fit=1,
        fixed_parameters={"objective":"binary:logistic"},
    )
    assert options.n_jobs_per_fit == 1

    print("TRB Batch 18 XGBoost focused acceptance")
    print(f"xgboost_version={xgboost_version()}")
    print(f"candidate_pool_size={pool_size}")
    print(f"frozen_random_candidates={len(first)}")
    print("candidate_bank_deterministic=true")
    print("runtime_n_jobs_per_fit=1")
    print("probability_contract=true")
    print("feature_importance_contract=true")
    print(FOCUSED_PASS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
