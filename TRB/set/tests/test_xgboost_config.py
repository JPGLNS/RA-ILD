#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd
import pytest

from ra_ild_trb.xgboost_config import (
    XGBoostConfigError,
    build_xgboost_candidate_bank,
    candidate_bank_frame,
    load_candidate_bank,
    sha256_file,
)


def _engine(n_iter=100, seed=20260807):
    return {
        "engine": "xgboost",
        "search": {"strategy": "random", "n_iter": n_iter, "random_state": seed, "replacement": False},
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


def test_user_search_space_is_972_and_samples_100_without_replacement():
    candidates, pool_size = build_xgboost_candidate_bank(_engine())
    assert pool_size == 972
    assert len(candidates) == 100
    assert len({c.pool_index for c in candidates}) == 100
    assert len({c.parameters_json for c in candidates}) == 100


def test_candidate_bank_is_deterministic_for_fixed_search_seed():
    a, _ = build_xgboost_candidate_bank(_engine())
    b, _ = build_xgboost_candidate_bank(_engine())
    c, _ = build_xgboost_candidate_bank(_engine(seed=20260808))
    assert [(x.candidate_id, x.pool_index) for x in a] == [(x.candidate_id, x.pool_index) for x in b]
    assert [x.pool_index for x in a] != [x.pool_index for x in c]


def test_candidate_bank_round_trip_and_sha(tmp_path: Path):
    candidates, pool_size = build_xgboost_candidate_bank(_engine(n_iter=7))
    path = tmp_path / "bank.csv"
    candidate_bank_frame(candidates, pool_size=pool_size).to_csv(path, index=False, lineterminator="\n")
    digest = sha256_file(path)
    loaded = load_candidate_bank(path, expected_sha256=digest)
    assert [(x.candidate_id, x.parameters_json) for x in loaded] == [(x.candidate_id, x.parameters_json) for x in candidates]


def test_random_search_refuses_more_candidates_than_pool():
    engine = _engine(n_iter=100)
    engine["hyperparameters"] = {"max_depth": [1, 2]}
    with pytest.raises(XGBoostConfigError, match="exceeds candidate pool"):
        build_xgboost_candidate_bank(engine)


def test_runtime_owned_parameters_are_rejected():
    engine = _engine(n_iter=2)
    engine["hyperparameters"] = {"max_depth": [1, 2], "n_jobs": [1, 2]}
    with pytest.raises(XGBoostConfigError, match="Runtime-owned"):
        build_xgboost_candidate_bank(engine)
