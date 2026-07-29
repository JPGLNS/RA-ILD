#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused tests for Batch 09 alpha-grid overrides and alpha endpoints."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
SET_DIR = SCRIPT_DIR.parent
REPOSITORY_ROOT = SET_DIR.parents[1]
SRC_DIR = SET_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.config import validate_experiment_mapping
from ra_ild_igh.modeling import fit_elastic_net
from ra_ild_igh.specifications import build_hyperparameter_grid


ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]


def load_preparer_module():
    path = SCRIPT_DIR / "prepare_feature_ablation_scheme.py"
    spec = importlib.util.spec_from_file_location("batch09_preparer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_alpha_grid_and_endpoints() -> None:
    candidates = build_hyperparameter_grid(ALPHAS, [0.1, 1.0])
    assert len(candidates) == 14
    assert candidates[0].alpha == 0.0
    assert candidates[-1].alpha == 1.0

    X = np.asarray(
        [
            [0.0, 0.0, 0.2],
            [0.0, 1.0, 0.1],
            [1.0, 0.0, 0.4],
            [1.0, 1.0, 0.8],
            [2.0, 1.0, 1.2],
            [1.0, 2.0, 1.1],
            [2.0, 2.0, 1.6],
            [0.2, 0.1, 0.0],
        ],
        dtype=float,
    )
    y = np.asarray([0, 0, 0, 1, 1, 1, 1, 0], dtype=int)
    for alpha in (0.0, 0.5, 1.0):
        fitted = fit_elastic_net(
            X,
            y,
            l1_ratio=alpha,
            lambda_value=1.0,
            class_weight="none",
            max_iter=20000,
            tolerance=1.0e-6,
            random_state=123,
        )
        probability = fitted.predict_probability(X)
        assert probability.shape == (len(X),)
        assert np.isfinite(probability).all()
        assert fitted.config.l1_ratio == alpha


def test_config_validation_accepts_zero() -> None:
    base_path = (
        REPOSITORY_ROOT
        / "IGH/set/experiments/igh_scheme_003_repeat3_no_clinical/00_config/resolved_config.yaml"
    )
    raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    raw["model_engine"]["alpha_grid"] = list(ALPHAS)
    model_count = len(raw["models"])
    candidate_count = len(ALPHAS) * len(raw["model_engine"]["lambda_grid"])
    raw["nested_cv"]["expected_inner_fits_per_task"] = (
        model_count * int(raw["cross_validation"]["inner_folds"]) * candidate_count
    )
    validate_experiment_mapping(raw)


def test_scheme_override_and_count_recalculation() -> None:
    preparer = load_preparer_module()
    template = (
        REPOSITORY_ROOT
        / "IGH/set/configs/schemes/igh_scheme_005_parameter_tuning_template.yaml"
    )
    source = yaml.safe_load(template.read_text(encoding="utf-8"))
    test_id = "batch09_alpha_grid_test_tmp"
    source["scheme"]["id"] = test_id
    source["scheme"]["output_root"] = f"IGH/set/experiments/{test_id}"

    temporary_scheme = (
        REPOSITORY_ROOT
        / "IGH/set/configs/schemes/_batch09_alpha_grid_test_tmp.yaml"
    )
    output_root = REPOSITORY_ROOT / "IGH/set/experiments" / test_id
    try:
        if output_root.exists():
            shutil.rmtree(output_root)
        temporary_scheme.write_text(
            yaml.safe_dump(source, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        resolved_path = preparer.prepare(
            temporary_scheme,
            REPOSITORY_ROOT,
            overwrite=False,
        )
        resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        assert resolved["model_engine"]["alpha_grid"] == ALPHAS
        assert len(resolved["models"]) == 2
        expected = (
            len(resolved["models"])
            * int(resolved["cross_validation"]["inner_folds"])
            * len(resolved["model_engine"]["alpha_grid"])
            * len(resolved["model_engine"]["lambda_grid"])
        )
        assert resolved["nested_cv"]["expected_inner_fits_per_task"] == expected
        assert expected == 560, (
            "The current base configuration is expected to use 5 inner folds "
            "and 8 lambda values."
        )
    finally:
        temporary_scheme.unlink(missing_ok=True)
        if output_root.exists():
            shutil.rmtree(output_root)


def main() -> int:
    tests = [
        test_alpha_grid_and_endpoints,
        test_config_validation_accepts_zero,
        test_scheme_override_and_count_recalculation,
    ]
    passed = 0
    for test in tests:
        test()
        passed += 1
        print(f"PASS: {test.__name__}")
    print(f"Batch 09 focused tests: {passed}/{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
