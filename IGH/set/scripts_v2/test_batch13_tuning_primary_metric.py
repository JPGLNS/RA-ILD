#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused acceptance tests for IGH Batch 13 tuning-primary-metric support."""
from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.config import ConfigError, validate_experiment_mapping  # noqa: E402
import ra_ild_igh.nested_cv as nested_cv  # noqa: E402
from ra_ild_igh.nested_cv import (  # noqa: E402
    NestedCVOptions,
    PreparedInnerFold,
    tune_models,
)
from ra_ild_igh.repeated_holdout_summary import (  # noqa: E402
    _hyperparameter_tables,
)
from ra_ild_igh.specifications import (  # noqa: E402
    HyperparameterCandidate,
    candidate_selection_policy_name,
    rank_tuning_candidates,
    select_best_tuning_candidate,
    tuning_sort_policy,
)


def test_config_default_and_log_loss_contract() -> None:
    path = ROOT / "IGH/set/configs/igh_baseline_m2_v1.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_experiment_mapping(raw)

    logloss = copy.deepcopy(raw)
    logloss["model_selection"]["tuning_primary_metric"] = "log_loss"
    columns, ascending = tuning_sort_policy("log_loss")
    logloss["model_selection"]["candidate_sort"] = [
        {"field": field, "ascending": bool(direction)}
        for field, direction in zip(columns, ascending)
    ]
    logloss["nested_cv"]["candidate_selection_policy"] = (
        candidate_selection_policy_name("log_loss")
    )
    validate_experiment_mapping(logloss)

    invalid = copy.deepcopy(logloss)
    invalid["model_selection"]["candidate_sort"][0]["ascending"] = False
    try:
        validate_experiment_mapping(invalid)
    except ConfigError:
        pass
    else:
        raise AssertionError("Invalid log-loss sort direction was accepted")
    print("PASS test_config_default_and_log_loss_contract")


def test_specification_ranking_policies() -> None:
    frame = pd.DataFrame(
        [
            {
                "l1_ratio_alpha": 0.1,
                "lambda": 1.0,
                "pooled_inner_roc_auc": 0.90,
                "pooled_inner_pr_auc": 0.60,
                "pooled_inner_log_loss": 0.75,
                "pooled_inner_brier_score": 0.28,
            },
            {
                "l1_ratio_alpha": 0.5,
                "lambda": 2.0,
                "pooled_inner_roc_auc": 0.80,
                "pooled_inner_pr_auc": 0.70,
                "pooled_inner_log_loss": 0.45,
                "pooled_inner_brier_score": 0.18,
            },
        ]
    )
    roc = select_best_tuning_candidate(frame, primary_metric="roc_auc")
    loss = select_best_tuning_candidate(frame, primary_metric="log_loss")
    assert roc.lambda_value == 1.0
    assert loss.lambda_value == 2.0
    assert float(
        rank_tuning_candidates(frame, primary_metric="log_loss").iloc[0]["lambda"]
    ) == 2.0
    print("PASS test_specification_ranking_policies")


class _FakeFit:
    def __init__(self, lambda_value: float):
        self.lambda_value = float(lambda_value)
        self.converged = True
        self.n_iter = 3

    def predict_probability(self, matrix: np.ndarray) -> np.ndarray:
        fold_id = int(round(float(matrix[0, 0])))
        if np.isclose(self.lambda_value, 1.0):
            return np.asarray([0.49, 0.51], dtype=float)
        if fold_id == 1:
            return np.asarray([0.10, 0.90], dtype=float)
        return np.asarray([0.75, 0.70], dtype=float)


def _fake_fit_elastic_net(
    X_train,
    y_train,
    *,
    l1_ratio,
    lambda_value,
    class_weight,
    max_iter,
    tolerance,
    random_state,
):
    return _FakeFit(lambda_value)


def _prepared_folds():
    folds = {}
    for fold_id in (1, 2):
        folds[fold_id] = PreparedInnerFold(
            X_train=np.zeros((2, 1), dtype=float),
            X_valid=np.full((2, 1), fold_id, dtype=float),
            y_train=np.asarray([0, 1], dtype=int),
            y_valid=np.asarray([0, 1], dtype=int),
            train_sample_ids=(f"T{fold_id}A", f"T{fold_id}B"),
            valid_sample_ids=(f"V{fold_id}A", f"V{fold_id}B"),
            feature_names=("x",),
            preprocessing_audit=pd.DataFrame(),
        )
    return {"M": folds}


def test_tune_models_honors_configured_metric() -> None:
    original = nested_cv.fit_elastic_net
    nested_cv.fit_elastic_net = _fake_fit_elastic_net
    candidates = (
        HyperparameterCandidate(alpha=0.1, lambda_value=1.0),
        HyperparameterCandidate(alpha=0.1, lambda_value=2.0),
    )
    try:
        roc_result = tune_models(
            _prepared_folds(),
            candidates,
            NestedCVOptions(base_seed=1, tuning_primary_metric="roc_auc"),
        )
        loss_result = tune_models(
            _prepared_folds(),
            candidates,
            NestedCVOptions(base_seed=1, tuning_primary_metric="log_loss"),
        )
    finally:
        nested_cv.fit_elastic_net = original

    assert roc_result.selected["M"].candidate.lambda_value == 1.0
    assert loss_result.selected["M"].candidate.lambda_value == 2.0
    required = {
        "pooled_inner_log_loss",
        "pooled_inner_brier_score",
        "tuning_primary_metric",
        "candidate_selection_policy",
    }
    assert required.issubset(loss_result.tuning.columns)
    assert set(loss_result.tuning["tuning_primary_metric"]) == {"log_loss"}
    print("PASS test_tune_models_honors_configured_metric")


def test_runner_writes_tuning_audit() -> None:
    source = (
        ROOT / "IGH/set/scripts_v2/run_nested_cv_task.py"
    ).read_text(encoding="utf-8")
    assert 'selection_config = config.section("model_selection")' in source
    assert "tuning_primary_metric=str(" in source
    assert '"candidate_selection_policy": next(' in source
    assert '"inner_log_loss": value.inner_log_loss' in source
    print("PASS test_runner_writes_tuning_audit")


def _metric_rows(primary_metric: str, policy: str) -> pd.DataFrame:
    rows = []
    for repeat, alpha in ((1, 0.1), (2, 0.5)):
        rows.append(
            {
                "split_set_id": "S",
                "split_id": f"S{repeat}",
                "task_id": f"T{repeat}",
                "task_index": repeat,
                "outer_repeat": repeat,
                "outer_fold": 1,
                "model": "M",
                "selected_l1_ratio_alpha": alpha,
                "selected_lambda": 1.0,
                "threshold": 0.5,
                "inner_selected_roc_auc": 0.7,
                "inner_selected_pr_auc": 0.6,
                "inner_selected_log_loss": 0.5,
                "inner_selected_brier_score": 0.2,
                "tuning_primary_metric": primary_metric,
                "candidate_selection_policy": policy,
            }
        )
    return pd.DataFrame(rows)


def test_generic_summary_legacy_and_new_metadata() -> None:
    legacy = _metric_rows(
        "roc_auc", "pooled_roc_pr_lambda_alpha"
    ).drop(
        columns=[
            "inner_selected_log_loss",
            "inner_selected_brier_score",
            "tuning_primary_metric",
            "candidate_selection_policy",
        ]
    )
    selected, frequency, _ = _hyperparameter_tables(legacy)
    assert set(selected["tuning_primary_metric"]) == {"roc_auc"}
    assert set(frequency["candidate_selection_policy"]) == {
        "pooled_roc_pr_lambda_alpha"
    }

    modern = _metric_rows(
        "log_loss", "pooled_log_loss_brier_roc_lambda_alpha"
    )
    selected, frequency, _ = _hyperparameter_tables(modern)
    assert set(selected["tuning_primary_metric"]) == {"log_loss"}
    assert "mean_inner_log_loss" in frequency.columns
    print("PASS test_generic_summary_legacy_and_new_metadata")


def _load_dedicated_aggregator():
    path = ROOT / "IGH/set/scripts_v2/aggregate_repeated_holdout_results.py"
    spec = importlib.util.spec_from_file_location(
        "igh_batch13_dedicated_aggregator", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dedicated_aggregator_preserves_policy() -> None:
    module = _load_dedicated_aggregator()
    frame = _metric_rows(
        "log_loss", "pooled_log_loss_brier_roc_lambda_alpha"
    )
    selected, frequency, *_ = module.selected_hyperparameters(
        frame, ("M",)
    )
    assert set(selected["tuning_primary_metric"]) == {"log_loss"}
    assert set(frequency["candidate_selection_policy"]) == {
        "pooled_log_loss_brier_roc_lambda_alpha"
    }
    assert "mean_inner_brier_score" in frequency.columns
    print("PASS test_dedicated_aggregator_preserves_policy")


def test_feature_ablation_preparer_log_loss() -> None:
    source_path = (
        ROOT
        / "IGH/set/configs/schemes/igh_scheme_pbmc_a1_a6_repeat100_v1.yaml"
    )
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    temporary_source = (
        ROOT
        / "IGH/set/configs/schemes/_batch13_logloss_acceptance_tmp.yaml"
    )
    temporary_id = "batch13_logloss_acceptance_tmp"
    temporary_output = ROOT / "IGH/set/experiments" / temporary_id
    preparer = ROOT / "IGH/set/scripts_v2/prepare_feature_ablation_scheme.py"
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    source["scheme"]["id"] = temporary_id
    source["scheme"]["description"] = (
        "Temporary IGH Batch 13 log-loss acceptance scheme."
    )
    source["scheme"]["output_root"] = f"IGH/set/experiments/{temporary_id}"
    # Use the already completed full-cohort Scheme 006 resolved config.
    # The PBMC material-specific base experiment is optional and may not yet
    # have been prepared on a server that has only frozen its inputs.
    source["scheme"]["base_resolved_config"] = (
        "IGH/set/experiments/"
        "igh_scheme_006_a1_a6_repeat100_fullgrid/"
        "00_config/resolved_config.yaml"
    )
    source["model_selection"] = {"tuning_primary_metric": "log_loss"}

    try:
        if temporary_output.exists():
            shutil.rmtree(temporary_output)
        temporary_source.write_text(
            yaml.safe_dump(source, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(preparer),
                "--scheme",
                str(temporary_source.relative_to(ROOT)),
                "--repository-root",
                str(ROOT),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        if completed.returncode != 0:
            raise RuntimeError(
                "Feature-ablation preparer acceptance command failed "
                f"with exit code {completed.returncode}"
            )
        resolved = yaml.safe_load(
            (
                temporary_output / "00_config/resolved_config.yaml"
            ).read_text(encoding="utf-8")
        )
        marker = json.loads(
            (
                temporary_output
                / "00_config/FEATURE_ABLATION_SCHEME_PREPARED.json"
            ).read_text(encoding="utf-8")
        )
        assert resolved["model_selection"]["tuning_primary_metric"] == "log_loss"
        assert (
            resolved["nested_cv"]["candidate_selection_policy"]
            == "pooled_log_loss_brier_roc_lambda_alpha"
        )
        assert marker["tuning_primary_metric"] == "log_loss"
    finally:
        temporary_source.unlink(missing_ok=True)
        if temporary_output.exists():
            shutil.rmtree(temporary_output)
    print("PASS test_feature_ablation_preparer_log_loss")


def main() -> int:
    tests = [
        test_config_default_and_log_loss_contract,
        test_specification_ranking_policies,
        test_tune_models_honors_configured_metric,
        test_runner_writes_tuning_audit,
        test_generic_summary_legacy_and_new_metadata,
        test_dedicated_aggregator_preserves_policy,
        test_feature_ablation_preparer_log_loss,
    ]
    for test in tests:
        test()
    print(f"IGH Batch 13 focused tests: {len(tests)}/{len(tests)} PASS")
    print("IGH_BATCH13_ACCEPTANCE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
