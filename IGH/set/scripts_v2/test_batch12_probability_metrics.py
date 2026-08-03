#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused acceptance tests for IGH Batch 12 probability metrics."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ra_ild_igh.metrics import (
    BOOTSTRAP_METRICS,
    classification_metrics,
    stratified_bootstrap_ci,
)
from ra_ild_igh.repeated_holdout_summary import (
    _pairwise_model_differences,
    compare_aggregated_schemes,
)
from backfill_probability_metrics import (
    BACKUP_FILE,
    COMPLETE_FILE,
    METRIC_FILE,
    PREDICTION_FILE,
    _recompute_task,
)


def metric_row(model, split_id, logloss, brier, higher):
    return {
        "split_id": split_id,
        "model": model,
        "roc_auc": higher,
        "pr_auc": higher,
        "log_loss": logloss,
        "brier_score": brier,
        "accuracy": higher,
        "sensitivity_recall": higher,
        "specificity": higher,
        "precision": higher,
        "f1": higher,
    }


def test_metric_values_and_bootstrap():
    y = np.asarray([0, 0, 1, 1], dtype=int)
    p = np.asarray([0.1, 0.3, 0.8, 0.9], dtype=float)
    result = classification_metrics(
        y, p, 0.5, include_brier=True, include_log_loss=True
    )
    assert 0.0 < result["log_loss"] < 1.0
    assert 0.0 < result["brier_score"] < 0.25
    assert {"log_loss", "brier_score"}.issubset(BOOTSTRAP_METRICS)
    ci = stratified_bootstrap_ci(y, p, 0.5, reps=100, seed=20260803)
    assert {"log_loss", "brier_score"}.issubset(set(ci["metric"]))
    print("PASS test_metric_values_and_bootstrap")


def test_native_outer_metrics_enabled():
    source = (SRC_DIR / "ra_ild_igh/nested_cv.py").read_text(encoding="utf-8")
    assert "include_brier=True," in source
    assert "include_log_loss=True," in source
    print("PASS test_native_outer_metrics_enabled")


def test_model_pairwise_direction():
    rows = []
    for split_id in ("split_01", "split_02"):
        rows += [
            metric_row("A", split_id, 0.70, 0.25, 0.60),
            metric_row("B", split_id, 0.50, 0.18, 0.70),
        ]
    result = _pairwise_model_differences(pd.DataFrame(rows))
    for name in ("log_loss", "brier_score"):
        row = result.loc[result["metric"] == name].iloc[0]
        assert row["metric_direction"] == "lower_is_better"
        assert float(row["comparison_win_frequency"]) == 1.0
    print("PASS test_model_pairwise_direction")


def test_scheme_pairwise_direction():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        directories = []
        for scheme, logloss, brier, higher in (
            ("scheme_a", 0.70, 0.25, 0.60),
            ("scheme_b", 0.50, 0.18, 0.70),
        ):
            directory = root / scheme
            directory.mkdir()
            pd.DataFrame([
                metric_row("A", split_id, logloss, brier, higher)
                for split_id in ("split_01", "split_02")
            ]).to_csv(directory / "08_holdout_task_metrics.csv", index=False)
            marker = {
                "status": "COMPLETE",
                "experiment_id": scheme,
                "assignment_sha256": "same_hash",
                "split_set_id": "same_split_set",
                "models": ["A"],
                "completed_split_count": 2,
                "metric_rows": 2,
                "prediction_rows": 4,
            }
            (directory / "08_REPEATED_HOLDOUT_AGGREGATION_COMPLETE.json").write_text(
                json.dumps(marker), encoding="utf-8"
            )
            directories.append(directory)
        _, pairwise, _ = compare_aggregated_schemes(directories)
        row = pairwise.loc[pairwise["metric"] == "log_loss"].iloc[0]
        assert row["metric_direction"] == "lower_is_better"
        assert float(row["comparison_win_frequency"]) == 1.0
    print("PASS test_scheme_pairwise_direction")


def load_dedicated_aggregator():
    path = SCRIPT_DIR / "aggregate_repeated_holdout_results.py"
    spec = importlib.util.spec_from_file_location("igh_batch12_aggregator", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dedicated_aggregator_direction():
    module = load_dedicated_aggregator()
    rows = []
    for repeat in (1, 2):
        a = metric_row("A", f"split_{repeat:02d}", 0.70, 0.25, 0.60)
        b = metric_row("B", f"split_{repeat:02d}", 0.50, 0.18, 0.70)
        a["outer_repeat"] = repeat
        b["outer_repeat"] = repeat
        rows += [a, b]
    metrics = pd.DataFrame(rows)
    _, summary = module.paired_differences(
        metrics, baseline_model="A", model_order=("A", "B")
    )
    row = summary.loc[summary["metric"] == "log_loss"].iloc[0]
    assert row["metric_direction"] == "lower_is_better"
    assert float(row["comparison_win_frequency"]) == 1.0
    detail, _ = module.model_ranks(metrics, ("A", "B"))
    best = detail.loc[
        (detail["metric"] == "log_loss") & (detail["model"] == "B"), "rank"
    ]
    assert np.allclose(best.to_numpy(float), 1.0)
    print("PASS test_dedicated_aggregator_direction")


def test_backfill_is_reproducible_and_idempotent():
    with tempfile.TemporaryDirectory() as td:
        task_dir = Path(td) / "repeat_01_fold_01"
        task_dir.mkdir()
        y = np.asarray([0, 0, 1, 1], dtype=int)
        p = np.asarray([0.1, 0.3, 0.8, 0.9], dtype=float)
        predictions = pd.DataFrame({
            "model": ["A"] * 4,
            "sample_id": ["S1", "S2", "S3", "S4"],
            "true_label": y,
            "probability_ILD": p,
            "threshold": [0.5] * 4,
        })
        metrics = classification_metrics(y, p, 0.5)
        pd.DataFrame([{"model": "A", **metrics}]).to_csv(
            task_dir / METRIC_FILE, index=False
        )
        predictions.to_csv(task_dir / PREDICTION_FILE, index=False)
        (task_dir / COMPLETE_FILE).write_text(
            json.dumps({"status": "COMPLETE"}), encoding="utf-8"
        )
        task = SimpleNamespace(
            task_index=1, task_id="repeat_01_fold_01",
            outer_repeat=1, outer_fold=1, output_dir=task_dir
        )
        dry = _recompute_task(task, tolerance=1e-10, dry_run=True)
        assert dry["action"] == "would_update"
        assert not (task_dir / BACKUP_FILE).exists()
        written = _recompute_task(task, tolerance=1e-10, dry_run=False)
        assert written["action"] == "updated"
        assert (task_dir / BACKUP_FILE).is_file()
        current = pd.read_csv(task_dir / METRIC_FILE)
        assert {"log_loss", "brier_score"}.issubset(current.columns)
        again = _recompute_task(task, tolerance=1e-10, dry_run=False)
        assert again["action"] == "already_current"
    print("PASS test_backfill_is_reproducible_and_idempotent")


def main():
    tests = [
        test_metric_values_and_bootstrap,
        test_native_outer_metrics_enabled,
        test_model_pairwise_direction,
        test_scheme_pairwise_direction,
        test_dedicated_aggregator_direction,
        test_backfill_is_reproducible_and_idempotent,
    ]
    for test in tests:
        test()
    print(f"IGH Batch 12 focused tests: {len(tests)}/{len(tests)} PASS")
    print("IGH_BATCH12_ACCEPTANCE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
