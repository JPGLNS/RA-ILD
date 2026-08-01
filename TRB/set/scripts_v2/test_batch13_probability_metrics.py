#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Acceptance checks for TRB V2 Batch 13 probability metrics."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

SCRIPT_DIR = Path(__file__).resolve().parent
SET_DIR = SCRIPT_DIR.parent
SRC_DIR = SET_DIR / "src"
for value in (str(SRC_DIR), str(SCRIPT_DIR)):
    if value not in sys.path:
        sys.path.insert(0, value)

from backfill_probability_metrics import _recompute_task  # noqa: E402
from ra_ild_trb.metrics import (  # noqa: E402
    BOOTSTRAP_METRICS,
    classification_metrics,
    stratified_bootstrap_ci,
)
from ra_ild_trb.repeated_holdout_summary import (  # noqa: E402
    LOWER_IS_BETTER_METRICS,
    METRIC_NAMES,
    _metric_summary,
    _model_ranking,
    _pairwise_model_differences,
)


def check_metrics() -> None:
    y = np.array([0, 0, 1, 1], dtype=int)
    p = np.array([0.10, 0.35, 0.70, 0.90], dtype=float)
    result = classification_metrics(
        y,
        p,
        0.5,
        include_brier=True,
        include_log_loss=True,
    )
    assert np.isclose(result["log_loss"], log_loss(y, p, labels=[0, 1]))
    assert np.isclose(result["brier_score"], brier_score_loss(y, p))
    assert "log_loss" in BOOTSTRAP_METRICS
    assert "brier_score" in BOOTSTRAP_METRICS
    bootstrap = stratified_bootstrap_ci(y, p, 0.5, reps=100, seed=13)
    assert set(bootstrap["metric"]) == set(BOOTSTRAP_METRICS)
    assert bootstrap["estimate"].map(np.isfinite).all()


def check_summary_direction() -> None:
    assert LOWER_IS_BETTER_METRICS == {"log_loss", "brier_score"}
    rows = []
    for split_id in ("s1", "s2"):
        rows.append(
            {
                "split_id": split_id,
                "model": "A",
                "roc_auc": 0.60,
                "pr_auc": 0.55,
                "log_loss": 0.70,
                "brier_score": 0.25,
                "accuracy": 0.60,
                "sensitivity_recall": 0.60,
                "specificity": 0.60,
                "precision": 0.55,
                "f1": 0.57,
            }
        )
        rows.append(
            {
                "split_id": split_id,
                "model": "B",
                "roc_auc": 0.70,
                "pr_auc": 0.65,
                "log_loss": 0.55,
                "brier_score": 0.20,
                "accuracy": 0.65,
                "sensitivity_recall": 0.65,
                "specificity": 0.65,
                "precision": 0.60,
                "f1": 0.62,
            }
        )
    frame = pd.DataFrame(rows)
    summary = _metric_summary(frame)
    ranking = _model_ranking(summary).set_index("model")
    assert int(ranking.loc["B", "rank_mean_log_loss"]) == 1
    assert int(ranking.loc["B", "rank_mean_brier_score"]) == 1

    pairwise = _pairwise_model_differences(frame)
    loss_row = pairwise.loc[pairwise["metric"] == "log_loss"].iloc[0]
    assert loss_row["metric_direction"] == "lower_is_better"
    assert float(loss_row["mean_difference_comparison_minus_reference"]) < 0
    assert np.isclose(float(loss_row["comparison_win_frequency"]), 1.0)
    auc_row = pairwise.loc[pairwise["metric"] == "roc_auc"].iloc[0]
    assert auc_row["metric_direction"] == "higher_is_better"
    assert float(auc_row["mean_difference_comparison_minus_reference"]) > 0
    assert np.isclose(float(auc_row["comparison_win_frequency"]), 1.0)
    assert set(METRIC_NAMES) >= {"log_loss", "brier_score"}


def check_backfill() -> None:
    y = np.array([0, 0, 1, 1], dtype=int)
    p = np.array([0.12, 0.30, 0.72, 0.88], dtype=float)
    old = classification_metrics(y, p, 0.5)
    row = {"model": "M", **old}

    with tempfile.TemporaryDirectory() as temporary:
        task_dir = Path(temporary) / "repeat_01_fold_01"
        task_dir.mkdir()
        (task_dir / "06_TASK_COMPLETE.json").write_text(
            json.dumps({"status": "COMPLETE"}) + "\n",
            encoding="utf-8",
        )
        pd.DataFrame(
            {
                "model": ["M"] * len(y),
                "sample_id": [f"S{i}" for i in range(len(y))],
                "true_label": y,
                "probability_ILD": p,
                "threshold": [0.5] * len(y),
            }
        ).to_csv(task_dir / "06_outer_validation_predictions.csv", index=False)
        pd.DataFrame([row]).to_csv(
            task_dir / "06_outer_validation_metrics.csv", index=False
        )
        task = SimpleNamespace(
            task_index=1,
            task_id="repeat_01_fold_01",
            outer_repeat=1,
            outer_fold=1,
            output_dir=task_dir,
        )
        dry = _recompute_task(task, tolerance=1.0e-8, dry_run=True)
        assert dry["action"] == "would_update"
        written = _recompute_task(task, tolerance=1.0e-8, dry_run=False)
        assert written["action"] == "updated"
        result = pd.read_csv(task_dir / "06_outer_validation_metrics.csv")
        assert "log_loss" in result.columns
        assert "brier_score" in result.columns
        assert (task_dir / "06_outer_validation_metrics.before_batch13.csv").is_file()
        current = _recompute_task(task, tolerance=1.0e-8, dry_run=False)
        assert current["action"] == "already_current"


def main() -> int:
    checks = [check_metrics, check_summary_direction, check_backfill]
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")
    print("TRB_BATCH13_ACCEPTANCE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
