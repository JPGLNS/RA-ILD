#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from pathlib import Path

import pandas as pd

from ra_ild_igh.xgboost_summary import discover_completed_tasks, write_aggregation


def _write_task(root: Path, repeat: int, fold: int, auc: float):
    task = root / f"repeat_{repeat:02d}_fold_{fold:02d}"
    task.mkdir(parents=True)
    split_id = f"repeat_{repeat:02d}_fold_{fold:02d}"
    metrics = pd.DataFrame([{
        "model": "M0", "split_id": split_id, "outer_repeat": repeat, "outer_fold": fold,
        "roc_auc": auc, "pr_auc": auc, "log_loss": 0.6, "brier_score": 0.2,
        "accuracy": 0.6, "sensitivity_recall": 0.5, "specificity": 0.7,
        "precision": 0.6, "f1": 0.55, "threshold": 0.5,
        "selected_candidate_id": "xgb_001", "selected_parameters_json": '{"max_depth":2}',
        "selected_xgb_max_depth": 2, "probability_metrics_available": True,
    }])
    metrics.to_csv(task / "06_outer_validation_metrics.csv", index=False)
    predictions = pd.DataFrame([
        {"model":"M0","split_id":split_id,"outer_repeat":repeat,"outer_fold":fold,"sample_id":"A","true_label":0,"true_cohort":"RA","probability_ILD":0.2,"threshold":0.5,"predicted_label":0},
        {"model":"M0","split_id":split_id,"outer_repeat":repeat,"outer_fold":fold,"sample_id":"B","true_label":1,"true_cohort":"ILD","probability_ILD":0.8,"threshold":0.5,"predicted_label":1},
    ])
    predictions.to_csv(task / "06_outer_validation_predictions.csv", index=False)
    importance = pd.DataFrame([
        {"model":"M0","split_id":split_id,"outer_repeat":repeat,"outer_fold":fold,"feature_name":"age","importance_gain":1.0,"importance_weight":2.0,"importance_cover":3.0,"importance_total_gain":4.0,"importance_total_cover":5.0},
        {"model":"M0","split_id":split_id,"outer_repeat":repeat,"outer_fold":fold,"feature_name":"sex_M","importance_gain":0.0,"importance_weight":0.0,"importance_cover":0.0,"importance_total_gain":0.0,"importance_total_cover":0.0},
    ])
    importance.to_csv(task / "06_final_model_feature_importance.csv", index=False)
    (task / "06_task_configuration.json").write_text("{}\n", encoding="utf-8")
    (task / "06_TASK_COMPLETE.json").write_text(json.dumps({"status":"COMPLETE","engine":"xgboost","outer_repeat":repeat,"outer_fold":fold})+"\n", encoding="utf-8")


def test_xgboost_summary_writes_probability_and_importance_outputs(tmp_path: Path):
    task_root = tmp_path / "tasks"
    _write_task(task_root, 1, 1, 0.70)
    _write_task(task_root, 2, 1, 0.80)
    tasks = discover_completed_tasks(task_root)
    assert len(tasks) == 2
    out = tmp_path / "summary"
    paths = write_aggregation(task_root, out)
    assert paths["metrics"].is_file()
    assert paths["predictions"].is_file()
    assert paths["importance_summary"].is_file()
    metric_summary = pd.read_csv(paths["metric_summary"])
    roc = metric_summary.query("model == 'M0' and metric == 'roc_auc'").iloc[0]
    assert abs(float(roc["mean"]) - 0.75) < 1e-12
    importance = pd.read_csv(paths["importance_summary"])
    age = importance.query("feature_name == 'age'").iloc[0]
    assert float(age["retained_frequency"]) == 1.0
    assert float(age["gain_nonzero_frequency_all_tasks"]) == 1.0
