#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finalize Batch 18 installation after the real-data smoke workflow passes."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.xgboost_config import load_xgboost_config  # noqa: E402

ACCEPTANCE = "TRB_BATCH18_XGBOOST_ACCEPTANCE_PASS"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Batch 18 real-data smoke outputs and finalize the installation marker.")
    parser.add_argument("--repository-root", required=True)
    parser.add_argument(
        "--config",
        default="TRB/set/experiments/trb_xgboost_smoke_m0_v1/00_config/resolved_config.yaml",
    )
    args = parser.parse_args()
    root = Path(args.repository_root).expanduser().resolve()
    try:
        config_path = Path(args.config).expanduser()
        if not config_path.is_absolute():
            config_path = root / config_path
        config = load_xgboost_config(config_path, repository_root=root)
        if config.experiment_id != "trb_xgboost_smoke_m0_v1":
            raise ValueError(f"Acceptance requires smoke config trb_xgboost_smoke_m0_v1; observed={config.experiment_id}")
        if len(config.section("models")) != 1 or "M0_clinical" not in config.section("models"):
            raise ValueError("Smoke config must contain only M0_clinical")
        if int(config.section("model_engine")["candidate_bank"]["candidate_count"]) != 2:
            raise ValueError("Smoke candidate bank must contain exactly 2 candidates")
        task = config.section("nested_cv")["regression_task"]
        repeat = int(task["outer_repeat"])
        fold = int(task["outer_fold"])
        task_dir = config.path("outer_tasks.output_root") / f"repeat_{repeat:02d}_fold_{fold:02d}"
        marker_path = task_dir / "06_TASK_COMPLETE.json"
        if not marker_path.is_file():
            raise FileNotFoundError(f"Smoke task completion marker missing: {marker_path}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("status") != "COMPLETE" or marker.get("engine") != "xgboost":
            raise ValueError("Smoke task completion marker is invalid")
        tuning = pd.read_csv(task_dir / "06_inner_tuning_results.csv")
        metrics = pd.read_csv(task_dir / "06_outer_validation_metrics.csv")
        predictions = pd.read_csv(task_dir / "06_outer_validation_predictions.csv")
        importance = pd.read_csv(task_dir / "06_final_model_feature_importance.csv")
        if len(tuning) != 2 or set(tuning["model"].astype(str)) != {"M0_clinical"}:
            raise ValueError("Smoke tuning must contain exactly 2 M0 candidate rows")
        if len(metrics) != 1 or str(metrics.iloc[0]["model"]) != "M0_clinical":
            raise ValueError("Smoke metrics must contain one M0 row")
        required_metrics = ["roc_auc", "pr_auc", "log_loss", "brier_score", "accuracy", "sensitivity_recall", "specificity", "precision", "f1"]
        values = pd.to_numeric(metrics.iloc[0][required_metrics], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("Smoke metrics contain non-finite values")
        probability = pd.to_numeric(predictions["probability_ILD"], errors="raise").to_numpy(float)
        if len(probability) < 2 or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError("Smoke predictions violate the probability contract")
        if importance.empty or "importance_gain" not in importance.columns:
            raise ValueError("Smoke feature-importance output is missing")

        summary_dir = task_dir.parent.parent / "03_summary"
        aggregate_marker = summary_dir / "08_REPEATED_HOLDOUT_AGGREGATION_COMPLETE.json"
        if not aggregate_marker.is_file():
            raise FileNotFoundError(f"Smoke aggregation marker missing: {aggregate_marker}")
        aggregate = json.loads(aggregate_marker.read_text(encoding="utf-8"))
        if aggregate.get("status") != "COMPLETE" or aggregate.get("model_engine") != "xgboost":
            raise ValueError("Smoke aggregation completion marker is invalid")
        aggregated_metrics = pd.read_csv(summary_dir / "08_holdout_task_metrics.csv")
        aggregated_predictions = pd.read_csv(summary_dir / "08_holdout_predictions.csv.gz")
        if len(aggregated_metrics) != 1 or len(aggregated_predictions) != len(predictions):
            raise ValueError("Smoke aggregation row counts do not match task outputs")

        install_path = root / "TRB/set/scripts_v2/BATCH18_XGBOOST_INSTALLATION.json"
        if not install_path.is_file():
            raise FileNotFoundError("Batch 18 installation marker missing")
        install = json.loads(install_path.read_text(encoding="utf-8"))
        install.update({
            "status": "INSTALLED",
            "accepted_at_utc": datetime.now(timezone.utc).isoformat(),
            "acceptance": ACCEPTANCE,
            "smoke_experiment": config.experiment_id,
            "smoke_task": f"{repeat}:{fold}",
            "smoke_tuning_candidates": 2,
            "smoke_probability_metrics": True,
            "smoke_feature_importance": True,
            "smoke_aggregation": True,
        })
        install_path.write_text(json.dumps(install, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("TRB Batch 18 XGBoost real-data smoke acceptance: PASS")
        print(f"Smoke config: {config_path}")
        print(f"Smoke task:   repeat={repeat}, fold={fold}")
        print(f"Metrics:      {required_metrics}")
        print(f"Predictions:  {len(predictions)}")
        print(f"Importance:   {len(importance)} rows")
        print(f"Aggregation:  {summary_dir}")
        print(ACCEPTANCE)
        return 0
    except Exception as exc:
        print(f"TRB Batch 18 XGBoost acceptance: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
