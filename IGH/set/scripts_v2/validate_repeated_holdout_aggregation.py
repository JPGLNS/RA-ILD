#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate IGH repeated-holdout aggregate outputs and their frozen contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.config import load_experiment_config
from ra_ild_igh.repeated_holdout_summary import METRIC_NAMES, OUTPUT_FILE_NAMES, sha256_file


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_experiment_config(
        Path(args.config),
        repository_root=Path(args.repository_root) if args.repository_root else None,
    )
    repeated = config.section("repeated_holdout_training")
    models = tuple(map(str, config.section("models")))
    split_count = int(repeated["split_count"])
    holdout_size = int(repeated["holdout_size"])
    expected_samples = int(config.raw["data"]["train"]["expected_samples"])
    expected_metric_rows = split_count * len(models)
    expected_prediction_rows = split_count * holdout_size * len(models)
    expected_summary_rows = len(models) * len(METRIC_NAMES)
    expected_pair_rows = (len(models) * (len(models) - 1) // 2) * len(METRIC_NAMES)
    expected_sample_rows = expected_samples * len(models)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else config.path("aggregation.output_dir")
    )
    paths = {key: output_dir / name for key, name in OUTPUT_FILE_NAMES.items()}
    for key, path in paths.items():
        require(path.is_file(), f"Missing aggregation output {key}: {path}")

    marker = json.loads(paths["complete"].read_text(encoding="utf-8"))
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    require(marker.get("status") == "COMPLETE", "Aggregation marker is not COMPLETE")
    require(marker.get("experiment_id") == config.experiment_id, "Experiment ID mismatch")
    require(marker.get("analysis_mode") == "frozen_repeated_holdout", "Analysis mode mismatch")
    require(marker.get("split_set_id") == str(repeated["split_set_id"]), "Split-set ID mismatch")
    assignments = config.path("repeated_holdout_training.assignments", must_exist=True, expect="file")
    require(marker.get("assignment_sha256") == sha256_file(assignments), "Assignment SHA256 mismatch")
    require(int(marker.get("completed_split_count", -1)) == split_count, "Completed split count mismatch")
    require(int(marker.get("holdout_size_per_split", -1)) == holdout_size, "Holdout size mismatch")
    require(tuple(map(str, marker.get("models", []))) == models, "Model order mismatch")
    require(int(marker.get("metric_rows", -1)) == expected_metric_rows, "Marker metric count mismatch")
    require(int(marker.get("prediction_rows", -1)) == expected_prediction_rows, "Marker prediction count mismatch")
    require(marker.get("ranking_is_descriptive_only") is True, "Ranking must be descriptive only")
    require(marker.get("automatic_final_model_selection") is False, "Automatic winner selection must be false")
    require(marker.get("independent_test_read") is False, "Independent test must not be read")
    require(summary == {k: marker[k] for k in summary}, "Summary and completion marker disagree")

    for key, item in marker.get("files", {}).items():
        path = Path(item["path"])
        require(path.is_file(), f"Hashed output is missing: {path}")
        require(sha256_file(path) == item["sha256"], f"Hashed output was modified: {path}")
        require(int(path.stat().st_size) == int(item["size_bytes"]), f"Size mismatch: {path}")

    metrics = pd.read_csv(paths["metrics"])
    predictions = pd.read_csv(paths["predictions"])
    metric_summary = pd.read_csv(paths["metric_summary"])
    ranking = pd.read_csv(paths["model_ranking"])
    pairwise = pd.read_csv(paths["pairwise_model_differences"])
    sample_summary = pd.read_csv(paths["sample_prediction_summary"])
    membership = pd.read_csv(paths["membership_audit"])
    hyperparameters = pd.read_csv(paths["hyperparameters"])

    require(len(metrics) == expected_metric_rows, "Metric row count mismatch")
    require(len(predictions) == expected_prediction_rows, "Prediction row count mismatch")
    require(len(metric_summary) == expected_summary_rows, "Metric summary row count mismatch")
    require(len(ranking) == len(models), "Model ranking row count mismatch")
    require(len(pairwise) == expected_pair_rows, "Pairwise model-difference row count mismatch")
    require(len(sample_summary) == expected_sample_rows, "Sample prediction summary row count mismatch")
    require(len(membership) == expected_metric_rows, "Membership audit row count mismatch")
    require(len(hyperparameters) == expected_metric_rows, "Selected hyperparameter row count mismatch")
    require(set(metrics["model"].astype(str)) == set(models), "Unexpected metric models")
    require(set(metric_summary["metric"].astype(str)) == set(METRIC_NAMES), "Metric summary names mismatch")
    require(membership["membership_exact_match"].astype(str).str.lower().eq("true").all(), "Holdout membership audit failed")
    require(ranking["automatic_final_model_selection"].astype(str).str.lower().eq("false").all(), "Ranking selected a winner")

    for model in models:
        require(int((metrics["model"].astype(str) == model).sum()) == split_count, f"Metric splits mismatch for {model}")
        require(int((predictions["model"].astype(str) == model).sum()) == split_count * holdout_size, f"Prediction rows mismatch for {model}")
        require(int((sample_summary["model"].astype(str) == model).sum()) == expected_samples, f"Coverage rows mismatch for {model}")

    print("RA-ILD IGH repeated-holdout aggregation validation: PASS")
    print(f"Experiment:       {config.experiment_id}")
    print(f"Completed splits: {split_count}/{split_count}")
    print(f"Models:           {len(models)}")
    print(f"Metric rows:      {len(metrics)}")
    print(f"Prediction rows:  {len(predictions)}")
    print(f"Metric summaries: {len(metric_summary)}")
    print(f"Pairwise rows:    {len(pairwise)}")
    print(f"Coverage rows:    {len(sample_summary)}")
    print(f"Stable features:  {int(marker.get('stable_feature_rows', 0))}")
    print(f"Output:           {output_dir}")
    print("IGH_REPEATED_HOLDOUT_AGGREGATION_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"RA-ILD IGH repeated-holdout aggregation validation: FAIL\n{exc}", file=sys.stderr)
        raise SystemExit(1)
