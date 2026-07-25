#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate three frozen RA-ILD repeated-holdout tasks without OOF assumptions."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.config import load_experiment_config
from ra_ild_trb.outer_cv import build_outer_tasks, manifest_frame, scan_outer_tasks
from ra_ild_trb.repeated_holdout_summary import (
    aggregate_repeated_holdout_results,
    load_frozen_assignments,
    write_repeated_holdout_aggregate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate frozen repeated-holdout model results.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_experiment_config(
            Path(args.config),
            repository_root=Path(args.repository_root) if args.repository_root else None,
        )
        repeated = config.section("repeated_holdout_training")
        if str(repeated.get("mode")) != "frozen_repeated_holdout":
            raise ValueError("This script requires repeated_holdout_training.mode=frozen_repeated_holdout")
        cv = config.section("cross_validation")
        engine = config.section("model_engine")
        models = tuple(config.section("models"))
        selection = config.section("model_selection")
        stability = config.section("stability")
        output_root = config.path("outer_tasks.output_root")
        configured = build_outer_tasks(int(cv["outer_repeats"]), int(cv["outer_folds"]), output_root)
        executed_folds = tuple(int(x) for x in cv.get("executed_outer_folds", (1,)))
        tasks = tuple(task for task in configured if task.outer_fold in executed_folds)
        if len(tasks) != int(repeated["split_count"]):
            raise ValueError("Executed tasks do not match repeated_holdout_training.split_count")
        manifest = manifest_frame(
            tasks,
            python_executable=sys.executable,
            runner_script=config.path("outer_tasks.runner_script", must_exist=True, expect="file"),
            config_path=config.source_path,
        )
        status = scan_outer_tasks(
            tasks,
            expected_models=models,
            expected_samples=int(config.raw["data"]["train"]["expected_samples"]),
            expected_candidates_per_model=len(engine["alpha_grid"]) * len(engine["lambda_grid"]),
            expected_static_features=int(selection["expected_static_feature_count"]),
        )
        assignments, _, assignment_sha, marker_sha = load_frozen_assignments(
            config.path("repeated_holdout_training.assignments", must_exist=True, expect="file"),
            config.path("repeated_holdout_training.frozen_marker", must_exist=True, expect="file"),
            expected_split_set_id=str(repeated["split_set_id"]),
            expected_splits=int(repeated["split_count"]),
            expected_train_size=int(repeated["train_size"]),
            expected_holdout_size=int(repeated["holdout_size"]),
        )
        aggregate = aggregate_repeated_holdout_results(
            tasks,
            status,
            assignments,
            expected_models=models,
            expected_split_set_id=str(repeated["split_set_id"]),
            expected_split_count=int(repeated["split_count"]),
            expected_holdout_size=int(repeated["holdout_size"]),
            coefficient_tolerance=float(stability["coefficient_tolerance"]),
            minimum_selection_frequency=float(stability["minimum_selection_frequency"]),
            minimum_sign_consistency=float(stability["minimum_sign_consistency"]),
            manifest=manifest,
        )
        output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else config.path("aggregation.output_dir")
        paths = write_repeated_holdout_aggregate(
            aggregate,
            output_dir,
            experiment_id=config.experiment_id,
            split_set_id=str(repeated["split_set_id"]),
            assignment_sha256=assignment_sha,
            split_marker_sha256=marker_sha,
            expected_split_count=int(repeated["split_count"]),
            expected_holdout_size=int(repeated["holdout_size"]),
            overwrite=args.overwrite,
        )
        print("TRB repeated-holdout aggregation: COMPLETE")
        print(f"Completed splits: {int(status['status'].eq('complete').sum())}/{len(tasks)}")
        print(f"Metric rows:      {len(aggregate.metrics)}")
        print(f"Prediction rows:  {len(aggregate.predictions)}")
        print(f"Stable features:  {len(aggregate.stable_features)}")
        print(f"Output:           {output_dir}")
        print(f"Completion marker:{paths['complete']}")
        return 0
    except Exception as exc:
        print(f"TRB repeated-holdout aggregation: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
