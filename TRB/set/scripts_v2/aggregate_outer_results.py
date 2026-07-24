#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate completed TRB V2 outer tasks and calculate stability summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_trb.config import load_experiment_config  # noqa: E402
from ra_ild_trb.outer_cv import (  # noqa: E402
    aggregate_outer_results,
    build_outer_tasks,
    manifest_frame,
    scan_outer_tasks,
    write_outer_aggregate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and summarize TRB V2 repeated nested-CV outer tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_experiment_config(
            Path(args.config),
            repository_root=Path(args.repository_root) if args.repository_root else None,
        )
        cv = config.section("cross_validation")
        engine = config.section("model_engine")
        models = tuple(config.section("models"))
        selection = config.section("model_selection")
        outer = config.section("outer_tasks")
        aggregation = config.section("aggregation")
        stability = config.section("stability")

        output_root = config.path("outer_tasks.output_root")
        tasks = build_outer_tasks(
            int(cv["outer_repeats"]), int(cv["outer_folds"]), output_root
        )
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
        require_all = bool(aggregation["require_all_tasks"]) and not args.allow_incomplete
        aggregate = aggregate_outer_results(
            tasks,
            status,
            expected_models=models,
            expected_samples=int(config.raw["data"]["train"]["expected_samples"]),
            expected_repeats=int(cv["outer_repeats"]),
            require_all_tasks=require_all,
            coefficient_tolerance=float(stability["coefficient_tolerance"]),
            minimum_selection_frequency=float(stability["minimum_selection_frequency"]),
            minimum_sign_consistency=float(stability["minimum_sign_consistency"]),
            manifest=manifest,
        )
        if args.output_dir:
            output_dir = Path(args.output_dir).expanduser().resolve()
        else:
            configured_output = config.path("aggregation.output_dir")
            output_dir = (
                configured_output
                if require_all
                else configured_output.parent / f"{configured_output.name}_partial"
            )
        paths = write_outer_aggregate(
            aggregate,
            output_dir,
            experiment_id=config.experiment_id,
            expected_tasks=len(tasks),
            require_all_tasks=require_all,
            overwrite=args.overwrite,
        )
        print(
            "TRB V2 outer aggregation: "
            + ("COMPLETE" if require_all else "PARTIAL")
        )
        print(f"Completed tasks: {(status['status'] == 'complete').sum()}/{len(tasks)}")
        print(f"Metric rows: {len(aggregate.all_metrics)}")
        print(f"Prediction rows: {len(aggregate.all_predictions)}")
        print(f"Stable feature rows: {len(aggregate.stable_features)}")
        print(f"Output: {output_dir}")
        print(f"Completion marker: {paths['complete']}")
        return 0
    except Exception as exc:
        print(f"TRB V2 outer aggregation: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
