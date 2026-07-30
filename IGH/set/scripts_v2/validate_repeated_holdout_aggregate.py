#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the formal IGH repeated-holdout aggregate outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, MutableMapping

import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required.") from exc


class ValidationError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate IGH repeated-holdout aggregation outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--baseline-model", default="A1_core83")
    return parser.parse_args()


def load_yaml(path: Path) -> MutableMapping[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, MutableMapping):
        raise ValidationError(f"YAML root must be a mapping: {path}")
    return value


def require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must be a mapping.")
    return value


def resolve_path(value: object, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValidationError(f"JSON root must be an object: {path}")
    return value


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.lower().isin({"true", "1"})


def main() -> int:
    args = parse_args()
    try:
        root = (
            Path(args.repository_root).expanduser().resolve()
            if args.repository_root
            else Path.cwd().resolve()
        )
        config_path = Path(args.config).expanduser()
        if not config_path.is_absolute():
            config_path = (root / config_path).resolve()
        config = load_yaml(config_path)
        cv = require_mapping(config["cross_validation"], "cross_validation")
        outer = require_mapping(config["outer_tasks"], "outer_tasks")
        aggregation = require_mapping(config["aggregation"], "aggregation")
        models = tuple(require_mapping(config["models"], "models").keys())
        repeated = require_mapping(
            config["repeated_holdout_training"], "repeated_holdout_training"
        )
        if args.output_dir:
            output_dir = Path(args.output_dir).expanduser().resolve()
        else:
            output_dir = resolve_path(aggregation["output_dir"], root)

        repeats = int(cv["outer_repeats"])
        tasks = int(outer["expected_tasks"])
        train_size = int(repeated["train_size"])
        holdout_size = int(repeated["holdout_size"])
        model_count = len(models)
        candidates = len(config["model_engine"]["alpha_grid"]) * len(
            config["model_engine"]["lambda_grid"]
        )
        patients = int(config["data"]["train"]["expected_samples"])

        marker = read_json(
            output_dir / "07_REPEATED_HOLDOUT_AGGREGATION_COMPLETE.json"
        )
        summary = read_json(output_dir / "07_aggregation_summary.json")
        if marker.get("status") != "COMPLETE":
            raise ValidationError("Aggregation completion marker is not COMPLETE.")
        if marker.get("aggregation_mode") != "repeated_holdout":
            raise ValidationError("Aggregation mode mismatch.")
        if int(marker["completed_outer_tasks"]) != tasks:
            raise ValidationError("Completed task count mismatch.")
        if bool(marker.get("final_model_selected", True)):
            raise ValidationError("Aggregator must not select a final model.")

        metrics = pd.read_csv(output_dir / "07_all_outer_metrics.csv")
        predictions = pd.read_csv(
            output_dir / "07_all_outer_predictions.csv.gz"
        )
        tuning = pd.read_csv(
            output_dir / "07_all_inner_tuning_results.csv.gz"
        )
        oof = pd.read_csv(
            output_dir / "07_all_inner_selected_oof_predictions.csv.gz"
        )
        selected = pd.read_csv(
            output_dir / "07_all_selected_hyperparameters.csv"
        )
        baseline_detail = pd.read_csv(
            output_dir / "07_baseline_paired_differences.csv.gz"
        )
        baseline_summary = pd.read_csv(
            output_dir / "07_baseline_paired_difference_summary.csv"
        )
        pairwise = pd.read_csv(
            output_dir / "07_pairwise_model_metric_differences.csv"
        )
        rank_detail = pd.read_csv(
            output_dir / "07_model_rank_by_repeat.csv.gz"
        )
        metric_summary = pd.read_csv(
            output_dir / "07_model_metric_summary_long.csv"
        )
        alpha_frequency = pd.read_csv(
            output_dir / "07_alpha_selection_frequency.csv"
        )
        family_frequency = pd.read_csv(
            output_dir / "07_regularization_family_frequency.csv"
        )
        inner_convergence = pd.read_csv(
            output_dir / "07_inner_tuning_convergence_summary.csv"
        )
        outer_convergence = pd.read_csv(
            output_dir / "07_outer_refit_convergence_summary.csv"
        )
        sample_counts = pd.read_csv(
            output_dir / "07_sample_holdout_count_audit.csv"
        )
        sample_stability = pd.read_csv(
            output_dir / "07_sample_holdout_prediction_stability.csv"
        )

        expected_counts = {
            "metrics": tasks * model_count,
            "predictions": tasks * holdout_size * model_count,
            "tuning": tasks * model_count * candidates,
            "oof": tasks * train_size * model_count,
            "selected": tasks * model_count,
            "baseline_detail": (model_count - 1) * repeats * 5,
            "baseline_summary": (model_count - 1) * 5,
            "pairwise": (model_count * (model_count - 1) // 2) * 5,
            "rank_detail": model_count * repeats * 5,
            "metric_summary": model_count * 7,
            "sample_counts": patients,
            "sample_stability": model_count * patients,
        }
        observed_counts = {
            "metrics": len(metrics),
            "predictions": len(predictions),
            "tuning": len(tuning),
            "oof": len(oof),
            "selected": len(selected),
            "baseline_detail": len(baseline_detail),
            "baseline_summary": len(baseline_summary),
            "pairwise": len(pairwise),
            "rank_detail": len(rank_detail),
            "metric_summary": len(metric_summary),
            "sample_counts": len(sample_counts),
            "sample_stability": len(sample_stability),
        }
        if observed_counts != expected_counts:
            raise ValidationError(
                f"Output row counts={observed_counts}, expected={expected_counts}."
            )

        if metrics["model"].nunique() != model_count:
            raise ValidationError("Metric model count mismatch.")
        if metrics["outer_repeat"].nunique() != repeats:
            raise ValidationError("Metric repeat count mismatch.")
        if metrics.duplicated(["task_id", "model"]).any():
            raise ValidationError("Duplicate metric task/model rows.")
        if predictions.duplicated(["task_id", "model", "sample_id"]).any():
            raise ValidationError("Duplicate prediction rows.")
        if not bool_series(metrics["fit_converged"]).all():
            raise ValidationError("Non-converged outer refits found.")
        if not bool_series(tuning["all_fits_converged"]).all():
            raise ValidationError("Non-converged inner fits found.")
        if not (inner_convergence["convergence_frequency"] == 1.0).all():
            raise ValidationError("Inner convergence summary is not complete.")
        if not (outer_convergence["convergence_frequency"] == 1.0).all():
            raise ValidationError("Outer convergence summary is not complete.")

        if args.baseline_model not in set(models):
            raise ValidationError("Configured baseline model is absent.")
        if set(baseline_summary["baseline_model"]) != {args.baseline_model}:
            raise ValidationError("Baseline summary model mismatch.")
        if 0.0 not in set(alpha_frequency["alpha"].astype(float)):
            raise ValidationError("Alpha=0 is absent from selection-frequency output.")
        expected_families = {
            "Ridge_alpha0",
            "ElasticNet_interior",
            "Lasso_alpha1",
        }
        if not set(family_frequency["regularization_family"]).issubset(
            expected_families
        ):
            raise ValidationError("Unexpected regularization family.")
        family_sum = family_frequency.groupby("model")["selection_frequency"].sum()
        if family_sum.sub(1.0).abs().max() > 1e-12:
            raise ValidationError(
                "Regularization-family frequencies do not sum to one."
            )

        holdout_total = int(sample_counts["holdout_appearances"].sum())
        if holdout_total != tasks * holdout_size:
            raise ValidationError(
                f"Total holdout appearances={holdout_total}, "
                f"expected={tasks * holdout_size}."
            )
        nonzero_expected = sample_stability[
            "expected_holdout_predictions"
        ].astype(int) > 0
        if not (
            sample_stability.loc[nonzero_expected, "coverage_frequency"] == 1.0
        ).all():
            raise ValidationError("Sample prediction coverage is incomplete.")

        contract = summary["scientific_contract"]
        if contract["final_model_selected"]:
            raise ValidationError("Summary incorrectly selects a final model.")
        if contract["p_values_calculated"]:
            raise ValidationError("Summary incorrectly reports p-values.")
        if contract["repeats_treated_as_independent_cohorts"]:
            raise ValidationError("Summary treats repeats as independent cohorts.")

        print("IGH repeated-holdout aggregate validation: PASS")
        print(f"Tasks:               {tasks}")
        print(f"Models:              {model_count}")
        print(f"Metric rows:         {len(metrics)}")
        print(f"Prediction rows:     {len(predictions)}")
        print(f"Inner tuning rows:   {len(tuning)}")
        print(f"Inner OOF rows:      {len(oof)}")
        print(f"Patients audited:    {len(sample_counts)}")
        print(f"Max inner iter:      {int(inner_convergence['max_iterations_used'].max())}")
        print(f"Max outer iter:      {int(outer_convergence['max_iterations_used'].max())}")
        print("IGH_REPEATED_HOLDOUT_AGGREGATE_VALIDATION_PASS")
        return 0
    except Exception as exc:
        print(f"IGH repeated-holdout aggregate validation: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
