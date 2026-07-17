#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Review M2 Elastic Net parameter-selection frequency for the IGH project.

The input is the 100-task selected-parameter table collected in step 06.  The
script summarizes alpha, lambda and classification-threshold distributions,
checks that all 20 x 5 outer tasks are present exactly once, and diagnoses
whether the original tuning grid is too narrow at either boundary.

Interpretation used by the modeling worker:
- larger alpha (l1_ratio) gives a more L1-like, sparse Elastic Net penalty;
- larger lambda means stronger shrinkage because C = 1 / lambda.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.2.0-IGH"
EXPECTED_RECEPTOR = "IGH"
ROOT = Path("/data/users/chenhaisheng/RA-ILD/IGH")

DEFAULT_INPUT = (
    ROOT
    / "set/train/result/06_cv_result_summary/collected"
    / "06_all_selected_parameters.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "set/train/result/06_cv_result_summary/analysis"
    / "m2_parameter_review"
)
DEFAULT_MODEL = "M2_static_igh_public"
DEFAULT_ALPHA_GRID = (0.1, 0.5, 0.9)
DEFAULT_LAMBDA_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)


def parse_float_list(text: str) -> List[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one numeric value is required.")
    if any(not np.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("Grid values must be finite.")
    return sorted(set(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize M2 alpha/lambda selection frequency and diagnose the "
            "candidate grid across repeated nested-CV outer tasks."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--expected-repeats", type=int, default=20)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--expected-tasks", type=int, default=100)
    parser.add_argument(
        "--alpha-grid",
        type=parse_float_list,
        default=list(DEFAULT_ALPHA_GRID),
        help="Alpha candidates used by the nested-CV worker.",
    )
    parser.add_argument(
        "--lambda-grid",
        type=parse_float_list,
        default=list(DEFAULT_LAMBDA_GRID),
        help="Lambda candidates used by the nested-CV worker.",
    )
    parser.add_argument(
        "--central-coverage",
        type=float,
        default=0.90,
        help="Target mass for the shortest contiguous selected-parameter interval.",
    )
    parser.add_argument(
        "--boundary-warning-frequency",
        type=float,
        default=0.20,
        help="Boundary selection frequency that triggers a grid-extension warning.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.expected_repeats < 1 or args.expected_folds < 2:
        parser.error("Expected repeats must be >=1 and folds must be >=2.")
    if args.expected_tasks != args.expected_repeats * args.expected_folds:
        parser.error("--expected-tasks must equal repeats × folds.")
    if not (0.50 <= args.central_coverage <= 1.0):
        parser.error("--central-coverage must be in [0.50, 1.0].")
    if not (0.0 <= args.boundary_warning_frequency <= 1.0):
        parser.error("--boundary-warning-frequency must be in [0, 1].")
    if any(value <= 0 or value > 1 for value in args.alpha_grid):
        parser.error("Every alpha candidate must be in (0, 1].")
    if any(value <= 0 for value in args.lambda_grid):
        parser.error("Every lambda candidate must be > 0.")
    return args


def get_output_paths(output_dir: Path) -> Dict[str, Path]:
    return {
        "alpha_frequency": output_dir / "06_M2_alpha_frequency.csv",
        "lambda_frequency": output_dir / "06_M2_lambda_frequency.csv",
        "combo_frequency": output_dir / "06_M2_alpha_lambda_frequency.csv",
        "threshold_summary": output_dir / "06_M2_threshold_summary.csv",
        "threshold_bins": output_dir / "06_M2_threshold_interval_frequency.csv",
        "threshold_by_repeat": output_dir / "06_M2_threshold_by_repeat.csv",
        "combo_performance": output_dir / "06_M2_parameter_inner_performance.csv",
        "task_parameters": output_dir / "06_M2_all_task_parameters.csv",
        "grid_diagnostics": output_dir / "06_M2_parameter_grid_diagnostics.csv",
        "recommendation": output_dir / "06_M2_candidate_range_recommendation.csv",
        "configuration": output_dir / "06_M2_parameter_review_configuration.json",
        "report": output_dir / "06_M2_parameter_review.txt",
    }


def check_overwrite(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output files already exist. Add --overwrite to replace them:\n"
            + "\n".join(f"  - {path}" for path in existing)
        )


def find_alpha_column(columns: Sequence[str]) -> str:
    for candidate in ("l1_ratio_alpha", "alpha", "l1_ratio"):
        if candidate in columns:
            return candidate
    raise ValueError(
        "No alpha column found. Expected one of: l1_ratio_alpha, alpha, l1_ratio"
    )


def match_grid_value(value: float, grid: Sequence[float], atol: float = 1e-10) -> float:
    matches = [candidate for candidate in grid if math.isclose(value, candidate, abs_tol=atol, rel_tol=1e-9)]
    if len(matches) != 1:
        raise ValueError(f"Selected value {value!r} is not uniquely present in grid {list(grid)}")
    return float(matches[0])


def complete_frequency_table(
    values: pd.Series,
    grid: Sequence[float],
    value_name: str,
) -> pd.DataFrame:
    counts = values.value_counts().to_dict()
    total = len(values)
    rows = []
    for value in grid:
        count = int(counts.get(float(value), 0))
        rows.append(
            {
                value_name: float(value),
                "count": count,
                "percentage": count / total * 100.0,
                "selection_frequency": count / total,
                "is_lower_grid_boundary": bool(value == min(grid)),
                "is_upper_grid_boundary": bool(value == max(grid)),
            }
        )
    return pd.DataFrame(rows).sort_values(value_name).reset_index(drop=True)


def complete_combo_table(
    data: pd.DataFrame,
    alpha_col: str,
    alpha_grid: Sequence[float],
    lambda_grid: Sequence[float],
) -> pd.DataFrame:
    observed = (
        data.groupby([alpha_col, "lambda"], dropna=False)
        .agg(
            count=("model", "size"),
            threshold_mean=("threshold", "mean"),
            threshold_median=("threshold", "median"),
            inner_roc_auc_mean=("inner_roc_auc", "mean"),
            inner_roc_auc_sd=("inner_roc_auc", "std"),
            inner_roc_auc_median=("inner_roc_auc", "median"),
            inner_pr_auc_mean=("inner_pr_auc", "mean"),
            inner_pr_auc_sd=("inner_pr_auc", "std"),
            inner_pr_auc_median=("inner_pr_auc", "median"),
        )
        .reset_index()
    )
    full = pd.MultiIndex.from_product(
        [list(alpha_grid), list(lambda_grid)],
        names=[alpha_col, "lambda"],
    ).to_frame(index=False)
    result = full.merge(observed, how="left", on=[alpha_col, "lambda"])
    result["count"] = result["count"].fillna(0).astype(int)
    result["percentage"] = result["count"] / len(data) * 100.0
    result["selection_frequency"] = result["count"] / len(data)
    return result.sort_values(
        ["count", alpha_col, "lambda"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def shortest_contiguous_interval(
    frequency: pd.DataFrame,
    value_col: str,
    target_coverage: float,
) -> Tuple[float, float, float]:
    table = frequency.sort_values(value_col).reset_index(drop=True)
    values = table[value_col].to_numpy(float)
    masses = table["selection_frequency"].to_numpy(float)
    best: Tuple[int, float, int, int] | None = None
    for left in range(len(values)):
        mass = 0.0
        for right in range(left, len(values)):
            mass += masses[right]
            if mass + 1e-12 >= target_coverage:
                width = right - left
                candidate = (width, values[right] - values[left], left, right)
                if best is None or candidate < best:
                    best = candidate
                break
    if best is None:
        return float(values.min()), float(values.max()), float(masses.sum())
    _, _, left, right = best
    coverage = float(masses[left : right + 1].sum())
    return float(values[left]), float(values[right]), coverage


def next_upper_grid_value(grid: Sequence[float], hard_cap: float | None = None) -> float:
    """Extend a positive grid using a readable 1-3-10 logarithmic sequence."""
    value = max(float(item) for item in grid)
    exponent = math.floor(math.log10(value))
    scale = 10.0 ** exponent
    mantissa = value / scale
    if mantissa < 1.5:
        candidate = 3.0 * scale
    elif mantissa < 5.0:
        candidate = 10.0 * scale
    else:
        candidate = 30.0 * scale
    if candidate <= value:
        candidate = value * 3.0
    if hard_cap is not None:
        candidate = min(candidate, hard_cap)
    return float(candidate)


def next_lower_grid_value(grid: Sequence[float], hard_floor: float | None = None) -> float:
    """Extend a positive grid downward using a readable 1-3-10 sequence."""
    value = min(float(item) for item in grid)
    exponent = math.floor(math.log10(value))
    scale = 10.0 ** exponent
    mantissa = value / scale
    if mantissa >= 7.0:
        candidate = 3.0 * scale
    elif mantissa >= 2.0:
        candidate = 1.0 * scale
    else:
        candidate = 3.0 * scale / 10.0
    if candidate >= value:
        candidate = value / 3.0
    if hard_floor is not None:
        candidate = max(candidate, hard_floor)
    return float(candidate)


def format_grid(values: Sequence[float]) -> str:
    return ",".join(f"{value:g}" for value in values)


def build_recommendation(
    alpha_frequency: pd.DataFrame,
    lambda_frequency: pd.DataFrame,
    alpha_grid: Sequence[float],
    lambda_grid: Sequence[float],
    coverage: float,
    boundary_warning: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    alpha_min, alpha_max, alpha_coverage = shortest_contiguous_interval(
        alpha_frequency, "l1_ratio_alpha", coverage
    )
    lambda_min, lambda_max, lambda_coverage = shortest_contiguous_interval(
        lambda_frequency, "lambda", coverage
    )

    alpha_lower_freq = float(
        alpha_frequency.loc[
            alpha_frequency["l1_ratio_alpha"] == min(alpha_grid),
            "selection_frequency",
        ].iloc[0]
    )
    alpha_upper_freq = float(
        alpha_frequency.loc[
            alpha_frequency["l1_ratio_alpha"] == max(alpha_grid),
            "selection_frequency",
        ].iloc[0]
    )
    lambda_lower_freq = float(
        lambda_frequency.loc[
            lambda_frequency["lambda"] == min(lambda_grid),
            "selection_frequency",
        ].iloc[0]
    )
    lambda_upper_freq = float(
        lambda_frequency.loc[
            lambda_frequency["lambda"] == max(lambda_grid),
            "selection_frequency",
        ].iloc[0]
    )

    alpha_weighted_mean = float(
        np.average(
            alpha_frequency["l1_ratio_alpha"],
            weights=alpha_frequency["count"],
        )
    )
    alpha_high_share = float(
        alpha_frequency.loc[
            alpha_frequency["l1_ratio_alpha"] >= 0.8,
            "selection_frequency",
        ].sum()
    )
    lambda_weighted_log10_mean = float(
        np.average(
            np.log10(lambda_frequency["lambda"]),
            weights=lambda_frequency["count"],
        )
    )
    lambda_ge_3_share = float(
        lambda_frequency.loc[
            lambda_frequency["lambda"] >= 3.0,
            "selection_frequency",
        ].sum()
    )
    lambda_ge_10_share = float(
        lambda_frequency.loc[
            lambda_frequency["lambda"] >= 10.0,
            "selection_frequency",
        ].sum()
    )

    suggested_alpha = sorted(set(float(value) for value in alpha_grid))
    alpha_upper_warning = alpha_upper_freq >= boundary_warning
    alpha_lower_warning = alpha_lower_freq >= boundary_warning
    if alpha_upper_warning and max(suggested_alpha) < 1.0:
        suggested_alpha.append(1.0)
    if alpha_lower_warning and min(suggested_alpha) > 0.01:
        suggested_alpha.append(next_lower_grid_value(suggested_alpha, hard_floor=0.01))
    suggested_alpha = sorted(set(round(value, 12) for value in suggested_alpha))

    suggested_lambda = sorted(set(float(value) for value in lambda_grid))
    lambda_upper_warning = lambda_upper_freq >= boundary_warning
    lambda_lower_warning = lambda_lower_freq >= boundary_warning
    if lambda_upper_warning:
        suggested_lambda.append(next_upper_grid_value(suggested_lambda))
    if lambda_lower_warning:
        suggested_lambda.append(next_lower_grid_value(suggested_lambda, hard_floor=1e-6))
    suggested_lambda = sorted(set(round(value, 12) for value in suggested_lambda))

    alpha_pattern = (
        "strongly_sparse_leaning"
        if alpha_high_share >= 0.50
        else "moderately_sparse_leaning"
        if alpha_weighted_mean >= 0.60
        else "mixed_penalty"
    )
    lambda_pattern = (
        "very_strong_regularization"
        if lambda_ge_10_share >= 0.50
        else "strong_regularization"
        if lambda_ge_3_share >= 0.50
        else "mixed_regularization"
    )

    diagnostics = pd.DataFrame(
        [
            {
                "parameter": "alpha",
                "grid_min": min(alpha_grid),
                "grid_max": max(alpha_grid),
                "lower_boundary_selection_frequency": alpha_lower_freq,
                "upper_boundary_selection_frequency": alpha_upper_freq,
                "lower_boundary_warning": alpha_lower_warning,
                "upper_boundary_warning": alpha_upper_warning,
                "central_interval_min": alpha_min,
                "central_interval_max": alpha_max,
                "central_interval_coverage": alpha_coverage,
                "weighted_mean": alpha_weighted_mean,
                "high_value_share": alpha_high_share,
                "pattern": alpha_pattern,
            },
            {
                "parameter": "lambda",
                "grid_min": min(lambda_grid),
                "grid_max": max(lambda_grid),
                "lower_boundary_selection_frequency": lambda_lower_freq,
                "upper_boundary_selection_frequency": lambda_upper_freq,
                "lower_boundary_warning": lambda_lower_warning,
                "upper_boundary_warning": lambda_upper_warning,
                "central_interval_min": lambda_min,
                "central_interval_max": lambda_max,
                "central_interval_coverage": lambda_coverage,
                "weighted_mean": 10 ** lambda_weighted_log10_mean,
                "high_value_share": lambda_ge_3_share,
                "pattern": lambda_pattern,
            },
        ]
    )

    recommendation = pd.DataFrame(
        [
            {
                "model": DEFAULT_MODEL,
                "alpha_pattern": alpha_pattern,
                "alpha_weighted_mean": alpha_weighted_mean,
                "alpha_ge_0_8_selection_frequency": alpha_high_share,
                "alpha_central_interval": f"[{alpha_min:g}, {alpha_max:g}]",
                "alpha_upper_boundary_warning": alpha_upper_warning,
                "suggested_alpha_grid": format_grid(suggested_alpha),
                "lambda_pattern": lambda_pattern,
                "lambda_ge_3_selection_frequency": lambda_ge_3_share,
                "lambda_ge_10_selection_frequency": lambda_ge_10_share,
                "lambda_central_interval": f"[{lambda_min:g}, {lambda_max:g}]",
                "lambda_upper_boundary_warning": lambda_upper_warning,
                "suggested_lambda_grid": format_grid(suggested_lambda),
                "interpretation": (
                    "Alpha controls the L1/L2 mixture; larger alpha is more sparse. "
                    "Lambda controls shrinkage; larger lambda is stronger because C=1/lambda. "
                    "A frequent upper-boundary selection suggests extending that grid upward."
                ),
            }
        ]
    )
    return diagnostics, recommendation


def threshold_outputs(model_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    threshold = model_data["threshold"]
    threshold_summary = pd.DataFrame(
        {
            "statistic": [
                "n", "mean", "sd", "min", "q05", "q10", "q25",
                "median", "q75", "q90", "q95", "max", "IQR",
            ],
            "value": [
                len(threshold), threshold.mean(), threshold.std(ddof=1),
                threshold.min(), threshold.quantile(0.05), threshold.quantile(0.10),
                threshold.quantile(0.25), threshold.median(), threshold.quantile(0.75),
                threshold.quantile(0.90), threshold.quantile(0.95), threshold.max(),
                threshold.quantile(0.75) - threshold.quantile(0.25),
            ],
        }
    )
    bins = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.000001]
    intervals = pd.cut(threshold, bins=bins, include_lowest=True, right=False)
    threshold_bins = (
        intervals.value_counts(sort=False)
        .rename_axis("threshold_interval")
        .reset_index(name="count")
    )
    threshold_bins["threshold_interval"] = threshold_bins["threshold_interval"].astype(str)
    threshold_bins["percentage"] = threshold_bins["count"] / len(threshold) * 100.0
    threshold_bins["selection_frequency"] = threshold_bins["count"] / len(threshold)

    threshold_by_repeat = (
        model_data.groupby("outer_repeat")
        .agg(
            n_folds=("threshold", "count"),
            mean=("threshold", "mean"),
            median=("threshold", "median"),
            minimum=("threshold", "min"),
            maximum=("threshold", "max"),
            sd=("threshold", "std"),
        )
        .reset_index()
        .sort_values("outer_repeat")
    )
    return threshold_summary, threshold_bins, threshold_by_repeat


def format_table(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda value: f"{value:.4f}")


def validate_and_prepare(args: argparse.Namespace) -> Tuple[pd.DataFrame, str]:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    data = pd.read_csv(input_path)
    if "model" not in data.columns:
        raise ValueError("Input file has no 'model' column.")

    alpha_col = find_alpha_column(data.columns.tolist())
    required = {
        "outer_repeat", "outer_fold", "model", alpha_col, "lambda", "threshold",
        "inner_roc_auc", "inner_pr_auc",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    model_data = data.loc[data["model"].astype(str) == args.model].copy()
    if model_data.empty:
        available = sorted(data["model"].astype(str).unique())
        raise ValueError(f"Model '{args.model}' not found. Available: {available}")

    numeric_cols = ["outer_repeat", "outer_fold", alpha_col, "lambda", "threshold", "inner_roc_auc", "inner_pr_auc"]
    for col in numeric_cols:
        model_data[col] = pd.to_numeric(model_data[col], errors="raise")

    if len(model_data) != args.expected_tasks:
        raise ValueError(
            f"Observed {len(model_data)} rows for {args.model}; expected {args.expected_tasks}."
        )
    if model_data.duplicated(["outer_repeat", "outer_fold"]).any():
        raise ValueError("Duplicate outer task rows detected for the selected model.")

    model_data["outer_repeat"] = model_data["outer_repeat"].astype(int)
    model_data["outer_fold"] = model_data["outer_fold"].astype(int)
    expected_tasks = {
        (repeat, fold)
        for repeat in range(1, args.expected_repeats + 1)
        for fold in range(1, args.expected_folds + 1)
    }
    observed_tasks = set(zip(model_data["outer_repeat"], model_data["outer_fold"]))
    if observed_tasks != expected_tasks:
        missing_tasks = sorted(expected_tasks - observed_tasks)[:10]
        extra_tasks = sorted(observed_tasks - expected_tasks)[:10]
        raise ValueError(
            f"Outer task grid mismatch. Missing examples={missing_tasks}; extra examples={extra_tasks}"
        )

    if not model_data[alpha_col].between(0, 1, inclusive="right").all() or (model_data[alpha_col] <= 0).any():
        raise ValueError("Alpha values must be in (0, 1].")
    if (model_data["lambda"] <= 0).any():
        raise ValueError("Lambda values must be > 0.")
    for col in ("threshold", "inner_roc_auc", "inner_pr_auc"):
        if not model_data[col].between(0, 1, inclusive="both").all():
            raise ValueError(f"{col} values must be in [0, 1].")

    model_data[alpha_col] = [match_grid_value(value, args.alpha_grid) for value in model_data[alpha_col]]
    model_data["lambda"] = [match_grid_value(value, args.lambda_grid) for value in model_data["lambda"]]
    model_data = model_data.sort_values(["outer_repeat", "outer_fold"]).reset_index(drop=True)
    return model_data, alpha_col


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = get_output_paths(output_dir)
    check_overwrite(output_paths.values(), args.overwrite)

    model_data, alpha_col = validate_and_prepare(args)
    if alpha_col != "l1_ratio_alpha":
        model_data = model_data.rename(columns={alpha_col: "l1_ratio_alpha"})
    alpha_col = "l1_ratio_alpha"

    alpha_frequency = complete_frequency_table(
        model_data[alpha_col], args.alpha_grid, alpha_col
    )
    lambda_frequency = complete_frequency_table(
        model_data["lambda"], args.lambda_grid, "lambda"
    )
    combo_performance = complete_combo_table(
        model_data, alpha_col, args.alpha_grid, args.lambda_grid
    )
    combo_frequency = combo_performance[
        [alpha_col, "lambda", "count", "percentage", "selection_frequency"]
    ].copy()
    threshold_summary, threshold_bins, threshold_by_repeat = threshold_outputs(model_data)
    diagnostics, recommendation = build_recommendation(
        alpha_frequency=alpha_frequency,
        lambda_frequency=lambda_frequency,
        alpha_grid=args.alpha_grid,
        lambda_grid=args.lambda_grid,
        coverage=args.central_coverage,
        boundary_warning=args.boundary_warning_frequency,
    )
    recommendation["model"] = args.model

    model_data.to_csv(output_paths["task_parameters"], index=False)
    alpha_frequency.to_csv(output_paths["alpha_frequency"], index=False)
    lambda_frequency.to_csv(output_paths["lambda_frequency"], index=False)
    combo_frequency.to_csv(output_paths["combo_frequency"], index=False)
    threshold_summary.to_csv(output_paths["threshold_summary"], index=False)
    threshold_bins.to_csv(output_paths["threshold_bins"], index=False)
    threshold_by_repeat.to_csv(output_paths["threshold_by_repeat"], index=False)
    combo_performance.to_csv(output_paths["combo_performance"], index=False)
    diagnostics.to_csv(output_paths["grid_diagnostics"], index=False)
    recommendation.to_csv(output_paths["recommendation"], index=False)

    config = {
        "script_version": SCRIPT_VERSION,
        "receptor": EXPECTED_RECEPTOR,
        "input": str(input_path),
        "output_dir": str(output_dir),
        "model": args.model,
        "expected_repeats": args.expected_repeats,
        "expected_folds": args.expected_folds,
        "expected_tasks": args.expected_tasks,
        "alpha_grid": args.alpha_grid,
        "lambda_grid": args.lambda_grid,
        "central_coverage": args.central_coverage,
        "boundary_warning_frequency": args.boundary_warning_frequency,
    }
    output_paths["configuration"].write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    top_combo = combo_frequency.sort_values(
        ["count", alpha_col, "lambda"], ascending=[False, False, False]
    ).iloc[0]
    rec = recommendation.iloc[0]
    report_lines = [
        "=" * 88,
        "M2 IGH ELASTIC NET PARAMETER SELECTION REVIEW",
        "=" * 88,
        f"Script version: {SCRIPT_VERSION}",
        f"Receptor: {EXPECTED_RECEPTOR}",
        f"Input file: {input_path}",
        f"Model: {args.model}",
        f"Observed outer tasks: {len(model_data)}",
        f"Alpha grid: {format_grid(args.alpha_grid)}",
        f"Lambda grid: {format_grid(args.lambda_grid)}",
        "",
        "=" * 88,
        "1. Alpha selection frequency",
        "=" * 88,
        format_table(alpha_frequency),
        "",
        "Interpretation: larger alpha is more L1-like and produces a sparser model.",
        f"Observed alpha pattern: {rec['alpha_pattern']}",
        f"Alpha >= 0.8 selection frequency: {rec['alpha_ge_0_8_selection_frequency']:.1%}",
        f"Suggested alpha grid for a future pre-specified tuning run: {rec['suggested_alpha_grid']}",
        "",
        "=" * 88,
        "2. Lambda selection frequency",
        "=" * 88,
        format_table(lambda_frequency),
        "",
        "Interpretation: larger lambda means stronger shrinkage because the worker uses C=1/lambda.",
        f"Observed lambda pattern: {rec['lambda_pattern']}",
        f"Lambda >= 3 selection frequency: {rec['lambda_ge_3_selection_frequency']:.1%}",
        f"Lambda >= 10 selection frequency: {rec['lambda_ge_10_selection_frequency']:.1%}",
        f"Suggested lambda grid for a future pre-specified tuning run: {rec['suggested_lambda_grid']}",
        "",
        "=" * 88,
        "3. Alpha + lambda combination frequency",
        "=" * 88,
        format_table(combo_frequency),
        "",
        "=" * 88,
        "4. Parameter-grid boundary diagnostics",
        "=" * 88,
        format_table(diagnostics),
        "",
        "A frequent selection at the largest candidate means the current grid may be truncated;",
        "it does not by itself prove that an even larger value will improve independent-test performance.",
        "",
        "=" * 88,
        "5. Threshold descriptive statistics",
        "=" * 88,
        format_table(threshold_summary),
        "",
        "=" * 88,
        "6. Threshold interval frequency",
        "=" * 88,
        format_table(threshold_bins),
        "",
        "=" * 88,
        "7. Threshold summary by repeat",
        "=" * 88,
        format_table(threshold_by_repeat),
        "",
        "=" * 88,
        "8. Parameter combination and inner-CV performance",
        "=" * 88,
        format_table(combo_performance),
        "",
        "=" * 88,
        "9. Most frequently selected combination",
        "=" * 88,
        f"Alpha: {top_combo[alpha_col]:g}",
        f"Lambda: {top_combo['lambda']:g}",
        f"Selected count: {int(top_combo['count'])}/{len(model_data)}",
        f"Selection percentage: {top_combo['percentage']:.1f}%",
        f"Overall threshold median: {model_data['threshold'].median():.4f}",
        f"Overall threshold mean: {model_data['threshold'].mean():.4f}",
        "",
        "=" * 88,
        "10. Use of this review",
        "=" * 88,
        "- These frequencies describe the nested-CV choices already made; they are not test-set results.",
        "- Do not change the completed repeated-CV analysis after seeing outer validation performance.",
        "- A refined grid is appropriate only for a separately pre-specified final-fit/sensitivity analysis.",
        "- Final independent-test evaluation must remain untouched while the candidate grid is chosen.",
        "",
        "=" * 88,
        "Output files",
        "=" * 88,
    ]
    for name, path in output_paths.items():
        report_lines.append(f"- {name}: {path}")
    report = "\n".join(report_lines) + "\n"
    output_paths["report"].write_text(report, encoding="utf-8")

    print("=" * 88)
    print("06 IGH M2 Elastic Net Parameter Review")
    print("=" * 88)
    print(f"Model: {args.model}")
    print(f"Outer tasks: {len(model_data)}")
    print(f"Most frequent alpha/lambda: {top_combo[alpha_col]:g} / {top_combo['lambda']:g}")
    print(f"Alpha pattern: {rec['alpha_pattern']}")
    print(f"Lambda pattern: {rec['lambda_pattern']}")
    print(f"Suggested alpha grid: {rec['suggested_alpha_grid']}")
    print(f"Suggested lambda grid: {rec['suggested_lambda_grid']}")
    print("[Output files]")
    for name, path in output_paths.items():
        print(f"- {name}: {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
