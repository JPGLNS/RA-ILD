#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Review M2 Elastic Net parameter-selection frequency and threshold distribution.

Default input:
  /data/users/chenhaisheng/RA-ILD/TRB/set/train/result/
  06_cv_result_summary/collected/06_all_selected_parameters.csv

Default output:
  /data/users/chenhaisheng/RA-ILD/TRB/set/train/result/
  06_cv_result_summary/analysis/m2_parameter_review/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


ROOT = Path("/data/users/chenhaisheng/RA-ILD/TRB")

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

DEFAULT_MODEL = "M2_static_tcr_public"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize M2 alpha/lambda selection frequency and threshold "
            "distribution across repeated nested-CV outer tasks."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Combined selected-parameter CSV from step 06.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for summary outputs.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name to analyze.",
    )
    parser.add_argument(
        "--expected-tasks",
        type=int,
        default=100,
        help="Expected number of outer tasks for the selected model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing outputs.",
    )
    return parser.parse_args()


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
        "report": output_dir / "06_M2_parameter_review.txt",
    }


def check_overwrite(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output files already exist. Add --overwrite to replace them:\n"
            + "\n".join(f"  - {path}" for path in existing)
        )


def find_alpha_column(columns: List[str]) -> str:
    candidates = ["l1_ratio_alpha", "alpha", "l1_ratio"]
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        "No alpha column found. Expected one of: "
        + ", ".join(candidates)
    )


def percentage_table(
    data: pd.DataFrame,
    group_columns: List[str],
    total: int,
) -> pd.DataFrame:
    result = (
        data.groupby(group_columns, dropna=False)
        .size()
        .reset_index(name="count")
    )
    result["percentage"] = result["count"] / total * 100
    return result.sort_values(
        ["count"] + group_columns,
        ascending=[False] + [True] * len(group_columns),
    ).reset_index(drop=True)


def format_table(df: pd.DataFrame) -> str:
    return df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )


def main() -> int:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = get_output_paths(output_dir)
    check_overwrite(output_paths.values(), args.overwrite)

    data = pd.read_csv(input_path)

    if "model" not in data.columns:
        raise ValueError("Input file has no 'model' column.")

    model_data = data.loc[
        data["model"].astype(str) == args.model
    ].copy()

    if model_data.empty:
        available = sorted(data["model"].astype(str).unique())
        raise ValueError(
            f"Model '{args.model}' was not found. "
            f"Available models: {available}"
        )

    alpha_column = find_alpha_column(model_data.columns.tolist())

    required_columns = [alpha_column, "lambda", "threshold"]
    missing = [
        column
        for column in required_columns
        if column not in model_data.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_columns = [
        alpha_column,
        "lambda",
        "threshold",
        "inner_roc_auc",
        "inner_pr_auc",
    ]
    for column in numeric_columns:
        if column in model_data.columns:
            model_data[column] = pd.to_numeric(
                model_data[column],
                errors="coerce",
            )

    if len(model_data) != args.expected_tasks:
        raise ValueError(
            f"Observed {len(model_data)} rows for {args.model}, "
            f"expected {args.expected_tasks}."
        )

    duplicate_keys = [
        column
        for column in ["outer_repeat", "outer_fold"]
        if column in model_data.columns
    ]
    if len(duplicate_keys) == 2:
        if model_data.duplicated(duplicate_keys).any():
            duplicates = model_data.loc[
                model_data.duplicated(duplicate_keys, keep=False),
                duplicate_keys,
            ]
            raise ValueError(
                "Duplicate outer task rows detected:\n"
                + duplicates.to_string(index=False)
            )

    for column in required_columns:
        if model_data[column].isna().any():
            raise ValueError(
                f"Column '{column}' contains "
                f"{int(model_data[column].isna().sum())} missing values."
            )

    alpha_frequency = percentage_table(
        model_data,
        [alpha_column],
        len(model_data),
    )
    lambda_frequency = percentage_table(
        model_data,
        ["lambda"],
        len(model_data),
    )
    combo_frequency = percentage_table(
        model_data,
        [alpha_column, "lambda"],
        len(model_data),
    )

    threshold = model_data["threshold"].dropna()

    threshold_summary = pd.DataFrame(
        {
            "statistic": [
                "n",
                "mean",
                "sd",
                "min",
                "q05",
                "q10",
                "q25",
                "median",
                "q75",
                "q90",
                "q95",
                "max",
                "IQR",
            ],
            "value": [
                len(threshold),
                threshold.mean(),
                threshold.std(ddof=1),
                threshold.min(),
                threshold.quantile(0.05),
                threshold.quantile(0.10),
                threshold.quantile(0.25),
                threshold.median(),
                threshold.quantile(0.75),
                threshold.quantile(0.90),
                threshold.quantile(0.95),
                threshold.max(),
                threshold.quantile(0.75)
                - threshold.quantile(0.25),
            ],
        }
    )

    bins = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.000001]
    interval = pd.cut(
        threshold,
        bins=bins,
        include_lowest=True,
        right=False,
    )
    threshold_bins = (
        interval.value_counts(sort=False)
        .rename_axis("threshold_interval")
        .reset_index(name="count")
    )
    threshold_bins["threshold_interval"] = (
        threshold_bins["threshold_interval"].astype(str)
    )
    threshold_bins["percentage"] = (
        threshold_bins["count"] / len(threshold) * 100
    )

    if "outer_repeat" in model_data.columns:
        repeat_aggregation = {
            "n_folds": ("threshold", "count"),
            "mean": ("threshold", "mean"),
            "median": ("threshold", "median"),
            "minimum": ("threshold", "min"),
            "maximum": ("threshold", "max"),
            "sd": ("threshold", "std"),
        }
        threshold_by_repeat = (
            model_data.groupby("outer_repeat")
            .agg(**repeat_aggregation)
            .reset_index()
            .sort_values("outer_repeat")
        )
    else:
        threshold_by_repeat = pd.DataFrame()

    aggregation = {
        "selected_count": ("model", "size"),
        "threshold_mean": ("threshold", "mean"),
        "threshold_median": ("threshold", "median"),
    }

    if "inner_roc_auc" in model_data.columns:
        aggregation.update(
            {
                "inner_roc_auc_mean": ("inner_roc_auc", "mean"),
                "inner_roc_auc_sd": ("inner_roc_auc", "std"),
                "inner_roc_auc_median": ("inner_roc_auc", "median"),
            }
        )

    if "inner_pr_auc" in model_data.columns:
        aggregation.update(
            {
                "inner_pr_auc_mean": ("inner_pr_auc", "mean"),
                "inner_pr_auc_sd": ("inner_pr_auc", "std"),
                "inner_pr_auc_median": ("inner_pr_auc", "median"),
            }
        )

    combo_performance = (
        model_data.groupby([alpha_column, "lambda"])
        .agg(**aggregation)
        .reset_index()
        .sort_values(
            ["selected_count", alpha_column, "lambda"],
            ascending=[False, True, True],
        )
    )

    task_sort_columns = [
        column
        for column in ["outer_repeat", "outer_fold"]
        if column in model_data.columns
    ]
    if task_sort_columns:
        model_data = model_data.sort_values(task_sort_columns)

    model_data.to_csv(
        output_paths["task_parameters"],
        index=False,
    )
    alpha_frequency.to_csv(
        output_paths["alpha_frequency"],
        index=False,
    )
    lambda_frequency.to_csv(
        output_paths["lambda_frequency"],
        index=False,
    )
    combo_frequency.to_csv(
        output_paths["combo_frequency"],
        index=False,
    )
    threshold_summary.to_csv(
        output_paths["threshold_summary"],
        index=False,
    )
    threshold_bins.to_csv(
        output_paths["threshold_bins"],
        index=False,
    )
    threshold_by_repeat.to_csv(
        output_paths["threshold_by_repeat"],
        index=False,
    )
    combo_performance.to_csv(
        output_paths["combo_performance"],
        index=False,
    )

    top_combo = combo_frequency.iloc[0]

    report_lines = [
        "=" * 80,
        "M2 PARAMETER SELECTION REVIEW",
        "=" * 80,
        f"Input file: {input_path}",
        f"Model: {args.model}",
        f"Observed tasks: {len(model_data)}",
        f"Alpha column: {alpha_column}",
        "",
        "=" * 80,
        "1. Alpha selection frequency",
        "=" * 80,
        format_table(alpha_frequency),
        "",
        "=" * 80,
        "2. Lambda selection frequency",
        "=" * 80,
        format_table(lambda_frequency),
        "",
        "=" * 80,
        "3. Alpha + lambda combination frequency",
        "=" * 80,
        format_table(combo_frequency),
        "",
        "=" * 80,
        "4. Threshold descriptive statistics",
        "=" * 80,
        format_table(threshold_summary),
        "",
        "=" * 80,
        "5. Threshold interval frequency",
        "=" * 80,
        format_table(threshold_bins),
        "",
    ]

    if not threshold_by_repeat.empty:
        report_lines.extend(
            [
                "=" * 80,
                "6. Threshold summary by repeat",
                "=" * 80,
                format_table(threshold_by_repeat),
                "",
            ]
        )

    report_lines.extend(
        [
            "=" * 80,
            "7. Parameter combination and inner-CV performance",
            "=" * 80,
            format_table(combo_performance),
            "",
            "=" * 80,
            "8. Most frequently selected combination",
            "=" * 80,
            f"Alpha: {top_combo[alpha_column]:.4f}",
            f"Lambda: {top_combo['lambda']:.4f}",
            f"Selected count: {int(top_combo['count'])}/{len(model_data)}",
            f"Selection percentage: {top_combo['percentage']:.1f}%",
            f"Overall threshold median: {threshold.median():.4f}",
            f"Overall threshold mean: {threshold.mean():.4f}",
            "",
            "=" * 80,
            "Output files",
            "=" * 80,
        ]
    )

    for name, path in output_paths.items():
        report_lines.append(f"- {name}: {path}")

    report = "\n".join(report_lines) + "\n"
    output_paths["report"].write_text(report, encoding="utf-8")

    print(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
