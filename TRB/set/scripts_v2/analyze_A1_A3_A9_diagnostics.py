#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A1/A3/A9 repeated-holdout diagnostics for the combined 14-model TRB study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

A1 = "A1_core83"
A3 = "A3_core83_unweighted500"
A9 = "A9_core83_unweighted200"
TARGET_MODELS = (A1, A3, A9)

METRICS = (
    "roc_auc",
    "pr_auc",
    "accuracy",
    "sensitivity_recall",
    "specificity",
    "precision",
    "f1",
)

EXPECTED_PREDICTOR_COUNTS = {
    A1: 83,
    A3: 583,
    A9: 283,
}

PAIR_TOLERANCE = 1.0e-15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize A9-vs-A3/A1 paired performance, metric distributions, "
            "selected alpha/lambda values, and coefficient sparsity."
        )
    )
    parser.add_argument("--combined-metrics", required=True)
    parser.add_argument("--scheme011-hyperparameters", required=True)
    parser.add_argument("--scheme012-hyperparameters", required=True)
    parser.add_argument("--scheme011-coefficients", required=True)
    parser.add_argument("--scheme012-coefficients", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--coefficient-tolerance",
        type=float,
        default=1.0e-12,
        help="Absolute threshold used to define a nonzero coefficient.",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def read_csv(path: str | Path, label: str) -> pd.DataFrame:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return pd.read_csv(resolved)


def load_target_metrics(path: str | Path) -> pd.DataFrame:
    frame = read_csv(path, "combined 14-model metrics")
    require_columns(
        frame,
        ("split_id", "outer_repeat", "model", *METRICS),
        "combined 14-model metrics",
    )

    frame["split_id"] = frame["split_id"].astype(str)
    frame["model"] = frame["model"].astype(str)
    frame["outer_repeat"] = pd.to_numeric(
        frame["outer_repeat"], errors="raise"
    ).astype(int)

    for metric in METRICS:
        frame[metric] = pd.to_numeric(frame[metric], errors="raise")
        if not np.isfinite(frame[metric].to_numpy(float)).all():
            raise ValueError(f"Non-finite values found in metric: {metric}")

    target = frame.loc[frame["model"].isin(TARGET_MODELS)].copy()
    if set(target["model"]) != set(TARGET_MODELS):
        raise ValueError(
            f"Target models are incomplete: observed={sorted(target['model'].unique())}"
        )
    if target.duplicated(["split_id", "model"]).any():
        raise ValueError("Duplicate split_id/model rows found in metrics")

    counts = target.groupby("model")["split_id"].nunique()
    if not (counts == 100).all():
        raise ValueError(f"Every target model must have 100 splits:\n{counts}")

    expected_repeats = set(range(1, 101))
    for model, group in target.groupby("model"):
        if set(group["outer_repeat"]) != expected_repeats:
            raise ValueError(f"{model} outer_repeat values are not exactly 1..100")

    return target


def paired_comparison(
    metrics: pd.DataFrame,
    focal_model: str,
    reference_model: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    focal = metrics.loc[
        metrics["model"] == focal_model,
        ["split_id", "outer_repeat", *METRICS],
    ].copy()
    reference = metrics.loc[
        metrics["model"] == reference_model,
        ["split_id", "outer_repeat", *METRICS],
    ].copy()

    paired = focal.merge(
        reference,
        on=["split_id", "outer_repeat"],
        how="inner",
        suffixes=("_focal", "_reference"),
        validate="one_to_one",
    ).sort_values("outer_repeat")

    if len(paired) != 100:
        raise ValueError(
            f"{focal_model} vs {reference_model}: paired splits={len(paired)}, expected=100"
        )

    detail = paired[["split_id", "outer_repeat"]].copy()
    summary_rows: list[dict] = []

    for metric in METRICS:
        focal_values = paired[f"{metric}_focal"].to_numpy(float)
        reference_values = paired[f"{metric}_reference"].to_numpy(float)
        difference = focal_values - reference_values
        wins = difference > PAIR_TOLERANCE
        losses = difference < -PAIR_TOLERANCE
        ties = ~(wins | losses)

        detail[f"{metric}_{focal_model}"] = focal_values
        detail[f"{metric}_{reference_model}"] = reference_values
        detail[f"{metric}_difference_A9_minus_reference"] = difference
        detail[f"{metric}_A9_result"] = np.select(
            [wins, losses], ["win", "loss"], default="tie"
        )

        summary_rows.append(
            {
                "focal_model": focal_model,
                "reference_model": reference_model,
                "metric": metric,
                "n_paired_splits": 100,
                "mean_focal": float(np.mean(focal_values)),
                "mean_reference": float(np.mean(reference_values)),
                "mean_difference_focal_minus_reference": float(np.mean(difference)),
                "sd_difference": float(np.std(difference, ddof=1)),
                "median_difference": float(np.median(difference)),
                "q025_difference": float(np.quantile(difference, 0.025)),
                "q975_difference": float(np.quantile(difference, 0.975)),
                "focal_win_count": int(np.sum(wins)),
                "tie_count": int(np.sum(ties)),
                "focal_loss_count": int(np.sum(losses)),
                "focal_win_frequency": float(np.mean(wins)),
                "tie_frequency": float(np.mean(ties)),
                "focal_loss_frequency": float(np.mean(losses)),
            }
        )

    return detail, pd.DataFrame(summary_rows)


def metric_distribution_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for model in TARGET_MODELS:
        group = metrics.loc[metrics["model"] == model]
        for metric in METRICS:
            values = group[metric].to_numpy(float)
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "n_splits": int(len(values)),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "median": float(np.median(values)),
                    "q025": float(np.quantile(values, 0.025)),
                    "q975": float(np.quantile(values, 0.975)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def normalize_hyperparameter_columns(
    frame: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    frame = frame.copy()
    if "alpha" not in frame.columns and "selected_l1_ratio_alpha" in frame.columns:
        frame = frame.rename(columns={"selected_l1_ratio_alpha": "alpha"})
    if "lambda" not in frame.columns and "selected_lambda" in frame.columns:
        frame = frame.rename(columns={"selected_lambda": "lambda"})

    require_columns(frame, ("split_id", "model", "alpha", "lambda"), label)
    frame["split_id"] = frame["split_id"].astype(str)
    frame["model"] = frame["model"].astype(str)
    frame["alpha"] = pd.to_numeric(frame["alpha"], errors="raise")
    frame["lambda"] = pd.to_numeric(frame["lambda"], errors="raise")
    return frame


def attach_and_validate_outer_repeat(
    frame: pd.DataFrame,
    repeat_map: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """Attach outer_repeat by split_id or validate an existing column."""
    frame = frame.copy()
    mapping = repeat_map.loc[:, ["split_id", "outer_repeat"]].copy()
    mapping["split_id"] = mapping["split_id"].astype(str)
    mapping["outer_repeat"] = pd.to_numeric(
        mapping["outer_repeat"], errors="raise"
    ).astype(int)

    if mapping.duplicated("split_id").any():
        raise ValueError(f"{label}: repeat_map contains duplicate split_id values")

    if "outer_repeat" in frame.columns:
        frame["outer_repeat"] = pd.to_numeric(
            frame["outer_repeat"], errors="raise"
        ).astype(int)
        checked = frame.merge(
            mapping.rename(
                columns={"outer_repeat": "expected_outer_repeat"}
            ),
            on="split_id",
            how="left",
            validate="many_to_one",
        )
        if checked["expected_outer_repeat"].isna().any():
            missing = sorted(
                checked.loc[
                    checked["expected_outer_repeat"].isna(),
                    "split_id",
                ]
                .astype(str)
                .unique()
                .tolist()
            )
            raise ValueError(
                f"{label}: split_id values absent from repeat_map: {missing[:20]}"
            )
        mismatch = checked.loc[
            checked["outer_repeat"]
            != checked["expected_outer_repeat"]
        ]
        if not mismatch.empty:
            examples = mismatch.loc[
                :, ["split_id", "outer_repeat", "expected_outer_repeat"]
            ].head(20)
            raise ValueError(
                f"{label}: existing outer_repeat conflicts with split_id mapping:\n"
                f"{examples.to_string(index=False)}"
            )
        return checked.drop(columns=["expected_outer_repeat"])

    attached = frame.merge(
        mapping,
        on="split_id",
        how="left",
        validate="many_to_one",
    )
    if attached["outer_repeat"].isna().any():
        missing = sorted(
            attached.loc[
                attached["outer_repeat"].isna(),
                "split_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            f"{label}: split_id values absent from repeat_map: {missing[:20]}"
        )
    attached["outer_repeat"] = attached["outer_repeat"].astype(int)
    return attached


def hyperparameter_outputs(
    scheme011_path: str | Path,
    scheme012_path: str | Path,
    repeat_map: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scheme011 = normalize_hyperparameter_columns(
        read_csv(scheme011_path, "Scheme 011 hyperparameters"),
        "Scheme 011 hyperparameters",
    )
    scheme012 = normalize_hyperparameter_columns(
        read_csv(scheme012_path, "Scheme 012 hyperparameters"),
        "Scheme 012 hyperparameters",
    )

    selected = pd.concat([scheme011, scheme012], ignore_index=True)
    selected = selected.loc[selected["model"].isin(TARGET_MODELS)].copy()
    selected = attach_and_validate_outer_repeat(
        selected,
        repeat_map,
        "combined hyperparameter data",
    )

    if selected.duplicated(["split_id", "model"]).any():
        raise ValueError("Duplicate split/model rows found in hyperparameters")
    counts = selected.groupby("model")["split_id"].nunique()
    if not (counts == 100).all():
        raise ValueError(f"Every target model must have 100 hyperparameter rows:\n{counts}")

    selected = selected.sort_values(["model", "outer_repeat"]).reset_index(drop=True)

    joint = (
        selected.groupby(["model", "alpha", "lambda"], as_index=False)
        .size()
        .rename(columns={"size": "selection_count"})
    )
    joint["selection_frequency"] = (
        joint["selection_count"]
        / joint.groupby("model")["selection_count"].transform("sum")
    )
    joint = joint.sort_values(
        ["model", "selection_count", "lambda", "alpha"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    alpha_frequency = (
        selected.groupby(["model", "alpha"], as_index=False)
        .size()
        .rename(columns={"size": "selection_count"})
    )
    alpha_frequency["selection_frequency"] = (
        alpha_frequency["selection_count"]
        / alpha_frequency.groupby("model")["selection_count"].transform("sum")
    )

    lambda_frequency = (
        selected.groupby(["model", "lambda"], as_index=False)
        .size()
        .rename(columns={"size": "selection_count"})
    )
    lambda_frequency["selection_frequency"] = (
        lambda_frequency["selection_count"]
        / lambda_frequency.groupby("model")["selection_count"].transform("sum")
    )

    return selected, joint, alpha_frequency, lambda_frequency


def coefficient_outputs(
    scheme011_path: str | Path,
    scheme012_path: str | Path,
    repeat_map: pd.DataFrame,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    coefficients = pd.concat(
        [
            read_csv(scheme011_path, "Scheme 011 coefficients"),
            read_csv(scheme012_path, "Scheme 012 coefficients"),
        ],
        ignore_index=True,
    )
    require_columns(
        coefficients,
        ("split_id", "model", "feature_name", "coefficient"),
        "combined coefficient data",
    )

    coefficients["split_id"] = coefficients["split_id"].astype(str)
    coefficients["model"] = coefficients["model"].astype(str)
    coefficients["feature_name"] = coefficients["feature_name"].astype(str)
    coefficients["coefficient"] = pd.to_numeric(
        coefficients["coefficient"], errors="raise"
    )
    coefficients = coefficients.loc[
        coefficients["model"].isin(TARGET_MODELS)
    ].copy()

    if coefficients.duplicated(["split_id", "model", "feature_name"]).any():
        raise ValueError("Duplicate split/model/feature rows found in coefficients")

    predictors = coefficients.loc[
        coefficients["feature_name"] != "__INTERCEPT__"
    ].copy()
    predictors["is_nonzero"] = predictors["coefficient"].abs() > float(tolerance)

    observed = (
        predictors.groupby(["model", "split_id"], as_index=False)
        .agg(
            coefficient_rows_observed=("feature_name", "size"),
            nonzero_coefficient_count=("is_nonzero", "sum"),
        )
    )

    split_ids = repeat_map.sort_values("outer_repeat")["split_id"].tolist()
    complete_grid = pd.MultiIndex.from_product(
        [TARGET_MODELS, split_ids], names=["model", "split_id"]
    ).to_frame(index=False)

    counts = complete_grid.merge(
        observed,
        on=["model", "split_id"],
        how="left",
        validate="one_to_one",
    )
    counts["coefficient_rows_observed"] = (
        counts["coefficient_rows_observed"].fillna(0).astype(int)
    )
    counts["nonzero_coefficient_count"] = (
        counts["nonzero_coefficient_count"].fillna(0).astype(int)
    )
    counts = counts.merge(
        repeat_map,
        on="split_id",
        how="left",
        validate="many_to_one",
    )
    counts["outer_repeat"] = counts["outer_repeat"].astype(int)
    counts["expected_predictor_count"] = counts["model"].map(
        EXPECTED_PREDICTOR_COUNTS
    )
    counts["nonzero_fraction_of_expected"] = (
        counts["nonzero_coefficient_count"]
        / counts["expected_predictor_count"]
    )
    counts = counts.sort_values(["model", "outer_repeat"]).reset_index(drop=True)

    warnings: list[str] = []
    for model, expected in EXPECTED_PREDICTOR_COUNTS.items():
        observed_counts = sorted(
            counts.loc[
                counts["model"] == model, "coefficient_rows_observed"
            ].unique().tolist()
        )
        if observed_counts != [expected]:
            warnings.append(
                f"{model}: coefficient_rows_observed={observed_counts}, expected={expected}. "
                "The coefficient file may contain only a subset of features."
            )

    summary_rows: list[dict] = []
    for model, group in counts.groupby("model", sort=False):
        values = group["nonzero_coefficient_count"].to_numpy(float)
        fractions = group["nonzero_fraction_of_expected"].to_numpy(float)
        summary_rows.append(
            {
                "model": model,
                "n_splits": int(len(values)),
                "expected_predictor_count": int(EXPECTED_PREDICTOR_COUNTS[model]),
                "mean_nonzero_coefficients": float(np.mean(values)),
                "sd_nonzero_coefficients": float(np.std(values, ddof=1)),
                "median_nonzero_coefficients": float(np.median(values)),
                "q025_nonzero_coefficients": float(np.quantile(values, 0.025)),
                "q975_nonzero_coefficients": float(np.quantile(values, 0.975)),
                "minimum_nonzero_coefficients": int(np.min(values)),
                "maximum_nonzero_coefficients": int(np.max(values)),
                "mean_nonzero_fraction": float(np.mean(fractions)),
                "coefficient_tolerance": float(tolerance),
                "intercept_excluded": True,
            }
        )

    return counts, pd.DataFrame(summary_rows), warnings


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_target_metrics(args.combined_metrics)
    repeat_map = (
        metrics[["split_id", "outer_repeat"]]
        .drop_duplicates()
        .sort_values("outer_repeat")
        .reset_index(drop=True)
    )
    if len(repeat_map) != 100:
        raise ValueError(f"Unique split count={len(repeat_map)}, expected=100")

    a9_a3_detail, a9_a3_summary = paired_comparison(metrics, A9, A3)
    a9_a1_detail, a9_a1_summary = paired_comparison(metrics, A9, A1)
    distributions = metric_distribution_summary(metrics)

    selected_hp, joint_hp, alpha_hp, lambda_hp = hyperparameter_outputs(
        args.scheme011_hyperparameters,
        args.scheme012_hyperparameters,
        repeat_map,
    )

    nonzero_by_repeat, nonzero_summary, coefficient_warnings = coefficient_outputs(
        args.scheme011_coefficients,
        args.scheme012_coefficients,
        repeat_map,
        args.coefficient_tolerance,
    )

    outputs = {
        "a9_vs_a3_detail": output_dir / "01_A9_vs_A3_paired_by_repeat.csv",
        "a9_vs_a3_summary": output_dir / "02_A9_vs_A3_paired_summary.csv",
        "a9_vs_a1_detail": output_dir / "03_A9_vs_A1_paired_by_repeat.csv",
        "a9_vs_a1_summary": output_dir / "04_A9_vs_A1_paired_summary.csv",
        "metric_distribution": output_dir / "05_A1_A3_A9_metric_distribution_summary.csv",
        "hyperparameters_by_repeat": output_dir / "06_A1_A3_A9_hyperparameters_by_repeat.csv",
        "alpha_lambda_frequency": output_dir / "07_A1_A3_A9_alpha_lambda_joint_frequency.csv",
        "alpha_frequency": output_dir / "08_A1_A3_A9_alpha_frequency.csv",
        "lambda_frequency": output_dir / "09_A1_A3_A9_lambda_frequency.csv",
        "nonzero_by_repeat": output_dir / "10_A1_A3_A9_nonzero_coefficients_by_repeat.csv",
        "nonzero_summary": output_dir / "11_A1_A3_A9_nonzero_coefficient_summary.csv",
        "audit": output_dir / "12_A1_A3_A9_diagnostics_audit.json",
    }

    a9_a3_detail.to_csv(outputs["a9_vs_a3_detail"], index=False)
    a9_a3_summary.to_csv(outputs["a9_vs_a3_summary"], index=False)
    a9_a1_detail.to_csv(outputs["a9_vs_a1_detail"], index=False)
    a9_a1_summary.to_csv(outputs["a9_vs_a1_summary"], index=False)
    distributions.to_csv(outputs["metric_distribution"], index=False)
    selected_hp.to_csv(outputs["hyperparameters_by_repeat"], index=False)
    joint_hp.to_csv(outputs["alpha_lambda_frequency"], index=False)
    alpha_hp.to_csv(outputs["alpha_frequency"], index=False)
    lambda_hp.to_csv(outputs["lambda_frequency"], index=False)
    nonzero_by_repeat.to_csv(outputs["nonzero_by_repeat"], index=False)
    nonzero_summary.to_csv(outputs["nonzero_summary"], index=False)

    audit = {
        "status": "COMPLETE",
        "models": list(TARGET_MODELS),
        "n_splits_per_model": 100,
        "difference_definition": "A9 minus reference model within the same frozen split",
        "coefficient_tolerance": float(args.coefficient_tolerance),
        "intercept_excluded_from_nonzero_count": True,
        "coefficient_warnings": coefficient_warnings,
        "outputs": {
            key: str(path) for key, path in outputs.items() if key != "audit"
        },
    }
    outputs["audit"].write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    paired_columns = [
        "metric",
        "mean_difference_focal_minus_reference",
        "sd_difference",
        "q025_difference",
        "q975_difference",
        "focal_win_frequency",
        "tie_frequency",
        "focal_loss_frequency",
    ]

    print("A1/A3/A9 diagnostics: COMPLETE")
    print(f"Output directory: {output_dir}")
    print("\nA9 minus A3 paired summary:")
    print(a9_a3_summary[paired_columns].round(4).to_string(index=False))
    print("\nA9 minus A1 paired summary:")
    print(a9_a1_summary[paired_columns].round(4).to_string(index=False))
    print("\nA1/A3/A9 metric distributions:")
    print(
        distributions[
            ["model", "metric", "mean", "sd", "q025", "q975"]
        ].round(4).to_string(index=False)
    )
    print("\nTop alpha/lambda combinations per model:")
    print(
        joint_hp.groupby("model", sort=False)
        .head(10)
        .round(4)
        .to_string(index=False)
    )
    print("\nNonzero coefficient summary; intercept excluded:")
    print(nonzero_summary.round(4).to_string(index=False))

    if coefficient_warnings:
        print("\nCoefficient warnings:")
        for warning in coefficient_warnings:
            print(f"  - {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
