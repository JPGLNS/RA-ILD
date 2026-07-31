#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare A9/A3 models with their public-addon counterparts over 100 repeats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


A3 = "A3_core83_unweighted500"
A9 = "A9_core83_unweighted200"
A13 = "A13_core83_unweighted200_public"
A14 = "A14_core83_unweighted500_public"

METRICS = (
    "roc_auc",
    "pr_auc",
    "accuracy",
    "sensitivity_recall",
    "specificity",
    "precision",
    "f1",
)

PAIRS = (
    (A9, A13, "A13_minus_A9"),
    (A3, A14, "A14_minus_A3"),
    (A14, A13, "A13_minus_A14"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine 14-model metrics with Scheme 013 and test public add-ons."
    )
    parser.add_argument("--combined14-metrics", required=True)
    parser.add_argument("--scheme013-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-repeats", type=int, default=100)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def load_metrics(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    frame = pd.read_csv(path)
    require_columns(
        frame,
        ("split_set_id", "split_id", "outer_repeat", "model", *METRICS),
        label,
    )
    frame["split_set_id"] = frame["split_set_id"].astype(str)
    frame["split_id"] = frame["split_id"].astype(str)
    frame["model"] = frame["model"].astype(str)
    frame["outer_repeat"] = pd.to_numeric(
        frame["outer_repeat"], errors="raise"
    ).astype(int)
    for metric in METRICS:
        frame[metric] = pd.to_numeric(frame[metric], errors="raise")
        if not np.isfinite(frame[metric].to_numpy(float)).all():
            raise ValueError(f"{label} contains non-finite values in {metric}")
    if frame.duplicated(["split_id", "model"]).any():
        raise ValueError(f"{label} contains duplicate split/model rows")
    return frame


def distribution_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in frame.groupby("model", sort=False):
        for metric in METRICS:
            values = group[metric].to_numpy(float)
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "n_repeats": int(len(values)),
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


def ranking(summary: pd.DataFrame) -> pd.DataFrame:
    wide = summary.pivot(index="model", columns="metric", values="mean").reset_index()
    wide = wide.sort_values(
        ["roc_auc", "pr_auc", "f1", "specificity", "sensitivity_recall", "model"],
        ascending=[False, False, False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    wide.insert(0, "descriptive_rank", np.arange(1, len(wide) + 1))
    wide["ranking_rule"] = (
        "mean_roc_auc_then_mean_pr_auc_then_mean_f1_descriptive_only"
    )
    return wide


def paired_outputs(
    frame: pd.DataFrame,
    reference: str,
    focal: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = frame.loc[
        frame["model"] == reference,
        ["split_set_id", "split_id", "outer_repeat", *METRICS],
    ].copy()
    foc = frame.loc[
        frame["model"] == focal,
        ["split_set_id", "split_id", "outer_repeat", *METRICS],
    ].copy()

    paired = ref.merge(
        foc,
        on=["split_set_id", "split_id", "outer_repeat"],
        how="inner",
        suffixes=("_reference", "_focal"),
        validate="one_to_one",
    )
    if paired.empty:
        raise ValueError(f"No paired rows for {focal} versus {reference}")

    summaries = []
    for metric in METRICS:
        difference = (
            paired[f"{metric}_focal"].to_numpy(float)
            - paired[f"{metric}_reference"].to_numpy(float)
        )
        paired[f"{metric}_difference_focal_minus_reference"] = difference
        paired[f"{metric}_result"] = np.where(
            difference > 0,
            "focal_win",
            np.where(
                np.isclose(difference, 0.0, atol=1e-15, rtol=0.0),
                "tie",
                "focal_loss",
            ),
        )
        summaries.append(
            {
                "reference_model": reference,
                "focal_model": focal,
                "metric": metric,
                "n_paired_repeats": int(len(difference)),
                "mean_difference_focal_minus_reference": float(
                    np.mean(difference)
                ),
                "sd_difference": float(np.std(difference, ddof=1)),
                "median_difference": float(np.median(difference)),
                "q025_difference": float(np.quantile(difference, 0.025)),
                "q975_difference": float(np.quantile(difference, 0.975)),
                "focal_win_count": int(np.sum(difference > 0)),
                "focal_win_frequency": float(np.mean(difference > 0)),
                "tie_count": int(
                    np.sum(np.isclose(difference, 0.0, atol=1e-15, rtol=0.0))
                ),
                "tie_frequency": float(
                    np.mean(np.isclose(difference, 0.0, atol=1e-15, rtol=0.0))
                ),
                "focal_loss_count": int(np.sum(difference < 0)),
                "focal_loss_frequency": float(np.mean(difference < 0)),
            }
        )
    paired.insert(0, "reference_model", reference)
    paired.insert(1, "focal_model", focal)
    return paired, pd.DataFrame(summaries)


def main() -> int:
    args = parse_args()
    old_path = Path(args.combined14_metrics).expanduser().resolve()
    new_path = Path(args.scheme013_metrics).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    old = load_metrics(old_path, "combined 14-model metrics")
    new = load_metrics(new_path, "Scheme 013 metrics")

    expected = int(args.expected_repeats)
    old_models = set(old["model"])
    new_models = set(new["model"])

    required_old = {A3, A9}
    required_new = {A13, A14}
    if not required_old.issubset(old_models):
        raise ValueError(f"Old metrics missing models: {sorted(required_old - old_models)}")
    if new_models != required_new:
        raise ValueError(
            f"Scheme 013 must contain exactly {sorted(required_new)}; "
            f"observed {sorted(new_models)}"
        )

    if set(old["split_set_id"]) != set(new["split_set_id"]):
        raise ValueError("Old and new metrics use different split_set_id values")
    if set(old["split_id"]) != set(new["split_id"]):
        raise ValueError("Old and new metrics contain different frozen split IDs")

    for label, frame in (("old", old), ("new", new)):
        counts = frame.groupby("model")["split_id"].nunique()
        if not (counts == expected).all():
            raise ValueError(f"{label} model coverage is not {expected} repeats:\n{counts}")

    old = old.copy()
    new = new.copy()
    if "source_experiment" not in old.columns:
        old.insert(0, "source_experiment", "scheme011_012")
    if "source_experiment" not in new.columns:
        new.insert(0, "source_experiment", "scheme013")

    combined = pd.concat([old, new], ignore_index=True, sort=False)
    if combined["model"].nunique() != 16:
        raise ValueError(
            f"Expected 16 models after combination; observed "
            f"{combined['model'].nunique()}"
        )
    if len(combined) != 16 * expected:
        raise ValueError(
            f"Expected {16 * expected} metric rows; observed {len(combined)}"
        )
    if combined.duplicated(["split_id", "model"]).any():
        raise ValueError("Combined metrics contain duplicate split/model rows")

    summary = distribution_summary(combined)
    rank = ranking(summary)

    focus = combined.loc[
        combined["model"].isin([A3, A9, A13, A14])
    ].copy()
    focus_summary = distribution_summary(focus)

    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / "01_16model_holdout_task_metrics.csv", index=False)
    summary.to_csv(output_dir / "02_16model_metric_summary.csv", index=False)
    rank.to_csv(output_dir / "03_16model_ranking.csv", index=False)
    focus_summary.to_csv(
        output_dir / "04_A3_A9_A13_A14_metric_distribution.csv", index=False
    )

    pair_files = {}
    all_pair_summaries = []
    for reference, focal, label in PAIRS:
        by_repeat, pair_summary = paired_outputs(focus, reference, focal)
        by_path = output_dir / f"05_{label}_by_repeat.csv"
        summary_path = output_dir / f"06_{label}_summary.csv"
        by_repeat.to_csv(by_path, index=False)
        pair_summary.to_csv(summary_path, index=False)
        pair_files[label] = {
            "by_repeat": str(by_path),
            "summary": str(summary_path),
        }
        all_pair_summaries.append(pair_summary)

    all_pairs = pd.concat(all_pair_summaries, ignore_index=True)
    all_pairs.to_csv(
        output_dir / "07_all_public_addon_paired_summaries.csv", index=False
    )

    audit = {
        "status": "COMPLETE",
        "expected_repeats": expected,
        "combined_model_count": int(combined["model"].nunique()),
        "combined_metric_rows": int(len(combined)),
        "models_of_interest": [A3, A9, A13, A14],
        "comparisons": pair_files,
    }
    (output_dir / "08_public_addon_comparison_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("TRB public-addon comparison: COMPLETE")
    print(f"Output directory: {output_dir}")
    print()
    print("16-model descriptive ranking:")
    print(
        rank[
            [
                "descriptive_rank",
                "model",
                "roc_auc",
                "pr_auc",
                "sensitivity_recall",
                "specificity",
                "f1",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print()
    print("Public-addon paired summaries:")
    print(
        all_pairs[
            [
                "reference_model",
                "focal_model",
                "metric",
                "mean_difference_focal_minus_reference",
                "q025_difference",
                "q975_difference",
                "focal_win_frequency",
                "tie_frequency",
                "focal_loss_frequency",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
