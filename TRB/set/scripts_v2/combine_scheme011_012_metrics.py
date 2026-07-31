#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


METRICS = (
    "roc_auc",
    "pr_auc",
    "accuracy",
    "sensitivity_recall",
    "specificity",
    "precision",
    "f1",
)

CHECKPOINTS = (10, 20, 30, 50, 75, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine Scheme 011 and Scheme 012 repeated-holdout "
            "metrics into one 14-model comparison."
        )
    )
    parser.add_argument("--scheme011", required=True)
    parser.add_argument("--scheme012", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    label: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(
            f"{label} is missing columns: {missing}"
        )


def load_metrics(path: Path, label: str) -> pd.DataFrame:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")

    frame = pd.read_csv(path)

    require_columns(
        frame,
        (
            "split_set_id",
            "split_id",
            "task_id",
            "outer_repeat",
            "outer_fold",
            "model",
            *METRICS,
        ),
        label,
    )

    frame["split_set_id"] = frame["split_set_id"].astype(str)
    frame["split_id"] = frame["split_id"].astype(str)
    frame["task_id"] = frame["task_id"].astype(str)
    frame["model"] = frame["model"].astype(str)

    frame["outer_repeat"] = pd.to_numeric(
        frame["outer_repeat"],
        errors="raise",
    ).astype(int)

    frame["outer_fold"] = pd.to_numeric(
        frame["outer_fold"],
        errors="raise",
    ).astype(int)

    for metric in METRICS:
        frame[metric] = pd.to_numeric(
            frame[metric],
            errors="raise",
        )

        if not np.isfinite(frame[metric].to_numpy(float)).all():
            raise ValueError(
                f"{label} contains non-finite {metric} values"
            )

    if frame.duplicated(["split_id", "model"]).any():
        duplicates = frame.loc[
            frame.duplicated(
                ["split_id", "model"],
                keep=False,
            ),
            ["split_id", "model"],
        ]

        raise ValueError(
            f"{label} contains duplicate split/model rows:\n"
            f"{duplicates.head(20)}"
        )

    return frame


def metric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for model, group in frame.groupby(
        "model",
        sort=False,
    ):
        for metric in METRICS:
            values = group[metric].to_numpy(float)

            rows.append(
                {
                    "model": str(model),
                    "metric": metric,
                    "n_splits": int(len(values)),
                    "mean": float(np.mean(values)),
                    "sd": (
                        float(np.std(values, ddof=1))
                        if len(values) > 1
                        else 0.0
                    ),
                    "median": float(np.median(values)),
                    "q025": float(
                        np.quantile(values, 0.025)
                    ),
                    "q25": float(
                        np.quantile(values, 0.25)
                    ),
                    "q75": float(
                        np.quantile(values, 0.75)
                    ),
                    "q975": float(
                        np.quantile(values, 0.975)
                    ),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )

    return pd.DataFrame(rows)


def model_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    wide = (
        summary
        .pivot(
            index="model",
            columns="metric",
            values="mean",
        )
        .reset_index()
    )

    for metric in METRICS:
        if metric not in wide.columns:
            wide[metric] = np.nan

    wide["rank_mean_roc_auc"] = (
        wide["roc_auc"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    wide["rank_mean_pr_auc"] = (
        wide["pr_auc"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    wide = (
        wide
        .sort_values(
            [
                "roc_auc",
                "pr_auc",
                "f1",
                "specificity",
                "sensitivity_recall",
                "model",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                False,
                True,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    wide.insert(
        0,
        "descriptive_rank",
        np.arange(1, len(wide) + 1),
    )

    wide["ranking_rule"] = (
        "mean_roc_auc_then_mean_pr_auc_then_mean_f1_"
        "descriptive_only"
    )

    wide["automatic_final_model_selection"] = False

    return wide


def pairwise_differences(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    models = list(
        dict.fromkeys(frame["model"].astype(str))
    )

    rows = []

    for reference, comparison in itertools.combinations(
        models,
        2,
    ):
        reference_rows = (
            frame.loc[
                frame["model"] == reference,
                ["split_id", *METRICS],
            ]
            .copy()
        )

        comparison_rows = (
            frame.loc[
                frame["model"] == comparison,
                ["split_id", *METRICS],
            ]
            .copy()
        )

        paired = reference_rows.merge(
            comparison_rows,
            on="split_id",
            how="inner",
            suffixes=("_reference", "_comparison"),
            validate="one_to_one",
        )

        if len(paired) != 100:
            raise ValueError(
                f"{reference} vs {comparison} has "
                f"{len(paired)} paired splits, expected 100"
            )

        for metric in METRICS:
            difference = (
                paired[f"{metric}_comparison"].to_numpy(float)
                - paired[f"{metric}_reference"].to_numpy(float)
            )

            rows.append(
                {
                    "reference_model": reference,
                    "comparison_model": comparison,
                    "metric": metric,
                    "n_paired_splits": int(len(difference)),
                    "mean_difference_comparison_minus_reference": (
                        float(np.mean(difference))
                    ),
                    "sd_difference": float(
                        np.std(difference, ddof=1)
                    ),
                    "median_difference": float(
                        np.median(difference)
                    ),
                    "comparison_win_frequency": float(
                        np.mean(difference > 0)
                    ),
                    "tie_frequency": float(
                        np.mean(
                            np.isclose(
                                difference,
                                0.0,
                                atol=1.0e-15,
                                rtol=0.0,
                            )
                        )
                    ),
                    "comparison_loss_frequency": float(
                        np.mean(difference < 0)
                    ),
                }
            )

    return pd.DataFrame(rows)


def cumulative_outputs(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_parts = []
    ranking_parts = []

    for checkpoint in CHECKPOINTS:
        subset = frame.loc[
            frame["outer_repeat"] <= checkpoint
        ].copy()

        counts = (
            subset.groupby("model")["outer_repeat"]
            .nunique()
        )

        if not (counts == checkpoint).all():
            raise ValueError(
                f"Checkpoint {checkpoint} has incomplete model coverage:\n"
                f"{counts}"
            )

        summary = metric_summary(subset)
        summary.insert(0, "repeat_checkpoint", checkpoint)
        summary_parts.append(summary)

        ranking = model_ranking(summary)
        ranking.insert(0, "repeat_checkpoint", checkpoint)
        ranking_parts.append(ranking)

    return (
        pd.concat(summary_parts, ignore_index=True),
        pd.concat(ranking_parts, ignore_index=True),
    )


def main() -> int:
    args = parse_args()

    scheme011_path = Path(args.scheme011).expanduser().resolve()
    scheme012_path = Path(args.scheme012).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    scheme011 = load_metrics(
        scheme011_path,
        "Scheme 011 metrics",
    )

    scheme012 = load_metrics(
        scheme012_path,
        "Scheme 012 metrics",
    )

    scheme011_models = set(
        scheme011["model"].unique()
    )
    scheme012_models = set(
        scheme012["model"].unique()
    )

    overlap = sorted(
        scheme011_models & scheme012_models
    )

    if overlap:
        raise ValueError(
            f"Model names overlap between experiments: {overlap}"
        )

    if len(scheme011_models) != 8:
        raise ValueError(
            f"Scheme 011 contains {len(scheme011_models)} "
            "models, expected 8"
        )

    if len(scheme012_models) != 6:
        raise ValueError(
            f"Scheme 012 contains {len(scheme012_models)} "
            "models, expected 6"
        )

    split_sets_011 = set(
        scheme011["split_set_id"]
    )
    split_sets_012 = set(
        scheme012["split_set_id"]
    )

    if split_sets_011 != split_sets_012:
        raise ValueError(
            "Scheme 011 and Scheme 012 use different "
            f"split sets: {split_sets_011} vs {split_sets_012}"
        )

    splits_011 = set(
        scheme011["split_id"]
    )
    splits_012 = set(
        scheme012["split_id"]
    )

    if splits_011 != splits_012:
        raise ValueError(
            "Scheme 011 and Scheme 012 do not contain "
            "the same frozen split IDs"
        )

    repeats_011 = set(
        scheme011["outer_repeat"]
    )
    repeats_012 = set(
        scheme012["outer_repeat"]
    )

    expected_repeats = set(range(1, 101))

    if repeats_011 != expected_repeats:
        raise ValueError(
            "Scheme 011 outer_repeat values are not 1..100"
        )

    if repeats_012 != expected_repeats:
        raise ValueError(
            "Scheme 012 outer_repeat values are not 1..100"
        )

    scheme011 = scheme011.copy()
    scheme012 = scheme012.copy()

    scheme011.insert(0, "source_experiment", "scheme011")
    scheme012.insert(0, "source_experiment", "scheme012")

    combined = pd.concat(
        [scheme011, scheme012],
        ignore_index=True,
    )

    if len(combined) != 1400:
        raise ValueError(
            f"Combined metric rows={len(combined)}, expected 1400"
        )

    if combined["model"].nunique() != 14:
        raise ValueError(
            "Combined result does not contain 14 unique models"
        )

    if combined.duplicated(
        ["split_id", "model"]
    ).any():
        raise ValueError(
            "Combined result contains duplicate split/model rows"
        )

    model_counts = (
        combined.groupby("model")["split_id"]
        .nunique()
    )

    if not (model_counts == 100).all():
        raise ValueError(
            f"Not every model has 100 splits:\n{model_counts}"
        )

    summary = metric_summary(combined)
    ranking = model_ranking(summary)
    pairwise = pairwise_differences(combined)

    cumulative_summary, cumulative_ranking = (
        cumulative_outputs(combined)
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "combined_metrics": (
            output_dir
            / "14model_holdout_task_metrics.csv"
        ),
        "metric_summary": (
            output_dir
            / "14model_metric_summary.csv"
        ),
        "ranking": (
            output_dir
            / "14model_ranking.csv"
        ),
        "pairwise": (
            output_dir
            / "14model_pairwise_metric_differences.csv"
        ),
        "cumulative_summary": (
            output_dir
            / "14model_cumulative_metric_summary.csv"
        ),
        "cumulative_ranking": (
            output_dir
            / "14model_cumulative_ranking.csv"
        ),
        "audit": (
            output_dir
            / "14model_merge_audit.json"
        ),
    }

    combined.to_csv(
        paths["combined_metrics"],
        index=False,
    )

    summary.to_csv(
        paths["metric_summary"],
        index=False,
    )

    ranking.to_csv(
        paths["ranking"],
        index=False,
    )

    pairwise.to_csv(
        paths["pairwise"],
        index=False,
    )

    cumulative_summary.to_csv(
        paths["cumulative_summary"],
        index=False,
    )

    cumulative_ranking.to_csv(
        paths["cumulative_ranking"],
        index=False,
    )

    audit = {
        "status": "COMPLETE",
        "scheme011_metrics": str(scheme011_path),
        "scheme012_metrics": str(scheme012_path),
        "split_set_ids": sorted(split_sets_011),
        "split_count": 100,
        "scheme011_model_count": 8,
        "scheme012_model_count": 6,
        "combined_model_count": 14,
        "combined_metric_rows": int(len(combined)),
        "checkpoints": list(CHECKPOINTS),
        "outputs": {
            key: str(value)
            for key, value in paths.items()
            if key != "audit"
        },
    }

    paths["audit"].write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("14-model metric combination: COMPLETE")
    print(f"Scheme 011 models: {len(scheme011_models)}")
    print(f"Scheme 012 models: {len(scheme012_models)}")
    print(f"Combined models:   {combined['model'].nunique()}")
    print(f"Metric rows:       {len(combined)}")
    print(f"Output directory:  {output_dir}")
    print()
    print(
        ranking[
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
