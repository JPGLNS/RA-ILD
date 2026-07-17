#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Visualize and compare repeated nested-CV performance for paired TRB+IGH models.

The statistically preferred plotting unit is one pooled OOF result per repeat:
20 repeats × 4 models = 80 plotted observations. This matches the paired
Friedman and Wilcoxon tests.

Use --plot-level fold only for a diagnostic view of the 100 outer-fold metrics;
the p-values are still calculated from the 20 paired repeat-level pooled OOF
results to avoid treating dependent folds as independent observations.

The frozen patient-level outer assignments are read to validate the real
23/24-patient validation folds produced by 118 patients split into five folds.
The independent test set is never read.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SCRIPT_VERSION = "1.0.0-PAIRED-TRB-IGH"
ANALYSIS_VIEW = "TRB_IGH"
CV_UNIT = "patient"

ROOT = Path("/data/users/chenhaisheng/RA-ILD/TRB_IGH")
COLLECT = ROOT / "set/train/result/06_cv_result_summary/collected"
OUT = ROOT / "set/train/result/06_cv_result_summary/visualization"
OUTER_ASSIGNMENTS = (
    ROOT / "set/train/result/05_modeling/cv_splits/05_outer_fold_assignments.csv"
)

MODELS = [
    "M0_clinical",
    "M1_static_trb_igh",
    "M2_static_trb_igh_public",
    "M3_static_trb_igh_public_material",
]
MODEL_SET = set(MODELS)

MODEL_LABELS = {
    "M0_clinical": "M0\nClinical",
    "M1_static_trb_igh": "M1\n+ static TRB/IGH",
    "M2_static_trb_igh_public": "M2\n+ dynamic public",
    "M3_static_trb_igh_public_material": "M3\n+ material",
}

MODEL_COLORS = {
    "M0_clinical": "#65c8cc",
    "M1_static_trb_igh": "#f0e94b",
    "M2_static_trb_igh_public": "#72c15a",
    "M3_static_trb_igh_public_material": "#f3793b",
}

METRICS: List[Tuple[str, str]] = [
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
    ("sensitivity_recall", "Sensitivity"),
    ("specificity", "Specificity"),
    ("f1", "F1 score"),
]

PLANNED_PAIRS = [
    ("M0_clinical", "M1_static_trb_igh"),
    ("M1_static_trb_igh", "M2_static_trb_igh_public"),
    ("M2_static_trb_igh_public", "M3_static_trb_igh_public_material"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot and statistically compare paired TRB+IGH repeated nested-CV model metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metrics",
        default=str(COLLECT / "06_all_outer_metrics.csv"),
        help="Collected 100-task outer-fold metric table.",
    )
    parser.add_argument(
        "--predictions",
        default=str(COLLECT / "06_all_outer_predictions.csv.gz"),
        help="Collected outer-validation prediction table.",
    )
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument(
        "--outer-assignments",
        default=str(OUTER_ASSIGNMENTS),
        help="Frozen patient-level outer-fold assignment table.",
    )
    parser.add_argument("--expected-repeats", type=int, default=20)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--expected-samples", type=int, default=118)
    parser.add_argument("--expected-models", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument(
        "--plot-level",
        choices=("repeat", "fold"),
        default="repeat",
        help=(
            "repeat: plot 20 pooled OOF values per model (recommended and aligned "
            "with significance tests); fold: plot 100 outer-fold values per model."
        ),
    )
    parser.add_argument(
        "--annotate-pairs",
        choices=("planned", "all"),
        default="planned",
        help="Pairwise Holm-adjusted p-values shown above each panel.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.expected_repeats < 2:
        parser.error("--expected-repeats must be >= 2.")
    if args.expected_folds < 2:
        parser.error("--expected-folds must be >= 2.")
    if args.expected_samples < 2:
        parser.error("--expected-samples must be >= 2.")
    if args.expected_models != len(MODELS):
        parser.error(
            f"--expected-models must equal the configured model count {len(MODELS)}."
        )
    if args.dpi < 72:
        parser.error("--dpi must be >= 72.")
    return args


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def ensure_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")



def sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_outer_assignments(
    path: Path,
    expected_repeats: int,
    expected_folds: int,
    expected_samples: int,
) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], frozenset[str]]]:
    ensure_file(path, "frozen outer assignments")
    outer = pd.read_csv(path, dtype=str)
    outer = outer.drop(
        columns=[c for c in outer.columns if str(c).startswith("Unnamed:")],
        errors="ignore",
    )
    ensure_columns(
        outer,
        ["outer_repeat", "outer_fold", "sample_id"],
        "frozen outer assignments",
    )

    outer["outer_repeat"] = pd.to_numeric(
        outer["outer_repeat"], errors="raise"
    ).astype(int)
    outer["outer_fold"] = pd.to_numeric(
        outer["outer_fold"], errors="raise"
    ).astype(int)
    outer["sample_id"] = outer["sample_id"].astype(str).str.strip()

    if outer["sample_id"].eq("").any():
        raise ValueError("Frozen outer assignments contain empty sample_id values.")
    if set(outer["outer_repeat"]) != set(range(1, expected_repeats + 1)):
        raise ValueError("Frozen outer repeat IDs are incomplete or unexpected.")
    if set(outer["outer_fold"]) != set(range(1, expected_folds + 1)):
        raise ValueError("Frozen outer fold IDs are incomplete or unexpected.")
    if outer.duplicated(["outer_repeat", "sample_id"]).any():
        raise ValueError(
            "A patient appears more than once within a frozen outer repeat."
        )

    repeat_counts = outer.groupby("outer_repeat")["sample_id"].nunique()
    if not (repeat_counts == expected_samples).all():
        raise ValueError(
            "Frozen outer assignments do not contain exactly "
            f"{expected_samples} patients per repeat: {repeat_counts.to_dict()}"
        )

    repeat_sets = [
        frozenset(group["sample_id"])
        for _, group in outer.groupby("outer_repeat", sort=True)
    ]
    if not repeat_sets or any(current != repeat_sets[0] for current in repeat_sets[1:]):
        raise ValueError("Frozen patient universe differs across outer repeats.")

    expectations: Dict[Tuple[int, int], frozenset[str]] = {}
    for (repeat, fold), group in outer.groupby(
        ["outer_repeat", "outer_fold"], sort=True
    ):
        ids = frozenset(group["sample_id"])
        if not ids:
            raise ValueError(f"Frozen task repeat={repeat}, fold={fold} is empty.")
        expectations[(int(repeat), int(fold))] = ids

    expected_tasks = expected_repeats * expected_folds
    if len(expectations) != expected_tasks:
        raise ValueError(
            f"Expected {expected_tasks} frozen outer tasks, observed {len(expectations)}."
        )

    fold_sizes = np.array([len(ids) for ids in expectations.values()], dtype=int)
    allowed_sizes = {
        expected_samples // expected_folds,
        int(math.ceil(expected_samples / expected_folds)),
    }
    if not set(fold_sizes).issubset(allowed_sizes):
        raise ValueError(
            "Frozen validation fold sizes are unexpected: "
            f"observed={sorted(set(fold_sizes))}, allowed={sorted(allowed_sizes)}"
        )

    return outer.sort_values(
        ["outer_repeat", "outer_fold", "sample_id"]
    ).reset_index(drop=True), expectations


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm family-wise error adjustment, preserving original order."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or len(p) == 0:
        return np.asarray([], dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("Non-finite p-value supplied to Holm adjustment.")

    order = np.argsort(p)
    adjusted_sorted = np.empty(len(p), dtype=float)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted_sorted[rank] = min(running, 1.0)

    out = np.empty(len(p), dtype=float)
    for rank, idx in enumerate(order):
        out[idx] = adjusted_sorted[rank]
    return out


def paired_wilcoxon(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("Paired vectors have different shapes.")
    diff = y - x
    if np.allclose(diff, 0.0, rtol=0.0, atol=1e-15):
        return 0.0, 1.0
    result = wilcoxon(
        x,
        y,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def validate_outer_metrics(
    fold_metrics: pd.DataFrame,
    expected_repeats: int,
    expected_folds: int,
    task_expectations: Dict[Tuple[int, int], frozenset[str]],
) -> pd.DataFrame:
    required = {
        "outer_repeat",
        "outer_fold",
        "model",
        *[metric for metric, _ in METRICS],
    }
    ensure_columns(fold_metrics, required, "outer metrics")

    out = fold_metrics.copy()
    out["outer_repeat"] = pd.to_numeric(out["outer_repeat"], errors="raise").astype(int)
    out["outer_fold"] = pd.to_numeric(out["outer_fold"], errors="raise").astype(int)
    out["model"] = out["model"].astype(str)

    observed_models = set(out["model"])
    if observed_models != MODEL_SET:
        raise ValueError(
            f"Outer metrics model set mismatch: expected {MODELS}, "
            f"observed {sorted(observed_models)}"
        )

    expected_rows = expected_repeats * expected_folds * len(MODELS)
    if len(out) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} outer metric rows, observed {len(out)}."
        )
    if out.duplicated(["outer_repeat", "outer_fold", "model"]).any():
        raise ValueError("Duplicated outer_repeat + outer_fold + model metric rows.")
    if set(out["outer_repeat"]) != set(range(1, expected_repeats + 1)):
        raise ValueError("Outer metrics repeat IDs are incomplete or unexpected.")
    if set(out["outer_fold"]) != set(range(1, expected_folds + 1)):
        raise ValueError("Outer metrics fold IDs are incomplete or unexpected.")

    task_counts = out.groupby(["outer_repeat", "outer_fold"])["model"].nunique()
    if not (task_counts == len(MODELS)).all():
        raise ValueError("One or more outer tasks do not contain all four models.")

    for metric, _ in METRICS:
        values = pd.to_numeric(out[metric], errors="raise")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"Outer metrics contain missing/non-finite {metric}.")
        numeric = values.to_numpy(dtype=float)
        tolerance = 1e-12
        if np.any(numeric < -tolerance) or np.any(numeric > 1.0 + tolerance):
            raise ValueError(f"Outer metric {metric} contains values outside [0, 1].")
        out[metric] = np.clip(numeric, 0.0, 1.0)

    if "fit_converged" in out.columns and not as_bool(out["fit_converged"]).all():
        raise ValueError("One or more final outer models did not converge.")

    if "n_outer_validation" in out.columns:
        valid_n = pd.to_numeric(out["n_outer_validation"], errors="raise").astype(int)
        expected_valid_n = np.array(
            [
                len(task_expectations[(int(r), int(f))])
                for r, f in zip(out["outer_repeat"], out["outer_fold"])
            ],
            dtype=int,
        )
        if not np.array_equal(valid_n.to_numpy(), expected_valid_n):
            raise ValueError("Outer metrics n_outer_validation differs from frozen folds.")

    if "n_outer_train" in out.columns:
        train_n = pd.to_numeric(out["n_outer_train"], errors="raise").astype(int)
        total_patients = sum(len(ids) for (r, _), ids in task_expectations.items() if r == 1)
        expected_train_n = np.array(
            [
                total_patients - len(task_expectations[(int(r), int(f))])
                for r, f in zip(out["outer_repeat"], out["outer_fold"])
            ],
            dtype=int,
        )
        if not np.array_equal(train_n.to_numpy(), expected_train_n):
            raise ValueError("Outer metrics n_outer_train differs from frozen folds.")

    return out.sort_values(["outer_repeat", "outer_fold", "model"]).reset_index(drop=True)


def validate_predictions(
    predictions: pd.DataFrame,
    expected_repeats: int,
    expected_folds: int,
    expected_samples: int,
    task_expectations: Dict[Tuple[int, int], frozenset[str]],
) -> pd.DataFrame:
    required = {
        "outer_repeat",
        "outer_fold",
        "model",
        "sample_id",
        "true_label",
        "probability_ILD",
        "predicted_label",
    }
    ensure_columns(predictions, required, "outer predictions")

    out = predictions.copy()
    out["outer_repeat"] = pd.to_numeric(out["outer_repeat"], errors="raise").astype(int)
    out["outer_fold"] = pd.to_numeric(out["outer_fold"], errors="raise").astype(int)
    out["model"] = out["model"].astype(str)
    out["sample_id"] = out["sample_id"].astype(str).str.strip()
    out["true_label"] = pd.to_numeric(out["true_label"], errors="raise").astype(int)
    out["predicted_label"] = pd.to_numeric(
        out["predicted_label"], errors="raise"
    ).astype(int)
    out["probability_ILD"] = pd.to_numeric(
        out["probability_ILD"], errors="raise"
    ).astype(float)

    observed_models = set(out["model"])
    if observed_models != MODEL_SET:
        raise ValueError(
            f"Prediction model set mismatch: expected {MODELS}, "
            f"observed {sorted(observed_models)}"
        )
    if set(out["outer_repeat"]) != set(range(1, expected_repeats + 1)):
        raise ValueError("Prediction repeat IDs are incomplete or unexpected.")
    if set(out["outer_fold"]) != set(range(1, expected_folds + 1)):
        raise ValueError("Prediction fold IDs are incomplete or unexpected.")
    if out.duplicated(["outer_repeat", "model", "sample_id"]).any():
        raise ValueError(
            "A patient has duplicate OOF predictions within a repeat and model."
        )

    if not out["true_label"].isin([0, 1]).all():
        raise ValueError("true_label must contain only 0/1.")
    if not out["predicted_label"].isin([0, 1]).all():
        raise ValueError("predicted_label must contain only 0/1.")
    probability_values = out["probability_ILD"].to_numpy(dtype=float)
    if not np.isfinite(probability_values).all():
        raise ValueError("probability_ILD contains missing/non-finite values.")
    tolerance = 1e-12
    if np.any(probability_values < -tolerance) or np.any(
        probability_values > 1.0 + tolerance
    ):
        raise ValueError("probability_ILD contains values outside [0, 1].")
    out["probability_ILD"] = np.clip(probability_values, 0.0, 1.0)

    expected_prediction_rows = expected_repeats * expected_samples * len(MODELS)
    if len(out) != expected_prediction_rows:
        raise ValueError(
            f"Expected {expected_prediction_rows:,} prediction rows, "
            f"observed {len(out):,}."
        )

    counts = out.groupby(["outer_repeat", "model"])["sample_id"].nunique()
    bad_counts = counts[counts != expected_samples]
    if not bad_counts.empty:
        raise ValueError(
            "Each repeat/model must contain exactly "
            f"{expected_samples} unique OOF patients: {bad_counts.to_dict()}"
        )

    for (repeat, fold, model), group in out.groupby(
        ["outer_repeat", "outer_fold", "model"], sort=True
    ):
        key = (int(repeat), int(fold))
        if key not in task_expectations:
            raise ValueError(f"Prediction task {key} is absent from frozen assignments.")
        observed_ids = frozenset(group["sample_id"])
        expected_ids = task_expectations[key]
        if observed_ids != expected_ids:
            missing = sorted(expected_ids - observed_ids)[:10]
            extra = sorted(observed_ids - expected_ids)[:10]
            raise ValueError(
                "Prediction patients differ from frozen validation patients for "
                f"repeat={repeat}, fold={fold}, model={model}; "
                f"missing={missing}, extra={extra}"
            )

    label_consistency = out.groupby(["outer_repeat", "sample_id"])[
        "true_label"
    ].nunique()
    if (label_consistency != 1).any():
        raise ValueError("A patient has inconsistent true labels across models.")

    sample_sets = {
        (repeat, model): frozenset(group["sample_id"])
        for (repeat, model), group in out.groupby(["outer_repeat", "model"])
    }
    reference_set = sample_sets[(1, MODELS[0])]
    for key, sample_set in sample_sets.items():
        if sample_set != reference_set:
            raise ValueError(
                f"OOF patient set for repeat/model {key} differs from the fixed cohort."
            )

    return out.sort_values(
        ["outer_repeat", "outer_fold", "model", "sample_id"]
    ).reset_index(drop=True)


def build_repeat_level_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for (repeat, model), group in predictions.groupby(
        ["outer_repeat", "model"], sort=True
    ):
        if group["sample_id"].duplicated().any():
            raise ValueError(
                f"Duplicate OOF sample: repeat={repeat}, model={model}."
            )

        y_true = group["true_label"].to_numpy(dtype=int)
        probability = group["probability_ILD"].to_numpy(dtype=float)
        predicted = group["predicted_label"].to_numpy(dtype=int)

        if len(np.unique(y_true)) != 2:
            raise ValueError(
                f"Both classes are required: repeat={repeat}, model={model}."
            )

        tn, fp, fn, tp = confusion_matrix(
            y_true, predicted, labels=[0, 1]
        ).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else np.nan

        rows.append(
            {
                "outer_repeat": int(repeat),
                "model": str(model),
                "n_samples": int(len(group)),
                "n_RA": int(np.sum(y_true == 0)),
                "n_ILD": int(np.sum(y_true == 1)),
                "roc_auc": float(roc_auc_score(y_true, probability)),
                "pr_auc": float(average_precision_score(y_true, probability)),
                "sensitivity_recall": float(
                    recall_score(y_true, predicted, zero_division=0)
                ),
                "specificity": float(specificity),
                "precision": float(
                    precision_score(y_true, predicted, zero_division=0)
                ),
                "f1": float(f1_score(y_true, predicted, zero_division=0)),
                "accuracy": float(accuracy_score(y_true, predicted)),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
            }
        )

    result = pd.DataFrame(rows).sort_values(
        ["outer_repeat", "model"]
    ).reset_index(drop=True)

    expected_rows = predictions["outer_repeat"].nunique() * len(MODELS)
    if len(result) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} repeat-level rows, observed {len(result)}."
        )
    return result


def safe_friedman(matrix: pd.DataFrame) -> Tuple[float, float]:
    values = matrix.to_numpy(dtype=float)
    if values.shape[0] < 2:
        raise ValueError("At least two paired repeats are required.")
    if np.allclose(values, values[:, [0]], rtol=0.0, atol=1e-15):
        return 0.0, 1.0

    with np.errstate(invalid="ignore"):
        result = friedmanchisquare(
            *[matrix[model].to_numpy(dtype=float) for model in MODELS]
        )
    statistic = float(result.statistic)
    p_value = float(result.pvalue)
    if not math.isfinite(statistic) or not math.isfinite(p_value):
        # scipy may return nan when every model is tied at every repeat.
        return 0.0, 1.0
    return statistic, p_value


def run_paired_tests(
    repeat_metrics: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    global_rows: List[Dict[str, object]] = []
    pair_rows: List[Dict[str, object]] = []

    for metric, metric_label in METRICS:
        wide = repeat_metrics.pivot(
            index="outer_repeat",
            columns="model",
            values=metric,
        )
        missing_models = [model for model in MODELS if model not in wide.columns]
        if missing_models:
            raise ValueError(
                f"Repeat-level {metric} missing models: {missing_models}"
            )
        wide = wide[MODELS].dropna()

        statistic, p_value = safe_friedman(wide)
        global_rows.append(
            {
                "metric": metric,
                "metric_label": metric_label,
                "n_repeats": int(len(wide)),
                "friedman_statistic": statistic,
                "friedman_p_value": p_value,
            }
        )

        metric_pairs: List[Dict[str, object]] = []
        for model_a, model_b in itertools.combinations(MODELS, 2):
            x = wide[model_a].to_numpy(dtype=float)
            y = wide[model_b].to_numpy(dtype=float)
            test_statistic, raw_p = paired_wilcoxon(x, y)
            difference = y - x

            positive = int(np.sum(difference > 1e-15))
            negative = int(np.sum(difference < -1e-15))
            ties = int(len(difference) - positive - negative)
            nonzero = positive + negative
            rank_biserial = (
                (positive - negative) / nonzero if nonzero else 0.0
            )

            metric_pairs.append(
                {
                    "metric": metric,
                    "metric_label": metric_label,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_repeats": int(len(wide)),
                    "median_difference_b_minus_a": float(
                        np.median(difference)
                    ),
                    "mean_difference_b_minus_a": float(
                        np.mean(difference)
                    ),
                    "wins_b_gt_a": positive,
                    "ties": ties,
                    "losses_b_lt_a": negative,
                    "paired_sign_rank_biserial": float(rank_biserial),
                    "wilcoxon_statistic": test_statistic,
                    "p_value_raw": raw_p,
                }
            )

        adjusted = holm_adjust(
            [row["p_value_raw"] for row in metric_pairs]
        )
        for row, adjusted_p in zip(metric_pairs, adjusted):
            row["p_value_holm"] = float(adjusted_p)
            pair_rows.append(row)

    return pd.DataFrame(global_rows), pd.DataFrame(pair_rows)


def format_p_value(value: float) -> str:
    if value < 1e-4:
        return f"{value:.1e}"
    return f"{value:.4f}"


def add_bracket(
    axis: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    height: float,
    text: str,
) -> None:
    axis.plot(
        [x1, x1, x2, x2],
        [y, y + height, y + height, y],
        linewidth=0.8,
        clip_on=False,
    )
    axis.text(
        (x1 + x2) / 2,
        y + height,
        text,
        ha="center",
        va="bottom",
        fontsize=7,
        clip_on=False,
    )


def plot_metric_panels(
    plotting_data: pd.DataFrame,
    global_tests: pd.DataFrame,
    pairwise_tests: pd.DataFrame,
    plot_level: str,
    annotate_pairs: str,
    png_path: Path,
    pdf_path: Path,
    dpi: int,
) -> None:
    annotation_pairs = (
        PLANNED_PAIRS
        if annotate_pairs == "planned"
        else list(itertools.combinations(MODELS, 2))
    )

    figure, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    axes_flat = axes.ravel()

    for metric_index, (metric, title) in enumerate(METRICS):
        axis = axes_flat[metric_index]
        arrays = [
            plotting_data.loc[
                plotting_data["model"] == model, metric
            ].dropna().to_numpy(dtype=float)
            for model in MODELS
        ]
        expected_n = len(arrays[0])
        if expected_n == 0 or any(len(values) != expected_n for values in arrays):
            raise ValueError(
                f"Unequal/empty plotting arrays for metric {metric}."
            )

        boxplot = axis.boxplot(
            arrays,
            positions=np.arange(1, len(MODELS) + 1),
            widths=0.58,
            patch_artist=True,
            showfliers=False,
            medianprops={"linewidth": 1.4},
        )
        for patch, model in zip(boxplot["boxes"], MODELS):
            patch.set_facecolor(MODEL_COLORS[model])
            patch.set_alpha(0.85)

        random = np.random.default_rng(20260711 + metric_index)
        point_size = 12 if plot_level == "repeat" else 7
        point_alpha = 0.55 if plot_level == "repeat" else 0.28
        for position, values in enumerate(arrays, start=1):
            jitter = random.normal(position, 0.045, len(values))
            axis.scatter(
                jitter,
                values,
                s=point_size,
                alpha=point_alpha,
                edgecolors="none",
            )

        global_row = global_tests.loc[global_tests["metric"] == metric]
        if len(global_row) != 1:
            raise RuntimeError(f"Missing Friedman result for {metric}.")
        friedman_p = float(global_row["friedman_p_value"].iloc[0])

        pair_subset = pairwise_tests.loc[
            pairwise_tests["metric"] == metric
        ]
        annotation_base = 1.015
        annotation_step = 0.047 if len(annotation_pairs) <= 3 else 0.041
        bracket_height = 0.010
        max_annotation = annotation_base + (
            max(len(annotation_pairs) - 1, 0) * annotation_step
        ) + bracket_height + 0.03

        for pair_index, (model_x, model_y) in enumerate(annotation_pairs):
            row = pair_subset.loc[
                (pair_subset["model_a"] == model_x)
                & (pair_subset["model_b"] == model_y)
            ]
            if row.empty:
                row = pair_subset.loc[
                    (pair_subset["model_a"] == model_y)
                    & (pair_subset["model_b"] == model_x)
                ]
            if len(row) != 1:
                raise RuntimeError(
                    f"Missing pairwise result for {metric}: "
                    f"{model_x} vs {model_y}"
                )
            adjusted_p = float(row["p_value_holm"].iloc[0])
            add_bracket(
                axis,
                MODELS.index(model_x) + 1,
                MODELS.index(model_y) + 1,
                annotation_base + pair_index * annotation_step,
                bracket_height,
                f"p={format_p_value(adjusted_p)}",
            )

        axis.set_title(title, fontsize=13, fontweight="bold")
        axis.set_xticks(range(1, len(MODELS) + 1))
        axis.set_xticklabels(
            [MODEL_LABELS[model] for model in MODELS],
            fontsize=8,
        )
        axis.set_ylabel(title)
        axis.set_ylim(0.0, max(1.10, max_annotation))
        axis.grid(axis="y", alpha=0.22)
        axis.text(
            0.02,
            0.98,
            f"Friedman p = {format_p_value(friedman_p)}",
            transform=axis.transAxes,
            va="top",
            fontsize=8.5,
        )

    axes_flat[5].axis("off")

    if plot_level == "repeat":
        observation_note = (
            "Boxes and points: 20 pooled repeat-level OOF values per model."
        )
    else:
        observation_note = (
            "Boxes and points: 100 outer-fold values per model. "
            "This fold-level view is diagnostic."
        )

    figure.suptitle(
        "Repeated Nested-CV Performance of Paired TRB+IGH Models",
        fontsize=17,
        fontweight="bold",
        y=0.99,
    )
    figure.text(
        0.5,
        0.015,
        observation_note
        + " P-values: paired tests on 20 repeat-level pooled OOF results; "
          "Holm-adjusted within each metric.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=[0.02, 0.05, 0.98, 0.965])
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()

    metrics_path = Path(args.metrics).expanduser().resolve()
    predictions_path = Path(args.predictions).expanduser().resolve()
    outer_path = Path(args.outer_assignments).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    ensure_file(metrics_path, "collected outer metrics")
    ensure_file(predictions_path, "collected outer predictions")
    ensure_file(outer_path, "frozen outer assignments")

    _frozen_outer, task_expectations = load_frozen_outer_assignments(
        outer_path,
        expected_repeats=args.expected_repeats,
        expected_folds=args.expected_folds,
        expected_samples=args.expected_samples,
    )

    output_paths = {
        "repeat_level_metrics": output_dir
        / "06_repeat_level_model_metrics.csv",
        "global_friedman": output_dir
        / "06_metric_global_friedman_tests.csv",
        "pairwise_wilcoxon": output_dir
        / "06_metric_pairwise_wilcoxon_tests.csv",
        "png": output_dir / "06_model_metric_boxplots.png",
        "pdf": output_dir / "06_model_metric_boxplots.pdf",
    }

    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite and not args.preflight_only:
        raise FileExistsError(
            "Outputs already exist; add --overwrite:\n"
            + "\n".join(f"  - {path}" for path in existing)
        )

    fold_metrics = validate_outer_metrics(
        pd.read_csv(metrics_path),
        expected_repeats=args.expected_repeats,
        expected_folds=args.expected_folds,
        task_expectations=task_expectations,
    )
    predictions = validate_predictions(
        pd.read_csv(predictions_path),
        expected_repeats=args.expected_repeats,
        expected_folds=args.expected_folds,
        expected_samples=args.expected_samples,
        task_expectations=task_expectations,
    )

    repeat_level = build_repeat_level_metrics(predictions)
    global_tests, pairwise_tests = run_paired_tests(repeat_level)

    expected_repeat_rows = args.expected_repeats * len(MODELS)
    if len(repeat_level) != expected_repeat_rows:
        raise RuntimeError(
            f"Expected {expected_repeat_rows} repeat-level metric rows, "
            f"observed {len(repeat_level)}."
        )
    if len(global_tests) != len(METRICS):
        raise RuntimeError("Unexpected number of global Friedman results.")
    expected_pair_rows = len(METRICS) * math.comb(len(MODELS), 2)
    if len(pairwise_tests) != expected_pair_rows:
        raise RuntimeError(
            f"Expected {expected_pair_rows} pairwise results, "
            f"observed {len(pairwise_tests)}."
        )

    if not args.preflight_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        repeat_level.to_csv(
            output_paths["repeat_level_metrics"], index=False
        )
        global_tests.to_csv(
            output_paths["global_friedman"], index=False
        )
        pairwise_tests.to_csv(
            output_paths["pairwise_wilcoxon"], index=False
        )

        plotting_data = (
            repeat_level if args.plot_level == "repeat" else fold_metrics
        )
        plot_metric_panels(
            plotting_data=plotting_data,
            global_tests=global_tests,
            pairwise_tests=pairwise_tests,
            plot_level=args.plot_level,
            annotate_pairs=args.annotate_pairs,
            png_path=output_paths["png"],
            pdf_path=output_paths["pdf"],
            dpi=args.dpi,
        )

    print("=" * 88)
    print("06 Paired TRB+IGH Repeated-CV Model Metric Visualization")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Analysis view: {ANALYSIS_VIEW}")
    print(f"CV unit: {CV_UNIT}")
    print(f"Collected metrics: {metrics_path}")
    print(f"Collected predictions: {predictions_path}")
    print(f"Frozen outer assignments: {outer_path}")
    print(f"Frozen outer SHA256: {sha256sum(outer_path)}")
    print(f"Outer metric rows: {len(fold_metrics):,}")
    print(f"Prediction rows: {len(predictions):,}")
    print(f"Repeat-level metric rows: {len(repeat_level):,}")
    print(f"Plot level: {args.plot_level}")
    print(f"Pair annotations: {args.annotate_pairs}")
    print("Independent test set: NOT READ")
    print("[Integrity checks]")
    print(
        f"- PASS: {args.expected_repeats} repeats × "
        f"{args.expected_folds} folds × {len(MODELS)} models"
    )
    print(
        f"- PASS: every repeat/model contains "
        f"{args.expected_samples} unique patient-level OOF predictions"
    )
    fold_sizes = sorted({len(ids) for ids in task_expectations.values()})
    print(
        "- PASS: every task/model prediction set exactly matches its frozen "
        f"validation patients; fold sizes={fold_sizes}"
    )
    print("- PASS: paired Friedman/Wilcoxon tests use repeat-level pooled OOF metrics")
    if args.preflight_only:
        print("Preflight-only: no output files were written.")
    else:
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
