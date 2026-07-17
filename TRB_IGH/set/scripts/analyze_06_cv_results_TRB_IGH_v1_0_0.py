#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Analyze repeated nested-CV performance and coefficient stability for paired TRB+IGH models.

Statistical units
-----------------
* Performance summaries and paired model comparisons use 20 pooled repeat-level
  out-of-fold (OOF) results. Repeats share the same patients and summarize
  resampling variability; they are not independent external cohorts.
* Feature stability uses the 100 outer fitted models for each model
  specification. A feature absent from a task after task-specific preprocessing
  is treated as not selected in that task, so the prespecified
  selection-frequency denominator remains all 100 outer tasks.

Coefficient direction
---------------------
The positive outcome is ILD. Positive coefficients indicate a higher predicted
probability of RA-ILD/ILD, whereas negative coefficients indicate the RA
reference direction.

The frozen patient-level outer assignments are read to validate the real
23/24-patient validation folds produced by 118 paired patients split into five
folds. The independent 49-patient test set is never read.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
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
OUT = ROOT / "set/train/result/06_cv_result_summary/analysis"
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
    "M0_clinical": "Clinical: age + sex",
    "M1_static_trb_igh": "Clinical + static TRB/IGH",
    "M2_static_trb_igh_public": (
        "Clinical + static TRB/IGH + dynamic TRB/IGH public"
    ),
    "M3_static_trb_igh_public_material": (
        "Clinical + static TRB/IGH + dynamic TRB/IGH public + material"
    ),
}

METRICS = [
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
    ("sensitivity_recall", "Sensitivity"),
    ("specificity", "Specificity"),
    ("f1", "F1 score"),
]

PLANNED_COMPARISONS = [
    ("M0_clinical", "M1_static_trb_igh"),
    ("M1_static_trb_igh", "M2_static_trb_igh_public"),
    ("M2_static_trb_igh_public", "M3_static_trb_igh_public_material"),
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize paired TRB+IGH repeated-CV model performance and "
            "outer-model coefficient stability."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predictions",
        default=str(COLLECT / "06_all_outer_predictions.csv.gz"),
    )
    parser.add_argument(
        "--coefficients",
        default=str(COLLECT / "06_all_model_coefficients.csv.gz"),
    )
    parser.add_argument(
        "--parameters",
        default=str(COLLECT / "06_all_selected_parameters.csv"),
    )
    parser.add_argument(
        "--outer-assignments",
        default=str(OUTER_ASSIGNMENTS),
        help="Frozen patient-level outer-fold assignment table.",
    )
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--expected-repeats", type=int, default=20)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--expected-samples", type=int, default=118)
    parser.add_argument("--sample-id-col", default="sample_id")
    parser.add_argument("--trb-id-col", default="trb_libraryid")
    parser.add_argument("--igh-id-col", default="igh_libraryid")
    parser.add_argument("--label-col", default="cohort")
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--stable-selection-frequency",
        type=float,
        default=0.50,
        help="Minimum fraction of all outer tasks with a nonzero coefficient.",
    )
    parser.add_argument(
        "--stable-sign-consistency",
        type=float,
        default=0.80,
        help=(
            "Minimum fraction of selected coefficients sharing the predominant "
            "sign."
        ),
    )
    parser.add_argument(
        "--coefficient-tolerance",
        type=float,
        default=1e-12,
        help="Absolute coefficient threshold used to define nonzero selection.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run all input, metric and stability checks without writing outputs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.expected_repeats < 1 or args.expected_folds < 2:
        parser.error("Expected repeats must be >=1 and folds must be >=2.")
    if args.expected_samples < 2:
        parser.error("--expected-samples must be >=2.")
    if args.bootstrap_reps < 100:
        parser.error("--bootstrap-reps must be >=100.")
    for name in ("stable_selection_frequency", "stable_sign_consistency"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1].")
    if args.coefficient_tolerance < 0:
        parser.error("--coefficient-tolerance must be >=0.")
    return args


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


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
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], pd.DataFrame], str]:
    ensure_file(path, "frozen outer assignments")
    outer = pd.read_csv(path, dtype=str)
    outer = outer.drop(
        columns=[c for c in outer.columns if str(c).startswith("Unnamed:")],
        errors="ignore",
    )
    required = [
        "outer_repeat",
        "outer_fold",
        args.sample_id_col,
        args.trb_id_col,
        args.igh_id_col,
        args.label_col,
    ]
    require_columns(outer, required, "frozen outer assignments")

    outer["outer_repeat"] = pd.to_numeric(
        outer["outer_repeat"], errors="raise"
    ).astype(int)
    outer["outer_fold"] = pd.to_numeric(
        outer["outer_fold"], errors="raise"
    ).astype(int)
    for column in [
        args.sample_id_col,
        args.trb_id_col,
        args.igh_id_col,
        args.label_col,
    ]:
        outer[column] = outer[column].astype(str).str.strip()
        if outer[column].eq("").any():
            raise ValueError(
                f"Frozen outer assignments contain empty {column} values."
            )

    expected_repeats = set(range(1, args.expected_repeats + 1))
    expected_folds = set(range(1, args.expected_folds + 1))
    if set(outer["outer_repeat"]) != expected_repeats:
        raise ValueError("Frozen outer repeat IDs are incomplete or unexpected.")
    if set(outer["outer_fold"]) != expected_folds:
        raise ValueError("Frozen outer fold IDs are incomplete or unexpected.")
    if outer.duplicated(["outer_repeat", args.sample_id_col]).any():
        raise ValueError(
            "A patient appears more than once within a frozen outer repeat."
        )

    repeat_counts = outer.groupby("outer_repeat")[args.sample_id_col].nunique()
    if not (repeat_counts == args.expected_samples).all():
        raise ValueError(
            "Frozen outer assignments do not contain exactly "
            f"{args.expected_samples} patients per repeat: {repeat_counts.to_dict()}"
        )

    mapping_columns = [
        args.sample_id_col,
        args.trb_id_col,
        args.igh_id_col,
        args.label_col,
    ]
    canonical = outer[mapping_columns].drop_duplicates()
    if canonical.duplicated(args.sample_id_col).any():
        raise ValueError(
            "Patient-to-TRB/IGH/cohort mapping changes across frozen repeats."
        )
    if len(canonical) != args.expected_samples:
        raise ValueError(
            f"Expected {args.expected_samples} canonical patients, observed "
            f"{len(canonical)}."
        )
    cohorts = set(canonical[args.label_col].str.upper())
    if cohorts != {"RA", "ILD"}:
        raise ValueError(
            f"Frozen cohort labels must be exactly RA/ILD; observed={sorted(cohorts)}"
        )

    expected_universe = frozenset(canonical[args.sample_id_col])
    for repeat, group in outer.groupby("outer_repeat", sort=True):
        if frozenset(group[args.sample_id_col]) != expected_universe:
            raise ValueError(
                f"Frozen patient universe differs in outer repeat {repeat}."
            )

    expectations: Dict[Tuple[int, int], pd.DataFrame] = {}
    for (repeat, fold), group in outer.groupby(
        ["outer_repeat", "outer_fold"], sort=True
    ):
        expected = group[mapping_columns].copy().sort_values(args.sample_id_col)
        if expected.empty:
            raise ValueError(f"Frozen task repeat={repeat}, fold={fold} is empty.")
        expectations[(int(repeat), int(fold))] = expected.reset_index(drop=True)

    expected_tasks = args.expected_repeats * args.expected_folds
    if len(expectations) != expected_tasks:
        raise ValueError(
            f"Expected {expected_tasks} frozen outer tasks, observed {len(expectations)}."
        )

    fold_sizes = np.array([len(frame) for frame in expectations.values()], dtype=int)
    allowed_sizes = {
        args.expected_samples // args.expected_folds,
        int(math.ceil(args.expected_samples / args.expected_folds)),
    }
    if not set(fold_sizes).issubset(allowed_sizes):
        raise ValueError(
            "Frozen validation fold sizes are unexpected: "
            f"observed={sorted(set(fold_sizes))}, allowed={sorted(allowed_sizes)}"
        )

    return (
        canonical.sort_values(args.sample_id_col).reset_index(drop=True),
        expectations,
        sha256sum(path),
    )


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    allowed = {"true", "1", "yes", "false", "0", "no"}
    bad = ~normalized.isin(allowed)
    if bad.any():
        examples = normalized[bad].drop_duplicates().head(10).tolist()
        raise ValueError(f"Invalid boolean values: {examples}")
    return normalized.isin({"true", "1", "yes"})


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def numeric_finite(
    df: pd.DataFrame,
    columns: Sequence[str],
    label: str,
) -> None:
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(float)).all():
            raise ValueError(f"{label}.{column} contains non-finite values.")
        df[column] = values


def expected_task_pairs(args: argparse.Namespace) -> set[Tuple[int, int]]:
    return {
        (repeat, fold)
        for repeat in range(1, args.expected_repeats + 1)
        for fold in range(1, args.expected_folds + 1)
    }


def validate_models(df: pd.DataFrame, label: str) -> None:
    observed = set(df["model"].astype(str))
    expected = set(MODELS)
    if observed != expected:
        raise ValueError(
            f"{label} model set mismatch; expected={sorted(expected)}, "
            f"observed={sorted(observed)}"
        )


def validate_inputs(
    pred: pd.DataFrame,
    coef: pd.DataFrame,
    param: pd.DataFrame,
    canonical_mapping: pd.DataFrame,
    task_expectations: Mapping[Tuple[int, int], pd.DataFrame],
    args: argparse.Namespace,
) -> List[str]:
    checks: List[str] = []
    tasks_expected = expected_task_pairs(args)
    n_tasks = len(tasks_expected)

    require_columns(
        pred,
        [
            "outer_repeat",
            "outer_fold",
            "model",
            args.sample_id_col,
            args.trb_id_col,
            args.igh_id_col,
            "true_label",
            "probability_ILD",
            "predicted_label",
        ],
        "predictions",
    )
    require_columns(
        coef,
        [
            "outer_repeat",
            "outer_fold",
            "model",
            "feature_name",
            "coefficient",
            "nonzero",
        ],
        "coefficients",
    )
    require_columns(
        param,
        [
            "outer_repeat",
            "outer_fold",
            "model",
            "l1_ratio_alpha",
            "lambda",
            "threshold",
            "inner_roc_auc",
            "inner_pr_auc",
        ],
        "parameters",
    )

    for frame, label in (
        (pred, "predictions"),
        (coef, "coefficients"),
        (param, "parameters"),
    ):
        numeric_finite(frame, ["outer_repeat", "outer_fold"], label)
        frame["outer_repeat"] = frame["outer_repeat"].astype(int)
        frame["outer_fold"] = frame["outer_fold"].astype(int)
        frame["model"] = frame["model"].astype(str)
        validate_models(frame, label)
        observed_tasks = set(
            frame[["outer_repeat", "outer_fold"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        if observed_tasks != tasks_expected:
            missing = sorted(tasks_expected - observed_tasks)[:10]
            extra = sorted(observed_tasks - tasks_expected)[:10]
            raise ValueError(
                f"{label} task coverage mismatch; missing={missing}, extra={extra}"
            )

    # Predictions: patient-level OOF coverage and paired-library mapping.
    for column in [args.sample_id_col, args.trb_id_col, args.igh_id_col]:
        pred[column] = pred[column].astype(str).str.strip()
        if pred[column].eq("").any():
            raise ValueError(f"predictions.{column} contains empty values.")
    numeric_finite(
        pred,
        ["true_label", "probability_ILD", "predicted_label"],
        "predictions",
    )
    pred["true_label"] = pred["true_label"].astype(int)
    pred["predicted_label"] = pred["predicted_label"].astype(int)
    if not set(pred["true_label"]).issubset({0, 1}):
        raise ValueError("predictions.true_label must contain only 0/1.")
    if not set(pred["predicted_label"]).issubset({0, 1}):
        raise ValueError("predictions.predicted_label must contain only 0/1.")
    tolerance = 1e-12
    probabilities = pred["probability_ILD"].to_numpy(float)
    if np.any(probabilities < -tolerance) or np.any(probabilities > 1 + tolerance):
        raise ValueError("predictions.probability_ILD must be in [0, 1].")
    pred["probability_ILD"] = np.clip(probabilities, 0.0, 1.0)

    prediction_key = [
        "outer_repeat",
        "outer_fold",
        "model",
        args.sample_id_col,
    ]
    if pred.duplicated(prediction_key).any():
        raise ValueError("Duplicate outer prediction rows detected.")
    expected_prediction_rows = (
        args.expected_repeats * args.expected_samples * len(MODELS)
    )
    if len(pred) != expected_prediction_rows:
        raise ValueError(
            f"Expected {expected_prediction_rows:,} prediction rows; got "
            f"{len(pred):,}."
        )

    fold_sizes: List[int] = []
    for repeat, fold in sorted(tasks_expected):
        expected = task_expectations[(repeat, fold)]
        expected_ids = set(expected[args.sample_id_col].astype(str))
        fold_sizes.append(len(expected_ids))
        task = pred[
            (pred["outer_repeat"] == repeat)
            & (pred["outer_fold"] == fold)
        ]
        for model in MODELS:
            observed = task.loc[
                task["model"] == model, args.sample_id_col
            ].astype(str)
            if set(observed) != expected_ids or len(observed) != len(expected_ids):
                raise ValueError(
                    "Prediction patient set differs from frozen validation fold: "
                    f"repeat={repeat}, fold={fold}, model={model}."
                )

    mapping_check = pred.merge(
        canonical_mapping[
            [args.sample_id_col, args.trb_id_col, args.igh_id_col, args.label_col]
        ].rename(
            columns={
                args.trb_id_col: "__frozen_trb",
                args.igh_id_col: "__frozen_igh",
                args.label_col: "__frozen_cohort",
            }
        ),
        on=args.sample_id_col,
        how="left",
        validate="many_to_one",
    )
    if mapping_check["__frozen_cohort"].isna().any():
        raise ValueError("Predictions contain patients absent from frozen assignments.")
    if not (
        mapping_check[args.trb_id_col].astype(str).to_numpy()
        == mapping_check["__frozen_trb"].astype(str).to_numpy()
    ).all():
        raise ValueError("Patient-to-TRB library mapping differs from frozen assignments.")
    if not (
        mapping_check[args.igh_id_col].astype(str).to_numpy()
        == mapping_check["__frozen_igh"].astype(str).to_numpy()
    ).all():
        raise ValueError("Patient-to-IGH library mapping differs from frozen assignments.")
    expected_binary = (
        mapping_check["__frozen_cohort"].astype(str).str.upper() == "ILD"
    ).astype(int)
    if not (mapping_check["true_label"].to_numpy() == expected_binary.to_numpy()).all():
        raise ValueError("Prediction true_label differs from frozen cohort.")

    repeat_model_counts = pred.groupby(
        ["outer_repeat", "model"]
    )[args.sample_id_col].nunique()
    if not (repeat_model_counts == args.expected_samples).all():
        bad = repeat_model_counts[
            repeat_model_counts != args.expected_samples
        ].head(10)
        raise ValueError(
            "Every repeat/model must contain all pooled OOF patients; "
            f"examples={bad.to_dict()}"
        )
    repeat_sets = pred.groupby(["outer_repeat", "model"])[args.sample_id_col].agg(
        lambda x: frozenset(x)
    )
    if len(set(repeat_sets.tolist())) != 1:
        raise ValueError("Repeat/model OOF patient cohorts are not identical.")
    label_counts = pred.groupby(args.sample_id_col)["true_label"].nunique()
    if (label_counts != 1).any():
        bad = label_counts[label_counts != 1].index[:10].tolist()
        raise ValueError(f"True labels change across tasks for patients: {bad}")
    checks.append(
        f"Predictions: {len(pred):,} rows; every repeat/model covers "
        f"{args.expected_samples} fixed paired patients exactly once."
    )
    checks.append(
        "Every task/model prediction set matches its frozen validation patients; "
        f"observed fold sizes={sorted(set(fold_sizes))}."
    )
    checks.append(
        "Patient labels and paired TRB/IGH library mappings match the frozen split."
    )

    # Selected parameters.
    numeric_finite(
        param,
        [
            "l1_ratio_alpha",
            "lambda",
            "threshold",
            "inner_roc_auc",
            "inner_pr_auc",
        ],
        "parameters",
    )
    parameter_key = ["outer_repeat", "outer_fold", "model"]
    if param.duplicated(parameter_key).any():
        raise ValueError("Duplicate selected-parameter rows detected.")
    expected_parameter_rows = n_tasks * len(MODELS)
    if len(param) != expected_parameter_rows:
        raise ValueError(
            f"Expected {expected_parameter_rows} parameter rows; got {len(param)}."
        )
    if not ((param["l1_ratio_alpha"] > 0) & (param["l1_ratio_alpha"] <= 1)).all():
        raise ValueError("parameters.l1_ratio_alpha must be in (0, 1].")
    if not (param["lambda"] > 0).all():
        raise ValueError("parameters.lambda must be >0.")
    for column in ("threshold", "inner_roc_auc", "inner_pr_auc"):
        values = param[column].to_numpy(float)
        if np.any(values < -tolerance) or np.any(values > 1 + tolerance):
            raise ValueError(f"parameters.{column} must be in [0, 1].")
        param[column] = np.clip(values, 0.0, 1.0)
    checks.append(f"Parameters: complete {expected_parameter_rows}-row task/model grid.")

    # Coefficients.
    coef["feature_name"] = coef["feature_name"].astype(str)
    coef["nonzero"] = as_bool(coef["nonzero"])
    numeric_finite(coef, ["coefficient"], "coefficients")
    coefficient_key = [
        "outer_repeat",
        "outer_fold",
        "model",
        "feature_name",
    ]
    if coef.duplicated(coefficient_key).any():
        raise ValueError("Duplicate model-feature coefficient rows detected.")
    model_task_counts = (
        coef[["model", "outer_repeat", "outer_fold"]]
        .drop_duplicates()
        .groupby("model")
        .size()
        .reindex(MODELS, fill_value=0)
    )
    if not (model_task_counts == n_tasks).all():
        raise ValueError(
            "Each model must have coefficients for all outer tasks; "
            f"observed={model_task_counts.to_dict()}"
        )
    intercept = coef[coef["feature_name"] == "__INTERCEPT__"]
    intercept_counts = intercept.groupby(
        ["outer_repeat", "outer_fold", "model"]
    ).size()
    if len(intercept_counts) != expected_parameter_rows or not (
        intercept_counts == 1
    ).all():
        raise ValueError("Every task/model must contain exactly one intercept.")
    non_intercept = coef[coef["feature_name"] != "__INTERCEPT__"]
    calculated_nonzero = (
        non_intercept["coefficient"].abs() > args.coefficient_tolerance
    )
    if not (
        calculated_nonzero.to_numpy() == non_intercept["nonzero"].to_numpy()
    ).all():
        mismatch = int(
            np.sum(
                calculated_nonzero.to_numpy()
                != non_intercept["nonzero"].to_numpy()
            )
        )
        raise ValueError(
            f"Coefficient nonzero flags disagree with tolerance in {mismatch} rows."
        )
    checks.append(
        f"Coefficients: all {expected_parameter_rows} task/model fits represented; "
        "intercepts valid and nonzero flags consistent."
    )
    return checks


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if len(p) == 0:
        return np.array([], dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def paired_wilcoxon(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    difference = np.asarray(y, float) - np.asarray(x, float)
    if np.allclose(difference, 0.0):
        return 0.0, 1.0
    result = wilcoxon(
        x,
        y,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def repeat_level_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (repeat, model), frame in pred.groupby(
        ["outer_repeat", "model"], sort=True
    ):
        if frame["sample_id"].duplicated().any():
            raise ValueError(
                f"Duplicate pooled OOF sample: repeat={repeat}, model={model}"
            )
        y = frame["true_label"].to_numpy(dtype=int)
        probability = frame["probability_ILD"].to_numpy(dtype=float)
        predicted = frame["predicted_label"].to_numpy(dtype=int)
        tn, fp, fn, tp = confusion_matrix(
            y, predicted, labels=[0, 1]
        ).ravel()
        rows.append({
            "outer_repeat": int(repeat),
            "model": model,
            "model_definition": MODEL_LABELS[model],
            "n_samples": len(frame),
            "n_RA": int(np.sum(y == 0)),
            "n_ILD": int(np.sum(y == 1)),
            "roc_auc": float(roc_auc_score(y, probability)),
            "pr_auc": float(average_precision_score(y, probability)),
            "sensitivity_recall": float(
                recall_score(y, predicted, zero_division=0)
            ),
            "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
            "precision": float(
                precision_score(y, predicted, zero_division=0)
            ),
            "f1": float(f1_score(y, predicted, zero_division=0)),
            "accuracy": float(accuracy_score(y, predicted)),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        })
    return pd.DataFrame(rows).sort_values(
        ["outer_repeat", "model"]
    ).reset_index(drop=True)


def bootstrap_mean_interval(
    values: np.ndarray,
    reps: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    indices = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def performance_summary(
    repeat_metrics: pd.DataFrame,
    bootstrap_reps: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    for model in MODELS:
        for metric, metric_label in METRICS:
            values = repeat_metrics.loc[
                repeat_metrics["model"] == model, metric
            ].dropna().to_numpy(float)
            lower, upper = bootstrap_mean_interval(
                values, bootstrap_reps, rng
            )
            rows.append({
                "model": model,
                "model_definition": MODEL_LABELS[model],
                "metric": metric,
                "metric_label": metric_label,
                "n_repeats": len(values),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
                "median": float(np.median(values)),
                "q1": float(np.quantile(values, 0.25)),
                "q3": float(np.quantile(values, 0.75)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean_repeat_bootstrap_interval95_lower": lower,
                "mean_repeat_bootstrap_interval95_upper": upper,
            })
    return pd.DataFrame(rows)


def rank_biserial_from_differences(difference: pd.Series) -> float:
    nonzero = difference[difference != 0]
    if nonzero.empty:
        return 0.0
    ranks = nonzero.abs().rank(method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    denominator = positive + negative
    return (positive - negative) / denominator if denominator else 0.0


def pairwise_comparisons(repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for metric, metric_label in METRICS:
        wide = repeat_metrics.pivot(
            index="outer_repeat", columns="model", values=metric
        )[MODELS].dropna()
        metric_rows: List[Dict[str, object]] = []
        for model_a, model_b in itertools.combinations(MODELS, 2):
            statistic, p_value = paired_wilcoxon(
                wide[model_a].to_numpy(), wide[model_b].to_numpy()
            )
            difference = wide[model_b] - wide[model_a]
            metric_rows.append({
                "metric": metric,
                "metric_label": metric_label,
                "model_a": model_a,
                "model_b": model_b,
                "comparison_is_prespecified": (
                    (model_a, model_b) in PLANNED_COMPARISONS
                ),
                "n_repeats": len(wide),
                "mean_a": float(wide[model_a].mean()),
                "mean_b": float(wide[model_b].mean()),
                "mean_difference_b_minus_a": float(difference.mean()),
                "median_difference_b_minus_a": float(difference.median()),
                "q1_difference": float(difference.quantile(0.25)),
                "q3_difference": float(difference.quantile(0.75)),
                "wins_b_gt_a": int((difference > 0).sum()),
                "ties": int((difference == 0).sum()),
                "losses_b_lt_a": int((difference < 0).sum()),
                "paired_rank_biserial_b_minus_a": float(
                    rank_biserial_from_differences(difference)
                ),
                "wilcoxon_statistic": statistic,
                "p_value_raw": p_value,
            })
        adjusted = holm_adjust([row["p_value_raw"] for row in metric_rows])
        for row, value in zip(metric_rows, adjusted):
            row["p_value_holm_within_metric"] = float(value)
            rows.append(row)
    return pd.DataFrame(rows)


def classify_feature_source(feature_name: str) -> str:
    name = str(feature_name)
    if name == "age" or name.startswith("sex__"):
        return "clinical"
    if name.startswith("trb_"):
        return "TRB"
    if name.startswith("igh_"):
        return "IGH"
    if name.startswith("material__") or name == "material":
        return "material"
    return "other"


def coefficient_stability(
    coefficients: pd.DataFrame,
    expected_tasks: int,
    tolerance: float,
    selection_threshold: float,
    sign_threshold: float,
) -> pd.DataFrame:
    frame = coefficients[
        coefficients["feature_name"] != "__INTERCEPT__"
    ].copy()
    frame["selected"] = frame["coefficient"].abs() > tolerance
    frame["absolute_coefficient"] = frame["coefficient"].abs()

    rows: List[Dict[str, object]] = []
    for (model, feature_name), group in frame.groupby(
        ["model", "feature_name"], sort=True
    ):
        observed_tasks = group[
            ["outer_repeat", "outer_fold"]
        ].drop_duplicates().shape[0]
        selected = group[group["selected"]]
        positive_count = int((selected["coefficient"] > 0).sum())
        negative_count = int((selected["coefficient"] < 0).sum())
        selected_count = len(selected)
        sign_consistency = (
            max(positive_count, negative_count) / selected_count
            if selected_count
            else np.nan
        )
        if positive_count > negative_count:
            direction = "positive_ILD"
        elif negative_count > positive_count:
            direction = "negative_RA"
        else:
            direction = "tie_or_not_selected"

        selection_frequency = selected_count / expected_tasks
        is_stable = bool(
            selection_frequency >= selection_threshold
            and selected_count > 0
            and sign_consistency >= sign_threshold
        )
        coefficient_sum = float(group["coefficient"].sum())
        # Missing feature rows (usually zero-variance filtered) and explicit zero
        # coefficients are both treated as zero over the fixed task denominator.
        all_task_mean_coefficient_zero_filled = coefficient_sum / expected_tasks
        rows.append({
            "model": model,
            "model_definition": MODEL_LABELS[model],
            "feature_name": feature_name,
            "feature_source": classify_feature_source(feature_name),
            "n_outer_tasks_expected": expected_tasks,
            "n_outer_tasks_observed_after_preprocessing": observed_tasks,
            "availability_frequency": observed_tasks / expected_tasks,
            "selected_count": selected_count,
            "selection_frequency": selection_frequency,
            "selection_frequency_among_observed": (
                selected_count / observed_tasks if observed_tasks else np.nan
            ),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "sign_consistency_among_selected": sign_consistency,
            "predominant_direction": direction,
            "signed_selection_frequency": (
                positive_count - negative_count
            ) / expected_tasks,
            "coefficient_mean_all_tasks_zero_filled": (
                all_task_mean_coefficient_zero_filled
            ),
            "coefficient_median_selected": (
                float(selected["coefficient"].median())
                if selected_count else 0.0
            ),
            "coefficient_q1_selected": (
                float(selected["coefficient"].quantile(0.25))
                if selected_count else 0.0
            ),
            "coefficient_q3_selected": (
                float(selected["coefficient"].quantile(0.75))
                if selected_count else 0.0
            ),
            "absolute_coefficient_median_selected": (
                float(selected["absolute_coefficient"].median())
                if selected_count else 0.0
            ),
            "absolute_coefficient_mean_selected": (
                float(selected["absolute_coefficient"].mean())
                if selected_count else 0.0
            ),
            "stable_selection_frequency_threshold": selection_threshold,
            "stable_sign_consistency_threshold": sign_threshold,
            "is_stable": is_stable,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        [
            "model",
            "is_stable",
            "selection_frequency",
            "sign_consistency_among_selected",
            "absolute_coefficient_median_selected",
            "feature_name",
        ],
        ascending=[True, False, False, False, False, True],
    ).reset_index(drop=True)


def parameter_selection_summary(param: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        param.groupby(["model", "l1_ratio_alpha", "lambda"], as_index=False)
        .agg(
            selection_count=("model", "size"),
            threshold_mean=("threshold", "mean"),
            threshold_median=("threshold", "median"),
            inner_roc_auc_mean=("inner_roc_auc", "mean"),
            inner_pr_auc_mean=("inner_pr_auc", "mean"),
        )
    )
    totals = param.groupby("model").size().rename("n_tasks").reset_index()
    grouped = grouped.merge(totals, on="model", how="left", validate="many_to_one")
    grouped["selection_frequency"] = grouped["selection_count"] / grouped["n_tasks"]
    grouped["model_definition"] = grouped["model"].map(MODEL_LABELS)
    return grouped.sort_values(
        ["model", "selection_count", "l1_ratio_alpha", "lambda"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)


def format_p(value: float) -> str:
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def write_summary(
    path: Path,
    args: argparse.Namespace,
    checks: Sequence[str],
    performance: pd.DataFrame,
    comparisons: pd.DataFrame,
    stability: pd.DataFrame,
    stable: pd.DataFrame,
    expected_tasks: int,
    outer_sha256: str,
) -> None:
    lines = [
        "# 06 Paired TRB+IGH Repeated-CV Model and Feature Stability Summary",
        "",
        "## Scope and statistical units",
        "",
        f"- Script version: `{SCRIPT_VERSION}`",
        f"- Analysis view: **{ANALYSIS_VIEW}**",
        f"- Cross-validation unit: **{CV_UNIT}**",
        f"- Frozen outer-assignment SHA256: `{outer_sha256}`",
        f"- Patients per repeat: **{args.expected_samples}**",
        "- Performance summaries and paired comparisons use pooled OOF metrics from each of 20 repeats.",
        f"- Feature stability uses **{expected_tasks} outer fitted models per model specification**.",
        "- Positive coefficients indicate a higher predicted probability of ILD/RA-ILD; negative coefficients indicate the RA direction.",
        "- Features absent after task-specific preprocessing are treated as not selected for the fixed selection-frequency denominator.",
        "- Bootstrap intervals summarize variation across repeated partitions; repeats share patients and are not independent external cohorts.",
        "- The independent test set was not read.",
        "",
        "## Integrity checks",
        "",
        *[f"- PASS: {check}" for check in checks],
        "",
        "## Repeat-level performance",
        "",
    ]

    for model in MODELS:
        lines.append(f"### {model}")
        subset = performance[performance["model"] == model]
        for _, row in subset.iterrows():
            lines.append(
                f"- {row['metric_label']}: mean **{row['mean']:.4f}**, "
                f"median {row['median']:.4f}, descriptive 95% repeat-bootstrap "
                f"interval {row['mean_repeat_bootstrap_interval95_lower']:.4f}–"
                f"{row['mean_repeat_bootstrap_interval95_upper']:.4f}."
            )
        lines.append("")

    lines.extend(["## Prespecified paired comparisons", ""])
    planned = comparisons[comparisons["comparison_is_prespecified"]]
    for _, row in planned.iterrows():
        lines.append(
            f"- {row['metric_label']}: `{row['model_a']}` vs "
            f"`{row['model_b']}`; median difference (B−A) "
            f"{row['median_difference_b_minus_a']:.4f}; Holm-adjusted "
            f"p={format_p(row['p_value_holm_within_metric'])}."
        )

    lines.extend([
        "",
        "## Stable-feature rule",
        "",
        f"- Selection frequency ≥ **{args.stable_selection_frequency:.0%}** across all {expected_tasks} outer tasks.",
        f"- Sign consistency ≥ **{args.stable_sign_consistency:.0%}** among tasks in which the feature was selected.",
        f"- Nonzero coefficient tolerance: **{args.coefficient_tolerance:g}**.",
        "",
        "## Stable-feature counts",
        "",
    ])
    stable_counts = stable.groupby("model").size().reindex(MODELS, fill_value=0)
    total_counts = stability.groupby("model").size().reindex(MODELS, fill_value=0)
    for model in MODELS:
        lines.append(
            f"- `{model}`: **{int(stable_counts[model])}** stable features "
            f"among {int(total_counts[model])} evaluated features."
        )

    lines.extend(["", "## Stable features by source", ""])
    if stable.empty:
        lines.append("- None met the prespecified rule.")
    else:
        source_counts = (
            stable.groupby(["model", "feature_source"])
            .size()
            .rename("n")
            .reset_index()
        )
        for _, row in source_counts.iterrows():
            lines.append(
                f"- `{row['model']}` / `{row['feature_source']}`: **{int(row['n'])}**."
            )

    lines.extend(["", "## Stable features", ""])
    if stable.empty:
        lines.append("- None met the prespecified rule.")
    else:
        for _, row in stable.iterrows():
            lines.append(
                f"- `{row['model']}` / `{row['feature_name']}` "
                f"({row['feature_source']}): selected "
                f"{row['selection_frequency']:.1%}; direction "
                f"**{row['predominant_direction']}**; sign consistency "
                f"{row['sign_consistency_among_selected']:.1%}; median selected "
                f"coefficient {row['coefficient_median_selected']:.6g}."
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    prediction_path = Path(args.predictions).expanduser().resolve()
    coefficient_path = Path(args.coefficients).expanduser().resolve()
    parameter_path = Path(args.parameters).expanduser().resolve()
    outer_path = Path(args.outer_assignments).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    ensure_file(prediction_path, "collected predictions")
    ensure_file(coefficient_path, "collected coefficients")
    ensure_file(parameter_path, "collected selected parameters")
    canonical_mapping, task_expectations, outer_sha256 = (
        load_frozen_outer_assignments(outer_path, args)
    )

    paths = {
        "repeat_metrics": output_dir / "06_repeat_level_model_metrics.csv",
        "performance": output_dir / "06_model_performance_summary.csv",
        "comparisons": output_dir / "06_model_pairwise_comparisons.csv",
        "stability": output_dir / "06_feature_stability_by_model.csv",
        "stable": output_dir / "06_stable_features_filtered.csv",
        "parameters": output_dir / "06_parameter_selection_summary.csv",
        "summary": output_dir / "06_analysis_summary.md",
    }
    if not args.preflight_only:
        existing = [path for path in paths.values() if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(
                "Outputs already exist; add --overwrite:\n"
                + "\n".join(f"  - {path}" for path in existing)
            )

    print("=" * 88)
    print("06 Paired TRB+IGH Repeated-CV Model and Feature Stability Analysis")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Analysis view: {ANALYSIS_VIEW}")
    print(f"CV unit: {CV_UNIT}")
    print(f"Predictions:  {prediction_path}")
    print(f"Coefficients: {coefficient_path}")
    print(f"Parameters:   {parameter_path}")
    print(f"Frozen outer assignments: {outer_path}")
    print(f"Frozen outer SHA256: {outer_sha256}")
    print(f"Expected patients: {args.expected_samples}")
    print("Independent test set: NOT READ")
    print(
        "Stable rule: selection frequency >= "
        f"{args.stable_selection_frequency:.0%}; sign consistency >= "
        f"{args.stable_sign_consistency:.0%}"
    )
    print()

    predictions = pd.read_csv(prediction_path)
    coefficients = pd.read_csv(coefficient_path)
    parameters = pd.read_csv(parameter_path)

    checks = validate_inputs(
        predictions,
        coefficients,
        parameters,
        canonical_mapping,
        task_expectations,
        args,
    )
    expected_tasks = args.expected_repeats * args.expected_folds

    repeat_metrics = repeat_level_metrics(predictions)
    expected_repeat_metric_rows = args.expected_repeats * len(MODELS)
    if len(repeat_metrics) != expected_repeat_metric_rows:
        raise RuntimeError(
            f"Expected {expected_repeat_metric_rows} repeat-level metric rows; "
            f"got {len(repeat_metrics)}."
        )
    performance = performance_summary(
        repeat_metrics, args.bootstrap_reps, args.seed
    )
    comparisons = pairwise_comparisons(repeat_metrics)
    stability = coefficient_stability(
        coefficients,
        expected_tasks=expected_tasks,
        tolerance=args.coefficient_tolerance,
        selection_threshold=args.stable_selection_frequency,
        sign_threshold=args.stable_sign_consistency,
    )
    stable = stability[stability["is_stable"]].copy()
    parameters_summary = parameter_selection_summary(parameters)

    print("[Integrity checks]")
    for check in checks:
        print(f"- PASS: {check}")
    print()
    print("[Analysis dimensions]")
    print(f"Repeat-level metric rows: {len(repeat_metrics):,}")
    print(f"Performance summary rows: {len(performance):,}")
    print(f"Pairwise comparison rows: {len(comparisons):,}")
    print(f"Feature stability rows: {len(stability):,}")
    print(f"Stable feature rows: {len(stable):,}")
    print(f"Parameter summary rows: {len(parameters_summary):,}")

    if args.preflight_only:
        print("Preflight-only: no output files were written.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    repeat_metrics.to_csv(paths["repeat_metrics"], index=False)
    performance.to_csv(paths["performance"], index=False)
    comparisons.to_csv(paths["comparisons"], index=False)
    stability.to_csv(paths["stability"], index=False)
    stable.to_csv(paths["stable"], index=False)
    parameters_summary.to_csv(paths["parameters"], index=False)
    write_summary(
        paths["summary"],
        args,
        checks,
        performance,
        comparisons,
        stability,
        stable,
        expected_tasks,
        outer_sha256,
    )

    print("[Output files]")
    for name, path in paths.items():
        print(f"- {name}: {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
