#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Collect and validate all repeated nested-CV outer-task results for paired TRB+IGH.

Expected workflow:
    20 repeats x 5 outer folds = 100 patient-level outer tasks.

The collector validates every task against the frozen patient-level outer split,
then combines metrics, predictions, coefficients and selected parameters.  It
never reads the independent test set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
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
DEFAULT_INPUT = (
    ROOT
    / "set/train/result/05_modeling/single_outer_trial_loo_TRB_IGH"
    / "public_raw_bilateral"
)
DEFAULT_OUTER = (
    ROOT
    / "set/train/result/05_modeling/cv_splits"
    / "05_outer_fold_assignments.csv"
)
DEFAULT_OUTPUT = ROOT / "set/train/result/06_cv_result_summary/collected"

MODELS = [
    "M0_clinical",
    "M1_static_trb_igh",
    "M2_static_trb_igh_public",
    "M3_static_trb_igh_public_material",
]
MODEL_SET = set(MODELS)

REQUIRED_TASK_FILES = [
    "05_trial_configuration.json",
    "05_outer_validation_metrics.csv",
    "05_outer_validation_predictions.csv",
    "05_final_model_coefficients.csv",
]

OUTPUT_FILENAMES = {
    "metrics": "06_all_outer_metrics.csv",
    "predictions": "06_all_outer_predictions.csv.gz",
    "coefficients": "06_all_model_coefficients.csv.gz",
    "parameters": "06_all_selected_parameters.csv",
    "integrity": "06_task_integrity_report.csv",
    "summary": "06_collection_summary.md",
}


class TaskValidationError(ValueError):
    """Raised when one outer task fails schema or integrity validation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect all paired TRB+IGH LOO outer-task CV results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT))
    parser.add_argument("--outer-assignments", default=str(DEFAULT_OUTER))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--expected-repeats", type=int, default=20)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--expected-samples", type=int, default=118)
    parser.add_argument(
        "--expected-public-feature-set",
        default="raw_bilateral",
        choices=("raw_bilateral", "log_ratio", "all18"),
    )
    parser.add_argument("--sample-id-col", default="sample_id")
    parser.add_argument("--label-col", default="cohort")
    parser.add_argument("--trb-id-col", default="trb_libraryid")
    parser.add_argument("--igh-id-col", default="igh_libraryid")
    parser.add_argument("--metric-tolerance", type=float, default=1e-10)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write valid partial collections and return success even when tasks are missing/invalid.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate and assemble in memory but do not write output files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for name in ("expected_repeats", "expected_folds", "expected_samples"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.metric_tolerance < 0:
        parser.error("--metric-tolerance must be >= 0")
    return args


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def numeric_series(
    frame: pd.DataFrame,
    column: str,
    label: str,
    *,
    finite: bool = True,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        raise TaskValidationError(f"{label}.{column} contains non-numeric/missing values")
    if finite and not np.isfinite(values.to_numpy(float)).all():
        raise TaskValidationError(f"{label}.{column} contains non-finite values")
    return values


def require_columns(frame: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise TaskValidationError(f"{label} missing columns: {missing}")


def add_or_validate_task_columns(
    frame: pd.DataFrame,
    repeat: int,
    fold: int,
    task_name: str,
) -> pd.DataFrame:
    frame = frame.copy()
    checks = [
        ("outer_repeat", repeat),
        ("outer_fold", fold),
        ("task_name", task_name),
    ]
    for column, expected in checks:
        if column not in frame.columns:
            frame.insert(0, column, expected)
            continue
        if column in {"outer_repeat", "outer_fold"}:
            observed = pd.to_numeric(frame[column], errors="coerce")
            if observed.isna().any() or not (observed.astype(int) == int(expected)).all():
                raise TaskValidationError(
                    f"{column} values do not match task directory {task_name}"
                )
            frame[column] = observed.astype(int)
        elif not (frame[column].astype(str) == str(expected)).all():
            raise TaskValidationError(
                f"{column} values do not match task directory {task_name}"
            )
    ordered = ["outer_repeat", "outer_fold", "task_name"]
    return frame[ordered + [c for c in frame.columns if c not in ordered]]


def load_frozen_outer(args: argparse.Namespace, path: Path) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], pd.DataFrame], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen outer assignments not found: {path}")
    outer = pd.read_csv(path, dtype=str)
    outer = outer.drop(
        columns=[c for c in outer.columns if str(c).startswith("Unnamed:")],
        errors="ignore",
    )
    required = [
        "outer_repeat",
        "outer_fold",
        args.sample_id_col,
        args.label_col,
        args.trb_id_col,
        args.igh_id_col,
    ]
    missing = [c for c in required if c not in outer.columns]
    if missing:
        raise ValueError(f"Frozen outer assignments missing columns: {missing}")

    outer["outer_repeat"] = pd.to_numeric(outer["outer_repeat"], errors="raise").astype(int)
    outer["outer_fold"] = pd.to_numeric(outer["outer_fold"], errors="raise").astype(int)
    for column in [args.sample_id_col, args.trb_id_col, args.igh_id_col]:
        outer[column] = outer[column].astype(str).str.strip()
        if outer[column].eq("").any():
            raise ValueError(f"Frozen outer assignments contain empty {column}")
    outer[args.label_col] = outer[args.label_col].astype(str).str.upper().str.strip()
    if set(outer[args.label_col]) != {"RA", "ILD"}:
        raise ValueError("Frozen outer labels must be exactly RA and ILD")

    expected_repeats = set(range(1, args.expected_repeats + 1))
    observed_repeats = set(outer["outer_repeat"])
    if observed_repeats != expected_repeats:
        raise ValueError(
            f"Frozen outer repeats mismatch: expected={sorted(expected_repeats)}, "
            f"observed={sorted(observed_repeats)}"
        )

    task_expectations: Dict[Tuple[int, int], pd.DataFrame] = {}
    canonical_mapping: pd.DataFrame | None = None
    for repeat in range(1, args.expected_repeats + 1):
        rep = outer[outer["outer_repeat"] == repeat].copy()
        if len(rep) != args.expected_samples:
            raise ValueError(
                f"Frozen repeat {repeat}: expected {args.expected_samples} patients, got {len(rep)}"
            )
        if rep[args.sample_id_col].duplicated().any():
            raise ValueError(f"Frozen repeat {repeat} has duplicated patient IDs")
        if set(rep["outer_fold"]) != set(range(1, args.expected_folds + 1)):
            raise ValueError(f"Frozen repeat {repeat} does not contain all outer folds")

        mapping = rep[
            [args.sample_id_col, args.label_col, args.trb_id_col, args.igh_id_col]
        ].sort_values(args.sample_id_col).reset_index(drop=True)
        if canonical_mapping is None:
            canonical_mapping = mapping
        elif not mapping.equals(canonical_mapping):
            raise ValueError(
                f"Patient/cohort/TRB/IGH mapping differs in frozen repeat {repeat}"
            )

        for fold in range(1, args.expected_folds + 1):
            valid = rep[rep["outer_fold"] == fold].copy()
            if valid.empty:
                raise ValueError(f"Frozen task repeat={repeat}, fold={fold} has no validation patients")
            if set(valid[args.label_col]) != {"RA", "ILD"}:
                raise ValueError(
                    f"Frozen task repeat={repeat}, fold={fold} lacks RA or ILD validation patients"
                )
            task_expectations[(repeat, fold)] = valid

    assert canonical_mapping is not None
    return canonical_mapping, task_expectations, sha256sum(path)


def validate_configuration(
    config: Mapping[str, object],
    repeat: int,
    fold: int,
    n_train: int,
    n_valid: int,
    frozen_outer_sha256: str,
    args: argparse.Namespace,
) -> Mapping[str, Mapping[str, object]]:
    problems: List[str] = []
    if str(config.get("analysis_view", "")) != ANALYSIS_VIEW:
        problems.append(
            f"analysis_view={config.get('analysis_view')!r}, expected {ANALYSIS_VIEW}"
        )
    if str(config.get("cv_unit", "")) != CV_UNIT:
        problems.append(f"cv_unit={config.get('cv_unit')!r}, expected {CV_UNIT}")
    if int(config.get("outer_repeat", -1)) != repeat:
        problems.append("outer_repeat mismatch")
    if int(config.get("outer_fold", -1)) != fold:
        problems.append("outer_fold mismatch")
    if int(config.get("n_outer_train", -1)) != n_train:
        problems.append(f"n_outer_train={config.get('n_outer_train')}, expected {n_train}")
    if int(config.get("n_outer_validation", -1)) != n_valid:
        problems.append(
            f"n_outer_validation={config.get('n_outer_validation')}, expected {n_valid}"
        )
    if config.get("public_feature_set_per_receptor") != args.expected_public_feature_set:
        problems.append(
            "public_feature_set_per_receptor="
            f"{config.get('public_feature_set_per_receptor')!r}, "
            f"expected {args.expected_public_feature_set!r}"
        )

    configured_models = config.get("models", {})
    if not isinstance(configured_models, Mapping) or set(configured_models) != MODEL_SET:
        problems.append("configuration model set is not the four expected TRB+IGH models")

    selected = config.get("selected_parameters", {})
    if not isinstance(selected, Mapping) or set(selected) != MODEL_SET:
        problems.append("selected_parameters model set is invalid")

    receptor_inputs = config.get("receptor_inputs", {})
    if not isinstance(receptor_inputs, Mapping) or set(receptor_inputs) != {"TRB", "IGH"}:
        problems.append("receptor_inputs must contain exactly TRB and IGH")

    full_counts = config.get("full_reference_counts", {})
    if not isinstance(full_counts, Mapping) or set(full_counts) != {"TRB", "IGH"}:
        problems.append("full_reference_counts must contain exactly TRB and IGH")

    input_sha = config.get("input_sha256", {})
    if isinstance(input_sha, Mapping) and "outer" in input_sha:
        if str(input_sha["outer"]) != frozen_outer_sha256:
            problems.append("configuration frozen outer SHA256 mismatch")
    else:
        problems.append("configuration input_sha256.outer is missing")

    if problems:
        raise TaskValidationError("configuration: " + "; ".join(problems))
    return selected  # type: ignore[return-value]


def validate_metrics(
    metrics: pd.DataFrame,
    repeat: int,
    fold: int,
    task_name: str,
    n_train: int,
    n_valid: int,
    tolerance: float,
) -> pd.DataFrame:
    required = [
        "model", "roc_auc", "pr_auc", "fit_converged",
        "n_outer_train", "n_outer_validation",
    ]
    require_columns(metrics, required, "metrics")
    if len(metrics) != len(MODELS):
        raise TaskValidationError(
            f"metrics expected {len(MODELS)} rows, observed {len(metrics)}"
        )
    if metrics["model"].duplicated().any() or set(metrics["model"].astype(str)) != MODEL_SET:
        raise TaskValidationError("metrics model rows are invalid")
    if not bool_series(metrics["fit_converged"]).all():
        raise TaskValidationError("one or more final model fits did not converge")
    for column in ("roc_auc", "pr_auc"):
        values = numeric_series(metrics, column, "metrics")
        if ((values < -tolerance) | (values > 1.0 + tolerance)).any():
            raise TaskValidationError(f"metrics.{column} is outside [0, 1] beyond tolerance")
    train_n = numeric_series(metrics, "n_outer_train", "metrics").astype(int)
    valid_n = numeric_series(metrics, "n_outer_validation", "metrics").astype(int)
    if not (train_n == n_train).all():
        raise TaskValidationError(f"metrics n_outer_train mismatch; expected {n_train}")
    if not (valid_n == n_valid).all():
        raise TaskValidationError(f"metrics n_outer_validation mismatch; expected {n_valid}")
    return add_or_validate_task_columns(metrics, repeat, fold, task_name)


def validate_predictions(
    predictions: pd.DataFrame,
    expected_valid: pd.DataFrame,
    repeat: int,
    fold: int,
    task_name: str,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, List[str]]:
    required = [
        "model", args.sample_id_col, args.trb_id_col, args.igh_id_col,
        "true_label", "true_cohort", "probability_ILD", "threshold",
        "predicted_label", "predicted_cohort",
    ]
    require_columns(predictions, required, "predictions")
    if set(predictions["model"].astype(str)) != MODEL_SET:
        raise TaskValidationError("predictions model set is invalid")
    if predictions.duplicated(["model", args.sample_id_col]).any():
        raise TaskValidationError("predictions contain duplicate model/patient rows")

    expected_ids = set(expected_valid[args.sample_id_col].astype(str))
    n_valid = len(expected_ids)
    expected_rows = len(MODELS) * n_valid
    if len(predictions) != expected_rows:
        raise TaskValidationError(
            f"predictions expected {expected_rows} rows, observed {len(predictions)}"
        )

    for model in MODELS:
        subset = predictions[predictions["model"].astype(str) == model]
        if set(subset[args.sample_id_col].astype(str)) != expected_ids:
            raise TaskValidationError(
                f"prediction patient set for {model} differs from frozen validation fold"
            )

    frozen = expected_valid[
        [args.sample_id_col, args.label_col, args.trb_id_col, args.igh_id_col]
    ].copy()
    frozen[args.sample_id_col] = frozen[args.sample_id_col].astype(str)
    frozen = frozen.rename(columns={args.label_col: "frozen_cohort"})
    check = predictions.merge(frozen, on=args.sample_id_col, how="left", validate="many_to_one")
    if check["frozen_cohort"].isna().any():
        raise TaskValidationError("predictions include patients absent from frozen validation fold")
    for column in [args.trb_id_col, args.igh_id_col]:
        if not (
            check[column + "_x"].astype(str).to_numpy()
            == check[column + "_y"].astype(str).to_numpy()
        ).all():
            raise TaskValidationError(f"prediction {column} mapping differs from frozen split")

    labels = numeric_series(predictions, "true_label", "predictions").astype(int)
    predicted = numeric_series(predictions, "predicted_label", "predictions").astype(int)
    if not labels.isin([0, 1]).all():
        raise TaskValidationError("true_label contains values other than 0/1")
    if not predicted.isin([0, 1]).all():
        raise TaskValidationError("predicted_label contains values other than 0/1")
    probability = numeric_series(predictions, "probability_ILD", "predictions")
    threshold = numeric_series(predictions, "threshold", "predictions")
    if not probability.between(0.0, 1.0).all():
        raise TaskValidationError("probability_ILD is outside [0, 1]")
    if not threshold.between(0.0, 1.0).all():
        raise TaskValidationError("threshold is outside [0, 1]")
    calculated = (probability >= threshold).astype(int)
    if not (calculated.to_numpy() == predicted.to_numpy()).all():
        raise TaskValidationError("predicted_label is inconsistent with probability/threshold")

    expected_cohort = predictions[args.sample_id_col].astype(str).map(
        frozen.set_index(args.sample_id_col)["frozen_cohort"]
    )
    if not (
        predictions["true_cohort"].astype(str).str.upper().to_numpy()
        == expected_cohort.astype(str).str.upper().to_numpy()
    ).all():
        raise TaskValidationError("true_cohort differs from frozen cohort")
    expected_binary = (expected_cohort.astype(str).str.upper() == "ILD").astype(int)
    if not (labels.to_numpy() == expected_binary.to_numpy()).all():
        raise TaskValidationError("true_label differs from frozen cohort")
    predicted_cohort = np.where(predicted.to_numpy() == 1, "ILD", "RA")
    if not (
        predictions["predicted_cohort"].astype(str).str.upper().to_numpy()
        == predicted_cohort
    ).all():
        raise TaskValidationError("predicted_cohort differs from predicted_label")

    label_counts = predictions.groupby(args.sample_id_col)["true_label"].nunique()
    if not (label_counts == 1).all():
        raise TaskValidationError("true_label differs among models for a patient")

    predictions = add_or_validate_task_columns(predictions, repeat, fold, task_name)
    return predictions, sorted(expected_ids)


def recompute_and_validate_metrics(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    tolerance: float,
) -> None:
    metric_index = metrics.set_index("model")
    for model in MODELS:
        pred = predictions[predictions["model"].astype(str) == model]
        y = pd.to_numeric(pred["true_label"], errors="raise").to_numpy(int)
        probability = pd.to_numeric(pred["probability_ILD"], errors="raise").to_numpy(float)
        predicted = pd.to_numeric(pred["predicted_label"], errors="raise").to_numpy(int)
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if tn + fp else np.nan
        expected = {
            "roc_auc": float(roc_auc_score(y, probability)),
            "pr_auc": float(average_precision_score(y, probability)),
            "accuracy": float(accuracy_score(y, predicted)),
            "sensitivity_recall": float(recall_score(y, predicted, zero_division=0)),
            "specificity": float(specificity),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "f1": float(f1_score(y, predicted, zero_division=0)),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        }
        row = metric_index.loc[model]
        for column, value in expected.items():
            if column not in metrics.columns:
                continue
            observed = float(row[column])
            if not math.isclose(observed, float(value), rel_tol=0.0, abs_tol=tolerance):
                raise TaskValidationError(
                    f"metrics.{column} for {model} does not match predictions: "
                    f"observed={observed}, recomputed={value}"
                )


def validate_coefficients(
    coefficients: pd.DataFrame,
    repeat: int,
    fold: int,
    task_name: str,
) -> pd.DataFrame:
    required = ["model", "feature_name", "receptor", "coefficient", "nonzero"]
    require_columns(coefficients, required, "coefficients")
    if set(coefficients["model"].astype(str)) != MODEL_SET:
        raise TaskValidationError("coefficients model set is invalid")
    if coefficients.duplicated(["model", "feature_name"]).any():
        raise TaskValidationError("coefficients contain duplicate model/feature rows")
    numeric_series(coefficients, "coefficient", "coefficients")

    intercept = coefficients[
        coefficients["feature_name"].astype(str) == "__INTERCEPT__"
    ]
    counts = intercept.groupby("model").size().reindex(MODELS, fill_value=0)
    if not (counts == 1).all():
        raise TaskValidationError(f"each model must have one intercept: {counts.to_dict()}")
    if not (intercept["receptor"].astype(str).str.lower() == "none").all():
        raise TaskValidationError("intercept receptor must be none")

    non_intercept = coefficients[
        coefficients["feature_name"].astype(str) != "__INTERCEPT__"
    ]
    expected_receptor = np.where(
        non_intercept["feature_name"].astype(str).str.startswith("trb_"),
        "TRB",
        np.where(
            non_intercept["feature_name"].astype(str).str.startswith("igh_"),
            "IGH",
            "clinical",
        ),
    )
    if not (
        non_intercept["receptor"].astype(str).to_numpy() == expected_receptor
    ).all():
        raise TaskValidationError("coefficient receptor labels do not match feature prefixes")
    return add_or_validate_task_columns(coefficients, repeat, fold, task_name)


def extract_selected_parameters(
    selected: Mapping[str, Mapping[str, object]],
    repeat: int,
    fold: int,
    task_name: str,
    tolerance: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model in MODELS:
        values = selected[model]
        rows.append(
            {
                "outer_repeat": repeat,
                "outer_fold": fold,
                "task_name": task_name,
                "model": model,
                "l1_ratio_alpha": values.get("l1_ratio", values.get("alpha")),
                "lambda": values.get("lambda"),
                "threshold": values.get("threshold"),
                "inner_roc_auc": values.get("inner_roc_auc"),
                "inner_pr_auc": values.get("inner_pr_auc"),
            }
        )
    frame = pd.DataFrame(rows)
    for column in [
        "l1_ratio_alpha", "lambda", "threshold", "inner_roc_auc", "inner_pr_auc"
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any() or not np.isfinite(frame[column]).all():
            raise TaskValidationError(
                f"selected parameter {column} contains missing/non-finite values"
            )
    if not ((frame["l1_ratio_alpha"] > 0) & (frame["l1_ratio_alpha"] <= 1)).all():
        raise TaskValidationError("selected alpha is outside (0, 1]")
    if not (frame["lambda"] > 0).all():
        raise TaskValidationError("selected lambda must be > 0")
    if ((frame["threshold"] < -tolerance) | (frame["threshold"] > 1.0 + tolerance)).any():
        raise TaskValidationError("selected threshold is outside [0, 1] beyond tolerance")
    for column in ["inner_roc_auc", "inner_pr_auc"]:
        if ((frame[column] < -tolerance) | (frame[column] > 1.0 + tolerance)).any():
            raise TaskValidationError(f"selected {column} is outside [0, 1] beyond tolerance")
    return frame


def validate_parameter_consistency(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    parameters: pd.DataFrame,
    tolerance: float,
) -> None:
    p = parameters.set_index("model")
    m = metrics.set_index("model")
    for model in MODELS:
        pred = predictions[predictions["model"].astype(str) == model]
        thresholds = pd.to_numeric(pred["threshold"], errors="raise").unique()
        if len(thresholds) != 1:
            raise TaskValidationError(f"prediction thresholds are not constant for {model}")
        checks = [
            (float(thresholds[0]), float(p.loc[model, "threshold"]), "threshold"),
        ]
        metric_pairs = [
            ("selected_l1_ratio_alpha", "l1_ratio_alpha"),
            ("selected_lambda", "lambda"),
            ("inner_selected_roc_auc", "inner_roc_auc"),
            ("inner_selected_pr_auc", "inner_pr_auc"),
            ("threshold", "threshold"),
        ]
        for metric_col, param_col in metric_pairs:
            if metric_col in metrics.columns:
                checks.append(
                    (float(m.loc[model, metric_col]), float(p.loc[model, param_col]), metric_col)
                )
        for observed, expected, label in checks:
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=tolerance):
                raise TaskValidationError(
                    f"selected parameter mismatch for {model}.{label}: "
                    f"observed={observed}, expected={expected}"
                )


def validate_global_collection(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    coefficients: pd.DataFrame,
    parameters: pd.DataFrame,
    canonical_mapping: pd.DataFrame,
    task_expectations: Mapping[Tuple[int, int], pd.DataFrame],
    args: argparse.Namespace,
) -> List[str]:
    checks: List[str] = []
    expected_tasks = args.expected_repeats * args.expected_folds
    expected_metric_rows = expected_tasks * len(MODELS)
    expected_prediction_rows = args.expected_repeats * args.expected_samples * len(MODELS)

    if len(metrics) != expected_metric_rows:
        raise RuntimeError(
            f"combined metrics expected {expected_metric_rows} rows, observed {len(metrics)}"
        )
    checks.append(f"Combined metrics contain exactly {expected_metric_rows} model/task rows.")

    if len(predictions) != expected_prediction_rows:
        raise RuntimeError(
            f"combined predictions expected {expected_prediction_rows} rows, observed {len(predictions)}"
        )
    checks.append(f"Combined predictions contain exactly {expected_prediction_rows} rows.")

    if len(parameters) != expected_metric_rows:
        raise RuntimeError(
            f"combined selected parameters expected {expected_metric_rows} rows, observed {len(parameters)}"
        )
    checks.append(
        f"Combined selected parameters contain exactly {expected_metric_rows} rows."
    )

    if metrics.duplicated(["outer_repeat", "outer_fold", "model"]).any():
        raise RuntimeError("duplicate metric task/model rows detected")
    if parameters.duplicated(["outer_repeat", "outer_fold", "model"]).any():
        raise RuntimeError("duplicate selected-parameter task/model rows detected")
    if predictions.duplicated(
        ["outer_repeat", "outer_fold", "model", args.sample_id_col]
    ).any():
        raise RuntimeError("duplicate global prediction rows detected")
    if coefficients.duplicated(
        ["outer_repeat", "outer_fold", "model", "feature_name"]
    ).any():
        raise RuntimeError("duplicate global coefficient rows detected")
    checks.append("No duplicate task/model/patient or task/model/feature rows were detected.")

    for (repeat, fold), expected in task_expectations.items():
        expected_ids = set(expected[args.sample_id_col].astype(str))
        for model in MODELS:
            observed_ids = set(
                predictions.loc[
                    (predictions["outer_repeat"] == repeat)
                    & (predictions["outer_fold"] == fold)
                    & (predictions["model"].astype(str) == model),
                    args.sample_id_col,
                ].astype(str)
            )
            if observed_ids != expected_ids:
                raise RuntimeError(
                    f"combined prediction validation set mismatch: repeat={repeat}, fold={fold}, model={model}"
                )
    checks.append("Every task/model prediction set exactly matches its frozen validation patients.")

    oof_counts = predictions.groupby(["outer_repeat", "model"])[args.sample_id_col].nunique()
    if not (oof_counts == args.expected_samples).all():
        bad = oof_counts[oof_counts != args.expected_samples].to_dict()
        raise RuntimeError(f"repeat/model OOF patient counts are invalid: {bad}")
    row_counts = predictions.groupby(["outer_repeat", "model"]).size()
    if not (row_counts == args.expected_samples).all():
        bad = row_counts[row_counts != args.expected_samples].to_dict()
        raise RuntimeError(f"repeat/model OOF row counts are invalid: {bad}")
    checks.append(
        f"Every repeat/model has exactly one OOF prediction for all {args.expected_samples} patients."
    )

    canonical_ids = set(canonical_mapping[args.sample_id_col].astype(str))
    grouped_sets = {
        key: set(group[args.sample_id_col].astype(str))
        for key, group in predictions.groupby(["outer_repeat", "model"])
    }
    if any(patient_set != canonical_ids for patient_set in grouped_sets.values()):
        raise RuntimeError("OOF patient universes differ across repeats/models")
    checks.append("All repeats and models use the same fixed 118-patient training cohort.")

    truth = canonical_mapping.set_index(args.sample_id_col)
    pred_unique = predictions[
        [args.sample_id_col, args.trb_id_col, args.igh_id_col, "true_cohort", "true_label"]
    ].drop_duplicates()
    if pred_unique[args.sample_id_col].duplicated().any():
        raise RuntimeError("a patient has inconsistent labels or library IDs across tasks")
    pred_unique = pred_unique.set_index(args.sample_id_col).reindex(truth.index)
    if pred_unique.isna().any().any():
        raise RuntimeError("global prediction mapping is missing one or more patients")
    if not (
        pred_unique[args.trb_id_col].astype(str).to_numpy()
        == truth[args.trb_id_col].astype(str).to_numpy()
    ).all():
        raise RuntimeError("global TRB library mapping is inconsistent")
    if not (
        pred_unique[args.igh_id_col].astype(str).to_numpy()
        == truth[args.igh_id_col].astype(str).to_numpy()
    ).all():
        raise RuntimeError("global IGH library mapping is inconsistent")
    if not (
        pred_unique["true_cohort"].astype(str).str.upper().to_numpy()
        == truth[args.label_col].astype(str).str.upper().to_numpy()
    ).all():
        raise RuntimeError("global true cohorts are inconsistent")
    checks.append("Patient labels and paired TRB/IGH library mappings are globally consistent.")

    metric_task_counts = metrics.groupby(["outer_repeat", "outer_fold"]).size()
    parameter_task_counts = parameters.groupby(["outer_repeat", "outer_fold"]).size()
    if not (metric_task_counts == len(MODELS)).all():
        raise RuntimeError("one or more tasks do not have exactly four metric rows")
    if not (parameter_task_counts == len(MODELS)).all():
        raise RuntimeError("one or more tasks do not have exactly four parameter rows")
    checks.append("Every task has exactly four metrics rows and four selected-parameter rows.")
    return checks


def write_summary(
    path: Path,
    args: argparse.Namespace,
    input_root: Path,
    outer_path: Path,
    outer_sha256: str,
    integrity: pd.DataFrame,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    coefficients: pd.DataFrame,
    parameters: pd.DataFrame,
    global_checks: Sequence[str],
    global_error: str,
) -> None:
    passed = int((integrity["status"] == "PASS").sum())
    failed = int((integrity["status"] == "FAIL").sum())
    expected_tasks = args.expected_repeats * args.expected_folds
    valid_sizes = sorted(
        pd.to_numeric(integrity.loc[integrity["status"] == "PASS", "n_validation_samples"], errors="coerce")
        .dropna().astype(int).unique().tolist()
    )
    lines = [
        "# 06 Paired TRB+IGH CV Result Collection Summary",
        "",
        "## Scope",
        "",
        f"- Script version: `{SCRIPT_VERSION}`",
        f"- Analysis view: **{ANALYSIS_VIEW}**",
        f"- CV unit: **{CV_UNIT}**",
        f"- Input root: `{input_root}`",
        f"- Frozen outer assignments: `{outer_path}`",
        f"- Frozen outer SHA256: `{outer_sha256}`",
        f"- Expected tasks: **{expected_tasks}**",
        f"- Expected repeats/folds: **{args.expected_repeats} × {args.expected_folds}**",
        f"- Fixed training patients: **{args.expected_samples}**",
        f"- Public feature set per receptor: `{args.expected_public_feature_set}`",
        "- Independent test set: **not read**",
        "",
        "## Task integrity",
        "",
        f"- Passed tasks: **{passed}**",
        f"- Failed tasks: **{failed}**",
        f"- Observed validation fold sizes: **{valid_sizes}**",
        "",
        "## Collected dimensions",
        "",
        f"- Metrics rows: **{len(metrics):,}**",
        f"- Prediction rows: **{len(predictions):,}**",
        f"- Coefficient rows: **{len(coefficients):,}**",
        f"- Selected-parameter rows: **{len(parameters):,}**",
        "",
        "## Collection-level integrity checks",
        "",
    ]
    if global_error:
        lines.append(f"- **FAIL:** {global_error}")
    elif global_checks:
        lines.extend(f"- PASS: {item}" for item in global_checks)
    else:
        lines.append("- Not run because incomplete collection was allowed.")

    lines.extend(["", "## Failed tasks", ""])
    failed_rows = integrity[integrity["status"] == "FAIL"]
    if failed_rows.empty:
        lines.append("- None")
    else:
        for _, row in failed_rows.iterrows():
            lines.append(f"- `{row['task_name']}`: {row['message']}")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- These files contain the 100 repeated outer-CV task results, not independent-test results.",
            "- Within each repeat, every patient has one OOF prediction per model.",
            "- Performance summaries should be calculated from patient-level OOF predictions.",
            "- Do not treat the 100 folds as 100 independent biological cohorts.",
            "- TRB-only, IGH-only and paired comparisons must use the same frozen patient splits.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    outer_path = Path(args.outer_assignments).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    canonical_mapping, task_expectations, outer_sha256 = load_frozen_outer(args, outer_path)

    output_paths = {key: output_dir / name for key, name in OUTPUT_FILENAMES.items()}
    if not args.preflight_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        existing = [path for path in output_paths.values() if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(
                "Outputs already exist; add --overwrite:\n"
                + "\n".join(f"  - {path}" for path in existing)
            )

    print("=" * 88)
    print("06 Paired TRB+IGH Repeated-CV Result Collector")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Analysis view: {ANALYSIS_VIEW}")
    print(f"CV unit: {CV_UNIT}")
    print(f"Input root: {input_root}")
    print(f"Frozen outer assignments: {outer_path}")
    print(f"Frozen outer SHA256: {outer_sha256}")
    print(f"Expected tasks: {args.expected_repeats * args.expected_folds}")
    print(f"Expected patients: {args.expected_samples}")
    print(f"Public feature set per receptor: {args.expected_public_feature_set}")
    print("Independent test set: NOT READ")
    print("")

    integrity_rows: List[Dict[str, object]] = []
    metric_parts: List[pd.DataFrame] = []
    prediction_parts: List[pd.DataFrame] = []
    coefficient_parts: List[pd.DataFrame] = []
    parameter_parts: List[pd.DataFrame] = []

    for repeat in range(1, args.expected_repeats + 1):
        for fold in range(1, args.expected_folds + 1):
            task_name = f"repeat_{repeat:02d}_fold_{fold:02d}"
            task_dir = input_root / task_name
            expected_valid = task_expectations[(repeat, fold)]
            n_valid = len(expected_valid)
            n_train = args.expected_samples - n_valid
            row: Dict[str, object] = {
                "outer_repeat": repeat,
                "outer_fold": fold,
                "task_name": task_name,
                "task_dir": str(task_dir),
                "status": "FAIL",
                "message": "",
                "expected_training_samples": n_train,
                "expected_validation_samples": n_valid,
                "n_validation_samples": "",
                "metric_rows": "",
                "prediction_rows": "",
                "coefficient_rows": "",
                "parameter_rows": "",
            }

            missing = [
                filename for filename in REQUIRED_TASK_FILES
                if not (task_dir / filename).is_file()
            ]
            if missing:
                row["message"] = f"missing files: {missing}"
                integrity_rows.append(row)
                continue

            try:
                config = json.loads(
                    (task_dir / "05_trial_configuration.json").read_text(encoding="utf-8")
                )
                selected = validate_configuration(
                    config, repeat, fold, n_train, n_valid, outer_sha256, args
                )
                metrics = validate_metrics(
                    pd.read_csv(task_dir / "05_outer_validation_metrics.csv"),
                    repeat, fold, task_name, n_train, n_valid, args.metric_tolerance,
                )
                predictions, validation_ids = validate_predictions(
                    pd.read_csv(task_dir / "05_outer_validation_predictions.csv"),
                    expected_valid, repeat, fold, task_name, args,
                )
                coefficients = validate_coefficients(
                    pd.read_csv(task_dir / "05_final_model_coefficients.csv"),
                    repeat, fold, task_name,
                )
                parameters = extract_selected_parameters(
                    selected, repeat, fold, task_name, args.metric_tolerance
                )
                recompute_and_validate_metrics(metrics, predictions, args.metric_tolerance)
                validate_parameter_consistency(
                    metrics, predictions, parameters, args.metric_tolerance
                )

                row.update(
                    status="PASS",
                    message="all paired task checks passed",
                    n_validation_samples=len(validation_ids),
                    metric_rows=len(metrics),
                    prediction_rows=len(predictions),
                    coefficient_rows=len(coefficients),
                    parameter_rows=len(parameters),
                )
                metric_parts.append(metrics)
                prediction_parts.append(predictions)
                coefficient_parts.append(coefficients)
                parameter_parts.append(parameters)
            except Exception as exc:
                row["message"] = f"read/validation error: {exc}"
            integrity_rows.append(row)

        if repeat == 1 or repeat % 5 == 0 or repeat == args.expected_repeats:
            passed_so_far = sum(r["status"] == "PASS" for r in integrity_rows)
            print(
                f"[repeat {repeat:>2}/{args.expected_repeats}] "
                f"tasks checked={len(integrity_rows)}, passed={passed_so_far}",
                flush=True,
            )

    integrity = pd.DataFrame(integrity_rows).sort_values(["outer_repeat", "outer_fold"])
    metrics = pd.concat(metric_parts, ignore_index=True) if metric_parts else pd.DataFrame()
    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    coefficients = pd.concat(coefficient_parts, ignore_index=True) if coefficient_parts else pd.DataFrame()
    parameters = pd.concat(parameter_parts, ignore_index=True) if parameter_parts else pd.DataFrame()

    for frame in [metrics, predictions, coefficients, parameters]:
        if not frame.empty:
            frame.sort_values(
                [c for c in ["outer_repeat", "outer_fold", "model", args.sample_id_col, "feature_name"] if c in frame.columns],
                inplace=True,
                kind="stable",
            )
            frame.reset_index(drop=True, inplace=True)

    passed = int((integrity["status"] == "PASS").sum())
    failed = int((integrity["status"] == "FAIL").sum())
    expected_tasks = args.expected_repeats * args.expected_folds

    global_checks: List[str] = []
    global_error = ""
    if failed == 0 and passed == expected_tasks:
        try:
            global_checks = validate_global_collection(
                metrics, predictions, coefficients, parameters,
                canonical_mapping, task_expectations, args,
            )
        except Exception as exc:
            global_error = str(exc)
    elif not args.allow_incomplete:
        global_error = (
            f"task collection incomplete: passed={passed}, failed={failed}, expected={expected_tasks}"
        )

    print("")
    print("[Collection completed]")
    print(f"Passed tasks: {passed}")
    print(f"Failed tasks: {failed}")
    print(f"Metrics rows: {len(metrics):,}")
    print(f"Prediction rows: {len(predictions):,}")
    print(f"Coefficient rows: {len(coefficients):,}")
    print(f"Parameter rows: {len(parameters):,}")
    if global_error:
        print(f"Global integrity: FAIL - {global_error}")
    elif global_checks:
        print("Global integrity: PASS")
        for check in global_checks:
            print(f"- PASS: {check}")
    else:
        print("Global integrity: not run (incomplete collection allowed)")

    if args.preflight_only:
        print("Preflight-only: no output files were written.")
    else:
        integrity.to_csv(output_paths["integrity"], index=False)
        metrics.to_csv(output_paths["metrics"], index=False)
        predictions.to_csv(output_paths["predictions"], index=False, compression="gzip")
        coefficients.to_csv(output_paths["coefficients"], index=False, compression="gzip")
        parameters.to_csv(output_paths["parameters"], index=False)
        write_summary(
            output_paths["summary"], args, input_root, outer_path, outer_sha256,
            integrity, metrics, predictions, coefficients, parameters,
            global_checks, global_error,
        )
        print("[Output files]")
        for key, path in output_paths.items():
            print(f"- {key}: {path}")

    strict_failed = (failed > 0 or bool(global_error)) and not args.allow_incomplete
    return 1 if strict_failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
