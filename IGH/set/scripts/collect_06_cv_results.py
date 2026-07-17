#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Collect and validate all repeated nested-CV outer-task results for IGH.

The collector expects the leakage-controlled LOO workflow produced by
``run_05_single_outer_trial_loo.py``. By default it collects 20 repeats ×
5 outer folds = 100 tasks and writes combined metrics, predictions,
coefficients, selected parameters, and an integrity report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.1.0-IGH"
EXPECTED_RECEPTOR = "IGH"

ROOT = Path("/data/users/chenhaisheng/RA-ILD/IGH")
DEFAULT_INPUT = ROOT / "set/train/result/05_modeling/single_outer_trial_loo"
DEFAULT_OUTPUT = ROOT / "set/train/result/06_cv_result_summary/collected"

MODELS = [
    "M0_clinical",
    "M1_static_igh",
    "M2_static_igh_public",
    "M3_static_igh_public_material",
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
        description="Collect all IGH LOO outer-task CV results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--expected-repeats", type=int, default=20)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--expected-samples", type=int, default=120)
    parser.add_argument("--expected-validation-samples", type=int, default=24)
    parser.add_argument("--expected-training-samples", type=int, default=96)
    parser.add_argument(
        "--expected-public-feature-set",
        default="raw_bilateral",
        choices=("raw_bilateral", "log_ratio", "all18"),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write partial collections and return success even if tasks are missing/invalid.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for name in (
        "expected_repeats",
        "expected_folds",
        "expected_samples",
        "expected_validation_samples",
        "expected_training_samples",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    return args


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
    name: str,
) -> pd.DataFrame:
    frame = frame.copy()
    checks = [
        ("outer_repeat", repeat),
        ("outer_fold", fold),
        ("task_name", name),
    ]
    for column, expected in checks:
        if column not in frame.columns:
            frame.insert(0, column, expected)
            continue
        if column in {"outer_repeat", "outer_fold"}:
            observed = pd.to_numeric(frame[column], errors="coerce")
            if observed.isna().any() or not (observed.astype(int) == int(expected)).all():
                raise TaskValidationError(
                    f"{column} values do not match task directory {name}"
                )
            frame[column] = observed.astype(int)
        elif not (frame[column].astype(str) == str(expected)).all():
            raise TaskValidationError(
                f"{column} values do not match task directory {name}"
            )

    ordered = ["outer_repeat", "outer_fold", "task_name"]
    return frame[ordered + [column for column in frame.columns if column not in ordered]]


def validate_configuration(
    config: Mapping[str, object],
    repeat: int,
    fold: int,
    args: argparse.Namespace,
) -> Mapping[str, Mapping[str, object]]:
    problems: List[str] = []

    if str(config.get("receptor", "")).upper() != EXPECTED_RECEPTOR:
        problems.append(
            f"receptor={config.get('receptor')!r}, expected {EXPECTED_RECEPTOR}"
        )
    if int(config.get("outer_repeat", -1)) != repeat:
        problems.append("outer_repeat mismatch")
    if int(config.get("outer_fold", -1)) != fold:
        problems.append("outer_fold mismatch")
    if int(config.get("n_outer_train", -1)) != args.expected_training_samples:
        problems.append(
            f"n_outer_train={config.get('n_outer_train')}, "
            f"expected {args.expected_training_samples}"
        )
    if int(config.get("n_outer_validation", -1)) != args.expected_validation_samples:
        problems.append(
            f"n_outer_validation={config.get('n_outer_validation')}, "
            f"expected {args.expected_validation_samples}"
        )
    if config.get("public_feature_set") != args.expected_public_feature_set:
        problems.append(
            f"public_feature_set={config.get('public_feature_set')!r}, "
            f"expected {args.expected_public_feature_set!r}"
        )

    configured_models = config.get("models", {})
    if not isinstance(configured_models, Mapping) or set(configured_models) != MODEL_SET:
        problems.append("configuration model set is not the four expected IGH models")

    selected = config.get("selected_parameters", {})
    if not isinstance(selected, Mapping) or set(selected) != MODEL_SET:
        problems.append("selected_parameters model set is invalid")

    if problems:
        raise TaskValidationError("configuration: " + "; ".join(problems))
    return selected  # type: ignore[return-value]


def validate_metrics(
    metrics: pd.DataFrame,
    repeat: int,
    fold: int,
    task_name: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    required = [
        "model",
        "roc_auc",
        "pr_auc",
        "fit_converged",
        "n_outer_train",
        "n_outer_validation",
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
        if not values.between(0.0, 1.0).all():
            raise TaskValidationError(f"metrics.{column} is outside [0, 1]")

    train_n = numeric_series(metrics, "n_outer_train", "metrics").astype(int)
    valid_n = numeric_series(metrics, "n_outer_validation", "metrics").astype(int)
    if not (train_n == args.expected_training_samples).all():
        raise TaskValidationError("metrics n_outer_train mismatch")
    if not (valid_n == args.expected_validation_samples).all():
        raise TaskValidationError("metrics n_outer_validation mismatch")

    return add_or_validate_task_columns(metrics, repeat, fold, task_name)


def validate_predictions(
    predictions: pd.DataFrame,
    repeat: int,
    fold: int,
    task_name: str,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, List[str]]:
    required = [
        "model",
        "sample_id",
        "true_label",
        "probability_ILD",
        "threshold",
        "predicted_label",
    ]
    require_columns(predictions, required, "predictions")

    if set(predictions["model"].astype(str)) != MODEL_SET:
        raise TaskValidationError("predictions model set is invalid")
    if predictions.duplicated(["model", "sample_id"]).any():
        raise TaskValidationError("predictions contain duplicate model/sample_id rows")

    expected_rows = len(MODELS) * args.expected_validation_samples
    if len(predictions) != expected_rows:
        raise TaskValidationError(
            f"predictions expected {expected_rows} rows, observed {len(predictions)}"
        )

    model_counts = predictions.groupby("model")["sample_id"].nunique()
    if set(model_counts.index.astype(str)) != MODEL_SET or not (
        model_counts == args.expected_validation_samples
    ).all():
        raise TaskValidationError(
            f"prediction counts per model invalid: {model_counts.to_dict()}"
        )

    sample_sets = [
        set(group["sample_id"].astype(str))
        for _, group in predictions.groupby("model", sort=False)
    ]
    if len(sample_sets) != len(MODELS) or any(
        sample_set != sample_sets[0] for sample_set in sample_sets[1:]
    ):
        raise TaskValidationError("prediction sample sets differ among models")

    labels = numeric_series(predictions, "true_label", "predictions").astype(int)
    predicted = numeric_series(
        predictions, "predicted_label", "predictions"
    ).astype(int)
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

    # Labels for the same validation sample must agree across all four models.
    label_counts = predictions.groupby("sample_id")["true_label"].nunique()
    if not (label_counts == 1).all():
        raise TaskValidationError("true_label differs among models for a sample")

    predictions = add_or_validate_task_columns(
        predictions, repeat, fold, task_name
    )
    return predictions, sorted(sample_sets[0])


def validate_coefficients(
    coefficients: pd.DataFrame,
    repeat: int,
    fold: int,
    task_name: str,
) -> pd.DataFrame:
    required = ["model", "feature_name", "coefficient", "nonzero"]
    require_columns(coefficients, required, "coefficients")

    if set(coefficients["model"].astype(str)) != MODEL_SET:
        raise TaskValidationError("coefficients model set is invalid")
    if coefficients.duplicated(["model", "feature_name"]).any():
        raise TaskValidationError("coefficients contain duplicate model/feature rows")
    numeric_series(coefficients, "coefficient", "coefficients")

    intercept_counts = (
        coefficients.loc[
            coefficients["feature_name"].astype(str) == "__INTERCEPT__"
        ]
        .groupby("model")
        .size()
        .reindex(MODELS, fill_value=0)
    )
    if not (intercept_counts == 1).all():
        raise TaskValidationError(
            f"each model must have one intercept: {intercept_counts.to_dict()}"
        )

    return add_or_validate_task_columns(
        coefficients, repeat, fold, task_name
    )


def extract_selected_parameters(
    selected: Mapping[str, Mapping[str, object]],
    repeat: int,
    fold: int,
    task_name: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model in MODELS:
        values = selected[model]
        alpha = values.get("alpha", values.get("l1_ratio"))
        record = {
            "outer_repeat": repeat,
            "outer_fold": fold,
            "task_name": task_name,
            "model": model,
            "l1_ratio_alpha": alpha,
            "lambda": values.get("lambda"),
            "threshold": values.get("threshold"),
            "inner_roc_auc": values.get("inner_roc_auc"),
            "inner_pr_auc": values.get("inner_pr_auc"),
        }
        rows.append(record)

    frame = pd.DataFrame(rows)
    for column in (
        "l1_ratio_alpha",
        "lambda",
        "threshold",
        "inner_roc_auc",
        "inner_pr_auc",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any() or not np.isfinite(frame[column]).all():
            raise TaskValidationError(
                f"selected parameter {column} contains missing/non-finite values"
            )

    if not frame["l1_ratio_alpha"].between(0.0, 1.0, inclusive="right").all():
        raise TaskValidationError("selected alpha is outside (0, 1]")
    if not (frame["lambda"] > 0).all():
        raise TaskValidationError("selected lambda must be > 0")
    if not frame["threshold"].between(0.0, 1.0).all():
        raise TaskValidationError("selected threshold is outside [0, 1]")
    for column in ("inner_roc_auc", "inner_pr_auc"):
        if not frame[column].between(0.0, 1.0).all():
            raise TaskValidationError(f"selected {column} is outside [0, 1]")
    return frame


def validate_global_collection(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    parameters: pd.DataFrame,
    args: argparse.Namespace,
) -> List[str]:
    checks: List[str] = []
    expected_tasks = args.expected_repeats * args.expected_folds
    expected_metric_rows = expected_tasks * len(MODELS)
    expected_prediction_rows = (
        args.expected_repeats * args.expected_samples * len(MODELS)
    )
    expected_parameter_rows = expected_metric_rows

    if len(metrics) != expected_metric_rows:
        raise RuntimeError(
            f"combined metrics expected {expected_metric_rows} rows, observed {len(metrics)}"
        )
    checks.append(f"Combined metrics contain exactly {expected_metric_rows} model/task rows.")

    if len(predictions) != expected_prediction_rows:
        raise RuntimeError(
            "combined predictions expected "
            f"{expected_prediction_rows} rows, observed {len(predictions)}"
        )
    checks.append(
        f"Combined predictions contain exactly {expected_prediction_rows} rows."
    )

    if len(parameters) != expected_parameter_rows:
        raise RuntimeError(
            "combined selected parameters expected "
            f"{expected_parameter_rows} rows, observed {len(parameters)}"
        )
    checks.append(
        f"Combined selected parameters contain exactly {expected_parameter_rows} rows."
    )

    if predictions.duplicated(
        ["outer_repeat", "outer_fold", "model", "sample_id"]
    ).any():
        raise RuntimeError("duplicate global prediction rows detected")
    checks.append("No duplicate repeat/fold/model/sample prediction rows were detected.")

    oof_counts = predictions.groupby(["outer_repeat", "model"])[
        "sample_id"
    ].nunique()
    if not (oof_counts == args.expected_samples).all():
        bad = oof_counts[oof_counts != args.expected_samples].to_dict()
        raise RuntimeError(f"repeat/model OOF sample counts are invalid: {bad}")
    checks.append(
        f"Every repeat/model has one OOF prediction for all {args.expected_samples} samples."
    )

    row_counts = predictions.groupby(["outer_repeat", "model"]).size()
    if not (row_counts == args.expected_samples).all():
        bad = row_counts[row_counts != args.expected_samples].to_dict()
        raise RuntimeError(f"repeat/model OOF row counts are invalid: {bad}")

    # All repeats and models must cover exactly the same fixed training cohort.
    grouped_sets = {
        key: frozenset(group["sample_id"].astype(str))
        for key, group in predictions.groupby(["outer_repeat", "model"])
    }
    cohort_sets = list(grouped_sets.values())
    if not cohort_sets or any(current != cohort_sets[0] for current in cohort_sets[1:]):
        raise RuntimeError("OOF sample universes differ across repeats/models")
    checks.append("All repeats and models use the same fixed 120-sample training cohort.")

    label_map = predictions[["sample_id", "true_label"]].drop_duplicates()
    if label_map["sample_id"].duplicated().any():
        raise RuntimeError("a sample has inconsistent true labels across tasks")
    if len(label_map) != args.expected_samples:
        raise RuntimeError(
            f"expected labels for {args.expected_samples} samples, observed {len(label_map)}"
        )
    checks.append("True labels are consistent for every sample across all repeated tasks.")

    metric_task_counts = metrics.groupby(["outer_repeat", "outer_fold"]).size()
    if not (metric_task_counts == len(MODELS)).all():
        raise RuntimeError("one or more tasks do not have exactly four metric rows")

    parameter_task_counts = parameters.groupby(["outer_repeat", "outer_fold"]).size()
    if not (parameter_task_counts == len(MODELS)).all():
        raise RuntimeError("one or more tasks do not have exactly four parameter rows")

    return checks


def write_summary(
    path: Path,
    args: argparse.Namespace,
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

    lines = [
        "# 06 IGH CV Result Collection Summary",
        "",
        "## Scope",
        "",
        f"- Script version: `{SCRIPT_VERSION}`",
        f"- Receptor: **{EXPECTED_RECEPTOR}**",
        f"- Input root: `{Path(args.input_root).expanduser().resolve()}`",
        f"- Expected tasks: **{expected_tasks}**",
        f"- Expected repeats/folds: **{args.expected_repeats} × {args.expected_folds}**",
        f"- Expected training samples: **{args.expected_samples}**",
        f"- Public feature set: `{args.expected_public_feature_set}`",
        "",
        "## Task integrity",
        "",
        f"- Passed tasks: **{passed}**",
        f"- Failed tasks: **{failed}**",
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

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- These files contain the 100 outer-task results; they are not the independent-test results.",
        "- Each repeat should contain one out-of-fold prediction per training sample and model.",
        "- Subsequent performance summaries should be calculated from the combined OOF predictions, not by treating the 100 folds as independent patients.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        key: output_dir / filename for key, filename in OUTPUT_FILENAMES.items()
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Outputs already exist; add --overwrite:\n"
            + "\n".join(f"  - {path}" for path in existing)
        )

    print("=" * 80)
    print("06 IGH Repeated-CV Result Collector")
    print("=" * 80)
    print(f"Input root: {input_root}")
    print(f"Output directory: {output_dir}")
    print(f"Expected tasks: {args.expected_repeats * args.expected_folds}")
    print(f"Expected receptor: {EXPECTED_RECEPTOR}")
    print(f"Public feature set: {args.expected_public_feature_set}")
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
            row: Dict[str, object] = {
                "outer_repeat": repeat,
                "outer_fold": fold,
                "task_name": task_name,
                "task_dir": str(task_dir),
                "status": "FAIL",
                "message": "",
                "n_validation_samples": "",
                "metric_rows": "",
                "prediction_rows": "",
                "coefficient_rows": "",
                "parameter_rows": "",
            }

            missing = [
                filename
                for filename in REQUIRED_TASK_FILES
                if not (task_dir / filename).is_file()
            ]
            if missing:
                row["message"] = f"missing files: {missing}"
                integrity_rows.append(row)
                continue

            try:
                config = json.loads(
                    (task_dir / "05_trial_configuration.json").read_text(
                        encoding="utf-8"
                    )
                )
                selected = validate_configuration(
                    config, repeat, fold, args
                )

                metrics = validate_metrics(
                    pd.read_csv(task_dir / "05_outer_validation_metrics.csv"),
                    repeat,
                    fold,
                    task_name,
                    args,
                )
                predictions, validation_ids = validate_predictions(
                    pd.read_csv(task_dir / "05_outer_validation_predictions.csv"),
                    repeat,
                    fold,
                    task_name,
                    args,
                )
                coefficients = validate_coefficients(
                    pd.read_csv(task_dir / "05_final_model_coefficients.csv"),
                    repeat,
                    fold,
                    task_name,
                )
                parameters = extract_selected_parameters(
                    selected, repeat, fold, task_name
                )

                row.update(
                    status="PASS",
                    message="all task checks passed",
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
            passed_so_far = sum(
                row["status"] == "PASS" for row in integrity_rows
            )
            print(
                f"[repeat {repeat:>2}/{args.expected_repeats}] "
                f"tasks checked={len(integrity_rows)}, passed={passed_so_far}",
                flush=True,
            )

    integrity = pd.DataFrame(integrity_rows).sort_values(
        ["outer_repeat", "outer_fold"]
    )
    metrics = (
        pd.concat(metric_parts, ignore_index=True)
        if metric_parts
        else pd.DataFrame()
    )
    predictions = (
        pd.concat(prediction_parts, ignore_index=True)
        if prediction_parts
        else pd.DataFrame()
    )
    coefficients = (
        pd.concat(coefficient_parts, ignore_index=True)
        if coefficient_parts
        else pd.DataFrame()
    )
    parameters = (
        pd.concat(parameter_parts, ignore_index=True)
        if parameter_parts
        else pd.DataFrame()
    )

    passed = int((integrity["status"] == "PASS").sum())
    failed = int((integrity["status"] == "FAIL").sum())
    expected_tasks = args.expected_repeats * args.expected_folds

    global_checks: List[str] = []
    global_error = ""
    if failed == 0 and passed == expected_tasks:
        try:
            global_checks = validate_global_collection(
                metrics, predictions, parameters, args
            )
        except Exception as exc:
            global_error = str(exc)
    elif not args.allow_incomplete:
        global_error = (
            f"task collection incomplete: passed={passed}, failed={failed}, "
            f"expected={expected_tasks}"
        )

    # Always write the integrity report and any valid partial collections.
    integrity.to_csv(output_paths["integrity"], index=False)
    metrics.to_csv(output_paths["metrics"], index=False)
    predictions.to_csv(
        output_paths["predictions"], index=False, compression="gzip"
    )
    coefficients.to_csv(
        output_paths["coefficients"], index=False, compression="gzip"
    )
    parameters.to_csv(output_paths["parameters"], index=False)
    write_summary(
        output_paths["summary"],
        args,
        integrity,
        metrics,
        predictions,
        coefficients,
        parameters,
        global_checks,
        global_error,
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

    print("[Output files]")
    for key, path in output_paths.items():
        print(f"- {key}: {path}")

    failed_strict = failed > 0 and not args.allow_incomplete
    global_failed = bool(global_error) and not args.allow_incomplete
    return 1 if failed_strict or global_failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
