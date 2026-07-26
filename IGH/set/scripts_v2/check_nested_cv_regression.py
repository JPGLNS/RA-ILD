#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression-check the V2 nested-CV engine against frozen V1 outer task 1/1.

Quick mode reconstructs selected parameters from V1 inner outputs and runs the
shared V2 outer stage (four model fits). Full mode additionally reruns every
inner candidate fit for the task and compares the complete tuning/OOF outputs.
No V1 file is modified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.config import load_experiment_config  # noqa: E402
from ra_ild_igh.nested_cv import (  # noqa: E402
    NestedCVOptions,
    SelectedTuning,
    fit_outer_models,
    run_nested_outer_task,
    validate_fixed_assignments,
)
from ra_ild_igh.public_reference import (  # noqa: E402
    ALL_PUBLIC_FEATURES,
    ThresholdScheme,
)
from ra_ild_igh.specifications import (  # noqa: E402
    build_hyperparameter_grid,
    rank_tuning_candidates,
    resolve_all_model_specifications,
    select_static_igh_features,
    validate_tuning_grid_frame,
)
from ra_ild_igh.thresholds import choose_threshold_youden  # noqa: E402


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str


def add(results: List[CheckResult], name: str, ok: bool, detail: str) -> None:
    results.append(CheckResult(name, "PASS" if ok else "FAIL", detail))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the IGH V2 nested-CV engine with frozen V1 outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--full", action="store_true", help="Rerun all 480 inner fits.")
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-12)
    return parser.parse_args()


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    frame = pd.read_csv(path)
    duplicates = frame.columns[frame.columns.duplicated()].tolist()
    if duplicates:
        raise ValueError(f"{label} has duplicated columns: {duplicates[:20]}")
    return frame


def as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def allclose_detail(left: np.ndarray, right: np.ndarray, rtol: float, atol: float) -> Tuple[bool, str]:
    observed = np.asarray(left, dtype=float)
    expected = np.asarray(right, dtype=float)
    if observed.shape != expected.shape:
        return False, f"shape V2={observed.shape}, V1={expected.shape}"
    difference = np.abs(observed - expected)
    maximum = float(np.max(difference)) if difference.size else 0.0
    return bool(np.allclose(observed, expected, rtol=rtol, atol=atol)), f"max_abs_diff={maximum:.12g}"


def compare_public(v2: pd.DataFrame, v1: pd.DataFrame, rtol: float, atol: float) -> Tuple[bool, str]:
    for frame, label in ((v2, "V2"), (v1, "V1")):
        if "sample_id" not in frame.columns:
            return False, f"{label} public frame lacks sample_id"
        if frame["sample_id"].astype(str).duplicated().any():
            return False, f"{label} public frame has duplicate sample_id"
    ids = v1["sample_id"].astype(str).tolist()
    if set(v2["sample_id"].astype(str)) != set(ids):
        return False, "sample ID sets differ"
    left = v2.assign(sample_id=v2["sample_id"].astype(str)).set_index("sample_id").loc[ids, list(ALL_PUBLIC_FEATURES)]
    right = v1.assign(sample_id=v1["sample_id"].astype(str)).set_index("sample_id").loc[ids, list(ALL_PUBLIC_FEATURES)]
    ok, detail = allclose_detail(left.to_numpy(float), right.to_numpy(float), rtol, atol)
    return ok, f"rows={len(ids)}, {detail}"


def compare_preprocessing(v2: pd.DataFrame, v1: pd.DataFrame, stage: str, rtol: float, atol: float) -> Tuple[bool, str]:
    left = v2.loc[v2["stage"].astype(str) == stage].copy()
    right = v1.loc[v1["stage"].astype(str) == stage].copy()
    keys = ["model", "inner_fold", "feature_name"] if stage == "inner" else ["model", "feature_name"]
    for frame in (left, right):
        frame["model"] = frame["model"].astype(str)
        frame["feature_name"] = frame["feature_name"].astype(str)
        if "inner_fold" in keys:
            frame["inner_fold"] = pd.to_numeric(frame["inner_fold"], errors="raise").astype(int)
    left = left.sort_values(keys, kind="mergesort").reset_index(drop=True)
    right = right.sort_values(keys, kind="mergesort").reset_index(drop=True)
    if left[keys].to_dict("records") != right[keys].to_dict("records"):
        return False, f"key/order mismatch V2={len(left)}, V1={len(right)}"
    type_ok = left["source_type"].astype(str).tolist() == right["source_type"].astype(str).tolist()
    keep_ok = np.array_equal(left["kept_after_zero_variance_filter"].map(as_bool), right["kept_after_zero_variance_filter"].map(as_bool))
    numeric_ok, detail = allclose_detail(
        left[["training_mean", "training_sd"]].to_numpy(float),
        right[["training_mean", "training_sd"]].to_numpy(float),
        rtol,
        atol,
    )
    return type_ok and keep_ok and numeric_ok, f"rows={len(left)}, type={type_ok}, keep={keep_ok}, {detail}"


def build_selected(v1_tuning: pd.DataFrame, v1_oof: pd.DataFrame, model_names: Sequence[str]) -> Tuple[Mapping[str, SelectedTuning], List[Tuple[str, bool, str]]]:
    selected: Dict[str, SelectedTuning] = {}
    checks = []
    for model in model_names:
        tuning = v1_tuning.loc[v1_tuning["model"].astype(str) == model].copy()
        best = rank_tuning_candidates(tuning).iloc[0]
        alpha = float(best["l1_ratio_alpha"])
        lambda_value = float(best["lambda"])
        oof = v1_oof.loc[v1_oof["model"].astype(str) == model].copy()
        pairs = oof[["l1_ratio_alpha", "lambda"]].astype(float).drop_duplicates()
        pair_ok = len(pairs) == 1 and math.isclose(float(pairs.iloc[0, 0]), alpha) and math.isclose(float(pairs.iloc[0, 1]), lambda_value)
        threshold = choose_threshold_youden(
            pd.to_numeric(oof["true_label"], errors="raise").to_numpy(int),
            pd.to_numeric(oof["probability"], errors="raise").to_numpy(float),
        )
        stored = pd.to_numeric(oof["selected_threshold"], errors="raise").to_numpy(float)
        threshold_ok = np.allclose(stored, threshold, rtol=1e-12, atol=1e-12)
        checks.append((model, pair_ok and threshold_ok, f"pair=({alpha:g},{lambda_value:g}), threshold={threshold:.16g}"))
        from ra_ild_igh.specifications import HyperparameterCandidate
        selected[model] = SelectedTuning(
            candidate=HyperparameterCandidate(alpha=alpha, lambda_value=lambda_value),
            threshold=float(threshold),
            inner_roc_auc=float(best["pooled_inner_roc_auc"]),
            inner_pr_auc=float(best["pooled_inner_pr_auc"]),
        )
    return MappingProxyType(selected), checks


def compare_outer_model(
    results: List[CheckResult],
    model: str,
    v2_predictions: pd.DataFrame,
    v2_metrics: pd.DataFrame,
    v2_coefficients: pd.DataFrame,
    v1_predictions: pd.DataFrame,
    v1_metrics: pd.DataFrame,
    v1_coefficients: pd.DataFrame,
    rtol: float,
    atol: float,
) -> None:
    left_pred = v2_predictions.loc[v2_predictions["model"] == model].copy()
    right_pred = v1_predictions.loc[v1_predictions["model"].astype(str) == model].copy()
    ids = right_pred["sample_id"].astype(str).tolist()
    left_pred["sample_id"] = left_pred["sample_id"].astype(str)
    right_pred["sample_id"] = right_pred["sample_id"].astype(str)
    if set(left_pred["sample_id"]) == set(ids):
        left_pred = left_pred.set_index("sample_id").loc[ids]
        right_pred = right_pred.set_index("sample_id").loc[ids]
        probability_ok, probability_detail = allclose_detail(
            left_pred["probability_ILD"], right_pred["probability_ILD"], rtol, atol
        )
    else:
        probability_ok, probability_detail = False, "sample IDs differ"
    add(results, f"{model} outer probability equality", probability_ok, probability_detail)

    labels_ok = probability_ok and np.array_equal(
        pd.to_numeric(left_pred["predicted_label"]).to_numpy(int),
        pd.to_numeric(right_pred["predicted_label"]).to_numpy(int),
    ) and np.allclose(left_pred["threshold"], right_pred["threshold"], rtol=rtol, atol=atol)
    add(results, f"{model} outer classification equality", labels_ok, f"n_rows={len(ids)}")

    left_metric = v2_metrics.loc[v2_metrics["model"] == model]
    right_metric = v1_metrics.loc[v1_metrics["model"].astype(str) == model]
    if len(left_metric) != 1 or len(right_metric) != 1:
        metric_ok, metric_detail = False, f"metric rows V2={len(left_metric)}, V1={len(right_metric)}"
    else:
        numeric = [
            "roc_auc", "pr_auc", "threshold", "accuracy", "sensitivity_recall",
            "specificity", "precision", "f1", "selected_l1_ratio_alpha",
            "selected_lambda", "inner_selected_roc_auc", "inner_selected_pr_auc",
        ]
        metric_ok, metric_detail = allclose_detail(
            left_metric[numeric].to_numpy(float), right_metric[numeric].to_numpy(float), rtol, atol
        )
        integer = ["TN", "FP", "FN", "TP", "n_outer_train", "n_outer_validation", "iterations_used", "n_final_predictors", "n_nonzero_coefficients"]
        metric_ok &= np.array_equal(left_metric[integer].to_numpy(int), right_metric[integer].to_numpy(int))
        metric_ok &= as_bool(left_metric.iloc[0]["fit_converged"]) == as_bool(right_metric.iloc[0]["fit_converged"])
        metric_detail += ", audit integers/bool compared"
    add(results, f"{model} outer metric equality", metric_ok, metric_detail)

    left_coef = v2_coefficients.loc[v2_coefficients["model"] == model].copy()
    right_coef = v1_coefficients.loc[v1_coefficients["model"].astype(str) == model].copy()
    names_ok = left_coef["feature_name"].astype(str).tolist() == right_coef["feature_name"].astype(str).tolist()
    if names_ok:
        coef_ok, coef_detail = allclose_detail(left_coef["coefficient"], right_coef["coefficient"], rtol, atol)
        coef_ok &= np.array_equal(left_coef["nonzero"].map(as_bool), right_coef["nonzero"].map(as_bool))
    else:
        coef_ok, coef_detail = False, f"feature order V2={len(left_coef)}, V1={len(right_coef)}"
    add(results, f"{model} coefficient equality", names_ok and coef_ok, coef_detail)


def main() -> int:
    args = parse_args()
    results: List[CheckResult] = []
    try:
        config = load_experiment_config(Path(args.config), repository_root=Path(args.repository_root) if args.repository_root else None)
        nested = config.section("nested_cv")
        task = nested["regression_task"]
        outer_repeat = int(task["outer_repeat"])
        outer_fold = int(task["outer_fold"])

        base = read_csv(config.path("data.train.base_matrix", must_exist=True, expect="file"), "base matrix")
        manifest = read_csv(config.path("data.train.feature_manifest", must_exist=True, expect="file"), "feature manifest")
        outer_assignments = read_csv(config.path("cross_validation.outer_assignments", must_exist=True, expect="file"), "outer assignments")
        inner_assignments = read_csv(config.path("cross_validation.inner_assignments", must_exist=True, expect="file"), "inner assignments")
        cache_meta = json.loads(config.path("public_reference.cache.metadata", must_exist=True, expect="file").read_text(encoding="utf-8"))
        sample_ids = [str(value) for value in cache_meta["sample_ids"]]
        base["sample_id"] = base["sample_id"].astype(str)
        order_ok = base["sample_id"].tolist() == sample_ids
        add(results, "Sparse cache/base exact sample order", order_ok, f"n_samples={len(sample_ids)}")
        if not order_ok:
            raise ValueError("Sparse cache and base order differ.")
        presence = sparse.load_npz(config.path("public_reference.cache.presence", must_exist=True, expect="file")).tocsr()
        frequency = sparse.load_npz(config.path("public_reference.cache.frequency", must_exist=True, expect="file")).tocsr()
        expected_shape = (len(base), int(config.raw["public_reference"]["expected_sizes"]["catalog"]))
        if presence.shape != expected_shape or frequency.shape != expected_shape:
            raise ValueError("Sparse cache shape mismatch.")

        split = validate_fixed_assignments(
            sample_ids, outer_assignments, inner_assignments,
            outer_repeat=outer_repeat, outer_fold=outer_fold,
            outer_folds=int(config.raw["cross_validation"]["outer_folds"]),
            inner_folds=int(config.raw["cross_validation"]["inner_folds"]),
        )
        task_outer = outer_assignments.loc[
            pd.to_numeric(
                outer_assignments["outer_repeat"],
                errors="raise",
            ).astype(int).eq(outer_repeat)
        ].copy()

        task_outer["sample_id"] = task_outer["sample_id"].astype(str)
        task_outer["outer_fold"] = pd.to_numeric(
            task_outer["outer_fold"],
            errors="raise",
        ).astype(int)

        expected_total_ids = set(task_outer["sample_id"])
        expected_validation_ids = set(
            task_outer.loc[
                task_outer["outer_fold"].eq(outer_fold),
                "sample_id",
            ]
        )
        expected_outer_valid = len(expected_validation_ids)
        expected_outer_train = len(
            expected_total_ids - expected_validation_ids
        )

        sample_count_ok = (
            split.n_outer_train == expected_outer_train
            and split.n_outer_valid == expected_outer_valid
            and split.n_outer_train + split.n_outer_valid == len(base)
        )

        add(
            results,
            "Outer task sample counts",
            sample_count_ok,
            (
                f"observed_train={split.n_outer_train}, "
                f"expected_train={expected_outer_train}, "
                f"observed_validation={split.n_outer_valid}, "
                f"expected_validation={expected_outer_valid}"
            ),
        )
        add(results, "Inner fixed-fold coverage", split.inner_folds == (1, 2, 3, 4, 5), f"folds={list(split.inner_folds)}")

        v1_static = read_csv(config.path("preprocessing.regression_task.static_feature_list", must_exist=True, expect="file"), "V1 static feature list")["feature_name"].astype(str).tolist()
        static_features = select_static_igh_features(manifest, available_columns=base.columns, require_all_available=True)
        add(results, "Static feature exact order", static_features == v1_static, f"V2={len(static_features)}, V1={len(v1_static)}")
        public = config.section("public_reference")
        model_specs = resolve_all_model_specifications(
            config.section("models"), static_features=static_features,
            allowed_dynamic_public=public["dynamic_features"],
            static_group_name=str(config.raw["model_selection"]["static_feature_group"]),
        )
        add(results, "Resolved model set", list(model_specs) == list(config.section("models")), f"models={list(model_specs)}")
        engine = config.section("model_engine")
        candidates = build_hyperparameter_grid(engine["alpha_grid"], engine["lambda_grid"])
        add(results, "Configured candidate grid", len(candidates) == 24, f"n_candidates={len(candidates)}")

        v1_tuning = read_csv(config.path("nested_cv.regression_task.inner_tuning_results", must_exist=True, expect="file"), "V1 inner tuning")
        validate_tuning_grid_frame(v1_tuning, model_names=list(model_specs), candidates=candidates)
        add(results, "V1 tuning exact grid coverage", len(v1_tuning) == 96, f"rows={len(v1_tuning)}")
        v1_oof = read_csv(config.path("nested_cv.regression_task.inner_selected_oof", must_exist=True, expect="file"), "V1 selected OOF")
        selected, selection_checks = build_selected(v1_tuning, v1_oof, list(model_specs))
        for model, ok, detail in selection_checks:
            add(results, f"{model} V1 selected pair/threshold", ok, detail)

        modeling = config.section("modeling")
        options = NestedCVOptions(
            base_seed=int(config.raw["experiment"]["random_seed"]),
            class_weight=str(engine["class_weight"]), max_iter=int(engine["max_iter"]),
            tolerance=float(engine["tolerance"]), zero_sd_tolerance=float(engine["zero_sd_tolerance"]),
            epsilon=float(public["epsilon"]), coefficient_nonzero_tolerance=float(modeling["coefficient_nonzero_tolerance"]),
            positive_label=str(modeling["positive_label"]), negative_label=str(modeling["negative_label"]),
        )
        row_lookup = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        labels = base["cohort"].astype(str).str.upper().to_numpy()
        aa_clone_numbers = pd.to_numeric(base["aa_clone_number"], errors="raise").to_numpy(float)
        scheme = ThresholdScheme.from_mapping(public)

        outer_result = fit_outer_models(
            base, split, row_lookup, presence, frequency, labels, aa_clone_numbers,
            model_specs, selected, scheme, options,
        )
        v1_train_public = read_csv(config.path("nested_cv.regression_task.outer_train_public", must_exist=True, expect="file"), "V1 outer train public")
        v1_valid_public = read_csv(config.path("nested_cv.regression_task.outer_validation_public", must_exist=True, expect="file"), "V1 outer validation public")
        ok, detail = compare_public(outer_result.outer_train_public, v1_train_public, args.rtol, args.atol)
        add(results, "Outer training LOO public equality", ok, detail)
        ok, detail = compare_public(outer_result.outer_validation_public, v1_valid_public, args.rtol, args.atol)
        add(results, "Outer validation public equality", ok, detail)

        v1_predictions = read_csv(config.path("nested_cv.regression_task.outer_predictions", must_exist=True, expect="file"), "V1 outer predictions")
        v1_metrics = read_csv(config.path("nested_cv.regression_task.outer_metrics", must_exist=True, expect="file"), "V1 outer metrics")
        v1_coefficients = read_csv(config.path("nested_cv.regression_task.coefficients", must_exist=True, expect="file"), "V1 coefficients")
        for model in model_specs:
            compare_outer_model(results, model, outer_result.predictions, outer_result.metrics, outer_result.coefficients, v1_predictions, v1_metrics, v1_coefficients, args.rtol, args.atol)

        v1_preprocessing = read_csv(config.path("nested_cv.regression_task.preprocessing_summary", must_exist=True, expect="file"), "V1 preprocessing")
        ok, detail = compare_preprocessing(outer_result.preprocessing_audit, v1_preprocessing, "outer_final", args.rtol, args.atol)
        add(results, "Outer preprocessing audit equality", ok, detail)
        v1_ref = read_csv(config.path("nested_cv.regression_task.public_reference_summary", must_exist=True, expect="file"), "V1 public reference audit")
        v1_outer_ref = v1_ref.loc[v1_ref["context"].astype(str).str.startswith("outer_")]
        add(results, "Outer public-reference audit row count", len(outer_result.public_reference_audit) == len(v1_outer_ref), f"V2={len(outer_result.public_reference_audit)}, V1={len(v1_outer_ref)}")
        v1_loo = read_csv(config.path("nested_cv.regression_task.public_loo_assignments", must_exist=True, expect="file"), "V1 LOO assignments")
        v1_outer_loo = v1_loo.loc[v1_loo["context"].astype(str) == "outer_training_leave_one_out"]
        add(results, "Outer LOO assignment count", len(outer_result.public_loo_assignments) == len(v1_outer_loo) == split.n_outer_train, f"V2={len(outer_result.public_loo_assignments)}, V1={len(v1_outer_loo)}")

        if args.full:
            full = run_nested_outer_task(
                base, outer_assignments, inner_assignments, row_lookup,
                presence, frequency, labels, aa_clone_numbers, model_specs,
                candidates, scheme, options,
                outer_repeat=outer_repeat, outer_fold=outer_fold,
                outer_folds=int(config.raw["cross_validation"]["outer_folds"]),
                inner_folds=int(config.raw["cross_validation"]["inner_folds"]),
            )
            tuning_numeric = ["C_inverse_lambda", "pooled_inner_roc_auc", "pooled_inner_pr_auc", "mean_fold_roc_auc", "sd_fold_roc_auc"]
            for model in model_specs:
                left = full.inner_tuning.loc[full.inner_tuning["model"] == model].sort_values(["l1_ratio_alpha", "lambda"]).reset_index(drop=True)
                right = v1_tuning.loc[v1_tuning["model"].astype(str) == model].sort_values(["l1_ratio_alpha", "lambda"]).reset_index(drop=True)
                keys_ok = np.array_equal(left[["l1_ratio_alpha", "lambda"]].to_numpy(float), right[["l1_ratio_alpha", "lambda"]].to_numpy(float))
                numeric_ok, detail = allclose_detail(left[tuning_numeric], right[tuning_numeric], args.rtol, args.atol)
                audit_ok = np.array_equal(left["max_iterations_used"].to_numpy(int), right["max_iterations_used"].to_numpy(int)) and np.array_equal(left["all_fits_converged"].map(as_bool), right["all_fits_converged"].map(as_bool))
                add(results, f"{model} full inner tuning equality", keys_ok and numeric_ok and audit_ok, detail + ", audit compared")

                left_oof = full.inner_selected_oof_predictions.loc[full.inner_selected_oof_predictions["model"] == model].copy()
                right_oof = v1_oof.loc[v1_oof["model"].astype(str) == model].copy()
                keys = ["inner_fold", "sample_id"]
                left_oof["sample_id"] = left_oof["sample_id"].astype(str)
                right_oof["sample_id"] = right_oof["sample_id"].astype(str)
                left_oof = left_oof.sort_values(keys).reset_index(drop=True)
                right_oof = right_oof.sort_values(keys).reset_index(drop=True)
                key_ok = left_oof[keys].to_dict("records") == right_oof[keys].to_dict("records")
                oof_ok, oof_detail = allclose_detail(
                    left_oof[["probability", "l1_ratio_alpha", "lambda", "selected_threshold"]],
                    right_oof[["probability", "l1_ratio_alpha", "lambda", "selected_threshold"]],
                    args.rtol, args.atol,
                )
                oof_ok &= np.array_equal(left_oof["true_label"].to_numpy(int), right_oof["true_label"].to_numpy(int))
                add(results, f"{model} selected inner OOF equality", key_ok and oof_ok, f"rows={len(left_oof)}, {oof_detail}")

            # V1 computes inner-fold preprocessing rows in memory but writes only
            # the outer-final preprocessing table to 05_preprocessing_summary.csv.
            # Therefore there are no frozen V1 inner rows available for an
            # equality comparison. Validate the V2 inner audit structurally and
            # explicitly confirm the V1 export boundary instead.
            v1_inner = v1_preprocessing.loc[
                v1_preprocessing["stage"].astype(str) == "inner"
            ].copy()
            v2_inner = full.preprocessing_audit.loc[
                full.preprocessing_audit["stage"].astype(str) == "inner"
            ].copy()
            required_inner_columns = {
                "stage", "model", "inner_fold", "feature_name", "source_type",
                "training_mean", "training_sd",
                "kept_after_zero_variance_filter",
            }
            columns_ok = required_inner_columns.issubset(v2_inner.columns)
            expected_models = list(model_specs)
            expected_folds = list(split.inner_folds)
            expected_per_model = (
                outer_result.preprocessing_audit
                .loc[outer_result.preprocessing_audit["stage"].astype(str) == "outer_final"]
                .groupby("model", sort=False)
                .size()
                .reindex(expected_models)
                .astype(int)
                .to_dict()
            )
            expected_rows = int(sum(expected_per_model.values()) * len(expected_folds))

            if columns_ok:
                v2_inner["model"] = v2_inner["model"].astype(str)
                v2_inner["feature_name"] = v2_inner["feature_name"].astype(str)
                v2_inner["inner_fold"] = pd.to_numeric(
                    v2_inner["inner_fold"], errors="raise"
                ).astype(int)
                observed_models = sorted(v2_inner["model"].unique().tolist())
                observed_folds = sorted(v2_inner["inner_fold"].unique().tolist())
                duplicate_count = int(
                    v2_inner.duplicated(
                        ["model", "inner_fold", "feature_name"]
                    ).sum()
                )
                counts = (
                    v2_inner.groupby(["model", "inner_fold"], sort=False)
                    .size()
                    .to_dict()
                )
                counts_ok = all(
                    int(counts.get((model, fold), -1))
                    == int(expected_per_model[model])
                    for model in expected_models
                    for fold in expected_folds
                )
                numeric = v2_inner[["training_mean", "training_sd"]].to_numpy(float)
                finite_ok = bool(np.isfinite(numeric).all())
                source_ok = set(v2_inner["source_type"].astype(str)).issubset(
                    {"numeric", "categorical_dummy"}
                )
                structure_ok = (
                    len(v2_inner) == expected_rows
                    and observed_models == sorted(expected_models)
                    and observed_folds == expected_folds
                    and duplicate_count == 0
                    and counts_ok
                    and finite_ok
                    and source_ok
                )
            else:
                observed_models = []
                observed_folds = []
                duplicate_count = -1
                counts_ok = False
                finite_ok = False
                source_ok = False
                structure_ok = False

            inner_ok = len(v1_inner) == 0 and columns_ok and structure_ok
            inner_detail = (
                "V1 exported inner rows=0 (expected); "
                f"V2 rows={len(v2_inner)}, expected={expected_rows}, "
                f"models={observed_models}, folds={observed_folds}, "
                f"duplicate_keys={duplicate_count}, per_fold_counts={counts_ok}, "
                f"finite={finite_ok}, source_types={source_ok}"
            )
            add(
                results,
                "Full inner preprocessing audit completeness",
                inner_ok,
                inner_detail,
            )
            context_counts_v2 = full.public_reference_audit["context"].astype(str).value_counts().sort_index().to_dict()
            context_counts_v1 = v1_ref["context"].astype(str).value_counts().sort_index().to_dict()
            add(results, "Full public-reference context counts", context_counts_v2 == context_counts_v1, f"V2={context_counts_v2}, V1={context_counts_v1}")
            loo_counts_v2 = full.public_loo_assignments["context"].astype(str).value_counts().sort_index().to_dict()
            loo_counts_v1 = v1_loo["context"].astype(str).value_counts().sort_index().to_dict()
            add(results, "Full LOO-assignment context counts", loo_counts_v2 == loo_counts_v1, f"V2={loo_counts_v2}, V1={loo_counts_v1}")

        passed = sum(result.status == "PASS" for result in results)
        failed = len(results) - passed
        mode = "full" if args.full else "quick"
        print(f"Mode={mode} Outer task={outer_repeat}/{outer_fold} Checks={len(results)} PASS={passed} FAIL={failed}")
        for result in results:
            print(f"[{result.status}] {result.name}: {result.detail}")
        return 0 if failed == 0 else 1
    except Exception as exc:
        print(f"Nested-CV regression check: FAIL\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
