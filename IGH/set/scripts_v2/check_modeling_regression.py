#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression-check the V2 model, threshold, and metric modules against V1.

The check reconstructs the frozen outer task ``repeat_01_fold_01`` without
changing any V1 output. For each M0--M3 model it:

1. reconstructs the V1 outer training/validation design matrices;
2. reads the hyperparameters selected by the existing V1 inner CV;
3. recomputes the selected Youden threshold from V1 inner OOF predictions;
4. refits the same scikit-learn Elastic Net model with the same derived seed;
5. compares probabilities, classifications, metrics, fit audit, and every
   coefficient with the existing V1 files;
6. recomputes the locked independent-test metrics from saved predictions.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ra_ild_igh.config import load_experiment_config  # noqa: E402
from ra_ild_igh.metrics import classification_metrics  # noqa: E402
from ra_ild_igh.modeling import (  # noqa: E402
    coefficient_frame,
    count_nonzero_coefficients,
    derive_seed,
    fit_elastic_net,
)
from ra_ild_igh.preprocessing import prepare_design_matrices  # noqa: E402
from ra_ild_igh.thresholds import apply_threshold, choose_threshold_youden  # noqa: E402


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str


def add(results: List[CheckResult], name: str, ok: bool, detail: str) -> None:
    results.append(CheckResult(name, "PASS" if ok else "FAIL", detail))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare IGH V2 modeling utilities with frozen V1 outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", default=None)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--skip-refit", action="store_true")
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


def require_unique_ids(frame: pd.DataFrame, id_col: str, label: str) -> None:
    if id_col not in frame.columns:
        raise ValueError(f"{label} is missing {id_col}")
    frame[id_col] = frame[id_col].astype(str)
    if frame[id_col].duplicated().any():
        examples = frame.loc[
            frame[id_col].duplicated(keep=False), id_col
        ].unique().tolist()[:10]
        raise ValueError(f"{label} has duplicated {id_col}: {examples}")


def load_static_features(path: Path) -> List[str]:
    frame = read_csv(path, "V1 static feature list")
    if "feature_name" not in frame.columns:
        raise ValueError("V1 static feature list is missing feature_name")
    features = frame["feature_name"].astype(str).tolist()
    if not features or len(features) != len(set(features)):
        raise ValueError("V1 static feature list is empty or duplicated")
    return features


def resolve_model_columns(
    specification: Mapping[str, Any],
    static_features: Sequence[str],
) -> Tuple[List[str], List[str]]:
    numeric = [str(value) for value in specification.get("numeric", [])]
    if specification.get("static_feature_groups"):
        numeric.extend(static_features)
    numeric.extend(str(value) for value in specification.get("dynamic_public", []))
    categorical = [str(value) for value in specification.get("categorical", [])]
    if len(numeric) != len(set(numeric)):
        raise ValueError(f"Resolved numeric columns contain duplicates: {numeric}")
    if len(categorical) != len(set(categorical)):
        raise ValueError(
            f"Resolved categorical columns contain duplicates: {categorical}"
        )
    return numeric, categorical


def merge_public(
    base: pd.DataFrame,
    public: pd.DataFrame,
    sample_ids: Sequence[str],
    label: str,
) -> pd.DataFrame:
    require_unique_ids(public, "sample_id", label)
    requested = list(map(str, sample_ids))
    observed = set(public["sample_id"])
    missing = [sample_id for sample_id in requested if sample_id not in observed]
    extra = sorted(observed - set(requested))
    if missing or extra:
        raise ValueError(
            f"{label} sample mismatch; missing={missing[:10]}, extra={extra[:10]}"
        )
    aligned = public.set_index("sample_id").loc[requested]
    output = base.loc[requested].copy()
    collisions = sorted(set(output.columns) & set(aligned.columns))
    if collisions:
        raise ValueError(f"{label} collides with base columns: {collisions}")
    for column in aligned.columns:
        output[column] = pd.to_numeric(aligned[column], errors="raise").to_numpy(float)
    return output


def compare_numeric_vectors(
    observed: np.ndarray,
    expected: np.ndarray,
    *,
    rtol: float,
    atol: float,
    labels: Optional[Sequence[str]] = None,
) -> Tuple[bool, str]:
    left = np.asarray(observed, dtype=float)
    right = np.asarray(expected, dtype=float)
    if left.shape != right.shape:
        return False, f"shape V2={left.shape}, V1={right.shape}"
    differences = np.abs(left - right)
    max_index = int(np.argmax(differences)) if differences.size else 0
    max_difference = float(differences.flat[max_index]) if differences.size else 0.0
    location = ""
    if labels is not None and differences.size:
        location = f", location={labels[max_index]}"
    return (
        bool(np.allclose(left, right, rtol=rtol, atol=atol)),
        f"max_abs_diff={max_difference:.12g}{location}, rtol={rtol:g}, atol={atol:g}",
    )


def compare_metric_mapping(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    names: Sequence[str],
    rtol: float,
    atol: float,
) -> Tuple[bool, str]:
    missing = [name for name in names if name not in expected]
    if missing:
        return False, f"V1 metrics missing columns: {missing}"
    for name in names:
        if name in {"TN", "FP", "FN", "TP", "n_test", "n_RA", "n_ILD"}:
            if int(observed[name]) != int(expected[name]):
                return False, f"{name}: V2={observed[name]}, V1={expected[name]}"
        else:
            if not math.isclose(
                float(observed[name]),
                float(expected[name]),
                rel_tol=rtol,
                abs_tol=atol,
            ):
                return False, f"{name}: V2={observed[name]}, V1={expected[name]}"
    return True, f"metrics={list(names)}"


def metric_row_mapping(frame: pd.DataFrame) -> Dict[str, Any]:
    if frame.empty:
        return {}
    if {"metric", "value"}.issubset(frame.columns):
        return {
            str(metric): value
            for metric, value in zip(frame["metric"], frame["value"])
        }
    if len(frame) == 1:
        return frame.iloc[0].to_dict()
    return {}


def first_column(frame: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"{label} lacks all candidate columns: {list(candidates)}")


def main() -> int:
    args = parse_args()
    results: List[CheckResult] = []

    try:
        config = load_experiment_config(
            Path(args.config),
            repository_root=(
                Path(args.repository_root) if args.repository_root else None
            ),
        )
        modeling = config.section("modeling")
        task = modeling["regression_task"]
        outer_repeat = int(task["outer_repeat"])
        outer_fold = int(task["outer_fold"])
        positive_label = str(modeling["positive_label"]).upper()
        negative_label = str(modeling["negative_label"]).upper()
        nonzero_tolerance = float(modeling["coefficient_nonzero_tolerance"])

        base = read_csv(
            config.path("data.train.base_matrix", must_exist=True, expect="file"),
            "train base matrix",
        )
        require_unique_ids(base, "sample_id", "train base matrix")
        base["cohort"] = base["cohort"].astype(str).str.upper()
        base = base.set_index("sample_id", drop=False)

        outer = read_csv(
            config.path(
                "cross_validation.outer_assignments", must_exist=True, expect="file"
            ),
            "outer assignments",
        )
        required_outer = {"outer_repeat", "outer_fold", "sample_id"}
        missing_outer = sorted(required_outer - set(outer.columns))
        if missing_outer:
            raise ValueError(f"Outer assignments missing columns: {missing_outer}")
        selected_outer = outer.loc[
            pd.to_numeric(outer["outer_repeat"], errors="raise").astype(int)
            == outer_repeat
        ].copy()
        selected_outer["sample_id"] = selected_outer["sample_id"].astype(str)
        train_set = set(
            selected_outer.loc[
                pd.to_numeric(selected_outer["outer_fold"], errors="raise").astype(int)
                != outer_fold,
                "sample_id",
            ]
        )
        valid_set = set(
            selected_outer.loc[
                pd.to_numeric(selected_outer["outer_fold"], errors="raise").astype(int)
                == outer_fold,
                "sample_id",
            ]
        )
        train_ids = [sample_id for sample_id in base.index if sample_id in train_set]
        valid_ids = [sample_id for sample_id in base.index if sample_id in valid_set]
        if train_set & valid_set or train_set | valid_set != set(base.index):
            raise ValueError("Outer train/validation partition is invalid")

        static_features = load_static_features(
            config.path(
                "preprocessing.regression_task.static_feature_list",
                must_exist=True,
                expect="file",
            )
        )
        outer_train_public = read_csv(
            config.path(
                "preprocessing.regression_task.outer_train_public",
                must_exist=True,
                expect="file",
            ),
            "V1 outer-train public features",
        )
        outer_valid_public = read_csv(
            config.path(
                "preprocessing.regression_task.outer_validation_public",
                must_exist=True,
                expect="file",
            ),
            "V1 outer-validation public features",
        )
        train_frame = merge_public(
            base,
            outer_train_public,
            train_ids,
            "V1 outer-train public features",
        )
        valid_frame = merge_public(
            base,
            outer_valid_public,
            valid_ids,
            "V1 outer-validation public features",
        )
        y_train = (train_frame["cohort"] == positive_label).astype(int).to_numpy()
        y_valid = (valid_frame["cohort"] == positive_label).astype(int).to_numpy()
        observed_labels = set(base["cohort"])
        if observed_labels != {positive_label, negative_label}:
            raise ValueError(
                f"Unexpected cohort labels: {sorted(observed_labels)}, "
                f"expected={[negative_label, positive_label]}"
            )

        inner_oof = read_csv(
            config.path(
                "modeling.regression_task.inner_selected_oof",
                must_exist=True,
                expect="file",
            ),
            "V1 selected inner OOF predictions",
        )
        outer_predictions = read_csv(
            config.path(
                "modeling.regression_task.outer_predictions",
                must_exist=True,
                expect="file",
            ),
            "V1 outer predictions",
        )
        outer_metrics = read_csv(
            config.path(
                "modeling.regression_task.outer_metrics",
                must_exist=True,
                expect="file",
            ),
            "V1 outer metrics",
        )
        v1_coefficients = read_csv(
            config.path(
                "modeling.regression_task.coefficients",
                must_exist=True,
                expect="file",
            ),
            "V1 coefficient table",
        )

        required_model_columns = {"model"}
        for label, frame in (
            ("inner OOF", inner_oof),
            ("outer predictions", outer_predictions),
            ("outer metrics", outer_metrics),
            ("coefficients", v1_coefficients),
        ):
            missing = sorted(required_model_columns - set(frame.columns))
            if missing:
                raise ValueError(f"V1 {label} missing columns: {missing}")

        add(
            results,
            "Outer task sample counts",
            len(train_ids) + len(valid_ids) == len(base),
            f"train={len(train_ids)}, validation={len(valid_ids)}, total={len(base)}",
        )
        add(
            results,
            "Static feature count",
            len(static_features) > 0,
            f"n_static={len(static_features)}",
        )

        engine = config.section("model_engine")
        model_specs = config.section("models")
        probability_column = first_column(
            outer_predictions,
            ("probability_ILD", "probability"),
            "V1 outer predictions",
        )

        for model_index, (model_name, specification) in enumerate(model_specs.items()):
            metric_rows = outer_metrics.loc[
                outer_metrics["model"].astype(str) == model_name
            ]
            if len(metric_rows) != 1:
                raise ValueError(
                    f"Expected one V1 outer metric row for {model_name}, observed={len(metric_rows)}"
                )
            expected_metric = metric_rows.iloc[0].to_dict()
            alpha = float(expected_metric["selected_l1_ratio_alpha"])
            lambda_value = float(expected_metric["selected_lambda"])
            expected_threshold = float(expected_metric["threshold"])

            oof = inner_oof.loc[inner_oof["model"].astype(str) == model_name].copy()
            if oof.empty:
                raise ValueError(f"No V1 selected inner OOF rows for {model_name}")
            recomputed_threshold = choose_threshold_youden(
                pd.to_numeric(oof["true_label"], errors="raise").to_numpy(int),
                pd.to_numeric(oof["probability"], errors="raise").to_numpy(float),
            )
            stored_thresholds = pd.to_numeric(
                oof["selected_threshold"], errors="raise"
            ).to_numpy(float)
            threshold_ok = (
                np.allclose(
                    stored_thresholds,
                    expected_threshold,
                    rtol=args.rtol,
                    atol=args.atol,
                )
                and math.isclose(
                    recomputed_threshold,
                    expected_threshold,
                    rel_tol=args.rtol,
                    abs_tol=args.atol,
                )
            )
            add(
                results,
                f"{model_name} selected Youden threshold",
                threshold_ok,
                f"recomputed={recomputed_threshold:.16g}, V1={expected_threshold:.16g}",
            )

            numeric_columns, categorical_columns = resolve_model_columns(
                specification,
                static_features,
            )
            design = prepare_design_matrices(
                train_frame,
                valid_frame,
                numeric_columns,
                categorical_columns,
                zero_sd_tolerance=float(engine["zero_sd_tolerance"]),
            )

            expected_predictions = outer_predictions.loc[
                outer_predictions["model"].astype(str) == model_name
            ].copy()
            require_unique_ids(
                expected_predictions,
                "sample_id",
                f"V1 outer predictions for {model_name}",
            )
            expected_predictions = expected_predictions.set_index("sample_id").loc[
                valid_ids
            ]
            expected_probability = pd.to_numeric(
                expected_predictions[probability_column], errors="raise"
            ).to_numpy(float)

            # Metric logic is validated independently from model refitting by using
            # the frozen V1 probabilities directly.
            metric_from_v1_probability = classification_metrics(
                y_valid,
                expected_probability,
                expected_threshold,
            )
            metric_names = (
                "roc_auc",
                "pr_auc",
                "threshold",
                "accuracy",
                "sensitivity_recall",
                "specificity",
                "precision",
                "f1",
                "TN",
                "FP",
                "FN",
                "TP",
            )
            metric_ok, metric_detail = compare_metric_mapping(
                metric_from_v1_probability,
                expected_metric,
                metric_names,
                args.rtol,
                args.atol,
            )
            add(
                results,
                f"{model_name} metric calculation",
                metric_ok,
                metric_detail,
            )

            expected_predicted = pd.to_numeric(
                expected_predictions["predicted_label"], errors="raise"
            ).to_numpy(int)
            observed_predicted = apply_threshold(
                expected_probability,
                expected_threshold,
            )
            add(
                results,
                f"{model_name} threshold application",
                np.array_equal(observed_predicted, expected_predicted),
                f"n_discordant={int(np.sum(observed_predicted != expected_predicted))}",
            )

            if args.skip_refit:
                continue

            seed = derive_seed(
                int(config.raw["experiment"]["random_seed"]),
                400,
                model_index,
            )
            fit = fit_elastic_net(
                design.X_train,
                y_train,
                l1_ratio=alpha,
                lambda_value=lambda_value,
                class_weight=engine["class_weight"],
                max_iter=int(engine["max_iter"]),
                tolerance=float(engine["tolerance"]),
                random_state=seed,
            )
            observed_probability = fit.predict_probability(design.X_valid)
            probability_ok, probability_detail = compare_numeric_vectors(
                observed_probability,
                expected_probability,
                rtol=args.rtol,
                atol=args.atol,
                labels=valid_ids,
            )
            add(
                results,
                f"{model_name} refit probability equality",
                probability_ok,
                probability_detail,
            )

            audit_checks = {
                "fit_converged": bool(fit.converged),
                "iterations_used": int(fit.n_iter),
                "n_final_predictors": int(len(design.feature_names)),
                "n_nonzero_coefficients": count_nonzero_coefficients(
                    fit.model,
                    tolerance=nonzero_tolerance,
                ),
            }
            audit_ok = True
            audit_details = []
            for name, observed in audit_checks.items():
                expected = expected_metric[name]
                if name == "fit_converged":
                    same = bool(observed) == as_bool(expected)
                else:
                    same = int(observed) == int(expected)
                audit_ok &= same
                audit_details.append(f"{name}:V2={observed},V1={expected}")
            add(
                results,
                f"{model_name} fit audit equality",
                audit_ok,
                "; ".join(audit_details),
            )

            observed_coefficients = coefficient_frame(
                fit.model,
                design.feature_names,
                nonzero_tolerance=nonzero_tolerance,
                model_name=model_name,
            )
            expected_coefficients = v1_coefficients.loc[
                v1_coefficients["model"].astype(str) == model_name
            ].copy()
            if expected_coefficients.empty:
                raise ValueError(f"No V1 coefficients for {model_name}")
            observed_names = observed_coefficients["feature_name"].astype(str).tolist()
            expected_names = expected_coefficients["feature_name"].astype(str).tolist()
            names_ok = observed_names == expected_names
            if names_ok:
                coefficients_ok, coefficient_detail = compare_numeric_vectors(
                    observed_coefficients["coefficient"].to_numpy(float),
                    pd.to_numeric(
                        expected_coefficients["coefficient"], errors="raise"
                    ).to_numpy(float),
                    rtol=args.rtol,
                    atol=args.atol,
                    labels=observed_names,
                )
                observed_nonzero = observed_coefficients["nonzero"].map(as_bool).to_numpy()
                expected_nonzero = expected_coefficients["nonzero"].map(as_bool).to_numpy()
                coefficients_ok &= np.array_equal(observed_nonzero, expected_nonzero)
                coefficient_detail += (
                    f", nonzero_discordant={int(np.sum(observed_nonzero != expected_nonzero))}"
                )
            else:
                coefficients_ok = False
                mismatch = next(
                    (
                        index,
                        observed,
                        expected,
                    )
                    for index, (observed, expected) in enumerate(
                        zip(observed_names, expected_names)
                    )
                    if observed != expected
                ) if len(observed_names) == len(expected_names) else None
                coefficient_detail = (
                    f"feature order differs; V2={len(observed_names)}, V1={len(expected_names)}, "
                    f"first_mismatch={mismatch}"
                )
            add(
                results,
                f"{model_name} coefficient equality",
                names_ok and coefficients_ok,
                coefficient_detail,
            )

        independent_predictions = read_csv(
            config.path(
                "modeling.regression_task.independent_predictions",
                must_exist=True,
                expect="file",
            ),
            "V1 independent predictions",
        )
        independent_metrics_frame = read_csv(
            config.path(
                "modeling.regression_task.independent_metrics",
                must_exist=True,
                expect="file",
            ),
            "V1 independent metrics",
        )
        true_column = first_column(
            independent_predictions,
            ("true_label", "label"),
            "V1 independent predictions",
        )
        independent_probability_column = first_column(
            independent_predictions,
            ("probability_ILD", "probability"),
            "V1 independent predictions",
        )
        threshold_column = first_column(
            independent_predictions,
            ("locked_threshold", "threshold"),
            "V1 independent predictions",
        )
        y_test = pd.to_numeric(
            independent_predictions[true_column], errors="raise"
        ).to_numpy(int)
        p_test = pd.to_numeric(
            independent_predictions[independent_probability_column], errors="raise"
        ).to_numpy(float)
        test_thresholds = pd.to_numeric(
            independent_predictions[threshold_column], errors="raise"
        ).to_numpy(float)
        if not np.allclose(test_thresholds, test_thresholds[0], rtol=0, atol=0):
            raise ValueError("Independent predictions contain multiple locked thresholds")
        test_metrics = classification_metrics(
            y_test,
            p_test,
            float(test_thresholds[0]),
            include_brier=True,
            include_sample_summary=True,
        )
        stored_test_metrics = metric_row_mapping(independent_metrics_frame)
        if "locked_threshold" in stored_test_metrics and "threshold" not in stored_test_metrics:
            stored_test_metrics["threshold"] = stored_test_metrics["locked_threshold"]
        test_metric_names = (
            "n_test",
            "n_RA",
            "n_ILD",
            "positive_prevalence",
            "roc_auc",
            "pr_auc",
            "accuracy",
            "sensitivity_recall",
            "specificity",
            "precision",
            "f1",
            "brier_score",
            "TN",
            "FP",
            "FN",
            "TP",
        )
        independent_ok, independent_detail = compare_metric_mapping(
            test_metrics,
            stored_test_metrics,
            test_metric_names,
            args.rtol,
            args.atol,
        )
        add(
            results,
            "Independent-test metric calculation",
            independent_ok,
            independent_detail,
        )

        predicted_column = first_column(
            independent_predictions,
            ("predicted_label",),
            "V1 independent predictions",
        )
        expected_test_predicted = pd.to_numeric(
            independent_predictions[predicted_column], errors="raise"
        ).to_numpy(int)
        observed_test_predicted = apply_threshold(p_test, float(test_thresholds[0]))
        add(
            results,
            "Independent-test threshold application",
            np.array_equal(observed_test_predicted, expected_test_predicted),
            f"n_discordant={int(np.sum(observed_test_predicted != expected_test_predicted))}",
        )

    except Exception as exc:
        add(results, "Modeling regression execution", False, str(exc))

    failed = [result for result in results if result.status == "FAIL"]
    print(
        f"Outer task={locals().get('outer_repeat', '?')}/{locals().get('outer_fold', '?')} "
        f"Checks={len(results)} PASS={len(results)-len(failed)} FAIL={len(failed)}"
    )
    for result in results:
        print(f"[{result.status}] {result.name}: {result.detail}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
