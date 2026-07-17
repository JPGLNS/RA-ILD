#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-time independent validation of the locked final IGH M2 model.

Locked model:
    M2_static_igh_public
    alpha = 0.9
    lambda = 10
    threshold = 0.4568009623694209

Important boundaries:
- The model is loaded from the step-07 IGH joblib bundle and is never refitted.
- Training means/SDs and categorical encodings are reused exactly.
- The independent test threshold is never optimized or changed.
- Test labels are used only after probabilities are generated, for evaluation.
- The script refuses to replace existing validation outputs unless the user
  explicitly supplies --overwrite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


SCRIPT_VERSION = "1.1.0-IGH"
EXPECTED_RECEPTOR = "IGH"
MODEL_NAME = "M2_static_igh_public"
ROOT = Path("/data/users/chenhaisheng/RA-ILD/IGH")

DEFAULT_TEST_MATRIX = (
    ROOT / "set/test/result/04_final_feature_matrix/04_test_final_feature_matrix.csv"
)
DEFAULT_TEST_REFERENCE_FEATURES = (
    ROOT / "set/test/result/03_public_features/03_test_reference_public_features.csv"
)
DEFAULT_TEST_REFERENCE_DEFINITION = (
    ROOT / "set/test/result/03_public_features/03_reference_definition_used.json"
)
DEFAULT_MODEL_BUNDLE = (
    ROOT / "set/train/result/07_final_M2_model/07_final_M2_model.joblib"
)
DEFAULT_MODEL_CONFIGURATION = (
    ROOT / "set/train/result/07_final_M2_model/07_final_M2_configuration.json"
)
DEFAULT_REFERENCE_MASKS = (
    ROOT / "set/train/result/07_final_M2_model/07_full_training_public_reference_masks.npz"
)
DEFAULT_OUTPUT_DIR = ROOT / "set/test/result/08_final_M2_validation"

LOCKED_ALPHA = 0.9
LOCKED_LAMBDA = 10.0
LOCKED_THRESHOLD = 0.4568009623694209


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the locked final IGH M2 model to the 49-sample independent test set once.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test-matrix", default=str(DEFAULT_TEST_MATRIX))
    parser.add_argument(
        "--test-reference-features",
        default=str(DEFAULT_TEST_REFERENCE_FEATURES),
    )
    parser.add_argument(
        "--test-reference-definition",
        default=str(DEFAULT_TEST_REFERENCE_DEFINITION),
    )
    parser.add_argument("--model-bundle", default=str(DEFAULT_MODEL_BUNDLE))
    parser.add_argument(
        "--model-configuration",
        default=str(DEFAULT_MODEL_CONFIGURATION),
    )
    parser.add_argument("--reference-masks", default=str(DEFAULT_REFERENCE_MASKS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    parser.add_argument("--sample-id-col", default="sample_id")
    parser.add_argument("--label-col", default="cohort")
    parser.add_argument("--positive-label", default="ILD")
    parser.add_argument("--negative-label", default="RA")
    parser.add_argument("--expected-test-samples", type=int, default=49)
    parser.add_argument("--expected-training-samples", type=int, default=120)
    parser.add_argument("--expected-training-ra", type=int, default=72)
    parser.add_argument("--expected-training-ild", type=int, default=48)
    parser.add_argument("--expected-catalog-size", type=int, default=41052)

    parser.add_argument("--locked-alpha", type=float, default=LOCKED_ALPHA)
    parser.add_argument("--locked-lambda", type=float, default=LOCKED_LAMBDA)
    parser.add_argument("--locked-threshold", type=float, default=LOCKED_THRESHOLD)
    parser.add_argument("--float-tolerance", type=float, default=1e-6)
    parser.add_argument("--public-value-tolerance", type=float, default=1e-10)

    parser.add_argument("--expected-global-size", type=int, default=6387)
    parser.add_argument("--expected-ra-specific-size", type=int, default=210)
    parser.add_argument("--expected-ild-specific-size", type=int, default=504)
    parser.add_argument("--expected-shared-size", type=int, default=263)

    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260713)
    parser.add_argument("--plot-dpi", type=int, default=400)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace prior validation outputs. Do not use for the real one-time "
            "independent validation unless a genuine technical failure occurred."
        ),
    )
    args = parser.parse_args()

    if args.expected_test_samples < 2:
        parser.error("--expected-test-samples must be >= 2.")
    if args.expected_training_samples < 2:
        parser.error("--expected-training-samples must be >= 2.")
    if args.expected_training_ra < 1 or args.expected_training_ild < 1:
        parser.error("Expected training class counts must both be >= 1.")
    if args.expected_training_ra + args.expected_training_ild != args.expected_training_samples:
        parser.error("Expected training RA + ILD counts must equal expected training samples.")
    if args.expected_catalog_size < 1:
        parser.error("--expected-catalog-size must be >= 1.")
    if not (0 < args.locked_alpha <= 1):
        parser.error("--locked-alpha must be in (0, 1].")
    if args.locked_lambda <= 0:
        parser.error("--locked-lambda must be > 0.")
    if not (0 < args.locked_threshold < 1):
        parser.error("--locked-threshold must be in (0, 1).")
    if args.bootstrap_reps < 100:
        parser.error("--bootstrap-reps must be >= 100.")
    return args


def sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def output_paths(output_dir: Path) -> Dict[str, Path]:
    return {
        "predictions": output_dir / "08_independent_test_predictions.csv",
        "metrics": output_dir / "08_independent_test_metrics.csv",
        "bootstrap_ci": output_dir / "08_independent_test_bootstrap_CI.csv",
        "confusion_matrix": output_dir / "08_independent_test_confusion_matrix.csv",
        "roc_curve": output_dir / "08_independent_test_ROC_curve.csv",
        "pr_curve": output_dir / "08_independent_test_PR_curve.csv",
        "figure_png": output_dir / "08_independent_test_evaluation.png",
        "figure_pdf": output_dir / "08_independent_test_evaluation.pdf",
        "configuration": output_dir / "08_independent_test_configuration.json",
        "summary": output_dir / "08_independent_test_summary.md",
        "complete_marker": output_dir / "08_VALIDATION_COMPLETE.json",
    }


def enforce_one_time_outputs(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Independent-test validation outputs already exist. The locked test "
            "set should not be repeatedly evaluated. Existing files:\n"
            + "\n".join(f"  - {path}" for path in existing)
            + "\nUse --overwrite only after documenting a genuine technical failure."
        )


def close_enough(observed: float, expected: float, tolerance: float) -> bool:
    return math.isclose(
        float(observed),
        float(expected),
        rel_tol=tolerance,
        abs_tol=tolerance,
    )


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_and_validate_locked_model(
    bundle_path: Path,
    config_path: Path,
    reference_masks_path: Path,
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, int]]:
    bundle = joblib.load(bundle_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    if not isinstance(bundle, dict):
        raise TypeError("Model bundle must be a dictionary.")
    if bundle.get("model_name") != MODEL_NAME:
        raise ValueError(
            f"Unexpected model bundle: {bundle.get('model_name')}; expected {MODEL_NAME}."
        )
    if config.get("model") != MODEL_NAME:
        raise ValueError(
            f"Unexpected model configuration: {config.get('model')}; expected {MODEL_NAME}."
        )
    if bundle.get("receptor") != EXPECTED_RECEPTOR:
        raise ValueError(
            f"Unexpected bundle receptor: {bundle.get('receptor')}; "
            f"expected {EXPECTED_RECEPTOR}."
        )
    if config.get("receptor") != EXPECTED_RECEPTOR:
        raise ValueError(
            f"Unexpected configuration receptor: {config.get('receptor')}; "
            f"expected {EXPECTED_RECEPTOR}."
        )
    training_ids = [str(x) for x in bundle.get("training_sample_ids", [])]
    if len(training_ids) != args.expected_training_samples:
        raise ValueError(
            f"Bundle contains {len(training_ids)} training IDs; "
            f"expected {args.expected_training_samples}."
        )
    if len(set(training_ids)) != len(training_ids):
        raise ValueError("Bundle training sample IDs are duplicated.")
    bundle_n = int(bundle.get("n_training_samples", len(training_ids)))
    if bundle_n != args.expected_training_samples:
        raise ValueError(
            f"Bundle n_training_samples={bundle_n}; "
            f"expected={args.expected_training_samples}."
        )
    config_n = int(config.get("n_training_samples", -1))
    if config_n != args.expected_training_samples:
        raise ValueError(
            f"Configuration n_training_samples={config_n}; "
            f"expected={args.expected_training_samples}."
        )
    for source_name, source in (("bundle", bundle), ("configuration", config)):
        if "n_RA" in source and int(source["n_RA"]) != args.expected_training_ra:
            raise ValueError(
                f"{source_name} n_RA={source['n_RA']}; "
                f"expected={args.expected_training_ra}."
            )
        if "n_ILD" in source and int(source["n_ILD"]) != args.expected_training_ild:
            raise ValueError(
                f"{source_name} n_ILD={source['n_ILD']}; "
                f"expected={args.expected_training_ild}."
            )
    if config.get("independent_test_read") is not False:
        raise ValueError(
            "Step-07 configuration does not confirm that the independent test set was unread."
        )
    if not as_bool(bundle.get("fit_converged", False)):
        raise ValueError("The saved final M2 fit is not marked as converged.")

    checks = [
        ("bundle alpha", bundle.get("selected_alpha"), args.locked_alpha),
        ("bundle lambda", bundle.get("selected_lambda"), args.locked_lambda),
        ("bundle threshold", bundle.get("locked_threshold"), args.locked_threshold),
    ]

    config_selected = config.get("selected_parameters", {})
    if config_selected:
        checks.extend([
            ("configuration alpha", config_selected.get("alpha"), args.locked_alpha),
            ("configuration lambda", config_selected.get("lambda"), args.locked_lambda),
        ])
    else:
        checks.extend([
            ("configuration alpha", config.get("selected_alpha"), args.locked_alpha),
            ("configuration lambda", config.get("selected_lambda"), args.locked_lambda),
        ])
    config_threshold = config.get("locked_threshold")
    if config_threshold is not None:
        checks.append(
            ("configuration threshold", config_threshold, args.locked_threshold)
        )

    for label, observed, expected in checks:
        if observed is None or not close_enough(observed, expected, args.float_tolerance):
            raise ValueError(
                f"Locked-value mismatch for {label}: observed={observed}, expected={expected}."
            )

    model = bundle.get("sklearn_model")
    if model is None or not hasattr(model, "predict_proba"):
        raise TypeError("Saved bundle does not contain a usable sklearn probability model.")
    classes = np.asarray(getattr(model, "classes_", []))
    if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
        raise ValueError(f"Unexpected sklearn class labels: {classes.tolist()}")

    masks_sha256 = sha256sum(reference_masks_path)
    bundle_masks_sha = bundle.get("public_reference_masks_sha256")
    if bundle_masks_sha and str(bundle_masks_sha) != masks_sha256:
        raise ValueError("Reference-mask SHA256 does not match the model bundle.")
    config_masks_sha = config.get("full_reference_masks_sha256")
    if config_masks_sha and str(config_masks_sha) != masks_sha256:
        raise ValueError("Reference-mask SHA256 does not match the step-07 configuration.")

    masks = np.load(reference_masks_path, allow_pickle=False)
    if "receptor" not in masks.files or str(np.asarray(masks["receptor"]).item()) != EXPECTED_RECEPTOR:
        raise ValueError("Reference-mask file is not marked as IGH.")
    metadata_expectations = {
        "n_reference": args.expected_training_samples,
        "n_RA_reference": args.expected_training_ra,
        "n_ILD_reference": args.expected_training_ild,
    }
    for key, expected in metadata_expectations.items():
        if key not in masks.files:
            raise ValueError(f"Reference-mask file is missing metadata '{key}'.")
        observed = int(np.asarray(masks[key]).item())
        if observed != expected:
            raise ValueError(
                f"Reference-mask metadata {key}={observed}; expected={expected}."
            )

    required_masks = {
        "global_mask": args.expected_global_size,
        "RA_specific_mask": args.expected_ra_specific_size,
        "ILD_specific_mask": args.expected_ild_specific_size,
        "shared_mask": args.expected_shared_size,
    }
    mask_counts: Dict[str, int] = {}
    mask_length = None
    for key, expected_size in required_masks.items():
        if key not in masks.files:
            raise ValueError(f"Reference-mask file is missing '{key}'.")
        mask = np.asarray(masks[key]).astype(bool)
        if mask_length is None:
            mask_length = len(mask)
        elif len(mask) != mask_length:
            raise ValueError("Reference masks have inconsistent lengths.")
        observed_size = int(mask.sum())
        mask_counts[key] = observed_size
        if expected_size > 0 and observed_size != expected_size:
            raise ValueError(
                f"Reference size mismatch for {key}: observed={observed_size}, "
                f"expected={expected_size}."
            )

    if mask_length != args.expected_catalog_size:
        raise ValueError(
            f"Reference-mask length={mask_length}; "
            f"expected catalog size={args.expected_catalog_size}."
        )
    if "public_catalog_sha256" in masks.files:
        masks_catalog_sha = str(np.asarray(masks["public_catalog_sha256"]).item())
        bundle_catalog_sha = str(bundle.get("public_catalog_sha256", ""))
        if bundle_catalog_sha and masks_catalog_sha != bundle_catalog_sha:
            raise ValueError("Public-catalog SHA256 differs between bundle and mask file.")
    if "reference_definition_sha256" in masks.files:
        masks_definition_sha = str(np.asarray(masks["reference_definition_sha256"]).item())
        bundle_definition_sha = str(bundle.get("reference_definition_sha256", ""))
        if bundle_definition_sha and masks_definition_sha != bundle_definition_sha:
            raise ValueError(
                "Reference-definition SHA256 differs between bundle and mask file."
            )

    # Specific sets and shared set should be mutually exclusive by construction.
    ra = np.asarray(masks["RA_specific_mask"]).astype(bool)
    ild = np.asarray(masks["ILD_specific_mask"]).astype(bool)
    shared = np.asarray(masks["shared_mask"]).astype(bool)
    if np.any(ra & ild) or np.any(ra & shared) or np.any(ild & shared):
        raise ValueError("Saved RA-specific, ILD-specific and shared masks overlap.")

    return bundle, config, mask_counts


def validate_test_reference_audit(
    definition_path: Path,
    expected_samples: int,
    expected_training_samples: int,
    reference_masks_path: Path,
) -> Dict[str, object]:
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    if definition.get("receptor") not in {None, EXPECTED_RECEPTOR}:
        raise ValueError(
            f"Test reference receptor={definition.get('receptor')}; "
            f"expected={EXPECTED_RECEPTOR}."
        )
    if definition.get("mode") != "test_transform":
        raise ValueError("Test reference-definition file is not in test_transform mode.")
    if definition.get("test_labels_used_to_build_reference") is not False:
        raise ValueError("Test labels may have been used to build public references.")
    observed_count = int(definition.get("test_sample_count", -1))
    if observed_count != expected_samples:
        raise ValueError(
            f"Reference-definition test sample count={observed_count}; "
            f"expected={expected_samples}."
        )
    expected_hash = definition.get("reference_file_sha256")
    verified_hash = definition.get("reference_file_sha256_verified")
    if expected_hash and verified_hash and expected_hash != verified_hash:
        raise ValueError("Training public-reference SHA256 was not verified in test mode.")
    actual_reference_sha = sha256sum(reference_masks_path)
    for key in ("training_reference_masks_sha256", "full_reference_masks_sha256"):
        value = definition.get(key)
        if value and str(value) != actual_reference_sha:
            raise ValueError(
                f"Test reference audit {key} does not match the locked reference masks."
            )
    if "training_sample_count" in definition:
        observed_training = int(definition["training_sample_count"])
        if observed_training != expected_training_samples:
            raise ValueError(
                f"Test reference audit training_sample_count={observed_training}; "
                f"expected={expected_training_samples}."
            )
    return definition


def validate_and_align_test_data(
    test_matrix_path: Path,
    test_reference_path: Path,
    bundle: Mapping[str, object],
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test = pd.read_csv(test_matrix_path)
    reference = pd.read_csv(test_reference_path)

    required_test = {args.sample_id_col, args.label_col}
    missing = required_test - set(test.columns)
    if missing:
        raise ValueError(f"Test matrix missing columns: {sorted(missing)}")
    if args.sample_id_col not in reference.columns:
        raise ValueError("Test reference-public feature file lacks sample_id.")

    test[args.sample_id_col] = test[args.sample_id_col].astype(str)
    reference[args.sample_id_col] = reference[args.sample_id_col].astype(str)

    if len(test) != args.expected_test_samples:
        raise ValueError(
            f"Test matrix has {len(test)} rows; expected {args.expected_test_samples}."
        )
    if test[args.sample_id_col].duplicated().any():
        raise ValueError("Test matrix contains duplicated sample IDs.")
    if reference[args.sample_id_col].duplicated().any():
        raise ValueError("Test public-feature file contains duplicated sample IDs.")
    if set(test[args.sample_id_col]) != set(reference[args.sample_id_col]):
        raise ValueError("Test matrix and test public-feature file have different sample sets.")

    train_ids = set(str(x) for x in bundle.get("training_sample_ids", []))
    overlap = train_ids & set(test[args.sample_id_col])
    if overlap:
        raise ValueError(f"Training/test sample overlap detected: {sorted(overlap)[:20]}")

    labels = test[args.label_col].astype(str).str.upper()
    allowed = {args.negative_label.upper(), args.positive_label.upper()}
    unexpected = sorted(set(labels) - allowed)
    if unexpected:
        raise ValueError(f"Unexpected test labels: {unexpected}")
    if labels.nunique() != 2:
        raise ValueError("Independent test set must contain both RA and ILD.")

    public_features = list(bundle.get("public_features", []))
    if not public_features:
        raise ValueError("Model bundle contains no public-feature list.")
    missing_public_test = [x for x in public_features if x not in test.columns]
    missing_public_ref = [x for x in public_features if x not in reference.columns]
    if missing_public_test or missing_public_ref:
        raise ValueError(
            f"Missing public features. test={missing_public_test}, "
            f"reference={missing_public_ref}"
        )

    test_indexed = test.set_index(args.sample_id_col, drop=False)
    ref_indexed = reference.set_index(args.sample_id_col).loc[test_indexed.index]
    for feature in public_features:
        left = pd.to_numeric(test_indexed[feature], errors="raise").to_numpy(float)
        right = pd.to_numeric(ref_indexed[feature], errors="raise").to_numpy(float)
        max_difference = float(np.max(np.abs(left - right)))
        if max_difference > args.public_value_tolerance:
            raise ValueError(
                f"Public feature '{feature}' differs between step-04 matrix and "
                f"step-03 test reference file; max abs difference={max_difference:.3g}."
            )

    return test_indexed.copy(), ref_indexed.copy()


def transform_with_locked_preprocessor(
    test: pd.DataFrame,
    bundle: Mapping[str, object],
) -> Tuple[np.ndarray, List[str]]:
    preprocessor = bundle.get("preprocessor")
    if not isinstance(preprocessor, dict):
        raise TypeError("Model bundle is missing preprocessor metadata.")

    numeric_columns = list(preprocessor.get("numeric_columns", []))
    categorical_columns = list(preprocessor.get("categorical_columns", []))
    categorical_levels = dict(preprocessor.get("categorical_levels", {}))
    categorical_reference = dict(preprocessor.get("categorical_reference", {}))
    final_feature_names = list(preprocessor.get("final_feature_names", []))
    preprocessing_rows = list(preprocessor.get("preprocessing_rows", []))

    if not final_feature_names or not preprocessing_rows:
        raise ValueError("Preprocessor metadata is incomplete.")

    missing_numeric = [x for x in numeric_columns if x not in test.columns]
    missing_categorical = [x for x in categorical_columns if x not in test.columns]
    if missing_numeric or missing_categorical:
        raise ValueError(
            f"Test predictors missing: numeric={missing_numeric}, "
            f"categorical={missing_categorical}"
        )

    raw_features: Dict[str, np.ndarray] = {}
    for column in numeric_columns:
        values = pd.to_numeric(test[column], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values in numeric predictor '{column}'.")
        raw_features[column] = values

    for column in categorical_columns:
        values = test[column].astype(str)
        levels = [str(x) for x in categorical_levels.get(column, [])]
        if not levels:
            raise ValueError(f"No stored training categories for '{column}'.")
        reference = str(categorical_reference.get(column, levels[0]))
        unseen = sorted(set(values) - set(levels))
        if unseen:
            raise ValueError(
                f"Unseen test categories for '{column}': {unseen}; "
                f"training levels={levels}."
            )
        for category in levels:
            if category == reference:
                continue
            feature_name = f"{column}__{category}_vs_{reference}"
            raw_features[feature_name] = (values == category).to_numpy(float)

    prep = pd.DataFrame(preprocessing_rows)
    required_prep = {
        "feature_name",
        "training_mean",
        "training_sd",
        "kept_after_zero_variance_filter",
    }
    missing_prep = required_prep - set(prep.columns)
    if missing_prep:
        raise ValueError(f"Preprocessing metadata missing: {sorted(missing_prep)}")
    if prep["feature_name"].duplicated().any():
        raise ValueError("Duplicated feature names in preprocessing metadata.")
    prep = prep.set_index("feature_name")

    transformed_parts: List[np.ndarray] = []
    for feature_name in final_feature_names:
        if feature_name not in raw_features:
            raise ValueError(
                f"Unable to reconstruct final model feature '{feature_name}'."
            )
        if feature_name not in prep.index:
            raise ValueError(
                f"No training preprocessing row for '{feature_name}'."
            )
        row = prep.loc[feature_name]
        if not as_bool(row["kept_after_zero_variance_filter"]):
            raise ValueError(
                f"Final model feature '{feature_name}' was not marked as kept."
            )
        mean = float(row["training_mean"])
        sd = float(row["training_sd"])
        if not np.isfinite(mean) or not np.isfinite(sd) or sd <= 0:
            raise ValueError(
                f"Invalid stored mean/SD for '{feature_name}': mean={mean}, sd={sd}."
            )
        transformed_parts.append(((raw_features[feature_name] - mean) / sd)[:, None])

    X = np.hstack(transformed_parts)
    if not np.isfinite(X).all():
        raise ValueError("Locked preprocessing produced non-finite values.")

    model = bundle["sklearn_model"]
    expected_features = int(getattr(model, "n_features_in_", X.shape[1]))
    if X.shape[1] != expected_features:
        raise ValueError(
            f"Transformed feature count={X.shape[1]}, model expects={expected_features}."
        )
    return X, final_feature_names


def calculate_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "n_test": int(len(y)),
        "n_RA": int(np.sum(y == 0)),
        "n_ILD": int(np.sum(y == 1)),
        "positive_prevalence": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "locked_threshold": float(threshold),
        "accuracy": float(accuracy_score(y, prediction)),
        "sensitivity_recall": float(recall_score(y, prediction, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "brier_score": float(brier_score_loss(y, probability)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def stratified_bootstrap_ci(
    y: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    metric_names = [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "sensitivity_recall",
        "specificity",
        "precision",
        "f1",
        "brier_score",
    ]
    values: Dict[str, List[float]] = {name: [] for name in metric_names}

    for _ in range(reps):
        sampled_negative = rng.choice(negative, size=len(negative), replace=True)
        sampled_positive = rng.choice(positive, size=len(positive), replace=True)
        index = np.concatenate([sampled_negative, sampled_positive])
        rng.shuffle(index)
        metrics = calculate_metrics(y[index], probability[index], threshold)
        for name in metric_names:
            values[name].append(float(metrics[name]))

    observed = calculate_metrics(y, probability, threshold)
    rows = []
    for name in metric_names:
        distribution = np.asarray(values[name], dtype=float)
        rows.append({
            "metric": name,
            "estimate": float(observed[name]),
            "bootstrap_reps": reps,
            "ci95_lower": float(np.quantile(distribution, 0.025)),
            "ci95_upper": float(np.quantile(distribution, 0.975)),
            "bootstrap_mean": float(np.mean(distribution)),
            "bootstrap_sd": float(np.std(distribution, ddof=1)),
            "bootstrap_method": "stratified percentile",
        })
    return pd.DataFrame(rows)


def make_evaluation_figure(
    y: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
    metrics: Mapping[str, float],
    output_png: Path,
    output_pdf: Path,
    dpi: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fpr, tpr, roc_thresholds = roc_curve(y, probability)
    precision, recall, pr_thresholds = precision_recall_curve(y, probability)

    roc_table = pd.DataFrame({
        "false_positive_rate": fpr,
        "true_positive_rate": tpr,
        "threshold": roc_thresholds,
    })
    pr_table = pd.DataFrame({
        "recall": recall,
        "precision": precision,
        "threshold": np.r_[pr_thresholds, np.nan],
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(fpr, tpr, linewidth=2)
    axes[0, 0].plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    axes[0, 0].set_xlabel("False positive rate")
    axes[0, 0].set_ylabel("True positive rate")
    axes[0, 0].set_title(f"ROC curve (AUC={metrics['roc_auc']:.3f})")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(recall, precision, linewidth=2)
    axes[0, 1].axhline(metrics["positive_prevalence"], linestyle="--", linewidth=1)
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].set_title(f"Precision–recall curve (AP={metrics['pr_auc']:.3f})")
    axes[0, 1].grid(alpha=0.25)

    cm = confusion_matrix(y, prediction, labels=[0, 1])
    image = axes[1, 0].imshow(cm)
    for row in range(2):
        for col in range(2):
            axes[1, 0].text(col, row, int(cm[row, col]), ha="center", va="center")
    axes[1, 0].set_xticks([0, 1], ["Predicted RA", "Predicted ILD"])
    axes[1, 0].set_yticks([0, 1], ["True RA", "True ILD"])
    axes[1, 0].set_title("Confusion matrix")
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)

    ra_probability = probability[y == 0]
    ild_probability = probability[y == 1]
    axes[1, 1].boxplot(
        [ra_probability, ild_probability],
        tick_labels=["RA", "RA-ILD"],
        showfliers=True,
    )
    axes[1, 1].axhline(metrics["locked_threshold"], linestyle="--", linewidth=1)
    axes[1, 1].set_ylabel("Predicted probability of RA-ILD")
    axes[1, 1].set_title("Locked-model probabilities")
    axes[1, 1].grid(axis="y", alpha=0.25)

    fig.suptitle("Independent Test Validation of the Locked Final IGH M2 Model", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    return roc_table, pr_table


def write_summary(
    path: Path,
    metrics: Mapping[str, float],
    ci: pd.DataFrame,
    args: argparse.Namespace,
    runtime_seconds: float,
) -> None:
    ci_lookup = ci.set_index("metric")

    def metric_line(name: str, label: str) -> str:
        row = ci_lookup.loc[name]
        return (
            f"- {label}: **{float(row['estimate']):.4f}** "
            f"(95% bootstrap CI {float(row['ci95_lower']):.4f}–"
            f"{float(row['ci95_upper']):.4f})"
        )

    lines = [
        "# 08 Independent Test Validation of Final IGH M2",
        "",
        "## Locked analysis",
        "",
        f"- Receptor: `{EXPECTED_RECEPTOR}`",
        f"- Model: `{MODEL_NAME}`",
        f"- Alpha: **{args.locked_alpha:.6g}**",
        f"- Lambda: **{args.locked_lambda:.6g}**",
        f"- Threshold: **{args.locked_threshold:.6f}**",
        "- The model was not refitted on the test set.",
        "- Training preprocessing parameters were reused unchanged.",
        "- No parameter, feature or threshold selection was performed on the test set.",
        "- Test labels were used only for final performance evaluation.",
        "",
        "## Test cohort",
        "",
        f"- Total: **{int(metrics['n_test'])}**",
        f"- RA: **{int(metrics['n_RA'])}**",
        f"- RA-ILD: **{int(metrics['n_ILD'])}**",
        "",
        "## Independent test performance",
        "",
        metric_line("roc_auc", "ROC-AUC"),
        metric_line("pr_auc", "PR-AUC"),
        metric_line("sensitivity_recall", "Sensitivity"),
        metric_line("specificity", "Specificity"),
        metric_line("precision", "Precision"),
        metric_line("f1", "F1"),
        metric_line("accuracy", "Accuracy"),
        metric_line("brier_score", "Brier score"),
        "",
        "## Confusion matrix",
        "",
        f"- TN: **{int(metrics['TN'])}**",
        f"- FP: **{int(metrics['FP'])}**",
        f"- FN: **{int(metrics['FN'])}**",
        f"- TP: **{int(metrics['TP'])}**",
        "",
        "## Interpretation boundary",
        "",
        "- These are the single locked independent-test results.",
        "- Bootstrap intervals quantify sampling uncertainty; they do not retune the model.",
        "- Any later alternative threshold analysis must be labeled exploratory and must not replace this primary result.",
        "",
        f"- Runtime: `{runtime_seconds:.2f}` seconds",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.time()

    input_paths = {
        "test_matrix": Path(args.test_matrix).expanduser().resolve(),
        "test_reference_features": Path(args.test_reference_features).expanduser().resolve(),
        "test_reference_definition": Path(args.test_reference_definition).expanduser().resolve(),
        "model_bundle": Path(args.model_bundle).expanduser().resolve(),
        "model_configuration": Path(args.model_configuration).expanduser().resolve(),
        "reference_masks": Path(args.reference_masks).expanduser().resolve(),
    }
    for label, path in input_paths.items():
        ensure_file(path, label)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)
    enforce_one_time_outputs(paths.values(), args.overwrite)

    print("=" * 80, flush=True)
    print("08 Locked Final IGH M2 — Independent Test Validation", flush=True)
    print("=" * 80, flush=True)
    print(f"Receptor: {EXPECTED_RECEPTOR}", flush=True)
    print(f"Expected test samples: {args.expected_test_samples}", flush=True)
    print(f"Locked alpha: {args.locked_alpha}", flush=True)
    print(f"Locked lambda: {args.locked_lambda}", flush=True)
    print(f"Locked threshold: {args.locked_threshold}", flush=True)
    print("No model fitting or test-set tuning will be performed.", flush=True)

    bundle, model_config, mask_counts = load_and_validate_locked_model(
        input_paths["model_bundle"],
        input_paths["model_configuration"],
        input_paths["reference_masks"],
        args,
    )
    reference_definition = validate_test_reference_audit(
        input_paths["test_reference_definition"],
        args.expected_test_samples,
        args.expected_training_samples,
        input_paths["reference_masks"],
    )
    test, _ = validate_and_align_test_data(
        input_paths["test_matrix"],
        input_paths["test_reference_features"],
        bundle,
        args,
    )
    X_test, feature_names = transform_with_locked_preprocessor(test, bundle)

    # Generate probabilities before consulting labels for evaluation.
    model = bundle["sklearn_model"]
    class_index = int(np.where(np.asarray(model.classes_) == 1)[0][0])
    probability = model.predict_proba(X_test)[:, class_index]
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Model produced invalid test probabilities.")

    labels_upper = test[args.label_col].astype(str).str.upper()
    y = (labels_upper == args.positive_label.upper()).astype(int).to_numpy()
    prediction = (probability >= args.locked_threshold).astype(int)

    metrics = calculate_metrics(y, probability, args.locked_threshold)
    metrics_table = pd.DataFrame([{
        "model": MODEL_NAME,
        "alpha": args.locked_alpha,
        "lambda": args.locked_lambda,
        **metrics,
        "parameter_selection_on_test": False,
        "threshold_selection_on_test": False,
        "model_refit_on_test": False,
    }])

    ci_table = stratified_bootstrap_ci(
        y,
        probability,
        args.locked_threshold,
        args.bootstrap_reps,
        args.bootstrap_seed,
    )

    predictions = pd.DataFrame({
        args.sample_id_col: test[args.sample_id_col].astype(str).to_numpy(),
        "true_cohort": test[args.label_col].astype(str).to_numpy(),
        "true_label": y,
        "probability_ILD": probability,
        "locked_threshold": args.locked_threshold,
        "predicted_label": prediction,
        "predicted_cohort": np.where(prediction == 1, args.positive_label, args.negative_label),
        "correct": prediction == y,
    })

    confusion = pd.DataFrame(
        confusion_matrix(y, prediction, labels=[0, 1]),
        index=["true_RA", "true_ILD"],
        columns=["predicted_RA", "predicted_ILD"],
    ).reset_index(names="true_class")

    roc_table, pr_table = make_evaluation_figure(
        y,
        probability,
        prediction,
        metrics,
        paths["figure_png"],
        paths["figure_pdf"],
        args.plot_dpi,
    )

    predictions.to_csv(paths["predictions"], index=False)
    metrics_table.to_csv(paths["metrics"], index=False)
    ci_table.to_csv(paths["bootstrap_ci"], index=False)
    confusion.to_csv(paths["confusion_matrix"], index=False)
    roc_table.to_csv(paths["roc_curve"], index=False)
    pr_table.to_csv(paths["pr_curve"], index=False)

    runtime = time.time() - started
    configuration = {
        "script_version": SCRIPT_VERSION,
        "receptor": EXPECTED_RECEPTOR,
        "model": MODEL_NAME,
        "analysis_type": "single_locked_independent_test_validation",
        "locked_parameters": {
            "alpha": args.locked_alpha,
            "lambda": args.locked_lambda,
            "threshold": args.locked_threshold,
        },
        "model_refit_on_test": False,
        "parameter_selection_on_test": False,
        "threshold_selection_on_test": False,
        "test_labels_used_for_prediction_features": False,
        "test_labels_used_for_evaluation_only": True,
        "input_files": {key: str(path) for key, path in input_paths.items()},
        "input_sha256": {key: sha256sum(path) for key, path in input_paths.items()},
        "model_bundle_script_version": bundle.get("script_version"),
        "model_fit_converged": bool(bundle.get("fit_converged")),
        "model_iterations_used": int(bundle.get("iterations_used", -1)),
        "model_feature_count": len(feature_names),
        "model_feature_names": feature_names,
        "training_sample_count": len(bundle.get("training_sample_ids", [])),
        "test_sample_count": len(test),
        "test_class_counts": {
            args.negative_label: int(metrics["n_RA"]),
            args.positive_label: int(metrics["n_ILD"]),
        },
        "public_reference_mask_counts": mask_counts,
        "test_reference_mode": reference_definition.get("mode"),
        "test_labels_used_to_build_reference": reference_definition.get(
            "test_labels_used_to_build_reference"
        ),
        "bootstrap": {
            "method": "stratified percentile",
            "reps": args.bootstrap_reps,
            "seed": args.bootstrap_seed,
        },
        "runtime_seconds": runtime,
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["configuration"].write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary(paths["summary"], metrics, ci_table, args, runtime)

    marker = {
        "status": "COMPLETE",
        "analysis_type": "single_locked_independent_test_validation",
        "receptor": EXPECTED_RECEPTOR,
        "model": MODEL_NAME,
        "alpha": args.locked_alpha,
        "lambda": args.locked_lambda,
        "threshold": args.locked_threshold,
        "test_matrix_sha256": sha256sum(input_paths["test_matrix"]),
        "model_bundle_sha256": sha256sum(input_paths["model_bundle"]),
        "configuration_sha256": sha256sum(paths["configuration"]),
        "metrics_sha256": sha256sum(paths["metrics"]),
    }
    paths["complete_marker"].write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n[Independent test metrics]", flush=True)
    print(metrics_table.to_string(index=False), flush=True)
    print("\n[95% stratified bootstrap confidence intervals]", flush=True)
    print(ci_table.to_string(index=False), flush=True)
    print("\n[Output files]", flush=True)
    for key, path in paths.items():
        print(f"- {key}: {path}", flush=True)
    print(f"Runtime: {runtime:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
