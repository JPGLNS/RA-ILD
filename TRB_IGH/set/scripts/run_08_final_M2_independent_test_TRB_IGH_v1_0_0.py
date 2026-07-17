#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""One-time independent validation of the locked paired TRB+IGH M2 model.

Locked model
------------
M2_static_trb_igh_public
    alpha = 0.5
    lambda = 30
    threshold = 0.5009465713068705

Strict boundary
---------------
* Load the fitted step-07 model bundle; never refit on the test cohort.
* Reuse the locked training preprocessing exactly.
* Never select parameters, features, or thresholds on the test cohort.
* Validate both receptor-specific test public-feature audits and verify that
  their values match the paired step-04 test matrix.
* Generate probabilities before test labels are used for outcome evaluation.
* Write a COMPLETE marker last; a completed validation cannot be overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

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


SCRIPT_VERSION = "1.0.0-PAIRED-TRB-IGH"
ANALYSIS_VIEW = "TRB_IGH"
CV_UNIT = "patient"
MODEL_NAME = "M2_static_trb_igh_public"
ROOT = Path("/data/users/chenhaisheng/RA-ILD/TRB_IGH")

DEFAULT_TEST_MATRIX = (
    ROOT / "set/test/result/04_final_feature_matrix/04_test_final_feature_matrix.csv"
)
DEFAULT_TRB_REFERENCE_FEATURES = (
    ROOT / "set/test/TRB/result/03_public_features/03_test_reference_public_features.csv"
)
DEFAULT_IGH_REFERENCE_FEATURES = (
    ROOT / "set/test/IGH/result/03_public_features/03_test_reference_public_features.csv"
)
DEFAULT_TRB_REFERENCE_DEFINITION = (
    ROOT / "set/test/TRB/result/03_public_features/03_reference_definition_used.json"
)
DEFAULT_IGH_REFERENCE_DEFINITION = (
    ROOT / "set/test/IGH/result/03_public_features/03_reference_definition_used.json"
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

LOCKED_ALPHA = 0.5
LOCKED_LAMBDA = 30.0
LOCKED_THRESHOLD = 0.5009465713068705

RECEPTORS = ("TRB", "IGH")
MASK_NAME_MAP = {
    "global_mask": "global",
    "RA_specific_mask": "RA_specific",
    "ILD_specific_mask": "ILD_specific",
    "shared_mask": "shared",
}
DEFINITION_SIZE_MAP = {
    "global_mask": "global_public",
    "RA_specific_mask": "RA_specific_ref",
    "ILD_specific_mask": "ILD_specific_ref",
    "shared_mask": "between_group_shared",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the locked final paired TRB+IGH M2 model once to the "
            "independent test cohort."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test-matrix", default=str(DEFAULT_TEST_MATRIX))
    parser.add_argument(
        "--trb-test-reference-features", default=str(DEFAULT_TRB_REFERENCE_FEATURES)
    )
    parser.add_argument(
        "--igh-test-reference-features", default=str(DEFAULT_IGH_REFERENCE_FEATURES)
    )
    parser.add_argument(
        "--trb-test-reference-definition", default=str(DEFAULT_TRB_REFERENCE_DEFINITION)
    )
    parser.add_argument(
        "--igh-test-reference-definition", default=str(DEFAULT_IGH_REFERENCE_DEFINITION)
    )
    parser.add_argument("--model-bundle", default=str(DEFAULT_MODEL_BUNDLE))
    parser.add_argument(
        "--model-configuration", default=str(DEFAULT_MODEL_CONFIGURATION)
    )
    parser.add_argument("--reference-masks", default=str(DEFAULT_REFERENCE_MASKS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    parser.add_argument("--sample-id-col", default="sample_id")
    parser.add_argument("--patient-col", default="patient")
    parser.add_argument("--trb-library-id-col", default="trb_libraryid")
    parser.add_argument("--igh-library-id-col", default="igh_libraryid")
    parser.add_argument("--label-col", default="cohort")
    parser.add_argument("--positive-label", default="ILD")
    parser.add_argument("--negative-label", default="RA")

    parser.add_argument("--expected-test-samples", type=int, default=49)
    parser.add_argument("--expected-training-samples", type=int, default=118)
    parser.add_argument("--expected-training-ra", type=int, default=72)
    parser.add_argument("--expected-training-ild", type=int, default=46)

    parser.add_argument("--locked-alpha", type=float, default=LOCKED_ALPHA)
    parser.add_argument("--locked-lambda", type=float, default=LOCKED_LAMBDA)
    parser.add_argument("--locked-threshold", type=float, default=LOCKED_THRESHOLD)
    parser.add_argument("--float-tolerance", type=float, default=1e-9)
    parser.add_argument("--public-value-tolerance", type=float, default=1e-9)

    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260717)
    parser.add_argument("--plot-dpi", type=int, default=400)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate locked artifacts, test schemas, public-feature audits, and "
            "preprocessing compatibility without generating probabilities, reading "
            "test labels, or writing outputs."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace incomplete prior outputs after a documented technical failure. "
            "A directory containing 08_VALIDATION_COMPLETE.json can never be overwritten."
        ),
    )
    args = parser.parse_args()

    if args.expected_test_samples < 2:
        parser.error("--expected-test-samples must be >=2.")
    if args.expected_training_samples < 2:
        parser.error("--expected-training-samples must be >=2.")
    if args.expected_training_ra < 1 or args.expected_training_ild < 1:
        parser.error("Expected training class counts must both be >=1.")
    if args.expected_training_ra + args.expected_training_ild != args.expected_training_samples:
        parser.error("Expected training RA + ILD must equal expected training samples.")
    if not (0 < args.locked_alpha <= 1):
        parser.error("--locked-alpha must be in (0,1].")
    if args.locked_lambda <= 0:
        parser.error("--locked-lambda must be >0.")
    if not (0 < args.locked_threshold < 1):
        parser.error("--locked-threshold must be in (0,1).")
    if args.float_tolerance < 0 or args.public_value_tolerance < 0:
        parser.error("Tolerances must be nonnegative.")
    if args.bootstrap_reps < 100:
        parser.error("--bootstrap-reps must be >=100.")
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


def close_enough(observed: object, expected: float, tolerance: float) -> bool:
    try:
        return math.isclose(
            float(observed), float(expected), rel_tol=tolerance, abs_tol=tolerance
        )
    except (TypeError, ValueError):
        return False


def as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


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
        "integrity": output_dir / "08_input_integrity_checks.csv",
        "configuration": output_dir / "08_independent_test_configuration.json",
        "summary": output_dir / "08_independent_test_summary.md",
        "complete_marker": output_dir / "08_VALIDATION_COMPLETE.json",
    }


def enforce_one_time_outputs(
    paths: Mapping[str, Path], overwrite: bool, preflight_only: bool
) -> None:
    complete = paths["complete_marker"]
    if complete.exists():
        raise FileExistsError(
            "Independent-test validation is already marked COMPLETE and cannot be "
            f"run again: {complete}"
        )
    existing = [path for path in paths.values() if path.exists()]
    if existing and not preflight_only and not overwrite:
        raise FileExistsError(
            "Incomplete prior validation outputs exist. Review the technical failure; "
            "use --overwrite only for recovery:\n"
            + "\n".join(f"  - {path}" for path in existing)
        )


def record_check(
    checks: List[Dict[str, object]], check: str, detail: str = "PASS"
) -> None:
    checks.append({"status": "PASS", "check": check, "detail": detail})


def resolve_input_paths(args: argparse.Namespace) -> Dict[str, Path]:
    return {
        "test_matrix": Path(args.test_matrix).expanduser().resolve(),
        "trb_test_reference_features": Path(
            args.trb_test_reference_features
        ).expanduser().resolve(),
        "igh_test_reference_features": Path(
            args.igh_test_reference_features
        ).expanduser().resolve(),
        "trb_test_reference_definition": Path(
            args.trb_test_reference_definition
        ).expanduser().resolve(),
        "igh_test_reference_definition": Path(
            args.igh_test_reference_definition
        ).expanduser().resolve(),
        "model_bundle": Path(args.model_bundle).expanduser().resolve(),
        "model_configuration": Path(args.model_configuration).expanduser().resolve(),
        "reference_masks": Path(args.reference_masks).expanduser().resolve(),
    }


def validate_reference_masks(
    path: Path,
    bundle: Mapping[str, object],
    config: Mapping[str, object],
    args: argparse.Namespace,
    checks: List[Dict[str, object]],
) -> Dict[str, Dict[str, int]]:
    actual_sha = sha256sum(path)
    for label, value in (
        ("bundle", bundle.get("public_reference_masks_sha256")),
        ("configuration", config.get("full_reference_masks_sha256")),
    ):
        if value and str(value) != actual_sha:
            raise ValueError(f"Reference-mask SHA256 differs from {label}.")
    record_check(checks, "reference-mask SHA256 matches step-07 bundle/configuration")

    with np.load(path, allow_pickle=False) as masks:
        if "analysis_view" not in masks.files:
            raise ValueError("Reference-mask NPZ lacks analysis_view.")
        observed_view = str(np.asarray(masks["analysis_view"]).item())
        if observed_view != ANALYSIS_VIEW:
            raise ValueError(
                f"Reference-mask analysis_view={observed_view}; expected={ANALYSIS_VIEW}."
            )
        if "cv_unit" in masks.files:
            observed_unit = str(np.asarray(masks["cv_unit"]).item())
            if observed_unit != CV_UNIT:
                raise ValueError(
                    f"Reference-mask cv_unit={observed_unit}; expected={CV_UNIT}."
                )

        counts: Dict[str, Dict[str, int]] = {}
        for receptor in RECEPTORS:
            prefix = receptor.lower()
            metadata_expected = {
                f"{prefix}_n_reference": args.expected_training_samples,
                f"{prefix}_n_RA_reference": args.expected_training_ra,
                f"{prefix}_n_ILD_reference": args.expected_training_ild,
            }
            for key, expected in metadata_expected.items():
                if key not in masks.files:
                    raise ValueError(f"Reference-mask NPZ missing {key}.")
                observed = int(np.asarray(masks[key]).item())
                if observed != expected:
                    raise ValueError(f"{key}={observed}; expected={expected}.")

            receptor_bundle = bundle.get("receptors", {}).get(receptor, {})
            if not isinstance(receptor_bundle, Mapping):
                raise ValueError(f"Bundle lacks receptor metadata for {receptor}.")
            expected_catalog_sha = str(receptor_bundle.get("public_catalog_sha256", ""))
            expected_definition_sha = str(
                receptor_bundle.get("reference_definition_sha256", "")
            )
            for suffix, expected_sha in (
                ("public_catalog_sha256", expected_catalog_sha),
                ("reference_definition_sha256", expected_definition_sha),
            ):
                key = f"{prefix}_{suffix}"
                if key not in masks.files:
                    raise ValueError(f"Reference-mask NPZ missing {key}.")
                observed_sha = str(np.asarray(masks[key]).item())
                if expected_sha and observed_sha != expected_sha:
                    raise ValueError(f"{receptor} {suffix} differs between mask and bundle.")

            mask_lengths: set[int] = set()
            receptor_counts: Dict[str, int] = {}
            for mask_suffix, result_name in MASK_NAME_MAP.items():
                key = f"{prefix}_{mask_suffix}"
                if key not in masks.files:
                    raise ValueError(f"Reference-mask NPZ missing {key}.")
                array = np.asarray(masks[key]).astype(bool)
                mask_lengths.add(len(array))
                receptor_counts[result_name] = int(array.sum())
            if len(mask_lengths) != 1:
                raise ValueError(f"{receptor} reference masks have unequal lengths.")

            ra = np.asarray(masks[f"{prefix}_RA_specific_mask"]).astype(bool)
            ild = np.asarray(masks[f"{prefix}_ILD_specific_mask"]).astype(bool)
            shared = np.asarray(masks[f"{prefix}_shared_mask"]).astype(bool)
            if np.any(ra & ild) or np.any(ra & shared) or np.any(ild & shared):
                raise ValueError(f"{receptor} specific/shared reference masks overlap.")

            summary = receptor_bundle.get("full_public_reference_summary", {})
            if isinstance(summary, Mapping):
                expected_count_keys = {
                    "global": "all_ref_public_size",
                    "RA_specific": "RA_specific_ref_size",
                    "ILD_specific": "ILD_specific_ref_size",
                    "shared": "shared_ref_size",
                }
                for name, summary_key in expected_count_keys.items():
                    if summary_key in summary and int(summary[summary_key]) != receptor_counts[name]:
                        raise ValueError(
                            f"{receptor} mask {name} count differs from bundle summary."
                        )
            counts[receptor] = receptor_counts
            record_check(
                checks,
                f"{receptor} full-training reference masks validated",
                ", ".join(f"{k}={v:,}" for k, v in receptor_counts.items()),
            )
    return counts


def validate_locked_model(
    bundle_path: Path,
    config_path: Path,
    masks_path: Path,
    args: argparse.Namespace,
    checks: List[Dict[str, object]],
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, Dict[str, int]]]:
    bundle = joblib.load(bundle_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(bundle, dict) or not isinstance(config, dict):
        raise TypeError("Step-07 bundle/configuration must be dictionaries.")

    expected_pairs = [
        ("bundle analysis_view", bundle.get("analysis_view"), ANALYSIS_VIEW),
        ("configuration analysis_view", config.get("analysis_view"), ANALYSIS_VIEW),
        ("bundle cv_unit", bundle.get("cv_unit"), CV_UNIT),
        ("configuration cv_unit", config.get("cv_unit"), CV_UNIT),
        ("bundle model", bundle.get("model_name"), MODEL_NAME),
        ("configuration model", config.get("model"), MODEL_NAME),
    ]
    for label, observed, expected in expected_pairs:
        if str(observed) != expected:
            raise ValueError(f"{label}={observed!r}; expected={expected!r}.")
    if config.get("independent_test_read") is not False:
        raise ValueError("Step-07 configuration does not confirm independent_test_read=false.")
    if not as_bool(bundle.get("fit_converged", False)):
        raise ValueError("Saved final model is not marked converged.")
    if not as_bool(config.get("final_fit_converged", False)):
        raise ValueError("Step-07 configuration final fit is not marked converged.")
    record_check(checks, "locked paired M2 bundle/configuration identity and convergence")

    training_ids = [str(value) for value in bundle.get("training_sample_ids", [])]
    if len(training_ids) != args.expected_training_samples:
        raise ValueError(
            f"Bundle contains {len(training_ids)} training patients; "
            f"expected={args.expected_training_samples}."
        )
    if len(set(training_ids)) != len(training_ids):
        raise ValueError("Bundle training patient IDs are duplicated.")
    for source_name, source in (("bundle", bundle), ("configuration", config)):
        for field, expected in (
            ("n_training_samples", args.expected_training_samples),
            ("n_RA", args.expected_training_ra),
            ("n_ILD", args.expected_training_ild),
        ):
            if int(source.get(field, -1)) != expected:
                raise ValueError(
                    f"{source_name} {field}={source.get(field)}; expected={expected}."
                )
    record_check(
        checks,
        "training cohort identity counts validated",
        f"n={args.expected_training_samples}, RA={args.expected_training_ra}, ILD={args.expected_training_ild}",
    )

    locked_checks = [
        ("bundle alpha", bundle.get("selected_alpha"), args.locked_alpha),
        ("bundle lambda", bundle.get("selected_lambda"), args.locked_lambda),
        ("bundle threshold", bundle.get("locked_threshold"), args.locked_threshold),
        ("configuration alpha", config.get("selected_alpha"), args.locked_alpha),
        ("configuration lambda", config.get("selected_lambda"), args.locked_lambda),
        ("configuration threshold", config.get("locked_threshold"), args.locked_threshold),
    ]
    for label, observed, expected in locked_checks:
        if not close_enough(observed, expected, args.float_tolerance):
            raise ValueError(
                f"Locked value mismatch for {label}: observed={observed}, expected={expected}."
            )
    record_check(
        checks,
        "alpha/lambda/threshold exactly match locked step-07 values",
        f"alpha={args.locked_alpha:g}, lambda={args.locked_lambda:g}, threshold={args.locked_threshold:.12f}",
    )

    model = bundle.get("sklearn_model")
    if model is None or not hasattr(model, "predict_proba"):
        raise TypeError("Bundle lacks a usable sklearn probability model.")
    classes = np.asarray(getattr(model, "classes_", []))
    if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
        raise ValueError(f"Unexpected sklearn class labels: {classes.tolist()}.")

    joint_public = [str(x) for x in bundle.get("joint_public_features", [])]
    if len(joint_public) != 16:
        raise ValueError(
            f"Bundle contains {len(joint_public)} joint public features; expected 16."
        )
    if sum(name.startswith("trb_") for name in joint_public) != 8 or sum(
        name.startswith("igh_") for name in joint_public
    ) != 8:
        raise ValueError("Joint public-feature list is not 8 TRB + 8 IGH.")
    record_check(checks, "joint dynamic-public feature list is 8 TRB + 8 IGH")

    mask_counts = validate_reference_masks(masks_path, bundle, config, args, checks)
    return bundle, config, mask_counts


def validate_test_reference_definition(
    path: Path,
    receptor: str,
    bundle: Mapping[str, object],
    mask_counts: Mapping[str, int],
    args: argparse.Namespace,
    checks: List[Dict[str, object]],
) -> Dict[str, object]:
    definition = json.loads(path.read_text(encoding="utf-8"))
    if str(definition.get("receptor", "")).upper() != receptor:
        raise ValueError(
            f"{receptor} test definition receptor={definition.get('receptor')!r}."
        )
    if definition.get("mode") != "test_transform":
        raise ValueError(f"{receptor} test definition is not test_transform mode.")
    if definition.get("test_labels_used_to_build_reference") is not False:
        raise ValueError(f"{receptor} test reference may have used test labels.")
    if int(definition.get("test_sample_count", -1)) != args.expected_test_samples:
        raise ValueError(f"{receptor} test definition sample count mismatch.")
    for field, expected in (
        ("n_training_samples", args.expected_training_samples),
        ("n_RA_training_samples", args.expected_training_ra),
        ("n_ILD_training_samples", args.expected_training_ild),
    ):
        if field in definition and int(definition[field]) != expected:
            raise ValueError(
                f"{receptor} test definition {field}={definition[field]}; expected={expected}."
            )

    original_hash = str(definition.get("reference_file_sha256", ""))
    verified_hash = str(definition.get("reference_file_sha256_verified", ""))
    if len(original_hash) != 64 or verified_hash != original_hash:
        raise ValueError(f"{receptor} training reference SHA256 was not verified in test mode.")
    receptor_bundle = bundle.get("receptors", {}).get(receptor, {})
    training_definition = receptor_bundle.get("reference_definition", {})
    if isinstance(training_definition, Mapping):
        bundle_reference_hash = str(training_definition.get("reference_file_sha256", ""))
        if bundle_reference_hash and bundle_reference_hash != verified_hash:
            raise ValueError(
                f"{receptor} test audit reference hash differs from locked bundle definition."
            )
        candidate_count = int(training_definition.get("candidate_sequence_count", -1))
        observed_candidate = int(definition.get("candidate_sequence_count", -1))
        if candidate_count > 0 and observed_candidate != candidate_count:
            raise ValueError(f"{receptor} candidate sequence count differs from bundle.")

    sizes = definition.get("reference_set_sizes", {})
    if not isinstance(sizes, Mapping):
        raise ValueError(f"{receptor} test definition lacks reference_set_sizes.")
    for mask_name, definition_name in DEFINITION_SIZE_MAP.items():
        result_name = MASK_NAME_MAP[mask_name]
        if int(sizes.get(definition_name, -1)) != int(mask_counts[result_name]):
            raise ValueError(
                f"{receptor} {definition_name} differs from locked reference mask count."
            )
    record_check(
        checks,
        f"{receptor} test public-reference audit validated",
        "test labels unused; locked training reference SHA256 verified",
    )
    return definition


def normalize_joint_test_matrix(
    path: Path, args: argparse.Namespace, checks: List[Dict[str, object]]
) -> pd.DataFrame:
    test = pd.read_csv(path)
    if args.sample_id_col not in test.columns:
        if args.patient_col in test.columns:
            test[args.sample_id_col] = test[args.patient_col]
        else:
            raise ValueError(
                f"Test matrix lacks both {args.sample_id_col!r} and {args.patient_col!r}."
            )
    required = {
        args.sample_id_col,
        args.trb_library_id_col,
        args.igh_library_id_col,
        args.label_col,
    }
    missing = sorted(required - set(test.columns))
    if missing:
        raise ValueError(f"Paired test matrix missing columns: {missing}.")
    for column in (
        args.sample_id_col,
        args.trb_library_id_col,
        args.igh_library_id_col,
    ):
        test[column] = test[column].astype(str)
        if test[column].duplicated().any():
            raise ValueError(f"Paired test matrix has duplicated {column}.")
        if test[column].str.strip().eq("").any():
            raise ValueError(f"Paired test matrix has blank {column}.")
    if len(test) != args.expected_test_samples:
        raise ValueError(
            f"Paired test matrix has {len(test)} rows; expected={args.expected_test_samples}."
        )
    if test.isna().any().any():
        missing_counts = test.isna().sum()
        raise ValueError(
            "Paired test matrix contains missing values: "
            + str(missing_counts[missing_counts > 0].to_dict())
        )
    record_check(
        checks,
        "paired test matrix structure validated",
        f"{len(test)} patients with unique patient/TRB/IGH IDs",
    )
    return test.set_index(args.sample_id_col, drop=False)


def validate_no_training_overlap(
    test: pd.DataFrame,
    bundle: Mapping[str, object],
    args: argparse.Namespace,
    checks: List[Dict[str, object]],
) -> None:
    training_patients = set(str(x) for x in bundle.get("training_sample_ids", []))
    test_patients = set(test[args.sample_id_col].astype(str))
    overlap = training_patients & test_patients
    if overlap:
        raise ValueError(f"Training/test patient overlap: {sorted(overlap)[:20]}.")

    mapping = bundle.get("training_patient_library_mapping", [])
    if isinstance(mapping, Sequence):
        train_trb = {
            str(row.get(args.trb_library_id_col))
            for row in mapping
            if isinstance(row, Mapping) and row.get(args.trb_library_id_col) is not None
        }
        train_igh = {
            str(row.get(args.igh_library_id_col))
            for row in mapping
            if isinstance(row, Mapping) and row.get(args.igh_library_id_col) is not None
        }
        trb_overlap = train_trb & set(test[args.trb_library_id_col].astype(str))
        igh_overlap = train_igh & set(test[args.igh_library_id_col].astype(str))
        if trb_overlap or igh_overlap:
            raise ValueError(
                "Training/test library overlap detected: "
                f"TRB={sorted(trb_overlap)[:10]}, IGH={sorted(igh_overlap)[:10]}."
            )
    record_check(checks, "no training/test overlap in patient or paired library IDs")


def validate_receptor_public_features(
    test: pd.DataFrame,
    path: Path,
    receptor: str,
    bundle: Mapping[str, object],
    args: argparse.Namespace,
    checks: List[Dict[str, object]],
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "sample_id" not in frame.columns:
        raise ValueError(f"{receptor} test public-feature file lacks sample_id.")
    frame["sample_id"] = frame["sample_id"].astype(str)
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{receptor} test public-feature file has duplicate sample_id.")
    if len(frame) != args.expected_test_samples:
        raise ValueError(
            f"{receptor} test public-feature file has {len(frame)} rows; "
            f"expected={args.expected_test_samples}."
        )

    library_col = (
        args.trb_library_id_col if receptor == "TRB" else args.igh_library_id_col
    )
    patient_ids = set(test[args.sample_id_col].astype(str))
    library_ids = set(test[library_col].astype(str))
    observed_ids = set(frame["sample_id"])
    if observed_ids == library_ids:
        patient_lookup = dict(
            zip(test[library_col].astype(str), test[args.sample_id_col].astype(str))
        )
        frame[args.sample_id_col] = frame["sample_id"].map(patient_lookup)
        id_mode = library_col
    elif observed_ids == patient_ids:
        frame[args.sample_id_col] = frame["sample_id"]
        id_mode = args.sample_id_col
    else:
        missing = sorted(library_ids - observed_ids)[:10]
        extra = sorted(observed_ids - library_ids)[:10]
        raise ValueError(
            f"{receptor} public-feature IDs match neither patients nor {library_col}; "
            f"missing examples={missing}, extra examples={extra}."
        )
    frame = frame.set_index(args.sample_id_col).loc[test.index]

    base_public = [str(x) for x in bundle.get("public_features_per_receptor", [])]
    if len(base_public) != 8:
        raise ValueError("Bundle public_features_per_receptor must contain 8 features.")
    missing_ref = [name for name in base_public if name not in frame.columns]
    if missing_ref:
        raise ValueError(f"{receptor} public-feature file missing: {missing_ref}.")

    prefix = receptor.lower() + "_"
    missing_joint = [prefix + name for name in base_public if prefix + name not in test.columns]
    if missing_joint:
        raise ValueError(f"Paired test matrix missing {receptor} public features: {missing_joint}.")

    maximum_difference = 0.0
    for base_name in base_public:
        joint_name = prefix + base_name
        left = pd.to_numeric(test[joint_name], errors="raise").to_numpy(float)
        right = pd.to_numeric(frame[base_name], errors="raise").to_numpy(float)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(f"Non-finite values in {receptor} public feature {base_name}.")
        difference = float(np.max(np.abs(left - right)))
        maximum_difference = max(maximum_difference, difference)
        if difference > args.public_value_tolerance:
            raise ValueError(
                f"{receptor} public feature {base_name} differs between step-04 and "
                f"step-03; max abs difference={difference:.6g}."
            )
    record_check(
        checks,
        f"{receptor} step-04 public values match receptor-specific step-03 output",
        f"ID alignment={id_mode}; max abs difference={maximum_difference:.3g}",
    )
    return frame


def transform_with_locked_preprocessor(
    test: pd.DataFrame,
    bundle: Mapping[str, object],
    checks: List[Dict[str, object]],
) -> Tuple[np.ndarray, List[str]]:
    preprocessor = bundle.get("preprocessor")
    if not isinstance(preprocessor, Mapping):
        raise TypeError("Model bundle lacks preprocessor metadata.")
    numeric_columns = [str(x) for x in preprocessor.get("numeric_columns", [])]
    categorical_columns = [str(x) for x in preprocessor.get("categorical_columns", [])]
    categorical_levels = dict(preprocessor.get("categorical_levels", {}))
    categorical_reference = dict(preprocessor.get("categorical_reference", {}))
    final_feature_names = [str(x) for x in preprocessor.get("final_feature_names", [])]
    preprocessing_rows = list(preprocessor.get("preprocessing_rows", []))
    if not numeric_columns or not final_feature_names or not preprocessing_rows:
        raise ValueError("Locked preprocessor metadata is incomplete.")

    missing_numeric = [name for name in numeric_columns if name not in test.columns]
    missing_categorical = [name for name in categorical_columns if name not in test.columns]
    if missing_numeric or missing_categorical:
        raise ValueError(
            f"Test predictors missing: numeric={missing_numeric}, categorical={missing_categorical}."
        )

    raw_features: Dict[str, np.ndarray] = {}
    for column in numeric_columns:
        values = pd.to_numeric(test[column], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values in predictor {column}.")
        raw_features[column] = values

    for column in categorical_columns:
        values = test[column].astype(str)
        levels = [str(value) for value in categorical_levels.get(column, [])]
        if not levels:
            raise ValueError(f"No stored training levels for categorical predictor {column}.")
        reference = str(categorical_reference.get(column, levels[0]))
        unseen = sorted(set(values) - set(levels))
        if unseen:
            raise ValueError(
                f"Unseen test categories for {column}: {unseen}; training levels={levels}."
            )
        for category in levels:
            if category == reference:
                continue
            name = f"{column}__{category}_vs_{reference}"
            raw_features[name] = (values == category).to_numpy(float)

    prep = pd.DataFrame(preprocessing_rows)
    required = {
        "feature_name",
        "training_mean",
        "training_sd",
        "kept_after_zero_variance_filter",
    }
    missing = sorted(required - set(prep.columns))
    if missing:
        raise ValueError(f"Preprocessor rows missing columns: {missing}.")
    if prep["feature_name"].astype(str).duplicated().any():
        raise ValueError("Preprocessor contains duplicated feature names.")
    prep["feature_name"] = prep["feature_name"].astype(str)
    prep = prep.set_index("feature_name")

    transformed: List[np.ndarray] = []
    for feature_name in final_feature_names:
        if feature_name not in raw_features:
            raise ValueError(f"Unable to reconstruct final feature {feature_name}.")
        if feature_name not in prep.index:
            raise ValueError(f"No preprocessing row for final feature {feature_name}.")
        row = prep.loc[feature_name]
        if not as_bool(row["kept_after_zero_variance_filter"]):
            raise ValueError(f"Final feature {feature_name} was not marked kept.")
        mean = float(row["training_mean"])
        sd = float(row["training_sd"])
        if not np.isfinite(mean) or not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"Invalid locked mean/SD for {feature_name}: {mean}, {sd}.")
        transformed.append(((raw_features[feature_name] - mean) / sd)[:, None])

    X = np.hstack(transformed)
    if not np.isfinite(X).all():
        raise ValueError("Locked preprocessing produced non-finite values.")
    model = bundle["sklearn_model"]
    expected_features = int(getattr(model, "n_features_in_", X.shape[1]))
    if X.shape[1] != expected_features:
        raise ValueError(
            f"Transformed feature count={X.shape[1]}; model expects={expected_features}."
        )
    record_check(
        checks,
        "locked training preprocessing reconstructs the exact model design",
        f"n_test={X.shape[0]}, n_model_features={X.shape[1]}",
    )
    return X, final_feature_names


def calculate_metrics(
    y: np.ndarray, probability: np.ndarray, threshold: float
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
    if len(negative) == 0 or len(positive) == 0:
        raise ValueError("Both RA and ILD are required for stratified bootstrap.")
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
    distributions: Dict[str, List[float]] = {name: [] for name in metric_names}
    for _ in range(reps):
        sampled_negative = rng.choice(negative, size=len(negative), replace=True)
        sampled_positive = rng.choice(positive, size=len(positive), replace=True)
        index = np.concatenate([sampled_negative, sampled_positive])
        rng.shuffle(index)
        current = calculate_metrics(y[index], probability[index], threshold)
        for name in metric_names:
            distributions[name].append(float(current[name]))
    observed = calculate_metrics(y, probability, threshold)
    rows = []
    for name in metric_names:
        values = np.asarray(distributions[name], dtype=float)
        rows.append(
            {
                "metric": name,
                "estimate": float(observed[name]),
                "bootstrap_reps": reps,
                "ci95_lower": float(np.quantile(values, 0.025)),
                "ci95_upper": float(np.quantile(values, 0.975)),
                "bootstrap_mean": float(np.mean(values)),
                "bootstrap_sd": float(np.std(values, ddof=1)),
                "bootstrap_method": "stratified percentile",
            }
        )
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
    roc_table = pd.DataFrame(
        {
            "false_positive_rate": fpr,
            "true_positive_rate": tpr,
            "threshold": roc_thresholds,
        }
    )
    pr_table = pd.DataFrame(
        {
            "recall": recall,
            "precision": precision,
            "threshold": np.r_[pr_thresholds, np.nan],
        }
    )

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
    axes[0, 1].set_title(f"Precision-recall curve (AP={metrics['pr_auc']:.3f})")
    axes[0, 1].grid(alpha=0.25)

    cm = confusion_matrix(y, prediction, labels=[0, 1])
    image = axes[1, 0].imshow(cm)
    for row in range(2):
        for column in range(2):
            axes[1, 0].text(column, row, int(cm[row, column]), ha="center", va="center")
    axes[1, 0].set_xticks([0, 1], labels=["Predicted RA", "Predicted RA-ILD"])
    axes[1, 0].set_yticks([0, 1], labels=["True RA", "True RA-ILD"])
    axes[1, 0].set_title("Confusion matrix")
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].boxplot(
        [probability[y == 0], probability[y == 1]], tick_labels=["RA", "RA-ILD"]
    )
    axes[1, 1].axhline(metrics["locked_threshold"], linestyle="--", linewidth=1)
    axes[1, 1].set_ylabel("Predicted probability of RA-ILD")
    axes[1, 1].set_title("Locked-model probabilities")
    axes[1, 1].grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Independent Test Validation of the Locked Paired TRB+IGH M2 Model",
        fontsize=15,
    )
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
    lookup = ci.set_index("metric")

    def metric_line(name: str, label: str) -> str:
        row = lookup.loc[name]
        return (
            f"- {label}: **{float(row['estimate']):.4f}** "
            f"(95% stratified bootstrap CI {float(row['ci95_lower']):.4f}–"
            f"{float(row['ci95_upper']):.4f})"
        )

    lines = [
        "# 08 Independent Test Validation of Final Paired TRB+IGH M2",
        "",
        "## Locked analysis",
        "",
        f"- Script version: `{SCRIPT_VERSION}`",
        f"- Analysis view: `{ANALYSIS_VIEW}`",
        f"- CV unit: `{CV_UNIT}`",
        f"- Model: `{MODEL_NAME}`",
        f"- Alpha: **{args.locked_alpha:.6g}**",
        f"- Lambda: **{args.locked_lambda:.6g}**",
        f"- Threshold: **{args.locked_threshold:.12f}**",
        "- The model was not refitted on the test set.",
        "- Training preprocessing parameters were reused unchanged.",
        "- Both receptor-specific public references were locked from the training set.",
        "- No parameter, feature, or threshold selection was performed on the test set.",
        "- Test labels were accessed only after locked probabilities were generated.",
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
        "- Bootstrap intervals quantify sampling uncertainty and do not retune the model.",
        "- Any later alternative-threshold analysis is exploratory and cannot replace this primary result.",
        "",
        f"- Runtime: `{runtime_seconds:.2f}` seconds",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.time()
    input_paths = resolve_input_paths(args)
    for label, path in input_paths.items():
        ensure_file(path, label)

    output_dir = Path(args.output_dir).expanduser().resolve()
    paths = output_paths(output_dir)
    enforce_one_time_outputs(paths, args.overwrite, args.preflight_only)

    print("=" * 88, flush=True)
    print("08 Locked Final Paired TRB+IGH M2 — One-Time Independent Validation", flush=True)
    print("=" * 88, flush=True)
    print(f"Script version: {SCRIPT_VERSION}", flush=True)
    print(f"Analysis view: {ANALYSIS_VIEW}", flush=True)
    print(f"CV unit: {CV_UNIT}", flush=True)
    print(f"Expected test patients: {args.expected_test_samples}", flush=True)
    print(f"Locked alpha: {args.locked_alpha}", flush=True)
    print(f"Locked lambda: {args.locked_lambda}", flush=True)
    print(f"Locked threshold: {args.locked_threshold:.12f}", flush=True)
    print("No refitting, parameter selection, feature selection, or threshold tuning.", flush=True)
    print("", flush=True)

    checks: List[Dict[str, object]] = []
    bundle, model_config, mask_counts = validate_locked_model(
        input_paths["model_bundle"],
        input_paths["model_configuration"],
        input_paths["reference_masks"],
        args,
        checks,
    )
    definitions = {
        "TRB": validate_test_reference_definition(
            input_paths["trb_test_reference_definition"],
            "TRB",
            bundle,
            mask_counts["TRB"],
            args,
            checks,
        ),
        "IGH": validate_test_reference_definition(
            input_paths["igh_test_reference_definition"],
            "IGH",
            bundle,
            mask_counts["IGH"],
            args,
            checks,
        ),
    }
    test = normalize_joint_test_matrix(input_paths["test_matrix"], args, checks)
    validate_no_training_overlap(test, bundle, args, checks)
    validate_receptor_public_features(
        test,
        input_paths["trb_test_reference_features"],
        "TRB",
        bundle,
        args,
        checks,
    )
    validate_receptor_public_features(
        test,
        input_paths["igh_test_reference_features"],
        "IGH",
        bundle,
        args,
        checks,
    )
    X_test, feature_names = transform_with_locked_preprocessor(test, bundle, checks)

    print("[Pre-prediction integrity checks]", flush=True)
    for row in checks:
        print(f"- PASS: {row['check']} — {row['detail']}", flush=True)

    if args.preflight_only:
        print("", flush=True)
        print("Preflight-only: no probabilities were generated, test labels were not used for class validation or performance evaluation,", flush=True)
        print("and no output files were written.", flush=True)
        return 0

    # From this point onward this is the single formal locked validation.
    output_dir.mkdir(parents=True, exist_ok=True)

    model = bundle["sklearn_model"]
    class_index = int(np.where(np.asarray(model.classes_) == 1)[0][0])
    probability = model.predict_proba(X_test)[:, class_index]
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Locked model produced invalid test probabilities.")
    record_check(checks, "locked test probabilities generated before label evaluation")

    # Test labels are accessed only after probabilities exist.
    labels_upper = test[args.label_col].astype(str).str.upper()
    allowed = {args.negative_label.upper(), args.positive_label.upper()}
    unexpected = sorted(set(labels_upper) - allowed)
    if unexpected:
        raise ValueError(f"Unexpected independent-test labels: {unexpected}.")
    if labels_upper.nunique() != 2:
        raise ValueError("Independent test cohort must contain both RA and ILD.")
    y = (labels_upper == args.positive_label.upper()).astype(int).to_numpy()
    prediction = (probability >= args.locked_threshold).astype(int)

    metrics = calculate_metrics(y, probability, args.locked_threshold)
    metrics_table = pd.DataFrame(
        [
            {
                "analysis_view": ANALYSIS_VIEW,
                "cv_unit": CV_UNIT,
                "model": MODEL_NAME,
                "alpha": args.locked_alpha,
                "lambda": args.locked_lambda,
                **metrics,
                "parameter_selection_on_test": False,
                "feature_selection_on_test": False,
                "threshold_selection_on_test": False,
                "model_refit_on_test": False,
            }
        ]
    )
    ci_table = stratified_bootstrap_ci(
        y,
        probability,
        args.locked_threshold,
        args.bootstrap_reps,
        args.bootstrap_seed,
    )

    predictions = pd.DataFrame(
        {
            "sample_id": test[args.sample_id_col].astype(str).to_numpy(),
            "patient": test[args.sample_id_col].astype(str).to_numpy(),
            "trb_libraryid": test[args.trb_library_id_col].astype(str).to_numpy(),
            "igh_libraryid": test[args.igh_library_id_col].astype(str).to_numpy(),
            "true_cohort": test[args.label_col].astype(str).to_numpy(),
            "true_label": y,
            "probability_ILD": probability,
            "locked_threshold": args.locked_threshold,
            "predicted_label": prediction,
            "predicted_cohort": np.where(
                prediction == 1, args.positive_label, args.negative_label
            ),
            "correct": prediction == y,
        }
    )
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
    pd.DataFrame(checks).to_csv(paths["integrity"], index=False)

    runtime = time.time() - started
    configuration: MutableMapping[str, object] = {
        "script_version": SCRIPT_VERSION,
        "analysis_view": ANALYSIS_VIEW,
        "cv_unit": CV_UNIT,
        "model": MODEL_NAME,
        "analysis_type": "single_locked_independent_test_validation",
        "locked_parameters": {
            "alpha": args.locked_alpha,
            "lambda": args.locked_lambda,
            "threshold": args.locked_threshold,
        },
        "model_refit_on_test": False,
        "parameter_selection_on_test": False,
        "feature_selection_on_test": False,
        "threshold_selection_on_test": False,
        "test_labels_used_for_prediction_features": False,
        "test_labels_used_for_evaluation_only": True,
        "input_files": {key: str(path) for key, path in input_paths.items()},
        "input_sha256": {key: sha256sum(path) for key, path in input_paths.items()},
        "step07_script_version": bundle.get("script_version"),
        "step07_model_fit_converged": bool(bundle.get("fit_converged")),
        "step07_model_iterations": int(bundle.get("iterations_used", -1)),
        "model_feature_count": len(feature_names),
        "model_feature_names": feature_names,
        "training_sample_count": len(bundle.get("training_sample_ids", [])),
        "test_sample_count": len(test),
        "test_class_counts": {
            args.negative_label: int(metrics["n_RA"]),
            args.positive_label: int(metrics["n_ILD"]),
        },
        "public_reference_mask_counts": mask_counts,
        "test_reference_audits": {
            receptor: {
                "mode": definitions[receptor].get("mode"),
                "test_labels_used_to_build_reference": definitions[receptor].get(
                    "test_labels_used_to_build_reference"
                ),
                "reference_file_sha256_verified": definitions[receptor].get(
                    "reference_file_sha256_verified"
                ),
            }
            for receptor in RECEPTORS
        },
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

    # COMPLETE marker is intentionally written last.
    marker = {
        "status": "COMPLETE",
        "analysis_type": "single_locked_independent_test_validation",
        "analysis_view": ANALYSIS_VIEW,
        "cv_unit": CV_UNIT,
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
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n[Independent test metrics]", flush=True)
    print(metrics_table.to_string(index=False), flush=True)
    print("\n[95% stratified bootstrap confidence intervals]", flush=True)
    print(ci_table.to_string(index=False), flush=True)
    print("\n[Output files]", flush=True)
    for key, path in paths.items():
        print(f"- {key}: {path}", flush=True)
    print(f"Runtime: {runtime:.2f}s", flush=True)
    print("Validation status: COMPLETE — this locked test must not be rerun.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
