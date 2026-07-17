#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Final paired TRB+IGH M2 tuning, threshold locking, and full-training fit.

Model
-----
M2_static_trb_igh_public:
    age + sex + static TRB + static IGH
    + dynamic TRB reference-public + dynamic IGH reference-public

Leakage boundary
----------------
* Uses only the paired training cohort and frozen repeated outer-fold assignments.
* Within each CV fold, training patients receive exact receptor-specific LOO
  public features; validation patients use references built only from that
  fold's training patients.
* After candidate selection, each training patient receives final LOO public
  features based on all other training patients.
* The independent test cohort is never read.

The training-patient count is inferred from the paired base matrix by default.
Use --expected-samples N to enforce a specific count; 0 means infer and validate
against the frozen assignments.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
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

DEFAULT_WORKER = ROOT / "set/scripts/run_05_single_outer_trial_loo_TRB_IGH_v1_0_0.py"
DEFAULT_BASE = (
    ROOT / "set/train/result/04_final_feature_matrix/04_train_base_feature_matrix.csv"
)
DEFAULT_MANIFEST = (
    ROOT / "set/train/result/04_final_feature_matrix/04_feature_manifest.csv"
)
DEFAULT_ASSIGNMENTS = (
    ROOT / "set/train/result/05_modeling/cv_splits/05_outer_fold_assignments.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "set/train/result/07_final_M2_model"

TRB_ROOT = ROOT / "set/train/TRB/result"
IGH_ROOT = ROOT / "set/train/IGH/result"

DEFAULT_CANDIDATE_PAIRS = "0.9:10,0.5:30"
DEFAULT_L1_GRID = "0.1,0.5,0.9,1.0"
DEFAULT_LAMBDA_GRID = "0.01,0.03,0.1,0.3,1,3,10,30,100"


# -----------------------------------------------------------------------------
# CLI and generic utilities
# -----------------------------------------------------------------------------

def parse_float_list(text: str) -> List[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one numeric value is required.")
    if any(not np.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("Numeric values must be finite.")
    return list(dict.fromkeys(values))


def parse_int_list(text: str) -> List[int]:
    values: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start, end = int(left), int(right)
            if end < start:
                raise argparse.ArgumentTypeError(f"Invalid range: {token}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return sorted(set(values))


def parse_candidate_pairs(text: str) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise argparse.ArgumentTypeError(
                "Candidate pairs must use alpha:lambda format."
            )
        alpha_text, lambda_text = token.split(":", 1)
        alpha = float(alpha_text)
        lambda_value = float(lambda_text)
        if not np.isfinite(alpha) or not np.isfinite(lambda_value):
            raise argparse.ArgumentTypeError("Candidate values must be finite.")
        if not (0 < alpha <= 1):
            raise argparse.ArgumentTypeError("Alpha must be in (0, 1].")
        if lambda_value <= 0:
            raise argparse.ArgumentTypeError("Lambda must be > 0.")
        pairs.append((alpha, lambda_value))
    if not pairs:
        raise argparse.ArgumentTypeError("At least one candidate pair is required.")
    return list(dict.fromkeys(pairs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune candidate Elastic Net pairs and fit the final paired TRB+IGH "
            "M2 model without reading the independent test cohort."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--loo-worker-script", default=str(DEFAULT_WORKER))
    parser.add_argument("--base-matrix", default=str(DEFAULT_BASE))
    parser.add_argument("--feature-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--cv-assignments", default=str(DEFAULT_ASSIGNMENTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    parser.add_argument(
        "--trb-public-catalog",
        default=str(TRB_ROOT / "03_public_features/03_public_aa_catalog.csv.gz"),
    )
    parser.add_argument(
        "--igh-public-catalog",
        default=str(IGH_ROOT / "03_public_features/03_public_aa_catalog.csv.gz"),
    )
    parser.add_argument(
        "--trb-reference-definition",
        default=str(TRB_ROOT / "03_public_features/03_reference_definition.json"),
    )
    parser.add_argument(
        "--igh-reference-definition",
        default=str(IGH_ROOT / "03_public_features/03_reference_definition.json"),
    )
    parser.add_argument(
        "--trb-clone-table-dir",
        default=str(TRB_ROOT / "01_AA_clone_table"),
    )
    parser.add_argument(
        "--igh-clone-table-dir",
        default=str(IGH_ROOT / "01_AA_clone_table"),
    )
    parser.add_argument(
        "--trb-cache-dir",
        default=str(ROOT / "set/train/result/05_modeling/cache_loo/TRB"),
    )
    parser.add_argument(
        "--igh-cache-dir",
        default=str(ROOT / "set/train/result/05_modeling/cache_loo/IGH"),
    )

    parser.add_argument("--repeats", type=parse_int_list, default=parse_int_list("1-20"))
    parser.add_argument("--folds", type=parse_int_list, default=parse_int_list("1-5"))

    parser.add_argument(
        "--tuning-mode",
        choices=("stable_pairs", "full_grid"),
        default="stable_pairs",
    )
    parser.add_argument(
        "--candidate-pairs",
        type=parse_candidate_pairs,
        default=parse_candidate_pairs(DEFAULT_CANDIDATE_PAIRS),
        help=(
            "Pre-specified candidate pairs from step 06. The paired-project "
            "defaults are alpha=0.9/lambda=10 and alpha=0.5/lambda=30."
        ),
    )
    parser.add_argument(
        "--l1-ratios",
        type=parse_float_list,
        default=parse_float_list(DEFAULT_L1_GRID),
    )
    parser.add_argument(
        "--lambdas",
        type=parse_float_list,
        default=parse_float_list(DEFAULT_LAMBDA_GRID),
    )
    parser.add_argument(
        "--selection-rule",
        choices=("one_se", "best_mean"),
        default="one_se",
    )
    parser.add_argument(
        "--threshold-rule",
        choices=("youden", "max_f1", "sensitivity_target"),
        default="youden",
    )
    parser.add_argument("--target-sensitivity", type=float, default=0.80)

    parser.add_argument(
        "--public-feature-set",
        default="raw_bilateral",
        choices=("raw_bilateral", "log_ratio", "all18"),
    )
    parser.add_argument("--sample-id-col", default="sample_id")
    parser.add_argument("--label-col", default="cohort")
    parser.add_argument("--positive-label", default="ILD")
    parser.add_argument("--age-col", default="age")
    parser.add_argument("--sex-col", default="sex")
    parser.add_argument("--material-col", default="material")
    parser.add_argument("--trb-library-id-col", default="trb_libraryid")
    parser.add_argument("--igh-library-id-col", default="igh_libraryid")
    parser.add_argument("--trb-aa-clone-number-col", default="trb_aa_clone_number")
    parser.add_argument("--igh-aa-clone-number-col", default="igh_aa_clone_number")
    parser.add_argument(
        "--trb-clone-file-pattern",
        default="{library_id}_TRB_CDR3_AA_clone_table.csv",
    )
    parser.add_argument(
        "--igh-clone-file-pattern",
        default="{library_id}_IGH-without-DJ_CDR3_AA_clone_table.csv",
    )
    parser.add_argument("--sequence-col", default="cdr3_aa")
    parser.add_argument("--frequency-col", default="frequency_norm")

    parser.add_argument("--global-min-prevalence", type=float, default=0.02)
    parser.add_argument("--global-min-count", type=int, default=3)
    parser.add_argument("--group-min-prevalence", type=float, default=0.05)
    parser.add_argument("--group-min-count", type=int, default=3)
    parser.add_argument("--specific-prevalence-delta", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=1e-8)

    parser.add_argument(
        "--class-weight", choices=("balanced", "none"), default="balanced"
    )
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--zero-sd-tolerance", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=20260711)

    parser.add_argument(
        "--expected-samples",
        type=int,
        default=0,
        help="Expected paired training patients; 0 infers from the base matrix.",
    )

    parser.add_argument("--expected-trb-catalog-size", type=int, default=965027)
    parser.add_argument("--expected-trb-global-size", type=int, default=385472)
    parser.add_argument("--expected-trb-ra-specific-size", type=int, default=44662)
    parser.add_argument("--expected-trb-ild-specific-size", type=int, default=27563)
    parser.add_argument("--expected-trb-shared-size", type=int, default=36119)

    parser.add_argument("--expected-igh-catalog-size", type=int, default=48594)
    parser.add_argument("--expected-igh-global-size", type=int, default=6786)
    parser.add_argument("--expected-igh-ra-specific-size", type=int, default=320)
    parser.add_argument("--expected-igh-ild-specific-size", type=int, default=571)
    parser.add_argument("--expected-igh-shared-size", type=int, default=274)

    parser.add_argument("--skip-full-reference-validation", action="store_true")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.repeats or min(args.repeats) < 1:
        parser.error("All repeat IDs must be >=1.")
    if not args.folds or min(args.folds) < 1:
        parser.error("All fold IDs must be >=1.")
    if not (0 < args.target_sensitivity <= 1):
        parser.error("--target-sensitivity must be in (0, 1].")
    if args.selection_rule == "one_se" and len(args.repeats) < 2:
        parser.error("--selection-rule one_se requires at least two repeats.")
    if args.expected_samples < 0:
        parser.error("--expected-samples must be >=0.")
    for value in args.l1_ratios:
        if not (0 < value <= 1):
            parser.error("Every l1 ratio must be in (0, 1].")
    for value in args.lambdas:
        if value <= 0:
            parser.error("Every lambda must be >0.")
    if args.epsilon <= 0:
        parser.error("--epsilon must be >0.")
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


def derive_seed(base_seed: int, *parts: int) -> int:
    text = ":".join(str(value) for value in (base_seed, *parts))
    return int.from_bytes(
        hashlib.sha256(text.encode("utf-8")).digest()[:8], "little"
    ) % (2**32 - 1)


def load_worker(path: Path) -> ModuleType:
    ensure_file(path, "paired LOO worker script")
    spec = importlib.util.spec_from_file_location("paired_loo_worker", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import paired LOO worker: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    required = [
        "PUBLIC_FEATURE_SETS",
        "receptor_specs",
        "static_joint_features_from_manifest",
        "load_and_validate_reference_definition",
        "build_or_load_sparse_cache",
        "validate_full_reference_for_receptor",
        "model_specifications",
        "build_joint_public_for_partition",
        "build_joint_external_public",
        "assemble_model_dataframe",
        "build_design",
        "fit_elastic_net",
        "build_public_reference",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise ImportError(f"Paired worker missing required objects: {missing}")

    worker_view = str(getattr(module, "SCRIPT_VERSION", ""))
    model_labels = getattr(module, "MODEL_LABELS", {})
    if MODEL_NAME not in model_labels:
        raise ImportError(
            f"Paired worker does not define {MODEL_NAME}; SCRIPT_VERSION={worker_view}"
        )
    return module


def output_paths(output_dir: Path) -> Dict[str, Path]:
    return {
        "cv_predictions": output_dir / "07_repeated_cv_candidate_predictions.csv.gz",
        "cv_fold_fits": output_dir / "07_repeated_cv_fold_fit_summary.csv",
        "cv_repeat_metrics": output_dir / "07_repeated_cv_repeat_metrics.csv",
        "candidate_summary": output_dir / "07_repeated_cv_candidate_summary.csv",
        "selected_oof": output_dir / "07_selected_candidate_sample_mean_OOF.csv",
        "threshold_candidates": output_dir / "07_threshold_candidates.csv",
        "repeat_thresholds": output_dir / "07_selected_candidate_repeat_thresholds.csv",
        "final_public": output_dir / "07_final_training_public_features_LOO.csv",
        "preprocessing": output_dir / "07_final_preprocessing.csv",
        "coefficients": output_dir / "07_final_model_coefficients.csv",
        "fitted_predictions": output_dir / "07_final_training_fitted_predictions.csv",
        "reference_masks": output_dir / "07_full_training_public_reference_masks.npz",
        "reference_summary": output_dir / "07_full_training_public_reference_summary.csv",
        "reference_log": output_dir / "07_public_reference_build_log.csv.gz",
        "loo_log": output_dir / "07_public_LOO_assignments.csv.gz",
        "model_bundle": output_dir / "07_final_M2_model.joblib",
        "configuration": output_dir / "07_final_M2_configuration.json",
        "summary": output_dir / "07_final_M2_summary.md",
    }


def enforce_overwrite(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Outputs already exist. Add --overwrite to replace them:\n"
            + "\n".join(f"  - {path}" for path in existing)
        )


def candidate_pairs(args: argparse.Namespace) -> List[Tuple[float, float]]:
    if args.tuning_mode == "stable_pairs":
        return list(args.candidate_pairs)
    return list(itertools.product(args.l1_ratios, args.lambdas))


# -----------------------------------------------------------------------------
# Input integrity
# -----------------------------------------------------------------------------

def normalize_assignments(
    assignments: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    required = {
        "outer_repeat",
        "outer_fold",
        args.sample_id_col,
        args.label_col,
    }
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise ValueError(f"CV assignments missing columns: {missing}")

    out = assignments.copy()
    out["outer_repeat"] = pd.to_numeric(out["outer_repeat"], errors="raise").astype(int)
    out["outer_fold"] = pd.to_numeric(out["outer_fold"], errors="raise").astype(int)
    out[args.sample_id_col] = out[args.sample_id_col].astype(str).str.strip()
    out[args.label_col] = out[args.label_col].astype(str).str.upper()
    return out


def validate_and_read_inputs(
    args: argparse.Namespace,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Path],
    int,
]:
    paths = {
        "base": Path(args.base_matrix).expanduser().resolve(),
        "manifest": Path(args.feature_manifest).expanduser().resolve(),
        "assignments": Path(args.cv_assignments).expanduser().resolve(),
        "worker": Path(args.loo_worker_script).expanduser().resolve(),
        "trb_catalog": Path(args.trb_public_catalog).expanduser().resolve(),
        "igh_catalog": Path(args.igh_public_catalog).expanduser().resolve(),
        "trb_reference_definition": Path(
            args.trb_reference_definition
        ).expanduser().resolve(),
        "igh_reference_definition": Path(
            args.igh_reference_definition
        ).expanduser().resolve(),
    }
    for key, path in paths.items():
        ensure_file(path, key)

    for label, directory in [
        ("TRB clone table directory", Path(args.trb_clone_table_dir)),
        ("IGH clone table directory", Path(args.igh_clone_table_dir)),
    ]:
        directory = directory.expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"{label} not found: {directory}")

    base = pd.read_csv(paths["base"])
    manifest = pd.read_csv(paths["manifest"])
    assignments = normalize_assignments(pd.read_csv(paths["assignments"]), args)

    required_base = {
        args.sample_id_col,
        args.label_col,
        args.age_col,
        args.sex_col,
        args.material_col,
        args.trb_library_id_col,
        args.igh_library_id_col,
        args.trb_aa_clone_number_col,
        args.igh_aa_clone_number_col,
    }
    missing = sorted(required_base - set(base.columns))
    if missing:
        raise ValueError(f"Paired base matrix missing columns: {missing}")
    if base.isna().any().any():
        bad = base.isna().sum()
        raise ValueError(
            "Paired base matrix contains missing values: "
            + str(bad[bad > 0].to_dict())
        )

    for column in [
        args.sample_id_col,
        args.trb_library_id_col,
        args.igh_library_id_col,
    ]:
        base[column] = base[column].astype(str).str.strip()
        if base[column].eq("").any():
            raise ValueError(f"Empty values in paired base column {column}.")
        if base[column].duplicated().any():
            raise ValueError(f"Duplicated values in paired base column {column}.")

    base[args.label_col] = base[args.label_col].astype(str).str.upper()
    if set(base[args.label_col]) != {"RA", "ILD"}:
        raise ValueError(
            f"Paired base labels must be RA/ILD; observed {sorted(base[args.label_col].unique())}."
        )

    n_samples = len(base)
    if args.expected_samples > 0 and n_samples != args.expected_samples:
        raise ValueError(
            f"Expected {args.expected_samples} paired training patients, observed {n_samples}."
        )

    required_manifest = {
        "analysis_view",
        "receptor",
        "feature_name",
        "present_in_train_base",
        "feature_role",
        "reference_drop",
    }
    missing_manifest = sorted(required_manifest - set(manifest.columns))
    if missing_manifest:
        raise ValueError(f"Feature manifest missing columns: {missing_manifest}")
    views = set(manifest["analysis_view"].astype(str))
    if views != {ANALYSIS_VIEW}:
        raise ValueError(
            f"Expected manifest analysis_view={ANALYSIS_VIEW}; observed {sorted(views)}."
        )

    selected = assignments[
        assignments["outer_repeat"].isin(args.repeats)
        & assignments["outer_fold"].isin(args.folds)
    ].copy()
    base_ids = set(base[args.sample_id_col])
    expected_fold_set = set(args.folds)

    for repeat in args.repeats:
        repeat_df = selected[selected["outer_repeat"] == repeat].copy()
        if len(repeat_df) != n_samples:
            raise ValueError(
                f"Repeat {repeat} has {len(repeat_df)} assignment rows; expected {n_samples}."
            )
        if repeat_df[args.sample_id_col].duplicated().any():
            raise ValueError(f"Repeat {repeat} contains duplicated patient IDs.")
        if set(repeat_df[args.sample_id_col]) != base_ids:
            raise ValueError(f"Repeat {repeat} patient IDs differ from the paired base matrix.")
        if set(repeat_df["outer_fold"]) != expected_fold_set:
            raise ValueError(
                f"Repeat {repeat} fold set mismatch: {sorted(repeat_df['outer_fold'].unique())}"
            )
        if set(repeat_df[args.label_col]) != {"RA", "ILD"}:
            raise ValueError(f"Repeat {repeat} does not contain both outcome classes.")

        mapping_cols = [
            column
            for column in [args.trb_library_id_col, args.igh_library_id_col]
            if column in repeat_df.columns
        ]
        for column in mapping_cols:
            check = base[[args.sample_id_col, column]].merge(
                repeat_df[[args.sample_id_col, column]],
                on=args.sample_id_col,
                how="inner",
                validate="one_to_one",
                suffixes=("_base", "_split"),
            )
            if not (
                check[f"{column}_base"].astype(str).to_numpy()
                == check[f"{column}_split"].astype(str).to_numpy()
            ).all():
                raise ValueError(
                    f"Frozen assignment mapping differs from base for {column}, repeat={repeat}."
                )

        label_check = base[[args.sample_id_col, args.label_col]].merge(
            repeat_df[[args.sample_id_col, args.label_col]],
            on=args.sample_id_col,
            how="inner",
            validate="one_to_one",
            suffixes=("_base", "_split"),
        )
        if not (
            label_check[f"{args.label_col}_base"].astype(str).str.upper().to_numpy()
            == label_check[f"{args.label_col}_split"].astype(str).str.upper().to_numpy()
        ).all():
            raise ValueError(f"Frozen labels differ from base for repeat={repeat}.")

    # Each repeat must partition the full cohort exactly once across requested folds.
    repeat_counts = selected.groupby(["outer_repeat", args.sample_id_col]).size()
    if not (repeat_counts == 1).all():
        raise ValueError("Each patient must have exactly one outer fold per repeat.")

    base = base.set_index(args.sample_id_col, drop=False)
    selected = selected.sort_values(
        ["outer_repeat", "outer_fold", args.sample_id_col]
    ).reset_index(drop=True)
    return base, manifest, selected, paths, n_samples


# -----------------------------------------------------------------------------
# Metrics, candidate selection, and thresholds
# -----------------------------------------------------------------------------

def safe_auc(y: np.ndarray, probability: np.ndarray) -> float:
    return float(roc_auc_score(y, probability))


def binary_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "roc_auc": safe_auc(y, probability),
        "pr_auc": float(average_precision_score(y, probability)),
        "threshold": float(threshold),
        "sensitivity_recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def threshold_grid(probability: np.ndarray) -> np.ndarray:
    values = np.unique(np.asarray(probability, dtype=float))
    candidates = np.unique(np.concatenate(([0.0], values, [0.5, 1.0])))
    return candidates[(candidates >= 0) & (candidates <= 1)]


def choose_threshold_youden(y: np.ndarray, probability: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, probability)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    j = tpr[finite] - fpr[finite]
    candidates = thresholds[finite][j == np.max(j)]
    return float(candidates[np.argmin(np.abs(candidates - 0.5))])


def choose_threshold_max_f1(y: np.ndarray, probability: np.ndarray) -> float:
    rows = []
    for threshold in threshold_grid(probability):
        metrics = binary_metrics(y, probability, float(threshold))
        rows.append((metrics["f1"], metrics["specificity"], threshold))
    best = sorted(rows, key=lambda item: (-item[0], -item[1], abs(item[2] - 0.5)))[0]
    return float(best[2])


def choose_threshold_sensitivity_target(
    y: np.ndarray,
    probability: np.ndarray,
    target: float,
) -> float:
    rows = []
    for threshold in threshold_grid(probability):
        metrics = binary_metrics(y, probability, float(threshold))
        if metrics["sensitivity_recall"] + 1e-12 >= target:
            rows.append((metrics["specificity"], metrics["precision"], threshold))
    if not rows:
        return 0.0
    best = sorted(rows, key=lambda item: (-item[0], -item[1], abs(item[2] - 0.5)))[0]
    return float(best[2])


def choose_thresholds(
    y: np.ndarray,
    probability: np.ndarray,
    target_sensitivity: float,
) -> pd.DataFrame:
    rules = {
        "youden": choose_threshold_youden(y, probability),
        "max_f1": choose_threshold_max_f1(y, probability),
        "sensitivity_target": choose_threshold_sensitivity_target(
            y, probability, target_sensitivity
        ),
    }
    rows: List[Dict[str, object]] = []
    for rule, threshold in rules.items():
        row: Dict[str, object] = {
            "threshold_rule": rule,
            "target_sensitivity": (
                target_sensitivity if rule == "sensitivity_target" else np.nan
            ),
        }
        row.update(binary_metrics(y, probability, threshold))
        rows.append(row)
    return pd.DataFrame(rows)


def repeat_metrics_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    group_cols = ["l1_ratio_alpha", "lambda", "outer_repeat"]
    for keys, group in predictions.groupby(group_cols, sort=True):
        alpha, lambda_value, repeat = keys
        if group["sample_id"].duplicated().any():
            raise ValueError(
                f"Duplicate OOF patient for alpha={alpha}, lambda={lambda_value}, repeat={repeat}."
            )
        y = group["true_label"].to_numpy(int)
        probability = group["probability_ILD"].to_numpy(float)
        rows.append(
            {
                "l1_ratio_alpha": float(alpha),
                "lambda": float(lambda_value),
                "outer_repeat": int(repeat),
                "n_samples": len(group),
                "roc_auc": safe_auc(y, probability),
                "pr_auc": float(average_precision_score(y, probability)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["l1_ratio_alpha", "lambda", "outer_repeat"]
    ).reset_index(drop=True)


def summarize_candidates(
    repeat_metrics: pd.DataFrame,
    fold_fits: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (alpha, lambda_value), repeat_group in repeat_metrics.groupby(
        ["l1_ratio_alpha", "lambda"], sort=True
    ):
        fold_group = fold_fits[
            (fold_fits["l1_ratio_alpha"] == alpha)
            & (fold_fits["lambda"] == lambda_value)
        ]
        pred_group = predictions[
            (predictions["l1_ratio_alpha"] == alpha)
            & (predictions["lambda"] == lambda_value)
        ]
        sample_mean = (
            pred_group.groupby("sample_id", as_index=False)
            .agg(
                true_label=("true_label", "first"),
                probability_ILD=("probability_ILD", "mean"),
                n_repeated_predictions=("probability_ILD", "size"),
            )
        )
        auc_values = repeat_group["roc_auc"].to_numpy(float)
        auc_sd = float(np.std(auc_values, ddof=1)) if len(auc_values) > 1 else 0.0
        rows.append(
            {
                "model": MODEL_NAME,
                "l1_ratio_alpha": float(alpha),
                "lambda": float(lambda_value),
                "n_repeats": len(repeat_group),
                "mean_repeat_roc_auc": float(np.mean(auc_values)),
                "sd_repeat_roc_auc": auc_sd,
                "se_repeat_roc_auc": float(auc_sd / math.sqrt(len(auc_values))),
                "median_repeat_roc_auc": float(np.median(auc_values)),
                "mean_repeat_pr_auc": float(repeat_group["pr_auc"].mean()),
                "sd_repeat_pr_auc": float(repeat_group["pr_auc"].std(ddof=1)),
                "sample_mean_oof_roc_auc": safe_auc(
                    sample_mean["true_label"].to_numpy(int),
                    sample_mean["probability_ILD"].to_numpy(float),
                ),
                "sample_mean_oof_pr_auc": float(
                    average_precision_score(
                        sample_mean["true_label"], sample_mean["probability_ILD"]
                    )
                ),
                "mean_nonzero_coefficients": float(
                    fold_group["n_nonzero_coefficients"].mean()
                ),
                "median_nonzero_coefficients": float(
                    fold_group["n_nonzero_coefficients"].median()
                ),
                "all_fits_converged": bool(fold_group["fit_converged"].all()),
                "max_iterations_used": int(fold_group["iterations_used"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_repeat_roc_auc", "mean_repeat_pr_auc"], ascending=[False, False]
    ).reset_index(drop=True)


def select_candidate(
    summary: pd.DataFrame,
    rule: str,
) -> Tuple[pd.Series, Dict[str, object]]:
    converged = summary[summary["all_fits_converged"]].copy()
    if converged.empty:
        raise RuntimeError("No candidate converged in every repeated-CV fold fit.")

    ranked = converged.sort_values(
        [
            "mean_repeat_roc_auc",
            "mean_repeat_pr_auc",
            "mean_nonzero_coefficients",
            "lambda",
            "l1_ratio_alpha",
        ],
        ascending=[False, False, True, False, False],
    ).reset_index(drop=True)
    best = ranked.iloc[0]

    if rule == "best_mean":
        return best, {
            "selection_rule": "best_mean",
            "best_mean_roc_auc": float(best["mean_repeat_roc_auc"]),
            "one_se_cutoff": None,
            "n_eligible": 1,
        }

    cutoff = float(best["mean_repeat_roc_auc"] - best["se_repeat_roc_auc"])
    eligible = converged[
        converged["mean_repeat_roc_auc"] >= cutoff - 1e-12
    ].copy()
    eligible = eligible.sort_values(
        [
            "mean_nonzero_coefficients",
            "mean_repeat_pr_auc",
            "lambda",
            "l1_ratio_alpha",
            "mean_repeat_roc_auc",
        ],
        ascending=[True, False, False, False, False],
    )
    selected = eligible.iloc[0]
    return selected, {
        "selection_rule": "one_se",
        "best_mean_roc_auc": float(best["mean_repeat_roc_auc"]),
        "best_mean_standard_error": float(best["se_repeat_roc_auc"]),
        "one_se_cutoff": cutoff,
        "n_eligible": int(len(eligible)),
    }


# -----------------------------------------------------------------------------
# Bundle helpers
# -----------------------------------------------------------------------------

def feature_source(feature_name: str) -> str:
    if feature_name == "__INTERCEPT__":
        return "intercept"
    if feature_name.startswith("trb_"):
        return "TRB"
    if feature_name.startswith("igh_"):
        return "IGH"
    if feature_name.startswith("age") or feature_name.startswith("sex__"):
        return "clinical"
    if feature_name.startswith("material__"):
        return "material"
    return "other"


def build_preprocessor_metadata(
    train_df: pd.DataFrame,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    preprocessing: pd.DataFrame,
    feature_names: Sequence[str],
) -> Dict[str, object]:
    categorical_levels = {
        column: sorted(train_df[column].astype(str).unique().tolist())
        for column in categorical_cols
    }
    categorical_reference = {
        column: levels[0] for column, levels in categorical_levels.items()
    }
    return {
        "numeric_columns": list(numeric_cols),
        "categorical_columns": list(categorical_cols),
        "categorical_levels": categorical_levels,
        "categorical_reference": categorical_reference,
        "final_feature_names": list(feature_names),
        "preprocessing_rows": preprocessing.to_dict(orient="records"),
    }


def save_joint_reference_masks(
    path: Path,
    receptor_data: Mapping[str, Mapping[str, object]],
    labels: np.ndarray,
    args: argparse.Namespace,
    worker: ModuleType,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    all_rows = np.arange(len(labels), dtype=int)
    payload: Dict[str, np.ndarray] = {
        "analysis_view": np.asarray(ANALYSIS_VIEW),
        "cv_unit": np.asarray(CV_UNIT),
    }
    summaries: List[Dict[str, object]] = []
    metadata: Dict[str, Dict[str, object]] = {}

    for receptor in ("TRB", "IGH"):
        data = receptor_data[receptor]
        spec = data["spec"]
        reference = worker.build_public_reference(
            presence=data["presence"],
            reference_rows=all_rows,
            labels=labels,
            args=args,
        )
        prefix = receptor.lower()
        catalog_path = Path(spec.public_catalog)
        definition_path = Path(spec.reference_definition)
        payload.update(
            {
                f"{prefix}_n_reference": np.asarray(reference.n_reference, dtype=np.int32),
                f"{prefix}_n_RA_reference": np.asarray(reference.n_ra, dtype=np.int32),
                f"{prefix}_n_ILD_reference": np.asarray(reference.n_ild, dtype=np.int32),
                f"{prefix}_global_threshold": np.asarray(
                    reference.global_threshold, dtype=np.int32
                ),
                f"{prefix}_RA_threshold": np.asarray(reference.ra_threshold, dtype=np.int32),
                f"{prefix}_ILD_threshold": np.asarray(reference.ild_threshold, dtype=np.int32),
                f"{prefix}_public_catalog_sha256": np.asarray(sha256sum(catalog_path)),
                f"{prefix}_reference_definition_sha256": np.asarray(
                    sha256sum(definition_path)
                ),
                f"{prefix}_global_mask": reference.global_mask.astype(np.uint8),
                f"{prefix}_RA_specific_mask": reference.ra_specific_mask.astype(np.uint8),
                f"{prefix}_ILD_specific_mask": reference.ild_specific_mask.astype(np.uint8),
                f"{prefix}_shared_mask": reference.shared_mask.astype(np.uint8),
            }
        )
        summary = {
            "receptor": receptor,
            "n_reference": reference.n_reference,
            "n_RA_reference": reference.n_ra,
            "n_ILD_reference": reference.n_ild,
            "global_threshold": reference.global_threshold,
            "RA_threshold": reference.ra_threshold,
            "ILD_threshold": reference.ild_threshold,
            "all_ref_public_size": int(reference.global_mask.sum()),
            "RA_specific_ref_size": int(reference.ra_specific_mask.sum()),
            "ILD_specific_ref_size": int(reference.ild_specific_mask.sum()),
            "shared_ref_size": int(reference.shared_mask.sum()),
            "public_catalog_sha256": sha256sum(catalog_path),
            "reference_definition_sha256": sha256sum(definition_path),
        }
        summaries.append(summary)
        metadata[receptor] = summary.copy()

    np.savez_compressed(path, **payload)
    return pd.DataFrame(summaries), metadata


def write_summary(
    path: Path,
    args: argparse.Namespace,
    n_samples: int,
    n_ra: int,
    n_ild: int,
    candidates: Sequence[Tuple[float, float]],
    candidate_summary: pd.DataFrame,
    selected: pd.Series,
    selection_details: Mapping[str, object],
    threshold_table: pd.DataFrame,
    selected_threshold: float,
    final_converged: bool,
    final_iterations: int,
    final_nonzero: int,
    full_reference_counts: Optional[Mapping[str, Mapping[str, int]]],
    runtime_seconds: float,
) -> None:
    chosen_threshold_row = threshold_table[
        threshold_table["threshold_rule"] == args.threshold_rule
    ].iloc[0]
    lines = [
        "# 07 Final Paired TRB+IGH M2 Training Summary",
        "",
        "## Scope and leakage boundary",
        "",
        f"- Script version: `{SCRIPT_VERSION}`",
        f"- Analysis view: `{ANALYSIS_VIEW}`",
        f"- CV unit: `{CV_UNIT}`",
        f"- Training patients: **{n_samples}** (RA={n_ra}, ILD={n_ild})",
        "- Independent test cohort: **NOT READ**.",
        "- CV-training public features used exact receptor-specific leave-one-out references.",
        "- CV-validation public features used only the corresponding fold-training partition.",
        f"- Final fitting public features used the other **{n_samples - 1}** training patients for each patient.",
        "",
        "## Candidate tuning",
        "",
        f"- Model: `{MODEL_NAME}`",
        f"- Tuning mode: `{args.tuning_mode}`",
        f"- Selection rule: `{args.selection_rule}`",
        f"- Repeats × folds: `{len(args.repeats)} × {len(args.folds)}`",
        f"- Candidate pairs: `{list(candidates)}`",
        f"- Selected alpha: **{float(selected['l1_ratio_alpha']):.4g}**",
        f"- Selected lambda: **{float(selected['lambda']):.4g}**",
        f"- Mean repeat ROC-AUC: **{float(selected['mean_repeat_roc_auc']):.4f}**",
        f"- Mean repeat PR-AUC: **{float(selected['mean_repeat_pr_auc']):.4f}**",
        f"- Mean nonzero coefficients: **{float(selected['mean_nonzero_coefficients']):.2f}**",
        "",
        "## Locked threshold",
        "",
        f"- Threshold rule: `{args.threshold_rule}`",
        f"- Locked threshold: **{selected_threshold:.6f}**",
        f"- Sample-mean OOF sensitivity: **{float(chosen_threshold_row['sensitivity_recall']):.4f}**",
        f"- Sample-mean OOF specificity: **{float(chosen_threshold_row['specificity']):.4f}**",
        f"- Sample-mean OOF F1: **{float(chosen_threshold_row['f1']):.4f}**",
        "- The step-06 threshold median near 0.4941 was descriptive; this locked threshold was re-estimated from the selected candidate's repeated OOF probabilities.",
        "",
        "## Final all-training fit",
        "",
        f"- Final fit converged: `{final_converged}`",
        f"- Iterations used: `{final_iterations}`",
        f"- Nonzero coefficients: `{final_nonzero}`",
        "- Full-training fitted metrics are apparent/in-sample and are not an unbiased performance estimate.",
        "",
        "## Full-training public references",
        "",
    ]
    if full_reference_counts is None:
        lines.append("- Step-03 full-reference validation was skipped.")
    else:
        for receptor in ("TRB", "IGH"):
            lines.append(f"### {receptor}")
            for key, value in full_reference_counts[receptor].items():
                lines.append(f"- {key}: `{int(value):,}`")
            lines.append("")

    lines.extend(["## Candidate summary", "", "```text"])
    lines.extend(candidate_summary.to_string(index=False).splitlines())
    lines.extend(["```", "", "## Selection details", ""])
    for key, value in selection_details.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", f"- Runtime: `{runtime_seconds:.1f}` seconds", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    started = time.time()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)
    if not args.preflight_only:
        enforce_overwrite(paths.values(), args.overwrite)

    base, manifest, assignments, input_paths, n_samples = validate_and_read_inputs(args)
    worker = load_worker(input_paths["worker"])
    specs = worker.receptor_specs(args)

    # Ensure worker-resolved paths agree with explicitly validated paths.
    expected_spec_paths = {
        "TRB": (
            input_paths["trb_catalog"],
            input_paths["trb_reference_definition"],
        ),
        "IGH": (
            input_paths["igh_catalog"],
            input_paths["igh_reference_definition"],
        ),
    }
    reference_definitions: Dict[str, Dict[str, object]] = {}
    reference_definition_paths: Dict[str, Path] = {}
    for receptor in ("TRB", "IGH"):
        spec = specs[receptor]
        catalog_expected, definition_expected = expected_spec_paths[receptor]
        if Path(spec.public_catalog).resolve() != catalog_expected:
            raise RuntimeError(f"{receptor} worker catalog path differs from validated input.")
        path, definition = worker.load_and_validate_reference_definition(args, spec)
        if Path(path).resolve() != definition_expected:
            raise RuntimeError(
                f"{receptor} worker reference-definition path differs from validated input."
            )
        reference_definition_paths[receptor] = Path(path).resolve()
        reference_definitions[receptor] = definition

    static_trb, static_igh = worker.static_joint_features_from_manifest(manifest, base)
    static_features = [*static_trb, *static_igh]
    selected_public_base = list(worker.PUBLIC_FEATURE_SETS[args.public_feature_set])
    public_features = [
        *[f"trb_{feature}" for feature in selected_public_base],
        *[f"igh_{feature}" for feature in selected_public_base],
    ]
    model_specs = worker.model_specifications(static_features, public_features, args)
    if MODEL_NAME not in model_specs:
        raise ValueError(
            f"Worker model specifications do not contain {MODEL_NAME}: {sorted(model_specs)}"
        )
    m2_spec = model_specs[MODEL_NAME]
    candidates = candidate_pairs(args)
    if args.selection_rule == "one_se" and len(candidates) < 2:
        raise ValueError("one_se selection requires at least two candidate pairs.")

    n_ra = int((base[args.label_col] == "RA").sum())
    n_ild = int((base[args.label_col] == "ILD").sum())
    assignment_sha = sha256sum(input_paths["assignments"])

    print("=" * 88, flush=True)
    print("07 Paired TRB+IGH Final Full-Training M2 Tuning and Fit", flush=True)
    print("=" * 88, flush=True)
    print(f"Script version: {SCRIPT_VERSION}", flush=True)
    print(f"Analysis view: {ANALYSIS_VIEW}", flush=True)
    print(f"CV unit: {CV_UNIT}", flush=True)
    print(f"Training patients: {n_samples} (RA={n_ra}, ILD={n_ild})", flush=True)
    print(f"Static TRB predictors: {len(static_trb)}", flush=True)
    print(f"Static IGH predictors: {len(static_igh)}", flush=True)
    print(
        f"Dynamic public predictors: {len(selected_public_base)} per receptor; "
        f"{len(public_features)} joint",
        flush=True,
    )
    print(f"Frozen assignments SHA256: {assignment_sha}", flush=True)
    print(f"Repeated CV tasks: {len(args.repeats) * len(args.folds)}", flush=True)
    print(f"Candidate pairs: {candidates}", flush=True)
    print(f"Selection rule: {args.selection_rule}", flush=True)
    print(f"Threshold rule: {args.threshold_rule}", flush=True)
    print("Independent test set: NOT READ", flush=True)
    print("", flush=True)

    if args.preflight_only:
        print("[Preflight checks]", flush=True)
        print(
            f"- PASS: paired base/manifest/frozen assignments cover {n_samples} patients",
            flush=True,
        )
        print(
            f"- PASS: {len(args.repeats)} repeats x {len(args.folds)} folds are complete",
            flush=True,
        )
        print(
            f"- PASS: candidate pairs={candidates}; model={MODEL_NAME}",
            flush=True,
        )
        print(
            f"- PASS: static predictors TRB={len(static_trb)}, IGH={len(static_igh)}; "
            f"public predictors={len(public_features)}",
            flush=True,
        )
        print("Preflight-only: no cache/model/output files were written.", flush=True)
        return 0

    labels = base[args.label_col].astype(str).str.upper().to_numpy()
    sample_ids = base.index.astype(str).tolist()
    row_lookup = {sample_id: index for index, sample_id in enumerate(sample_ids)}

    receptor_data: Dict[str, Dict[str, object]] = {}
    cache_metadata: Dict[str, Dict[str, object]] = {}
    for receptor in ("TRB", "IGH"):
        spec = specs[receptor]
        presence, frequency, cache_meta = worker.build_or_load_sparse_cache(
            args, base, spec
        )
        aa_clone_numbers = pd.to_numeric(
            base[spec.aa_clone_number_col], errors="raise"
        ).to_numpy(float)
        receptor_data[receptor] = {
            "spec": spec,
            "presence": presence,
            "frequency": frequency,
            "aa_clone_numbers": aa_clone_numbers,
        }
        cache_metadata[receptor] = cache_meta

    full_reference_counts: Optional[Dict[str, Dict[str, int]]] = None
    if not args.skip_full_reference_validation:
        full_reference_counts = {}
        for receptor in ("TRB", "IGH"):
            data = receptor_data[receptor]
            counts = worker.validate_full_reference_for_receptor(
                presence=data["presence"],
                labels=labels,
                args=args,
                spec=data["spec"],
            )
            full_reference_counts[receptor] = counts
            print(
                f"[PASS] {receptor} full-training reference reproduces step-03: "
                + ", ".join(f"{key}={value:,}" for key, value in counts.items()),
                flush=True,
            )

    prediction_rows: List[Dict[str, object]] = []
    fold_fit_rows: List[Dict[str, object]] = []
    reference_rows_log: List[Dict[str, object]] = []
    loo_assignment_log: List[Dict[str, object]] = []

    task_number = 0
    n_tasks = len(args.repeats) * len(args.folds)
    for repeat in args.repeats:
        repeat_assignments = assignments[
            assignments["outer_repeat"] == repeat
        ].copy()
        fold_map = dict(
            zip(
                repeat_assignments[args.sample_id_col].astype(str),
                repeat_assignments["outer_fold"].astype(int),
            )
        )

        for fold in args.folds:
            task_number += 1
            args.outer_repeat = int(repeat)
            args.outer_fold = int(fold)

            valid_ids = [sample_id for sample_id in sample_ids if fold_map[sample_id] == fold]
            train_ids = [sample_id for sample_id in sample_ids if fold_map[sample_id] != fold]
            train_rows = np.asarray([row_lookup[sample_id] for sample_id in train_ids], dtype=int)
            valid_rows = np.asarray([row_lookup[sample_id] for sample_id in valid_ids], dtype=int)

            if not valid_ids or not train_ids:
                raise ValueError(f"Empty train/validation partition: repeat={repeat}, fold={fold}.")
            train_labels = set(labels[train_rows])
            valid_labels = set(labels[valid_rows])
            if train_labels != {"RA", "ILD"} or valid_labels != {"RA", "ILD"}:
                raise ValueError(
                    f"Both classes required in train/validation: repeat={repeat}, fold={fold}."
                )

            print(
                f"[{task_number:03d}/{n_tasks:03d}] repeat={repeat}, fold={fold}: "
                f"train={len(train_ids)}, valid={len(valid_ids)}",
                flush=True,
            )

            train_public = worker.build_joint_public_for_partition(
                receptor_data=receptor_data,
                sample_rows=train_rows,
                labels=labels,
                args=args,
                context="final_tuning_cv_training_leave_one_out",
                inner_fold=fold,
                reference_rows_log=reference_rows_log,
                loo_assignment_log=loo_assignment_log,
            )
            valid_public = worker.build_joint_external_public(
                receptor_data=receptor_data,
                reference_rows=train_rows,
                target_rows=valid_rows,
                labels=labels,
                args=args,
                context="final_tuning_cv_validation_application",
                inner_fold=fold,
                reference_rows_log=reference_rows_log,
            )

            train_df = worker.assemble_model_dataframe(
                base, train_ids, train_public, row_lookup
            )
            valid_df = worker.assemble_model_dataframe(
                base, valid_ids, valid_public, row_lookup
            )
            X_train, X_valid, design_features, _ = worker.build_design(
                train_df=train_df,
                valid_df=valid_df,
                numeric_cols=m2_spec["numeric"],
                categorical_cols=m2_spec["categorical"],
                zero_sd_tolerance=args.zero_sd_tolerance,
            )
            y_train = (
                train_df[args.label_col].str.upper() == args.positive_label.upper()
            ).astype(int).to_numpy()
            y_valid = (
                valid_df[args.label_col].str.upper() == args.positive_label.upper()
            ).astype(int).to_numpy()

            for candidate_index, (alpha, lambda_value) in enumerate(candidates):
                model, converged, n_iter = worker.fit_elastic_net(
                    X=X_train,
                    y=y_train,
                    l1_ratio=float(alpha),
                    lambda_value=float(lambda_value),
                    args=args,
                    seed=derive_seed(
                        args.seed,
                        700,
                        int(repeat),
                        int(fold),
                        candidate_index,
                        int(round(alpha * 1000)),
                        int(round(lambda_value * 1000)),
                    ),
                )
                probability = model.predict_proba(X_valid)[:, 1]
                nonzero = int(np.sum(np.abs(model.coef_.ravel()) > 1e-12))

                fold_fit_rows.append(
                    {
                        "analysis_view": ANALYSIS_VIEW,
                        "cv_unit": CV_UNIT,
                        "model": MODEL_NAME,
                        "outer_repeat": int(repeat),
                        "outer_fold": int(fold),
                        "l1_ratio_alpha": float(alpha),
                        "lambda": float(lambda_value),
                        "n_train": len(train_ids),
                        "n_valid": len(valid_ids),
                        "n_design_features": len(design_features),
                        "n_nonzero_coefficients": nonzero,
                        "fit_converged": bool(converged),
                        "iterations_used": int(n_iter),
                        "fold_roc_auc": safe_auc(y_valid, probability),
                        "fold_pr_auc": float(average_precision_score(y_valid, probability)),
                    }
                )

                for patient_id, y_value, prob in zip(valid_ids, y_valid, probability):
                    prediction_rows.append(
                        {
                            "analysis_view": ANALYSIS_VIEW,
                            "cv_unit": CV_UNIT,
                            "model": MODEL_NAME,
                            "outer_repeat": int(repeat),
                            "outer_fold": int(fold),
                            "sample_id": patient_id,
                            "patient": patient_id,
                            "trb_libraryid": str(base.loc[patient_id, args.trb_library_id_col]),
                            "igh_libraryid": str(base.loc[patient_id, args.igh_library_id_col]),
                            "true_label": int(y_value),
                            "probability_ILD": float(prob),
                            "l1_ratio_alpha": float(alpha),
                            "lambda": float(lambda_value),
                        }
                    )

    cv_predictions = pd.DataFrame(prediction_rows)
    fold_fits = pd.DataFrame(fold_fit_rows)
    expected_prediction_rows = len(args.repeats) * n_samples * len(candidates)
    expected_fold_fits = n_tasks * len(candidates)
    if len(cv_predictions) != expected_prediction_rows:
        raise RuntimeError(
            f"Expected {expected_prediction_rows} CV prediction rows, observed {len(cv_predictions)}."
        )
    if len(fold_fits) != expected_fold_fits:
        raise RuntimeError(
            f"Expected {expected_fold_fits} fold fits, observed {len(fold_fits)}."
        )
    if cv_predictions.duplicated(
        ["outer_repeat", "l1_ratio_alpha", "lambda", "sample_id"]
    ).any():
        raise RuntimeError("Duplicate repeated-OOF patient predictions detected.")
    candidate_counts = cv_predictions.groupby(
        ["outer_repeat", "l1_ratio_alpha", "lambda"]
    )["sample_id"].nunique()
    if not (candidate_counts == n_samples).all():
        raise RuntimeError("A repeat/candidate does not cover all training patients exactly once.")

    repeat_metrics = repeat_metrics_from_predictions(cv_predictions)
    candidate_summary = summarize_candidates(repeat_metrics, fold_fits, cv_predictions)
    selected, selection_details = select_candidate(candidate_summary, args.selection_rule)
    selected_alpha = float(selected["l1_ratio_alpha"])
    selected_lambda = float(selected["lambda"])

    selected_predictions = cv_predictions[
        (cv_predictions["l1_ratio_alpha"] == selected_alpha)
        & (cv_predictions["lambda"] == selected_lambda)
    ].copy()
    sample_mean_oof = (
        selected_predictions.groupby("sample_id", as_index=False)
        .agg(
            patient=("patient", "first"),
            trb_libraryid=("trb_libraryid", "first"),
            igh_libraryid=("igh_libraryid", "first"),
            true_label=("true_label", "first"),
            probability_ILD_mean=("probability_ILD", "mean"),
            probability_ILD_sd=("probability_ILD", "std"),
            probability_ILD_median=("probability_ILD", "median"),
            probability_ILD_min=("probability_ILD", "min"),
            probability_ILD_max=("probability_ILD", "max"),
            n_repeated_predictions=("probability_ILD", "size"),
        )
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    if len(sample_mean_oof) != n_samples:
        raise RuntimeError(
            f"Selected-candidate sample-mean OOF has {len(sample_mean_oof)} patients; expected {n_samples}."
        )
    if not (sample_mean_oof["n_repeated_predictions"] == len(args.repeats)).all():
        raise RuntimeError("Each patient must have one selected-candidate OOF prediction per repeat.")

    threshold_table = choose_thresholds(
        sample_mean_oof["true_label"].to_numpy(int),
        sample_mean_oof["probability_ILD_mean"].to_numpy(float),
        args.target_sensitivity,
    )
    selected_threshold = float(
        threshold_table.loc[
            threshold_table["threshold_rule"] == args.threshold_rule, "threshold"
        ].iloc[0]
    )
    sample_mean_oof["selected_alpha"] = selected_alpha
    sample_mean_oof["selected_lambda"] = selected_lambda
    sample_mean_oof["locked_threshold"] = selected_threshold
    sample_mean_oof["predicted_label"] = (
        sample_mean_oof["probability_ILD_mean"] >= selected_threshold
    ).astype(int)

    repeat_threshold_rows: List[Dict[str, object]] = []
    for repeat, group in selected_predictions.groupby("outer_repeat", sort=True):
        repeat_table = choose_thresholds(
            group["true_label"].to_numpy(int),
            group["probability_ILD"].to_numpy(float),
            args.target_sensitivity,
        )
        row = repeat_table[
            repeat_table["threshold_rule"] == args.threshold_rule
        ].iloc[0].to_dict()
        row.update(
            {
                "outer_repeat": int(repeat),
                "l1_ratio_alpha": selected_alpha,
                "lambda": selected_lambda,
            }
        )
        repeat_threshold_rows.append(row)
    repeat_thresholds = pd.DataFrame(repeat_threshold_rows).sort_values("outer_repeat")

    # Final all-training receptor-specific LOO features and joint model fit.
    args.outer_repeat = 0
    args.outer_fold = 0
    all_rows = np.arange(n_samples, dtype=int)
    final_public = worker.build_joint_public_for_partition(
        receptor_data=receptor_data,
        sample_rows=all_rows,
        labels=labels,
        args=args,
        context="final_all_training_leave_one_out",
        inner_fold=None,
        reference_rows_log=reference_rows_log,
        loo_assignment_log=loo_assignment_log,
    )
    final_df = worker.assemble_model_dataframe(
        base, sample_ids, final_public, row_lookup
    )
    X_final, _, final_feature_names, preprocessing_rows = worker.build_design(
        train_df=final_df,
        valid_df=final_df,
        numeric_cols=m2_spec["numeric"],
        categorical_cols=m2_spec["categorical"],
        zero_sd_tolerance=args.zero_sd_tolerance,
    )
    y_final = (
        final_df[args.label_col].str.upper() == args.positive_label.upper()
    ).astype(int).to_numpy()
    final_model, final_converged, final_iterations = worker.fit_elastic_net(
        X=X_final,
        y=y_final,
        l1_ratio=selected_alpha,
        lambda_value=selected_lambda,
        args=args,
        seed=derive_seed(args.seed, 799, n_samples),
    )
    if not final_converged:
        raise RuntimeError(
            "Final full-training paired M2 fit did not converge; model bundle was not saved."
        )
    final_probability = final_model.predict_proba(X_final)[:, 1]
    final_nonzero = int(np.sum(np.abs(final_model.coef_.ravel()) > 1e-12))

    preprocessing = pd.DataFrame(preprocessing_rows)
    preprocessing["model"] = MODEL_NAME
    preprocessing["analysis_view"] = ANALYSIS_VIEW
    preprocessing["final_training_n"] = n_samples

    prep_kept = preprocessing[
        preprocessing["kept_after_zero_variance_filter"]
    ].set_index("feature_name")
    coefficient_rows: List[Dict[str, object]] = [
        {
            "analysis_view": ANALYSIS_VIEW,
            "model": MODEL_NAME,
            "feature_name": "__INTERCEPT__",
            "feature_source": "intercept",
            "coefficient": float(final_model.intercept_[0]),
            "absolute_coefficient": abs(float(final_model.intercept_[0])),
            "odds_ratio_per_1SD": np.nan,
            "nonzero": True,
            "training_mean": np.nan,
            "training_sd": np.nan,
        }
    ]
    for name, coefficient in zip(final_feature_names, final_model.coef_.ravel()):
        prep = prep_kept.loc[name]
        coefficient_rows.append(
            {
                "analysis_view": ANALYSIS_VIEW,
                "model": MODEL_NAME,
                "feature_name": name,
                "feature_source": feature_source(name),
                "coefficient": float(coefficient),
                "absolute_coefficient": abs(float(coefficient)),
                "odds_ratio_per_1SD": float(np.exp(coefficient)),
                "nonzero": bool(abs(coefficient) > 1e-12),
                "training_mean": float(prep["training_mean"]),
                "training_sd": float(prep["training_sd"]),
            }
        )
    coefficients = pd.DataFrame(coefficient_rows).sort_values(
        ["nonzero", "absolute_coefficient", "feature_name"],
        ascending=[False, False, True],
    )

    fitted_predictions = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "patient": sample_ids,
            "trb_libraryid": base[args.trb_library_id_col].astype(str).to_numpy(),
            "igh_libraryid": base[args.igh_library_id_col].astype(str).to_numpy(),
            "true_label": y_final,
            "probability_ILD": final_probability,
            "locked_threshold": selected_threshold,
            "predicted_label": (final_probability >= selected_threshold).astype(int),
            "prediction_type": "apparent_full_training_fit",
        }
    )

    final_public_out = final_public.copy()
    final_public_out.insert(0, "sample_id", sample_ids)
    final_public_out.insert(1, "patient", sample_ids)
    final_public_out.insert(
        2, "trb_libraryid", base[args.trb_library_id_col].astype(str).to_numpy()
    )
    final_public_out.insert(
        3, "igh_libraryid", base[args.igh_library_id_col].astype(str).to_numpy()
    )

    full_reference_summary, full_reference_metadata = save_joint_reference_masks(
        paths["reference_masks"], receptor_data, labels, args, worker
    )

    preprocessor_metadata = build_preprocessor_metadata(
        final_df,
        m2_spec["numeric"],
        m2_spec["categorical"],
        preprocessing,
        final_feature_names,
    )
    training_mapping = base[
        [args.sample_id_col, args.trb_library_id_col, args.igh_library_id_col, args.label_col]
    ].reset_index(drop=True).to_dict(orient="records")

    model_bundle = {
        "script_version": SCRIPT_VERSION,
        "analysis_view": ANALYSIS_VIEW,
        "cv_unit": CV_UNIT,
        "model_name": MODEL_NAME,
        "model_definition": (
            "age + sex + static TRB + static IGH + receptor-specific dynamic public"
        ),
        "sklearn_model": final_model,
        "selected_alpha": selected_alpha,
        "selected_lambda": selected_lambda,
        "locked_threshold": selected_threshold,
        "threshold_rule": args.threshold_rule,
        "positive_label": args.positive_label,
        "class_mapping": {"RA": 0, "ILD": 1},
        "public_feature_set": args.public_feature_set,
        "public_features_per_receptor": selected_public_base,
        "joint_public_features": public_features,
        "static_features": static_features,
        "static_trb_features": static_trb,
        "static_igh_features": static_igh,
        "preprocessor": preprocessor_metadata,
        "receptors": {
            receptor: {
                "library_id_column": specs[receptor].library_id_col,
                "aa_clone_number_column": specs[receptor].aa_clone_number_col,
                "clone_file_pattern": specs[receptor].clone_file_pattern,
                "sequence_column": args.sequence_col,
                "frequency_column": args.frequency_col,
                "public_catalog_path": str(specs[receptor].public_catalog),
                "public_catalog_sha256": sha256sum(Path(specs[receptor].public_catalog)),
                "reference_definition_path": str(reference_definition_paths[receptor]),
                "reference_definition_sha256": sha256sum(
                    reference_definition_paths[receptor]
                ),
                "reference_definition": reference_definitions[receptor],
                "full_public_reference_summary": full_reference_metadata[receptor],
                "cache_metadata": cache_metadata[receptor],
            }
            for receptor in ("TRB", "IGH")
        },
        "public_reference_masks_file": paths["reference_masks"].name,
        "public_reference_masks_sha256": sha256sum(paths["reference_masks"]),
        "training_sample_ids": sample_ids,
        "training_patient_library_mapping": training_mapping,
        "n_training_samples": n_samples,
        "n_RA": n_ra,
        "n_ILD": n_ild,
        "training_public_generation": (
            f"exact receptor-specific leave-one-out, n_reference={n_samples - 1}"
        ),
        "new_sample_public_generation": (
            f"complete paired {n_samples}-patient training reference per receptor"
        ),
        "fit_converged": bool(final_converged),
        "iterations_used": int(final_iterations),
    }
    joblib.dump(model_bundle, paths["model_bundle"], compress=3)

    # Save all tables after all model/reference checks have passed.
    cv_predictions.to_csv(paths["cv_predictions"], index=False, compression="gzip")
    fold_fits.to_csv(paths["cv_fold_fits"], index=False)
    repeat_metrics.to_csv(paths["cv_repeat_metrics"], index=False)
    candidate_summary.to_csv(paths["candidate_summary"], index=False)
    sample_mean_oof.to_csv(paths["selected_oof"], index=False)
    threshold_table.to_csv(paths["threshold_candidates"], index=False)
    repeat_thresholds.to_csv(paths["repeat_thresholds"], index=False)
    final_public_out.to_csv(paths["final_public"], index=False)
    preprocessing.to_csv(paths["preprocessing"], index=False)
    coefficients.to_csv(paths["coefficients"], index=False)
    fitted_predictions.to_csv(paths["fitted_predictions"], index=False)
    full_reference_summary.to_csv(paths["reference_summary"], index=False)
    pd.DataFrame(reference_rows_log).to_csv(
        paths["reference_log"], index=False, compression="gzip"
    )
    pd.DataFrame(loo_assignment_log).to_csv(
        paths["loo_log"], index=False, compression="gzip"
    )

    config = {
        "script_version": SCRIPT_VERSION,
        "analysis_view": ANALYSIS_VIEW,
        "cv_unit": CV_UNIT,
        "model": MODEL_NAME,
        "independent_test_read": False,
        "input_files": {key: str(path) for key, path in input_paths.items()},
        "input_sha256": {key: sha256sum(path) for key, path in input_paths.items()},
        "trb_clone_table_dir": str(Path(args.trb_clone_table_dir).resolve()),
        "igh_clone_table_dir": str(Path(args.igh_clone_table_dir).resolve()),
        "trb_cache_dir": str(Path(args.trb_cache_dir).resolve()),
        "igh_cache_dir": str(Path(args.igh_cache_dir).resolve()),
        "cache_metadata": cache_metadata,
        "n_training_samples": n_samples,
        "n_RA": n_ra,
        "n_ILD": n_ild,
        "n_static_trb_features": len(static_trb),
        "n_static_igh_features": len(static_igh),
        "public_feature_set": args.public_feature_set,
        "public_features_per_receptor": selected_public_base,
        "joint_public_features": public_features,
        "tuning_mode": args.tuning_mode,
        "candidate_pairs": [
            {"alpha": alpha, "lambda": lambda_value}
            for alpha, lambda_value in candidates
        ],
        "repeats": args.repeats,
        "folds": args.folds,
        "frozen_assignments_sha256": assignment_sha,
        "selection_rule": args.selection_rule,
        "selection_details": selection_details,
        "selected_alpha": selected_alpha,
        "selected_lambda": selected_lambda,
        "threshold_rule": args.threshold_rule,
        "target_sensitivity": args.target_sensitivity,
        "locked_threshold": selected_threshold,
        "class_weight": args.class_weight,
        "max_iter": args.max_iter,
        "tolerance": args.tolerance,
        "seed": args.seed,
        "reference_definitions": reference_definitions,
        "full_reference_counts": full_reference_counts,
        "full_reference_masks_sha256": sha256sum(paths["reference_masks"]),
        "final_fit_converged": bool(final_converged),
        "final_iterations": int(final_iterations),
        "final_nonzero_coefficients": final_nonzero,
        "output_files": {key: str(path) for key, path in paths.items()},
    }
    paths["configuration"].write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    runtime = time.time() - started
    write_summary(
        paths["summary"],
        args,
        n_samples,
        n_ra,
        n_ild,
        candidates,
        candidate_summary,
        selected,
        selection_details,
        threshold_table,
        selected_threshold,
        final_converged,
        final_iterations,
        final_nonzero,
        full_reference_counts,
        runtime,
    )

    print("", flush=True)
    print("[Candidate summary]", flush=True)
    print(candidate_summary.to_string(index=False), flush=True)
    print("", flush=True)
    print("[Selected final M2 parameters]", flush=True)
    print(f"alpha={selected_alpha}", flush=True)
    print(f"lambda={selected_lambda}", flush=True)
    print(f"selection_rule={args.selection_rule}", flush=True)
    print("", flush=True)
    print(
        f"[Threshold candidates on {n_samples} patient-level mean repeated OOF probabilities]",
        flush=True,
    )
    print(threshold_table.to_string(index=False), flush=True)
    print(f"Locked threshold ({args.threshold_rule})={selected_threshold:.6f}", flush=True)
    print("", flush=True)
    print("[Final full-training fit]", flush=True)
    print(f"converged={final_converged}", flush=True)
    print(f"iterations={final_iterations}", flush=True)
    print(f"nonzero_coefficients={final_nonzero}", flush=True)
    print("Independent test set: NOT READ", flush=True)
    print("", flush=True)
    print("[Output files]", flush=True)
    for key, path in paths.items():
        print(f"- {key}: {path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
