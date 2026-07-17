#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final full-training M2 tuning, threshold locking and model fitting.

Project model:
    M2_static_igh_public
    age + sex + static IGH + dynamic reference-public features

Default workflow:
1. Read ONLY the 120-sample IGH training base matrix.
2. Reuse the leakage-controlled LOO/public/preprocessing functions from
   run_05_single_outer_trial_loo.py.
3. Use the frozen 20 repeated stratified 5-fold splits over all 120 training samples.
4. Compare the two stable candidate pairs identified in repeated nested CV:
       alpha=0.9, lambda=10
       alpha=0.5, lambda=10
   A full original grid remains available through --tuning-mode full_grid.
5. Select the final pair with a repeat-level one-standard-error rule:
   candidates within one SE of the best mean repeat ROC-AUC are eligible;
   among them prefer fewer nonzero coefficients, then higher PR-AUC.
6. Average each sample's 20 repeated OOF probabilities and lock one threshold
   without treating 2,400 repeated predictions as independent observations.
7. Generate full-training LOO public features (each sample uses the other 119),
   fit the final M2 on all 120 training samples, and save a deployable bundle.
8. The independent 49-sample test set is NEVER read.
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
from scipy import sparse
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


SCRIPT_VERSION = "1.1.0-IGH"
MODEL_NAME = "M2_static_igh_public"
ROOT = Path("/data/users/chenhaisheng/RA-ILD/IGH")

DEFAULT_WORKER = ROOT / "set/scripts/run_05_single_outer_trial_loo.py"
DEFAULT_BASE = (
    ROOT / "set/train/result/04_final_feature_matrix/"
           "04_train_base_feature_matrix.csv"
)
DEFAULT_MANIFEST = (
    ROOT / "set/train/result/04_final_feature_matrix/"
           "04_feature_manifest.csv"
)
DEFAULT_ASSIGNMENTS = (
    ROOT / "set/train/result/05_modeling/cv_splits/"
           "05_outer_fold_assignments.csv"
)
DEFAULT_CATALOG = (
    ROOT / "set/train/result/03_public_features/"
           "03_public_aa_catalog.csv.gz"
)
DEFAULT_REFERENCE_DEFINITION = (
    ROOT / "set/train/result/03_public_features/"
           "03_reference_definition.json"
)
DEFAULT_CLONE_DIR = ROOT / "set/train/result/01_AA_clone_table"
DEFAULT_CACHE_DIR = ROOT / "set/train/result/05_modeling/cache"
DEFAULT_OUTPUT_DIR = ROOT / "set/train/result/07_final_M2_model"


def parse_float_list(text: str) -> List[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one numeric value is required.")
    return values


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
        description="Tune and fit the final full-training M2 model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--loo-worker-script", default=str(DEFAULT_WORKER))
    parser.add_argument("--base-matrix", default=str(DEFAULT_BASE))
    parser.add_argument("--feature-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--cv-assignments", default=str(DEFAULT_ASSIGNMENTS))
    parser.add_argument("--public-catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument(
        "--reference-definition", default=str(DEFAULT_REFERENCE_DEFINITION)
    )
    parser.add_argument("--clone-table-dir", default=str(DEFAULT_CLONE_DIR))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    parser.add_argument("--repeats", type=parse_int_list, default=parse_int_list("1-20"))
    parser.add_argument("--folds", type=parse_int_list, default=parse_int_list("1-5"))

    parser.add_argument(
        "--tuning-mode",
        choices=("stable_pairs", "full_grid"),
        default="stable_pairs",
        help=(
            "stable_pairs evaluates the two candidates locked from step 06; "
            "full_grid evaluates the original Cartesian grid."
        ),
    )
    parser.add_argument(
        "--candidate-pairs",
        type=parse_candidate_pairs,
        default=parse_candidate_pairs("0.9:10,0.5:10"),
    )
    parser.add_argument(
        "--l1-ratios",
        type=parse_float_list,
        default=parse_float_list("0.1,0.5,0.9,1.0"),
    )
    parser.add_argument(
        "--lambdas",
        type=parse_float_list,
        default=parse_float_list("0.01,0.03,0.1,0.3,1,3,10,30,100"),
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

    parser.add_argument("--public-feature-set", default="raw_bilateral")
    parser.add_argument("--sample-id-col", default="sample_id")
    parser.add_argument("--label-col", default="cohort")
    parser.add_argument("--positive-label", default="ILD")
    parser.add_argument("--age-col", default="age")
    parser.add_argument("--sex-col", default="sex")
    parser.add_argument("--material-col", default="material")
    parser.add_argument("--aa-clone-number-col", default="aa_clone_number")
    parser.add_argument(
        "--clone-file-pattern",
        default="{sample_id}_IGH-without-DJ_CDR3_AA_clone_table.csv",
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

    parser.add_argument("--expected-samples", type=int, default=120)
    parser.add_argument("--expected-catalog-size", type=int, default=41052)
    parser.add_argument("--expected-global-size", type=int, default=6387)
    parser.add_argument("--expected-ra-specific-size", type=int, default=210)
    parser.add_argument("--expected-ild-specific-size", type=int, default=504)
    parser.add_argument("--expected-shared-size", type=int, default=263)
    parser.add_argument("--skip-full-reference-validation", action="store_true")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if not (0 < args.target_sensitivity <= 1):
        parser.error("--target-sensitivity must be in (0, 1].")
    if args.selection_rule == "one_se" and len(args.repeats) < 2:
        parser.error("--selection-rule one_se requires at least two repeats.")
    for x in args.l1_ratios:
        if not (0 < x <= 1):
            parser.error("Every l1 ratio must be in (0, 1].")
    for x in args.lambdas:
        if x <= 0:
            parser.error("Every lambda must be > 0.")
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
    text = ":".join(str(x) for x in (base_seed, *parts))
    return int.from_bytes(
        hashlib.sha256(text.encode("utf-8")).digest()[:8], "little"
    ) % (2**32 - 1)


def load_worker(path: Path) -> ModuleType:
    ensure_file(path, "LOO worker script")
    spec = importlib.util.spec_from_file_location("loo_worker", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import worker script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    required = [
        "PUBLIC_FEATURE_SETS",
        "static_igh_features_from_manifest",
        "load_and_validate_reference_definition",
        "build_or_load_sparse_cache",
        "validate_full_reference",
        "model_specifications",
        "leave_one_out_public_features",
        "external_public_features",
        "assemble_model_dataframe",
        "build_design",
        "fit_elastic_net",
        "build_public_reference",
        "classification_metrics",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise ImportError(f"Worker script missing required objects: {missing}")
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


def validate_and_read_inputs(args: argparse.Namespace):
    paths = {
        "base": Path(args.base_matrix).expanduser().resolve(),
        "manifest": Path(args.feature_manifest).expanduser().resolve(),
        "assignments": Path(args.cv_assignments).expanduser().resolve(),
        "catalog": Path(args.public_catalog).expanduser().resolve(),
        "reference_definition": Path(args.reference_definition).expanduser().resolve(),
        "worker": Path(args.loo_worker_script).expanduser().resolve(),
    }
    for key, path in paths.items():
        ensure_file(path, key)

    clone_dir = Path(args.clone_table_dir).expanduser().resolve()
    if not clone_dir.is_dir():
        raise FileNotFoundError(f"clone table directory not found: {clone_dir}")

    base = pd.read_csv(paths["base"])
    manifest = pd.read_csv(paths["manifest"])
    assignments = pd.read_csv(paths["assignments"])

    required_base = {
        args.sample_id_col,
        args.label_col,
        args.age_col,
        args.sex_col,
        args.material_col,
        args.aa_clone_number_col,
    }
    missing = required_base - set(base.columns)
    if missing:
        raise ValueError(f"Base matrix missing columns: {sorted(missing)}")
    if base[args.sample_id_col].duplicated().any():
        raise ValueError("Base matrix has duplicated sample IDs.")
    if base.isna().any().any():
        bad = base.isna().sum()
        raise ValueError(
            "Base matrix contains missing values: "
            + str(bad[bad > 0].to_dict())
        )

    base[args.sample_id_col] = base[args.sample_id_col].astype(str)
    base[args.label_col] = base[args.label_col].astype(str).str.upper()
    if args.expected_samples > 0 and len(base) != args.expected_samples:
        raise ValueError(
            f"Expected {args.expected_samples} training samples, observed {len(base)}."
        )
    label_set = set(base[args.label_col])
    if label_set != {"RA", "ILD"}:
        raise ValueError(f"Expected RA/ILD labels, observed: {sorted(label_set)}")

    required_assignment = {
        "outer_repeat", "outer_fold", args.sample_id_col
    }
    missing = required_assignment - set(assignments.columns)
    if missing:
        raise ValueError(f"CV assignments missing columns: {sorted(missing)}")
    assignments[args.sample_id_col] = assignments[args.sample_id_col].astype(str)

    selected_assignments = assignments[
        assignments["outer_repeat"].isin(args.repeats)
        & assignments["outer_fold"].isin(args.folds)
    ].copy()

    expected_ids = set(base[args.sample_id_col])
    for repeat in args.repeats:
        repeat_df = selected_assignments[
            selected_assignments["outer_repeat"] == repeat
        ]
        if len(repeat_df) != len(base):
            raise ValueError(
                f"Repeat {repeat} has {len(repeat_df)} rows; expected {len(base)}."
            )
        if repeat_df[args.sample_id_col].duplicated().any():
            raise ValueError(f"Repeat {repeat} has duplicated sample IDs.")
        if set(repeat_df[args.sample_id_col]) != expected_ids:
            raise ValueError(f"Repeat {repeat} sample IDs do not match base matrix.")
        if set(repeat_df["outer_fold"]) != set(args.folds):
            raise ValueError(
                f"Repeat {repeat} fold set mismatch: "
                f"{sorted(repeat_df['outer_fold'].unique())}"
            )

    base = base.set_index(args.sample_id_col, drop=False)
    return base, manifest, selected_assignments, paths


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
    best = sorted(rows, key=lambda x: (-x[0], -x[1], abs(x[2] - 0.5)))[0]
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
            rows.append(
                (
                    metrics["specificity"],
                    metrics["precision"],
                    threshold,
                )
            )
    if not rows:
        return 0.0
    best = sorted(rows, key=lambda x: (-x[0], -x[1], abs(x[2] - 0.5)))[0]
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
    for keys, group in predictions.groupby(group_cols):
        alpha, lambda_value, repeat = keys
        if group["sample_id"].duplicated().any():
            raise ValueError(
                f"Duplicate OOF sample for alpha={alpha}, lambda={lambda_value}, "
                f"repeat={repeat}."
            )
        y = group["true_label"].to_numpy(int)
        probability = group["probability_ILD"].to_numpy(float)
        rows.append({
            "l1_ratio_alpha": float(alpha),
            "lambda": float(lambda_value),
            "outer_repeat": int(repeat),
            "n_samples": len(group),
            "roc_auc": safe_auc(y, probability),
            "pr_auc": float(average_precision_score(y, probability)),
        })
    return pd.DataFrame(rows).sort_values(
        ["l1_ratio_alpha", "lambda", "outer_repeat"]
    )


def summarize_candidates(
    repeat_metrics: pd.DataFrame,
    fold_fits: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (alpha, lambda_value), repeat_group in repeat_metrics.groupby(
        ["l1_ratio_alpha", "lambda"]
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
        rows.append({
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
        })
    return pd.DataFrame(rows).sort_values(
        ["mean_repeat_roc_auc", "mean_repeat_pr_auc"],
        ascending=[False, False],
    ).reset_index(drop=True)


def select_candidate(
    summary: pd.DataFrame,
    rule: str,
) -> Tuple[pd.Series, Dict[str, object]]:
    converged_summary = summary[summary["all_fits_converged"]].copy()
    if converged_summary.empty:
        raise RuntimeError("No candidate converged in every repeated-CV fold fit.")

    ranked = converged_summary.sort_values(
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

    cutoff = float(
        best["mean_repeat_roc_auc"] - best["se_repeat_roc_auc"]
    )
    eligible = converged_summary[
        converged_summary["mean_repeat_roc_auc"] >= cutoff - 1e-12
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


def build_preprocessor_metadata(
    train_df: pd.DataFrame,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    preprocessing: pd.DataFrame,
    feature_names: Sequence[str],
) -> Dict[str, object]:
    categorical_levels = {
        col: sorted(train_df[col].astype(str).unique().tolist())
        for col in categorical_cols
    }
    categorical_reference = {
        col: levels[0] for col, levels in categorical_levels.items()
    }
    return {
        "numeric_columns": list(numeric_cols),
        "categorical_columns": list(categorical_cols),
        "categorical_levels": categorical_levels,
        "categorical_reference": categorical_reference,
        "final_feature_names": list(feature_names),
        "preprocessing_rows": preprocessing.to_dict(orient="records"),
    }


def write_summary(
    path: Path,
    args: argparse.Namespace,
    candidates: Sequence[Tuple[float, float]],
    candidate_summary: pd.DataFrame,
    selected: pd.Series,
    selection_details: Mapping[str, object],
    threshold_table: pd.DataFrame,
    selected_threshold: float,
    final_converged: bool,
    final_iterations: int,
    final_nonzero: int,
    full_reference_counts: Optional[Mapping[str, int]],
    runtime_seconds: float,
) -> None:
    chosen_threshold_row = threshold_table[
        threshold_table["threshold_rule"] == args.threshold_rule
    ].iloc[0]
    lines = [
        "# 07 Final M2 Training Summary",
        "",
        "## Leakage boundary",
        "",
        f"- Only the {args.expected_samples}-sample IGH training set was read.",
        "- The independent 49-sample IGH test set was not read.",
        "- CV-training public features used exact leave-one-out references.",
        "- CV-validation public features used only the corresponding CV-training partition.",
        f"- Final fitting public features used all other {args.expected_samples - 1} training samples for each sample.",
        "",
        "## Candidate tuning",
        "",
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
        "",
        "## Final all-training fit",
        "",
        f"- Final fit converged: `{final_converged}`",
        f"- Iterations used: `{final_iterations}`",
        f"- Nonzero coefficients: `{final_nonzero}`",
        "- Final fitted training metrics are apparent/in-sample and are not an unbiased performance estimate.",
        "",
        "## Full training reference",
        "",
    ]
    if full_reference_counts is None:
        lines.append("- Full-reference step-03 validation was skipped.")
    else:
        for key, value in full_reference_counts.items():
            lines.append(f"- {key}: `{int(value):,}`")
    lines.extend([
        "",
        "## Selection details",
        "",
    ])
    for key, value in selection_details.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend([
        "",
        f"- Runtime: `{runtime_seconds:.1f}` seconds",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.time()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)
    enforce_overwrite(paths.values(), args.overwrite)

    base, manifest, assignments, input_paths = validate_and_read_inputs(args)
    worker = load_worker(input_paths["worker"])

    worker_receptor = str(getattr(worker, "EXPECTED_RECEPTOR", "")).upper()
    if worker_receptor != "IGH":
        raise ValueError(
            f"LOO worker receptor={worker_receptor or '<missing>'}; expected IGH."
        )
    reference_definition_path, reference_definition = (
        worker.load_and_validate_reference_definition(args)
    )
    if reference_definition_path != input_paths["reference_definition"]:
        raise RuntimeError("Validated reference-definition path changed unexpectedly.")

    if args.public_feature_set not in worker.PUBLIC_FEATURE_SETS:
        raise ValueError(
            f"Unknown public feature set '{args.public_feature_set}'. "
            f"Available: {sorted(worker.PUBLIC_FEATURE_SETS)}"
        )

    static_features = worker.static_igh_features_from_manifest(manifest, base)
    public_features = worker.PUBLIC_FEATURE_SETS[args.public_feature_set]
    model_specs = worker.model_specifications(static_features, public_features, args)
    if MODEL_NAME not in model_specs:
        raise ValueError(
            f"Worker model specifications do not contain {MODEL_NAME}: "
            f"{sorted(model_specs)}"
        )
    m2_spec = model_specs[MODEL_NAME]
    candidates = candidate_pairs(args)
    if len(candidates) < 2 and args.selection_rule == "one_se":
        raise ValueError("one_se selection requires at least two candidate pairs.")

    print("=" * 80, flush=True)
    print("07 IGH Final Full-Training M2 Tuning and Fit", flush=True)
    print("=" * 80, flush=True)
    print("Receptor: IGH", flush=True)
    print(f"Training samples: {len(base)}", flush=True)
    print(f"Static IGH predictors: {len(static_features)}", flush=True)
    print(f"Dynamic public predictors: {len(public_features)}", flush=True)
    print(f"Repeated CV tasks: {len(args.repeats) * len(args.folds)}", flush=True)
    print(f"Candidate pairs: {candidates}", flush=True)
    print("Independent test set: NOT READ", flush=True)
    print("", flush=True)

    presence, frequency, cache_meta = worker.build_or_load_sparse_cache(args, base)
    sample_ids = base.index.astype(str).tolist()
    row_lookup = {sample_id: i for i, sample_id in enumerate(sample_ids)}
    labels = base[args.label_col].astype(str).str.upper().to_numpy()
    aa_clone_numbers = pd.to_numeric(
        base[args.aa_clone_number_col], errors="raise"
    ).to_numpy(float)

    full_reference_counts = None
    if not args.skip_full_reference_validation:
        full_reference_counts = worker.validate_full_reference(
            presence, labels, args
        )
        print(
            "[PASS] Full-train public reference sizes reproduce step-03: "
            + ", ".join(
                f"{key}={value:,}" for key, value in full_reference_counts.items()
            ),
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

            valid_ids = [sid for sid in sample_ids if fold_map[sid] == fold]
            train_ids = [sid for sid in sample_ids if fold_map[sid] != fold]
            train_rows = np.array([row_lookup[sid] for sid in train_ids], dtype=int)
            valid_rows = np.array([row_lookup[sid] for sid in valid_ids], dtype=int)

            print(
                f"[{task_number:03d}/{n_tasks:03d}] repeat={repeat}, fold={fold}: "
                f"train={len(train_ids)}, valid={len(valid_ids)}",
                flush=True,
            )

            train_public = worker.leave_one_out_public_features(
                presence=presence,
                frequency=frequency,
                sample_rows=train_rows,
                labels=labels,
                aa_clone_numbers=aa_clone_numbers,
                args=args,
                context="final_tuning_cv_training_leave_one_out",
                inner_fold=fold,
                reference_rows_log=reference_rows_log,
                loo_assignment_log=loo_assignment_log,
            )
            valid_public = worker.external_public_features(
                presence=presence,
                frequency=frequency,
                reference_rows=train_rows,
                target_rows=valid_rows,
                labels=labels,
                aa_clone_numbers=aa_clone_numbers,
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
            X_train, X_valid, feature_names, _ = worker.build_design(
                train_df=train_df,
                valid_df=valid_df,
                numeric_cols=m2_spec["numeric"],
                categorical_cols=m2_spec["categorical"],
                zero_sd_tolerance=args.zero_sd_tolerance,
            )
            y_train = (
                train_df[args.label_col].str.upper()
                == args.positive_label.upper()
            ).astype(int).to_numpy()
            y_valid = (
                valid_df[args.label_col].str.upper()
                == args.positive_label.upper()
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

                fold_fit_rows.append({
                    "model": MODEL_NAME,
                    "outer_repeat": int(repeat),
                    "outer_fold": int(fold),
                    "l1_ratio_alpha": float(alpha),
                    "lambda": float(lambda_value),
                    "n_train": len(train_ids),
                    "n_valid": len(valid_ids),
                    "n_design_features": len(feature_names),
                    "n_nonzero_coefficients": nonzero,
                    "fit_converged": bool(converged),
                    "iterations_used": int(n_iter),
                    "fold_roc_auc": safe_auc(y_valid, probability),
                    "fold_pr_auc": float(
                        average_precision_score(y_valid, probability)
                    ),
                })

                for sample_id, y_value, prob in zip(
                    valid_ids, y_valid, probability
                ):
                    prediction_rows.append({
                        "model": MODEL_NAME,
                        "outer_repeat": int(repeat),
                        "outer_fold": int(fold),
                        "sample_id": sample_id,
                        "true_label": int(y_value),
                        "probability_ILD": float(prob),
                        "l1_ratio_alpha": float(alpha),
                        "lambda": float(lambda_value),
                    })

    cv_predictions = pd.DataFrame(prediction_rows)
    fold_fits = pd.DataFrame(fold_fit_rows)

    expected_prediction_rows = len(args.repeats) * len(base) * len(candidates)
    if len(cv_predictions) != expected_prediction_rows:
        raise RuntimeError(
            f"Expected {expected_prediction_rows} CV prediction rows, "
            f"observed {len(cv_predictions)}."
        )
    expected_fold_fits = n_tasks * len(candidates)
    if len(fold_fits) != expected_fold_fits:
        raise RuntimeError(
            f"Expected {expected_fold_fits} fold fits, observed {len(fold_fits)}."
        )

    repeat_metrics = repeat_metrics_from_predictions(cv_predictions)
    candidate_summary = summarize_candidates(
        repeat_metrics, fold_fits, cv_predictions
    )
    selected, selection_details = select_candidate(
        candidate_summary, args.selection_rule
    )
    selected_alpha = float(selected["l1_ratio_alpha"])
    selected_lambda = float(selected["lambda"])

    selected_predictions = cv_predictions[
        (cv_predictions["l1_ratio_alpha"] == selected_alpha)
        & (cv_predictions["lambda"] == selected_lambda)
    ].copy()
    sample_mean_oof = (
        selected_predictions.groupby("sample_id", as_index=False)
        .agg(
            true_label=("true_label", "first"),
            probability_ILD_mean=("probability_ILD", "mean"),
            probability_ILD_sd=("probability_ILD", "std"),
            probability_ILD_median=("probability_ILD", "median"),
            probability_ILD_min=("probability_ILD", "min"),
            probability_ILD_max=("probability_ILD", "max"),
            n_repeated_predictions=("probability_ILD", "size"),
        )
        .sort_values("sample_id")
    )
    if len(sample_mean_oof) != len(base):
        raise RuntimeError(
            f"Selected-candidate sample-mean OOF table has {len(sample_mean_oof)} "
            f"samples; expected {len(base)}."
        )
    if not (sample_mean_oof["n_repeated_predictions"] == len(args.repeats)).all():
        raise RuntimeError(
            "Each sample does not have one selected-candidate OOF prediction per repeat."
        )

    threshold_table = choose_thresholds(
        sample_mean_oof["true_label"].to_numpy(int),
        sample_mean_oof["probability_ILD_mean"].to_numpy(float),
        args.target_sensitivity,
    )
    selected_threshold = float(
        threshold_table.loc[
            threshold_table["threshold_rule"] == args.threshold_rule,
            "threshold",
        ].iloc[0]
    )
    sample_mean_oof["locked_threshold"] = selected_threshold
    sample_mean_oof["predicted_label"] = (
        sample_mean_oof["probability_ILD_mean"] >= selected_threshold
    ).astype(int)

    repeat_threshold_rows: List[Dict[str, object]] = []
    for repeat, group in selected_predictions.groupby("outer_repeat"):
        y = group["true_label"].to_numpy(int)
        probability = group["probability_ILD"].to_numpy(float)
        repeat_table = choose_thresholds(y, probability, args.target_sensitivity)
        row = repeat_table[
            repeat_table["threshold_rule"] == args.threshold_rule
        ].iloc[0].to_dict()
        row.update({
            "outer_repeat": int(repeat),
            "l1_ratio_alpha": selected_alpha,
            "lambda": selected_lambda,
        })
        repeat_threshold_rows.append(row)
    repeat_thresholds = pd.DataFrame(repeat_threshold_rows).sort_values("outer_repeat")

    # Final full-training LOO features and fit.
    args.outer_repeat = 0
    args.outer_fold = 0
    all_rows = np.arange(len(base), dtype=int)
    final_public = worker.leave_one_out_public_features(
        presence=presence,
        frequency=frequency,
        sample_rows=all_rows,
        labels=labels,
        aa_clone_numbers=aa_clone_numbers,
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
        seed=derive_seed(args.seed, 799, len(base)),
    )
    if not final_converged:
        raise RuntimeError(
            "Final full-training M2 fit did not converge; model bundle was not saved."
        )
    final_probability = final_model.predict_proba(X_final)[:, 1]
    final_nonzero = int(np.sum(np.abs(final_model.coef_.ravel()) > 1e-12))

    preprocessing = pd.DataFrame(preprocessing_rows)
    preprocessing["model"] = MODEL_NAME
    preprocessing["final_training_n"] = len(base)

    coefficient_rows: List[Dict[str, object]] = [{
        "model": MODEL_NAME,
        "feature_name": "__INTERCEPT__",
        "coefficient": float(final_model.intercept_[0]),
        "absolute_coefficient": abs(float(final_model.intercept_[0])),
        "odds_ratio_per_1SD": np.nan,
        "nonzero": True,
        "source_type": "intercept",
        "training_mean": np.nan,
        "training_sd": np.nan,
    }]
    prep_kept = preprocessing[
        preprocessing["kept_after_zero_variance_filter"]
    ].set_index("feature_name")
    for feature_name, coefficient in zip(
        final_feature_names, final_model.coef_.ravel()
    ):
        prep = prep_kept.loc[feature_name]
        coefficient_rows.append({
            "model": MODEL_NAME,
            "feature_name": feature_name,
            "coefficient": float(coefficient),
            "absolute_coefficient": abs(float(coefficient)),
            "odds_ratio_per_1SD": float(np.exp(coefficient)),
            "nonzero": bool(abs(coefficient) > 1e-12),
            "source_type": prep["source_type"],
            "training_mean": float(prep["training_mean"]),
            "training_sd": float(prep["training_sd"]),
        })
    coefficients = pd.DataFrame(coefficient_rows).sort_values(
        ["nonzero", "absolute_coefficient"], ascending=[False, False]
    )

    fitted_predictions = pd.DataFrame({
        "sample_id": sample_ids,
        "true_label": y_final,
        "probability_ILD": final_probability,
        "locked_threshold": selected_threshold,
        "predicted_label": (final_probability >= selected_threshold).astype(int),
        "prediction_type": "apparent_full_training_fit",
    })

    final_public_out = final_public.copy()
    final_public_out.insert(0, "sample_id", sample_ids)

    # Full 120-sample public reference for future test/new-sample application.
    full_reference = worker.build_public_reference(
        presence=presence,
        reference_rows=all_rows,
        labels=labels,
        args=args,
    )
    np.savez_compressed(
        paths["reference_masks"],
        receptor=np.asarray("IGH"),
        n_reference=np.asarray(full_reference.n_reference, dtype=np.int32),
        n_RA_reference=np.asarray(full_reference.n_ra, dtype=np.int32),
        n_ILD_reference=np.asarray(full_reference.n_ild, dtype=np.int32),
        global_threshold=np.asarray(full_reference.global_threshold, dtype=np.int32),
        RA_threshold=np.asarray(full_reference.ra_threshold, dtype=np.int32),
        ILD_threshold=np.asarray(full_reference.ild_threshold, dtype=np.int32),
        public_catalog_sha256=np.asarray(sha256sum(input_paths["catalog"])),
        reference_definition_sha256=np.asarray(
            sha256sum(input_paths["reference_definition"])
        ),
        global_mask=full_reference.global_mask.astype(np.uint8),
        RA_specific_mask=full_reference.ra_specific_mask.astype(np.uint8),
        ILD_specific_mask=full_reference.ild_specific_mask.astype(np.uint8),
        shared_mask=full_reference.shared_mask.astype(np.uint8),
    )
    full_reference_summary = pd.DataFrame([{
        "n_reference": full_reference.n_reference,
        "n_RA_reference": full_reference.n_ra,
        "n_ILD_reference": full_reference.n_ild,
        "global_threshold": full_reference.global_threshold,
        "RA_threshold": full_reference.ra_threshold,
        "ILD_threshold": full_reference.ild_threshold,
        "all_ref_public_size": int(full_reference.global_mask.sum()),
        "RA_specific_ref_size": int(full_reference.ra_specific_mask.sum()),
        "ILD_specific_ref_size": int(full_reference.ild_specific_mask.sum()),
        "shared_ref_size": int(full_reference.shared_mask.sum()),
    }])

    preprocessor_metadata = build_preprocessor_metadata(
        final_df,
        m2_spec["numeric"],
        m2_spec["categorical"],
        preprocessing,
        final_feature_names,
    )
    model_bundle = {
        "script_version": SCRIPT_VERSION,
        "receptor": "IGH",
        "model_name": MODEL_NAME,
        "model_definition": "age + sex + static IGH + dynamic public",
        "sklearn_model": final_model,
        "selected_alpha": selected_alpha,
        "selected_lambda": selected_lambda,
        "locked_threshold": selected_threshold,
        "threshold_rule": args.threshold_rule,
        "positive_label": args.positive_label,
        "class_mapping": {"RA": 0, "ILD": 1},
        "public_feature_set": args.public_feature_set,
        "public_features": list(public_features),
        "static_features": list(static_features),
        "static_igh_features": list(static_features),
        "preprocessor": preprocessor_metadata,
        "clone_file_pattern": args.clone_file_pattern,
        "sequence_column": args.sequence_col,
        "frequency_column": args.frequency_col,
        "reference_definition": reference_definition,
        "reference_definition_path": str(input_paths["reference_definition"]),
        "reference_definition_sha256": sha256sum(
            input_paths["reference_definition"]
        ),
        "public_reference_masks_file": paths["reference_masks"].name,
        "public_reference_masks_sha256": sha256sum(paths["reference_masks"]),
        "public_catalog_path": str(input_paths["catalog"]),
        "public_catalog_sha256": sha256sum(input_paths["catalog"]),
        "training_sample_ids": sample_ids,
        "n_training_samples": len(base),
        "n_RA": int((labels == "RA").sum()),
        "n_ILD": int((labels == "ILD").sum()),
        "full_public_reference_summary": full_reference_summary.iloc[0].to_dict(),
        "training_public_generation": f"exact leave-one-out, n_reference={len(base)-1}",
        "new_sample_public_generation": f"complete {len(base)}-sample training reference",
        "fit_converged": bool(final_converged),
        "iterations_used": int(final_iterations),
    }
    joblib.dump(model_bundle, paths["model_bundle"], compress=3)

    # Save all tables.
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
        "receptor": "IGH",
        "model": MODEL_NAME,
        "independent_test_read": False,
        "input_files": {key: str(path) for key, path in input_paths.items()},
        "input_sha256": {
            key: sha256sum(path) for key, path in input_paths.items()
        },
        "clone_table_dir": str(Path(args.clone_table_dir).resolve()),
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "cache_metadata": cache_meta,
        "n_training_samples": len(base),
        "n_RA": int((labels == "RA").sum()),
        "n_ILD": int((labels == "ILD").sum()),
        "n_static_igh_features": len(static_features),
        "public_feature_set": args.public_feature_set,
        "public_features": list(public_features),
        "tuning_mode": args.tuning_mode,
        "candidate_pairs": [
            {"alpha": alpha, "lambda": lambda_value}
            for alpha, lambda_value in candidates
        ],
        "repeats": args.repeats,
        "folds": args.folds,
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
        "reference_definition": reference_definition,
        "full_reference_counts": full_reference_counts,
        "full_reference_masks_sha256": sha256sum(paths["reference_masks"]),
        "final_fit_converged": bool(final_converged),
        "final_iterations": int(final_iterations),
        "final_nonzero_coefficients": final_nonzero,
        "output_files": {key: str(path) for key, path in paths.items()},
    }
    paths["configuration"].write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    runtime = time.time() - started
    write_summary(
        paths["summary"],
        args,
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
    print(f"[Threshold candidates on {len(base)} sample-mean repeated OOF probabilities]", flush=True)
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
    raise SystemExit(main())
