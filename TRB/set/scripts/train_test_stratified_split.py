#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Select a fixed 70/30 train/test split by repeated stratified sampling.

Designed for RA / RA-ILD TRB repertoire metadata. The default stratification
variables are cohort + batch. Batch is used only to balance the split; this
script does not perform feature extraction or model training.

Default output layout under --output-dir:
    train/metadata_train_70.csv
    test/metadata_test_30.csv
    train_test_split_summary.md
    all_seed_split_scores.csv

Example:
    python3 train_test_stratified_split.py \
        --metadata /data/users/chenhaisheng/RA-ILD/TRB/metadata.csv \
        --output-dir /data/users/chenhaisheng/RA-ILD/TRB/set \
        --train-ratio 0.70 \
        --seed-start 1 \
        --seed-end 5000 \
        --strata-cols cohort,batch \
        --id-col libraryid
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


MISSING_TOKEN = "__MISSING__"

# Distribution components and weights used in the imbalance score.
# Higher weights mean that a component has greater influence on seed selection.
FEATURE_CONFIG: Tuple[Tuple[str, Tuple[str, ...], float], ...] = (
    ("cohort", ("cohort",), 6.0),
    ("batch", ("batch",), 3.0),
    ("material", ("material",), 3.5),
    ("sex", ("sex",), 1.0),
    ("cohort_x_batch", ("cohort", "batch"), 6.0),
    ("cohort_x_material", ("cohort", "material"), 3.5),
    ("material_x_batch", ("material", "batch"), 3.0),
    ("cohort_x_sex", ("cohort", "sex"), 1.5),
)

# Penalties added when the test set completely misses a category that exists
# in the full metadata. These intentionally dominate small proportion shifts.
TEST_MISSING_CATEGORY_PENALTIES: Mapping[str, float] = {
    "cohort": 100.0,
    "batch": 45.0,
    "material": 70.0,
    "sex": 20.0,
}

# A cohort x batch stratum with >=2 samples should be represented in both sets
# under the default allocation rule. Missing such a non-singleton stratum is
# therefore penalized heavily. Singleton strata are only reported as warnings.
NONRARE_COHORT_BATCH_MISSING_PENALTY = 100.0


@dataclass(frozen=True)
class DistributionSpec:
    """Precomputed coding information for one scored distribution."""

    name: str
    columns: Tuple[str, ...]
    weight: float
    labels: np.ndarray
    codes: np.ndarray
    overall_counts: np.ndarray
    overall_proportions: np.ndarray


@dataclass(frozen=True)
class SplitResult:
    """One candidate split and its score components."""

    seed: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    metrics: Mapping[str, float]


class SplitError(RuntimeError):
    """Raised for invalid input or an invalid split."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repeated cohort+batch-stratified train/test splitting. "
            "The seed with the smallest distribution-imbalance score is selected."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata",
        required=True,
        type=Path,
        help="Input metadata CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output root. Train/test metadata are written into train/ and test/.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Target train fraction within each stratum.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1,
        help="First random seed, inclusive.",
    )
    parser.add_argument(
        "--seed-end",
        type=int,
        default=5000,
        help="Last random seed, inclusive.",
    )
    parser.add_argument(
        "--strata-cols",
        default="cohort,batch",
        help="Comma-separated stratification columns.",
    )
    parser.add_argument(
        "--id-col",
        default="auto",
        help=(
            "Unique sample ID column. With 'auto', libraryid is preferred and "
            "sample_id is used as fallback."
        ),
    )
    parser.add_argument(
        "--missing-policy",
        choices=("error", "warn"),
        default="error",
        help=(
            "How to handle missing values in cohort/batch/material/sex/age and "
            "stratification columns. 'warn' keeps rows and treats missing categorical "
            "values as a separate internal category."
        ),
    )
    parser.add_argument(
        "--singleton-policy",
        choices=("train", "test", "random"),
        default="train",
        help=(
            "Allocation for a stratum containing exactly one sample. Such a stratum "
            "cannot appear in both sets. 'random' assigns it according to train-ratio."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing output files.",
    )
    parser.add_argument(
        "--keep-unnamed-columns",
        action="store_true",
        help="Keep CSV columns such as 'Unnamed: 0'. By default they are removed.",
    )
    return parser.parse_args()


def parse_column_list(raw: str) -> List[str]:
    columns = [item.strip() for item in raw.split(",") if item.strip()]
    if not columns:
        raise SplitError("--strata-cols must contain at least one column name.")
    if len(columns) != len(set(columns)):
        raise SplitError(f"Duplicate names were found in --strata-cols: {columns}")
    return columns


def ratio_tag(value: float) -> str:
    """Convert 0.70 to '70' and 0.675 to '67p5' for file names."""
    pct = value * 100.0
    if math.isclose(pct, round(pct), abs_tol=1e-12):
        return str(int(round(pct)))
    text = f"{pct:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def normalized_text(series: pd.Series) -> pd.Series:
    """Normalize values for grouping/checking without modifying output metadata."""
    result = series.astype("string").str.strip()
    result = result.mask(result.eq(""), pd.NA)
    return result.fillna(MISSING_TOKEN)


def missing_mask(series: pd.Series) -> pd.Series:
    """Treat NaN and blank strings as missing."""
    mask = series.isna()
    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
        mask = mask | series.astype("string").str.strip().eq("").fillna(False)
    return mask


def resolve_id_column(metadata: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        if requested not in metadata.columns:
            raise SplitError(
                f"ID column '{requested}' is absent. Available columns: "
                f"{', '.join(map(str, metadata.columns))}"
            )
        return requested

    for candidate in ("libraryid", "sample_id"):
        if candidate in metadata.columns:
            return candidate
    raise SplitError(
        "--id-col auto could not find 'libraryid' or 'sample_id'. "
        "Please provide --id-col explicitly."
    )


def read_and_validate_metadata(
    path: Path,
    id_col_requested: str,
    strata_cols: Sequence[str],
    missing_policy: str,
    keep_unnamed_columns: bool,
) -> Tuple[pd.DataFrame, str, List[str]]:
    if not path.exists():
        raise SplitError(f"Metadata file does not exist: {path}")
    if not path.is_file():
        raise SplitError(f"Metadata path is not a regular file: {path}")

    try:
        metadata = pd.read_csv(path, low_memory=False)
    except Exception as exc:  # pragma: no cover - error text depends on parser
        raise SplitError(f"Failed to read metadata CSV: {exc}") from exc

    warnings: List[str] = []
    if metadata.empty:
        raise SplitError("Metadata contains zero rows.")

    unnamed = [col for col in metadata.columns if str(col).startswith("Unnamed:")]
    if unnamed and not keep_unnamed_columns:
        metadata = metadata.drop(columns=unnamed)
        warnings.append(
            "Removed CSV index-like column(s): " + ", ".join(map(str, unnamed))
        )

    id_col = resolve_id_column(metadata, id_col_requested)

    # Age is validated for missingness but deliberately not used in the score,
    # because the requested score components are categorical distributions only.
    required = set(strata_cols) | {
        id_col,
        "cohort",
        "batch",
        "material",
        "sex",
        "age",
    }
    absent = sorted(required - set(metadata.columns))
    if absent:
        raise SplitError(
            "Required metadata column(s) are absent: " + ", ".join(absent)
        )

    id_missing = missing_mask(metadata[id_col])
    if id_missing.any():
        example_rows = metadata.index[id_missing].tolist()[:10]
        raise SplitError(
            f"ID column '{id_col}' contains {int(id_missing.sum())} missing/blank "
            f"value(s), for example input row indices {example_rows}."
        )

    normalized_ids = normalized_text(metadata[id_col])
    duplicate_id_mask = normalized_ids.duplicated(keep=False)
    if duplicate_id_mask.any():
        duplicates = sorted(normalized_ids[duplicate_id_mask].unique().tolist())
        raise SplitError(
            f"ID column '{id_col}' is not unique. Duplicate ID example(s): "
            + ", ".join(duplicates[:20])
        )

    columns_checked_for_missing = sorted(
        set(strata_cols) | {"cohort", "batch", "material", "sex", "age"}
    )
    missing_counts = {
        col: int(missing_mask(metadata[col]).sum())
        for col in columns_checked_for_missing
        if missing_mask(metadata[col]).any()
    }
    if missing_counts:
        detail = ", ".join(f"{key}={value}" for key, value in missing_counts.items())
        if missing_policy == "error":
            raise SplitError(
                "Missing/blank values were found in required analysis columns: "
                f"{detail}. Use --missing-policy warn to continue."
            )
        warnings.append(
            "Missing/blank values were retained and treated as an internal category "
            f"named {MISSING_TOKEN}: {detail}"
        )

    # Keep original values in output but reset to a clean positional index.
    metadata = metadata.reset_index(drop=True)
    return metadata, id_col, warnings


def make_joint_key(metadata: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    normalized = [normalized_text(metadata[col]).to_numpy(dtype=str) for col in columns]
    if len(normalized) == 1:
        return normalized[0]

    # Use a visible separator that is very unlikely to occur in metadata values.
    key = normalized[0].copy()
    for next_values in normalized[1:]:
        key = np.char.add(np.char.add(key, " || "), next_values)
    return key


def build_distribution_specs(metadata: pd.DataFrame) -> List[DistributionSpec]:
    specs: List[DistributionSpec] = []
    for name, columns, weight in FEATURE_CONFIG:
        keys = make_joint_key(metadata, columns)
        labels, codes = np.unique(keys, return_inverse=True)
        counts = np.bincount(codes, minlength=len(labels)).astype(np.int64)
        proportions = counts / counts.sum()
        specs.append(
            DistributionSpec(
                name=name,
                columns=columns,
                weight=weight,
                labels=labels,
                codes=codes.astype(np.int64),
                overall_counts=counts,
                overall_proportions=proportions,
            )
        )
    return specs


def round_half_up(value: float) -> int:
    """Round positive values with .5 going upward, unlike Python banker's round."""
    return int(math.floor(value + 0.5))


def stratum_train_size(n: int, train_ratio: float) -> int:
    if n <= 0:
        raise SplitError(f"Invalid stratum size: {n}")
    if n == 1:
        return 1
    proposed = round_half_up(n * train_ratio)
    return min(max(proposed, 1), n - 1)


def prepare_strata_groups(
    metadata: pd.DataFrame,
    strata_cols: Sequence[str],
    id_col: str,
) -> Tuple[List[Tuple[str, np.ndarray]], pd.DataFrame]:
    keys = make_joint_key(metadata, strata_cols)
    ids = normalized_text(metadata[id_col]).to_numpy(dtype=str)

    groups: List[Tuple[str, np.ndarray]] = []
    rows: List[Dict[str, object]] = []
    for key in sorted(np.unique(keys).tolist()):
        indices = np.flatnonzero(keys == key)
        # Sorting by ID makes results reproducible even if input row order changes.
        order = np.argsort(ids[indices], kind="stable")
        indices = indices[order].astype(np.int64)
        groups.append((key, indices))

        row: Dict[str, object] = {
            "stratum": key.replace(" || ", " × "),
            "total_n": int(len(indices)),
        }
        for col in strata_cols:
            # All rows in a group have the same normalized value for this column.
            row[col] = normalized_text(metadata.loc[indices, col]).iloc[0]
        rows.append(row)

    return groups, pd.DataFrame(rows)


def generate_split_indices(
    n_rows: int,
    groups: Sequence[Tuple[str, np.ndarray]],
    train_ratio: float,
    seed: int,
    singleton_policy: str,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    is_train = np.zeros(n_rows, dtype=bool)

    for _, indices in groups:
        n = len(indices)
        if n == 1:
            if singleton_policy == "train":
                put_in_train = True
            elif singleton_policy == "test":
                put_in_train = False
            else:
                put_in_train = bool(rng.random() < train_ratio)
            if put_in_train:
                is_train[indices[0]] = True
            continue

        n_train = stratum_train_size(n, train_ratio)
        permuted = rng.permutation(indices)
        is_train[permuted[:n_train]] = True

    train_indices = np.flatnonzero(is_train).astype(np.int64)
    test_indices = np.flatnonzero(~is_train).astype(np.int64)
    return train_indices, test_indices


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def score_split(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    specs: Sequence[DistributionSpec],
) -> Dict[str, float]:
    n_train = len(train_indices)
    n_test = len(test_indices)
    if n_train == 0 or n_test == 0:
        # This should only occur in pathological all-singleton inputs with an
        # extreme singleton policy. Return an unmistakably bad score.
        return {
            "n_train": float(n_train),
            "n_test": float(n_test),
            "actual_train_ratio": float(n_train / max(n_train + n_test, 1)),
            "base_score": float("inf"),
            "missing_test_category_penalty": float("inf"),
            "missing_nonrare_cohort_batch_penalty": float("inf"),
            "total_penalty": float("inf"),
            "score": float("inf"),
        }

    metrics: Dict[str, float] = {
        "n_train": float(n_train),
        "n_test": float(n_test),
        "actual_train_ratio": float(n_train / (n_train + n_test)),
    }

    weighted_sum = 0.0
    total_weight = 0.0
    missing_test_category_penalty = 0.0
    missing_nonrare_cb_penalty = 0.0

    for spec in specs:
        k = len(spec.labels)
        train_counts = np.bincount(spec.codes[train_indices], minlength=k)
        test_counts = np.bincount(spec.codes[test_indices], minlength=k)
        train_p = train_counts / n_train
        test_p = test_counts / n_test

        # Percentage-point-like deviation: average total-variation distance of
        # train vs full and test vs full, multiplied by 100.
        tv_train = total_variation_distance(train_p, spec.overall_proportions)
        tv_test = total_variation_distance(test_p, spec.overall_proportions)
        deviation = 100.0 * 0.5 * (tv_train + tv_test)

        metrics[f"dev_{spec.name}"] = deviation
        metrics[f"tv_train_{spec.name}"] = 100.0 * tv_train
        metrics[f"tv_test_{spec.name}"] = 100.0 * tv_test
        weighted_sum += spec.weight * deviation
        total_weight += spec.weight

        if spec.name in TEST_MISSING_CATEGORY_PENALTIES:
            missing_count = int(np.sum((spec.overall_counts > 0) & (test_counts == 0)))
            metrics[f"missing_test_categories_{spec.name}"] = float(missing_count)
            missing_test_category_penalty += (
                missing_count * TEST_MISSING_CATEGORY_PENALTIES[spec.name]
            )

        if spec.name == "cohort_x_batch":
            missing_nonrare = int(
                np.sum((spec.overall_counts >= 2) & (test_counts == 0))
            )
            missing_singleton = int(
                np.sum((spec.overall_counts == 1) & (test_counts == 0))
            )
            metrics["missing_test_nonrare_cohort_x_batch"] = float(missing_nonrare)
            metrics["missing_test_singleton_cohort_x_batch"] = float(
                missing_singleton
            )
            missing_nonrare_cb_penalty += (
                missing_nonrare * NONRARE_COHORT_BATCH_MISSING_PENALTY
            )

    base_score = weighted_sum / total_weight
    total_penalty = missing_test_category_penalty + missing_nonrare_cb_penalty
    metrics["base_score"] = base_score
    metrics["missing_test_category_penalty"] = missing_test_category_penalty
    metrics[
        "missing_nonrare_cohort_batch_penalty"
    ] = missing_nonrare_cb_penalty
    metrics["total_penalty"] = total_penalty
    metrics["score"] = base_score + total_penalty
    return metrics


def evaluate_all_seeds(
    metadata: pd.DataFrame,
    groups: Sequence[Tuple[str, np.ndarray]],
    specs: Sequence[DistributionSpec],
    train_ratio: float,
    seed_start: int,
    seed_end: int,
    singleton_policy: str,
) -> Tuple[SplitResult, pd.DataFrame]:
    rows: List[Dict[str, float]] = []
    best: Optional[SplitResult] = None
    best_key: Optional[Tuple[float, float, int]] = None

    for seed in range(seed_start, seed_end + 1):
        train_indices, test_indices = generate_split_indices(
            n_rows=len(metadata),
            groups=groups,
            train_ratio=train_ratio,
            seed=seed,
            singleton_policy=singleton_policy,
        )
        metrics = score_split(train_indices, test_indices, specs)
        row: Dict[str, float] = {"seed": float(seed), **metrics}
        rows.append(row)

        # Primary criterion: score. Secondary criterion: closeness to target
        # global ratio. Final deterministic tie-break: smaller seed.
        key = (
            float(metrics["score"]),
            abs(float(metrics["actual_train_ratio"]) - train_ratio),
            seed,
        )
        if best_key is None or key < best_key:
            best_key = key
            best = SplitResult(
                seed=seed,
                train_indices=train_indices,
                test_indices=test_indices,
                metrics=metrics,
            )

    if best is None:
        raise SplitError("No seed was evaluated.")

    scores = pd.DataFrame(rows)
    scores["seed"] = scores["seed"].astype(int)
    scores["n_train"] = scores["n_train"].astype(int)
    scores["n_test"] = scores["n_test"].astype(int)
    scores["rank"] = scores["score"].rank(method="min", ascending=True).astype(int)
    scores["is_selected_best"] = scores["seed"].eq(best.seed)
    return best, scores


def split_integrity_checks(
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    id_col: str,
) -> Dict[str, object]:
    train = metadata.iloc[train_indices]
    test = metadata.iloc[test_indices]
    train_ids = set(normalized_text(train[id_col]).tolist())
    test_ids = set(normalized_text(test[id_col]).tolist())
    all_ids = set(normalized_text(metadata[id_col]).tolist())

    overlap = sorted(train_ids & test_ids)
    union = train_ids | test_ids
    return {
        "train_duplicate_ids": int(normalized_text(train[id_col]).duplicated().sum()),
        "test_duplicate_ids": int(normalized_text(test[id_col]).duplicated().sum()),
        "train_test_overlap_count": len(overlap),
        "train_test_overlap_examples": overlap[:20],
        "coverage_matches_input": union == all_ids,
        "row_count_matches_input": len(train_indices) + len(test_indices) == len(metadata),
        "covered_unique_ids": len(union),
        "input_unique_ids": len(all_ids),
    }


def patient_overlap_check(
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> Optional[Dict[str, object]]:
    if "patient" not in metadata.columns:
        return None
    train_patients = set(normalized_text(metadata.iloc[train_indices]["patient"]).tolist())
    test_patients = set(normalized_text(metadata.iloc[test_indices]["patient"]).tolist())
    train_patients.discard(MISSING_TOKEN)
    test_patients.discard(MISSING_TOKEN)
    overlap = sorted(train_patients & test_patients)
    return {
        "overlap_count": len(overlap),
        "overlap_examples": overlap[:20],
        "input_duplicate_patient_rows": int(
            normalized_text(metadata["patient"])
            .replace(MISSING_TOKEN, pd.NA)
            .duplicated(keep=False)
            .sum()
        ),
    }


def category_label_from_key(key: str) -> str:
    return key.replace(" || ", " × ").replace(MISSING_TOKEN, "<MISSING>")


def distribution_table(
    overall: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    overall_keys = make_joint_key(overall, columns)
    train_keys = make_joint_key(train, columns)
    test_keys = make_joint_key(test, columns)

    overall_counts = pd.Series(overall_keys).value_counts(sort=False)
    train_counts = pd.Series(train_keys).value_counts(sort=False)
    test_counts = pd.Series(test_keys).value_counts(sort=False)

    categories = sorted(
        overall_counts.index.tolist(),
        key=lambda key: (-int(overall_counts.get(key, 0)), str(key)),
    )
    rows: List[Dict[str, object]] = []
    for category in categories:
        overall_n = int(overall_counts.get(category, 0))
        train_n = int(train_counts.get(category, 0))
        test_n = int(test_counts.get(category, 0))
        rows.append(
            {
                "category": category_label_from_key(str(category)),
                "overall_n": overall_n,
                "overall_pct": f"{100.0 * overall_n / len(overall):.2f}%",
                "train_n": train_n,
                "train_pct": f"{100.0 * train_n / len(train):.2f}%",
                "test_n": test_n,
                "test_pct": f"{100.0 * test_n / len(test):.2f}%",
            }
        )
    return pd.DataFrame(rows)


def escape_markdown_cell(value: object) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def dataframe_to_markdown(table: pd.DataFrame) -> str:
    if table.empty:
        return "_No rows._"
    headers = [escape_markdown_cell(col) for col in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(escape_markdown_cell(x) for x in row) + " |")
    return "\n".join(lines)


def build_strata_status_table(
    metadata: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    strata_cols: Sequence[str],
) -> pd.DataFrame:
    overall_keys = make_joint_key(metadata, strata_cols)
    train_keys = make_joint_key(train, strata_cols)
    test_keys = make_joint_key(test, strata_cols)

    overall_counts = pd.Series(overall_keys).value_counts(sort=False)
    train_counts = pd.Series(train_keys).value_counts(sort=False)
    test_counts = pd.Series(test_keys).value_counts(sort=False)

    rows: List[Dict[str, object]] = []
    for key in sorted(overall_counts.index.tolist()):
        total_n = int(overall_counts.get(key, 0))
        train_n = int(train_counts.get(key, 0))
        test_n = int(test_counts.get(key, 0))
        warning = ""
        if total_n == 1:
            warning = "singleton: impossible to place in both train and test"
        elif train_n == 0 or test_n == 0:
            warning = "missing from one split despite total_n >= 2"
        elif test_n == 1:
            warning = "test contains only 1 sample; acceptable but fragile"
        rows.append(
            {
                "stratum": category_label_from_key(str(key)),
                "total_n": total_n,
                "train_n": train_n,
                "test_n": test_n,
                "warning": warning,
            }
        )
    return pd.DataFrame(rows)


def format_bool(value: object) -> str:
    return "PASS" if bool(value) else "FAIL"


def build_summary(
    metadata_path: Path,
    output_paths: Mapping[str, Path],
    metadata: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    best: SplitResult,
    all_scores: pd.DataFrame,
    id_col: str,
    strata_cols: Sequence[str],
    train_ratio: float,
    seed_start: int,
    seed_end: int,
    missing_policy: str,
    singleton_policy: str,
    initial_warnings: Sequence[str],
    integrity: Mapping[str, object],
    patient_check: Optional[Mapping[str, object]],
) -> str:
    selected_score = float(best.metrics["score"])
    tied = int(np.isclose(all_scores["score"], selected_score, rtol=0, atol=1e-12).sum())
    strata_status = build_strata_status_table(metadata, train, test, strata_cols)
    strata_warnings = strata_status[strata_status["warning"].ne("")]
    missing_test_strata = strata_status[strata_status["test_n"].eq(0)]

    warnings = list(initial_warnings)
    if not missing_test_strata.empty:
        warnings.append(
            f"{len(missing_test_strata)} stratification stratum/strata are absent "
            "from the test set; see the stratum-status table."
        )
    if patient_check and int(patient_check["overlap_count"]) > 0:
        warnings.append(
            f"Patient IDs overlap between train and test: "
            f"{patient_check['overlap_count']}. The split unit is sample ID, not patient ID."
        )

    lines: List[str] = []
    lines.append("# Train/Test Split Summary")
    lines.append("")
    lines.append("## Run configuration")
    lines.append("")
    lines.append(f"- Input metadata: `{metadata_path}`")
    lines.append(f"- ID column: `{id_col}`")
    lines.append(f"- Stratification columns: `{', '.join(strata_cols)}`")
    lines.append(f"- Target train ratio: `{train_ratio:.4f}`")
    lines.append(f"- Seed range: `{seed_start}` to `{seed_end}` (inclusive)")
    lines.append(f"- Missing-value policy: `{missing_policy}`")
    lines.append(f"- Singleton-stratum policy: `{singleton_policy}`")
    lines.append("")
    lines.append("## Selected split")
    lines.append("")
    lines.append(f"- Input samples: **{len(metadata)}**")
    lines.append(f"- Best seed: **{best.seed}**")
    lines.append(f"- Best total score: **{selected_score:.10f}**")
    lines.append(f"- Base distribution score: **{best.metrics['base_score']:.10f}**")
    lines.append(f"- Total missing-category penalty: **{best.metrics['total_penalty']:.10f}**")
    lines.append(f"- Train samples: **{len(train)}** ({100.0 * len(train) / len(metadata):.2f}%)")
    lines.append(f"- Test samples: **{len(test)}** ({100.0 * len(test) / len(metadata):.2f}%)")
    lines.append(f"- Number of seeds tied at the minimum score: **{tied}**")
    if tied > 1:
        lines.append("- Tie rule: closest global train ratio, then the smallest seed.")
    lines.append("")
    lines.append("### Output files")
    lines.append("")
    for label, path in output_paths.items():
        lines.append(f"- {label}: `{path}`")

    lines.append("")
    lines.append("## Score definition")
    lines.append("")
    lines.append(
        "For each distribution, the script calculates total-variation distance "
        "between train and overall, and between test and overall. The component "
        "deviation is `100 × mean(TV_train, TV_test)`. The base score is the "
        "weighted mean of component deviations; explicit missing-category penalties "
        "are then added. Lower is better."
    )
    weight_rows = []
    for name, columns, weight in FEATURE_CONFIG:
        weight_rows.append(
            {
                "component": name,
                "columns": " × ".join(columns),
                "weight": weight,
                "best_deviation": f"{best.metrics[f'dev_{name}']:.6f}",
            }
        )
    lines.append("")
    lines.append(dataframe_to_markdown(pd.DataFrame(weight_rows)))
    lines.append("")
    lines.append("Test missing-category penalties per absent category:")
    lines.append("")
    penalty_rows = [
        {"component": key, "penalty_per_missing_category": value}
        for key, value in TEST_MISSING_CATEGORY_PENALTIES.items()
    ]
    penalty_rows.append(
        {
            "component": "cohort × batch with overall n >= 2",
            "penalty_per_missing_category": NONRARE_COHORT_BATCH_MISSING_PENALTY,
        }
    )
    lines.append(dataframe_to_markdown(pd.DataFrame(penalty_rows)))

    lines.append("")
    lines.append("## Integrity checks")
    lines.append("")
    check_rows = [
        {
            "check": "Duplicate sample IDs within train",
            "result": "PASS" if int(integrity["train_duplicate_ids"]) == 0 else "FAIL",
            "detail": integrity["train_duplicate_ids"],
        },
        {
            "check": "Duplicate sample IDs within test",
            "result": "PASS" if int(integrity["test_duplicate_ids"]) == 0 else "FAIL",
            "detail": integrity["test_duplicate_ids"],
        },
        {
            "check": "Sample ID overlap between train and test",
            "result": "PASS" if int(integrity["train_test_overlap_count"]) == 0 else "FAIL",
            "detail": integrity["train_test_overlap_count"],
        },
        {
            "check": "train + test row count equals input",
            "result": format_bool(integrity["row_count_matches_input"]),
            "detail": f"{len(train)} + {len(test)} = {len(metadata)}",
        },
        {
            "check": "train/test unique-ID union covers all input IDs",
            "result": format_bool(integrity["coverage_matches_input"]),
            "detail": f"{integrity['covered_unique_ids']} / {integrity['input_unique_ids']}",
        },
    ]
    lines.append(dataframe_to_markdown(pd.DataFrame(check_rows)))

    if patient_check is not None:
        lines.append("")
        lines.append("### Patient-level leakage check (informational)")
        lines.append("")
        lines.append(
            f"- Patient IDs appearing in both train and test: "
            f"**{patient_check['overlap_count']}**"
        )
        if patient_check["overlap_examples"]:
            lines.append(
                "- Overlap examples: `"
                + "`, `".join(map(str, patient_check["overlap_examples"]))
                + "`"
            )
        lines.append(
            "- Note: the requested split unit is the sample ID. If one patient has "
            "multiple samples, patient-grouped splitting should be used instead."
        )

    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    if warnings:
        for warning in warnings:
            lines.append(f"- WARNING: {warning}")
    else:
        lines.append("- No run-level warnings.")

    lines.append("")
    lines.append("## Stratification-stratum status")
    lines.append("")
    lines.append(
        "Singleton strata cannot be represented in both sets. A test count of 1 is "
        "allowed and is shown explicitly."
    )
    lines.append("")
    lines.append(dataframe_to_markdown(strata_status))

    table_sections = [
        ("Cohort distribution", ("cohort",)),
        ("Batch distribution", ("batch",)),
        ("Material distribution", ("material",)),
        ("Sex distribution", ("sex",)),
        ("Cohort × batch distribution", ("cohort", "batch")),
        ("Cohort × material distribution", ("cohort", "material")),
        ("Material × batch distribution", ("material", "batch")),
        ("Cohort × sex distribution", ("cohort", "sex")),
    ]
    for title, columns in table_sections:
        lines.append("")
        lines.append(f"## {title}")
        lines.append("")
        lines.append(
            dataframe_to_markdown(distribution_table(metadata, train, test, columns))
        )

    # Age is checked for missingness but not scored, as requested.
    age_rows = []
    for name, frame in (("overall", metadata), ("train", train), ("test", test)):
        numeric_age = pd.to_numeric(frame["age"], errors="coerce")
        age_rows.append(
            {
                "set": name,
                "n": len(frame),
                "nonmissing_numeric_age": int(numeric_age.notna().sum()),
                "mean": f"{numeric_age.mean():.3f}" if numeric_age.notna().any() else "NA",
                "median": f"{numeric_age.median():.3f}" if numeric_age.notna().any() else "NA",
                "min": f"{numeric_age.min():.3f}" if numeric_age.notna().any() else "NA",
                "max": f"{numeric_age.max():.3f}" if numeric_age.notna().any() else "NA",
            }
        )
    lines.append("")
    lines.append("## Age descriptive statistics (not included in score)")
    lines.append("")
    lines.append(dataframe_to_markdown(pd.DataFrame(age_rows)))

    lines.append("")
    lines.append("## Seed search diagnostics")
    lines.append("")
    top_columns = [
        "seed",
        "score",
        "base_score",
        "total_penalty",
        "actual_train_ratio",
    ]
    top = all_scores.sort_values(
        ["score", "actual_train_ratio", "seed"], ascending=[True, False, True]
    ).head(20)[top_columns]
    top = top.copy()
    for col in ("score", "base_score", "total_penalty", "actual_train_ratio"):
        top[col] = top[col].map(lambda value: f"{value:.10f}")
    lines.append("Top 20 candidate seeds:")
    lines.append("")
    lines.append(dataframe_to_markdown(top))

    lines.append("")
    return "\n".join(lines)


def ensure_output_paths(
    output_dir: Path,
    train_ratio: float,
    overwrite: bool,
) -> Dict[str, Path]:
    train_tag = ratio_tag(train_ratio)
    test_tag = ratio_tag(1.0 - train_ratio)
    paths = {
        "train metadata": output_dir / "train" / f"metadata_train_{train_tag}.csv",
        "test metadata": output_dir / "test" / f"metadata_test_{test_tag}.csv",
        "summary": output_dir / "train_test_split_summary.md",
        "all seed scores": output_dir / "all_seed_split_scores.csv",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise SplitError(
            "Output file(s) already exist. Use --overwrite to replace them:\n"
            + formatted
        )
    return paths


def validate_arguments(args: argparse.Namespace) -> List[str]:
    if not (0.0 < args.train_ratio < 1.0):
        raise SplitError("--train-ratio must be strictly between 0 and 1.")
    if args.seed_start < 0 or args.seed_end < 0:
        raise SplitError("Seeds must be non-negative integers.")
    if args.seed_end < args.seed_start:
        raise SplitError("--seed-end must be greater than or equal to --seed-start.")
    return parse_column_list(args.strata_cols)


def main() -> int:
    args = parse_args()
    try:
        strata_cols = validate_arguments(args)
        output_paths = ensure_output_paths(
            args.output_dir, args.train_ratio, args.overwrite
        )
        metadata, id_col, initial_warnings = read_and_validate_metadata(
            path=args.metadata,
            id_col_requested=args.id_col,
            strata_cols=strata_cols,
            missing_policy=args.missing_policy,
            keep_unnamed_columns=args.keep_unnamed_columns,
        )

        groups, _ = prepare_strata_groups(metadata, strata_cols, id_col)
        specs = build_distribution_specs(metadata)
        best, all_scores = evaluate_all_seeds(
            metadata=metadata,
            groups=groups,
            specs=specs,
            train_ratio=args.train_ratio,
            seed_start=args.seed_start,
            seed_end=args.seed_end,
            singleton_policy=args.singleton_policy,
        )

        train = metadata.iloc[best.train_indices].copy().sort_index()
        test = metadata.iloc[best.test_indices].copy().sort_index()

        integrity = split_integrity_checks(
            metadata, best.train_indices, best.test_indices, id_col
        )
        if (
            int(integrity["train_duplicate_ids"]) > 0
            or int(integrity["test_duplicate_ids"]) > 0
            or int(integrity["train_test_overlap_count"]) > 0
            or not bool(integrity["coverage_matches_input"])
            or not bool(integrity["row_count_matches_input"])
        ):
            raise SplitError(f"Internal split-integrity check failed: {integrity}")

        patient_check = patient_overlap_check(
            metadata, best.train_indices, best.test_indices
        )

        # Create directories only after input validation and seed search succeed.
        output_paths["train metadata"].parent.mkdir(parents=True, exist_ok=True)
        output_paths["test metadata"].parent.mkdir(parents=True, exist_ok=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        train.to_csv(output_paths["train metadata"], index=False)
        test.to_csv(output_paths["test metadata"], index=False)

        # Keep seed order in the tracking CSV. Use high precision for auditing.
        all_scores.sort_values("seed").to_csv(
            output_paths["all seed scores"], index=False, float_format="%.12g"
        )

        summary = build_summary(
            metadata_path=args.metadata,
            output_paths=output_paths,
            metadata=metadata,
            train=train,
            test=test,
            best=best,
            all_scores=all_scores,
            id_col=id_col,
            strata_cols=strata_cols,
            train_ratio=args.train_ratio,
            seed_start=args.seed_start,
            seed_end=args.seed_end,
            missing_policy=args.missing_policy,
            singleton_policy=args.singleton_policy,
            initial_warnings=initial_warnings,
            integrity=integrity,
            patient_check=patient_check,
        )
        output_paths["summary"].write_text(summary, encoding="utf-8")

        print(summary)
        print("\nCompleted successfully.")
        return 0

    except SplitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("ERROR: Interrupted by user.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
