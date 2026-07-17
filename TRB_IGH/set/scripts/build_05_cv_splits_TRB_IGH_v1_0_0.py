#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create one fixed repeated nested-CV split system for paired TRB+IGH modeling.

The split key is the patient ID from the patient-level TRB+IGH step-04 training
base matrix. The same outer and inner partitions are intended to be reused by:

1. TRB-only models;
2. IGH-only models;
3. paired TRB+IGH models.

Only selected metadata/identifier columns from TRAIN step-04 base matrices are
read. The independent test set and all immune-repertoire feature values are
never read or used to optimize the partitions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.0-PAIRED-TRB-IGH"
ANALYSIS_VIEWS = ("TRB", "IGH", "TRB_IGH")
ROOT = Path("/data/users/chenhaisheng/RA-ILD/TRB_IGH")

DEFAULT_JOINT_INPUT = (
    ROOT / "set/train/result/04_final_feature_matrix/04_train_base_feature_matrix.csv"
)
DEFAULT_TRB_INPUT = (
    ROOT
    / "set/train/TRB/result/04_final_feature_matrix/04_train_base_feature_matrix.csv"
)
DEFAULT_IGH_INPUT = (
    ROOT
    / "set/train/IGH/result/04_final_feature_matrix/04_train_base_feature_matrix.csv"
)
DEFAULT_OUTPUT = ROOT / "set/train/result/05_modeling/cv_splits"

BALANCE_WEIGHTS = {
    "batch": 1.0,
    "material": 0.75,
    "sex": 0.5,
}


def parse_csv_list(text: str) -> List[str]:
    values: List[str] = []
    for item in str(text).split(","):
        item = item.strip()
        if item and item not in values:
            values.append(item)
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one patient-level repeated nested-CV split system shared by "
            "TRB-only, IGH-only and paired TRB+IGH models."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--joint-input", default=str(DEFAULT_JOINT_INPUT))
    parser.add_argument("--trb-input", default=str(DEFAULT_TRB_INPUT))
    parser.add_argument("--igh-input", default=str(DEFAULT_IGH_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))

    parser.add_argument("--sample-id-col", default="sample_id")
    parser.add_argument(
        "--patient-col",
        default="patient",
        help=(
            "Patient trace column written to outputs. In the joint matrix, "
            "sample_id is already the patient ID; this column is added as an alias."
        ),
    )
    parser.add_argument("--trb-id-col", default="trb_libraryid")
    parser.add_argument("--igh-id-col", default="igh_libraryid")
    parser.add_argument(
        "--receptor-patient-col",
        default="patient",
        help="Patient column in receptor-specific TRB/IGH step-04 base matrices.",
    )
    parser.add_argument("--label-col", default="cohort")
    parser.add_argument("--balance-cols", default="batch,material,sex")
    parser.add_argument(
        "--trace-cols",
        default="age,sex,material,batch,igh_batch,split_batch",
        help=(
            "Training metadata columns retained in assignment outputs and "
            "cross-checked across joint/TRB/IGH matrices."
        ),
    )

    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=20)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--outer-candidates", type=int, default=300)
    parser.add_argument("--inner-candidates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--expected-samples", type=int, default=118)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Build and validate all partitions in memory without writing outputs.",
    )
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    args.balance_cols = parse_csv_list(args.balance_cols)
    args.trace_cols = parse_csv_list(args.trace_cols)

    if args.outer_folds < 2 or args.inner_folds < 2:
        parser.error("Fold counts must be >= 2.")
    if args.outer_repeats < 1:
        parser.error("--outer-repeats must be >= 1.")
    if args.outer_candidates < 1 or args.inner_candidates < 1:
        parser.error("Candidate counts must be >= 1.")
    if args.expected_samples < 0:
        parser.error("--expected-samples must be >= 0.")
    if not args.balance_cols:
        parser.error("--balance-cols must contain at least one column.")
    return args


def unique_ordered(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derived_seed(base: int, *parts: int) -> int:
    text = ":".join(str(x) for x in (base, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") % (
        2**32 - 1
    )


def output_paths(output_dir: Path) -> Dict[str, Path]:
    return {
        "patient_mapping": output_dir / "05_patient_library_mapping.csv",
        "outer_assignments": output_dir / "05_outer_fold_assignments.csv",
        "outer_tasks": output_dir / "05_outer_tasks_long.csv.gz",
        "inner_assignments": output_dir / "05_inner_fold_assignments.csv.gz",
        "outer_balance": output_dir / "05_outer_fold_balance.csv",
        "inner_balance": output_dir / "05_inner_fold_balance.csv",
        "configuration": output_dir / "05_cv_split_configuration.json",
        "summary": output_dir / "05_cv_split_summary.md",
    }


def check_overwrite(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Outputs already exist; add --overwrite:\n"
            + "\n".join(f"  - {path}" for path in existing)
        )


def read_selected_columns(path: Path, columns: Sequence[str], label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")

    header = pd.read_csv(path, nrows=0)
    header_columns = [
        column
        for column in header.columns
        if not str(column).startswith("Unnamed:")
    ]
    missing = [column for column in columns if column not in header_columns]
    if missing:
        raise ValueError(
            f"{label} missing required metadata columns: {missing}; "
            f"available={header_columns[:30]}"
        )

    frame = pd.read_csv(path, usecols=list(columns), dtype=str)
    frame = frame[list(columns)].copy()
    for column in columns:
        frame[column] = frame[column].astype("string").str.strip()
        if frame[column].isna().any() or frame[column].eq("").any():
            raise ValueError(f"{label} contains empty values in {column}")
    return frame


def check_unique(frame: pd.DataFrame, column: str, label: str) -> None:
    if frame[column].duplicated().any():
        values = (
            frame.loc[frame[column].duplicated(keep=False), column]
            .astype(str)
            .unique()
            .tolist()[:10]
        )
        raise ValueError(f"{label} duplicated {column}; examples={values}")


def normalize_label(frame: pd.DataFrame, label_col: str, label: str) -> None:
    frame[label_col] = frame[label_col].astype(str).str.upper()
    observed = set(frame[label_col].unique())
    if observed != {"RA", "ILD"}:
        raise ValueError(
            f"{label} expected cohort labels RA/ILD only; observed={sorted(observed)}"
        )


def compare_columns_exact(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: Sequence[str],
    left_name: str,
    right_name: str,
) -> None:
    for column in columns:
        mismatch = left[column].astype(str).to_numpy() != right[column].astype(str).to_numpy()
        if np.any(mismatch):
            index = int(np.flatnonzero(mismatch)[0])
            raise RuntimeError(
                f"Metadata mismatch for {column}: {left_name}={left.iloc[index][column]!r}, "
                f"{right_name}={right.iloc[index][column]!r}, row={index}"
            )


def load_and_validate_training_mapping(
    args: argparse.Namespace,
    joint_path: Path,
    trb_path: Path,
    igh_path: Path,
) -> pd.DataFrame:
    shared_metadata = unique_ordered(
        [args.label_col, *args.trace_cols, *args.balance_cols]
    )
    joint_columns = unique_ordered(
        [
            args.sample_id_col,
            args.trb_id_col,
            args.igh_id_col,
            *shared_metadata,
        ]
    )
    receptor_columns = unique_ordered(
        [
            args.sample_id_col,
            args.receptor_patient_col,
            *shared_metadata,
        ]
    )

    joint = read_selected_columns(joint_path, joint_columns, "joint train base")
    trb = read_selected_columns(trb_path, receptor_columns, "TRB train base")
    igh = read_selected_columns(igh_path, receptor_columns, "IGH train base")

    if args.expected_samples > 0:
        for label, frame in [("joint", joint), ("TRB", trb), ("IGH", igh)]:
            if len(frame) != args.expected_samples:
                raise ValueError(
                    f"{label} training rows expected {args.expected_samples}; got {len(frame)}"
                )

    for column in [args.sample_id_col, args.trb_id_col, args.igh_id_col]:
        check_unique(joint, column, "joint train base")
    check_unique(trb, args.sample_id_col, "TRB train base")
    check_unique(trb, args.receptor_patient_col, "TRB train base")
    check_unique(igh, args.sample_id_col, "IGH train base")
    check_unique(igh, args.receptor_patient_col, "IGH train base")

    normalize_label(joint, args.label_col, "joint train base")
    normalize_label(trb, args.label_col, "TRB train base")
    normalize_label(igh, args.label_col, "IGH train base")

    trb_map = trb.rename(
        columns={
            args.sample_id_col: args.trb_id_col,
            args.receptor_patient_col: args.sample_id_col,
        }
    )
    igh_map = igh.rename(
        columns={
            args.sample_id_col: args.igh_id_col,
            args.receptor_patient_col: args.sample_id_col,
        }
    )

    joint_patients = set(joint[args.sample_id_col])
    if set(trb_map[args.sample_id_col]) != joint_patients:
        raise RuntimeError("TRB patient IDs do not exactly match joint patient IDs.")
    if set(igh_map[args.sample_id_col]) != joint_patients:
        raise RuntimeError("IGH patient IDs do not exactly match joint patient IDs.")

    trb_map = joint[[args.sample_id_col]].merge(
        trb_map,
        on=args.sample_id_col,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    igh_map = joint[[args.sample_id_col]].merge(
        igh_map,
        on=args.sample_id_col,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if joint[args.trb_id_col].tolist() != trb_map[args.trb_id_col].tolist():
        raise RuntimeError("Joint trb_libraryid mapping disagrees with TRB train base.")
    if joint[args.igh_id_col].tolist() != igh_map[args.igh_id_col].tolist():
        raise RuntimeError("Joint igh_libraryid mapping disagrees with IGH train base.")

    compare_columns_exact(
        joint,
        trb_map,
        shared_metadata,
        "joint",
        "TRB",
    )
    compare_columns_exact(
        joint,
        igh_map,
        shared_metadata,
        "joint",
        "IGH",
    )

    joint.insert(1, args.patient_col, joint[args.sample_id_col].astype(str))
    joint.insert(0, "row_index", np.arange(len(joint), dtype=int))
    return joint


def stratified_candidate(
    labels: np.ndarray,
    folds_n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Hard-stratify cohort and keep total fold sizes within one sample."""
    assignments = np.full(len(labels), -1, dtype=np.int16)
    current_sizes = np.zeros(folds_n, dtype=int)

    label_order = sorted(
        np.unique(labels),
        key=lambda value: (-int(np.sum(labels == value)), str(value)),
    )
    for label in label_order:
        indices = rng.permutation(np.flatnonzero(labels == label))
        base, remainder = divmod(len(indices), folds_n)
        counts = np.full(folds_n, base, dtype=int)

        if remainder:
            jitter = rng.random(folds_n) * 1e-6
            chosen = np.argsort(current_sizes + jitter)[:remainder]
            counts[chosen] += 1

        fold_ids = np.concatenate(
            [
                np.full(counts[fold], fold + 1, dtype=np.int16)
                for fold in range(folds_n)
            ]
        )
        assignments[indices] = rng.permutation(fold_ids)
        current_sizes += counts

    if np.any(assignments < 1):
        raise RuntimeError("Incomplete fold assignment.")
    if current_sizes.max() - current_sizes.min() > 1:
        raise RuntimeError("Unable to balance total fold sizes within one sample.")
    return assignments


def category_score(values: np.ndarray, folds: np.ndarray, folds_n: int) -> float:
    score = 0.0
    for category in sorted(np.unique(values)):
        mask = values == category
        expected = mask.sum() / folds_n
        observed = np.array(
            [(mask & (folds == fold)).sum() for fold in range(1, folds_n + 1)],
            dtype=float,
        )
        score += float(
            np.sum(((observed - expected) / max(expected, 1.0)) ** 2)
        )
    return score


def split_score(
    frame: pd.DataFrame,
    folds: np.ndarray,
    args: argparse.Namespace,
    folds_n: int,
) -> float:
    expected_size = len(frame) / folds_n
    sizes = np.array(
        [(folds == fold).sum() for fold in range(1, folds_n + 1)],
        dtype=float,
    )
    score = 0.25 * float(
        np.sum(((sizes - expected_size) / max(expected_size, 1.0)) ** 2)
    )

    labels = frame[args.label_col].astype(str)
    for column in args.balance_cols:
        weight = BALANCE_WEIGHTS.get(column, 0.5)
        values = frame[column].astype(str).to_numpy()
        score += weight * category_score(values, folds, folds_n)

        joint_values = (labels + "|" + frame[column].astype(str)).to_numpy()
        score += 1.5 * weight * category_score(joint_values, folds, folds_n)
    return score


def partition_signature(ids: Sequence[str], folds: np.ndarray) -> str:
    text = "\n".join(
        f"{sample_id}\t{int(fold)}"
        for sample_id, fold in sorted(zip(ids, folds))
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def optimize_split(
    frame: pd.DataFrame,
    folds_n: int,
    candidates: int,
    base_seed: int,
    args: argparse.Namespace,
    forbidden: Optional[set[str]] = None,
) -> Tuple[np.ndarray, float, int, str]:
    labels = frame[args.label_col].to_numpy()
    ids = frame[args.sample_id_col].astype(str).tolist()
    best = None
    fallback = None

    for candidate in range(1, candidates + 1):
        seed = derived_seed(base_seed, candidate)
        folds = stratified_candidate(labels, folds_n, np.random.default_rng(seed))
        score = split_score(frame, folds, args, folds_n)
        signature = partition_signature(ids, folds)
        record = (score, seed, folds.copy(), signature)

        if fallback is None or score < fallback[0]:
            fallback = record
        if forbidden and signature in forbidden:
            continue
        if best is None or score < best[0]:
            best = record

    chosen = best or fallback
    if chosen is None:
        raise RuntimeError("No candidate split generated.")
    score, seed, folds, signature = chosen
    return folds, float(score), int(seed), str(signature)


def balance_table(
    frame: pd.DataFrame,
    folds: np.ndarray,
    folds_n: int,
    variables: Sequence[str],
    level: str,
    repeat: int,
    outer_fold: Optional[int] = None,
) -> pd.DataFrame:
    rows: List[dict] = []
    for fold in range(1, folds_n + 1):
        subset = frame.loc[folds == fold]
        base = {
            "level": level,
            "outer_repeat": repeat,
            "outer_fold": "" if outer_fold is None else outer_fold,
            "fold": fold,
            "fold_size": len(subset),
        }
        rows.append(
            {
                **base,
                "variable": "__fold__",
                "category": "all",
                "count": len(subset),
                "proportion_within_fold": 1.0,
            }
        )
        for variable in variables:
            categories = sorted(frame[variable].astype(str).unique())
            counts = subset[variable].astype(str).value_counts()
            for category in categories:
                count = int(counts.get(category, 0))
                rows.append(
                    {
                        **base,
                        "variable": variable,
                        "category": category,
                        "count": count,
                        "proportion_within_fold": (
                            count / len(subset) if len(subset) else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def validate_outer(
    metadata: pd.DataFrame,
    outer: pd.DataFrame,
    args: argparse.Namespace,
) -> List[str]:
    expected_rows = len(metadata) * args.outer_repeats
    if len(outer) != expected_rows:
        raise RuntimeError(
            f"Outer rows expected {expected_rows}; observed {len(outer)}"
        )

    signatures: List[str] = []
    for repeat in range(1, args.outer_repeats + 1):
        current = outer[outer["outer_repeat"] == repeat]
        if len(current) != len(metadata):
            raise RuntimeError(f"Outer repeat {repeat} has incorrect sample count.")
        if current[args.sample_id_col].duplicated().any():
            raise RuntimeError(f"Outer repeat {repeat} contains duplicate patients.")
        if set(current["outer_fold"]) != set(range(1, args.outer_folds + 1)):
            raise RuntimeError(f"Outer repeat {repeat} is missing a fold.")

        fold_sizes = current["outer_fold"].value_counts()
        if fold_sizes.max() - fold_sizes.min() > 1:
            raise RuntimeError(f"Outer fold-size imbalance >1 in repeat {repeat}.")

        signatures.append(
            partition_signature(
                current[args.sample_id_col].astype(str),
                current["outer_fold"].to_numpy(),
            )
        )
        for label in ["RA", "ILD"]:
            counts = (
                current.loc[current[args.label_col] == label, "outer_fold"]
                .value_counts()
                .reindex(range(1, args.outer_folds + 1), fill_value=0)
            )
            if counts.max() - counts.min() > 1:
                raise RuntimeError(
                    f"Outer cohort imbalance >1: repeat={repeat}, cohort={label}"
                )
            if (counts == 0).any():
                raise RuntimeError(
                    f"Outer fold without {label}: repeat={repeat}, counts={counts.tolist()}"
                )

    if len(set(signatures)) != args.outer_repeats:
        raise RuntimeError("Duplicate outer partitions detected.")

    return [
        "Every outer repeat contains each training patient exactly once.",
        "All outer-repeat partitions are unique.",
        "Outer fold sizes differ by at most one within every repeat.",
        "RA and ILD counts differ by at most one across outer folds.",
        "Every outer validation fold contains both RA and ILD patients.",
    ]


def validate_inner(
    metadata: pd.DataFrame,
    outer: pd.DataFrame,
    inner: pd.DataFrame,
    args: argparse.Namespace,
) -> List[str]:
    expected_rows = (
        len(metadata) * args.outer_repeats * (args.outer_folds - 1)
    )
    if len(inner) != expected_rows:
        raise RuntimeError(
            f"Inner rows expected {expected_rows}; observed {len(inner)}"
        )

    for repeat in range(1, args.outer_repeats + 1):
        outer_repeat = outer[outer["outer_repeat"] == repeat]
        for task_fold in range(1, args.outer_folds + 1):
            training_ids = set(
                outer_repeat.loc[
                    outer_repeat["outer_fold"] != task_fold,
                    args.sample_id_col,
                ]
            )
            validation_ids = set(
                outer_repeat.loc[
                    outer_repeat["outer_fold"] == task_fold,
                    args.sample_id_col,
                ]
            )
            current = inner[
                (inner["outer_repeat"] == repeat)
                & (inner["outer_fold"] == task_fold)
            ]

            if set(current[args.sample_id_col]) != training_ids:
                raise RuntimeError(
                    f"Inner IDs mismatch: repeat={repeat}, outer_fold={task_fold}"
                )
            if set(current[args.sample_id_col]) & validation_ids:
                raise RuntimeError(
                    f"Outer-validation leakage: repeat={repeat}, outer_fold={task_fold}"
                )
            if current[args.sample_id_col].duplicated().any():
                raise RuntimeError(
                    f"Duplicate inner patient: repeat={repeat}, outer_fold={task_fold}"
                )
            if set(current["inner_fold"]) != set(
                range(1, args.inner_folds + 1)
            ):
                raise RuntimeError(
                    f"Missing inner fold: repeat={repeat}, outer_fold={task_fold}"
                )

            fold_sizes = current["inner_fold"].value_counts()
            if fold_sizes.max() - fold_sizes.min() > 1:
                raise RuntimeError(
                    f"Inner fold-size imbalance >1: repeat={repeat}, "
                    f"outer_fold={task_fold}"
                )

            for label in ["RA", "ILD"]:
                counts = (
                    current.loc[current[args.label_col] == label, "inner_fold"]
                    .value_counts()
                    .reindex(range(1, args.inner_folds + 1), fill_value=0)
                )
                if counts.max() - counts.min() > 1:
                    raise RuntimeError(
                        f"Inner cohort imbalance >1: repeat={repeat}, "
                        f"outer_fold={task_fold}, cohort={label}"
                    )
                if (counts == 0).any():
                    raise RuntimeError(
                        f"Inner fold without {label}: repeat={repeat}, "
                        f"outer_fold={task_fold}, counts={counts.tolist()}"
                    )

    return [
        "Every inner task contains exactly its corresponding outer-training patients.",
        "No outer-validation patient appears in the corresponding inner CV.",
        "Inner fold sizes differ by at most one within every outer task.",
        "RA and ILD counts differ by at most one across inner folds.",
        "Every inner validation fold contains both RA and ILD patients.",
    ]


def write_dataframe_atomic(
    frame: pd.DataFrame,
    path: Path,
    compression: Optional[str] = None,
) -> None:
    if compression == "gzip":
        temporary = path.with_name(path.name + ".tmp.gz")
        frame.to_csv(temporary, index=False, compression="gzip")
    else:
        temporary = path.with_name(path.name + ".tmp")
        frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_text_atomic(text: str, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_summary(
    path: Path,
    input_paths: Mapping[str, Path],
    metadata: pd.DataFrame,
    outer: pd.DataFrame,
    inner: pd.DataFrame,
    args: argparse.Namespace,
    outer_scores: Sequence[float],
    inner_scores: Sequence[float],
    checks: Sequence[str],
    runtime: float,
) -> None:
    outer_sizes = outer.groupby(["outer_repeat", "outer_fold"]).size()
    inner_sizes = inner.groupby(
        ["outer_repeat", "outer_fold", "inner_fold"]
    ).size()
    label_counts = metadata[args.label_col].value_counts()

    lines = [
        "# 05 Paired TRB+IGH Repeated Nested Cross-Validation Split Summary",
        "",
        "## Scope",
        "",
        "- One patient-level split system was generated and frozen.",
        "- The same partitions must be reused by TRB-only, IGH-only, and paired TRB+IGH models.",
        "- Only selected identifiers and metadata columns from TRAIN step-04 base matrices were read.",
        "- The independent test set was not read.",
        "- No TRB or IGH immune-repertoire feature value was read or used to optimize a partition.",
        "- Cohort was hard-stratified; batch, material, and sex were secondary balance targets.",
        "",
        "## Configuration",
        "",
        f"- Script version: `{SCRIPT_VERSION}`",
        f"- Analysis views sharing the split: `{', '.join(ANALYSIS_VIEWS)}`",
        f"- Training patients: **{len(metadata)}**",
        f"- RA: **{int(label_counts.get('RA', 0))}**",
        f"- ILD: **{int(label_counts.get('ILD', 0))}**",
        f"- Outer CV: **{args.outer_folds} folds × {args.outer_repeats} repeats**",
        f"- Outer tasks: **{args.outer_folds * args.outer_repeats}**",
        f"- Inner CV: **{args.inner_folds} folds per outer task**",
        f"- Outer candidate partitions per repeat: **{args.outer_candidates}**",
        f"- Inner candidate partitions per outer task: **{args.inner_candidates}**",
        f"- Seed: **{args.seed}**",
        f"- Secondary balance columns: `{', '.join(args.balance_cols)}`",
        "",
        "## Inputs",
        "",
    ]
    for name, input_path in input_paths.items():
        lines.append(f"- `{name}`: `{input_path}`")

    lines.extend(
        [
            "",
            "## Output dimensions",
            "",
            f"- Patient-library mapping rows: **{len(metadata)}**",
            f"- Outer assignment rows: **{len(outer)}**",
            f"- Inner assignment rows: **{len(inner)}**",
            f"- Outer validation fold sizes: **{outer_sizes.min()}–{outer_sizes.max()}**",
            f"- Inner validation fold sizes: **{inner_sizes.min()}–{inner_sizes.max()}**",
            "",
            "## Optimization scores",
            "",
            f"- Outer median: **{np.median(outer_scores):.6f}**",
            f"- Outer range: **{min(outer_scores):.6f}–{max(outer_scores):.6f}**",
            f"- Inner median: **{np.median(inner_scores):.6f}**",
            f"- Inner range: **{min(inner_scores):.6f}–{max(inner_scores):.6f}**",
            "",
            "## Integrity checks",
            "",
            *[f"- {check}" for check in checks],
            "- Joint patient IDs and TRB/IGH library mappings were cross-checked against all three step-04 training matrices.",
            "",
            "## Use",
            "",
            "- In outer task `(outer_repeat=r, outer_fold=f)`, rows with `outer_fold == f` are outer validation patients; all others are outer training patients.",
            "- Use the matching rows in `05_inner_fold_assignments.csv.gz` only to tune hyperparameters within that outer-training subset.",
            "- Join TRB-only data by `trb_libraryid`, IGH-only data by `igh_libraryid`, and paired data by patient-level `sample_id`.",
            "- All model variants must reuse these exact partitions for fair comparison.",
            "- Dynamic TRB and IGH reference-public sets must be rebuilt inside every relevant training partition.",
            "- The complete-training reference sets are reserved for the final model and independent-test transformation.",
            "",
            f"- Runtime: **{runtime:.2f} seconds**",
            "",
        ]
    )
    write_text_atomic("\n".join(lines), path)


def main() -> int:
    args = parse_args()
    started = time.time()

    joint_path = Path(args.joint_input).expanduser().resolve()
    trb_path = Path(args.trb_input).expanduser().resolve()
    igh_path = Path(args.igh_input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = output_paths(output_dir)

    if not args.preflight_only:
        check_overwrite(outputs.values(), args.overwrite)

    metadata = load_and_validate_training_mapping(
        args=args,
        joint_path=joint_path,
        trb_path=trb_path,
        igh_path=igh_path,
    )

    print("=" * 88)
    print("05 Paired TRB+IGH Repeated Nested Cross-Validation Split Builder")
    print("=" * 88)
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Training patients: {len(metadata)}")
    print(metadata[args.label_col].value_counts().to_string())
    print(f"Outer: {args.outer_folds} folds × {args.outer_repeats} repeats")
    print(f"Inner: {args.inner_folds} folds per outer task")
    print(f"Secondary balance: {args.balance_cols}")
    print("Shared by: TRB-only, IGH-only, paired TRB+IGH")
    print("Independent test set: NOT READ")
    print("Immune-repertoire feature values: NOT READ")
    print("Patient/TRB/IGH mapping cross-check: PASS")
    print()

    outer_parts: List[pd.DataFrame] = []
    outer_task_parts: List[pd.DataFrame] = []
    inner_parts: List[pd.DataFrame] = []
    outer_balance_parts: List[pd.DataFrame] = []
    inner_balance_parts: List[pd.DataFrame] = []
    outer_scores: List[float] = []
    inner_scores: List[float] = []
    outer_meta: List[dict] = []
    inner_meta: List[dict] = []
    used_outer_signatures: set[str] = set()

    for repeat in range(1, args.outer_repeats + 1):
        folds, score, chosen_seed, signature = optimize_split(
            metadata,
            args.outer_folds,
            args.outer_candidates,
            derived_seed(args.seed, 1, repeat),
            args,
            used_outer_signatures,
        )
        used_outer_signatures.add(signature)
        outer_scores.append(score)
        outer_meta.append(
            {
                "outer_repeat": repeat,
                "score": score,
                "chosen_seed": chosen_seed,
                "signature": signature,
            }
        )

        assignment = metadata.copy()
        assignment.insert(1, "outer_repeat", repeat)
        assignment.insert(2, "outer_fold", folds.astype(int))
        outer_parts.append(assignment)
        outer_balance_parts.append(
            balance_table(
                metadata,
                folds,
                args.outer_folds,
                [args.label_col, *args.balance_cols],
                "outer",
                repeat,
            )
        )

        for task_fold in range(1, args.outer_folds + 1):
            task = assignment.copy()
            task["outer_task_fold"] = task_fold
            task["outer_role"] = np.where(
                task["outer_fold"] == task_fold,
                "validation",
                "training",
            )
            outer_task_parts.append(task)

            inner_training = metadata.loc[folds != task_fold].reset_index(drop=True)
            inner_folds, inner_score, inner_seed, inner_signature = optimize_split(
                inner_training,
                args.inner_folds,
                args.inner_candidates,
                derived_seed(args.seed, 2, repeat, task_fold),
                args,
            )
            inner_scores.append(inner_score)
            inner_meta.append(
                {
                    "outer_repeat": repeat,
                    "outer_fold": task_fold,
                    "score": inner_score,
                    "chosen_seed": inner_seed,
                    "signature": inner_signature,
                }
            )

            inner_assignment = inner_training.copy()
            inner_assignment.insert(1, "outer_repeat", repeat)
            inner_assignment.insert(2, "outer_fold", task_fold)
            inner_assignment.insert(3, "inner_fold", inner_folds.astype(int))
            inner_parts.append(inner_assignment)
            inner_balance_parts.append(
                balance_table(
                    inner_training,
                    inner_folds,
                    args.inner_folds,
                    [args.label_col, *args.balance_cols],
                    "inner",
                    repeat,
                    task_fold,
                )
            )

        if repeat == 1 or repeat % 5 == 0 or repeat == args.outer_repeats:
            fold_sizes = [
                int((folds == fold).sum())
                for fold in range(1, args.outer_folds + 1)
            ]
            print(
                f"[repeat {repeat:>2}/{args.outer_repeats}] "
                f"score={score:.6f} | fold sizes={fold_sizes}"
            )

    outer = pd.concat(outer_parts, ignore_index=True)
    outer_tasks = pd.concat(outer_task_parts, ignore_index=True)
    inner = pd.concat(inner_parts, ignore_index=True)
    outer_balance = pd.concat(outer_balance_parts, ignore_index=True)
    inner_balance = pd.concat(inner_balance_parts, ignore_index=True)

    checks = validate_outer(metadata, outer, args) + validate_inner(
        metadata,
        outer,
        inner,
        args,
    )

    trace_columns = unique_ordered(
        [
            args.sample_id_col,
            args.patient_col,
            args.trb_id_col,
            args.igh_id_col,
            args.label_col,
            *args.trace_cols,
            *args.balance_cols,
        ]
    )
    mapping = metadata[[*trace_columns, "row_index"]].copy()

    outer = outer[
        ["outer_repeat", "outer_fold", *trace_columns, "row_index"]
    ].sort_values(
        ["outer_repeat", "outer_fold", args.sample_id_col],
        kind="mergesort",
    )
    outer_tasks = outer_tasks[
        [
            "outer_repeat",
            "outer_task_fold",
            "outer_role",
            "outer_fold",
            *trace_columns,
            "row_index",
        ]
    ].sort_values(
        ["outer_repeat", "outer_task_fold", "outer_role", args.sample_id_col],
        kind="mergesort",
    )
    inner = inner[
        [
            "outer_repeat",
            "outer_fold",
            "inner_fold",
            *trace_columns,
            "row_index",
        ]
    ].sort_values(
        ["outer_repeat", "outer_fold", "inner_fold", args.sample_id_col],
        kind="mergesort",
    )

    runtime = time.time() - started

    input_paths = {
        "joint_train_base": joint_path,
        "trb_train_base": trb_path,
        "igh_train_base": igh_path,
    }
    configuration = {
        "script_version": SCRIPT_VERSION,
        "analysis_views": list(ANALYSIS_VIEWS),
        "split_key": "patient",
        "inputs": {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for name, path in input_paths.items()
        },
        "output_dir": str(output_dir),
        "sample_id_col": args.sample_id_col,
        "patient_col": args.patient_col,
        "trb_id_col": args.trb_id_col,
        "igh_id_col": args.igh_id_col,
        "label_col": args.label_col,
        "balance_cols": args.balance_cols,
        "trace_cols": args.trace_cols,
        "balance_weights": BALANCE_WEIGHTS,
        "expected_samples": args.expected_samples,
        "observed_samples": len(metadata),
        "outer_folds": args.outer_folds,
        "outer_repeats": args.outer_repeats,
        "inner_folds": args.inner_folds,
        "outer_candidates": args.outer_candidates,
        "inner_candidates": args.inner_candidates,
        "seed": args.seed,
        "outer_partitions": outer_meta,
        "inner_partitions": inner_meta,
        "outer_assignment_sha256": hashlib.sha256(
            outer.to_csv(index=False).encode("utf-8")
        ).hexdigest(),
        "inner_assignment_sha256": hashlib.sha256(
            inner.to_csv(index=False).encode("utf-8")
        ).hexdigest(),
        "independent_test_set_read": False,
        "immune_feature_values_read": False,
        "important_note": (
            "All model views must reuse these partitions. Dynamic TRB/IGH public "
            "reference sets must be rebuilt inside each training partition."
        ),
    }

    if args.preflight_only:
        print()
        print("[Preflight-only completed]")
        print(f"Patient mapping rows:  {len(mapping):,}")
        print(f"Outer assignment rows: {len(outer):,}")
        print(f"Outer task rows:       {len(outer_tasks):,}")
        print(f"Inner assignment rows: {len(inner):,}")
        print("Integrity checks: PASS")
        print("No output files were written.")
        print(f"Runtime: {runtime:.2f}s")
        return 0

    write_dataframe_atomic(mapping, outputs["patient_mapping"])
    write_dataframe_atomic(outer, outputs["outer_assignments"])
    write_dataframe_atomic(outer_tasks, outputs["outer_tasks"], compression="gzip")
    write_dataframe_atomic(inner, outputs["inner_assignments"], compression="gzip")
    write_dataframe_atomic(outer_balance, outputs["outer_balance"])
    write_dataframe_atomic(inner_balance, outputs["inner_balance"])
    write_text_atomic(
        json.dumps(configuration, ensure_ascii=False, indent=2) + "\n",
        outputs["configuration"],
    )
    write_summary(
        path=outputs["summary"],
        input_paths=input_paths,
        metadata=metadata,
        outer=outer,
        inner=inner,
        args=args,
        outer_scores=outer_scores,
        inner_scores=inner_scores,
        checks=checks,
        runtime=runtime,
    )

    print()
    print("[Completed]")
    print(f"Patient mapping rows:  {len(mapping):,}")
    print(f"Outer assignment rows: {len(outer):,}")
    print(f"Outer task rows:       {len(outer_tasks):,}")
    print(f"Inner assignment rows: {len(inner):,}")
    print("[Integrity checks]")
    for check in checks:
        print(f"- PASS: {check}")
    print("- PASS: Joint/TRB/IGH patient-library mappings are identical.")
    print("[Output files]")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    print(f"Runtime: {runtime:.2f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
