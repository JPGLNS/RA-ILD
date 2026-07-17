#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build step-04 feature matrices for paired TRB + IGH immune-repertoire data.

Three analysis views are generated:
  1. TRB-only
  2. IGH-only
  3. patient-level paired TRB+IGH

Important leakage rule
----------------------
The complete-training descriptive public features from step 03 are self-inclusive.
They are written only to descriptive matrices and must not be used as fixed
predictors in ordinary cross-validation. Reference-public predictors for training
must be rebuilt inside each training fold. The independent test matrices use the
fixed reference sets generated from the complete training set.

The script always reads and validates both train and test inputs so that schemas,
ID separation, and receptor pairing are checked consistently. --mode controls
which validated outputs are written.
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

import numpy as np
import pandas as pd


VERSION = "1.0.0-PAIRED-TRB-IGH"
ROOT = Path("/data/users/chenhaisheng/RA-ILD/TRB_IGH")

DEFAULTS = {
    "train_metadata": ROOT / "set/train/metadata_train_70.csv",
    "test_metadata": ROOT / "set/test/metadata_test_30.csv",

    "trb_train_02": ROOT / "set/train/TRB/result/02_sample_level_features/02_sample_level_features_merged.csv",
    "trb_test_02": ROOT / "set/test/TRB/result/02_sample_level_features/02_sample_level_features_merged.csv",
    "igh_train_02": ROOT / "set/train/IGH/result/02_sample_level_features/02_sample_level_features_merged.csv",
    "igh_test_02": ROOT / "set/test/IGH/result/02_sample_level_features/02_sample_level_features_merged.csv",

    "trb_train_03_desc": ROOT / "set/train/TRB/result/03_public_features/03_descriptive_public_features.csv",
    "trb_test_03_desc": ROOT / "set/test/TRB/result/03_public_features/03_descriptive_public_features.csv",
    "igh_train_03_desc": ROOT / "set/train/IGH/result/03_public_features/03_descriptive_public_features.csv",
    "igh_test_03_desc": ROOT / "set/test/IGH/result/03_public_features/03_descriptive_public_features.csv",

    "trb_test_03_ref": ROOT / "set/test/TRB/result/03_public_features/03_test_reference_public_features.csv",
    "igh_test_03_ref": ROOT / "set/test/IGH/result/03_public_features/03_test_reference_public_features.csv",
    "trb_test_03_definition": ROOT / "set/test/TRB/result/03_public_features/03_reference_definition_used.json",
    "igh_test_03_definition": ROOT / "set/test/IGH/result/03_public_features/03_reference_definition_used.json",

    "trb_train_out": ROOT / "set/train/TRB/result/04_final_feature_matrix",
    "trb_test_out": ROOT / "set/test/TRB/result/04_final_feature_matrix",
    "igh_train_out": ROOT / "set/train/IGH/result/04_final_feature_matrix",
    "igh_test_out": ROOT / "set/test/IGH/result/04_final_feature_matrix",
    "joint_train_out": ROOT / "set/train/result/04_final_feature_matrix",
    "joint_test_out": ROOT / "set/test/result/04_final_feature_matrix",
}

REFERENCE_DROP = {
    "Y_frequency",
    "weighted_Y_frequency",
    "weighted_length_other_freq",
    "nonpolar_aa_ratio",
    "weighted_nonpolar_aa_ratio",
}

QC_FEATURES = {
    "total_reads_all_nt",
    "total_reads_valid_aa",
    "valid_read_ratio",
    "nt_clone_number_all",
    "valid_nt_clone_number",
    "valid_nt_row_ratio",
    "aa_clone_number",
    "log1p_aa_clone_number",
}

DIVERSITY_FEATURES = {
    "Shannon",
    "Simpson",
    "inverse_Simpson",
    "Gini",
    "clonality",
}

EXPANSION_FEATURES = {
    "top1_frequency",
    "top5_cumulative_frequency",
    "top10_cumulative_frequency",
    "top20_cumulative_frequency",
    "top50_cumulative_frequency",
}

PUBLIC_DYNAMIC_PREFIXES = (
    "all_ref_public_",
    "RA_specific_ref_",
    "ILD_specific_ref_",
    "shared_ref_",
    "ILD_RA_specific_ref_",
)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build TRB-only, IGH-only, and paired TRB+IGH step-04 feature matrices."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both")

    parser.add_argument("--train-metadata", default=str(DEFAULTS["train_metadata"]))
    parser.add_argument("--test-metadata", default=str(DEFAULTS["test_metadata"]))
    parser.add_argument("--patient-col", default="patient")
    parser.add_argument("--trb-id-col", default="trb_libraryid")
    parser.add_argument("--igh-id-col", default="igh_libraryid")
    parser.add_argument("--cohort-col", default="cohort")
    parser.add_argument("--sample-id-col", default="sample_id")
    parser.add_argument(
        "--receptor-metadata-output-cols",
        default="cohort,patient,age,sex,material,batch,igh_batch,split_batch",
        help="Metadata columns retained in each receptor-specific matrix.",
    )
    parser.add_argument(
        "--joint-metadata-output-cols",
        default=(
            "cohort,age,sex,material,batch,igh_batch,split_batch,"
            "trb_libraryid,igh_libraryid"
        ),
        help=(
            "Metadata columns retained after patient is renamed to sample_id in the "
            "paired matrix. Do not include patient itself."
        ),
    )

    for receptor in ("trb", "igh"):
        parser.add_argument(
            f"--{receptor}-train-02",
            default=str(DEFAULTS[f"{receptor}_train_02"]),
        )
        parser.add_argument(
            f"--{receptor}-test-02",
            default=str(DEFAULTS[f"{receptor}_test_02"]),
        )
        parser.add_argument(
            f"--{receptor}-train-03-descriptive",
            default=str(DEFAULTS[f"{receptor}_train_03_desc"]),
        )
        parser.add_argument(
            f"--{receptor}-test-03-descriptive",
            default=str(DEFAULTS[f"{receptor}_test_03_desc"]),
        )
        parser.add_argument(
            f"--{receptor}-test-03-reference",
            default=str(DEFAULTS[f"{receptor}_test_03_ref"]),
        )
        parser.add_argument(
            f"--{receptor}-test-03-definition",
            default=str(DEFAULTS[f"{receptor}_test_03_definition"]),
        )
        parser.add_argument(
            f"--{receptor}-train-output-dir",
            default=str(DEFAULTS[f"{receptor}_train_out"]),
        )
        parser.add_argument(
            f"--{receptor}-test-output-dir",
            default=str(DEFAULTS[f"{receptor}_test_out"]),
        )

    parser.add_argument(
        "--joint-train-output-dir",
        default=str(DEFAULTS["joint_train_out"]),
    )
    parser.add_argument(
        "--joint-test-output-dir",
        default=str(DEFAULTS["joint_test_out"]),
    )

    parser.add_argument("--expected-train-samples", type=int, default=118)
    parser.add_argument("--expected-test-samples", type=int, default=49)
    parser.add_argument("--expected-02-columns", type=int, default=1097)
    parser.add_argument("--expected-03-descriptive-columns", type=int, default=26)
    parser.add_argument("--expected-test-reference-columns", type=int, default=19)
    parser.add_argument("--scheme-name", default="main")
    parser.add_argument("--float-tolerance", type=float, default=1e-10)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate all inputs and construct matrices in memory without writing outputs.",
    )
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    args.receptor_metadata_output_cols = parse_csv_list(
        args.receptor_metadata_output_cols
    )
    args.joint_metadata_output_cols = parse_csv_list(args.joint_metadata_output_cols)

    if args.expected_train_samples < 1 or args.expected_test_samples < 1:
        parser.error("Expected sample counts must be >= 1.")
    if args.expected_02_columns < 2:
        parser.error("--expected-02-columns must be >= 2.")
    if args.expected_03_descriptive_columns < 2:
        parser.error("--expected-03-descriptive-columns must be >= 2.")
    if args.expected_test_reference_columns < 2:
        parser.error("--expected-test-reference-columns must be >= 2.")
    if args.float_tolerance <= 0:
        parser.error("--float-tolerance must be > 0.")

    return args


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


# -----------------------------------------------------------------------------
# Generic validation and IO
# -----------------------------------------------------------------------------

def path_from_arg(args: argparse.Namespace, name: str) -> Path:
    return Path(getattr(args, name)).expanduser().resolve()


def read_csv(path: Path, label: str, dtype=None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    df = pd.read_csv(path, dtype=dtype)
    unnamed = [column for column in df.columns if str(column).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"{label} duplicated columns: {duplicates[:20]}")
    return df


def read_json(path: Path, label: str) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}: {exc}") from exc


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def check_expected(actual: int, expected: int, label: str) -> None:
    if expected > 0 and actual != expected:
        raise ValueError(f"{label}: expected {expected}, observed {actual}")


def check_ids(df: pd.DataFrame, id_col: str, label: str) -> None:
    if id_col not in df.columns:
        raise ValueError(f"{label} missing ID column: {id_col}")
    ids = df[id_col]
    if ids.isna().any():
        raise ValueError(f"{label} has missing {id_col}")
    ids = ids.astype(str).str.strip()
    if ids.eq("").any():
        raise ValueError(f"{label} has empty {id_col}")
    if ids.duplicated().any():
        duplicates = ids[ids.duplicated(keep=False)].unique().tolist()[:20]
        raise ValueError(f"{label} duplicated {id_col}: {duplicates}")
    df[id_col] = ids


def check_no_missing(df: pd.DataFrame, label: str) -> None:
    total = int(df.isna().sum().sum())
    if total:
        counts = df.isna().sum()
        raise ValueError(
            f"{label} has {total} missing values: "
            f"{counts[counts > 0].head(20).to_dict()}"
        )


def check_finite_numeric(df: pd.DataFrame, id_col: str, label: str) -> None:
    numeric = df.drop(columns=[id_col], errors="ignore")
    if numeric.empty:
        raise ValueError(f"{label} contains no feature columns")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains NaN or infinite numeric values")


def ensure_exact_id_set(
    left: pd.DataFrame,
    right: pd.DataFrame,
    id_col: str,
    left_label: str,
    right_label: str,
) -> None:
    left_ids = set(left[id_col])
    right_ids = set(right[id_col])
    if left_ids != right_ids:
        raise ValueError(
            f"ID mismatch {left_label} vs {right_label}; "
            f"only_{left_label}={sorted(left_ids - right_ids)[:20]}, "
            f"only_{right_label}={sorted(right_ids - left_ids)[:20]}"
        )


def compare_numeric(
    left: pd.Series,
    right: pd.Series,
    tolerance: float,
    label: str,
) -> None:
    a = pd.to_numeric(left, errors="raise").to_numpy(dtype=float)
    b = pd.to_numeric(right, errors="raise").to_numpy(dtype=float)
    if not np.allclose(a, b, rtol=tolerance, atol=tolerance):
        difference = np.abs(a - b)
        index = int(np.argmax(difference))
        raise ValueError(
            f"{label} mismatch at row {index}; "
            f"left={a[index]}, right={b[index]}, diff={difference[index]}"
        )


def ensure_output_policy(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        text = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Output files already exist. Add --overwrite to replace them:\n" + text
        )


def write_dataframe_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp, index=False)
    temp.replace(path)


def write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


# -----------------------------------------------------------------------------
# Metadata and feature loading
# -----------------------------------------------------------------------------

def load_paired_metadata(
    path: Path,
    args: argparse.Namespace,
    expected_rows: int,
    label: str,
) -> pd.DataFrame:
    df = read_csv(path, label, dtype=str)
    required = {
        args.patient_col,
        args.trb_id_col,
        args.igh_id_col,
        args.cohort_col,
        *args.receptor_metadata_output_cols,
        *args.joint_metadata_output_cols,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")

    df = df.reset_index(drop=True)
    for column in [args.patient_col, args.trb_id_col, args.igh_id_col]:
        values = df[column]
        if values.isna().any():
            raise ValueError(f"{label}.{column} contains missing values")
        values = values.astype(str).str.strip()
        if values.eq("").any():
            raise ValueError(f"{label}.{column} contains empty values")
        if values.duplicated().any():
            duplicates = values[values.duplicated(keep=False)].unique().tolist()[:20]
            raise ValueError(f"{label}.{column} duplicated IDs: {duplicates}")
        df[column] = values

    check_expected(len(df), expected_rows, f"{label} rows")
    check_no_missing(df[list(required)], label)

    df[args.cohort_col] = df[args.cohort_col].astype(str).str.strip().str.upper()
    if set(df[args.cohort_col]) != {"RA", "ILD"}:
        raise ValueError(
            f"{label}.{args.cohort_col} must contain exactly RA and ILD; "
            f"observed={sorted(set(df[args.cohort_col]))}"
        )
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="raise")
    return df


def check_train_test_disjoint(
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    for column in [args.patient_col, args.trb_id_col, args.igh_id_col]:
        overlap = set(train_metadata[column]) & set(test_metadata[column])
        if overlap:
            raise ValueError(
                f"Train/test overlap in {column}: {sorted(overlap)[:20]}"
            )


def load_numeric_matrix(
    path: Path,
    id_col: str,
    expected_rows: int,
    expected_columns: int,
    label: str,
) -> pd.DataFrame:
    df = read_csv(path, label)
    check_ids(df, id_col, label)
    check_no_missing(df, label)
    check_expected(len(df), expected_rows, f"{label} rows")
    check_expected(len(df.columns), expected_columns, f"{label} columns")

    for column in df.columns:
        if column == id_col:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"{label}.{column} is not fully numeric")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{label}.{column} contains non-finite values")
        df[column] = values
    return df


def validate_reference_definition(
    definition: Mapping[str, object],
    receptor: str,
    expected_samples: int,
    scheme_name: str,
    label: str,
) -> None:
    observed_receptor = str(definition.get("receptor", "")).upper()
    if observed_receptor != receptor:
        raise RuntimeError(
            f"{label} receptor={observed_receptor!r}, expected {receptor}"
        )

    scheme = definition.get("scheme", {})
    observed_scheme = scheme.get("name") if isinstance(scheme, Mapping) else None
    if observed_scheme != scheme_name:
        raise RuntimeError(
            f"{label} scheme={observed_scheme!r}, expected {scheme_name!r}"
        )

    sample_count = int(definition.get("test_sample_count", -1))
    if sample_count != expected_samples:
        raise RuntimeError(
            f"{label} test_sample_count={sample_count}, expected {expected_samples}"
        )

    labels_used = definition.get("test_labels_used_to_build_reference")
    if labels_used is not False:
        raise RuntimeError(
            f"{label} does not confirm test_labels_used_to_build_reference=False"
        )

    expected_hash = definition.get("reference_file_sha256_verified")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise RuntimeError(f"{label} missing verified reference SHA256")


def receptor_metadata_frame(
    metadata: pd.DataFrame,
    receptor: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    id_source = args.trb_id_col if receptor == "TRB" else args.igh_id_col
    columns = [id_source] + list(args.receptor_metadata_output_cols)
    frame = metadata[columns].copy()
    frame = frame.rename(columns={id_source: args.sample_id_col})
    check_ids(frame, args.sample_id_col, f"{receptor} metadata frame")
    return frame


def joint_metadata_frame(
    metadata: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    columns = [args.patient_col] + list(args.joint_metadata_output_cols)
    frame = metadata[columns].copy()
    frame = frame.rename(columns={args.patient_col: args.sample_id_col})
    check_ids(frame, args.sample_id_col, "joint metadata frame")
    return frame


# -----------------------------------------------------------------------------
# Matrix construction
# -----------------------------------------------------------------------------

def merge_receptor_base(
    metadata_frame: pd.DataFrame,
    feature_02: pd.DataFrame,
    sample_id_col: str,
    label: str,
) -> pd.DataFrame:
    ensure_exact_id_set(
        metadata_frame,
        feature_02,
        sample_id_col,
        f"{label}_metadata",
        f"{label}_02",
    )
    out = metadata_frame.merge(
        feature_02,
        on=sample_id_col,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    ordered = (
        metadata_frame.columns.tolist()
        + [column for column in feature_02.columns if column != sample_id_col]
    )
    out = out[ordered]
    check_no_missing(out, f"{label} base matrix")
    return out


def merge_receptor_descriptive(
    base: pd.DataFrame,
    descriptive: pd.DataFrame,
    sample_id_col: str,
    tolerance: float,
    label: str,
) -> Tuple[pd.DataFrame, List[str]]:
    ensure_exact_id_set(
        base,
        descriptive,
        sample_id_col,
        f"{label}_base",
        f"{label}_03_descriptive",
    )
    to_add = descriptive.copy()

    if "aa_clone_number" in to_add.columns:
        if "aa_clone_number" not in base.columns:
            raise ValueError(f"{label} base matrix missing aa_clone_number")
        check = base[[sample_id_col, "aa_clone_number"]].merge(
            to_add[[sample_id_col, "aa_clone_number"]],
            on=sample_id_col,
            validate="one_to_one",
            suffixes=("_02", "_03"),
            sort=False,
        )
        compare_numeric(
            check["aa_clone_number_02"],
            check["aa_clone_number_03"],
            tolerance,
            f"{label} aa_clone_number",
        )
        to_add = to_add.drop(columns="aa_clone_number")

    added_columns = [column for column in to_add.columns if column != sample_id_col]
    collisions = sorted((set(base.columns) & set(added_columns)))
    if collisions:
        raise ValueError(f"{label} descriptive column collisions: {collisions}")

    out = base.merge(
        to_add,
        on=sample_id_col,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    check_no_missing(out, f"{label} descriptive matrix")
    return out, added_columns


def merge_receptor_test_final(
    base: pd.DataFrame,
    reference: pd.DataFrame,
    sample_id_col: str,
    label: str,
) -> Tuple[pd.DataFrame, List[str]]:
    ensure_exact_id_set(
        base,
        reference,
        sample_id_col,
        f"{label}_base",
        f"{label}_03_reference",
    )
    added_columns = [column for column in reference.columns if column != sample_id_col]
    collisions = sorted(set(base.columns) & set(added_columns))
    if collisions:
        raise ValueError(f"{label} test-reference column collisions: {collisions}")

    out = base.merge(
        reference,
        on=sample_id_col,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    check_no_missing(out, f"{label} test-final matrix")
    return out, added_columns


def feature_block_by_patient(
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    receptor: str,
    source_id_col: str,
    patient_col: str,
    sample_id_col: str,
    feature_columns: Sequence[str],
    prefix: str,
    label: str,
) -> pd.DataFrame:
    mapping = metadata[[patient_col, source_id_col]].copy()
    mapping = mapping.rename(columns={source_id_col: sample_id_col})
    check_ids(mapping, sample_id_col, f"{label} ID mapping")

    ensure_exact_id_set(
        mapping,
        matrix,
        sample_id_col,
        f"{label}_mapping",
        f"{label}_matrix",
    )

    block = mapping.merge(
        matrix[[sample_id_col] + list(feature_columns)],
        on=sample_id_col,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    block = block.drop(columns=sample_id_col)
    renamed = {column: f"{prefix}{column}" for column in feature_columns}
    block = block.rename(columns=renamed)
    block = block.rename(columns={patient_col: sample_id_col})
    check_ids(block, sample_id_col, f"{label} patient block")
    check_no_missing(block, f"{label} patient block")
    return block


def merge_joint_blocks(
    joint_metadata: pd.DataFrame,
    trb_block: pd.DataFrame,
    igh_block: pd.DataFrame,
    sample_id_col: str,
    label: str,
) -> pd.DataFrame:
    ensure_exact_id_set(
        joint_metadata,
        trb_block,
        sample_id_col,
        f"{label}_metadata",
        f"{label}_TRB",
    )
    ensure_exact_id_set(
        joint_metadata,
        igh_block,
        sample_id_col,
        f"{label}_metadata",
        f"{label}_IGH",
    )

    collisions = (
        (set(joint_metadata.columns) & set(trb_block.columns))
        | (set(joint_metadata.columns) & set(igh_block.columns))
        | (set(trb_block.columns) & set(igh_block.columns))
    ) - {sample_id_col}
    if collisions:
        raise RuntimeError(f"{label} joint column collisions: {sorted(collisions)[:20]}")

    out = (
        joint_metadata
        .merge(trb_block, on=sample_id_col, how="left", validate="one_to_one", sort=False)
        .merge(igh_block, on=sample_id_col, how="left", validate="one_to_one", sort=False)
    )
    check_no_missing(out, f"{label} joint matrix")
    return out


# -----------------------------------------------------------------------------
# Feature manifest
# -----------------------------------------------------------------------------

def infer_data_type(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def strip_receptor_prefix(feature_name: str) -> Tuple[str, str]:
    if feature_name.startswith("trb_"):
        return "TRB", feature_name[len("trb_"):]
    if feature_name.startswith("igh_"):
        return "IGH", feature_name[len("igh_"):]
    return "metadata", feature_name


def classify_static_feature(base_name: str, receptor: str) -> str:
    prefix = receptor.lower()
    if base_name in QC_FEATURES:
        return f"{prefix}_qc_depth"
    if base_name in DIVERSITY_FEATURES:
        return f"{prefix}_diversity"
    if base_name in EXPANSION_FEATURES:
        return f"{prefix}_clonal_expansion"
    if base_name.startswith("unweighted_3mer_"):
        return f"{prefix}_3mer_unweighted"
    if base_name.startswith("weighted_3mer_"):
        return f"{prefix}_3mer_weighted"
    if "aa_length" in base_name or base_name.startswith("weighted_length_"):
        return f"{prefix}_aa_length"
    if base_name.endswith("_frequency"):
        core = base_name.removeprefix("weighted_").split("_")[0]
        if len(core) == 1:
            suffix = "weighted" if base_name.startswith("weighted_") else "unweighted"
            return f"{prefix}_aa_composition_{suffix}"
    if any(
        token in base_name
        for token in (
            "hydrophobicity",
            "charge",
            "aromaticity",
            "basic_aa_ratio",
            "acidic_aa_ratio",
            "polar_aa_ratio",
            "nonpolar_aa_ratio",
        )
    ):
        return f"{prefix}_physicochemical"
    return f"{prefix}_other_static"


def metadata_role(feature_name: str) -> Tuple[str, str, str]:
    if feature_name == "sample_id":
        return "identifier", "join_key", "no"
    if feature_name == "cohort":
        return "outcome", "outcome_label", "no"
    if feature_name in {"patient", "trb_libraryid", "igh_libraryid"}:
        return "identifier_trace", "trace_only", "no"
    if feature_name in {"age", "sex"}:
        return "clinical", "primary_predictor", "yes"
    if feature_name == "material":
        return "clinical_optional", "optional_predictor", "optional"
    if feature_name in {"batch", "igh_batch", "split_batch"}:
        return "technical_qc", "qc_only", "no"
    return "metadata_review", "review", "review"


def transformation_text(
    feature_name: str,
    data_type: str,
    role: str,
    base_name: str,
) -> str:
    if feature_name == "cohort":
        return "binary encode in model pipeline"
    if feature_name in {"sex", "material"}:
        return "one-hot encode in model pipeline"
    if role in {"qc_only", "trace_only", "join_key", "outcome_label", "descriptive_only"}:
        return "none"
    if base_name in REFERENCE_DROP:
        return "drop before model fitting"
    if data_type in {"numeric", "integer"}:
        return "standardize within each training fold"
    return "none"


def build_manifest(
    analysis_view: str,
    matrices: Mapping[str, pd.DataFrame],
    source_by_matrix: Mapping[str, str],
    sample_id_col: str,
    metadata_columns: Sequence[str],
    static_base_columns: Mapping[str, Sequence[str]],
    descriptive_added_columns: Mapping[str, Sequence[str]],
    reference_added_columns: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    ordered_features: List[str] = []
    for matrix_name in ["train_base", "test_base", "train_descriptive", "test_descriptive", "test_final"]:
        frame = matrices[matrix_name]
        for column in frame.columns:
            if column not in ordered_features:
                ordered_features.append(column)

    metadata_set = set(metadata_columns)
    rows: List[Dict[str, object]] = []

    for order, feature_name in enumerate(ordered_features, start=1):
        receptor, base_name = strip_receptor_prefix(feature_name)
        source = "unknown"
        feature_group = "review"
        role = "review"
        model_default = "review"
        dynamic = False
        reference_drop = False
        notes = ""

        if feature_name in metadata_set:
            source = "metadata"
            feature_group, role, model_default = metadata_role(feature_name)
        else:
            # Determine receptor for receptor-specific views, where columns are unprefixed.
            if analysis_view in {"TRB", "IGH"}:
                receptor = analysis_view
                base_name = feature_name

            receptor_key = receptor.lower() if receptor in {"TRB", "IGH"} else ""
            static_set = set(static_base_columns.get(receptor, []))
            desc_set = set(descriptive_added_columns.get(receptor, []))
            ref_set = set(reference_added_columns.get(receptor, []))

            if base_name in static_set:
                source = f"{receptor}_02_sample_level_features_merged"
                feature_group = classify_static_feature(base_name, receptor)
                if base_name in REFERENCE_DROP:
                    role = "composition_reference_drop"
                    model_default = "no"
                    reference_drop = True
                elif base_name in QC_FEATURES:
                    role = "qc_or_sensitivity_predictor"
                    model_default = "sensitivity"
                else:
                    role = "candidate_predictor"
                    model_default = "yes"
            elif base_name in ref_set:
                source = f"{receptor}_03_test_reference_public_features"
                if base_name in desc_set:
                    source = (
                        f"{receptor}_03_descriptive_public_features;"
                        f"{receptor}_03_test_reference_public_features"
                    )
                feature_group = f"{receptor_key}_public_reference_dynamic"
                role = "dynamic_predictor_train_fixed_predictor_test"
                model_default = "dynamic"
                dynamic = True
                notes = (
                    "Generate the training value inside each CV fold. Any value with the "
                    "same name in the complete-training descriptive matrix is self-inclusive "
                    "and must not be used as a fixed predictor; the independent-test value "
                    "uses the fixed complete-training reference."
                )
            elif base_name in desc_set:
                source = f"{receptor}_03_descriptive_public_features"
                feature_group = f"{receptor_key}_public_descriptive_self_inclusive"
                role = "descriptive_only"
                model_default = "no"
                notes = (
                    "Complete-training descriptive public values are self-inclusive and "
                    "must not be used as fixed cross-validation predictors."
                )

        exemplar = next(
            frame[feature_name]
            for frame in matrices.values()
            if feature_name in frame.columns
        )
        data_type = infer_data_type(exemplar)

        rows.append(
            {
                "analysis_view": analysis_view,
                "receptor": receptor,
                "feature_name": feature_name,
                "feature_order": order,
                "source_file": source,
                "feature_group": feature_group,
                "feature_role": role,
                "data_type": data_type,
                "present_in_train_base": feature_name in matrices["train_base"].columns,
                "present_in_test_base": feature_name in matrices["test_base"].columns,
                "present_in_train_descriptive": feature_name in matrices["train_descriptive"].columns,
                "present_in_test_descriptive": feature_name in matrices["test_descriptive"].columns,
                "present_in_test_final": feature_name in matrices["test_final"].columns,
                "eligible_for_fixed_train_cv_input": (
                    feature_name in matrices["train_base"].columns
                    and role not in {
                        "join_key",
                        "outcome_label",
                        "trace_only",
                        "qc_only",
                        "descriptive_only",
                        "composition_reference_drop",
                    }
                ),
                "dynamic_in_training": dynamic,
                "model_default": model_default,
                "transformation": transformation_text(
                    feature_name, data_type, role, base_name
                ),
                "reference_drop": reference_drop,
                "notes": notes,
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Outputs and summaries
# -----------------------------------------------------------------------------

def output_paths(train_dir: Path, test_dir: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    train = {
        "train_base": train_dir / "04_train_base_feature_matrix.csv",
        "train_descriptive": train_dir / "04_train_descriptive_feature_matrix.csv",
        "manifest": train_dir / "04_feature_manifest.csv",
        "provenance": train_dir / "04_feature_matrix_provenance.json",
        "summary": train_dir / "04_feature_matrix_build_summary.md",
    }
    test = {
        "test_base": test_dir / "04_test_base_feature_matrix.csv",
        "test_descriptive": test_dir / "04_test_descriptive_feature_matrix.csv",
        "test_final": test_dir / "04_test_final_feature_matrix.csv",
        "summary": test_dir / "04_feature_matrix_build_summary.md",
    }
    return train, test


def build_provenance(
    analysis_view: str,
    inputs: Mapping[str, Path],
    shapes: Mapping[str, Tuple[int, int]],
    metadata: Mapping[str, object],
) -> Dict[str, object]:
    input_records = {}
    for label, path in inputs.items():
        input_records[label] = {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "script_version": VERSION,
        "analysis_view": analysis_view,
        "inputs": input_records,
        "output_shapes": {key: list(value) for key, value in shapes.items()},
        **metadata,
        "important_note": (
            "No fixed training final matrix with complete-training reference-public "
            "predictors is generated. Training public predictors remain fold-dynamic."
        ),
    }


def make_summary(
    analysis_view: str,
    inputs: Mapping[str, Path],
    outputs: Mapping[str, Path],
    shapes: Mapping[str, Tuple[int, int]],
    runtime: float,
    sample_id_description: str,
) -> str:
    lines = [
        f"# 04 {analysis_view} Feature Matrix Build Summary",
        "",
        f"- Script version: `{VERSION}`",
        f"- Analysis view: `{analysis_view}`",
        f"- Sample key: {sample_id_description}",
        f"- Runtime: **{runtime:.2f} seconds**",
        "",
        "## Inputs",
        "",
    ]
    for label, path in inputs.items():
        lines.append(f"- `{label}`: `{path}`")

    lines.extend(["", "## Output shapes", ""])
    for name, shape in shapes.items():
        lines.append(f"- `{name}`: **{shape[0]} × {shape[1]}**")

    lines.extend(
        [
            "",
            "## Validation",
            "",
            "- Train and test patient, TRB-library, and IGH-library IDs are disjoint.",
            "- Every merge is one-to-one and preserves metadata order.",
            "- Train/test step-02 schemas are identical within each receptor.",
            "- Train/test descriptive schemas are identical within each receptor.",
            "- Step-02 and step-03 `aa_clone_number` values agree.",
            "- TRB and IGH patient mappings are complete and one-to-one.",
            "- No missing or non-finite feature values were found.",
            "- Test reference definitions confirm that test labels were not used.",
            "",
            "## Modeling notes",
            "",
            "- `04_train_base_feature_matrix.csv` is the fixed input for ordinary CV.",
            "- `04_train_descriptive_feature_matrix.csv` contains self-inclusive public summaries and is descriptive only.",
            "- No fixed `04_train_final_feature_matrix.csv` is generated.",
            "- Training reference-public features must be rebuilt inside each CV training fold.",
            "- `04_test_final_feature_matrix.csv` contains fixed training-reference public features for independent testing.",
            "- `patient`/library IDs are trace fields, not predictors.",
            "- `batch`, `igh_batch`, and `split_batch` are retained for QC/sensitivity, not the primary model.",
            "- Composition reference columns are retained in files but flagged for removal in the manifest.",
            "",
            "## Outputs",
            "",
        ]
    )
    for label, path in outputs.items():
        lines.append(f"- `{label}`: `{path}`")
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    started = time.time()

    # Resolve all input paths.
    paths = {
        name: path_from_arg(args, name)
        for name in [
            "train_metadata",
            "test_metadata",
            "trb_train_02",
            "trb_test_02",
            "igh_train_02",
            "igh_test_02",
            "trb_train_03_descriptive",
            "trb_test_03_descriptive",
            "igh_train_03_descriptive",
            "igh_test_03_descriptive",
            "trb_test_03_reference",
            "igh_test_03_reference",
            "trb_test_03_definition",
            "igh_test_03_definition",
        ]
    }

    # Read and validate paired metadata.
    train_metadata = load_paired_metadata(
        paths["train_metadata"],
        args,
        args.expected_train_samples,
        "train metadata",
    )
    test_metadata = load_paired_metadata(
        paths["test_metadata"],
        args,
        args.expected_test_samples,
        "test metadata",
    )
    check_train_test_disjoint(train_metadata, test_metadata, args)

    # Receptor-specific input matrices.
    receptor_inputs: Dict[str, Dict[str, pd.DataFrame]] = {}
    for receptor in ("TRB", "IGH"):
        key = receptor.lower()
        train_02 = load_numeric_matrix(
            paths[f"{key}_train_02"],
            args.sample_id_col,
            args.expected_train_samples,
            args.expected_02_columns,
            f"{receptor} train 02",
        )
        test_02 = load_numeric_matrix(
            paths[f"{key}_test_02"],
            args.sample_id_col,
            args.expected_test_samples,
            args.expected_02_columns,
            f"{receptor} test 02",
        )
        if train_02.columns.tolist() != test_02.columns.tolist():
            raise ValueError(f"{receptor} train/test step-02 columns or order differ")

        train_desc = load_numeric_matrix(
            paths[f"{key}_train_03_descriptive"],
            args.sample_id_col,
            args.expected_train_samples,
            args.expected_03_descriptive_columns,
            f"{receptor} train 03 descriptive",
        )
        test_desc = load_numeric_matrix(
            paths[f"{key}_test_03_descriptive"],
            args.sample_id_col,
            args.expected_test_samples,
            args.expected_03_descriptive_columns,
            f"{receptor} test 03 descriptive",
        )
        if train_desc.columns.tolist() != test_desc.columns.tolist():
            raise ValueError(
                f"{receptor} train/test step-03 descriptive schemas differ"
            )

        test_ref = load_numeric_matrix(
            paths[f"{key}_test_03_reference"],
            args.sample_id_col,
            args.expected_test_samples,
            args.expected_test_reference_columns,
            f"{receptor} test 03 reference",
        )
        definition = read_json(
            paths[f"{key}_test_03_definition"],
            f"{receptor} test 03 definition",
        )
        validate_reference_definition(
            definition,
            receptor,
            args.expected_test_samples,
            args.scheme_name,
            f"{receptor} test 03 definition",
        )

        receptor_inputs[receptor] = {
            "train_02": train_02,
            "test_02": test_02,
            "train_desc": train_desc,
            "test_desc": test_desc,
            "test_ref": test_ref,
        }

    # Ensure TRB and IGH input IDs match paired metadata.
    for receptor, id_col in [("TRB", args.trb_id_col), ("IGH", args.igh_id_col)]:
        metadata_train_ids = set(train_metadata[id_col])
        metadata_test_ids = set(test_metadata[id_col])
        observed_train_ids = set(receptor_inputs[receptor]["train_02"][args.sample_id_col])
        observed_test_ids = set(receptor_inputs[receptor]["test_02"][args.sample_id_col])
        if metadata_train_ids != observed_train_ids:
            raise ValueError(f"{receptor} train IDs do not match paired metadata")
        if metadata_test_ids != observed_test_ids:
            raise ValueError(f"{receptor} test IDs do not match paired metadata")

    # Construct receptor-specific matrices.
    receptor_results: Dict[str, Dict[str, object]] = {}
    for receptor in ("TRB", "IGH"):
        id_col = args.trb_id_col if receptor == "TRB" else args.igh_id_col
        train_meta_frame = receptor_metadata_frame(train_metadata, receptor, args)
        test_meta_frame = receptor_metadata_frame(test_metadata, receptor, args)
        data = receptor_inputs[receptor]

        train_base = merge_receptor_base(
            train_meta_frame,
            data["train_02"],
            args.sample_id_col,
            f"{receptor} train",
        )
        test_base = merge_receptor_base(
            test_meta_frame,
            data["test_02"],
            args.sample_id_col,
            f"{receptor} test",
        )
        if train_base.columns.tolist() != test_base.columns.tolist():
            raise RuntimeError(f"{receptor} train/test base schemas differ")

        train_descriptive, desc_columns = merge_receptor_descriptive(
            train_base,
            data["train_desc"],
            args.sample_id_col,
            args.float_tolerance,
            f"{receptor} train",
        )
        test_descriptive, test_desc_columns = merge_receptor_descriptive(
            test_base,
            data["test_desc"],
            args.sample_id_col,
            args.float_tolerance,
            f"{receptor} test",
        )
        if desc_columns != test_desc_columns:
            raise RuntimeError(f"{receptor} train/test added descriptive columns differ")
        if train_descriptive.columns.tolist() != test_descriptive.columns.tolist():
            raise RuntimeError(f"{receptor} train/test descriptive schemas differ")

        test_final, reference_columns = merge_receptor_test_final(
            test_base,
            data["test_ref"],
            args.sample_id_col,
            f"{receptor} test",
        )

        static_columns = [
            column for column in data["train_02"].columns
            if column != args.sample_id_col
        ]
        metadata_columns = train_meta_frame.columns.tolist()
        matrices = {
            "train_base": train_base,
            "test_base": test_base,
            "train_descriptive": train_descriptive,
            "test_descriptive": test_descriptive,
            "test_final": test_final,
        }
        manifest = build_manifest(
            analysis_view=receptor,
            matrices=matrices,
            source_by_matrix={},
            sample_id_col=args.sample_id_col,
            metadata_columns=metadata_columns,
            static_base_columns={receptor: static_columns},
            descriptive_added_columns={receptor: desc_columns},
            reference_added_columns={receptor: reference_columns},
        )

        receptor_results[receptor] = {
            "id_col": id_col,
            "train_base": train_base,
            "test_base": test_base,
            "train_descriptive": train_descriptive,
            "test_descriptive": test_descriptive,
            "test_final": test_final,
            "manifest": manifest,
            "static_columns": static_columns,
            "desc_columns": desc_columns,
            "reference_columns": reference_columns,
        }

    # Construct patient-level paired matrices.
    train_joint_meta = joint_metadata_frame(train_metadata, args)
    test_joint_meta = joint_metadata_frame(test_metadata, args)

    def make_patient_block(
        receptor: str,
        split: str,
        columns: Sequence[str],
        prefix: str,
    ) -> pd.DataFrame:
        id_col = args.trb_id_col if receptor == "TRB" else args.igh_id_col
        metadata = train_metadata if split == "train" else test_metadata
        matrix = receptor_results[receptor][f"{split}_base"]
        # Base matrices contain metadata; select only receptor-derived columns.
        return feature_block_by_patient(
            matrix=matrix,
            metadata=metadata,
            receptor=receptor,
            source_id_col=id_col,
            patient_col=args.patient_col,
            sample_id_col=args.sample_id_col,
            feature_columns=columns,
            prefix=prefix,
            label=f"{receptor} {split} static",
        )

    trb_train_static = make_patient_block(
        "TRB", "train", receptor_results["TRB"]["static_columns"], "trb_"
    )
    trb_test_static = make_patient_block(
        "TRB", "test", receptor_results["TRB"]["static_columns"], "trb_"
    )
    igh_train_static = make_patient_block(
        "IGH", "train", receptor_results["IGH"]["static_columns"], "igh_"
    )
    igh_test_static = make_patient_block(
        "IGH", "test", receptor_results["IGH"]["static_columns"], "igh_"
    )

    joint_train_base = merge_joint_blocks(
        train_joint_meta,
        trb_train_static,
        igh_train_static,
        args.sample_id_col,
        "train base",
    )
    joint_test_base = merge_joint_blocks(
        test_joint_meta,
        trb_test_static,
        igh_test_static,
        args.sample_id_col,
        "test base",
    )
    if joint_train_base.columns.tolist() != joint_test_base.columns.tolist():
        raise RuntimeError("Joint train/test base schemas differ")

    def extra_patient_block(
        receptor: str,
        split: str,
        matrix_key: str,
        columns: Sequence[str],
        prefix: str,
        label_suffix: str,
    ) -> pd.DataFrame:
        id_col = args.trb_id_col if receptor == "TRB" else args.igh_id_col
        metadata = train_metadata if split == "train" else test_metadata
        matrix = receptor_results[receptor][matrix_key]
        return feature_block_by_patient(
            matrix=matrix,
            metadata=metadata,
            receptor=receptor,
            source_id_col=id_col,
            patient_col=args.patient_col,
            sample_id_col=args.sample_id_col,
            feature_columns=columns,
            prefix=prefix,
            label=f"{receptor} {split} {label_suffix}",
        )

    trb_train_desc = extra_patient_block(
        "TRB", "train", "train_descriptive",
        receptor_results["TRB"]["desc_columns"], "trb_", "descriptive"
    )
    trb_test_desc = extra_patient_block(
        "TRB", "test", "test_descriptive",
        receptor_results["TRB"]["desc_columns"], "trb_", "descriptive"
    )
    igh_train_desc = extra_patient_block(
        "IGH", "train", "train_descriptive",
        receptor_results["IGH"]["desc_columns"], "igh_", "descriptive"
    )
    igh_test_desc = extra_patient_block(
        "IGH", "test", "test_descriptive",
        receptor_results["IGH"]["desc_columns"], "igh_", "descriptive"
    )

    joint_train_descriptive = merge_joint_blocks(
        joint_train_base,
        trb_train_desc,
        igh_train_desc,
        args.sample_id_col,
        "train descriptive",
    )
    joint_test_descriptive = merge_joint_blocks(
        joint_test_base,
        trb_test_desc,
        igh_test_desc,
        args.sample_id_col,
        "test descriptive",
    )
    if joint_train_descriptive.columns.tolist() != joint_test_descriptive.columns.tolist():
        raise RuntimeError("Joint train/test descriptive schemas differ")

    trb_test_ref = extra_patient_block(
        "TRB", "test", "test_final",
        receptor_results["TRB"]["reference_columns"], "trb_", "reference"
    )
    igh_test_ref = extra_patient_block(
        "IGH", "test", "test_final",
        receptor_results["IGH"]["reference_columns"], "igh_", "reference"
    )
    joint_test_final = merge_joint_blocks(
        joint_test_base,
        trb_test_ref,
        igh_test_ref,
        args.sample_id_col,
        "test final",
    )

    joint_matrices = {
        "train_base": joint_train_base,
        "test_base": joint_test_base,
        "train_descriptive": joint_train_descriptive,
        "test_descriptive": joint_test_descriptive,
        "test_final": joint_test_final,
    }
    joint_manifest = build_manifest(
        analysis_view="TRB_IGH",
        matrices=joint_matrices,
        source_by_matrix={},
        sample_id_col=args.sample_id_col,
        metadata_columns=train_joint_meta.columns.tolist(),
        static_base_columns={
            "TRB": receptor_results["TRB"]["static_columns"],
            "IGH": receptor_results["IGH"]["static_columns"],
        },
        descriptive_added_columns={
            "TRB": receptor_results["TRB"]["desc_columns"],
            "IGH": receptor_results["IGH"]["desc_columns"],
        },
        reference_added_columns={
            "TRB": receptor_results["TRB"]["reference_columns"],
            "IGH": receptor_results["IGH"]["reference_columns"],
        },
    )

    # Final global checks.
    for view, matrices in {
        "TRB": {key: receptor_results["TRB"][key] for key in [
            "train_base", "test_base", "train_descriptive", "test_descriptive", "test_final"
        ]},
        "IGH": {key: receptor_results["IGH"][key] for key in [
            "train_base", "test_base", "train_descriptive", "test_descriptive", "test_final"
        ]},
        "TRB_IGH": joint_matrices,
    }.items():
        for matrix_name, matrix in matrices.items():
            check_ids(matrix, args.sample_id_col, f"{view} {matrix_name}")
            check_no_missing(matrix, f"{view} {matrix_name}")
            # Only feature columns need to be finite. Metadata categorical columns are excluded.
            numeric_columns = matrix.select_dtypes(include=[np.number]).columns
            if numeric_columns.size and not np.isfinite(
                matrix[numeric_columns].to_numpy(dtype=float)
            ).all():
                raise RuntimeError(f"{view} {matrix_name} contains non-finite values")

    print("=" * 88)
    print("04 Paired TRB+IGH Final Feature Matrix Build")
    print("=" * 88)
    print(f"Script version: {VERSION}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Train paired patients: {len(train_metadata)}")
    print(f"Test paired patients: {len(test_metadata)}")
    print("Preflight and in-memory matrix construction: PASS")
    print("No fixed training final matrix with reference-public predictors is generated.")
    print("")

    # Output directories and files.
    directories = {
        "TRB": (
            path_from_arg(args, "trb_train_output_dir"),
            path_from_arg(args, "trb_test_output_dir"),
        ),
        "IGH": (
            path_from_arg(args, "igh_train_output_dir"),
            path_from_arg(args, "igh_test_output_dir"),
        ),
        "TRB_IGH": (
            path_from_arg(args, "joint_train_output_dir"),
            path_from_arg(args, "joint_test_output_dir"),
        ),
    }

    outputs_by_view: Dict[str, Tuple[Dict[str, Path], Dict[str, Path]]] = {
        view: output_paths(*directories[view]) for view in directories
    }

    selected_outputs: List[Path] = []
    for train_outputs, test_outputs in outputs_by_view.values():
        if args.mode in {"train", "both"}:
            selected_outputs.extend(train_outputs.values())
        if args.mode in {"test", "both"}:
            selected_outputs.extend(test_outputs.values())
    if not args.preflight_only:
        ensure_output_policy(selected_outputs, args.overwrite)

    view_payloads = {
        "TRB": {
            "matrices": {key: receptor_results["TRB"][key] for key in [
                "train_base", "test_base", "train_descriptive", "test_descriptive", "test_final"
            ]},
            "manifest": receptor_results["TRB"]["manifest"],
            "sample_key": "TRB library ID; patient retained as trace metadata",
            "inputs": {
                "train_metadata": paths["train_metadata"],
                "test_metadata": paths["test_metadata"],
                "train_02": paths["trb_train_02"],
                "test_02": paths["trb_test_02"],
                "train_03_descriptive": paths["trb_train_03_descriptive"],
                "test_03_descriptive": paths["trb_test_03_descriptive"],
                "test_03_reference": paths["trb_test_03_reference"],
                "test_03_definition": paths["trb_test_03_definition"],
            },
        },
        "IGH": {
            "matrices": {key: receptor_results["IGH"][key] for key in [
                "train_base", "test_base", "train_descriptive", "test_descriptive", "test_final"
            ]},
            "manifest": receptor_results["IGH"]["manifest"],
            "sample_key": "IGH library ID; patient retained as trace metadata",
            "inputs": {
                "train_metadata": paths["train_metadata"],
                "test_metadata": paths["test_metadata"],
                "train_02": paths["igh_train_02"],
                "test_02": paths["igh_test_02"],
                "train_03_descriptive": paths["igh_train_03_descriptive"],
                "test_03_descriptive": paths["igh_test_03_descriptive"],
                "test_03_reference": paths["igh_test_03_reference"],
                "test_03_definition": paths["igh_test_03_definition"],
            },
        },
        "TRB_IGH": {
            "matrices": joint_matrices,
            "manifest": joint_manifest,
            "sample_key": "patient ID; receptor-derived columns prefixed trb_ and igh_",
            "inputs": {
                "train_metadata": paths["train_metadata"],
                "test_metadata": paths["test_metadata"],
                "TRB_train_02": paths["trb_train_02"],
                "TRB_test_02": paths["trb_test_02"],
                "IGH_train_02": paths["igh_train_02"],
                "IGH_test_02": paths["igh_test_02"],
                "TRB_train_03_descriptive": paths["trb_train_03_descriptive"],
                "TRB_test_03_descriptive": paths["trb_test_03_descriptive"],
                "IGH_train_03_descriptive": paths["igh_train_03_descriptive"],
                "IGH_test_03_descriptive": paths["igh_test_03_descriptive"],
                "TRB_test_03_reference": paths["trb_test_03_reference"],
                "IGH_test_03_reference": paths["igh_test_03_reference"],
                "TRB_test_03_definition": paths["trb_test_03_definition"],
                "IGH_test_03_definition": paths["igh_test_03_definition"],
            },
        },
    }

    runtime = time.time() - started

    if args.preflight_only:
        for view, payload in view_payloads.items():
            matrices = payload["matrices"]
            print(
                f"{view}: train_base={matrices['train_base'].shape}, "
                f"train_desc={matrices['train_descriptive'].shape}, "
                f"test_base={matrices['test_base'].shape}, "
                f"test_desc={matrices['test_descriptive'].shape}, "
                f"test_final={matrices['test_final'].shape}, "
                f"manifest={payload['manifest'].shape}"
            )
        print("Preflight-only completed successfully; no output files were written.")
        print(f"Runtime: {runtime:.2f}s")
        return 0

    for view, payload in view_payloads.items():
        train_outputs, test_outputs = outputs_by_view[view]
        matrices = payload["matrices"]
        manifest = payload["manifest"]

        shapes = {
            "04_train_base_feature_matrix.csv": matrices["train_base"].shape,
            "04_train_descriptive_feature_matrix.csv": matrices["train_descriptive"].shape,
            "04_test_base_feature_matrix.csv": matrices["test_base"].shape,
            "04_test_descriptive_feature_matrix.csv": matrices["test_descriptive"].shape,
            "04_test_final_feature_matrix.csv": matrices["test_final"].shape,
            "04_feature_manifest.csv": manifest.shape,
        }

        provenance = build_provenance(
            analysis_view=view,
            inputs=payload["inputs"],
            shapes=shapes,
            metadata={
                "train_sample_count": len(train_metadata),
                "test_sample_count": len(test_metadata),
                "scheme_name": args.scheme_name,
                "sample_key": payload["sample_key"],
            },
        )

        if args.mode in {"train", "both"}:
            write_dataframe_atomic(matrices["train_base"], train_outputs["train_base"])
            write_dataframe_atomic(
                matrices["train_descriptive"], train_outputs["train_descriptive"]
            )
            write_dataframe_atomic(manifest, train_outputs["manifest"])
            write_text_atomic(
                json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
                train_outputs["provenance"],
            )
            train_summary_outputs = {
                key: value for key, value in train_outputs.items()
            }
            write_text_atomic(
                make_summary(
                    analysis_view=f"{view} TRAIN",
                    inputs=payload["inputs"],
                    outputs=train_summary_outputs,
                    shapes={
                        "04_train_base_feature_matrix.csv": matrices["train_base"].shape,
                        "04_train_descriptive_feature_matrix.csv": matrices["train_descriptive"].shape,
                        "04_feature_manifest.csv": manifest.shape,
                    },
                    runtime=runtime,
                    sample_id_description=payload["sample_key"],
                ),
                train_outputs["summary"],
            )

        if args.mode in {"test", "both"}:
            write_dataframe_atomic(matrices["test_base"], test_outputs["test_base"])
            write_dataframe_atomic(
                matrices["test_descriptive"], test_outputs["test_descriptive"]
            )
            write_dataframe_atomic(matrices["test_final"], test_outputs["test_final"])
            write_text_atomic(
                make_summary(
                    analysis_view=f"{view} TEST",
                    inputs=payload["inputs"],
                    outputs=test_outputs,
                    shapes={
                        "04_test_base_feature_matrix.csv": matrices["test_base"].shape,
                        "04_test_descriptive_feature_matrix.csv": matrices["test_descriptive"].shape,
                        "04_test_final_feature_matrix.csv": matrices["test_final"].shape,
                    },
                    runtime=runtime,
                    sample_id_description=payload["sample_key"],
                ),
                test_outputs["summary"],
            )

        print(
            f"{view}: train_base={matrices['train_base'].shape}, "
            f"train_desc={matrices['train_descriptive'].shape}, "
            f"test_base={matrices['test_base'].shape}, "
            f"test_desc={matrices['test_descriptive'].shape}, "
            f"test_final={matrices['test_final'].shape}, "
            f"manifest={manifest.shape}"
        )

    print("")
    print("04 paired TRB+IGH feature-matrix construction completed successfully.")
    print(f"Total runtime: {runtime:.2f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
