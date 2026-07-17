#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build 02_sample_level_features from 01_AA_clone_table outputs.

TRAIN mode
----------
1. Read per-sample AA clone tables and the 01 sample summary.
2. Build core sample-level features.
3. Enumerate amino-acid 3-mers from TRAIN samples only.
4. Calculate training prevalence and variance.
5. Fix the retained 3-mer vocabulary.
6. Write unweighted and weighted sample × 3-mer matrices.

TEST mode
---------
1. Read per-sample AA clone tables and the 01 sample summary.
2. Build core sample-level features using identical definitions.
3. Read the vocabulary generated from TRAIN.
4. Calculate only the retained TRAIN 3-mers.
5. A TRAIN 3-mer absent from a TEST sample is filled with zero.
6. TEST-only 3-mers are ignored.

No cohort, batch, material, sex, age, or other metadata features are added here.
The metadata file is used only to define sample IDs and sample order.

Core frequency weight:
    p_i = frequency_norm

3-mer definitions:
    unweighted_3mer_k =
        number of distinct AA clones containing k / total AA clone number

    weighted_3mer_k =
        sum_i p_i * I(k is present in clone i)

A 3-mer appearing multiple times in one clone is counted once for that clone.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.0"

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_ORDER)

# Kyte-Doolittle hydrophobicity scale.
KYTE_DOOLITTLE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# Mutually exclusive groups covering all 20 standard amino acids.
AA_GROUPS = {
    "basic": set("KRH"),
    "acidic": set("DE"),
    "polar": set("STNQCY"),
    "nonpolar": set("AVLIMFWGP"),
}

AA_INPUT_SUFFIX = "_TRB_CDR3_AA_clone_table.csv"

REQUIRED_AA_COLUMNS = {
    "sample_id",
    "cdr3_aa",
    "aa_length",
    "nt_clone_number",
    "read_count",
    "read_fraction",
    "input_frequency_sum",
    "input_cell_frequency_sum",
    "frequency",
    "frequency_norm",
    "top_nt_cdr3",
}

REQUIRED_SUMMARY_COLUMNS = {
    "sample_id",
    "input_nt_rows",
    "valid_nt_rows",
    "valid_nt_row_ratio",
    "total_reads_all_nt",
    "total_reads_valid_nt",
    "unique_valid_nt_clones",
    "aa_clone_number",
}


@dataclass
class SampleBuildLog:
    sample_id: str
    input_file: str
    status: str
    aa_clone_number: int
    total_reads_valid_aa: int
    valid_nt_clone_number: int
    frequency_norm_sum_input: float
    read_fraction_sum_input: float
    Shannon: float
    top1_frequency: float
    nonzero_unweighted_3mers: int
    nonzero_weighted_3mers: int
    invalid_aa_sequence_count: int
    summary_mismatch_count: int
    summary_mismatch_detail: str
    processing_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build core and 3-mer sample-level TRB AA features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        required=True,
        help="TRAIN creates the vocabulary; TEST must reuse the TRAIN vocabulary.",
    )
    parser.add_argument(
        "--aa-dir",
        required=True,
        help="01_AA_clone_table directory containing per-sample AA clone tables.",
    )
    parser.add_argument(
        "--summary-file",
        required=True,
        help="01_AA_clone_table_summary.csv generated during step 01.",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Train or test metadata CSV used to define sample IDs and order.",
    )
    parser.add_argument(
        "--id-col",
        default="libraryid",
        help="Sample ID column in metadata.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for 02 sample-level features.",
    )
    parser.add_argument(
        "--aa-input-suffix",
        default=AA_INPUT_SUFFIX,
        help="Suffix of per-sample AA clone tables.",
    )

    parser.add_argument(
        "--vocabulary-file",
        default=None,
        help=(
            "TRAIN vocabulary CSV. Required in TEST mode and ignored in TRAIN mode."
        ),
    )
    parser.add_argument(
        "--min-kmer-sample-count",
        type=int,
        default=5,
        help="Minimum number of TRAIN samples in which a 3-mer must be nonzero.",
    )
    parser.add_argument(
        "--min-kmer-prevalence",
        type=float,
        default=0.05,
        help="Minimum TRAIN sample prevalence for retaining a 3-mer.",
    )
    parser.add_argument(
        "--min-kmer-unweighted-variance",
        type=float,
        default=1e-12,
        help="Minimum TRAIN variance of the unweighted 3-mer feature.",
    )
    parser.add_argument(
        "--min-kmer-weighted-variance",
        type=float,
        default=1e-12,
        help="Minimum TRAIN variance of the weighted 3-mer feature.",
    )
    parser.add_argument(
        "--max-kmers",
        type=int,
        default=0,
        help=(
            "Optional maximum number of retained 3-mers after filtering. "
            "0 means no maximum. Ranking uses prevalence then total variance."
        ),
    )

    parser.add_argument(
        "--norm-tolerance",
        type=float,
        default=1e-8,
        help="Tolerance for frequency_norm and read_fraction sum checks.",
    )
    parser.add_argument(
        "--summary-mismatch-policy",
        choices=["error", "warn"],
        default="error",
        help="How to handle disagreement between AA tables and the 01 summary.",
    )
    parser.add_argument(
        "--invalid-aa-policy",
        choices=["error", "skip"],
        default="error",
        help="How to handle a non-standard or empty AA sequence in an AA table.",
    )

    parser.add_argument(
        "--write-merged",
        action="store_true",
        help=(
            "Also write core + unweighted 3-mer + weighted 3-mer as one wide CSV."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform complete calculations and QC without writing output files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing 02 output files.",
    )

    args = parser.parse_args()

    if not (0.0 <= args.min_kmer_prevalence <= 1.0):
        parser.error("--min-kmer-prevalence must be between 0 and 1.")
    if args.min_kmer_sample_count < 1:
        parser.error("--min-kmer-sample-count must be >= 1.")
    if args.min_kmer_unweighted_variance < 0:
        parser.error("--min-kmer-unweighted-variance must be >= 0.")
    if args.min_kmer_weighted_variance < 0:
        parser.error("--min-kmer-weighted-variance must be >= 0.")
    if args.max_kmers < 0:
        parser.error("--max-kmers must be >= 0.")
    if args.mode == "test" and not args.vocabulary_file:
        parser.error("--vocabulary-file is required in TEST mode.")

    return args


def read_metadata_ids(path: Path, id_col: str) -> List[str]:
    df = pd.read_csv(path, dtype=str)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)

    if id_col not in df.columns:
        raise ValueError(
            f"metadata 中不存在 ID 列 '{id_col}'。可用列: {list(df.columns)}"
        )

    ids = df[id_col].astype(str).str.strip()
    if ids.eq("").any() or ids.isna().any():
        raise ValueError(f"metadata 的 '{id_col}' 中存在空值。")
    if ids.duplicated().any():
        duplicate_values = ids[ids.duplicated(keep=False)].unique().tolist()[:10]
        raise ValueError(
            f"metadata 的 '{id_col}' 中存在重复 ID，例如: {duplicate_values}"
        )
    return ids.tolist()


def read_step01_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = sorted(REQUIRED_SUMMARY_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            f"01 summary 缺少必要列: {missing}\n"
            f"实际列: {list(df.columns)}"
        )
    if df["sample_id"].astype(str).duplicated().any():
        raise ValueError("01 summary 中 sample_id 存在重复。")
    df["sample_id"] = df["sample_id"].astype(str)
    return df.set_index("sample_id", drop=False)


def expected_output_paths(output_dir: Path, mode: str, write_merged: bool) -> Dict[str, Path]:
    outputs = {
        "core": output_dir / "02_core_sample_features.csv",
        "unweighted": output_dir / "02_3mer_unweighted_features.csv",
        "weighted": output_dir / "02_3mer_weighted_features.csv",
        "build_log": output_dir / "02_sample_level_features_build_log.csv",
        "summary": output_dir / "02_sample_level_features_build_summary.md",
    }
    if mode == "train":
        outputs["vocabulary"] = output_dir / "02_3mer_vocabulary.csv"
    else:
        outputs["vocabulary_used"] = output_dir / "02_3mer_vocabulary_used.csv"
    if write_merged:
        outputs["merged"] = output_dir / "02_sample_level_features_merged.csv"
    return outputs


def check_overwrite(paths: Mapping[str, Path], overwrite: bool) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        shown = "\n".join(f"  - {p}" for p in existing)
        raise FileExistsError(
            "以下输出文件已存在。若确认覆盖，请添加 --overwrite：\n" + shown
        )


def safe_numeric(series: pd.Series, column: str, path: Path) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        count = int(values.isna().sum())
        raise ValueError(f"{path} 的 {column} 有 {count} 个无法解析的数值。")
    if (~np.isfinite(values.to_numpy(dtype=float))).any():
        raise ValueError(f"{path} 的 {column} 包含非有限数值。")
    return values


def calculate_gini(probabilities: np.ndarray) -> float:
    """Gini coefficient of clone frequencies."""
    x = np.asarray(probabilities, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if np.any(x < 0):
        raise ValueError("Gini 输入包含负数。")
    total = float(x.sum())
    if total <= 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    index = np.arange(1, n + 1, dtype=float)
    gini = (2.0 * float(np.dot(index, x)) / (n * total)) - ((n + 1.0) / n)
    # Numerical rounding can create tiny values outside [0,1].
    return float(min(1.0, max(0.0, gini)))


def unique_3mers(sequence: str) -> Set[str]:
    if len(sequence) < 3:
        return set()
    return {sequence[i:i + 3] for i in range(len(sequence) - 2)}


def mismatch_message(
    name: str,
    computed: float,
    expected: float,
    tolerance: float = 0.0,
) -> Optional[str]:
    if tolerance == 0.0:
        mismatch = computed != expected
    else:
        mismatch = not math.isclose(
            float(computed), float(expected),
            rel_tol=tolerance, abs_tol=tolerance
        )
    if mismatch:
        return f"{name}: AA表计算={computed}, 01 summary={expected}"
    return None


def process_one_sample(
    sample_id: str,
    aa_file: Path,
    summary_row: pd.Series,
    retained_kmers: Optional[Set[str]],
    enumerate_all_kmers: bool,
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], Dict[str, float], Dict[str, float], SampleBuildLog]:
    started = time.time()

    df = pd.read_csv(aa_file)
    missing = sorted(REQUIRED_AA_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            f"{aa_file} 缺少必要列: {missing}\n实际列: {list(df.columns)}"
        )
    if df.empty:
        raise ValueError(f"AA clone table 为空: {aa_file}")

    file_sample_ids = df["sample_id"].astype(str).unique().tolist()
    if file_sample_ids != [sample_id]:
        raise ValueError(
            f"{aa_file} 内 sample_id 与预期不一致。"
            f"预期={sample_id}, 文件中={file_sample_ids[:10]}"
        )

    df["cdr3_aa"] = df["cdr3_aa"].astype(str).str.strip().str.upper()
    df["aa_length"] = safe_numeric(df["aa_length"], "aa_length", aa_file).astype(int)
    df["nt_clone_number"] = safe_numeric(
        df["nt_clone_number"], "nt_clone_number", aa_file
    ).astype(int)
    df["read_count"] = safe_numeric(df["read_count"], "read_count", aa_file)
    df["read_fraction"] = safe_numeric(
        df["read_fraction"], "read_fraction", aa_file
    )
    df["frequency_norm"] = safe_numeric(
        df["frequency_norm"], "frequency_norm", aa_file
    )

    if (df["read_count"] < 0).any():
        raise ValueError(f"{aa_file} 的 read_count 存在负数。")
    if (df["nt_clone_number"] < 1).any():
        raise ValueError(f"{aa_file} 的 nt_clone_number 存在小于 1 的值。")
    if (df["frequency_norm"] < 0).any():
        raise ValueError(f"{aa_file} 的 frequency_norm 存在负数。")

    invalid_sequence_mask = (
        df["cdr3_aa"].eq("")
        | df["cdr3_aa"].map(lambda x: not set(x).issubset(AA_SET))
        | (df["cdr3_aa"].str.len() != df["aa_length"])
    )
    invalid_aa_count = int(invalid_sequence_mask.sum())
    if invalid_aa_count:
        if args.invalid_aa_policy == "error":
            examples = df.loc[
                invalid_sequence_mask, ["cdr3_aa", "aa_length"]
            ].head(5).to_dict("records")
            raise ValueError(
                f"{aa_file} 有 {invalid_aa_count} 条异常 AA 序列，例如 {examples}"
            )
        df = df.loc[~invalid_sequence_mask].copy()

    if df.empty:
        raise ValueError(f"过滤异常 AA 后无可用 clone: {aa_file}")

    aa_clone_number = int(df.shape[0])
    total_reads_valid_aa = int(round(float(df["read_count"].sum())))
    valid_nt_clone_number = int(df["nt_clone_number"].sum())

    frequency_norm_sum_input = float(df["frequency_norm"].sum())
    read_fraction_sum_input = float(df["read_fraction"].sum())

    if frequency_norm_sum_input <= 0:
        raise ValueError(f"{aa_file} 的 frequency_norm 总和 <= 0。")
    if abs(frequency_norm_sum_input - 1.0) > args.norm_tolerance:
        raise ValueError(
            f"{aa_file} 的 frequency_norm 总和为 {frequency_norm_sum_input:.12g}，"
            f"超出容差 {args.norm_tolerance}。"
        )
    if abs(read_fraction_sum_input - 1.0) > args.norm_tolerance:
        raise ValueError(
            f"{aa_file} 的 read_fraction 总和为 {read_fraction_sum_input:.12g}，"
            f"超出容差 {args.norm_tolerance}。"
        )

    p = df["frequency_norm"].to_numpy(dtype=float)
    p = p / p.sum()

    # Summary consistency checks.
    mismatch_details: List[str] = []
    checks = [
        mismatch_message(
            "total_reads_valid_aa",
            total_reads_valid_aa,
            int(summary_row["total_reads_valid_nt"]),
        ),
        mismatch_message(
            "valid_nt_clone_number",
            valid_nt_clone_number,
            int(summary_row["unique_valid_nt_clones"]),
        ),
        mismatch_message(
            "aa_clone_number",
            aa_clone_number,
            int(summary_row["aa_clone_number"]),
        ),
    ]
    mismatch_details = [x for x in checks if x is not None]
    if mismatch_details and args.summary_mismatch_policy == "error":
        raise ValueError(
            f"{sample_id} 的 AA 表与 01 summary 不一致："
            + "；".join(mismatch_details)
        )

    # Diversity.
    positive_p = p[p > 0]
    Shannon = float(-np.sum(positive_p * np.log(positive_p)))
    Simpson = float(np.sum(p ** 2))
    inverse_Simpson = float(1.0 / Simpson) if Simpson > 0 else float("nan")
    Gini = calculate_gini(p)
    if aa_clone_number > 1:
        clonality = float(1.0 - Shannon / math.log(aa_clone_number))
        clonality = min(1.0, max(0.0, clonality))
    else:
        clonality = 1.0

    sorted_p = np.sort(p)[::-1]
    top_values = {
        "top1_frequency": float(sorted_p[:1].sum()),
        "top5_cumulative_frequency": float(sorted_p[:5].sum()),
        "top10_cumulative_frequency": float(sorted_p[:10].sum()),
        "top20_cumulative_frequency": float(sorted_p[:20].sum()),
        "top50_cumulative_frequency": float(sorted_p[:50].sum()),
    }

    # Length.
    lengths = df["aa_length"].to_numpy(dtype=float)
    mean_aa_length = float(np.mean(lengths))
    median_aa_length = float(np.median(lengths))
    sd_aa_length = float(np.std(lengths, ddof=0))
    weighted_mean_aa_length = float(np.sum(p * lengths))
    weighted_sd_aa_length = float(
        np.sqrt(np.sum(p * (lengths - weighted_mean_aa_length) ** 2))
    )

    weighted_length_features: Dict[str, float] = {}
    covered = np.zeros_like(p, dtype=bool)
    for length_value in range(8, 26):
        mask = lengths == length_value
        covered |= mask
        weighted_length_features[f"weighted_length_{length_value}_freq"] = float(
            p[mask].sum()
        )
    weighted_length_features["weighted_length_other_freq"] = float(p[~covered].sum())

    # Sequence-level composition and physicochemical summaries.
    unweighted_aa_sum = {aa: 0.0 for aa in AA_ORDER}
    weighted_aa_sum = {aa: 0.0 for aa in AA_ORDER}

    hydro_sum = 0.0
    weighted_hydro_sum = 0.0
    charge_sum = 0.0
    weighted_charge_sum = 0.0
    aromaticity_sum = 0.0
    weighted_aromaticity_sum = 0.0

    group_sum = {group: 0.0 for group in AA_GROUPS}
    weighted_group_sum = {group: 0.0 for group in AA_GROUPS}

    kmer_clone_counts: Dict[str, int] = defaultdict(int)
    kmer_weight_sums: Dict[str, float] = defaultdict(float)

    sequences = df["cdr3_aa"].tolist()

    for sequence, clone_weight in zip(sequences, p):
        length = len(sequence)
        counts = Counter(sequence)

        for aa in AA_ORDER:
            ratio = counts.get(aa, 0) / length
            unweighted_aa_sum[aa] += ratio
            weighted_aa_sum[aa] += float(clone_weight) * ratio

        hydrophobicity_i = sum(
            KYTE_DOOLITTLE[aa] * count for aa, count in counts.items()
        ) / length
        charge_i = (
            counts.get("K", 0)
            + counts.get("R", 0)
            + 0.1 * counts.get("H", 0)
            - counts.get("D", 0)
            - counts.get("E", 0)
        ) / length
        aromaticity_i = (
            counts.get("F", 0) + counts.get("W", 0) + counts.get("Y", 0)
        ) / length

        hydro_sum += hydrophobicity_i
        weighted_hydro_sum += float(clone_weight) * hydrophobicity_i
        charge_sum += charge_i
        weighted_charge_sum += float(clone_weight) * charge_i
        aromaticity_sum += aromaticity_i
        weighted_aromaticity_sum += float(clone_weight) * aromaticity_i

        for group, residues in AA_GROUPS.items():
            ratio = sum(counts.get(aa, 0) for aa in residues) / length
            group_sum[group] += ratio
            weighted_group_sum[group] += float(clone_weight) * ratio

        kmers = unique_3mers(sequence)
        if retained_kmers is not None:
            kmers = kmers.intersection(retained_kmers)
        elif not enumerate_all_kmers:
            kmers = set()

        for kmer in kmers:
            kmer_clone_counts[kmer] += 1
            kmer_weight_sums[kmer] += float(clone_weight)

    unweighted_aa_features = {
        f"{aa}_frequency": unweighted_aa_sum[aa] / aa_clone_number
        for aa in AA_ORDER
    }
    weighted_aa_features = {
        f"weighted_{aa}_frequency": weighted_aa_sum[aa]
        for aa in AA_ORDER
    }

    unweighted_kmers = {
        kmer: count / aa_clone_number
        for kmer, count in kmer_clone_counts.items()
    }
    weighted_kmers = dict(kmer_weight_sums)

    core: Dict[str, object] = {
        "sample_id": sample_id,
        "total_reads_all_nt": int(summary_row["total_reads_all_nt"]),
        "total_reads_valid_aa": total_reads_valid_aa,
        "valid_read_ratio": (
            total_reads_valid_aa / int(summary_row["total_reads_all_nt"])
            if int(summary_row["total_reads_all_nt"]) > 0
            else float("nan")
        ),
        "nt_clone_number_all": int(summary_row["input_nt_rows"]),
        "valid_nt_clone_number": valid_nt_clone_number,
        "valid_nt_row_ratio": float(summary_row["valid_nt_row_ratio"]),
        "aa_clone_number": aa_clone_number,
        "log1p_aa_clone_number": float(math.log1p(aa_clone_number)),
        "Shannon": Shannon,
        "Simpson": Simpson,
        "inverse_Simpson": inverse_Simpson,
        "Gini": Gini,
        "clonality": clonality,
        **top_values,
        "mean_aa_length": mean_aa_length,
        "median_aa_length": median_aa_length,
        "sd_aa_length": sd_aa_length,
        "weighted_mean_aa_length": weighted_mean_aa_length,
        "weighted_sd_aa_length": weighted_sd_aa_length,
        **weighted_length_features,
        **unweighted_aa_features,
        **weighted_aa_features,
        "mean_hydrophobicity": hydro_sum / aa_clone_number,
        "weighted_hydrophobicity": weighted_hydro_sum,
        "mean_charge": charge_sum / aa_clone_number,
        "weighted_charge": weighted_charge_sum,
        "aromaticity": aromaticity_sum / aa_clone_number,
        "weighted_aromaticity": weighted_aromaticity_sum,
        "basic_aa_ratio": group_sum["basic"] / aa_clone_number,
        "weighted_basic_aa_ratio": weighted_group_sum["basic"],
        "acidic_aa_ratio": group_sum["acidic"] / aa_clone_number,
        "weighted_acidic_aa_ratio": weighted_group_sum["acidic"],
        "polar_aa_ratio": group_sum["polar"] / aa_clone_number,
        "weighted_polar_aa_ratio": weighted_group_sum["polar"],
        "nonpolar_aa_ratio": group_sum["nonpolar"] / aa_clone_number,
        "weighted_nonpolar_aa_ratio": weighted_group_sum["nonpolar"],
    }

    log = SampleBuildLog(
        sample_id=sample_id,
        input_file=str(aa_file),
        status="CALCULATED",
        aa_clone_number=aa_clone_number,
        total_reads_valid_aa=total_reads_valid_aa,
        valid_nt_clone_number=valid_nt_clone_number,
        frequency_norm_sum_input=frequency_norm_sum_input,
        read_fraction_sum_input=read_fraction_sum_input,
        Shannon=Shannon,
        top1_frequency=top_values["top1_frequency"],
        nonzero_unweighted_3mers=len(unweighted_kmers),
        nonzero_weighted_3mers=len(weighted_kmers),
        invalid_aa_sequence_count=invalid_aa_count,
        summary_mismatch_count=len(mismatch_details),
        summary_mismatch_detail="; ".join(mismatch_details),
        processing_seconds=time.time() - started,
    )

    return core, unweighted_kmers, weighted_kmers, log


def determine_vocabulary(
    sample_ids: Sequence[str],
    unweighted_by_sample: Mapping[str, Mapping[str, float]],
    weighted_by_sample: Mapping[str, Mapping[str, float]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    all_kmers: Set[str] = set()
    for values in unweighted_by_sample.values():
        all_kmers.update(values.keys())
    for values in weighted_by_sample.values():
        all_kmers.update(values.keys())

    n_samples = len(sample_ids)
    rows: List[Dict[str, object]] = []

    for kmer in sorted(all_kmers):
        unweighted_values = np.array(
            [unweighted_by_sample[sample_id].get(kmer, 0.0) for sample_id in sample_ids],
            dtype=float,
        )
        weighted_values = np.array(
            [weighted_by_sample[sample_id].get(kmer, 0.0) for sample_id in sample_ids],
            dtype=float,
        )

        sample_count = int(np.count_nonzero(unweighted_values > 0))
        prevalence = sample_count / n_samples
        mean_unweighted = float(np.mean(unweighted_values))
        variance_unweighted = float(np.var(unweighted_values, ddof=0))
        mean_weighted = float(np.mean(weighted_values))
        variance_weighted = float(np.var(weighted_values, ddof=0))

        reasons: List[str] = []
        if sample_count < args.min_kmer_sample_count:
            reasons.append("low_sample_count")
        if prevalence < args.min_kmer_prevalence:
            reasons.append("low_prevalence")
        if (
            variance_unweighted < args.min_kmer_unweighted_variance
            and variance_weighted < args.min_kmer_weighted_variance
        ):
            reasons.append("low_variance_both")

        keep = len(reasons) == 0
        rows.append(
            {
                "kmer": kmer,
                "training_sample_count": sample_count,
                "training_sample_prevalence": prevalence,
                "mean_unweighted_value": mean_unweighted,
                "variance_unweighted_value": variance_unweighted,
                "mean_weighted_value": mean_weighted,
                "variance_weighted_value": variance_weighted,
                "keep": keep,
                "filter_reason": "retained" if keep else ";".join(reasons),
            }
        )

    vocab = pd.DataFrame(rows)
    if vocab.empty:
        raise ValueError("训练集中没有发现任何合法 3-mer。")

    # Optional cap after the ordinary filters.
    if args.max_kmers > 0:
        kept = vocab[vocab["keep"]].copy()
        if kept.shape[0] > args.max_kmers:
            kept["ranking_total_variance"] = (
                kept["variance_unweighted_value"]
                + kept["variance_weighted_value"]
            )
            kept = kept.sort_values(
                [
                    "training_sample_prevalence",
                    "ranking_total_variance",
                    "kmer",
                ],
                ascending=[False, False, True],
            )
            keep_set = set(kept.head(args.max_kmers)["kmer"])
            cap_mask = vocab["keep"] & ~vocab["kmer"].isin(keep_set)
            vocab.loc[cap_mask, "keep"] = False
            vocab.loc[cap_mask, "filter_reason"] = "excluded_by_max_kmers"

    vocab["keep"] = vocab["keep"].astype(bool)
    vocab = vocab.sort_values(
        ["keep", "training_sample_prevalence", "kmer"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    vocab.insert(0, "vocabulary_rank", np.arange(1, len(vocab) + 1))
    return vocab


def read_training_vocabulary(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    vocab = pd.read_csv(path)
    required = {
        "kmer",
        "training_sample_count",
        "training_sample_prevalence",
        "mean_unweighted_value",
        "mean_weighted_value",
        "keep",
        "filter_reason",
    }
    missing = sorted(required - set(vocab.columns))
    if missing:
        raise ValueError(f"训练 vocabulary 缺少必要列: {missing}")

    keep_raw = vocab["keep"]
    if keep_raw.dtype == bool:
        keep_mask = keep_raw
    else:
        keep_mask = (
            keep_raw.astype(str).str.strip().str.lower()
            .isin({"true", "1", "yes"})
        )

    retained = vocab.loc[keep_mask, "kmer"].astype(str).tolist()
    if not retained:
        raise ValueError("训练 vocabulary 中没有 keep=True 的 3-mer。")
    if len(retained) != len(set(retained)):
        raise ValueError("训练 vocabulary 中保留的 kmer 存在重复。")
    return vocab, retained


def build_kmer_matrix(
    sample_ids: Sequence[str],
    values_by_sample: Mapping[str, Mapping[str, float]],
    kmers: Sequence[str],
    prefix: str,
) -> pd.DataFrame:
    matrix = np.zeros((len(sample_ids), len(kmers)), dtype=np.float64)
    kmer_index = {kmer: i for i, kmer in enumerate(kmers)}

    for row_idx, sample_id in enumerate(sample_ids):
        for kmer, value in values_by_sample[sample_id].items():
            col_idx = kmer_index.get(kmer)
            if col_idx is not None:
                matrix[row_idx, col_idx] = value

    columns = [f"{prefix}_3mer_{kmer}" for kmer in kmers]
    out = pd.DataFrame(matrix, columns=columns)
    out.insert(0, "sample_id", list(sample_ids))
    return out


def write_dataframe_atomic(df: pd.DataFrame, path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(path)


def make_markdown_summary(
    args: argparse.Namespace,
    sample_ids: Sequence[str],
    core_df: pd.DataFrame,
    vocab_df: pd.DataFrame,
    retained_kmers: Sequence[str],
    logs: Sequence[SampleBuildLog],
    outputs: Mapping[str, Path],
    total_seconds: float,
) -> str:
    lines: List[str] = [
        "# 02 Sample-level Features Build Summary",
        "",
        "## Configuration",
        "",
        f"- Script version: `{SCRIPT_VERSION}`",
        f"- Mode: `{args.mode}`",
        f"- AA directory: `{Path(args.aa_dir).resolve()}`",
        f"- Step-01 summary: `{Path(args.summary_file).resolve()}`",
        f"- Metadata: `{Path(args.metadata).resolve()}`",
        f"- Sample ID column: `{args.id_col}`",
        f"- Output directory: `{Path(args.output_dir).resolve()}`",
        f"- Samples: **{len(sample_ids)}**",
        f"- Dry run: `{args.dry_run}`",
        f"- Write merged: `{args.write_merged}`",
        "",
        "## Core feature definitions",
        "",
        "- Diversity and expansion features use `frequency_norm` as clone probability.",
        "- Shannon uses natural logarithm.",
        "- Simpson is concentration: `sum(p_i^2)`.",
        "- inverse_Simpson is `1 / Simpson`.",
        "- clonality is `1 - Shannon / log(S)` for `S > 1`.",
        "- Length standard deviations use population definition (`ddof=0`).",
        "- Hydrophobicity uses the Kyte-Doolittle scale and is averaged per residue.",
        "- Charge per clone is `(K + R + 0.1H - D - E) / L`.",
        "- Aromaticity per clone is `(F + W + Y) / L`.",
        "- AA groups are mutually exclusive: basic=KRH, acidic=DE, polar=STNQCY, nonpolar=AVLIMFWGP.",
        "",
        "## 3-mer definition",
        "",
        "- A 3-mer is counted at most once within one AA clone.",
        "- Unweighted value = fraction of distinct AA clones containing the 3-mer.",
        "- Weighted value = sum of `frequency_norm` for AA clones containing the 3-mer.",
        "",
        "## Results",
        "",
        f"- Successfully calculated samples: **{len(core_df)}**",
        f"- Core feature columns including sample_id: **{core_df.shape[1]}**",
        f"- Candidate vocabulary rows: **{len(vocab_df)}**",
        f"- Retained 3-mers: **{len(retained_kmers)}**",
        f"- Samples with summary mismatch warnings: **{sum(x.summary_mismatch_count > 0 for x in logs)}**",
        f"- Total runtime: **{total_seconds:.2f} seconds**",
        "",
    ]

    if args.mode == "train":
        reason_counts = vocab_df["filter_reason"].value_counts().to_dict()
        lines.extend(["## Vocabulary filtering", ""])
        for reason, count in reason_counts.items():
            lines.append(f"- `{reason}`: **{count}**")
        lines.append("")

    lines.extend(
        [
            "## Core feature QC overview",
            "",
            f"- `frequency_norm` input-sum range: "
            f"{min(x.frequency_norm_sum_input for x in logs):.12g} – "
            f"{max(x.frequency_norm_sum_input for x in logs):.12g}",
            f"- `read_fraction` input-sum range: "
            f"{min(x.read_fraction_sum_input for x in logs):.12g} – "
            f"{max(x.read_fraction_sum_input for x in logs):.12g}",
            f"- Shannon range: {core_df['Shannon'].min():.6g} – {core_df['Shannon'].max():.6g}",
            f"- AA clone number range: "
            f"{int(core_df['aa_clone_number'].min())} – "
            f"{int(core_df['aa_clone_number'].max())}",
            "",
            "## Output files",
            "",
        ]
    )
    for name, path in outputs.items():
        lines.append(f"- `{name}`: `{path}`")

    lines.extend(
        [
            "",
            "## Modeling notes",
            "",
            "- `frequency_norm` is a clone-level weight and is not itself a sample-level model feature.",
            "- QC/depth columns are retained for auditing; they should not all be included in the primary disease model by default.",
            "- Before a linear model, one reference column may be removed from each compositional block, e.g. "
            "`Y_frequency`, `weighted_Y_frequency`, `weighted_length_other_freq`, "
            "`nonpolar_aa_ratio`, and `weighted_nonpolar_aa_ratio`.",
            "- 3-mer vocabulary selection must remain training-only. TEST must reuse the fixed TRAIN vocabulary.",
            "",
        ]
    )
    return "\n".join(lines)


def print_config(
    args: argparse.Namespace,
    sample_ids: Sequence[str],
    outputs: Mapping[str, Path],
) -> None:
    print("")
    print("=" * 78)
    print("02_sample_level_features：运行配置")
    print("=" * 78)
    print(f"脚本版本: {SCRIPT_VERSION}")
    print(f"运行模式: {args.mode.upper()}")
    print(f"AA clone 表目录: {Path(args.aa_dir).resolve()}")
    print(f"01 样本总结表: {Path(args.summary_file).resolve()}")
    print(f"metadata: {Path(args.metadata).resolve()}")
    print(f"样本 ID 列: {args.id_col}")
    print(f"待处理样本数: {len(sample_ids)}")
    print(f"输出目录: {Path(args.output_dir).resolve()}")
    if args.mode == "train":
        print(f"3-mer 最低样本数: {args.min_kmer_sample_count}")
        print(f"3-mer 最低 prevalence: {args.min_kmer_prevalence}")
        print(
            "3-mer 最低 variance: "
            f"unweighted={args.min_kmer_unweighted_variance}, "
            f"weighted={args.min_kmer_weighted_variance}"
        )
        print(f"3-mer 最大保留数: {args.max_kmers if args.max_kmers else '不限制'}")
    else:
        print(f"训练 vocabulary: {Path(args.vocabulary_file).resolve()}")
    print(f"生成 merged 大表: {'是' if args.write_merged else '否'}")
    print(f"dry-run: {'是' if args.dry_run else '否'}")
    print(f"覆盖旧文件: {'是' if args.overwrite else '否'}")
    print("")


def print_final_report(
    args: argparse.Namespace,
    core_df: pd.DataFrame,
    vocab_df: pd.DataFrame,
    retained_kmers: Sequence[str],
    logs: Sequence[SampleBuildLog],
    outputs: Mapping[str, Path],
    total_seconds: float,
) -> None:
    print("")
    print("=" * 78)
    print("02_sample_level_features：运行完成")
    print("=" * 78)
    print(f"运行模式: {'dry-run' if args.dry_run else '正式运行'}")
    print(f"完成样本数: {len(core_df)}")
    print(f"核心特征列数（含 sample_id）: {core_df.shape[1]}")
    print(f"候选 3-mer 数: {len(vocab_df):,}")
    print(f"保留 3-mer 数: {len(retained_kmers):,}")
    print(
        f"3-mer 矩阵列数: unweighted={len(retained_kmers):,}, "
        f"weighted={len(retained_kmers):,}"
    )

    print("")
    print("[样本级核心特征概览]")
    for column in [
        "aa_clone_number",
        "Shannon",
        "Simpson",
        "Gini",
        "clonality",
        "top1_frequency",
        "weighted_mean_aa_length",
    ]:
        print(
            f"- {column}: mean={core_df[column].mean():.6g}, "
            f"median={core_df[column].median():.6g}, "
            f"min={core_df[column].min():.6g}, "
            f"max={core_df[column].max():.6g}"
        )

    mismatch_logs = [x for x in logs if x.summary_mismatch_count > 0]
    print("")
    print("[01 summary 一致性检查]")
    if mismatch_logs:
        print(f"- WARNING：{len(mismatch_logs)} 个样本存在不一致。")
        for item in mismatch_logs[:10]:
            print(f"  - {item.sample_id}: {item.summary_mismatch_detail}")
        if len(mismatch_logs) > 10:
            print(f"  - 其余 {len(mismatch_logs) - 10} 个见 build log。")
    else:
        print("- PASS：AA 表重新计算结果与 01 summary 一致。")

    if args.mode == "train":
        print("")
        print("[3-mer vocabulary 筛选]")
        reason_counts = vocab_df["filter_reason"].value_counts()
        for reason, count in reason_counts.items():
            print(f"- {reason}: {int(count):,}")

        kept_preview = vocab_df[vocab_df["keep"]].head(10)
        print("")
        print("[保留 3-mer 示例 Top 10]")
        for _, row in kept_preview.iterrows():
            print(
                f"- {row['kmer']}: sample_count={int(row['training_sample_count'])}, "
                f"prevalence={row['training_sample_prevalence']:.4f}, "
                f"mean_unweighted={row['mean_unweighted_value']:.6g}, "
                f"mean_weighted={row['mean_weighted_value']:.6g}"
            )
    else:
        print("")
        print("[TEST vocabulary 使用规则]")
        print("- vocabulary 完全来自训练集。")
        print("- 测试样本中未出现的训练 3-mer 已填 0。")
        print("- 仅在测试集中出现、但训练 vocabulary 未保留的 3-mer 已忽略。")

    print("")
    print("[组成型特征提示]")
    print("- 02 中保留全部 AA、长度和氨基酸类别比例，便于追踪和比较。")
    print("- 建模预处理时再从每个总和为 1 的特征组删除固定参考列。")
    print("- 建议参考列: Y_frequency、weighted_Y_frequency、")
    print("  weighted_length_other_freq、nonpolar_aa_ratio、weighted_nonpolar_aa_ratio。")

    if args.dry_run:
        print("")
        print("[dry-run]")
        print("- 已完成全部计算和检查，但未写入任何 02 文件。")
    else:
        print("")
        print("[输出文件]")
        for name, path in outputs.items():
            print(f"- {name}: {path}")

    print("")
    print(f"总运行时间: {total_seconds:.2f} 秒")


def main() -> int:
    args = parse_args()
    started_all = time.time()

    aa_dir = Path(args.aa_dir).expanduser().resolve()
    summary_file = Path(args.summary_file).expanduser().resolve()
    metadata_file = Path(args.metadata).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not aa_dir.is_dir():
        raise NotADirectoryError(f"AA clone 表目录不存在: {aa_dir}")
    if not summary_file.is_file():
        raise FileNotFoundError(f"01 summary 不存在: {summary_file}")
    if not metadata_file.is_file():
        raise FileNotFoundError(f"metadata 不存在: {metadata_file}")

    sample_ids = read_metadata_ids(metadata_file, args.id_col)
    summary_df = read_step01_summary(summary_file)

    missing_summary_ids = [x for x in sample_ids if x not in summary_df.index]
    extra_summary_ids = [x for x in summary_df.index if x not in set(sample_ids)]
    if missing_summary_ids:
        raise ValueError(
            f"以下 metadata 样本在 01 summary 中缺失: {missing_summary_ids[:20]}"
        )
    if extra_summary_ids:
        print(
            f"WARNING: 01 summary 中有 {len(extra_summary_ids)} 个不属于当前 metadata "
            "的样本，将忽略。"
        )

    aa_files = {
        sample_id: aa_dir / f"{sample_id}{args.aa_input_suffix}"
        for sample_id in sample_ids
    }
    missing_files = [path for path in aa_files.values() if not path.is_file()]
    if missing_files:
        preview = "\n".join(f"  - {p}" for p in missing_files[:20])
        raise FileNotFoundError(
            f"缺失 {len(missing_files)} 个 AA clone 表：\n{preview}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = expected_output_paths(output_dir, args.mode, args.write_merged)
    check_overwrite(outputs, args.overwrite)

    training_vocab_df: Optional[pd.DataFrame] = None
    retained_kmers_from_train: Optional[List[str]] = None
    retained_set: Optional[Set[str]] = None

    if args.mode == "test":
        vocab_path = Path(args.vocabulary_file).expanduser().resolve()
        if not vocab_path.is_file():
            raise FileNotFoundError(f"训练 vocabulary 不存在: {vocab_path}")
        training_vocab_df, retained_kmers_from_train = read_training_vocabulary(
            vocab_path
        )
        retained_set = set(retained_kmers_from_train)

    print_config(args, sample_ids, outputs)

    core_rows: List[Dict[str, object]] = []
    unweighted_by_sample: Dict[str, Dict[str, float]] = {}
    weighted_by_sample: Dict[str, Dict[str, float]] = {}
    logs: List[SampleBuildLog] = []

    for index, sample_id in enumerate(sample_ids, start=1):
        core, unweighted, weighted, log = process_one_sample(
            sample_id=sample_id,
            aa_file=aa_files[sample_id],
            summary_row=summary_df.loc[sample_id],
            retained_kmers=retained_set,
            enumerate_all_kmers=(args.mode == "train"),
            args=args,
        )
        core_rows.append(core)
        unweighted_by_sample[sample_id] = unweighted
        weighted_by_sample[sample_id] = weighted
        logs.append(log)

        status = "WARN" if log.summary_mismatch_count else "PASS"
        print(
            f"[{index:>3}/{len(sample_ids)}] {sample_id} | "
            f"AA={log.aa_clone_number:,} | "
            f"reads={log.total_reads_valid_aa:,} | "
            f"Shannon={log.Shannon:.4f} | "
            f"top1={log.top1_frequency:.6f} | "
            f"3mer={log.nonzero_unweighted_3mers:,} | "
            f"{status} | CALCULATED"
        )

    core_df = pd.DataFrame(core_rows)

    if args.mode == "train":
        vocab_df = determine_vocabulary(
            sample_ids=sample_ids,
            unweighted_by_sample=unweighted_by_sample,
            weighted_by_sample=weighted_by_sample,
            args=args,
        )
        retained_kmers = (
            vocab_df.loc[vocab_df["keep"], "kmer"].astype(str).tolist()
        )
    else:
        assert training_vocab_df is not None
        assert retained_kmers_from_train is not None
        vocab_df = training_vocab_df.copy()
        retained_kmers = retained_kmers_from_train

    unweighted_df = build_kmer_matrix(
        sample_ids,
        unweighted_by_sample,
        retained_kmers,
        prefix="unweighted",
    )
    weighted_df = build_kmer_matrix(
        sample_ids,
        weighted_by_sample,
        retained_kmers,
        prefix="weighted",
    )

    merged_df: Optional[pd.DataFrame] = None
    if args.write_merged:
        merged_df = (
            core_df
            .merge(unweighted_df, on="sample_id", how="left", validate="one_to_one")
            .merge(weighted_df, on="sample_id", how="left", validate="one_to_one")
        )

    # Final integrity checks.
    for name, df in [
        ("core", core_df),
        ("unweighted", unweighted_df),
        ("weighted", weighted_df),
    ]:
        if df["sample_id"].tolist() != sample_ids:
            raise RuntimeError(f"{name} 矩阵样本顺序与 metadata 不一致。")
        if df["sample_id"].duplicated().any():
            raise RuntimeError(f"{name} 矩阵 sample_id 存在重复。")
        numeric = df.drop(columns=["sample_id"]).to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise RuntimeError(f"{name} 矩阵存在 NaN 或无穷值。")

    total_seconds = time.time() - started_all

    if not args.dry_run:
        write_dataframe_atomic(core_df, outputs["core"])
        write_dataframe_atomic(unweighted_df, outputs["unweighted"])
        write_dataframe_atomic(weighted_df, outputs["weighted"])

        if args.mode == "train":
            write_dataframe_atomic(vocab_df, outputs["vocabulary"])
        else:
            vocab_used = vocab_df.copy()
            vocab_used.insert(0, "vocabulary_source", str(Path(args.vocabulary_file).resolve()))
            write_dataframe_atomic(vocab_used, outputs["vocabulary_used"])

        log_df = pd.DataFrame([asdict(x) for x in logs])
        write_dataframe_atomic(log_df, outputs["build_log"])

        if args.write_merged and merged_df is not None:
            write_dataframe_atomic(merged_df, outputs["merged"])

        markdown = make_markdown_summary(
            args=args,
            sample_ids=sample_ids,
            core_df=core_df,
            vocab_df=vocab_df,
            retained_kmers=retained_kmers,
            logs=logs,
            outputs=outputs,
            total_seconds=total_seconds,
        )
        temp_summary = outputs["summary"].with_suffix(".md.tmp")
        temp_summary.write_text(markdown + "\n", encoding="utf-8")
        temp_summary.replace(outputs["summary"])

    print_final_report(
        args=args,
        core_df=core_df,
        vocab_df=vocab_df,
        retained_kmers=retained_kmers,
        logs=logs,
        outputs=outputs,
        total_seconds=total_seconds,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
