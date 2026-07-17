#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate step-03 public CDR3-AA threshold stability using TRAINING DATA ONLY.

The script:
  1. reads the 123 training AA-clone tables and training metadata;
  2. creates a reusable sparse sample × public-candidate sequence cache;
  3. performs stratified 80% reference / 20% held-out subsampling;
  4. compares loose, main, and strict threshold schemes on identical splits;
  5. evaluates set sizes, Jaccard-to-full-training stability, sequence
     selection frequency, and held-out sample feature stability;
  6. never reads the independent test set and never calculates classification
     AUC or uses outcome performance to select thresholds.

Expected project paths are provided as defaults and can be overridden.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from array import array
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from public_feature_utils import (
    CATEGORY_ORDER,
    DEFAULT_SCHEMES,
    ThresholdScheme,
    build_reference_masks,
    classify_full_reference_category,
    jaccard_similarity,
    stratified_reference_split,
)


SCRIPT_VERSION = "1.0.0"
AA_INPUT_SUFFIX = "_TRB_CDR3_AA_clone_table.csv"
AA_SET = set("ACDEFGHIKLMNPQRSTVWY")

DEFAULT_ROOT = Path("/data/users/chenhaisheng/RA-ILD/TRB")
DEFAULT_AA_DIR = DEFAULT_ROOT / "set/train/result/01_AA_clone_table"
DEFAULT_SUMMARY = DEFAULT_AA_DIR / "01_AA_clone_table_summary.csv"
DEFAULT_METADATA = DEFAULT_ROOT / "set/train/metadata_train_70.csv"
DEFAULT_OUTPUT = (
    DEFAULT_ROOT
    / "set/train/result/03_public_features/threshold_stability"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate public CDR3-AA threshold stability on training data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--aa-dir", default=str(DEFAULT_AA_DIR))
    parser.add_argument("--summary-file", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--id-col", default="libraryid")
    parser.add_argument("--cohort-col", default="cohort")
    parser.add_argument("--batch-col", default="batch")
    parser.add_argument(
        "--strata-cols",
        default="cohort,batch",
        help="Comma-separated metadata columns used for stratified subsampling.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--aa-input-suffix", default=AA_INPUT_SUFFIX)

    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--reference-fraction", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--epsilon", type=float, default=1e-8)

    parser.add_argument(
        "--candidate-min-total-count",
        type=int,
        default=2,
        help=(
            "Only sequences present in at least this many full-training samples "
            "enter the sparse cache. With the default schemes, count=2 is exact "
            "and safe because a full-training singleton can never become public "
            "in an 80%% subset."
        ),
    )

    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Reusable sparse-cache directory; default is OUTPUT/cache.",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse an existing compatible cache instead of rebuilding it.",
    )
    parser.add_argument(
        "--sort-memory",
        default="4G",
        help="Memory setting passed to GNU sort -S.",
    )
    parser.add_argument(
        "--sort-parallel",
        type=int,
        default=4,
        help="Threads passed to GNU sort --parallel.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the large unsorted/sorted temporary TSV files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing stability output files.",
    )
    parser.add_argument(
        "--max-sequence-output-rows",
        type=int,
        default=0,
        help=(
            "Optional limit for sequence-selection output after sorting by "
            "maximum selection frequency. 0 writes all selected candidates."
        ),
    )

    args = parser.parse_args()

    if args.iterations < 2:
        parser.error("--iterations must be >= 2.")
    if not 0.0 < args.reference_fraction < 1.0:
        parser.error("--reference-fraction must be between 0 and 1.")
    if args.epsilon <= 0:
        parser.error("--epsilon must be > 0.")
    if args.candidate_min_total_count < 2:
        parser.error("--candidate-min-total-count must be >= 2.")
    if args.sort_parallel < 1:
        parser.error("--sort-parallel must be >= 1.")
    if args.max_sequence_output_rows < 0:
        parser.error("--max-sequence-output-rows must be >= 0.")

    args.strata_cols = [
        x.strip() for x in args.strata_cols.split(",") if x.strip()
    ]
    if not args.strata_cols:
        parser.error("--strata-cols must contain at least one column.")

    return args


def output_paths(output_dir: Path) -> Dict[str, Path]:
    return {
        "configuration": output_dir / "03_threshold_stability_configuration.json",
        "iterations": output_dir / "03_threshold_stability_iterations.csv",
        "overview": output_dir / "03_threshold_stability_overview.csv",
        "sequence_selection": output_dir / "03_sequence_selection_frequency.csv.gz",
        "oof_values": output_dir / "03_oof_public_feature_values.csv.gz",
        "oof_stability": output_dir / "03_oof_public_feature_stability.csv",
        "oof_overview": output_dir / "03_oof_public_feature_overview.csv",
        "full_set_sizes": output_dir / "03_full_train_reference_set_sizes.csv",
        "summary": output_dir / "03_threshold_stability_summary.md",
    }


def cache_paths(cache_dir: Path) -> Dict[str, Path]:
    return {
        "presence": cache_dir / "candidate_presence_matrix.npz",
        "frequency": cache_dir / "candidate_frequency_matrix.npz",
        "sequences": cache_dir / "candidate_sequences.txt.gz",
        "sample_info": cache_dir / "sample_info.csv",
        "clone_numbers": cache_dir / "aa_clone_numbers.npy",
        "manifest": cache_dir / "cache_manifest.json",
    }


def ensure_overwrite(paths: Mapping[str, Path], overwrite: bool) -> None:
    existing = [p for p in paths.values() if p.exists()]
    if existing and not overwrite:
        text = "\n".join(f"  - {p}" for p in existing)
        raise FileExistsError(
            "以下输出文件已存在。确认覆盖时添加 --overwrite：\n" + text
        )


def read_metadata(
    path: Path,
    id_col: str,
    cohort_col: str,
    batch_col: str,
    strata_cols: Sequence[str],
) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)

    required = {id_col, cohort_col, *strata_cols}
    if batch_col:
        required.add(batch_col)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"metadata 缺少列: {missing}; 实际列: {list(df.columns)}"
        )

    df = df.reset_index(drop=True)
    df[id_col] = df[id_col].astype(str).str.strip()
    df[cohort_col] = df[cohort_col].astype(str).str.strip().str.upper()

    if df[id_col].eq("").any() or df[id_col].isna().any():
        raise ValueError(f"metadata 的 {id_col} 存在空值。")
    if df[id_col].duplicated().any():
        duplicated = df.loc[
            df[id_col].duplicated(keep=False), id_col
        ].unique().tolist()[:10]
        raise ValueError(f"metadata 样本ID重复，例如: {duplicated}")

    cohorts = set(df[cohort_col].unique())
    if cohorts != {"RA", "ILD"}:
        raise ValueError(
            f"{cohort_col} 必须且只能包含 RA/ILD，实际为: {sorted(cohorts)}"
        )

    for col in strata_cols:
        if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"分层列 {col} 存在空值。")

    return df


def read_summary(path: Path, sample_ids: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sample_id", "aa_clone_number"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"01 summary 缺少列: {missing}")
    df["sample_id"] = df["sample_id"].astype(str)
    if df["sample_id"].duplicated().any():
        raise ValueError("01 summary 中 sample_id 重复。")
    df = df.set_index("sample_id", drop=False)

    missing_ids = [x for x in sample_ids if x not in df.index]
    if missing_ids:
        raise ValueError(
            f"metadata 中有 {len(missing_ids)} 个样本不在01 summary，例如: "
            f"{missing_ids[:10]}"
        )
    return df.loc[list(sample_ids)].copy()


def file_signature(path: Path) -> Dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def build_manifest_signature(
    metadata_file: Path,
    summary_file: Path,
    aa_files: Sequence[Path],
    sample_ids: Sequence[str],
    candidate_min_total_count: int,
) -> Dict[str, object]:
    return {
        "script_version": SCRIPT_VERSION,
        "metadata": file_signature(metadata_file),
        "summary": file_signature(summary_file),
        "sample_ids": list(sample_ids),
        "candidate_min_total_count": int(candidate_min_total_count),
        "aa_files": [file_signature(p) for p in aa_files],
    }


def manifests_compatible(
    cached: Mapping[str, object],
    expected: Mapping[str, object],
) -> bool:
    keys = [
        "metadata",
        "summary",
        "sample_ids",
        "candidate_min_total_count",
        "aa_files",
    ]
    return all(cached.get(k) == expected.get(k) for k in keys)


def write_raw_occurrence_file(
    raw_path: Path,
    aa_files: Sequence[Path],
    sample_ids: Sequence[str],
    summary_df: pd.DataFrame,
) -> Tuple[int, List[int]]:
    total_rows = 0
    observed_clone_numbers: List[int] = []

    with raw_path.open("w", encoding="utf-8", newline="") as handle:
        for index, (sample_id, aa_file) in enumerate(
            zip(sample_ids, aa_files), start=1
        ):
            started = time.time()
            df = pd.read_csv(
                aa_file,
                usecols=["sample_id", "cdr3_aa", "frequency_norm"],
                dtype={"sample_id": str, "cdr3_aa": str},
            )
            if df.empty:
                raise ValueError(f"AA clone table为空: {aa_file}")

            file_ids = df["sample_id"].astype(str).unique().tolist()
            if file_ids != [sample_id]:
                raise ValueError(
                    f"{aa_file} 内sample_id不一致；预期={sample_id}, "
                    f"文件内={file_ids[:5]}"
                )

            sequences = df["cdr3_aa"].astype(str).str.strip().str.upper()
            invalid = (
                sequences.eq("")
                | sequences.map(lambda x: not set(x).issubset(AA_SET))
            )
            if invalid.any():
                examples = sequences[invalid].head(5).tolist()
                raise ValueError(
                    f"{aa_file} 有 {int(invalid.sum())} 条非标准AA序列，例如 {examples}"
                )

            frequency = pd.to_numeric(df["frequency_norm"], errors="coerce")
            if frequency.isna().any() or (~np.isfinite(frequency)).any():
                raise ValueError(f"{aa_file} 的frequency_norm存在非法数值。")
            if (frequency < 0).any():
                raise ValueError(f"{aa_file} 的frequency_norm存在负数。")
            if not math.isclose(
                float(frequency.sum()), 1.0, rel_tol=1e-8, abs_tol=1e-8
            ):
                raise ValueError(
                    f"{aa_file} 的frequency_norm总和不是1: {frequency.sum()}"
                )

            observed = int(len(df))
            expected = int(summary_df.loc[sample_id, "aa_clone_number"])
            if observed != expected:
                raise ValueError(
                    f"{sample_id} AA行数与01 summary不一致: "
                    f"AA表={observed}, summary={expected}"
                )

            out = pd.DataFrame(
                {
                    "cdr3_aa": sequences,
                    "sample_index": index - 1,
                    "frequency_norm": frequency.astype(float),
                }
            )
            out.to_csv(
                handle,
                sep="\t",
                index=False,
                header=False,
                float_format="%.17g",
                lineterminator="\n",
            )

            total_rows += observed
            observed_clone_numbers.append(observed)
            print(
                f"[缓存输入 {index:>3}/{len(sample_ids)}] {sample_id} | "
                f"AA={observed:,} | {time.time() - started:.2f}s"
            )

    return total_rows, observed_clone_numbers


def run_external_sort(
    raw_path: Path,
    sorted_path: Path,
    temp_dir: Path,
    sort_memory: str,
    sort_parallel: int,
) -> None:
    sort_exe = shutil.which("sort")
    if not sort_exe:
        raise RuntimeError(
            "未找到GNU sort。请确认Linux环境可使用sort命令。"
        )

    command = [
        sort_exe,
        "--parallel", str(sort_parallel),
        "-S", sort_memory,
        "-T", str(temp_dir),
        "-t", "\t",
        "-k1,1",
        "-k2,2n",
        str(raw_path),
        "-o", str(sorted_path),
    ]

    env = os.environ.copy()
    env["LC_ALL"] = "C"

    print("")
    print("[缓存构建] 开始外部排序；该步骤主要消耗磁盘I/O。")
    print("命令:", " ".join(command))
    subprocess.run(command, check=True, env=env)
    print("[缓存构建] 外部排序完成。")


def stream_build_sparse_cache(
    sorted_path: Path,
    candidate_min_total_count: int,
    n_samples: int,
    sequences_path: Path,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, int, int]:
    row_indices = array("I")
    col_indices = array("I")
    frequencies = array("d")

    candidate_count = 0
    candidate_incidence_count = 0
    processed_lines = 0

    current_sequence: Optional[str] = None
    current_occurrences: Dict[int, float] = {}

    def flush_current(
        sequence: Optional[str],
        occurrences: Dict[int, float],
        sequence_writer,
    ) -> None:
        nonlocal candidate_count, candidate_incidence_count
        if sequence is None:
            return
        if len(occurrences) < candidate_min_total_count:
            return

        sequence_writer.write(sequence + "\n")
        sequence_id = candidate_count

        for sample_index, frequency in sorted(occurrences.items()):
            row_indices.append(int(sample_index))
            col_indices.append(int(sequence_id))
            frequencies.append(float(frequency))
            candidate_incidence_count += 1

        candidate_count += 1

    with sorted_path.open("r", encoding="utf-8") as source, gzip.open(
        sequences_path, "wt", encoding="utf-8", newline=""
    ) as sequence_writer:
        for line in source:
            processed_lines += 1
            sequence, sample_text, frequency_text = line.rstrip("\n").split("\t")
            sample_index = int(sample_text)
            frequency = float(frequency_text)

            if current_sequence is None:
                current_sequence = sequence

            if sequence != current_sequence:
                flush_current(
                    current_sequence,
                    current_occurrences,
                    sequence_writer,
                )
                current_sequence = sequence
                current_occurrences = {}

            # Duplicate sequence/sample rows should not occur after AA aggregation.
            # Summing is defensive and preserves one presence observation.
            current_occurrences[sample_index] = (
                current_occurrences.get(sample_index, 0.0) + frequency
            )

            if processed_lines % 1_000_000 == 0:
                print(
                    f"[缓存聚合] 已读取 {processed_lines:,} 行 | "
                    f"候选序列 {candidate_count:,} | "
                    f"候选incidence {candidate_incidence_count:,}"
                )

        flush_current(
            current_sequence,
            current_occurrences,
            sequence_writer,
        )

    if candidate_count == 0:
        raise RuntimeError("没有序列达到candidate_min_total_count。")

    rows = np.frombuffer(row_indices, dtype=np.uint32).astype(np.int32, copy=False)
    cols = np.frombuffer(col_indices, dtype=np.uint32).astype(np.int32, copy=False)
    freq = np.frombuffer(frequencies, dtype=np.float64)

    presence = sparse.csr_matrix(
        (
            np.ones(len(rows), dtype=np.uint8),
            (rows, cols),
        ),
        shape=(n_samples, candidate_count),
        dtype=np.uint8,
    )
    frequency = sparse.csr_matrix(
        (freq, (rows, cols)),
        shape=(n_samples, candidate_count),
        dtype=np.float64,
    )

    presence.sum_duplicates()
    frequency.sum_duplicates()
    presence.sort_indices()
    frequency.sort_indices()

    if presence.nnz != candidate_incidence_count:
        raise RuntimeError(
            "候选presence矩阵nnz与聚合incidence数量不一致。"
        )
    if frequency.nnz != candidate_incidence_count:
        raise RuntimeError(
            "候选frequency矩阵nnz与聚合incidence数量不一致。"
        )

    return presence, frequency, candidate_count, candidate_incidence_count


def load_sequences(path: Path) -> List[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def build_or_load_cache(
    args: argparse.Namespace,
    metadata: pd.DataFrame,
    summary_df: pd.DataFrame,
    metadata_file: Path,
    summary_file: Path,
    aa_files: Sequence[Path],
    output_dir: Path,
) -> Tuple[
    sparse.csr_matrix,
    sparse.csr_matrix,
    List[str],
    np.ndarray,
    Dict[str, object],
]:
    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir
        else output_dir / "cache"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = cache_paths(cache_dir)

    sample_ids = metadata[args.id_col].tolist()
    expected_manifest = build_manifest_signature(
        metadata_file=metadata_file,
        summary_file=summary_file,
        aa_files=aa_files,
        sample_ids=sample_ids,
        candidate_min_total_count=args.candidate_min_total_count,
    )

    cache_complete = all(p.exists() for p in paths.values())
    if args.reuse_cache and cache_complete:
        cached_manifest = json.loads(
            paths["manifest"].read_text(encoding="utf-8")
        )
        if not manifests_compatible(cached_manifest, expected_manifest):
            raise RuntimeError(
                "现有cache与当前输入文件或参数不兼容。"
                "请移除--reuse-cache以重建cache。"
            )

        print("[缓存] 读取现有稀疏cache。")
        presence = sparse.load_npz(paths["presence"]).tocsr()
        frequency = sparse.load_npz(paths["frequency"]).tocsr()
        sequences = load_sequences(paths["sequences"])
        clone_numbers = np.load(paths["clone_numbers"])

        if presence.shape != frequency.shape:
            raise RuntimeError("cache中的presence/frequency矩阵形状不一致。")
        if presence.shape[0] != len(metadata):
            raise RuntimeError("cache样本数与metadata不一致。")
        if presence.shape[1] != len(sequences):
            raise RuntimeError("cache候选序列数与sequence文件不一致。")

        return (
            presence,
            frequency,
            sequences,
            clone_numbers,
            cached_manifest,
        )

    if args.reuse_cache and not cache_complete:
        print("[缓存] --reuse-cache已指定，但cache不完整，将重新构建。")

    for p in paths.values():
        if p.exists():
            if p.is_file():
                p.unlink()
            else:
                shutil.rmtree(p)

    temp_root = cache_dir / "temporary_sort_files"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    raw_path = temp_root / "all_occurrences.unsorted.tsv"
    sorted_path = temp_root / "all_occurrences.sorted.tsv"

    print("")
    print("=" * 80)
    print("建立可复用的训练集public候选稀疏cache")
    print("=" * 80)
    print(
        "说明：完整训练集中只出现1次的私有序列不可能满足默认最宽松的"
        "最低2人public标准，因此不进入稳定性稀疏矩阵。"
    )

    total_rows, observed_clone_numbers = write_raw_occurrence_file(
        raw_path=raw_path,
        aa_files=aa_files,
        sample_ids=sample_ids,
        summary_df=summary_df,
    )

    run_external_sort(
        raw_path=raw_path,
        sorted_path=sorted_path,
        temp_dir=temp_root,
        sort_memory=args.sort_memory,
        sort_parallel=args.sort_parallel,
    )

    presence, frequency, candidate_count, incidence_count = (
        stream_build_sparse_cache(
            sorted_path=sorted_path,
            candidate_min_total_count=args.candidate_min_total_count,
            n_samples=len(metadata),
            sequences_path=paths["sequences"],
        )
    )

    clone_numbers = np.asarray(observed_clone_numbers, dtype=np.int64)

    sparse.save_npz(paths["presence"], presence, compressed=True)
    sparse.save_npz(paths["frequency"], frequency, compressed=True)
    np.save(paths["clone_numbers"], clone_numbers)

    sample_info = metadata.copy()
    sample_info.insert(0, "sample_index", np.arange(len(metadata)))
    sample_info.to_csv(paths["sample_info"], index=False)

    manifest = {
        **expected_manifest,
        "created_at_unix": time.time(),
        "total_aa_clone_rows": int(total_rows),
        "candidate_sequence_count": int(candidate_count),
        "candidate_incidence_count": int(incidence_count),
        "presence_shape": list(presence.shape),
        "presence_nnz": int(presence.nnz),
        "frequency_nnz": int(frequency.nnz),
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if not args.keep_temp:
        shutil.rmtree(temp_root)
    else:
        print(f"[缓存] 临时排序文件已保留: {temp_root}")

    sequences = load_sequences(paths["sequences"])
    return presence, frequency, sequences, clone_numbers, manifest


def dense_vector(matrix_result) -> np.ndarray:
    return np.asarray(matrix_result).reshape(-1)


def calculate_full_counts(
    presence_transposed: sparse.csr_matrix,
    cohort_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    ra_vector = (cohort_values == "RA").astype(np.int16)
    ild_vector = (cohort_values == "ILD").astype(np.int16)

    ra_counts = dense_vector(presence_transposed.dot(ra_vector)).astype(
        np.int16, copy=False
    )
    ild_counts = dense_vector(presence_transposed.dot(ild_vector)).astype(
        np.int16, copy=False
    )
    return ra_counts, ild_counts


def summarize_numeric(values: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "sd": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "sd": float(np.std(x, ddof=0)),
        "q1": float(np.quantile(x, 0.25)),
        "q3": float(np.quantile(x, 0.75)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def make_stability_overview(
    iterations_df: pd.DataFrame,
    schemes: Sequence[ThresholdScheme],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for scheme in schemes:
        subset = iterations_df[iterations_df["scheme"] == scheme.name]
        for category in CATEGORY_ORDER:
            size_col = f"{category}_set_size"
            jac_col = f"{category}_jaccard_to_full"

            size_stats = summarize_numeric(subset[size_col])
            jac_stats = summarize_numeric(subset[jac_col])
            mean_size = size_stats["mean"]
            cv = (
                size_stats["sd"] / mean_size
                if np.isfinite(mean_size) and mean_size > 0
                else np.nan
            )

            rows.append(
                {
                    "scheme": scheme.name,
                    "category": category,
                    "full_train_set_size": int(
                        subset[f"{category}_full_set_size"].iloc[0]
                    ),
                    "mean_set_size": size_stats["mean"],
                    "median_set_size": size_stats["median"],
                    "sd_set_size": size_stats["sd"],
                    "q1_set_size": size_stats["q1"],
                    "q3_set_size": size_stats["q3"],
                    "min_set_size": size_stats["min"],
                    "max_set_size": size_stats["max"],
                    "cv_set_size": cv,
                    "empty_rate": float((subset[size_col] == 0).mean()),
                    "mean_jaccard_to_full": jac_stats["mean"],
                    "median_jaccard_to_full": jac_stats["median"],
                    "min_jaccard_to_full": jac_stats["min"],
                    "max_jaccard_to_full": jac_stats["max"],
                }
            )
    return pd.DataFrame(rows)


def make_oof_stability(
    oof_df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    group_columns = ["scheme", "sample_id", "cohort", "batch"]
    for keys, group in oof_df.groupby(group_columns, sort=True):
        scheme, sample_id, cohort, batch = keys
        for feature in feature_columns:
            values = group[feature].to_numpy(dtype=float)
            mean_value = float(np.mean(values))
            sd_value = float(np.std(values, ddof=0))
            cv = (
                sd_value / abs(mean_value)
                if abs(mean_value) > 1e-12
                else np.nan
            )
            rows.append(
                {
                    "scheme": scheme,
                    "sample_id": sample_id,
                    "cohort": cohort,
                    "batch": batch,
                    "feature": feature,
                    "repeat_count": int(len(values)),
                    "mean": mean_value,
                    "median": float(np.median(values)),
                    "sd": sd_value,
                    "cv": cv,
                    "zero_rate": float(np.mean(np.isclose(values, 0.0))),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def make_oof_overview(
    oof_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for scheme, scheme_df in oof_df.groupby("scheme", sort=True):
        for feature in feature_columns:
            values = scheme_df[feature].to_numpy(dtype=float)
            sample_stability = stability_df[
                (stability_df["scheme"] == scheme)
                & (stability_df["feature"] == feature)
            ]
            finite_cv = sample_stability["cv"].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()

            rows.append(
                {
                    "scheme": scheme,
                    "feature": feature,
                    "oof_observation_count": int(len(values)),
                    "nonzero_rate": float(np.mean(~np.isclose(values, 0.0))),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "sd": float(np.std(values, ddof=0)),
                    "q1": float(np.quantile(values, 0.25)),
                    "q3": float(np.quantile(values, 0.75)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median_within_sample_cv": (
                        float(finite_cv.median()) if len(finite_cv) else np.nan
                    ),
                    "median_within_sample_zero_rate": float(
                        sample_stability["zero_rate"].median()
                    ),
                    "median_repeat_count_per_sample": float(
                        sample_stability["repeat_count"].median()
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_sequence_selection_frequency(
    path: Path,
    sequences: Sequence[str],
    schemes: Sequence[ThresholdScheme],
    selection_counts: np.ndarray,
    full_ra_counts: np.ndarray,
    full_ild_counts: np.ndarray,
    full_masks: Mapping[str, Mapping[str, np.ndarray]],
    n_iterations: int,
    n_ra: int,
    n_ild: int,
    max_rows: int,
) -> int:
    """
    Write one row per selected sequence per threshold scheme.

    selection_counts shape:
        scheme × category × candidate_sequence
    """
    written = 0
    header = [
        "scheme",
        "cdr3_aa",
        "full_total_sample_count",
        "full_total_sample_prevalence",
        "full_RA_sample_count",
        "full_RA_sample_prevalence",
        "full_ILD_sample_count",
        "full_ILD_sample_prevalence",
        "full_reference_category",
        "global_selection_count",
        "global_selection_frequency",
        "RA_specific_selection_count",
        "RA_specific_selection_frequency",
        "ILD_specific_selection_count",
        "ILD_specific_selection_frequency",
        "shared_selection_count",
        "shared_selection_frequency",
        "max_selection_frequency",
    ]

    n_total = n_ra + n_ild

    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)

        for scheme_index, scheme in enumerate(schemes):
            counts_for_scheme = selection_counts[scheme_index]
            max_frequency = (
                counts_for_scheme.max(axis=0).astype(np.float64) / n_iterations
            )
            selected_indices = np.flatnonzero(max_frequency > 0)

            if max_rows > 0 and len(selected_indices) > max_rows:
                ranking = np.argsort(max_frequency[selected_indices])[::-1]
                selected_indices = selected_indices[ranking[:max_rows]]
            else:
                selected_indices = selected_indices[
                    np.argsort(max_frequency[selected_indices])[::-1]
                ]

            full_category = classify_full_reference_category(
                full_masks[scheme.name]
            )

            for seq_index in selected_indices:
                category_counts = counts_for_scheme[:, seq_index]
                total_count = int(
                    full_ra_counts[seq_index] + full_ild_counts[seq_index]
                )
                writer.writerow(
                    [
                        scheme.name,
                        sequences[seq_index],
                        total_count,
                        total_count / n_total,
                        int(full_ra_counts[seq_index]),
                        int(full_ra_counts[seq_index]) / n_ra,
                        int(full_ild_counts[seq_index]),
                        int(full_ild_counts[seq_index]) / n_ild,
                        full_category[seq_index],
                        int(category_counts[0]),
                        category_counts[0] / n_iterations,
                        int(category_counts[1]),
                        category_counts[1] / n_iterations,
                        int(category_counts[2]),
                        category_counts[2] / n_iterations,
                        int(category_counts[3]),
                        category_counts[3] / n_iterations,
                        max_frequency[seq_index],
                    ]
                )
                written += 1

    return written


def screening_status(
    overview_df: pd.DataFrame,
    oof_overview_df: pd.DataFrame,
    scheme_name: str,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    subset = overview_df[overview_df["scheme"] == scheme_name].set_index(
        "category"
    )

    for category in ["RA_specific", "ILD_specific"]:
        empty_rate = float(subset.loc[category, "empty_rate"])
        if empty_rate > 0.05:
            reasons.append(f"{category} empty_rate={empty_rate:.3f}>0.05")

    jaccard_targets = {
        "global_public": 0.50,
        "shared": 0.50,
        "RA_specific": 0.30,
        "ILD_specific": 0.30,
    }
    for category, target in jaccard_targets.items():
        value = float(subset.loc[category, "median_jaccard_to_full"])
        if value < target:
            reasons.append(
                f"{category} median_jaccard={value:.3f}<{target:.2f}"
            )

    feature_targets = [
        "RA_specific_ref_frequency_sum",
        "ILD_specific_ref_frequency_sum",
    ]
    feature_df = oof_overview_df[
        (oof_overview_df["scheme"] == scheme_name)
        & (oof_overview_df["feature"].isin(feature_targets))
    ].set_index("feature")
    for feature in feature_targets:
        if feature not in feature_df.index:
            reasons.append(f"missing OOF feature summary: {feature}")
            continue
        value = float(feature_df.loc[feature, "nonzero_rate"])
        if value < 0.20:
            reasons.append(f"{feature} nonzero_rate={value:.3f}<0.20")

    status = "PASS_EMPIRICAL_SCREEN" if not reasons else "REVIEW"
    return status, reasons


def build_markdown_summary(
    args: argparse.Namespace,
    metadata: pd.DataFrame,
    manifest: Mapping[str, object],
    overview_df: pd.DataFrame,
    oof_overview_df: pd.DataFrame,
    full_set_sizes_df: pd.DataFrame,
    outputs: Mapping[str, Path],
    sequence_rows_written: int,
    runtime_seconds: float,
) -> str:
    lines: List[str] = [
        "# 03 Public CDR3-AA Threshold Stability Summary",
        "",
        "## Scope",
        "",
        "- Only the fixed training set was used.",
        "- The independent test set was not read.",
        "- No classification AUC or label-supervised feature selection was used.",
        "- Each iteration created an 80% reference subset and evaluated public overlap on the held-out 20%.",
        "- Jaccard is calculated between each resampled reference set and the corresponding full-training reference set.",
        "",
        "## Configuration",
        "",
        f"- Script version: `{SCRIPT_VERSION}`",
        f"- Training samples: **{len(metadata)}**",
        f"- RA samples: **{int((metadata[args.cohort_col] == 'RA').sum())}**",
        f"- ILD samples: **{int((metadata[args.cohort_col] == 'ILD').sum())}**",
        f"- Iterations: **{args.iterations}**",
        f"- Reference fraction: **{args.reference_fraction:.2f}**",
        f"- Random seed: **{args.seed}**",
        f"- Stratification: `{', '.join(args.strata_cols)}`",
        f"- Candidate sequences: **{int(manifest['candidate_sequence_count']):,}**",
        f"- Candidate incidences: **{int(manifest['candidate_incidence_count']):,}**",
        f"- Candidate minimum full-training sample count: **{args.candidate_min_total_count}**",
        "",
        "## Full-training set sizes",
        "",
    ]

    for _, row in full_set_sizes_df.iterrows():
        lines.append(
            f"- `{row['scheme']}`: "
            f"global={int(row['global_public_set_size']):,}, "
            f"RA-specific={int(row['RA_specific_set_size']):,}, "
            f"ILD-specific={int(row['ILD_specific_set_size']):,}, "
            f"shared={int(row['shared_set_size']):,}"
        )

    lines.extend(["", "## Stability overview", ""])
    for scheme in [x.name for x in DEFAULT_SCHEMES]:
        status, reasons = screening_status(
            overview_df=overview_df,
            oof_overview_df=oof_overview_df,
            scheme_name=scheme,
        )
        lines.append(f"### {scheme}: {status}")
        lines.append("")
        subset = overview_df[overview_df["scheme"] == scheme]
        for _, row in subset.iterrows():
            lines.append(
                f"- `{row['category']}`: "
                f"median size={row['median_set_size']:.0f}, "
                f"CV={row['cv_set_size']:.3f}, "
                f"empty rate={row['empty_rate']:.3f}, "
                f"median Jaccard-to-full={row['median_jaccard_to_full']:.3f}"
            )
        if reasons:
            lines.append("- Review reasons:")
            for reason in reasons:
                lines.append(f"  - {reason}")
        else:
            lines.append("- Passed the script's empirical screening rules.")
        lines.append("")

    lines.extend(
        [
            "## Interpretation rules",
            "",
            "- The screening cutoffs are practical diagnostics, not universal biological standards.",
            "- Prefer the pre-specified main scheme when it is not empty, has acceptable Jaccard stability, and produces non-zero held-out overlap.",
            "- Do not choose a threshold scheme using independent-test performance.",
            "- Even after a scheme is fixed, model cross-validation must rebuild reference sets inside each training fold.",
            "",
            "## Outputs",
            "",
        ]
    )
    for name, path in outputs.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(
        [
            f"- Sequence-selection rows written: **{sequence_rows_written:,}**",
            f"- Runtime: **{runtime_seconds:.2f} seconds**",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    started = time.time()

    aa_dir = Path(args.aa_dir).expanduser().resolve()
    summary_file = Path(args.summary_file).expanduser().resolve()
    metadata_file = Path(args.metadata).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not aa_dir.is_dir():
        raise NotADirectoryError(f"AA目录不存在: {aa_dir}")
    if not summary_file.is_file():
        raise FileNotFoundError(f"01 summary不存在: {summary_file}")
    if not metadata_file.is_file():
        raise FileNotFoundError(f"metadata不存在: {metadata_file}")

    outputs = output_paths(output_dir)
    ensure_overwrite(outputs, args.overwrite)

    metadata = read_metadata(
        path=metadata_file,
        id_col=args.id_col,
        cohort_col=args.cohort_col,
        batch_col=args.batch_col,
        strata_cols=args.strata_cols,
    )
    sample_ids = metadata[args.id_col].tolist()
    summary_df = read_summary(summary_file, sample_ids)

    aa_files = [
        aa_dir / f"{sample_id}{args.aa_input_suffix}"
        for sample_id in sample_ids
    ]
    missing_files = [p for p in aa_files if not p.is_file()]
    if missing_files:
        preview = "\n".join(f"  - {p}" for p in missing_files[:20])
        raise FileNotFoundError(
            f"缺失{len(missing_files)}个AA clone表：\n{preview}"
        )

    print("")
    print("=" * 80)
    print("03A Public CDR3-AA 阈值稳定性评估")
    print("=" * 80)
    print(f"训练样本数: {len(metadata)}")
    print(f"RA: {(metadata[args.cohort_col] == 'RA').sum()}")
    print(f"ILD: {(metadata[args.cohort_col] == 'ILD').sum()}")
    print(f"迭代次数: {args.iterations}")
    print(f"reference比例: {args.reference_fraction}")
    print(f"分层列: {args.strata_cols}")
    print(f"随机种子: {args.seed}")
    print(f"输出目录: {output_dir}")
    print("注意：脚本不读取test目录，也不计算模型AUC。")
    print("")

    (
        presence,
        frequency,
        sequences,
        clone_numbers,
        manifest,
    ) = build_or_load_cache(
        args=args,
        metadata=metadata,
        summary_df=summary_df,
        metadata_file=metadata_file,
        summary_file=summary_file,
        aa_files=aa_files,
        output_dir=output_dir,
    )

    if presence.shape[0] != len(metadata):
        raise RuntimeError("presence矩阵样本数与metadata不一致。")
    if presence.shape[1] != len(sequences):
        raise RuntimeError("presence矩阵候选序列数与序列列表不一致。")
    if frequency.shape != presence.shape:
        raise RuntimeError("frequency矩阵与presence矩阵形状不一致。")
    if len(clone_numbers) != len(metadata):
        raise RuntimeError("aa_clone_numbers长度与metadata不一致。")

    schemes = list(DEFAULT_SCHEMES)
    cohort_values = metadata[args.cohort_col].to_numpy(dtype=str)
    n_ra_full = int(np.sum(cohort_values == "RA"))
    n_ild_full = int(np.sum(cohort_values == "ILD"))

    print("")
    print("[稀疏cache]")
    print(f"- shape: {presence.shape}")
    print(f"- candidate sequences: {len(sequences):,}")
    print(f"- candidate incidences: {presence.nnz:,}")
    print(f"- total AA clone rows: {int(manifest['total_aa_clone_rows']):,}")

    presence_transposed = presence.transpose().tocsr()
    full_ra_counts, full_ild_counts = calculate_full_counts(
        presence_transposed=presence_transposed,
        cohort_values=cohort_values,
    )

    full_masks: Dict[str, Dict[str, np.ndarray]] = {}
    full_thresholds: Dict[str, Dict[str, int]] = {}
    full_set_size_rows: List[Dict[str, object]] = []

    for scheme in schemes:
        masks, thresholds = build_reference_masks(
            ra_sample_counts=full_ra_counts,
            ild_sample_counts=full_ild_counts,
            n_ra_reference=n_ra_full,
            n_ild_reference=n_ild_full,
            scheme=scheme,
        )
        full_masks[scheme.name] = masks
        full_thresholds[scheme.name] = thresholds
        full_set_size_rows.append(
            {
                "scheme": scheme.name,
                **scheme.to_dict(),
                **thresholds,
                **{
                    f"{category}_set_size": int(np.count_nonzero(masks[category]))
                    for category in CATEGORY_ORDER
                },
            }
        )

    full_set_sizes_df = pd.DataFrame(full_set_size_rows)

    n_candidates = len(sequences)
    n_schemes = len(schemes)
    n_categories = len(CATEGORY_ORDER)
    selection_counts = np.zeros(
        (n_schemes, n_categories, n_candidates),
        dtype=np.uint16,
    )

    rng = np.random.default_rng(args.seed)
    iteration_rows: List[Dict[str, object]] = []
    oof_rows: List[Dict[str, object]] = []

    feature_columns = [
        "all_ref_public_clone_number",
        "all_ref_public_clone_ratio",
        "all_ref_public_frequency_sum",
        "RA_specific_ref_clone_number",
        "RA_specific_ref_clone_ratio",
        "RA_specific_ref_frequency_sum",
        "ILD_specific_ref_clone_number",
        "ILD_specific_ref_clone_ratio",
        "ILD_specific_ref_frequency_sum",
        "shared_ref_clone_number",
        "shared_ref_clone_ratio",
        "shared_ref_frequency_sum",
        "ILD_RA_specific_ref_frequency_delta",
        "ILD_RA_specific_ref_frequency_log_ratio",
    ]

    print("")
    print("[稳定性重采样]")
    for iteration in range(1, args.iterations + 1):
        iteration_started = time.time()
        ref_indices, held_indices = stratified_reference_split(
            metadata=metadata,
            reference_fraction=args.reference_fraction,
            rng=rng,
            strata_columns=args.strata_cols,
        )

        ref_cohorts = cohort_values[ref_indices]
        n_ra_ref = int(np.sum(ref_cohorts == "RA"))
        n_ild_ref = int(np.sum(ref_cohorts == "ILD"))
        if n_ra_ref == 0 or n_ild_ref == 0:
            raise RuntimeError(
                f"第{iteration}轮reference中RA或ILD为空。"
            )

        ra_ref_vector = np.zeros(len(metadata), dtype=np.int16)
        ild_ref_vector = np.zeros(len(metadata), dtype=np.int16)
        ra_ref_vector[ref_indices[ref_cohorts == "RA"]] = 1
        ild_ref_vector[ref_indices[ref_cohorts == "ILD"]] = 1

        ra_counts = dense_vector(
            presence_transposed.dot(ra_ref_vector)
        ).astype(np.int16, copy=False)
        ild_counts = dense_vector(
            presence_transposed.dot(ild_ref_vector)
        ).astype(np.int16, copy=False)

        category_columns: List[np.ndarray] = []
        scheme_iteration_data: List[
            Tuple[ThresholdScheme, Dict[str, np.ndarray], Dict[str, int]]
        ] = []

        for scheme_index, scheme in enumerate(schemes):
            masks, thresholds = build_reference_masks(
                ra_sample_counts=ra_counts,
                ild_sample_counts=ild_counts,
                n_ra_reference=n_ra_ref,
                n_ild_reference=n_ild_ref,
                scheme=scheme,
            )
            scheme_iteration_data.append((scheme, masks, thresholds))

            for category_index, category in enumerate(CATEGORY_ORDER):
                mask = masks[category]
                selection_counts[
                    scheme_index, category_index
                ] += mask.astype(np.uint16)
                category_columns.append(mask)

        # Candidate sequence × 12 category matrix; same held-out split for all schemes.
        category_matrix = np.column_stack(category_columns).astype(
            np.float32, copy=False
        )

        held_presence = presence[held_indices]
        held_frequency = frequency[held_indices]
        held_clone_counts = dense_vector(
            held_presence.dot(category_matrix)
        ).reshape(len(held_indices), -1)
        held_frequency_sums = dense_vector(
            held_frequency.dot(category_matrix)
        ).reshape(len(held_indices), -1)

        for scheme_index, (scheme, masks, thresholds) in enumerate(
            scheme_iteration_data
        ):
            iteration_row: Dict[str, object] = {
                "iteration": iteration,
                "scheme": scheme.name,
                "n_reference": int(len(ref_indices)),
                "n_held_out": int(len(held_indices)),
                "n_RA_reference": n_ra_ref,
                "n_ILD_reference": n_ild_ref,
                **scheme.to_dict(),
                **thresholds,
            }

            for category in CATEGORY_ORDER:
                iteration_row[f"{category}_set_size"] = int(
                    np.count_nonzero(masks[category])
                )
                iteration_row[f"{category}_full_set_size"] = int(
                    np.count_nonzero(full_masks[scheme.name][category])
                )
                iteration_row[f"{category}_jaccard_to_full"] = (
                    jaccard_similarity(
                        masks[category],
                        full_masks[scheme.name][category],
                    )
                )
            iteration_rows.append(iteration_row)

            column_start = scheme_index * n_categories
            column_map = {
                category: column_start + category_index
                for category_index, category in enumerate(CATEGORY_ORDER)
            }

            for local_row, sample_index in enumerate(held_indices):
                denominator = float(clone_numbers[sample_index])
                global_col = column_map["global_public"]
                ra_col = column_map["RA_specific"]
                ild_col = column_map["ILD_specific"]
                shared_col = column_map["shared"]

                global_count = float(
                    held_clone_counts[local_row, global_col]
                )
                ra_count = float(held_clone_counts[local_row, ra_col])
                ild_count = float(held_clone_counts[local_row, ild_col])
                shared_count = float(
                    held_clone_counts[local_row, shared_col]
                )

                global_freq = float(
                    held_frequency_sums[local_row, global_col]
                )
                ra_freq = float(held_frequency_sums[local_row, ra_col])
                ild_freq = float(
                    held_frequency_sums[local_row, ild_col]
                )
                shared_freq = float(
                    held_frequency_sums[local_row, shared_col]
                )

                oof_rows.append(
                    {
                        "iteration": iteration,
                        "scheme": scheme.name,
                        "sample_index": int(sample_index),
                        "sample_id": metadata.loc[
                            sample_index, args.id_col
                        ],
                        "cohort": metadata.loc[
                            sample_index, args.cohort_col
                        ],
                        "batch": metadata.loc[
                            sample_index, args.batch_col
                        ],
                        "aa_clone_number": int(
                            clone_numbers[sample_index]
                        ),
                        "all_ref_public_clone_number": global_count,
                        "all_ref_public_clone_ratio": (
                            global_count / denominator
                        ),
                        "all_ref_public_frequency_sum": global_freq,
                        "RA_specific_ref_clone_number": ra_count,
                        "RA_specific_ref_clone_ratio": (
                            ra_count / denominator
                        ),
                        "RA_specific_ref_frequency_sum": ra_freq,
                        "ILD_specific_ref_clone_number": ild_count,
                        "ILD_specific_ref_clone_ratio": (
                            ild_count / denominator
                        ),
                        "ILD_specific_ref_frequency_sum": ild_freq,
                        "shared_ref_clone_number": shared_count,
                        "shared_ref_clone_ratio": (
                            shared_count / denominator
                        ),
                        "shared_ref_frequency_sum": shared_freq,
                        "ILD_RA_specific_ref_frequency_delta": (
                            ild_freq - ra_freq
                        ),
                        "ILD_RA_specific_ref_frequency_log_ratio": math.log(
                            (ild_freq + args.epsilon)
                            / (ra_freq + args.epsilon)
                        ),
                    }
                )

        if iteration == 1 or iteration % 5 == 0 or iteration == args.iterations:
            elapsed = time.time() - iteration_started
            main_data = next(
                x for x in scheme_iteration_data if x[0].name == "main"
            )
            main_masks = main_data[1]
            print(
                f"[{iteration:>3}/{args.iterations}] "
                f"ref={len(ref_indices)} held={len(held_indices)} | "
                f"main global={np.count_nonzero(main_masks['global_public']):,} "
                f"RA-spec={np.count_nonzero(main_masks['RA_specific']):,} "
                f"ILD-spec={np.count_nonzero(main_masks['ILD_specific']):,} "
                f"shared={np.count_nonzero(main_masks['shared']):,} | "
                f"{elapsed:.2f}s"
            )

    iterations_df = pd.DataFrame(iteration_rows)
    oof_df = pd.DataFrame(oof_rows)

    overview_df = make_stability_overview(
        iterations_df=iterations_df,
        schemes=schemes,
    )
    oof_stability_df = make_oof_stability(
        oof_df=oof_df,
        feature_columns=feature_columns,
    )
    oof_overview_df = make_oof_overview(
        oof_df=oof_df,
        stability_df=oof_stability_df,
        feature_columns=feature_columns,
    )

    configuration = {
        "script_version": SCRIPT_VERSION,
        "aa_dir": str(aa_dir),
        "summary_file": str(summary_file),
        "metadata": str(metadata_file),
        "output_dir": str(output_dir),
        "id_col": args.id_col,
        "cohort_col": args.cohort_col,
        "batch_col": args.batch_col,
        "strata_cols": args.strata_cols,
        "iterations": args.iterations,
        "reference_fraction": args.reference_fraction,
        "seed": args.seed,
        "epsilon": args.epsilon,
        "candidate_min_total_count": args.candidate_min_total_count,
        "schemes": [x.to_dict() for x in schemes],
        "cache_manifest": manifest,
    }

    outputs["configuration"].write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    iterations_df.to_csv(outputs["iterations"], index=False)
    overview_df.to_csv(outputs["overview"], index=False)
    oof_df.to_csv(
        outputs["oof_values"],
        index=False,
        compression="gzip",
    )
    oof_stability_df.to_csv(outputs["oof_stability"], index=False)
    oof_overview_df.to_csv(outputs["oof_overview"], index=False)
    full_set_sizes_df.to_csv(outputs["full_set_sizes"], index=False)

    sequence_rows_written = write_sequence_selection_frequency(
        path=outputs["sequence_selection"],
        sequences=sequences,
        schemes=schemes,
        selection_counts=selection_counts,
        full_ra_counts=full_ra_counts,
        full_ild_counts=full_ild_counts,
        full_masks=full_masks,
        n_iterations=args.iterations,
        n_ra=n_ra_full,
        n_ild=n_ild_full,
        max_rows=args.max_sequence_output_rows,
    )

    runtime_seconds = time.time() - started
    markdown = build_markdown_summary(
        args=args,
        metadata=metadata,
        manifest=manifest,
        overview_df=overview_df,
        oof_overview_df=oof_overview_df,
        full_set_sizes_df=full_set_sizes_df,
        outputs=outputs,
        sequence_rows_written=sequence_rows_written,
        runtime_seconds=runtime_seconds,
    )
    outputs["summary"].write_text(markdown + "\n", encoding="utf-8")

    print("")
    print("=" * 80)
    print("03A 阈值稳定性评估完成")
    print("=" * 80)
    print("")
    print("[完整训练集reference集合大小]")
    print(full_set_sizes_df[
        [
            "scheme",
            "global_public_set_size",
            "RA_specific_set_size",
            "ILD_specific_set_size",
            "shared_set_size",
        ]
    ].to_string(index=False))

    print("")
    print("[稳定性核心结果]")
    display_columns = [
        "scheme",
        "category",
        "median_set_size",
        "cv_set_size",
        "empty_rate",
        "median_jaccard_to_full",
    ]
    print(overview_df[display_columns].to_string(index=False))

    print("")
    print("[经验筛查]")
    for scheme in schemes:
        status, reasons = screening_status(
            overview_df=overview_df,
            oof_overview_df=oof_overview_df,
            scheme_name=scheme.name,
        )
        print(f"- {scheme.name}: {status}")
        for reason in reasons:
            print(f"    {reason}")

    print("")
    print("[输出文件]")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    print(f"- sequence rows written: {sequence_rows_written:,}")
    print(f"- total runtime: {runtime_seconds:.2f}s")
    print("")
    print(
        "请重点查看03_threshold_stability_summary.md、"
        "03_threshold_stability_overview.csv和"
        "03_oof_public_feature_overview.csv。"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
