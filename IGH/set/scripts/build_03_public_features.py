#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build final step-03 IGH public CDR3-AA files after threshold stability evaluation.

Modes
-----
train
    Uses the fixed TRAINING sparse cache and the confirmed threshold scheme to
    create the sequence-level public catalog, final training reference sets,
    and sample-level descriptive public features.

test
    Applies the fixed training reference sets to independent test samples.
    Test labels are not used to construct or modify any reference set.

Important
---------
The fixed full-training IGH descriptive features generated in train mode are NOT
valid as ordinary cross-validation predictors because each training sample is
contained in the full-training reference construction. Future model CV must
rebuild reference sets inside each training fold using public_feature_utils.py.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from public_feature_utils import (
    ThresholdScheme,
    build_reference_masks,
    classify_full_reference_category,
    calculate_scheme_thresholds,
)


SCRIPT_VERSION = "1.1.0-IGH"
AA_INPUT_SUFFIX = "_IGH-without-DJ_CDR3_AA_clone_table.csv"
AA_SET = set("ACDEFGHIKLMNPQRSTVWY")
EXPECTED_RECEPTOR = "IGH"

ROOT = Path("/data/users/chenhaisheng/RA-ILD/IGH")
TRAIN_METADATA = ROOT / "set/train/metadata_train_70.csv"
TRAIN_SUMMARY = ROOT / "set/train/result/01_AA_clone_table/01_AA_clone_table_summary.csv"
TRAIN_AA_DIR = ROOT / "set/train/result/01_AA_clone_table"
TRAIN_OUTPUT = ROOT / "set/train/result/03_public_features"
STABILITY_DIR = TRAIN_OUTPUT / "threshold_stability"
STABILITY_CONFIG = STABILITY_DIR / "03_threshold_stability_configuration.json"
STABILITY_CACHE = STABILITY_DIR / "cache"

TEST_METADATA = ROOT / "set/test/metadata_test_30.csv"
TEST_SUMMARY = ROOT / "set/test/result/01_AA_clone_table/01_AA_clone_table_summary.csv"
TEST_AA_DIR = ROOT / "set/test/result/01_AA_clone_table"
TEST_OUTPUT = ROOT / "set/test/result/03_public_features"

TRAIN_REFERENCE_FILE = TRAIN_OUTPUT / "03_final_reference_public_sets.csv.gz"
TRAIN_REFERENCE_DEFINITION = TRAIN_OUTPUT / "03_reference_definition.json"

BIT_GLOBAL = np.uint8(1)
BIT_RA_GROUP = np.uint8(2)
BIT_ILD_GROUP = np.uint8(4)
BIT_SHARED = np.uint8(8)
BIT_RA_SPECIFIC = np.uint8(16)
BIT_ILD_SPECIFIC = np.uint8(32)

SET_ORDER = (
    "global_public",
    "RA_group_public",
    "ILD_group_public",
    "between_group_shared",
    "RA_specific_ref",
    "ILD_specific_ref",
)

DESCRIPTIVE_PREFIX = {
    "global_public": "global_public_aa",
    "RA_group_public": "RA_group_public_aa",
    "ILD_group_public": "ILD_group_public_aa",
    "between_group_shared": "between_group_shared_aa",
    "RA_specific_ref": "RA_specific_ref",
    "ILD_specific_ref": "ILD_specific_ref",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final step-03 IGH public CDR3-AA files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", required=True, choices=["train", "test"])
    parser.add_argument("--id-col", default="libraryid")
    parser.add_argument("--cohort-col", default="cohort")
    parser.add_argument("--scheme-name", default="main")
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--aa-input-suffix", default=AA_INPUT_SUFFIX)
    parser.add_argument("--catalog-chunk-size", type=int, default=100_000)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--metadata", default=None)
    parser.add_argument("--summary-file", default=None)
    parser.add_argument("--aa-dir", default=None)
    parser.add_argument("--output-dir", default=None)

    parser.add_argument(
        "--stability-configuration",
        default=str(STABILITY_CONFIG),
        help="Threshold-stability configuration JSON used to retrieve the confirmed scheme.",
    )
    parser.add_argument(
        "--stability-cache-dir",
        default=str(STABILITY_CACHE),
        help="Training sparse cache created by evaluate_03_public_thresholds.py.",
    )

    parser.add_argument(
        "--reference-file",
        default=str(TRAIN_REFERENCE_FILE),
        help="Fixed training reference sets used in test mode.",
    )
    parser.add_argument(
        "--reference-definition",
        default=str(TRAIN_REFERENCE_DEFINITION),
        help="Training reference definition JSON used in test mode.",
    )

    args = parser.parse_args()
    if args.epsilon <= 0:
        parser.error("--epsilon must be > 0.")
    if args.catalog_chunk_size < 1:
        parser.error("--catalog-chunk-size must be >= 1.")

    if args.mode == "train":
        args.metadata = args.metadata or str(TRAIN_METADATA)
        args.summary_file = args.summary_file or str(TRAIN_SUMMARY)
        args.aa_dir = args.aa_dir or str(TRAIN_AA_DIR)
        args.output_dir = args.output_dir or str(TRAIN_OUTPUT)
    else:
        args.metadata = args.metadata or str(TEST_METADATA)
        args.summary_file = args.summary_file or str(TEST_SUMMARY)
        args.aa_dir = args.aa_dir or str(TEST_AA_DIR)
        args.output_dir = args.output_dir or str(TEST_OUTPUT)

    return args


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def output_paths(output_dir: Path, mode: str) -> Dict[str, Path]:
    common = {
        "descriptive": output_dir / "03_descriptive_public_features.csv",
        "build_log": output_dir / "03_public_feature_build_log.csv",
        "summary": output_dir / "03_public_feature_build_summary.md",
    }
    if mode == "train":
        common.update(
            {
                "catalog": output_dir / "03_public_aa_catalog.csv.gz",
                "reference_sets": output_dir / "03_final_reference_public_sets.csv.gz",
                "reference_summary": output_dir / "03_reference_set_summary.csv",
                "reference_definition": output_dir / "03_reference_definition.json",
            }
        )
    else:
        common.update(
            {
                "test_reference_features": output_dir / "03_test_reference_public_features.csv",
                "reference_definition_used": output_dir / "03_reference_definition_used.json",
            }
        )
    return common


def ensure_outputs(paths: Mapping[str, Path], overwrite: bool) -> None:
    existing = [p for p in paths.values() if p.exists()]
    if existing and not overwrite:
        text = "\n".join(f"  - {p}" for p in existing)
        raise FileExistsError(
            "以下输出已存在；确认覆盖时添加 --overwrite：\n" + text
        )


def read_metadata(path: Path, id_col: str, cohort_col: str, require_cohort: bool) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    required = {id_col}
    if require_cohort:
        required.add(cohort_col)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"metadata缺少列: {missing}; 实际列: {list(df.columns)}")

    df = df.reset_index(drop=True)
    df[id_col] = df[id_col].astype(str).str.strip()
    if df[id_col].isna().any() or df[id_col].eq("").any():
        raise ValueError(f"metadata的{id_col}存在空值。")
    if df[id_col].duplicated().any():
        duplicates = df.loc[df[id_col].duplicated(keep=False), id_col].unique().tolist()[:10]
        raise ValueError(f"metadata样本ID重复，例如: {duplicates}")

    if cohort_col in df.columns:
        df[cohort_col] = df[cohort_col].astype(str).str.strip().str.upper()
    if require_cohort:
        cohorts = set(df[cohort_col].unique())
        if cohorts != {"RA", "ILD"}:
            raise ValueError(f"训练metadata的{cohort_col}必须且只能包含RA/ILD，实际={sorted(cohorts)}")
    return df


def read_summary(path: Path, sample_ids: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sample_id", "aa_clone_number"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"01 summary缺少列: {missing}")
    df["sample_id"] = df["sample_id"].astype(str)
    if df["sample_id"].duplicated().any():
        raise ValueError("01 summary中sample_id重复。")
    df = df.set_index("sample_id", drop=False)
    missing_ids = [x for x in sample_ids if x not in df.index]
    if missing_ids:
        raise ValueError(f"有{len(missing_ids)}个metadata样本不在01 summary，例如: {missing_ids[:10]}")
    return df.loc[list(sample_ids)].copy()


def load_scheme(configuration_file: Path, scheme_name: str) -> Tuple[ThresholdScheme, Dict[str, object]]:
    if not configuration_file.is_file():
        raise FileNotFoundError(f"稳定性配置不存在: {configuration_file}")
    config = json.loads(configuration_file.read_text(encoding="utf-8"))
    matches = [x for x in config.get("schemes", []) if x.get("name") == scheme_name]
    if len(matches) != 1:
        raise ValueError(
            f"稳定性配置中应恰有一个scheme={scheme_name}，实际找到{len(matches)}个。"
        )
    item = matches[0]
    scheme = ThresholdScheme(
        name=str(item["name"]),
        global_min_prevalence=float(item["global_min_prevalence"]),
        global_min_count=int(item["global_min_count"]),
        group_min_prevalence=float(item["group_min_prevalence"]),
        group_min_count=int(item["group_min_count"]),
        specific_prevalence_delta=float(item["specific_prevalence_delta"]),
    )
    return scheme, config


def validate_igh_stability_inputs(
    configuration: Mapping[str, object],
    cache_manifest: Mapping[str, object],
    aa_input_suffix: str,
) -> None:
    """Validate that stability configuration and cache belong to this IGH run."""
    config_suffix = configuration.get("aa_input_suffix")
    if config_suffix is not None and str(config_suffix) != aa_input_suffix:
        raise RuntimeError(
            "稳定性配置的AA输入后缀与当前命令不一致: "
            f"config={config_suffix}, current={aa_input_suffix}"
        )

    receptor = cache_manifest.get("receptor")
    if receptor is not None and str(receptor).upper() != EXPECTED_RECEPTOR:
        raise RuntimeError(
            f"稳定性cache receptor={receptor}，不是预期的{EXPECTED_RECEPTOR}。"
        )

    cache_suffix = cache_manifest.get("aa_input_suffix")
    if cache_suffix is not None and str(cache_suffix) != aa_input_suffix:
        raise RuntimeError(
            "稳定性cache的AA输入后缀与当前命令不一致: "
            f"cache={cache_suffix}, current={aa_input_suffix}"
        )

    configured_manifest = configuration.get("cache_manifest")
    if isinstance(configured_manifest, Mapping):
        keys = [
            "receptor",
            "aa_input_suffix",
            "metadata",
            "summary",
            "sample_ids",
            "candidate_min_total_count",
            "aa_files",
            "candidate_sequence_count",
            "candidate_incidence_count",
            "presence_shape",
            "presence_nnz",
            "frequency_nnz",
        ]
        mismatches = [
            key for key in keys
            if key in configured_manifest
            and configured_manifest.get(key) != cache_manifest.get(key)
        ]
        if mismatches:
            raise RuntimeError(
                "稳定性配置记录的cache与当前cache不一致，字段: "
                + ", ".join(mismatches)
            )


def validate_reference_definition_for_igh(
    definition: Mapping[str, object],
    scheme_name: str,
    aa_input_suffix: str,
) -> None:
    receptor = definition.get("receptor")
    if receptor is not None and str(receptor).upper() != EXPECTED_RECEPTOR:
        raise RuntimeError(
            f"reference definition receptor={receptor}，不是预期的{EXPECTED_RECEPTOR}。"
        )

    definition_suffix = definition.get("aa_input_suffix")
    if definition_suffix is not None and str(definition_suffix) != aa_input_suffix:
        raise RuntimeError(
            "reference definition的AA输入后缀与当前命令不一致: "
            f"definition={definition_suffix}, current={aa_input_suffix}"
        )

    actual_scheme = definition.get("scheme", {}).get("name")
    if actual_scheme != scheme_name:
        raise RuntimeError(
            f"reference scheme={actual_scheme}，但命令要求scheme={scheme_name}。"
        )


def cache_paths(cache_dir: Path) -> Dict[str, Path]:
    return {
        "presence": cache_dir / "candidate_presence_matrix.npz",
        "frequency": cache_dir / "candidate_frequency_matrix.npz",
        "sequences": cache_dir / "candidate_sequences.txt.gz",
        "sample_info": cache_dir / "sample_info.csv",
        "clone_numbers": cache_dir / "aa_clone_numbers.npy",
        "manifest": cache_dir / "cache_manifest.json",
    }


def load_sequences(path: Path) -> List[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def load_training_cache(
    cache_dir: Path,
    metadata: pd.DataFrame,
    id_col: str,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, List[str], np.ndarray, pd.DataFrame, Dict[str, object]]:
    paths = cache_paths(cache_dir)
    missing = [p for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("训练cache不完整:\n" + "\n".join(f"  - {p}" for p in missing))

    presence = sparse.load_npz(paths["presence"]).tocsr()
    frequency = sparse.load_npz(paths["frequency"]).tocsr()
    sequences = load_sequences(paths["sequences"])
    clone_numbers = np.load(paths["clone_numbers"])
    sample_info = pd.read_csv(paths["sample_info"], dtype=str)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

    if presence.shape != frequency.shape:
        raise RuntimeError("cache presence和frequency矩阵形状不一致。")
    if presence.shape[1] != len(sequences):
        raise RuntimeError("cache候选序列数量与矩阵列数不一致。")
    if presence.shape[0] != len(clone_numbers):
        raise RuntimeError("cache样本数与clone_numbers长度不一致。")
    if id_col not in sample_info.columns:
        raise ValueError(f"cache sample_info缺少{id_col}。")

    cache_ids = sample_info[id_col].astype(str).tolist()
    metadata_ids = metadata[id_col].astype(str).tolist()
    if cache_ids != metadata_ids:
        raise RuntimeError("当前训练metadata的样本顺序与稳定性cache不一致。")
    if len(metadata) != presence.shape[0]:
        raise RuntimeError("训练metadata样本数与cache不一致。")
    return presence, frequency, sequences, clone_numbers, sample_info, manifest


def dense_vector(value) -> np.ndarray:
    return np.asarray(value).reshape(-1)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.zeros_like(np.asarray(numerator, dtype=float), dtype=float)
    np.divide(numerator, denominator, out=out, where=np.asarray(denominator) != 0)
    return out


def build_full_train_statistics(
    presence: sparse.csr_matrix,
    frequency: sparse.csr_matrix,
    cohorts: np.ndarray,
) -> Dict[str, np.ndarray]:
    pt = presence.transpose().tocsr()
    ft = frequency.transpose().tocsr()
    ra = (cohorts == "RA").astype(np.int16)
    ild = (cohorts == "ILD").astype(np.int16)
    all_samples = np.ones(len(cohorts), dtype=np.int16)

    stats = {
        "total_sample_count": dense_vector(pt.dot(all_samples)).astype(np.int16, copy=False),
        "RA_sample_count": dense_vector(pt.dot(ra)).astype(np.int16, copy=False),
        "ILD_sample_count": dense_vector(pt.dot(ild)).astype(np.int16, copy=False),
        "total_frequency_sum": dense_vector(ft.dot(all_samples)).astype(np.float64, copy=False),
        "RA_frequency_sum": dense_vector(ft.dot(ra)).astype(np.float64, copy=False),
        "ILD_frequency_sum": dense_vector(ft.dot(ild)).astype(np.float64, copy=False),
    }
    return stats


def derive_group_masks(
    stats: Mapping[str, np.ndarray],
    n_ra: int,
    n_ild: int,
    scheme: ThresholdScheme,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    reference_masks, thresholds = build_reference_masks(
        ra_sample_counts=stats["RA_sample_count"],
        ild_sample_counts=stats["ILD_sample_count"],
        n_ra_reference=n_ra,
        n_ild_reference=n_ild,
        scheme=scheme,
    )
    ra_group = stats["RA_sample_count"] >= thresholds["RA_count_threshold"]
    ild_group = stats["ILD_sample_count"] >= thresholds["ILD_count_threshold"]
    masks = {
        "global_public": reference_masks["global_public"],
        "RA_group_public": ra_group,
        "ILD_group_public": ild_group,
        "between_group_shared": reference_masks["shared"],
        "RA_specific_ref": reference_masks["RA_specific"],
        "ILD_specific_ref": reference_masks["ILD_specific"],
    }
    return masks, thresholds


def catalog_columns() -> List[str]:
    return [
        "cdr3_aa",
        "total_sample_count",
        "total_sample_prevalence",
        "RA_sample_count",
        "RA_sample_prevalence",
        "ILD_sample_count",
        "ILD_sample_prevalence",
        "prevalence_delta_ILD_minus_RA",
        "total_frequency_sum",
        "total_mean_frequency_all_samples",
        "total_mean_frequency_present_samples",
        "RA_frequency_sum",
        "RA_mean_frequency_all_RA_samples",
        "RA_mean_frequency_present_RA_samples",
        "ILD_frequency_sum",
        "ILD_mean_frequency_all_ILD_samples",
        "ILD_mean_frequency_present_ILD_samples",
        "is_global_public",
        "is_RA_group_public",
        "is_ILD_group_public",
        "is_between_group_shared",
        "is_RA_specific_ref",
        "is_ILD_specific_ref",
        "reference_category",
    ]


def catalog_chunk_dataframe(
    start: int,
    end: int,
    sequences: Sequence[str],
    stats: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    categories: np.ndarray,
    n_total: int,
    n_ra: int,
    n_ild: int,
) -> pd.DataFrame:
    sl = slice(start, end)
    total_count = stats["total_sample_count"][sl]
    ra_count = stats["RA_sample_count"][sl]
    ild_count = stats["ILD_sample_count"][sl]
    total_freq = stats["total_frequency_sum"][sl]
    ra_freq = stats["RA_frequency_sum"][sl]
    ild_freq = stats["ILD_frequency_sum"][sl]

    df = pd.DataFrame(
        {
            "cdr3_aa": sequences[start:end],
            "total_sample_count": total_count,
            "total_sample_prevalence": total_count / n_total,
            "RA_sample_count": ra_count,
            "RA_sample_prevalence": ra_count / n_ra,
            "ILD_sample_count": ild_count,
            "ILD_sample_prevalence": ild_count / n_ild,
            "prevalence_delta_ILD_minus_RA": (ild_count / n_ild) - (ra_count / n_ra),
            "total_frequency_sum": total_freq,
            "total_mean_frequency_all_samples": total_freq / n_total,
            "total_mean_frequency_present_samples": safe_divide(total_freq, total_count),
            "RA_frequency_sum": ra_freq,
            "RA_mean_frequency_all_RA_samples": ra_freq / n_ra,
            "RA_mean_frequency_present_RA_samples": safe_divide(ra_freq, ra_count),
            "ILD_frequency_sum": ild_freq,
            "ILD_mean_frequency_all_ILD_samples": ild_freq / n_ild,
            "ILD_mean_frequency_present_ILD_samples": safe_divide(ild_freq, ild_count),
            "is_global_public": masks["global_public"][sl].astype(np.uint8),
            "is_RA_group_public": masks["RA_group_public"][sl].astype(np.uint8),
            "is_ILD_group_public": masks["ILD_group_public"][sl].astype(np.uint8),
            "is_between_group_shared": masks["between_group_shared"][sl].astype(np.uint8),
            "is_RA_specific_ref": masks["RA_specific_ref"][sl].astype(np.uint8),
            "is_ILD_specific_ref": masks["ILD_specific_ref"][sl].astype(np.uint8),
            "reference_category": categories[sl],
        }
    )
    return df[catalog_columns()]


def write_catalogs_chunked(
    catalog_path: Path,
    reference_path: Path,
    sequences: Sequence[str],
    stats: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    categories: np.ndarray,
    n_total: int,
    n_ra: int,
    n_ild: int,
    chunk_size: int,
) -> Tuple[int, int]:
    catalog_rows = 0
    reference_rows = 0
    with gzip.open(catalog_path, "wt", encoding="utf-8", newline="") as catalog_handle, gzip.open(
        reference_path, "wt", encoding="utf-8", newline=""
    ) as reference_handle:
        first_catalog = True
        first_reference = True
        for start in range(0, len(sequences), chunk_size):
            end = min(len(sequences), start + chunk_size)
            chunk = catalog_chunk_dataframe(
                start=start,
                end=end,
                sequences=sequences,
                stats=stats,
                masks=masks,
                categories=categories,
                n_total=n_total,
                n_ra=n_ra,
                n_ild=n_ild,
            )
            chunk.to_csv(
                catalog_handle,
                index=False,
                header=first_catalog,
                float_format="%.12g",
                lineterminator="\n",
            )
            first_catalog = False
            catalog_rows += len(chunk)

            reference_chunk = chunk[chunk["is_global_public"] == 1]
            if len(reference_chunk):
                reference_chunk.to_csv(
                    reference_handle,
                    index=False,
                    header=first_reference,
                    float_format="%.12g",
                    lineterminator="\n",
                )
                first_reference = False
                reference_rows += len(reference_chunk)
            print(
                f"[目录写出] {end:,}/{len(sequences):,} | "
                f"catalog={catalog_rows:,} reference={reference_rows:,}"
            )
    return catalog_rows, reference_rows


def build_sample_feature_frame_from_sparse(
    sample_ids: Sequence[str],
    clone_numbers: np.ndarray,
    presence: sparse.csr_matrix,
    frequency: sparse.csr_matrix,
    masks: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    mask_matrix = np.column_stack([masks[name] for name in SET_ORDER]).astype(np.float64)
    counts = np.asarray(presence.dot(mask_matrix))
    frequency_sums = np.asarray(frequency.dot(mask_matrix))
    set_sizes = np.array([np.count_nonzero(masks[name]) for name in SET_ORDER], dtype=float)

    data: Dict[str, object] = {
        "sample_id": list(sample_ids),
        "aa_clone_number": clone_numbers.astype(np.int64),
    }
    for index, set_name in enumerate(SET_ORDER):
        prefix = DESCRIPTIVE_PREFIX[set_name]
        values = counts[:, index]
        data[f"{prefix}_clone_number"] = values.astype(np.int64)
        data[f"{prefix}_clone_ratio"] = values / clone_numbers
        data[f"{prefix}_frequency_sum"] = frequency_sums[:, index]
        data[f"{prefix}_set_coverage"] = (
            values / set_sizes[index] if set_sizes[index] > 0 else np.zeros(len(sample_ids))
        )
    return pd.DataFrame(data)


def reference_feature_columns() -> List[str]:
    columns = ["sample_id"]
    mapping = {
        "global_public_aa": "all_ref_public",
        "RA_specific_ref": "RA_specific_ref",
        "ILD_specific_ref": "ILD_specific_ref",
        "between_group_shared_aa": "shared_ref",
    }
    for source_prefix, target_prefix in mapping.items():
        for suffix in ["clone_number", "clone_ratio", "frequency_sum", "set_coverage"]:
            columns.append(f"{target_prefix}_{suffix}")
    columns.extend(
        [
            "ILD_RA_specific_ref_frequency_delta",
            "ILD_RA_specific_ref_frequency_log_ratio",
        ]
    )
    return columns


def make_reference_feature_frame(descriptive: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    out = pd.DataFrame({"sample_id": descriptive["sample_id"]})
    mapping = {
        "global_public_aa": "all_ref_public",
        "RA_specific_ref": "RA_specific_ref",
        "ILD_specific_ref": "ILD_specific_ref",
        "between_group_shared_aa": "shared_ref",
    }
    for source_prefix, target_prefix in mapping.items():
        for suffix in ["clone_number", "clone_ratio", "frequency_sum", "set_coverage"]:
            out[f"{target_prefix}_{suffix}"] = descriptive[f"{source_prefix}_{suffix}"]

    ild = out["ILD_specific_ref_frequency_sum"].to_numpy(dtype=float)
    ra = out["RA_specific_ref_frequency_sum"].to_numpy(dtype=float)
    out["ILD_RA_specific_ref_frequency_delta"] = ild - ra
    out["ILD_RA_specific_ref_frequency_log_ratio"] = np.log((ild + epsilon) / (ra + epsilon))
    return out[reference_feature_columns()]


def reference_set_summary(
    masks: Mapping[str, np.ndarray],
    thresholds: Mapping[str, int],
    scheme: ThresholdScheme,
    n_total: int,
    n_ra: int,
    n_ild: int,
) -> pd.DataFrame:
    rows = []
    definitions = {
        "global_public": "total_sample_count >= global_count_threshold",
        "RA_group_public": "RA_sample_count >= RA_count_threshold",
        "ILD_group_public": "ILD_sample_count >= ILD_count_threshold",
        "between_group_shared": "RA and ILD group-public; not RA/ILD-specific",
        "RA_specific_ref": "RA group-public and RA_prevalence - ILD_prevalence >= delta",
        "ILD_specific_ref": "ILD group-public and ILD_prevalence - RA_prevalence >= delta",
    }
    for name in SET_ORDER:
        rows.append(
            {
                "scheme": scheme.name,
                "set_name": name,
                "set_size": int(np.count_nonzero(masks[name])),
                "definition": definitions[name],
                "n_training_samples": n_total,
                "n_RA_training_samples": n_ra,
                "n_ILD_training_samples": n_ild,
                **thresholds,
                "global_min_prevalence": scheme.global_min_prevalence,
                "global_min_count": scheme.global_min_count,
                "group_min_prevalence": scheme.group_min_prevalence,
                "group_min_count": scheme.group_min_count,
                "specific_prevalence_delta": scheme.specific_prevalence_delta,
            }
        )
    return pd.DataFrame(rows)


def validate_descriptive_frame(df: pd.DataFrame, sample_ids: Sequence[str]) -> List[str]:
    warnings: List[str] = []
    if df["sample_id"].tolist() != list(sample_ids):
        raise RuntimeError("样本级public特征的sample_id顺序不一致。")
    if df["sample_id"].duplicated().any():
        raise RuntimeError("样本级public特征出现重复sample_id。")
    numeric = df.drop(columns="sample_id")
    if numeric.isna().any().any():
        raise RuntimeError("样本级public特征存在缺失值。")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("样本级public特征存在非有限值。")

    ratio_cols = [c for c in df.columns if c.endswith("_clone_ratio")]
    freq_cols = [c for c in df.columns if c.endswith("_frequency_sum")]
    coverage_cols = [c for c in df.columns if c.endswith("_set_coverage")]
    for col in ratio_cols + freq_cols + coverage_cols:
        min_value = float(df[col].min())
        max_value = float(df[col].max())
        if min_value < -1e-10 or max_value > 1 + 1e-8:
            raise RuntimeError(f"{col}超出[0,1]范围: min={min_value}, max={max_value}")
        if np.isclose(df[col], 0.0).all():
            warnings.append(f"{col}全部为0")
    return warnings


def write_train_summary(
    path: Path,
    args: argparse.Namespace,
    scheme: ThresholdScheme,
    thresholds: Mapping[str, int],
    metadata: pd.DataFrame,
    reference_summary_df: pd.DataFrame,
    descriptive: pd.DataFrame,
    outputs: Mapping[str, Path],
    catalog_rows: int,
    reference_rows: int,
    warnings: Sequence[str],
    runtime: float,
) -> None:
    lines = [
        "# 03 IGH Public CDR3-AA Build Summary — TRAIN",
        "",
        "## Scope",
        "",
        "- The confirmed threshold scheme was loaded from the threshold-stability configuration.",
        "- Only the fixed training set and its validated sparse cache were used.",
        "- `03_descriptive_public_features.csv` is descriptive and self-inclusive; do not use it as a fixed cross-validation predictor.",
        "- Model cross-validation must rebuild reference sets within each training fold.",
        "",
        "## Thresholds",
        "",
        f"- Scheme: `{scheme.name}`",
        f"- Training samples: **{len(metadata)}**",
        f"- RA samples: **{int((metadata[args.cohort_col] == 'RA').sum())}**",
        f"- ILD samples: **{int((metadata[args.cohort_col] == 'ILD').sum())}**",
        f"- Effective global count threshold: **{thresholds['global_count_threshold']}**",
        f"- Effective RA count threshold: **{thresholds['RA_count_threshold']}**",
        f"- Effective ILD count threshold: **{thresholds['ILD_count_threshold']}**",
        f"- Specific prevalence delta: **{scheme.specific_prevalence_delta:.3f}**",
        "",
        "## Reference-set sizes",
        "",
    ]
    for _, row in reference_summary_df.iterrows():
        lines.append(f"- `{row['set_name']}`: **{int(row['set_size']):,}**")
    lines.extend(
        [
            "",
            "## Output dimensions",
            "",
            f"- Catalog rows: **{catalog_rows:,}**",
            f"- Final all-public reference rows: **{reference_rows:,}**",
            f"- Descriptive feature matrix: **{descriptive.shape[0]} × {descriptive.shape[1]}**",
            "",
            "## Warnings",
            "",
        ]
    )
    if warnings:
        lines.extend([f"- {x}" for x in warnings])
    else:
        lines.append("- None.")
    lines.extend(["", "## Outputs", ""])
    for name, output in outputs.items():
        lines.append(f"- `{name}`: `{output}`")
    lines.extend(["", f"Runtime: **{runtime:.2f} seconds**", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_test_summary(
    path: Path,
    metadata: pd.DataFrame,
    descriptive: pd.DataFrame,
    reference_features: pd.DataFrame,
    definition: Mapping[str, object],
    outputs: Mapping[str, Path],
    warnings: Sequence[str],
    runtime: float,
) -> None:
    sizes = definition["reference_set_sizes"]
    lines = [
        "# 03 IGH Public CDR3-AA Build Summary — TEST",
        "",
        "## Scope",
        "",
        "- The test set was transformed using reference sets constructed from the complete training set.",
        "- Test cohort labels were not used to construct, filter, or modify reference sets.",
        "- Test-only CDR3-AA sequences outside the training all-public reference were ignored for reference-overlap features.",
        "",
        "## Reference definition",
        "",
        f"- Scheme: `{definition['scheme']['name']}`",
        f"- Training samples used for references: **{definition['n_training_samples']}**",
        f"- all_ref_public size: **{sizes['global_public']:,}**",
        f"- RA_specific_ref size: **{sizes['RA_specific_ref']:,}**",
        f"- ILD_specific_ref size: **{sizes['ILD_specific_ref']:,}**",
        f"- shared_ref size: **{sizes['between_group_shared']:,}**",
        "",
        "## Output dimensions",
        "",
        f"- Test samples: **{len(metadata)}**",
        f"- Descriptive feature matrix: **{descriptive.shape[0]} × {descriptive.shape[1]}**",
        f"- Model reference-feature matrix: **{reference_features.shape[0]} × {reference_features.shape[1]}**",
        "",
        "## Warnings",
        "",
    ]
    if warnings:
        lines.extend([f"- {x}" for x in warnings])
    else:
        lines.append("- None.")
    lines.extend(["", "## Outputs", ""])
    for name, output in outputs.items():
        lines.append(f"- `{name}`: `{output}`")
    lines.extend(["", f"Runtime: **{runtime:.2f} seconds**", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_train(args: argparse.Namespace) -> int:
    started = time.time()
    metadata_path = Path(args.metadata).expanduser().resolve()
    summary_path = Path(args.summary_file).expanduser().resolve()
    aa_dir = Path(args.aa_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    stability_config_path = Path(args.stability_configuration).expanduser().resolve()
    cache_dir = Path(args.stability_cache_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = output_paths(output_dir, "train")
    ensure_outputs(outputs, args.overwrite)

    for path, label in [(metadata_path, "metadata"), (summary_path, "01 summary")]:
        if not path.is_file():
            raise FileNotFoundError(f"{label}不存在: {path}")
    if not aa_dir.is_dir():
        raise NotADirectoryError(f"AA目录不存在: {aa_dir}")

    metadata = read_metadata(metadata_path, args.id_col, args.cohort_col, require_cohort=True)
    sample_ids = metadata[args.id_col].tolist()
    summary_df = read_summary(summary_path, sample_ids)
    scheme, stability_config = load_scheme(stability_config_path, args.scheme_name)
    presence, frequency, sequences, clone_numbers, sample_info, cache_manifest = load_training_cache(
        cache_dir=cache_dir,
        metadata=metadata,
        id_col=args.id_col,
    )
    validate_igh_stability_inputs(
        configuration=stability_config,
        cache_manifest=cache_manifest,
        aa_input_suffix=args.aa_input_suffix,
    )

    expected_clone_numbers = summary_df["aa_clone_number"].astype(int).to_numpy()
    if not np.array_equal(clone_numbers.astype(int), expected_clone_numbers):
        raise RuntimeError("cache中的aa_clone_numbers与01 summary不一致。")

    cohorts = metadata[args.cohort_col].to_numpy(dtype=str)
    n_ra = int(np.sum(cohorts == "RA"))
    n_ild = int(np.sum(cohorts == "ILD"))
    n_total = len(metadata)
    stats = build_full_train_statistics(presence, frequency, cohorts)
    masks, thresholds = derive_group_masks(stats, n_ra, n_ild, scheme)

    # The compact final reference file contains all global-public sequences.
    # Therefore every group-public/specific/shared set must be a subset of it.
    for set_name in [
        "RA_group_public",
        "ILD_group_public",
        "between_group_shared",
        "RA_specific_ref",
        "ILD_specific_ref",
    ]:
        outside = masks[set_name] & ~masks["global_public"]
        if np.any(outside):
            raise RuntimeError(
                f"{set_name}有{int(np.count_nonzero(outside))}条序列不属于global_public；"
                "当前reference文件压缩策略不适用于该阈值组合。"
            )

    base_reference_masks = {
        "global_public": masks["global_public"],
        "RA_specific": masks["RA_specific_ref"],
        "ILD_specific": masks["ILD_specific_ref"],
        "shared": masks["between_group_shared"],
    }
    categories = classify_full_reference_category(base_reference_masks)

    print("=" * 80)
    print("03 IGH Public CDR3-AA 正式构建 — TRAIN")
    print("=" * 80)
    print(f"训练样本: {n_total} | RA={n_ra} | ILD={n_ild}")
    print(f"receptor: {EXPECTED_RECEPTOR}")
    print(f"AA输入后缀: {args.aa_input_suffix}")
    print(f"固定scheme: {scheme.name}")
    print(
        "有效阈值: "
        f"global={thresholds['global_count_threshold']}, "
        f"RA={thresholds['RA_count_threshold']}, "
        f"ILD={thresholds['ILD_count_threshold']}, "
        f"delta={scheme.specific_prevalence_delta}"
    )
    print(f"候选序列: {len(sequences):,}")

    catalog_rows, reference_rows = write_catalogs_chunked(
        catalog_path=outputs["catalog"],
        reference_path=outputs["reference_sets"],
        sequences=sequences,
        stats=stats,
        masks=masks,
        categories=categories,
        n_total=n_total,
        n_ra=n_ra,
        n_ild=n_ild,
        chunk_size=args.catalog_chunk_size,
    )

    descriptive = build_sample_feature_frame_from_sparse(
        sample_ids=sample_ids,
        clone_numbers=clone_numbers,
        presence=presence,
        frequency=frequency,
        masks=masks,
    )
    warnings = validate_descriptive_frame(descriptive, sample_ids)
    descriptive.to_csv(outputs["descriptive"], index=False, float_format="%.12g")

    reference_summary_df = reference_set_summary(
        masks=masks,
        thresholds=thresholds,
        scheme=scheme,
        n_total=n_total,
        n_ra=n_ra,
        n_ild=n_ild,
    )
    reference_summary_df.to_csv(outputs["reference_summary"], index=False)

    build_log = descriptive.copy()
    build_log.insert(1, "status", "PASS")
    build_log.to_csv(outputs["build_log"], index=False, float_format="%.12g")

    reference_hash = sha256_file(outputs["reference_sets"])
    definition = {
        "script_version": SCRIPT_VERSION,
        "receptor": EXPECTED_RECEPTOR,
        "aa_input_suffix": args.aa_input_suffix,
        "mode": "train",
        "scheme": scheme.to_dict(),
        "effective_thresholds": thresholds,
        "n_training_samples": n_total,
        "n_RA_training_samples": n_ra,
        "n_ILD_training_samples": n_ild,
        "candidate_min_full_training_sample_count": int(
            cache_manifest.get("candidate_min_total_count", 2)
        ),
        "candidate_sequence_count": len(sequences),
        "reference_set_sizes": {
            name: int(np.count_nonzero(masks[name])) for name in SET_ORDER
        },
        "reference_file": str(outputs["reference_sets"]),
        "reference_file_sha256": reference_hash,
        "stability_configuration": str(stability_config_path),
        "stability_configuration_scheme": args.scheme_name,
        "stability_cache_manifest": cache_manifest,
        "important_note": (
            "Full-training descriptive features are self-inclusive and must not be used "
            "as fixed predictors in ordinary cross-validation."
        ),
    }
    outputs["reference_definition"].write_text(
        json.dumps(definition, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    runtime = time.time() - started
    write_train_summary(
        path=outputs["summary"],
        args=args,
        scheme=scheme,
        thresholds=thresholds,
        metadata=metadata,
        reference_summary_df=reference_summary_df,
        descriptive=descriptive,
        outputs=outputs,
        catalog_rows=catalog_rows,
        reference_rows=reference_rows,
        warnings=warnings,
        runtime=runtime,
    )

    print("\n[reference set sizes]")
    print(reference_summary_df[["set_name", "set_size"]].to_string(index=False))
    print(f"\n描述性矩阵: {descriptive.shape}")
    print(f"catalog: {catalog_rows:,} rows")
    print(f"final reference: {reference_rows:,} rows")
    print(f"reference SHA256: {reference_hash}")
    print("\n[输出文件]")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    print(f"总运行时间: {runtime:.2f}s")
    return 0


def load_reference_mapping(
    reference_file: Path,
    definition: Mapping[str, object],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    required_cols = [
        "cdr3_aa",
        "is_global_public",
        "is_RA_group_public",
        "is_ILD_group_public",
        "is_between_group_shared",
        "is_RA_specific_ref",
        "is_ILD_specific_ref",
    ]
    df = pd.read_csv(reference_file, usecols=required_cols, dtype={"cdr3_aa": str})
    if df["cdr3_aa"].duplicated().any():
        raise ValueError("训练reference文件中cdr3_aa重复。")

    bit_values = (
        df["is_global_public"].astype(np.uint8) * BIT_GLOBAL
        + df["is_RA_group_public"].astype(np.uint8) * BIT_RA_GROUP
        + df["is_ILD_group_public"].astype(np.uint8) * BIT_ILD_GROUP
        + df["is_between_group_shared"].astype(np.uint8) * BIT_SHARED
        + df["is_RA_specific_ref"].astype(np.uint8) * BIT_RA_SPECIFIC
        + df["is_ILD_specific_ref"].astype(np.uint8) * BIT_ILD_SPECIFIC
    ).astype(np.uint8)
    mapping = dict(zip(df["cdr3_aa"].tolist(), bit_values.astype(int).tolist()))

    sizes = definition["reference_set_sizes"]
    expected_global = int(sizes["global_public"])
    if len(df) != expected_global:
        raise RuntimeError(
            f"训练reference文件行数({len(df)})与definition global_public({expected_global})不一致。"
        )
    size_map = {
        "global_public": expected_global,
        "RA_group_public": int(sizes["RA_group_public"]),
        "ILD_group_public": int(sizes["ILD_group_public"]),
        "between_group_shared": int(sizes["between_group_shared"]),
        "RA_specific_ref": int(sizes["RA_specific_ref"]),
        "ILD_specific_ref": int(sizes["ILD_specific_ref"]),
    }
    return mapping, size_map


def calculate_test_sample_features(
    sample_id: str,
    aa_file: Path,
    expected_clone_number: int,
    bit_mapping: Mapping[str, int],
    set_sizes: Mapping[str, int],
) -> Tuple[Dict[str, object], Dict[str, object]]:
    started = time.time()
    df = pd.read_csv(
        aa_file,
        usecols=["sample_id", "cdr3_aa", "frequency_norm"],
        dtype={"sample_id": str, "cdr3_aa": str},
    )
    if len(df) != expected_clone_number:
        raise ValueError(
            f"{sample_id} AA行数与01 summary不一致: AA表={len(df)}, summary={expected_clone_number}"
        )
    file_ids = df["sample_id"].astype(str).unique().tolist()
    if file_ids != [sample_id]:
        raise ValueError(f"{aa_file}内sample_id不一致: {file_ids[:5]}")

    sequences = df["cdr3_aa"].astype(str).str.strip().str.upper()
    invalid = sequences.eq("") | sequences.map(lambda x: not set(x).issubset(AA_SET))
    if invalid.any():
        raise ValueError(f"{sample_id}存在{int(invalid.sum())}条非标准AA序列。")
    frequency_values = pd.to_numeric(df["frequency_norm"], errors="coerce")
    if frequency_values.isna().any() or (~np.isfinite(frequency_values)).any():
        raise ValueError(f"{sample_id} frequency_norm存在非法值。")
    if (frequency_values < 0).any():
        raise ValueError(f"{sample_id} frequency_norm存在负数。")
    if not math.isclose(float(frequency_values.sum()), 1.0, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError(f"{sample_id} frequency_norm总和不是1: {frequency_values.sum()}")

    bits = sequences.map(bit_mapping).fillna(0).to_numpy(dtype=np.uint8)
    frequency_array = frequency_values.to_numpy(dtype=float)
    definitions = {
        "global_public": BIT_GLOBAL,
        "RA_group_public": BIT_RA_GROUP,
        "ILD_group_public": BIT_ILD_GROUP,
        "between_group_shared": BIT_SHARED,
        "RA_specific_ref": BIT_RA_SPECIFIC,
        "ILD_specific_ref": BIT_ILD_SPECIFIC,
    }

    row: Dict[str, object] = {
        "sample_id": sample_id,
        "aa_clone_number": int(expected_clone_number),
    }
    for set_name in SET_ORDER:
        flag = definitions[set_name]
        mask = (bits & flag) != 0
        count = int(np.count_nonzero(mask))
        prefix = DESCRIPTIVE_PREFIX[set_name]
        row[f"{prefix}_clone_number"] = count
        row[f"{prefix}_clone_ratio"] = count / expected_clone_number
        row[f"{prefix}_frequency_sum"] = float(frequency_array[mask].sum())
        row[f"{prefix}_set_coverage"] = (
            count / set_sizes[set_name] if set_sizes[set_name] > 0 else 0.0
        )

    log_row = {
        "sample_id": sample_id,
        "status": "PASS",
        "aa_clone_number": int(expected_clone_number),
        "matched_all_ref_clone_number": row["global_public_aa_clone_number"],
        "matched_all_ref_frequency_sum": row["global_public_aa_frequency_sum"],
        "ignored_nonreference_clone_number": int(expected_clone_number - row["global_public_aa_clone_number"]),
        "runtime_seconds": time.time() - started,
    }
    return row, log_row


def run_test(args: argparse.Namespace) -> int:
    started = time.time()
    metadata_path = Path(args.metadata).expanduser().resolve()
    summary_path = Path(args.summary_file).expanduser().resolve()
    aa_dir = Path(args.aa_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    reference_file = Path(args.reference_file).expanduser().resolve()
    reference_definition_file = Path(args.reference_definition).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = output_paths(output_dir, "test")
    ensure_outputs(outputs, args.overwrite)

    for path, label in [
        (metadata_path, "metadata"),
        (summary_path, "01 summary"),
        (reference_file, "训练reference文件"),
        (reference_definition_file, "训练reference definition"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"{label}不存在: {path}")
    if not aa_dir.is_dir():
        raise NotADirectoryError(f"AA目录不存在: {aa_dir}")

    metadata = read_metadata(metadata_path, args.id_col, args.cohort_col, require_cohort=False)
    sample_ids = metadata[args.id_col].tolist()
    summary_df = read_summary(summary_path, sample_ids)
    definition = json.loads(reference_definition_file.read_text(encoding="utf-8"))

    actual_hash = sha256_file(reference_file)
    expected_hash = definition.get("reference_file_sha256")
    if actual_hash != expected_hash:
        raise RuntimeError(
            "训练reference文件SHA256与definition不一致；可能文件被修改或引用错误。"
        )
    validate_reference_definition_for_igh(
        definition=definition,
        scheme_name=args.scheme_name,
        aa_input_suffix=args.aa_input_suffix,
    )

    bit_mapping, set_sizes = load_reference_mapping(reference_file, definition)
    print("=" * 80)
    print("03 IGH Public CDR3-AA 正式构建 — TEST")
    print("=" * 80)
    print(f"测试样本: {len(metadata)}")
    print(f"训练reference: {reference_file}")
    print(f"reference SHA256: {actual_hash}")
    print(f"receptor: {EXPECTED_RECEPTOR}")
    print(f"AA输入后缀: {args.aa_input_suffix}")
    print(f"scheme: {definition['scheme']['name']}")

    feature_rows: List[Dict[str, object]] = []
    log_rows: List[Dict[str, object]] = []
    for index, sample_id in enumerate(sample_ids, start=1):
        aa_file = aa_dir / f"{sample_id}{args.aa_input_suffix}"
        if not aa_file.is_file():
            raise FileNotFoundError(f"缺少AA clone表: {aa_file}")
        expected = int(summary_df.loc[sample_id, "aa_clone_number"])
        row, log_row = calculate_test_sample_features(
            sample_id=sample_id,
            aa_file=aa_file,
            expected_clone_number=expected,
            bit_mapping=bit_mapping,
            set_sizes=set_sizes,
        )
        feature_rows.append(row)
        log_rows.append(log_row)
        print(
            f"[{index:>3}/{len(sample_ids)}] {sample_id} | AA={expected:,} | "
            f"all_ref={row['global_public_aa_clone_number']:,} | "
            f"RA-spec={row['RA_specific_ref_clone_number']:,} | "
            f"ILD-spec={row['ILD_specific_ref_clone_number']:,} | PASS"
        )

    descriptive = pd.DataFrame(feature_rows)
    warnings = validate_descriptive_frame(descriptive, sample_ids)
    reference_features = make_reference_feature_frame(descriptive, args.epsilon)
    if reference_features.isna().any().any() or not np.isfinite(
        reference_features.drop(columns="sample_id").to_numpy(dtype=float)
    ).all():
        raise RuntimeError("test reference features存在缺失或非有限值。")

    descriptive.to_csv(outputs["descriptive"], index=False, float_format="%.12g")
    reference_features.to_csv(
        outputs["test_reference_features"], index=False, float_format="%.12g"
    )
    pd.DataFrame(log_rows).to_csv(outputs["build_log"], index=False, float_format="%.12g")

    used_definition = {
        **definition,
        "mode": "test_transform",
        "test_metadata": str(metadata_path),
        "test_sample_count": len(metadata),
        "reference_file_used": str(reference_file),
        "reference_file_sha256_verified": actual_hash,
        "test_labels_used_to_build_reference": False,
    }
    outputs["reference_definition_used"].write_text(
        json.dumps(used_definition, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    runtime = time.time() - started
    write_test_summary(
        path=outputs["summary"],
        metadata=metadata,
        descriptive=descriptive,
        reference_features=reference_features,
        definition=definition,
        outputs=outputs,
        warnings=warnings,
        runtime=runtime,
    )

    print(f"\n描述性矩阵: {descriptive.shape}")
    print(f"模型reference矩阵: {reference_features.shape}")
    print("\n[输出文件]")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    print(f"总运行时间: {runtime:.2f}s")
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "train":
        return run_train(args)
    return run_test(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
