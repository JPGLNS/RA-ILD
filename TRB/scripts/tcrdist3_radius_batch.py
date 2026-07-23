#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量进行患者内TRB TCRdist3稀疏距离计算和不同radius网络聚类。

流程：
1. 遍历single_table目录下所有单样本输入表。
2. 每个样本独立构建TCRrep。
3. 使用compute_pw_sparse_out_of_memory2计算患者内稀疏距离矩阵。
4. 统一保留TCRdist <= 50的距离。
5. 分别使用radius = 20, 22, 24, 26, 28, 30, 32构建网络。
6. 使用无向网络connected components作为cluster。
7. 保存cluster结果、clone-cluster映射和样本radius汇总。
8. 支持断点续跑。

注意：
- 每条clonotype是一个节点。
- TCRdist <= radius时建立无向边。
- 每个connected component定义为一个cluster。
- 无邻居的clonotype保留为singleton cluster。
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import re
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from tcrdist.rep_funcs import compute_pw_sparse_out_of_memory2
from tcrdist.repertoire import TCRrep


# ============================================================
# 1. 路径与全局参数
# ============================================================

INPUT_DIR = Path("/data/users/chenhaisheng/RA-ILD/TRB/vdj/result/single_table")

OUTPUT_ROOT = Path("/data/users/chenhaisheng/RA-ILD/TRB/vdj/result/tcrdist_all_samples")

INPUT_SUFFIX = "_TRB_tcrdist_input.csv"

# 正式运行时设为174；测试模式可以暂时设为None
EXPECTED_SAMPLE_COUNT = 174

# 测试前2个样本；正式运行时改为None
TEST_SAMPLE_LIMIT = None

DB_FILE = "alphabeta_gammadelta_db.tsv"

MAX_DISTANCE = 50

RADIUS_LIST = list(range(20, 33, 2))
# [20, 22, 24, 26, 28, 30, 32]

ROW_SIZE = 100

# 推荐将并行主要放在外层分块
PM_PROCESSES = 32
TR_CPUS = 1

# 已完成样本是否重新计算
OVERWRITE_COMPLETED = False

# 已存在距离矩阵时是否强制重新计算
FORCE_RECOMPUTE_DISTANCE = False

# gzip压缩级别；1速度较快
GZIP_COMPRESSION = {
    "method": "gzip",
    "compresslevel": 1,
}


# ============================================================
# 2. 输出目录和日志
# ============================================================

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_DIR = OUTPUT_ROOT / "logs"
LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BATCH_STATUS_FILE = OUTPUT_ROOT / "batch_processing_status.csv"

BATCH_ERROR_FILE = OUTPUT_ROOT / "batch_processing_errors.csv"


STATUS_FIELDS = [
    "timestamp",
    "sample_index",
    "total_samples",
    "sample_id",
    "subject",
    "status",
    "n_clonotypes",
    "elapsed_seconds",
    "message",
]

ERROR_FIELDS = [
    "timestamp",
    "sample_index",
    "total_samples",
    "sample_id",
    "input_file",
    "error_type",
    "error_message",
    "log_file",
]


# ============================================================
# 3. 通用函数
# ============================================================


def append_csv_row(
    file_path: Path,
    fieldnames: list,
    row: dict,
) -> None:
    """
    追加写入CSV日志。
    文件不存在时自动写入表头。
    """

    file_exists = file_path.is_file()

    with file_path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def write_json(
    data: dict,
    file_path: Path,
) -> None:
    """
    保存JSON。
    """

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            data,
            handle,
            indent=2,
            ensure_ascii=False,
        )


def write_csv_gzip(
    df: pd.DataFrame,
    file_path: Path,
) -> None:
    """
    使用较低gzip压缩级别保存CSV，以平衡速度和磁盘空间。
    """

    df.to_csv(
        file_path,
        index=False,
        compression=GZIP_COMPRESSION,
    )


def get_sample_id(
    input_file: Path,
) -> str:
    """
    从文件名中提取libraryid/sample_id。
    """

    if not input_file.name.endswith(INPUT_SUFFIX):
        raise ValueError(f"输入文件名不符合预期格式: {input_file.name}")

    return input_file.name[: -len(INPUT_SUFFIX)]


# ============================================================
# 4. 输入数据检查与整理
# ============================================================


def prepare_cell_df(
    input_file: Path,
) -> tuple[pd.DataFrame, str]:
    """
    读取并检查单样本输入文件，返回TCRrep使用的cell_df和subject。
    """

    required_cols = [
        "subject",
        "count",
        "v_b_gene",
        "j_b_gene",
        "cdr3_b_aa",
        "clone_id",
    ]

    df = pd.read_csv(input_file)

    missing_cols = [column for column in required_cols if column not in df.columns]

    if missing_cols:
        raise ValueError("输入文件缺少必要列: " + ", ".join(missing_cols))

    missing_summary = df[required_cols].isna().sum()

    if int(missing_summary.sum()) > 0:
        raise ValueError(
            "必要字段存在缺失值: "
            + missing_summary[missing_summary > 0].to_dict().__str__()
        )

    subjects = df["subject"].astype(str).str.strip().unique()

    if len(subjects) != 1:
        raise ValueError(
            f"单样本文件应只包含一个subject，实际发现: {subjects.tolist()}"
        )

    subject = subjects[0]

    # count检查
    df["count"] = pd.to_numeric(
        df["count"],
        errors="raise",
    )

    if (df["count"] <= 0).any():
        raise ValueError("发现count <= 0的记录")

    if (df["count"] % 1 != 0).any():
        raise ValueError("发现非整数count")

    df["count"] = df["count"].astype("int64")

    # V/J等位基因检查
    v_without_allele = ~(df["v_b_gene"].astype(str).str.contains(r"\*", regex=True))

    j_without_allele = ~(df["j_b_gene"].astype(str).str.contains(r"\*", regex=True))

    if v_without_allele.any():
        examples = (
            df.loc[
                v_without_allele,
                "v_b_gene",
            ]
            .drop_duplicates()
            .head(20)
            .tolist()
        )

        raise ValueError(
            "发现未包含等位基因后缀的V基因: " + ", ".join(map(str, examples))
        )

    if j_without_allele.any():
        examples = (
            df.loc[
                j_without_allele,
                "j_b_gene",
            ]
            .drop_duplicates()
            .head(20)
            .tolist()
        )

        raise ValueError(
            "发现未包含等位基因后缀的J基因: " + ", ".join(map(str, examples))
        )

    # CDR3-AA合法性检查
    valid_aa_pattern = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

    invalid_aa = ~(df["cdr3_b_aa"].astype(str).str.fullmatch(valid_aa_pattern))

    if invalid_aa.any():
        examples = (
            df.loc[
                invalid_aa,
                "cdr3_b_aa",
            ]
            .head(20)
            .tolist()
        )

        raise ValueError("发现不合法的CDR3-AA序列: " + ", ".join(map(str, examples)))

    # 原始clone_id唯一性
    if df["clone_id"].duplicated().any():
        raise ValueError("原始clone_id在样本内不是唯一值")

    # 患者内clonotype重复检查
    clone_key = [
        "subject",
        "v_b_gene",
        "j_b_gene",
        "cdr3_b_aa",
    ]

    duplicated_mask = df.duplicated(
        subset=clone_key,
        keep=False,
    )

    if duplicated_mask.any():
        raise ValueError(
            "输入表中存在重复的"
            "subject+V+J+CDR3-AA记录，"
            f"重复行数: {int(duplicated_mask.sum())}"
        )

    # 保留上游clone编号
    df = df.rename(
        columns={
            "clone_id": "source_clone_id",
        }
    )

    cell_df = df[
        [
            "subject",
            "count",
            "v_b_gene",
            "j_b_gene",
            "cdr3_b_aa",
            "source_clone_id",
        ]
    ].copy()

    return cell_df, subject


# ============================================================
# 5. 构建或读取TCRdist稀疏距离矩阵
# ============================================================


def get_sparse_distance(
    cell_df: pd.DataFrame,
    subject: str,
    sample_id: str,
    sample_dir: Path,
) -> tuple[pd.DataFrame, sparse.csr_matrix, dict]:
    """
    如果距离矩阵和clone index已经存在，则直接读取。
    否则建立TCRrep并进行分块稀疏距离计算。
    """

    sparse_dir = sample_dir / "sparse_distance"

    sparse_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    clone_index_file = sample_dir / f"{sample_id}_TCRrep_clone_index.csv.gz"

    sparse_matrix_file = sparse_dir / (f"{sample_id}_rw_beta_maxdist{MAX_DISTANCE}.npz")

    distance_summary_file = sparse_dir / f"{sample_id}_sparse_distance_summary.csv"

    parameter_file = sparse_dir / f"{sample_id}_sparse_distance_parameters.json"

    can_reuse = (
        clone_index_file.is_file()
        and sparse_matrix_file.is_file()
        and not FORCE_RECOMPUTE_DISTANCE
    )

    if can_reuse:
        clone_index_df = pd.read_csv(clone_index_file)

        rw_beta = sparse.load_npz(sparse_matrix_file).tocsr()

        distance_elapsed_seconds = np.nan

    else:
        distance_start = time.time()

        tr = TCRrep(
            cell_df=cell_df,
            organism="human",
            chains=["beta"],
            db_file=DB_FILE,
            compute_distances=False,
            store_all_cdr=False,
        )

        if len(tr.clone_df) != len(cell_df):
            tr.show_incomplete()

            raise ValueError(
                "TCRrep初始化后clone数量发生变化: "
                f"input={len(cell_df):,}, "
                f"clone_df={len(tr.clone_df):,}"
            )

        clone_index_df = tr.clone_df.copy()

        clone_index_df.insert(
            0,
            "tcrdist_row_index",
            np.arange(
                len(clone_index_df),
                dtype=np.int64,
            ),
        )

        write_csv_gzip(
            clone_index_df,
            clone_index_file,
        )

        tr.cpus = TR_CPUS

        original_working_directory = Path.cwd()

        try:
            # tcrdist3中间分块目录创建在当前工作目录；
            # 切换到样本目录，避免临时文件出现在项目根目录。
            os.chdir(sample_dir)

            sparse_matrices, _ = compute_pw_sparse_out_of_memory2(
                tr=tr,
                row_size=ROW_SIZE,
                pm_processes=PM_PROCESSES,
                pm_pbar=False,
                max_distance=MAX_DISTANCE,
                reassemble=True,
                cleanup=True,
                assign=True,
            )

        finally:
            os.chdir(original_working_directory)

        if not isinstance(
            sparse_matrices,
            dict,
        ):
            raise TypeError("稀疏距离函数返回结果不是字典")

        if "beta" not in sparse_matrices:
            raise KeyError("稀疏距离结果中没有beta矩阵")

        rw_beta = sparse_matrices["beta"].tocsr()

        sparse.save_npz(
            sparse_matrix_file,
            rw_beta,
            compressed=True,
        )

        distance_elapsed_seconds = time.time() - distance_start

        del tr
        del sparse_matrices

        gc.collect()

    # --------------------------------------------------------
    # 距离矩阵检查
    # --------------------------------------------------------

    n_clones = len(clone_index_df)

    expected_shape = (
        n_clones,
        n_clones,
    )

    if rw_beta.shape != expected_shape:
        raise ValueError(
            "距离矩阵维度与clone index不一致: "
            f"matrix={rw_beta.shape}, "
            f"clone_index={expected_shape}"
        )

    if rw_beta.data.size == 0:
        raise ValueError("稀疏距离矩阵没有保存任何元素")

    stored_min = int(rw_beta.data.min())

    stored_max = int(rw_beta.data.max())

    if stored_min < 1:
        raise ValueError(f"发现小于1的异常已保存距离值: {stored_min}")

    if stored_max > MAX_DISTANCE:
        raise ValueError(f"发现大于MAX_DISTANCE的距离: {stored_max}")

    diagonal = rw_beta.diagonal()

    diagonal_one_n = int(np.count_nonzero(diagonal == 1))

    if diagonal_one_n != n_clones:
        raise ValueError(f"稀疏距离矩阵对角线不全为1: {diagonal_one_n:,}/{n_clones:,}")

    asymmetry_nnz = int((rw_beta != rw_beta.transpose()).nnz)

    if asymmetry_nnz != 0:
        raise ValueError(f"患者内稀疏距离矩阵不对称: {asymmetry_nnz:,}")

    undirected_pairs = int((rw_beta.nnz - n_clones) // 2)

    distance_summary = {
        "sample_id": sample_id,
        "subject": subject,
        "n_clonotypes": int(n_clones),
        "matrix_rows": int(rw_beta.shape[0]),
        "matrix_columns": int(rw_beta.shape[1]),
        "stored_elements_nnz": int(rw_beta.nnz),
        "stored_minimum": stored_min,
        "stored_maximum": stored_max,
        "diagonal_entries_equal_1": (diagonal_one_n),
        "asymmetry_nnz": asymmetry_nnz,
        "undirected_pairs_within_max_distance": (undirected_pairs),
        "max_distance": MAX_DISTANCE,
        "row_size": ROW_SIZE,
        "pm_processes": PM_PROCESSES,
        "tr_cpus": TR_CPUS,
        "distance_elapsed_seconds": (distance_elapsed_seconds),
    }

    pd.DataFrame([distance_summary]).to_csv(
        distance_summary_file,
        index=False,
    )

    write_json(
        {
            "sample_id": sample_id,
            "subject": subject,
            "distance_definition": ("default full beta-chain TCRdist"),
            "max_distance": MAX_DISTANCE,
            "row_size": ROW_SIZE,
            "pm_processes": PM_PROCESSES,
            "tr_cpus": TR_CPUS,
            "self_distance_zero_encoding": 1,
            "reassemble": True,
            "cleanup": True,
        },
        parameter_file,
    )

    return (
        clone_index_df,
        rw_beta,
        distance_summary,
    )


# ============================================================
# 6. 单个radius网络聚类
# ============================================================


def cluster_one_radius(
    rw_beta: sparse.csr_matrix,
    clone_index_df: pd.DataFrame,
    sample_id: str,
    subject: str,
    radius: int,
    radius_dir: Path,
) -> dict:
    """
    根据指定radius建立无向网络并使用connected components聚类。
    """

    n_nodes = len(clone_index_df)

    clone_counts = pd.to_numeric(
        clone_index_df["count"],
        errors="raise",
    ).to_numpy(dtype=np.int64)

    if np.any(clone_counts <= 0):
        raise ValueError("clone index中存在count <= 0")

    total_clone_reads = int(clone_counts.sum())

    # --------------------------------------------------------
    # 构建当前radius的二值邻接矩阵
    # --------------------------------------------------------

    adjacency = rw_beta.copy().tocsr()

    adjacency.data = (adjacency.data <= radius).astype(np.int8)

    adjacency.eliminate_zeros()

    # 删除自身连接
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    # 确保无向、对称
    adjacency = adjacency.maximum(adjacency.transpose()).tocsr()

    adjacency.sort_indices()

    if np.count_nonzero(adjacency.diagonal()) != 0:
        raise ValueError(f"radius={radius}的邻接矩阵对角线非零")

    adjacency_asymmetry = int((adjacency != adjacency.transpose()).nnz)

    if adjacency_asymmetry != 0:
        raise ValueError(f"radius={radius}邻接矩阵不对称")

    if adjacency.nnz % 2 != 0:
        raise ValueError(f"radius={radius}邻接矩阵的非对角线元素数量不是偶数")

    node_degree = np.diff(adjacency.indptr).astype(np.int64)

    undirected_edges = int(adjacency.nnz // 2)

    # --------------------------------------------------------
    # 连通分量聚类
    # --------------------------------------------------------

    n_components, labels = connected_components(
        csgraph=adjacency,
        directed=False,
        return_labels=True,
    )

    labels = labels.astype(np.int64)

    if len(labels) != n_nodes:
        raise ValueError(f"radius={radius}的cluster label数量错误")

    cluster_clone_counts = np.bincount(
        labels,
        minlength=n_components,
    ).astype(np.int64)

    cluster_read_counts = np.rint(
        np.bincount(
            labels,
            weights=clone_counts,
            minlength=n_components,
        )
    ).astype(np.int64)

    if int(cluster_clone_counts.sum()) != n_nodes:
        raise ValueError(f"radius={radius}的cluster clone总数错误")

    if int(cluster_read_counts.sum()) != total_clone_reads:
        raise ValueError(f"radius={radius}的cluster read总数错误")

    cluster_read_fractions = cluster_read_counts / total_clone_reads

    # --------------------------------------------------------
    # 网络和cluster指标
    # --------------------------------------------------------

    isolated_clone_count = int(np.count_nonzero(node_degree == 0))

    singleton_mask = cluster_clone_counts == 1

    singleton_cluster_count = int(singleton_mask.sum())

    if singleton_cluster_count != isolated_clone_count:
        raise ValueError(f"radius={radius}: singleton cluster数量与孤立节点数量不一致")

    non_singleton_cluster_count = int(n_components - singleton_cluster_count)

    largest_cluster_clone_count = int(cluster_clone_counts.max())

    largest_cluster_clone_fraction = float(largest_cluster_clone_count / n_nodes)

    largest_cluster_read_count = int(cluster_read_counts.max())

    largest_cluster_read_fraction = float(
        largest_cluster_read_count / total_clone_reads
    )

    clones_in_non_singleton_clusters = int(
        cluster_clone_counts[cluster_clone_counts > 1].sum()
    )

    # --------------------------------------------------------
    # 多样性指标
    # --------------------------------------------------------

    positive_fractions = cluster_read_fractions[cluster_read_fractions > 0]

    cluster_shannon = float(-np.sum(positive_fractions * np.log(positive_fractions)))

    cluster_richness = int(n_components)

    if cluster_richness > 1:
        cluster_pielou_evenness = float(cluster_shannon / math.log(cluster_richness))

        cluster_clonality = float(1.0 - cluster_pielou_evenness)

    else:
        cluster_pielou_evenness = np.nan
        cluster_clonality = np.nan

    effective_cluster_number = float(math.exp(cluster_shannon))

    # --------------------------------------------------------
    # 保存cluster汇总结果
    # --------------------------------------------------------

    cluster_ids = np.arange(
        1,
        n_components + 1,
        dtype=np.int64,
    )

    cluster_summary_df = pd.DataFrame(
        {
            "sample_id": sample_id,
            "subject": subject,
            "radius": radius,
            "cluster_id": cluster_ids,
            "clone_count": (cluster_clone_counts),
            "read_count": (cluster_read_counts),
            "read_fraction": (cluster_read_fractions),
            "is_singleton": (singleton_mask),
        }
    )

    cluster_summary_df["cluster_rank_by_read_count"] = (
        cluster_summary_df["read_count"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    cluster_summary_df = cluster_summary_df.sort_values(
        [
            "read_count",
            "clone_count",
            "cluster_id",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    cluster_summary_file = radius_dir / (
        f"{sample_id}_radius{radius:02d}_cluster_summary.csv.gz"
    )

    write_csv_gzip(
        cluster_summary_df,
        cluster_summary_file,
    )

    # --------------------------------------------------------
    # 保存clone到cluster映射
    # --------------------------------------------------------

    one_based_labels = labels + 1

    clone_assignment_df = clone_index_df.copy()

    clone_assignment_df.insert(
        1,
        "radius",
        radius,
    )

    clone_assignment_df.insert(
        2,
        "cluster_id",
        one_based_labels,
    )

    clone_assignment_df["cluster_clone_count"] = cluster_clone_counts[labels]

    clone_assignment_df["cluster_read_count"] = cluster_read_counts[labels]

    clone_assignment_df["cluster_read_fraction"] = cluster_read_fractions[labels]

    clone_assignment_df["node_degree"] = node_degree

    clone_assignment_df["is_singleton_cluster"] = cluster_clone_counts[labels] == 1

    clone_assignment_file = radius_dir / (
        f"{sample_id}_radius{radius:02d}_clone_cluster_assignment.csv.gz"
    )

    write_csv_gzip(
        clone_assignment_df,
        clone_assignment_file,
    )

    possible_edges = n_nodes * (n_nodes - 1) / 2

    network_density = (
        undirected_edges / possible_edges if possible_edges > 0 else np.nan
    )

    radius_summary = {
        "sample_id": sample_id,
        "subject": subject,
        "radius": radius,
        "n_clonotypes": int(n_nodes),
        "total_clone_reads": int(total_clone_reads),
        "undirected_edges": int(undirected_edges),
        "network_density": float(network_density),
        "mean_node_degree": float(node_degree.mean()),
        "median_node_degree": float(np.median(node_degree)),
        "maximum_node_degree": int(node_degree.max()),
        "isolated_clone_count": int(isolated_clone_count),
        "isolated_clone_fraction": float(isolated_clone_count / n_nodes),
        "cluster_richness": int(cluster_richness),
        "singleton_cluster_count": int(singleton_cluster_count),
        "singleton_cluster_fraction": float(singleton_cluster_count / cluster_richness),
        "non_singleton_cluster_count": int(non_singleton_cluster_count),
        "clones_in_non_singleton_clusters": int(clones_in_non_singleton_clusters),
        "clones_in_non_singleton_fraction": float(
            clones_in_non_singleton_clusters / n_nodes
        ),
        "cluster_size_mean": float(cluster_clone_counts.mean()),
        "cluster_size_median": float(np.median(cluster_clone_counts)),
        "cluster_size_p90": float(
            np.quantile(
                cluster_clone_counts,
                0.90,
            )
        ),
        "cluster_size_p95": float(
            np.quantile(
                cluster_clone_counts,
                0.95,
            )
        ),
        "cluster_size_p99": float(
            np.quantile(
                cluster_clone_counts,
                0.99,
            )
        ),
        "largest_cluster_clone_count": int(largest_cluster_clone_count),
        "largest_cluster_clone_fraction": float(largest_cluster_clone_fraction),
        "largest_cluster_read_count": int(largest_cluster_read_count),
        "largest_cluster_read_fraction": float(largest_cluster_read_fraction),
        "cluster_shannon": float(cluster_shannon),
        "cluster_pielou_evenness": float(cluster_pielou_evenness),
        "cluster_clonality": float(cluster_clonality),
        "effective_cluster_number": float(effective_cluster_number),
        "cluster_summary_file": str(cluster_summary_file),
        "clone_assignment_file": str(clone_assignment_file),
    }

    del adjacency
    del labels
    del node_degree
    del cluster_clone_counts
    del cluster_read_counts
    del cluster_summary_df
    del clone_assignment_df

    gc.collect()

    return radius_summary


# ============================================================
# 7. 处理单个样本
# ============================================================


def process_sample(
    input_file: Path,
) -> dict:
    """
    完整处理一个样本。
    """

    sample_start = time.time()

    sample_id = get_sample_id(input_file)

    sample_dir = OUTPUT_ROOT / sample_id

    sample_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    radius_dir = sample_dir / "radius_results"

    radius_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    success_file = sample_dir / "_SUCCESS.json"

    if success_file.is_file() and not OVERWRITE_COMPLETED:
        success_info = json.loads(success_file.read_text(encoding="utf-8"))

        return {
            "status": "skipped_complete",
            "sample_id": sample_id,
            "subject": success_info.get(
                "subject",
                "",
            ),
            "n_clonotypes": success_info.get(
                "n_clonotypes",
                "",
            ),
            "elapsed_seconds": 0.0,
        }

    cell_df, subject = prepare_cell_df(input_file)

    (
        clone_index_df,
        rw_beta,
        distance_summary,
    ) = get_sparse_distance(
        cell_df=cell_df,
        subject=subject,
        sample_id=sample_id,
        sample_dir=sample_dir,
    )

    radius_summary_rows = []

    for radius in RADIUS_LIST:
        radius_summary = cluster_one_radius(
            rw_beta=rw_beta,
            clone_index_df=(clone_index_df),
            sample_id=sample_id,
            subject=subject,
            radius=radius,
            radius_dir=radius_dir,
        )

        radius_summary_rows.append(radius_summary)

    radius_summary_df = (
        pd.DataFrame(radius_summary_rows).sort_values("radius").reset_index(drop=True)
    )

    # 基本单调性检查
    if not radius_summary_df["undirected_edges"].is_monotonic_increasing:
        raise ValueError("edge数量没有随radius单调增加")

    if not radius_summary_df["cluster_richness"].is_monotonic_decreasing:
        raise ValueError("cluster richness没有随radius单调减少")

    if not radius_summary_df["singleton_cluster_count"].is_monotonic_decreasing:
        raise ValueError("singleton数量没有随radius单调减少")

    radius_summary_file = sample_dir / f"{sample_id}_radius_summary.csv"

    radius_summary_df.to_csv(
        radius_summary_file,
        index=False,
    )

    elapsed_seconds = time.time() - sample_start

    success_info = {
        "sample_id": sample_id,
        "subject": subject,
        "n_clonotypes": int(len(clone_index_df)),
        "max_distance": MAX_DISTANCE,
        "radius_list": RADIUS_LIST,
        "clustering_rule": ("undirected connected components"),
        "completed_at": (datetime.now().isoformat(timespec="seconds")),
        "elapsed_seconds": float(elapsed_seconds),
        "radius_summary_file": str(radius_summary_file),
        "sparse_matrix_nnz": int(rw_beta.nnz),
        "distance_summary": (distance_summary),
    }

    write_json(
        success_info,
        success_file,
    )

    del cell_df
    del clone_index_df
    del rw_beta
    del radius_summary_df

    gc.collect()

    return {
        "status": "completed",
        "sample_id": sample_id,
        "subject": subject,
        "n_clonotypes": int(success_info["n_clonotypes"]),
        "elapsed_seconds": float(elapsed_seconds),
    }


# ============================================================
# 8. 批量主程序
# ============================================================


def main() -> None:
    """
    遍历全部样本并顺序处理。
    """

    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {INPUT_DIR}")

    input_files = sorted(INPUT_DIR.glob(f"*{INPUT_SUFFIX}"))

    if TEST_SAMPLE_LIMIT is not None:
        input_files = input_files[:TEST_SAMPLE_LIMIT]

    total_samples = len(input_files)

    if total_samples == 0:
        raise FileNotFoundError(f"输入目录中没有找到单样本文件: {INPUT_DIR}")

    if EXPECTED_SAMPLE_COUNT is not None and total_samples != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            "检测到的样本数量与预期不一致: "
            f"检测到={total_samples}, "
            f"预期={EXPECTED_SAMPLE_COUNT}"
        )

    completed_count = 0
    skipped_count = 0
    failed_count = 0

    for sample_index, input_file in enumerate(
        input_files,
        start=1,
    ):
        sample_id = get_sample_id(input_file)

        sample_log_file = LOG_DIR / f"{sample_id}.log"

        try:
            # 捕获TCRrep和tcrdist3内部输出，
            # 避免终端打印大量信息。
            with sample_log_file.open(
                "a",
                encoding="utf-8",
            ) as log_handle:
                with redirect_stdout(log_handle), redirect_stderr(log_handle):
                    result = process_sample(input_file)

            status = result["status"]

            if status == "completed":
                completed_count += 1

                print(
                    f"[{sample_index:03d}/{total_samples:03d}] "
                    f"COMPLETED  "
                    f"{result['sample_id']}  "
                    f"subject={result['subject']}  "
                    f"clones={result['n_clonotypes']:,}",
                    flush=True,
                )

            elif status == "skipped_complete":
                skipped_count += 1

                n_clonotypes = result.get(
                    "n_clonotypes",
                    "",
                )

                clone_text = f"{int(n_clonotypes):,}" if n_clonotypes != "" else ""

                print(
                    f"[{sample_index:03d}/{total_samples:03d}] "
                    f"SKIPPED    "
                    f"{result['sample_id']}  "
                    f"subject={result['subject']}  "
                    f"clones={clone_text}",
                    flush=True,
                )

            append_csv_row(
                BATCH_STATUS_FILE,
                STATUS_FIELDS,
                {
                    "timestamp": (datetime.now().isoformat(timespec="seconds")),
                    "sample_index": (sample_index),
                    "total_samples": (total_samples),
                    "sample_id": result["sample_id"],
                    "subject": result["subject"],
                    "status": status,
                    "n_clonotypes": result["n_clonotypes"],
                    "elapsed_seconds": (result["elapsed_seconds"]),
                    "message": "",
                },
            )

        except Exception as exc:
            failed_count += 1

            with sample_log_file.open(
                "a",
                encoding="utf-8",
            ) as log_handle:
                log_handle.write("\n\n" + "=" * 80 + "\nERROR\n" + "=" * 80 + "\n")

                traceback.print_exc(file=log_handle)

            error_message = str(exc).replace(
                "\n",
                " ",
            )

            append_csv_row(
                BATCH_ERROR_FILE,
                ERROR_FIELDS,
                {
                    "timestamp": (datetime.now().isoformat(timespec="seconds")),
                    "sample_index": (sample_index),
                    "total_samples": (total_samples),
                    "sample_id": sample_id,
                    "input_file": str(input_file),
                    "error_type": type(exc).__name__,
                    "error_message": (error_message),
                    "log_file": str(sample_log_file),
                },
            )

            append_csv_row(
                BATCH_STATUS_FILE,
                STATUS_FIELDS,
                {
                    "timestamp": (datetime.now().isoformat(timespec="seconds")),
                    "sample_index": (sample_index),
                    "total_samples": (total_samples),
                    "sample_id": sample_id,
                    "subject": "",
                    "status": "failed",
                    "n_clonotypes": "",
                    "elapsed_seconds": "",
                    "message": (error_message),
                },
            )

            print(
                f"[{sample_index:03d}/{total_samples:03d}] "
                f"FAILED     "
                f"{sample_id}  "
                f"log={sample_log_file}",
                flush=True,
            )

        gc.collect()

    print(
        "Batch finished: "
        f"completed={completed_count}, "
        f"skipped={skipped_count}, "
        f"failed={failed_count}, "
        f"total={total_samples}",
        flush=True,
    )


if __name__ == "__main__":
    main()
