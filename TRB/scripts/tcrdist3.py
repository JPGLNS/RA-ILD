import re
import pandas as pd
from tcrdist.repertoire import TCRrep
from pathlib import Path

# 循环处理每个样本
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# 1. 文件路径
# ============================================================

input_file = Path(
    "/data/users/chenhaisheng/RA-ILD/TRB/vdj/result/single_table/"
    "MGI260115R01-10_TRB_tcrdist_input.csv"
)

output_dir = Path(
    "/data/users/chenhaisheng/RA-ILD/TRB/vdj/result/single_tcrdist_test/MGI260115R01-10"
)

output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 读取输入文件
# ============================================================

if not input_file.is_file():
    raise FileNotFoundError(f"输入文件不存在: {input_file}")

df = pd.read_csv(input_file)

print("=" * 70)
print("Input file:", input_file)
print("Input shape:", df.shape)
print("Input columns:", df.columns.tolist())
print("=" * 70)


# ============================================================
# 3. 检查必要列
# ============================================================

required_cols = [
    "subject",
    "count",
    "v_b_gene",
    "j_b_gene",
    "cdr3_b_aa",
    "clone_id",
]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError("输入文件缺少必要列: " + ", ".join(missing_cols))


# ============================================================
# 4. 只允许一个患者
# ============================================================

subjects = df["subject"].dropna().astype(str).unique()

if len(subjects) != 1:
    raise ValueError(f"该文件应只包含一个subject，实际发现: {subjects.tolist()}")

subject = subjects[0]

print("Subject:", subject)
print("Input clonotypes:", f"{len(df):,}")


# ============================================================
# 5. 必要字段缺失检查
# ============================================================

missing_summary = df[required_cols].isna().sum()

print("\nMissing-value summary:")
print(missing_summary)

if missing_summary.sum() > 0:
    raise ValueError("必要字段存在缺失值")


# ============================================================
# 6. count检查
# ============================================================

df["count"] = pd.to_numeric(
    df["count"],
    errors="raise",
)

if (df["count"] <= 0).any():
    raise ValueError("发现 count <= 0 的记录")

if (df["count"] % 1 != 0).any():
    raise ValueError("发现非整数 count")

df["count"] = df["count"].astype("int64")


# ============================================================
# 7. V/J基因格式检查
# ============================================================

v_without_allele = ~df["v_b_gene"].astype(str).str.contains(
    r"\*",
    regex=True,
)

j_without_allele = ~df["j_b_gene"].astype(str).str.contains(
    r"\*",
    regex=True,
)

if v_without_allele.any():
    examples = df.loc[v_without_allele, "v_b_gene"].drop_duplicates().head(20).tolist()

    raise ValueError("发现未包含等位基因后缀的V基因，例如: " + ", ".join(examples))

if j_without_allele.any():
    examples = df.loc[j_without_allele, "j_b_gene"].drop_duplicates().head(20).tolist()

    raise ValueError("发现未包含等位基因后缀的J基因，例如: " + ", ".join(examples))


# ============================================================
# 8. 检查患者内是否仍有重复clonotype
# ============================================================

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

duplicated_n = int(duplicated_mask.sum())

print("\nDuplicated subject+V+J+CDR3-AA rows:", duplicated_n)

if duplicated_n > 0:
    print(
        df.loc[
            duplicated_mask,
            clone_key + ["count", "clone_id"],
        ]
        .sort_values(clone_key)
        .head(30)
    )

    raise ValueError("输入表中仍有重复的患者内V+J+CDR3-AA记录")


# ============================================================
# 9. 保留原始clone_id
#
# tcrdist3会创建自己的内部clone_id，因此将原始编号改名为
# source_clone_id，方便后续将距离和聚类结果映射回原始记录。
# ============================================================

df = df.rename(
    columns={
        "clone_id": "source_clone_id",
    }
)


# ============================================================
# 10. 构建精简的TCRrep输入表
# ============================================================

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

print("\nTCRrep input preview:")
print(cell_df.head())

print("\nTCRrep input shape:", cell_df.shape)


# ============================================================
# 11. 初始化TCRrep
#
# 注意：
# compute_distances=False
# 当前只建立TCRrep，不计算N×N距离矩阵。
# ============================================================

tr = TCRrep(
    cell_df=cell_df,
    organism="human",
    chains=["beta"],
    db_file="alphabeta_gammadelta_db.tsv",
    compute_distances=False,
    store_all_cdr=False,
)


# ============================================================
# 12. 检查TCRrep结果
# ============================================================

print("\n" + "=" * 70)
print("TCRrep initialization completed")
print("=" * 70)

print("Original input rows:", f"{len(cell_df):,}")
print("tr.cell_df rows:", f"{len(tr.cell_df):,}")
print("tr.clone_df rows:", f"{len(tr.clone_df):,}")

print("\nTCRrep clone_df columns:")
print(tr.clone_df.columns.tolist())

print("\nTCRrep clone_df preview:")
print(tr.clone_df.head())


# ============================================================
# 13. 显示未成功进入clone_df的记录
# ============================================================

print("\nIncomplete records reported by tcrdist3:")
tr.show_incomplete()


# ============================================================
# 14. 检查是否有记录被删除
# ============================================================

n_removed = len(cell_df) - len(tr.clone_df)

print("\nRecords removed during TCRrep initialization:", n_removed)

if n_removed != 0:
    print("WARNING: 有记录未进入tr.clone_df，请检查上方show_incomplete()结果。")
else:
    print("All input clonotypes were accepted by TCRrep.")


# ============================================================
# 15. 添加与距离矩阵对应的行号
#
# 后续稀疏距离矩阵的行和列，均对应clone_df的当前顺序。
# ============================================================

clone_index_df = tr.clone_df.copy()

clone_index_df.insert(
    0,
    "tcrdist_row_index",
    range(len(clone_index_df)),
)


# ============================================================
# 16. 保存TCRrep处理结果
# ============================================================

clone_df_file = output_dir / "MGI260115R01-10_TCRrep_clone_df.csv"

clone_index_file = output_dir / "MGI260115R01-10_TCRrep_clone_index.csv"

cell_df_file = output_dir / "MGI260115R01-10_TCRrep_cell_df.csv"

tr.clone_df.to_csv(
    clone_df_file,
    index=False,
)

clone_index_df.to_csv(
    clone_index_file,
    index=False,
)

tr.cell_df.to_csv(
    cell_df_file,
    index=False,
)


# ============================================================
# 17. 最终汇总
# ============================================================

print("\n" + "=" * 70)
print("Finished")
print("=" * 70)

print("Subject:", subject)
print("Accepted clonotypes:", f"{len(tr.clone_df):,}")
print("Output directory:", output_dir)
print("clone_df:", clone_df_file)
print("clone index:", clone_index_file)
print("cell_df:", cell_df_file)
print("=" * 70)


# ============================================================
# 18. 患者内分块稀疏TCRdist计算
#
# 使用默认完整 beta-chain TCRdist：
# CDR1β + CDR2β + CDR2.5β + 3 × CDR3β
#
# 注意：
# max_distance只是本次保存距离的最大值，
# 后续实际聚类radius必须 <= max_distance。
# ============================================================

import json
import time

import numpy as np
from scipy import sparse
from tcrdist.rep_funcs import compute_pw_sparse_out_of_memory2


# ------------------------------------------------------------
# 18.1 计算参数
# ------------------------------------------------------------

MAX_DISTANCE = 50
ROW_SIZE = 100
PM_PROCESSES = 16

# 每个外层进程内部只使用1个CPU，
# 避免多层并行造成CPU过度占用
tr.cpus = 1


# ------------------------------------------------------------
# 18.2 创建稀疏距离输出目录
# ------------------------------------------------------------

sparse_output_dir = output_dir / "sparse_distance"
sparse_output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

sparse_matrix_file = sparse_output_dir / "MGI260115R01-10_rw_beta_maxdist50.npz"

parameter_file = sparse_output_dir / "MGI260115R01-10_sparse_distance_parameters.json"

summary_file = sparse_output_dir / "MGI260115R01-10_sparse_distance_summary.csv"


# ------------------------------------------------------------
# 18.3 显示计算规模
# ------------------------------------------------------------

n_clones = len(tr.clone_df)

print("")
print("=" * 70)
print("Starting sparse TCRdist calculation")
print("=" * 70)
print("Subject:", subject)
print("Number of clonotypes:", f"{n_clones:,}")
print("Maximum stored distance:", MAX_DISTANCE)
print("Rows per chunk:", ROW_SIZE)
print("Parallel processes:", PM_PROCESSES)
print("Internal CPUs per process:", tr.cpus)
print("Output directory:", sparse_output_dir)
print("=" * 70)


# ------------------------------------------------------------
# 18.4 分块、稀疏、磁盘辅助计算
#
# 返回值：
# sparse_matrices：
#     字典；单独beta链时使用 sparse_matrices["beta"]
#
# fragments：
#     中间分块信息
#
# reassemble=True：
#     将分块重新组合为一个CSR稀疏矩阵
#
# cleanup=True：
#     组合完成后删除临时分块目录
#
# assign=True：
#     同时将结果赋值给 tr.pw_beta
# ------------------------------------------------------------

start_time = time.time()

sparse_matrices, fragments = compute_pw_sparse_out_of_memory2(
    tr=tr,
    row_size=ROW_SIZE,
    pm_processes=PM_PROCESSES,
    pm_pbar=True,
    max_distance=MAX_DISTANCE,
    reassemble=True,
    cleanup=True,
    assign=True,
)

elapsed_seconds = time.time() - start_time


# ------------------------------------------------------------
# 18.5 提取beta链稀疏距离矩阵
# ------------------------------------------------------------

if not isinstance(sparse_matrices, dict):
    raise TypeError(
        f"compute_pw_sparse_out_of_memory2返回结果不是字典: {type(sparse_matrices)}"
    )

if "beta" not in sparse_matrices:
    raise KeyError(
        f"稀疏距离结果中没有beta矩阵，现有键: {list(sparse_matrices.keys())}"
    )

rw_beta = sparse_matrices["beta"]


# ------------------------------------------------------------
# 18.6 基本类型和维度检查
# ------------------------------------------------------------

if not sparse.issparse(rw_beta):
    raise TypeError(f"beta距离结果不是scipy稀疏矩阵: {type(rw_beta)}")

rw_beta = rw_beta.tocsr()

expected_shape = (n_clones, n_clones)

if rw_beta.shape != expected_shape:
    raise ValueError(f"稀疏矩阵维度错误: 实际={rw_beta.shape}, 预期={expected_shape}")

print("")
print("Sparse matrix calculation completed.")
print("Matrix type:", type(rw_beta))
print("Matrix shape:", rw_beta.shape)
print("Stored elements (nnz):", f"{rw_beta.nnz:,}")


# ------------------------------------------------------------
# 18.7 检查距离范围
#
# compute_pw_sparse_out_of_memory2中：
# - 稀疏矩阵中的0：未保存，通常表示距离 > MAX_DISTANCE
# - 原始距离0：通常编码为1
# - 其他正整数：保留的TCRdist
# ------------------------------------------------------------

stored_values = rw_beta.data

if stored_values.size == 0:
    raise ValueError("稀疏矩阵没有保存任何元素")

stored_min = int(stored_values.min())
stored_max = int(stored_values.max())

if stored_min < 1:
    raise ValueError(f"发现小于1的异常距离值: {stored_min}")

if stored_max > MAX_DISTANCE:
    raise ValueError(f"发现大于MAX_DISTANCE的已保存距离: {stored_max}")

print("Minimum stored value:", stored_min)
print("Maximum stored value:", stored_max)
print("Stored elements:", f"{rw_beta.nnz:,}")


# ------------------------------------------------------------
# 18.8 检查对角线
#
# 自身与自身的真实距离为0；
# 在compute_pw_sparse_out_of_memory2输出中通常编码为1。
# ------------------------------------------------------------

diagonal = rw_beta.diagonal()

diagonal_one_n = int(np.count_nonzero(diagonal == 1))

diagonal_nonzero_n = int(np.count_nonzero(diagonal))

print(
    "Diagonal entries represented as 1:",
    f"{diagonal_one_n:,} / {n_clones:,}",
)

print(
    "Nonzero diagonal entries:",
    f"{diagonal_nonzero_n:,} / {n_clones:,}",
)

if diagonal_one_n != n_clones:
    unique_values, unique_counts = np.unique(
        diagonal,
        return_counts=True,
    )

    print(
        "WARNING: 对角线不全为1。Diagonal value distribution:",
        dict(zip(unique_values.tolist(), unique_counts.tolist())),
    )


# ------------------------------------------------------------
# 18.9 检查矩阵是否对称
#
# 患者内all-versus-all TCRdist矩阵应当对称。
# ------------------------------------------------------------

asymmetry_nnz = int((rw_beta != rw_beta.transpose()).nnz)

print(
    "Asymmetric stored positions:",
    f"{asymmetry_nnz:,}",
)

if asymmetry_nnz != 0:
    raise ValueError("患者内TCRdist稀疏矩阵不是对称矩阵")


# ------------------------------------------------------------
# 18.10 计算简单稀疏度指标
# ------------------------------------------------------------

matrix_positions = n_clones * n_clones

stored_density = rw_beta.nnz / matrix_positions if matrix_positions > 0 else 0.0

# 每行平均保存元素数，包括自身对角线
mean_stored_per_row = rw_beta.nnz / n_clones if n_clones > 0 else 0.0

# 有方向的非对角线存储元素数
off_diagonal_nnz = int(rw_beta.nnz - np.count_nonzero(diagonal))

# 对称矩阵中，每条无向边一般出现两次
approx_undirected_edges = off_diagonal_nnz // 2

print("Stored density:", f"{stored_density:.8%}")
print(
    "Mean stored elements per row:",
    f"{mean_stored_per_row:.2f}",
)
print(
    "Approximate undirected off-diagonal edges:",
    f"{approx_undirected_edges:,}",
)


# ------------------------------------------------------------
# 18.11 保存最终稀疏距离矩阵
# ------------------------------------------------------------

sparse.save_npz(
    sparse_matrix_file,
    rw_beta,
    compressed=True,
)

print("")
print(
    "Sparse beta distance matrix saved to:",
    sparse_matrix_file,
)


# ------------------------------------------------------------
# 18.12 保存运行参数
# ------------------------------------------------------------

parameters = {
    "libraryid": "MGI260115R01-10",
    "subject": str(subject),
    "n_clonotypes": int(n_clones),
    "distance_definition": ("default full beta-chain TCRdist"),
    "max_distance": int(MAX_DISTANCE),
    "row_size": int(ROW_SIZE),
    "pm_processes": int(PM_PROCESSES),
    "tr_cpus": int(tr.cpus),
    "reassemble": True,
    "cleanup": True,
    "assign": True,
    "self_distance_zero_encoding": 1,
}

with parameter_file.open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        parameters,
        handle,
        indent=2,
        ensure_ascii=False,
    )


# ------------------------------------------------------------
# 18.13 保存计算摘要
# ------------------------------------------------------------

summary_df = pd.DataFrame(
    [
        {
            "libraryid": "MGI260115R01-10",
            "subject": subject,
            "n_clonotypes": n_clones,
            "matrix_rows": rw_beta.shape[0],
            "matrix_columns": rw_beta.shape[1],
            "diagonal_entries_equal_1": diagonal_one_n,
            "stored_elements_nnz": rw_beta.nnz,
            "minimum_stored_value": stored_min,
            "maximum_stored_value": stored_max,
            "stored_density": stored_density,
            "mean_stored_elements_per_row": (mean_stored_per_row),
            "approx_undirected_edges": (approx_undirected_edges),
            "asymmetry_nnz": asymmetry_nnz,
            "max_distance": MAX_DISTANCE,
            "row_size": ROW_SIZE,
            "pm_processes": PM_PROCESSES,
            "elapsed_seconds": elapsed_seconds,
        }
    ]
)

summary_df.to_csv(
    summary_file,
    index=False,
)


# ------------------------------------------------------------
# 18.14 确认TCRrep中也已赋值
#
# compute_pw_sparse_out_of_memory2(assign=True)
# 会把重组后的beta稀疏矩阵赋给tr.pw_beta。
# ------------------------------------------------------------

if not hasattr(tr, "pw_beta"):
    raise AttributeError("assign=True后tr中没有pw_beta属性")

if tr.pw_beta.shape != rw_beta.shape:
    raise ValueError("tr.pw_beta与返回矩阵维度不一致")


# ------------------------------------------------------------
# 18.15 最终汇总
# ------------------------------------------------------------

print("")
print("=" * 70)
print("Sparse TCRdist calculation finished")
print("=" * 70)
print("Subject:", subject)
print("Clonotypes:", f"{n_clones:,}")
print("Matrix shape:", rw_beta.shape)
print("Stored elements:", f"{rw_beta.nnz:,}")
print("Stored density:", f"{stored_density:.8%}")
print(
    "Approximate undirected edges:",
    f"{approx_undirected_edges:,}",
)
print(
    "Elapsed seconds:",
    f"{elapsed_seconds:.2f}",
)
print("Sparse matrix:", sparse_matrix_file)
print("Parameter file:", parameter_file)
print("Summary file:", summary_file)
print("=" * 70)

# ============================================================
# 19. 单样本不同radius下的网络聚类与多样性计算
#
# 当前暂定聚类规则：
# 1. 每条clonotype是一个节点
# 2. 两条clonotype的TCRdist <= radius时建立无向边
# 3. 每个connected component定义为一个cluster
# 4. 没有邻居的clonotype作为singleton cluster保留
#
# 注意：
# 这一步用于跑通单样本流程和观察不同radius的网络结构。
# 暂不据此确定最终统一radius。
# ============================================================

import gc
import math
import time

from scipy.sparse.csgraph import connected_components


# ------------------------------------------------------------
# 19.1 设置候选radius
#
# 所有radius必须 <= MAX_DISTANCE
# ------------------------------------------------------------

RADIUS_LIST = [
    8,
    12,
    16,
    20,
    24,
    28,
    32,
    36,
    40,
    44,
    48,
]

if not RADIUS_LIST:
    raise ValueError("RADIUS_LIST不能为空")

if min(RADIUS_LIST) < 1:
    raise ValueError("radius必须为正整数")

if max(RADIUS_LIST) > MAX_DISTANCE:
    raise ValueError(
        f"候选radius最大值为{max(RADIUS_LIST)}，超过MAX_DISTANCE={MAX_DISTANCE}"
    )

if len(set(RADIUS_LIST)) != len(RADIUS_LIST):
    raise ValueError("RADIUS_LIST中存在重复值")

RADIUS_LIST = sorted(RADIUS_LIST)


# ------------------------------------------------------------
# 19.2 生成样本名称
# ------------------------------------------------------------

input_suffix = "_TRB_tcrdist_input.csv"

if input_file.name.endswith(input_suffix):
    sample_name = input_file.name[: -len(input_suffix)]
else:
    sample_name = input_file.stem


# ------------------------------------------------------------
# 19.3 创建radius结果目录
# ------------------------------------------------------------

radius_output_dir = output_dir / "radius_test"

radius_output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

radius_summary_file = radius_output_dir / f"{sample_name}_radius_test_summary.csv"


# ------------------------------------------------------------
# 19.4 检查距离矩阵和clone信息是否一致
# ------------------------------------------------------------

if rw_beta.shape[0] != rw_beta.shape[1]:
    raise ValueError(f"距离矩阵不是方阵: {rw_beta.shape}")

if len(clone_index_df) != rw_beta.shape[0]:
    raise ValueError(
        "clone_index_df行数与距离矩阵维度不一致: "
        f"clone_index_df={len(clone_index_df):,}, "
        f"matrix={rw_beta.shape[0]:,}"
    )

if "count" not in clone_index_df.columns:
    raise ValueError("clone_index_df中缺少count列")

if "tcrdist_row_index" not in clone_index_df.columns:
    raise ValueError("clone_index_df中缺少tcrdist_row_index列")

expected_row_index = np.arange(
    len(clone_index_df),
    dtype=np.int64,
)

actual_row_index = clone_index_df["tcrdist_row_index"].to_numpy(dtype=np.int64)

if not np.array_equal(
    expected_row_index,
    actual_row_index,
):
    raise ValueError("tcrdist_row_index不是从0开始的连续行号，无法安全映射距离矩阵")


# ------------------------------------------------------------
# 19.5 准备clone abundance
#
# 后续cluster丰度：
# 同一cluster中所有clonotype的count之和
# ------------------------------------------------------------

clone_counts = pd.to_numeric(
    clone_index_df["count"],
    errors="raise",
).to_numpy(dtype=np.int64)

if np.any(clone_counts <= 0):
    raise ValueError("clone_index_df中发现count <= 0")

n_nodes = len(clone_index_df)
total_clone_reads = int(clone_counts.sum())

print("")
print("=" * 70)
print("Starting radius-based clustering")
print("=" * 70)
print("Sample:", sample_name)
print("Subject:", subject)
print("Number of clonotypes:", f"{n_nodes:,}")
print("Total retained clone counts:", f"{total_clone_reads:,}")
print("Candidate radii:", RADIUS_LIST)
print("Clustering rule: undirected connected components")
print("Output directory:", radius_output_dir)
print("=" * 70)


# ------------------------------------------------------------
# 19.6 存储每个radius的汇总结果
# ------------------------------------------------------------

radius_summary_rows = []


# ------------------------------------------------------------
# 19.7 逐个radius建立网络并划分cluster
# ------------------------------------------------------------

for radius in RADIUS_LIST:
    radius_start_time = time.time()

    print("")
    print("-" * 70)
    print(f"Processing radius = {radius}")
    print("-" * 70)

    # --------------------------------------------------------
    # A. 从rw_beta构建二值邻接矩阵
    #
    # rw_beta中已经只保存distance <= MAX_DISTANCE的元素。
    # 此处进一步保留distance <= 当前radius的元素。
    # --------------------------------------------------------

    adjacency = rw_beta.copy().tocsr()

    # 满足radius条件的元素设为1；
    # 不满足条件的元素设为0，随后从稀疏矩阵删除。
    adjacency.data = (adjacency.data <= radius).astype(np.int8)

    adjacency.eliminate_zeros()

    # 删除自身连接。
    # 原始对角线为1，但自身不应作为网络边。
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    # 确保矩阵为无向对称邻接矩阵
    adjacency = adjacency.maximum(adjacency.transpose()).tocsr()

    adjacency.sort_indices()

    # --------------------------------------------------------
    # B. 邻接矩阵质量检查
    # --------------------------------------------------------

    diagonal_nonzero = int(np.count_nonzero(adjacency.diagonal()))

    if diagonal_nonzero != 0:
        raise ValueError(
            f"radius={radius}的邻接矩阵仍有{diagonal_nonzero}个非零对角线元素"
        )

    adjacency_asymmetry = int((adjacency != adjacency.transpose()).nnz)

    if adjacency_asymmetry != 0:
        raise ValueError(
            f"radius={radius}的邻接矩阵不对称，不对称位置数={adjacency_asymmetry:,}"
        )

    if adjacency.nnz % 2 != 0:
        raise ValueError(f"radius={radius}的无向邻接矩阵非对角线元素数不是偶数")

    # --------------------------------------------------------
    # C. 网络基本指标
    # --------------------------------------------------------

    node_degree = np.diff(adjacency.indptr).astype(np.int64)

    undirected_edges = int(adjacency.nnz // 2)

    isolated_clone_count = int(np.count_nonzero(node_degree == 0))

    connected_clone_count = int(n_nodes - isolated_clone_count)

    isolated_clone_fraction = isolated_clone_count / n_nodes if n_nodes > 0 else np.nan

    connected_clone_fraction = (
        connected_clone_count / n_nodes if n_nodes > 0 else np.nan
    )

    mean_degree = float(node_degree.mean()) if n_nodes > 0 else np.nan

    median_degree = float(np.median(node_degree)) if n_nodes > 0 else np.nan

    max_degree = int(node_degree.max()) if n_nodes > 0 else 0

    possible_edges = n_nodes * (n_nodes - 1) / 2

    network_density = (
        undirected_edges / possible_edges if possible_edges > 0 else np.nan
    )

    # --------------------------------------------------------
    # D. 连通分量聚类
    #
    # labels长度等于clonotype数量；
    # labels[i]表示第i个clonotype所属component。
    # --------------------------------------------------------

    n_components, labels = connected_components(
        csgraph=adjacency,
        directed=False,
        return_labels=True,
    )

    labels = labels.astype(np.int64)

    if len(labels) != n_nodes:
        raise ValueError(f"radius={radius}返回的cluster label数量错误")

    if n_components < 1:
        raise ValueError(f"radius={radius}没有产生任何cluster")

    # --------------------------------------------------------
    # E. 统计每个cluster包含的clone数量
    # --------------------------------------------------------

    cluster_clone_counts = np.bincount(
        labels,
        minlength=n_components,
    ).astype(np.int64)

    if int(cluster_clone_counts.sum()) != n_nodes:
        raise ValueError(f"radius={radius}的cluster clone总数与输入clonotype数量不一致")

    # --------------------------------------------------------
    # F. 汇总每个cluster的read/count abundance
    # --------------------------------------------------------

    cluster_read_counts_float = np.bincount(
        labels,
        weights=clone_counts,
        minlength=n_components,
    )

    # np.bincount使用weights时返回float，
    # 但输入count为整数，因此四舍五入恢复int64。
    cluster_read_counts = np.rint(cluster_read_counts_float).astype(np.int64)

    if int(cluster_read_counts.sum()) != total_clone_reads:
        raise ValueError(f"radius={radius}的cluster count总和与原始count总和不一致")

    cluster_read_fractions = cluster_read_counts / total_clone_reads

    # --------------------------------------------------------
    # G. singleton与cluster结构
    # --------------------------------------------------------

    singleton_cluster_mask = cluster_clone_counts == 1

    singleton_cluster_count = int(singleton_cluster_mask.sum())

    non_singleton_cluster_count = int(n_components - singleton_cluster_count)

    # 对于无向简单网络：
    # singleton component数量应等于degree=0的节点数量。
    if singleton_cluster_count != isolated_clone_count:
        raise ValueError(
            f"radius={radius}: singleton cluster数量"
            f"({singleton_cluster_count:,})与孤立节点数量"
            f"({isolated_clone_count:,})不一致"
        )

    singleton_cluster_fraction = (
        singleton_cluster_count / n_components if n_components > 0 else np.nan
    )

    clones_in_non_singleton_clusters = int(
        cluster_clone_counts[cluster_clone_counts > 1].sum()
    )

    clones_in_non_singleton_fraction = (
        clones_in_non_singleton_clusters / n_nodes if n_nodes > 0 else np.nan
    )

    largest_cluster_clone_count = int(cluster_clone_counts.max())

    largest_cluster_clone_fraction = (
        largest_cluster_clone_count / n_nodes if n_nodes > 0 else np.nan
    )

    largest_cluster_read_count = int(cluster_read_counts.max())

    largest_cluster_read_fraction = (
        largest_cluster_read_count / total_clone_reads
        if total_clone_reads > 0
        else np.nan
    )

    # --------------------------------------------------------
    # H. cluster size分布
    # --------------------------------------------------------

    cluster_size_median = float(np.median(cluster_clone_counts))

    cluster_size_mean = float(cluster_clone_counts.mean())

    cluster_size_p90 = float(
        np.quantile(
            cluster_clone_counts,
            0.90,
        )
    )

    cluster_size_p95 = float(
        np.quantile(
            cluster_clone_counts,
            0.95,
        )
    )

    cluster_size_p99 = float(
        np.quantile(
            cluster_clone_counts,
            0.99,
        )
    )

    # --------------------------------------------------------
    # I. 基于cluster abundance计算多样性
    #
    # cluster richness = cluster数量
    #
    # Shannon使用cluster汇总后的count比例：
    # p_k = cluster_count_k / 所有cluster count总和
    #
    # Pielou evenness = Shannon / log(richness)
    # clonality = 1 - evenness
    # --------------------------------------------------------

    positive_cluster_fractions = cluster_read_fractions[cluster_read_fractions > 0]

    cluster_shannon = float(
        -np.sum(positive_cluster_fractions * np.log(positive_cluster_fractions))
    )

    cluster_richness = int(n_components)

    if cluster_richness > 1:
        cluster_pielou_evenness = float(cluster_shannon / math.log(cluster_richness))

        cluster_clonality = float(1.0 - cluster_pielou_evenness)
    else:
        cluster_pielou_evenness = np.nan
        cluster_clonality = np.nan

    effective_cluster_number = float(math.exp(cluster_shannon))

    # --------------------------------------------------------
    # J. 生成cluster级结果表
    # --------------------------------------------------------

    cluster_ids = np.arange(
        1,
        n_components + 1,
        dtype=np.int64,
    )

    cluster_result_df = pd.DataFrame(
        {
            "sample_id": sample_name,
            "subject": subject,
            "radius": radius,
            "cluster_id": cluster_ids,
            "clone_count": cluster_clone_counts,
            "read_count": cluster_read_counts,
            "read_fraction": cluster_read_fractions,
            "is_singleton": singleton_cluster_mask,
        }
    )

    # 按read abundance生成cluster rank
    cluster_result_df["cluster_rank_by_read_count"] = (
        cluster_result_df["read_count"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    cluster_result_df = cluster_result_df.sort_values(
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

    # --------------------------------------------------------
    # K. 生成clone到cluster的映射表
    # --------------------------------------------------------

    one_based_cluster_labels = labels + 1

    clone_cluster_df = clone_index_df.copy()

    clone_cluster_df.insert(
        1,
        "radius",
        radius,
    )

    clone_cluster_df.insert(
        2,
        "cluster_id",
        one_based_cluster_labels,
    )

    clone_cluster_df["cluster_clone_count"] = cluster_clone_counts[labels]

    clone_cluster_df["cluster_read_count"] = cluster_read_counts[labels]

    clone_cluster_df["cluster_read_fraction"] = cluster_read_fractions[labels]

    clone_cluster_df["node_degree"] = node_degree

    clone_cluster_df["is_singleton_cluster"] = cluster_clone_counts[labels] == 1

    # --------------------------------------------------------
    # L. 保存当前radius的结果
    # --------------------------------------------------------

    radius_label = f"{radius:02d}"

    cluster_result_file = radius_output_dir / (
        f"{sample_name}_radius{radius_label}_cluster_summary.csv"
    )

    clone_cluster_file = radius_output_dir / (
        f"{sample_name}_radius{radius_label}_clone_cluster_assignment.csv"
    )

    cluster_result_df.to_csv(
        cluster_result_file,
        index=False,
    )

    clone_cluster_df.to_csv(
        clone_cluster_file,
        index=False,
    )

    # --------------------------------------------------------
    # M. 保存当前radius汇总指标
    # --------------------------------------------------------

    radius_elapsed_seconds = time.time() - radius_start_time

    radius_summary_rows.append(
        {
            "sample_id": sample_name,
            "subject": subject,
            "radius": radius,
            "n_clonotypes": n_nodes,
            "total_clone_reads": total_clone_reads,
            "undirected_edges": undirected_edges,
            "network_density": network_density,
            "mean_node_degree": mean_degree,
            "median_node_degree": median_degree,
            "maximum_node_degree": max_degree,
            "isolated_clone_count": isolated_clone_count,
            "isolated_clone_fraction": isolated_clone_fraction,
            "connected_clone_count": connected_clone_count,
            "connected_clone_fraction": connected_clone_fraction,
            "cluster_richness": cluster_richness,
            "singleton_cluster_count": singleton_cluster_count,
            "singleton_cluster_fraction": singleton_cluster_fraction,
            "non_singleton_cluster_count": (non_singleton_cluster_count),
            "clones_in_non_singleton_clusters": (clones_in_non_singleton_clusters),
            "clones_in_non_singleton_fraction": (clones_in_non_singleton_fraction),
            "cluster_size_mean": cluster_size_mean,
            "cluster_size_median": cluster_size_median,
            "cluster_size_p90": cluster_size_p90,
            "cluster_size_p95": cluster_size_p95,
            "cluster_size_p99": cluster_size_p99,
            "largest_cluster_clone_count": (largest_cluster_clone_count),
            "largest_cluster_clone_fraction": (largest_cluster_clone_fraction),
            "largest_cluster_read_count": (largest_cluster_read_count),
            "largest_cluster_read_fraction": (largest_cluster_read_fraction),
            "cluster_shannon": cluster_shannon,
            "cluster_pielou_evenness": (cluster_pielou_evenness),
            "cluster_clonality": cluster_clonality,
            "effective_cluster_number": (effective_cluster_number),
            "radius_elapsed_seconds": (radius_elapsed_seconds),
        }
    )

    # --------------------------------------------------------
    # N. 打印当前radius结果
    # --------------------------------------------------------

    print(
        f"radius={radius}: "
        f"edges={undirected_edges:,}, "
        f"clusters={cluster_richness:,}, "
        f"singletons={singleton_cluster_count:,} "
        f"({isolated_clone_fraction:.2%}), "
        f"largest_cluster={largest_cluster_clone_count:,} "
        f"({largest_cluster_clone_fraction:.2%}), "
        f"Shannon={cluster_shannon:.6f}, "
        f"Evenness={cluster_pielou_evenness:.6f}, "
        f"Clonality={cluster_clonality:.6f}"
    )

    print(
        "Cluster summary:",
        cluster_result_file,
    )

    print(
        "Clone-cluster assignment:",
        clone_cluster_file,
    )

    # 释放当前radius的临时对象
    del adjacency
    del cluster_result_df
    del clone_cluster_df
    del labels
    del node_degree

    gc.collect()


# ------------------------------------------------------------
# 19.8 汇总并保存所有radius结果
# ------------------------------------------------------------

radius_summary_df = (
    pd.DataFrame(radius_summary_rows).sort_values("radius").reset_index(drop=True)
)

radius_summary_df.to_csv(
    radius_summary_file,
    index=False,
)


# ------------------------------------------------------------
# 19.9 检查随radius变化的基本单调性
#
# radius增大时通常应满足：
# - edge数量不减少
# - cluster数量不增加
# - singleton数量不增加
# ------------------------------------------------------------

if not radius_summary_df["undirected_edges"].is_monotonic_increasing:
    print("WARNING: edge数量未随radius单调增加")

if not radius_summary_df["cluster_richness"].is_monotonic_decreasing:
    print("WARNING: cluster richness未随radius单调减少")

if not radius_summary_df["singleton_cluster_count"].is_monotonic_decreasing:
    print("WARNING: singleton数量未随radius单调减少")


# ------------------------------------------------------------
# 19.10 打印核心radius比较表
# ------------------------------------------------------------

display_columns = [
    "radius",
    "undirected_edges",
    "mean_node_degree",
    "cluster_richness",
    "isolated_clone_fraction",
    "non_singleton_cluster_count",
    "largest_cluster_clone_count",
    "largest_cluster_clone_fraction",
    "largest_cluster_read_fraction",
    "cluster_shannon",
    "cluster_pielou_evenness",
    "cluster_clonality",
]

print("")
print("=" * 100)
print("Radius test summary")
print("=" * 100)

print(radius_summary_df[display_columns].to_string(index=False))

print("")
print("Radius summary file:", radius_summary_file)
print("Radius output directory:", radius_output_dir)
print("=" * 100)
