#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TCRdist3全样本radius结果统计分析与可视化。

本脚本承接批处理脚本tcrdist3_radius_batch.py生成的结果：

    /data/users/chenhaisheng/RA-ILD/TRB/vdj/result/tcrdist_all_samples/
        <sample_id>/
            <sample_id>_radius_summary.csv

运行分为两个阶段：

阶段1：全体样本radius诊断（始终执行）
    - 汇总全部样本在radius=20,22,...,32下的结果
    - 完整性、数值范围和单调性QC
    - 网络结构和多样性指标的总体描述
    - 巨型连通分量风险分析
    - 样本规模相关性分析
    - batch和material描述性汇总
    - radius趋势、患者轨迹、热图和样本规模散点图

阶段2：固定主radius后的cohort比较（可选）
    - 默认SELECTED_RADIUS = None，不执行阶段2
    - 根据阶段1结果确定主radius后，将SELECTED_RADIUS改成整数，例如24
    - 比较RA与ILD/RA-ILD患者的多样性和网络指标
    - Mann-Whitney U、Cliff's delta和bootstrap 95% CI
    - HC3稳健标准误的多变量线性回归
    - 邻近及全部radius敏感性分析
    - 组间分布图、森林图、相关性热图和效应随radius变化图

设计原则：
    - 主radius的选择只依据无监督网络结构和稳定性结果。
    - 不依据RA与ILD的显著性或P值反向选择radius。
    - 阶段2只有在SELECTED_RADIUS明确设置后才运行。

Python兼容性：Python 3.8+
主要依赖：numpy、pandas、scipy、matplotlib
"""

from __future__ import annotations

import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# 1. 用户配置区
# ============================================================

# 前一个全量批处理脚本的根输出目录。
RESULT_ROOT = Path("/data/users/chenhaisheng/RA-ILD/TRB/vdj/result/tcrdist_all_samples")

# 项目元数据。默认使用已知的RA-ILD项目元数据路径。
METADATA_FILE = Path("/data/users/chenhaisheng/RA-ILD/metadata.csv")

# 本统计脚本的输出目录。
STATISTICS_OUTPUT_ROOT = Path(
    "/data/users/chenhaisheng/RA-ILD/TRB/vdj/result/tcrdist_statistics"
)

# 批处理时计算的全部候选radius。
EXPECTED_RADIUS_LIST = [20, 22, 24, 26, 28, 30, 32]

# 全体预期样本数。设置为None时不强制检查。
EXPECTED_SAMPLE_COUNT: Optional[int] = 174

# ------------------------------------------------------------
# 最重要的开关
# ------------------------------------------------------------

# 默认None：只运行阶段1，不运行固定radius后的cohort分析。
# 根据阶段1结果确定主radius后，例如改成：
# SELECTED_RADIUS = 24
SELECTED_RADIUS: Optional[int] = 24

# ------------------------------------------------------------
# 元数据列名
# ------------------------------------------------------------

METADATA_SAMPLE_COLUMN = "libraryid"
METADATA_SUBJECT_COLUMN = "patient"
COHORT_COLUMN = "cohort"
AGE_COLUMN = "age"
SEX_COLUMN = "sex"
BATCH_COLUMN = "batch"
MATERIAL_COLUMN = "material"

# cohort编码。用户当前元数据中为RA和ILD。
REFERENCE_COHORT = "RA"
TARGET_COHORT = "ILD"

# ------------------------------------------------------------
# 固定radius后的统计配置
# ------------------------------------------------------------

# 回归模型连续协变量。
# log10_n_clonotypes由脚本自动生成。
ADJUSTMENT_CONTINUOUS = [
    AGE_COLUMN,
    "log10_n_clonotypes",
]

# 回归模型分类协变量。
# 默认校正sex和batch。
# 若需要改为material，可改成：[SEX_COLUMN, MATERIAL_COLUMN]
# 不建议在batch和material高度重合时同时加入两者。
ADJUSTMENT_CATEGORICAL = [
    SEX_COLUMN,
    BATCH_COLUMN,
]

# Cliff's delta bootstrap次数。
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 20260721

# 图片设置。
FIGURE_DPI = 180
SAVE_PDF = True
SAVE_PNG = True

# 是否绘制所有患者轨迹。174条线可正常绘制。
DRAW_PATIENT_TRAJECTORIES = True

# 是否绘制每个radius下样本规模相关散点图。
DRAW_SAMPLE_SIZE_SCATTERS = True


# ============================================================
# 2. 分析指标定义
# ============================================================

# 前一个批处理脚本应当输出的必要列。
REQUIRED_RADIUS_COLUMNS = [
    "sample_id",
    "subject",
    "radius",
    "n_clonotypes",
    "total_clone_reads",
    "undirected_edges",
    "network_density",
    "mean_node_degree",
    "median_node_degree",
    "maximum_node_degree",
    "isolated_clone_count",
    "isolated_clone_fraction",
    "cluster_richness",
    "singleton_cluster_count",
    "singleton_cluster_fraction",
    "non_singleton_cluster_count",
    "clones_in_non_singleton_clusters",
    "clones_in_non_singleton_fraction",
    "cluster_size_mean",
    "cluster_size_median",
    "cluster_size_p90",
    "cluster_size_p95",
    "cluster_size_p99",
    "largest_cluster_clone_count",
    "largest_cluster_clone_fraction",
    "largest_cluster_read_count",
    "largest_cluster_read_fraction",
    "cluster_shannon",
    "cluster_pielou_evenness",
    "cluster_clonality",
    "effective_cluster_number",
]

# radius选择阶段重点观察的指标。
RADIUS_DIAGNOSTIC_METRICS = [
    "isolated_clone_fraction",
    "clones_in_non_singleton_fraction",
    "cluster_richness_ratio",
    "mean_node_degree",
    "largest_cluster_clone_fraction",
    "largest_cluster_read_fraction",
    "cluster_shannon",
    "cluster_pielou_evenness",
    "cluster_clonality",
    "effective_cluster_number",
]

# 绘制所有患者轨迹的重点指标。
TRAJECTORY_METRICS = [
    "isolated_clone_fraction",
    "largest_cluster_clone_fraction",
    "largest_cluster_read_fraction",
    "cluster_shannon",
]

# 热图指标。
HEATMAP_METRICS = [
    "isolated_clone_fraction",
    "largest_cluster_clone_fraction",
    "cluster_shannon",
]

# 样本规模相关性重点指标。
SAMPLE_SIZE_CORRELATION_METRICS = [
    "isolated_clone_fraction",
    "clones_in_non_singleton_fraction",
    "cluster_richness_ratio",
    "largest_cluster_clone_fraction",
    "largest_cluster_read_fraction",
    "cluster_shannon",
    "cluster_pielou_evenness",
]

# 固定radius后进行正式统计检验的主要指标。
PRIMARY_COHORT_METRICS = [
    "cluster_shannon",
    "cluster_pielou_evenness",
    "effective_cluster_number",
    "cluster_richness_ratio",
    "clones_in_non_singleton_fraction",
    "largest_cluster_read_fraction",
]

# 补充网络结构指标。
SECONDARY_COHORT_METRICS = [
    "mean_node_degree",
    "isolated_clone_fraction",
    "non_singleton_cluster_count",
    "largest_cluster_clone_fraction",
    "cluster_size_p95",
    "cluster_size_p99",
]

ALL_COHORT_METRICS = list(
    dict.fromkeys(PRIMARY_COHORT_METRICS + SECONDARY_COHORT_METRICS)
)

# 固定radius后额外绘制的四个RA/ILD箱线图指标。
# cluster_richness表示TCRdist网络中的cluster总数（连通分量数），
# 不是原始CDR3 clonotype数量。
COHORT_BOXPLOT_METRICS = [
    "cluster_richness",
    "cluster_shannon",
    "cluster_clonality",
    "cluster_pielou_evenness",
]

# 0到1之间的比例指标，回归前做logit变换。
FRACTION_METRICS = {
    "isolated_clone_fraction",
    "clones_in_non_singleton_fraction",
    "cluster_richness_ratio",
    "largest_cluster_clone_fraction",
    "largest_cluster_read_fraction",
    "cluster_pielou_evenness",
    "cluster_clonality",
    "singleton_cluster_fraction",
    "network_density",
}

# 正值且常见右偏指标，回归前做log1p变换。
LOG1P_METRICS = {
    "effective_cluster_number",
    "mean_node_degree",
    "non_singleton_cluster_count",
    "cluster_size_p95",
    "cluster_size_p99",
    "cluster_richness",
    "undirected_edges",
}

# 指标显示名称。
METRIC_LABELS = {
    "isolated_clone_fraction": "Isolated clonotype fraction",
    "clones_in_non_singleton_fraction": "Clonotypes in non-singleton clusters",
    "cluster_richness": "TCRdist cluster richness (cluster count)",
    "cluster_richness_ratio": "Cluster richness / clonotype count",
    "mean_node_degree": "Mean node degree",
    "largest_cluster_clone_fraction": "Largest cluster clone fraction",
    "largest_cluster_read_fraction": "Largest cluster read fraction",
    "cluster_shannon": "Cluster Shannon diversity",
    "cluster_pielou_evenness": "Cluster Pielou evenness",
    "cluster_clonality": "Cluster clonality",
    "effective_cluster_number": "Effective cluster number",
    "non_singleton_cluster_count": "Non-singleton cluster count",
    "cluster_size_p95": "Cluster size 95th percentile",
    "cluster_size_p99": "Cluster size 99th percentile",
}


# ============================================================
# 3. 输出目录结构
# ============================================================

PHASE1_DIR = STATISTICS_OUTPUT_ROOT / "phase1_radius_selection"
PHASE1_TABLE_DIR = PHASE1_DIR / "tables"
PHASE1_FIGURE_DIR = PHASE1_DIR / "figures"

PHASE2_DIR = STATISTICS_OUTPUT_ROOT / "phase2_selected_radius"
PHASE2_TABLE_DIR = PHASE2_DIR / "tables"
PHASE2_FIGURE_DIR = PHASE2_DIR / "figures"


# ============================================================
# 4. 工具函数
# ============================================================


def ensure_directories() -> None:
    """创建输出目录。"""

    for directory in [
        STATISTICS_OUTPUT_ROOT,
        PHASE1_TABLE_DIR,
        PHASE1_FIGURE_DIR,
        PHASE2_TABLE_DIR,
        PHASE2_FIGURE_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def safe_filename(text: str) -> str:
    """将任意文本转换为适合文件名的形式。"""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def metric_label(metric: str) -> str:
    """返回指标显示名称。"""

    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    """将单张图同时保存为PNG和PDF。"""

    fig.tight_layout()

    if SAVE_PNG:
        fig.savefig(
            base_path.with_suffix(".png"),
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )

    if SAVE_PDF:
        fig.savefig(
            base_path.with_suffix(".pdf"),
            bbox_inches="tight",
        )

    plt.close(fig)


def write_json(data: Dict, file_path: Path) -> None:
    """保存JSON。"""

    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=str)


def bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR校正，保留NaN。"""

    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)

    if not valid.any():
        return adjusted

    p_valid = p[valid]
    order = np.argsort(p_valid)
    ranked = p_valid[order]
    n = len(ranked)

    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    reordered = np.empty_like(q)
    reordered[order] = q
    adjusted[valid] = reordered

    return adjusted


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    计算Cliff's delta：P(x>y)-P(x<y)。

    本脚本固定传入TARGET cohort为x、REFERENCE cohort为y，
    正值表示TARGET cohort数值更高。
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    comparison = x[:, None] - y[None, :]
    greater = np.count_nonzero(comparison > 0)
    lower = np.count_nonzero(comparison < 0)

    return float((greater - lower) / comparison.size)


def bootstrap_cliffs_delta_ci(
    target_values: np.ndarray,
    reference_values: np.ndarray,
    iterations: int,
    seed: int,
) -> Tuple[float, float]:
    """分组内bootstrap获得Cliff's delta 95% CI。"""

    target_values = np.asarray(target_values, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)
    target_values = target_values[np.isfinite(target_values)]
    reference_values = reference_values[np.isfinite(reference_values)]

    if len(target_values) == 0 or len(reference_values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)

    for index in range(iterations):
        target_sample = rng.choice(
            target_values,
            size=len(target_values),
            replace=True,
        )
        reference_sample = rng.choice(
            reference_values,
            size=len(reference_values),
            replace=True,
        )
        estimates[index] = cliffs_delta(target_sample, reference_sample)

    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def mann_whitney_p(target_values: np.ndarray, reference_values: np.ndarray) -> float:
    """兼容不同SciPy版本的双侧Mann-Whitney U检验。"""

    target_values = np.asarray(target_values, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)
    target_values = target_values[np.isfinite(target_values)]
    reference_values = reference_values[np.isfinite(reference_values)]

    if len(target_values) == 0 or len(reference_values) == 0:
        return np.nan

    try:
        result = stats.mannwhitneyu(
            target_values,
            reference_values,
            alternative="two-sided",
            method="asymptotic",
        )
    except TypeError:
        result = stats.mannwhitneyu(
            target_values,
            reference_values,
            alternative="two-sided",
        )

    return float(result.pvalue)


def transform_metric_for_regression(
    values: pd.Series,
    metric: str,
) -> Tuple[pd.Series, str]:
    """
    根据指标类型完成变换，并标准化为均值0、标准差1。

    标准化后，cohort回归系数可解释为调整协变量后，
    TARGET cohort相对于REFERENCE cohort相差多少个标准差。
    """

    numeric = pd.to_numeric(values, errors="coerce").astype(float)

    if metric in FRACTION_METRICS:
        epsilon = 1e-6
        clipped = numeric.clip(epsilon, 1.0 - epsilon)
        transformed = np.log(clipped / (1.0 - clipped))
        transform_name = "logit_then_zscore"
    elif metric in LOG1P_METRICS:
        transformed = np.log1p(numeric.clip(lower=0.0))
        transform_name = "log1p_then_zscore"
    else:
        transformed = numeric
        transform_name = "identity_then_zscore"

    transformed = pd.Series(transformed, index=values.index, dtype=float)
    mean_value = transformed.mean(skipna=True)
    sd_value = transformed.std(skipna=True, ddof=1)

    if not np.isfinite(sd_value) or sd_value <= 0:
        standardized = pd.Series(np.nan, index=values.index, dtype=float)
    else:
        standardized = (transformed - mean_value) / sd_value

    return standardized, transform_name


@dataclass
class HC3Result:
    coefficient: float
    standard_error: float
    confidence_low: float
    confidence_high: float
    p_value: float
    n_observations: int
    n_parameters: int
    residual_df: int


def fit_ols_hc3(
    y: np.ndarray,
    x: np.ndarray,
    coefficient_index: int,
) -> HC3Result:
    """
    使用numpy实现OLS和HC3稳健协方差。

    这样不依赖statsmodels，适合用户现有conda环境。
    """

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if y.ndim != 1:
        raise ValueError("y必须是一维数组")
    if x.ndim != 2:
        raise ValueError("x必须是二维矩阵")
    if len(y) != x.shape[0]:
        raise ValueError("y长度和x行数不一致")

    n_observations, n_parameters = x.shape
    residual_df = n_observations - n_parameters

    if residual_df <= 0:
        raise ValueError("回归样本数不足以估计模型")

    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residuals = y - x @ beta

    # Hat matrix对角线，无需构建完整n×n矩阵。
    leverage = np.sum((x @ xtx_inv) * x, axis=1)
    leverage = np.clip(leverage, 0.0, 1.0 - 1e-10)

    adjusted_residual_sq = (residuals / (1.0 - leverage)) ** 2
    meat = x.T @ (x * adjusted_residual_sq[:, None])
    covariance = xtx_inv @ meat @ xtx_inv

    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))

    coefficient = float(beta[coefficient_index])
    standard_error = float(standard_errors[coefficient_index])

    if not np.isfinite(standard_error) or standard_error <= 0:
        p_value = np.nan
        confidence_low = np.nan
        confidence_high = np.nan
    else:
        t_statistic = coefficient / standard_error
        p_value = float(2.0 * stats.t.sf(abs(t_statistic), df=residual_df))
        critical_value = float(stats.t.ppf(0.975, df=residual_df))
        confidence_low = coefficient - critical_value * standard_error
        confidence_high = coefficient + critical_value * standard_error

    return HC3Result(
        coefficient=coefficient,
        standard_error=standard_error,
        confidence_low=float(confidence_low),
        confidence_high=float(confidence_high),
        p_value=p_value,
        n_observations=n_observations,
        n_parameters=n_parameters,
        residual_df=residual_df,
    )


def build_adjusted_design_matrix(
    analysis_df: pd.DataFrame,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], str, pd.Index]:
    """建立固定radius多变量回归设计矩阵。"""

    required_columns = [
        metric,
        COHORT_COLUMN,
        *ADJUSTMENT_CONTINUOUS,
        *ADJUSTMENT_CATEGORICAL,
    ]
    required_columns = list(dict.fromkeys(required_columns))

    missing = [
        column for column in required_columns if column not in analysis_df.columns
    ]
    if missing:
        raise ValueError("调整模型缺少列: " + ", ".join(missing))

    model_df = analysis_df[required_columns].copy()
    model_df[COHORT_COLUMN] = model_df[COHORT_COLUMN].astype(str).str.strip()

    model_df = model_df[
        model_df[COHORT_COLUMN].isin([REFERENCE_COHORT, TARGET_COHORT])
    ].copy()

    transformed_y, transform_name = transform_metric_for_regression(
        model_df[metric], metric
    )
    model_df["__outcome__"] = transformed_y

    # cohort效应列，TARGET=1、REFERENCE=0。
    model_df["__cohort_target__"] = (model_df[COHORT_COLUMN] == TARGET_COHORT).astype(
        float
    )

    continuous_parts = []
    continuous_names = []

    for column in ADJUSTMENT_CONTINUOUS:
        numeric = pd.to_numeric(model_df[column], errors="coerce")
        continuous_parts.append(numeric.rename(column))
        continuous_names.append(column)

    categorical_source = model_df[ADJUSTMENT_CATEGORICAL].copy()
    for column in ADJUSTMENT_CATEGORICAL:
        categorical_source[column] = categorical_source[column].astype(str).str.strip()
        categorical_source.loc[model_df[column].isna(), column] = np.nan

    categorical_dummies = pd.get_dummies(
        categorical_source,
        prefix=ADJUSTMENT_CATEGORICAL,
        drop_first=True,
        dtype=float,
    )

    design_df = pd.DataFrame(
        {
            "Intercept": 1.0,
            "cohort_target": model_df["__cohort_target__"],
        },
        index=model_df.index,
    )

    for part in continuous_parts:
        design_df[part.name] = part

    if not categorical_dummies.empty:
        design_df = pd.concat([design_df, categorical_dummies], axis=1)

    combined = pd.concat(
        [model_df[["__outcome__"]], design_df],
        axis=1,
    ).dropna(axis=0, how="any")

    if combined.empty:
        raise ValueError(f"指标{metric}没有可用于调整模型的完整样本")

    y = combined["__outcome__"].to_numpy(dtype=float)
    x_df = combined.drop(columns="__outcome__")

    # 删除零方差协变量，但始终保留截距和cohort效应。
    keep_columns = []
    for column in x_df.columns:
        if column in {"Intercept", "cohort_target"}:
            keep_columns.append(column)
        elif x_df[column].nunique(dropna=True) > 1:
            keep_columns.append(column)

    x_df = x_df[keep_columns]

    if x_df["cohort_target"].nunique() != 2:
        raise ValueError(f"指标{metric}模型中没有同时包含两个cohort")

    x = x_df.to_numpy(dtype=float)
    coefficient_index = x_df.columns.get_loc("cohort_target")

    return y, x, list(x_df.columns), transform_name, combined.index


# ============================================================
# 5. 读取和合并数据
# ============================================================


def discover_radius_summary_files() -> List[Path]:
    """发现全部样本的radius summary文件。"""

    if not RESULT_ROOT.is_dir():
        raise FileNotFoundError(f"全量计算结果目录不存在: {RESULT_ROOT}")

    files = sorted(RESULT_ROOT.glob("*/*_radius_summary.csv"))

    if not files:
        raise FileNotFoundError(f"没有在{RESULT_ROOT}下发现*/*_radius_summary.csv")

    return files


def load_all_radius_results(files: Sequence[Path]) -> pd.DataFrame:
    """读取并合并所有样本的radius summary。"""

    frames: List[pd.DataFrame] = []

    for file_path in files:
        frame = pd.read_csv(file_path)

        missing = [
            column for column in REQUIRED_RADIUS_COLUMNS if column not in frame.columns
        ]
        if missing:
            raise ValueError(f"文件{file_path}缺少必要列: {', '.join(missing)}")

        frame = frame[REQUIRED_RADIUS_COLUMNS].copy()
        frame["source_file"] = str(file_path)
        frames.append(frame)

    all_radius = pd.concat(frames, ignore_index=True)

    all_radius["sample_id"] = all_radius["sample_id"].astype(str).str.strip()
    all_radius["subject"] = all_radius["subject"].astype(str).str.strip()
    all_radius["radius"] = pd.to_numeric(all_radius["radius"], errors="raise").astype(
        int
    )

    numeric_columns = [
        column
        for column in REQUIRED_RADIUS_COLUMNS
        if column not in {"sample_id", "subject", "radius"}
    ]

    for column in numeric_columns:
        all_radius[column] = pd.to_numeric(all_radius[column], errors="coerce")

    # 派生指标。
    all_radius["cluster_richness_ratio"] = (
        all_radius["cluster_richness"] / all_radius["n_clonotypes"]
    )
    all_radius["clustered_clone_fraction"] = all_radius[
        "clones_in_non_singleton_fraction"
    ]
    all_radius["log10_n_clonotypes"] = np.log10(all_radius["n_clonotypes"])
    all_radius["log10_total_clone_reads"] = np.log10(all_radius["total_clone_reads"])

    return all_radius


def load_metadata() -> pd.DataFrame:
    """读取并标准化元数据。"""

    if not METADATA_FILE.is_file():
        raise FileNotFoundError(f"元数据文件不存在: {METADATA_FILE}")

    metadata = pd.read_csv(METADATA_FILE)

    required = [
        METADATA_SAMPLE_COLUMN,
        COHORT_COLUMN,
        AGE_COLUMN,
        SEX_COLUMN,
        BATCH_COLUMN,
        MATERIAL_COLUMN,
    ]
    missing = [column for column in required if column not in metadata.columns]

    if missing:
        raise ValueError("元数据缺少必要列: " + ", ".join(missing))

    metadata = metadata.copy()
    metadata[METADATA_SAMPLE_COLUMN] = (
        metadata[METADATA_SAMPLE_COLUMN].astype(str).str.strip()
    )

    for column in [COHORT_COLUMN, SEX_COLUMN, BATCH_COLUMN, MATERIAL_COLUMN]:
        metadata[column] = metadata[column].astype(str).str.strip()

    metadata[AGE_COLUMN] = pd.to_numeric(metadata[AGE_COLUMN], errors="coerce")

    if metadata[METADATA_SAMPLE_COLUMN].duplicated().any():
        duplicated = metadata.loc[
            metadata[METADATA_SAMPLE_COLUMN].duplicated(keep=False),
            METADATA_SAMPLE_COLUMN,
        ].tolist()
        raise ValueError(
            "元数据libraryid/sample_id不唯一: " + ", ".join(map(str, duplicated[:20]))
        )

    return metadata


def merge_radius_with_metadata(
    all_radius: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """按sample_id与metadata中的libraryid合并。"""

    metadata_for_merge = metadata.rename(
        columns={METADATA_SAMPLE_COLUMN: "sample_id"}
    ).copy()

    merged = all_radius.merge(
        metadata_for_merge,
        on="sample_id",
        how="left",
        validate="many_to_one",
        indicator=True,
        suffixes=("", "_metadata"),
    )

    missing_metadata_samples = sorted(
        merged.loc[merged["_merge"] != "both", "sample_id"].unique().tolist()
    )

    if missing_metadata_samples:
        raise ValueError(
            "以下结果样本未匹配到metadata: " + ", ".join(missing_metadata_samples[:30])
        )

    merged = merged.drop(columns="_merge")

    # 若元数据含patient列，检查subject一致性。
    if METADATA_SUBJECT_COLUMN in merged.columns:
        result_subject = merged["subject"].astype(str).str.strip()
        metadata_subject = merged[METADATA_SUBJECT_COLUMN].astype(str).str.strip()
        inconsistent = merged[METADATA_SUBJECT_COLUMN].notna() & (
            result_subject != metadata_subject
        )

        if inconsistent.any():
            examples = (
                merged.loc[
                    inconsistent,
                    ["sample_id", "subject", METADATA_SUBJECT_COLUMN],
                ]
                .drop_duplicates()
                .head(20)
            )
            raise ValueError(
                "部分样本的结果subject与metadata patient不一致:\n"
                + examples.to_string(index=False)
            )

    return merged


# ============================================================
# 6. 阶段1：QC和无监督radius诊断
# ============================================================


def run_radius_qc(all_radius: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """完成样本级和全局radius结果QC。"""

    sample_ids = sorted(all_radius["sample_id"].unique().tolist())
    expected_radius_set = set(EXPECTED_RADIUS_LIST)
    qc_rows = []

    duplicate_pairs = all_radius.duplicated(subset=["sample_id", "radius"], keep=False)
    duplicate_pair_set = set(
        map(tuple, all_radius.loc[duplicate_pairs, ["sample_id", "radius"]].to_numpy())
    )

    fraction_columns = [
        "network_density",
        "isolated_clone_fraction",
        "singleton_cluster_fraction",
        "clones_in_non_singleton_fraction",
        "largest_cluster_clone_fraction",
        "largest_cluster_read_fraction",
        "cluster_pielou_evenness",
        "cluster_clonality",
        "cluster_richness_ratio",
    ]

    for sample_id in sample_ids:
        sample_df = all_radius[all_radius["sample_id"] == sample_id].copy()
        sample_df = sample_df.sort_values("radius")
        failures: List[str] = []

        observed_radius_set = set(sample_df["radius"].astype(int).tolist())
        if observed_radius_set != expected_radius_set:
            failures.append(f"radius集合错误:{sorted(observed_radius_set)}")

        if len(sample_df) != len(EXPECTED_RADIUS_LIST):
            failures.append(f"行数错误:{len(sample_df)}")

        if any(pair[0] == sample_id for pair in duplicate_pair_set):
            failures.append("sample_id+radius重复")

        numeric_missing = int(sample_df[REQUIRED_RADIUS_COLUMNS].isna().sum().sum())
        if numeric_missing > 0:
            failures.append(f"必要字段缺失:{numeric_missing}")

        for column in fraction_columns:
            invalid = ~sample_df[column].between(0.0, 1.0, inclusive="both")
            invalid = invalid & sample_df[column].notna()
            if invalid.any():
                failures.append(f"{column}超出[0,1]")

        if (sample_df["cluster_richness"] > sample_df["n_clonotypes"]).any():
            failures.append("cluster_richness>n_clonotypes")

        if (sample_df["singleton_cluster_count"] > sample_df["cluster_richness"]).any():
            failures.append("singleton_cluster_count>cluster_richness")

        if (sample_df["largest_cluster_clone_count"] > sample_df["n_clonotypes"]).any():
            failures.append("largest_cluster_clone_count>n_clonotypes")

        complement_error = np.abs(
            sample_df["cluster_pielou_evenness"] + sample_df["cluster_clonality"] - 1.0
        )
        if (complement_error.dropna() > 1e-8).any():
            failures.append("evenness+clonality不等于1")

        if not sample_df["undirected_edges"].is_monotonic_increasing:
            failures.append("undirected_edges非单调增加")

        if not sample_df["mean_node_degree"].is_monotonic_increasing:
            failures.append("mean_node_degree非单调增加")

        if not sample_df["cluster_richness"].is_monotonic_decreasing:
            failures.append("cluster_richness非单调减少")

        if not sample_df["isolated_clone_fraction"].is_monotonic_decreasing:
            failures.append("isolated_clone_fraction非单调减少")

        if not sample_df["singleton_cluster_count"].is_monotonic_decreasing:
            failures.append("singleton_cluster_count非单调减少")

        if not sample_df["largest_cluster_clone_fraction"].is_monotonic_increasing:
            failures.append("largest_cluster_clone_fraction非单调增加")

        if not sample_df["largest_cluster_read_fraction"].is_monotonic_increasing:
            failures.append("largest_cluster_read_fraction非单调增加")

        qc_rows.append(
            {
                "sample_id": sample_id,
                "subject": sample_df["subject"].iloc[0],
                "n_radius_rows": len(sample_df),
                "observed_radii": ",".join(map(str, sorted(observed_radius_set))),
                "qc_status": "PASS" if not failures else "FAIL",
                "qc_message": "; ".join(failures),
            }
        )

    sample_qc = pd.DataFrame(qc_rows)

    global_qc = pd.DataFrame(
        [
            {
                "n_result_files": all_radius["source_file"].nunique(),
                "n_unique_samples": all_radius["sample_id"].nunique(),
                "n_rows": len(all_radius),
                "expected_sample_count": EXPECTED_SAMPLE_COUNT,
                "expected_radius_count": len(EXPECTED_RADIUS_LIST),
                "expected_rows": (
                    EXPECTED_SAMPLE_COUNT * len(EXPECTED_RADIUS_LIST)
                    if EXPECTED_SAMPLE_COUNT is not None
                    else np.nan
                ),
                "n_qc_pass": int((sample_qc["qc_status"] == "PASS").sum()),
                "n_qc_fail": int((sample_qc["qc_status"] == "FAIL").sum()),
            }
        ]
    )

    if EXPECTED_SAMPLE_COUNT is not None:
        detected = all_radius["sample_id"].nunique()
        if detected != EXPECTED_SAMPLE_COUNT:
            warnings.warn(
                f"检测到{detected}个完整结果样本，预期为{EXPECTED_SAMPLE_COUNT}。"
                "阶段1仍会输出当前已检测结果，但正式解读前应确认全量批处理已经完成。"
            )

    return sample_qc, global_qc


def summarize_radius_metrics(all_radius: pd.DataFrame) -> pd.DataFrame:
    """按radius汇总诊断指标的分布。"""

    rows = []

    for radius in EXPECTED_RADIUS_LIST:
        radius_df = all_radius[all_radius["radius"] == radius]

        for metric in RADIUS_DIAGNOSTIC_METRICS:
            values = pd.to_numeric(radius_df[metric], errors="coerce").dropna()
            if values.empty:
                continue

            rows.append(
                {
                    "radius": radius,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "minimum": float(values.min()),
                    "p05": float(values.quantile(0.05)),
                    "q1": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "q3": float(values.quantile(0.75)),
                    "p95": float(values.quantile(0.95)),
                    "maximum": float(values.max()),
                }
            )

    return pd.DataFrame(rows)


def calculate_giant_component_risk(all_radius: pd.DataFrame) -> pd.DataFrame:
    """统计不同radius形成较大/巨型连通分量的患者比例。"""

    rows = []
    thresholds = [0.10, 0.20, 0.50]

    for radius in EXPECTED_RADIUS_LIST:
        radius_df = all_radius[all_radius["radius"] == radius]

        for metric in [
            "largest_cluster_clone_fraction",
            "largest_cluster_read_fraction",
        ]:
            values = pd.to_numeric(radius_df[metric], errors="coerce").dropna()

            for threshold in thresholds:
                rows.append(
                    {
                        "radius": radius,
                        "metric": metric,
                        "threshold": threshold,
                        "n_samples": int(values.size),
                        "n_above_threshold": int((values > threshold).sum()),
                        "fraction_above_threshold": float((values > threshold).mean()),
                    }
                )

    return pd.DataFrame(rows)


def calculate_sample_size_correlations(all_radius: pd.DataFrame) -> pd.DataFrame:
    """计算样本规模与radius指标之间的Spearman相关。"""

    rows = []

    for radius in EXPECTED_RADIUS_LIST:
        radius_df = all_radius[all_radius["radius"] == radius]

        for size_variable in ["n_clonotypes", "total_clone_reads"]:
            for metric in SAMPLE_SIZE_CORRELATION_METRICS:
                pair = radius_df[[size_variable, metric]].dropna()

                if len(pair) < 3:
                    rho = np.nan
                    p_value = np.nan
                else:
                    rho, p_value = stats.spearmanr(pair[size_variable], pair[metric])

                rows.append(
                    {
                        "radius": radius,
                        "size_variable": size_variable,
                        "metric": metric,
                        "n": len(pair),
                        "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                        "p_value": (float(p_value) if np.isfinite(p_value) else np.nan),
                    }
                )

    result = pd.DataFrame(rows)
    result["fdr_within_size_metric"] = np.nan

    for _, group_index in result.groupby(["size_variable", "metric"]).groups.items():
        result.loc[group_index, "fdr_within_size_metric"] = bh_adjust(
            result.loc[group_index, "p_value"].to_numpy(dtype=float)
        )

    return result


def calculate_stratified_descriptives(merged: pd.DataFrame) -> pd.DataFrame:
    """按material和batch描述关键radius结构指标。"""

    rows = []

    for stratifier in [MATERIAL_COLUMN, BATCH_COLUMN]:
        if stratifier not in merged.columns:
            continue

        for (radius, level), group in merged.groupby(["radius", stratifier]):
            for metric in [
                "isolated_clone_fraction",
                "largest_cluster_clone_fraction",
                "largest_cluster_read_fraction",
                "cluster_shannon",
                "cluster_pielou_evenness",
            ]:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                if values.empty:
                    continue

                rows.append(
                    {
                        "stratifier": stratifier,
                        "level": str(level),
                        "radius": int(radius),
                        "metric": metric,
                        "n": int(values.size),
                        "median": float(values.median()),
                        "q1": float(values.quantile(0.25)),
                        "q3": float(values.quantile(0.75)),
                        "minimum": float(values.min()),
                        "maximum": float(values.max()),
                    }
                )

    return pd.DataFrame(rows)


def calculate_adjacent_radius_changes(all_radius: pd.DataFrame) -> pd.DataFrame:
    """计算每个患者相邻radius之间的指标变化。"""

    rows = []
    change_metrics = [
        "undirected_edges",
        "isolated_clone_fraction",
        "cluster_richness",
        "largest_cluster_clone_fraction",
        "largest_cluster_read_fraction",
        "cluster_shannon",
        "cluster_pielou_evenness",
    ]

    for sample_id, sample_df in all_radius.groupby("sample_id"):
        sample_df = sample_df.sort_values("radius").set_index("radius")

        for lower, upper in zip(EXPECTED_RADIUS_LIST[:-1], EXPECTED_RADIUS_LIST[1:]):
            if lower not in sample_df.index or upper not in sample_df.index:
                continue

            for metric in change_metrics:
                lower_value = float(sample_df.loc[lower, metric])
                upper_value = float(sample_df.loc[upper, metric])

                rows.append(
                    {
                        "sample_id": sample_id,
                        "subject": sample_df.loc[lower, "subject"],
                        "radius_from": lower,
                        "radius_to": upper,
                        "metric": metric,
                        "value_from": lower_value,
                        "value_to": upper_value,
                        "absolute_change": upper_value - lower_value,
                        "absolute_change_magnitude": abs(upper_value - lower_value),
                    }
                )

    return pd.DataFrame(rows)


def summarize_adjacent_radius_changes(changes: pd.DataFrame) -> pd.DataFrame:
    """汇总相邻radius变化的分布。"""

    if changes.empty:
        return pd.DataFrame()

    rows = []

    for (radius_from, radius_to, metric), group in changes.groupby(
        ["radius_from", "radius_to", "metric"]
    ):
        values = group["absolute_change"].dropna()
        magnitudes = group["absolute_change_magnitude"].dropna()

        rows.append(
            {
                "radius_from": radius_from,
                "radius_to": radius_to,
                "metric": metric,
                "n": len(values),
                "median_change": float(values.median()),
                "q1_change": float(values.quantile(0.25)),
                "q3_change": float(values.quantile(0.75)),
                "median_absolute_change": float(magnitudes.median()),
                "p95_absolute_change": float(magnitudes.quantile(0.95)),
                "maximum_absolute_change": float(magnitudes.max()),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 7. 阶段1可视化
# ============================================================


def plot_radius_trends(radius_summary: pd.DataFrame) -> None:
    """每个关键指标单独绘制radius总体趋势。"""

    for metric in RADIUS_DIAGNOSTIC_METRICS:
        plot_df = radius_summary[radius_summary["metric"] == metric].sort_values(
            "radius"
        )

        if plot_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        x = plot_df["radius"].to_numpy(dtype=float)
        median = plot_df["median"].to_numpy(dtype=float)
        q1 = plot_df["q1"].to_numpy(dtype=float)
        q3 = plot_df["q3"].to_numpy(dtype=float)
        p05 = plot_df["p05"].to_numpy(dtype=float)
        p95 = plot_df["p95"].to_numpy(dtype=float)

        ax.fill_between(x, p05, p95, alpha=0.12, label="5th–95th percentile")
        ax.fill_between(x, q1, q3, alpha=0.25, label="IQR")
        ax.plot(x, median, marker="o", linewidth=2.0, label="Median")
        ax.set_xlabel("TCRdist radius")
        ax.set_ylabel(metric_label(metric))
        ax.set_title(f"{metric_label(metric)} across candidate radii")
        ax.set_xticks(EXPECTED_RADIUS_LIST)
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)

        save_figure(
            fig,
            PHASE1_FIGURE_DIR / f"radius_trend_{safe_filename(metric)}",
        )


def plot_patient_trajectories(all_radius: pd.DataFrame) -> None:
    """绘制每位患者随radius变化的轨迹。"""

    if not DRAW_PATIENT_TRAJECTORIES:
        return

    for metric in TRAJECTORY_METRICS:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))

        for _, sample_df in all_radius.groupby("sample_id"):
            sample_df = sample_df.sort_values("radius")
            ax.plot(
                sample_df["radius"],
                sample_df[metric],
                linewidth=0.7,
                alpha=0.10,
            )

        median_df = (
            all_radius.groupby("radius", as_index=False)[metric]
            .median()
            .sort_values("radius")
        )
        ax.plot(
            median_df["radius"],
            median_df[metric],
            marker="o",
            linewidth=2.5,
            label="Median",
        )

        ax.set_xlabel("TCRdist radius")
        ax.set_ylabel(metric_label(metric))
        ax.set_title(f"Patient trajectories: {metric_label(metric)}")
        ax.set_xticks(EXPECTED_RADIUS_LIST)
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)

        save_figure(
            fig,
            PHASE1_FIGURE_DIR / f"patient_trajectories_{safe_filename(metric)}",
        )


def plot_giant_component_risk(risk_df: pd.DataFrame) -> None:
    """绘制不同radius下巨型连通分量风险。"""

    for metric in [
        "largest_cluster_clone_fraction",
        "largest_cluster_read_fraction",
    ]:
        metric_df = risk_df[risk_df["metric"] == metric]
        if metric_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(7.2, 4.9))

        for threshold, group in metric_df.groupby("threshold"):
            group = group.sort_values("radius")
            ax.plot(
                group["radius"],
                group["fraction_above_threshold"],
                marker="o",
                linewidth=2.0,
                label=f"> {threshold:.0%}",
            )

        ax.set_xlabel("TCRdist radius")
        ax.set_ylabel("Fraction of patients")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(EXPECTED_RADIUS_LIST)
        ax.set_title(f"Giant-component risk: {metric_label(metric)}")
        ax.legend(title="Largest cluster", frameon=False)
        ax.grid(alpha=0.2)

        save_figure(
            fig,
            PHASE1_FIGURE_DIR / f"giant_component_risk_{safe_filename(metric)}",
        )


def plot_radius_heatmaps(all_radius: pd.DataFrame) -> None:
    """以样本为行、radius为列绘制关键指标热图。"""

    # 按n_clonotypes从小到大排序，使样本规模依赖更容易观察。
    sample_order = (
        all_radius[["sample_id", "n_clonotypes"]]
        .drop_duplicates("sample_id")
        .sort_values("n_clonotypes")["sample_id"]
        .tolist()
    )

    for metric in HEATMAP_METRICS:
        matrix = all_radius.pivot(
            index="sample_id",
            columns="radius",
            values=metric,
        )
        matrix = matrix.reindex(index=sample_order, columns=EXPECTED_RADIUS_LIST)

        fig_height = max(6.0, min(14.0, 0.045 * len(matrix)))
        fig, ax = plt.subplots(figsize=(7.0, fig_height))
        image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto")
        ax.set_xlabel("TCRdist radius")
        ax.set_ylabel("Patients ordered by clonotype count")
        ax.set_title(f"Heatmap: {metric_label(metric)}")
        ax.set_xticks(np.arange(len(EXPECTED_RADIUS_LIST)))
        ax.set_xticklabels(EXPECTED_RADIUS_LIST)
        ax.set_yticks([])
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label(metric_label(metric))

        save_figure(
            fig,
            PHASE1_FIGURE_DIR / f"radius_heatmap_{safe_filename(metric)}",
        )


def plot_sample_size_scatter(all_radius: pd.DataFrame) -> None:
    """绘制n_clonotypes与关键网络指标的散点关系。"""

    if not DRAW_SAMPLE_SIZE_SCATTERS:
        return

    metrics = [
        "isolated_clone_fraction",
        "largest_cluster_clone_fraction",
    ]

    for metric in metrics:
        for radius in EXPECTED_RADIUS_LIST:
            plot_df = all_radius[all_radius["radius"] == radius][
                ["n_clonotypes", metric]
            ].dropna()

            if len(plot_df) < 3:
                continue

            rho, p_value = stats.spearmanr(plot_df["n_clonotypes"], plot_df[metric])

            fig, ax = plt.subplots(figsize=(6.2, 4.7))
            ax.scatter(
                plot_df["n_clonotypes"],
                plot_df[metric],
                alpha=0.7,
                s=25,
            )
            ax.set_xscale("log")
            ax.set_xlabel("Number of clonotypes (log scale)")
            ax.set_ylabel(metric_label(metric))
            ax.set_title(
                f"Radius {radius}: sample size vs {metric_label(metric)}\n"
                f"Spearman rho={rho:.3f}, P={p_value:.3g}"
            )
            ax.grid(alpha=0.2)

            save_figure(
                fig,
                PHASE1_FIGURE_DIR
                / f"sample_size_radius{radius:02d}_{safe_filename(metric)}",
            )


# ============================================================
# 8. 阶段2：固定radius后的cohort统计
# ============================================================


def cohort_descriptive_statistics(
    selected_df: pd.DataFrame,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """输出固定radius下各cohort描述统计。"""

    rows = []

    for metric in metrics:
        for cohort, group in selected_df.groupby(COHORT_COLUMN):
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue

            rows.append(
                {
                    "metric": metric,
                    "cohort": cohort,
                    "n": int(values.size),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "minimum": float(values.min()),
                    "q1": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "q3": float(values.quantile(0.75)),
                    "maximum": float(values.max()),
                }
            )

    return pd.DataFrame(rows)


def unadjusted_cohort_comparison(
    selected_df: pd.DataFrame,
    metrics: Sequence[str],
    bootstrap_iterations: int,
) -> pd.DataFrame:
    """Mann-Whitney U、Cliff's delta和bootstrap CI。"""

    rows = []

    for metric_index, metric in enumerate(metrics):
        target = (
            pd.to_numeric(
                selected_df.loc[
                    selected_df[COHORT_COLUMN] == TARGET_COHORT,
                    metric,
                ],
                errors="coerce",
            )
            .dropna()
            .to_numpy(dtype=float)
        )

        reference = (
            pd.to_numeric(
                selected_df.loc[
                    selected_df[COHORT_COLUMN] == REFERENCE_COHORT,
                    metric,
                ],
                errors="coerce",
            )
            .dropna()
            .to_numpy(dtype=float)
        )

        delta = cliffs_delta(target, reference)
        ci_low, ci_high = bootstrap_cliffs_delta_ci(
            target,
            reference,
            iterations=bootstrap_iterations,
            seed=BOOTSTRAP_SEED + metric_index,
        )
        p_value = mann_whitney_p(target, reference)

        rows.append(
            {
                "metric": metric,
                "target_cohort": TARGET_COHORT,
                "reference_cohort": REFERENCE_COHORT,
                "n_target": len(target),
                "n_reference": len(reference),
                "target_median": float(np.median(target)) if len(target) else np.nan,
                "reference_median": (
                    float(np.median(reference)) if len(reference) else np.nan
                ),
                "median_difference_target_minus_reference": (
                    float(np.median(target) - np.median(reference))
                    if len(target) and len(reference)
                    else np.nan
                ),
                "cliffs_delta": delta,
                "cliffs_delta_ci_low": ci_low,
                "cliffs_delta_ci_high": ci_high,
                "mann_whitney_p_value": p_value,
            }
        )

    result = pd.DataFrame(rows)
    result["mann_whitney_fdr"] = bh_adjust(
        result["mann_whitney_p_value"].to_numpy(dtype=float)
    )

    return result


def adjusted_cohort_comparison(
    selected_df: pd.DataFrame,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """使用HC3稳健标准误的多变量回归比较cohort。"""

    rows = []

    for metric in metrics:
        try:
            y, x, column_names, transform_name, used_index = (
                build_adjusted_design_matrix(selected_df, metric)
            )
            cohort_index = column_names.index("cohort_target")
            result = fit_ols_hc3(y, x, cohort_index)

            rows.append(
                {
                    "metric": metric,
                    "target_cohort": TARGET_COHORT,
                    "reference_cohort": REFERENCE_COHORT,
                    "outcome_transform": transform_name,
                    "cohort_beta_standardized": result.coefficient,
                    "cohort_beta_se_hc3": result.standard_error,
                    "cohort_beta_ci_low": result.confidence_low,
                    "cohort_beta_ci_high": result.confidence_high,
                    "cohort_p_value": result.p_value,
                    "n_observations": result.n_observations,
                    "n_parameters": result.n_parameters,
                    "residual_df": result.residual_df,
                    "model_columns": ",".join(column_names),
                    "status": "OK",
                    "message": "",
                }
            )

        except Exception as exc:
            rows.append(
                {
                    "metric": metric,
                    "target_cohort": TARGET_COHORT,
                    "reference_cohort": REFERENCE_COHORT,
                    "outcome_transform": "",
                    "cohort_beta_standardized": np.nan,
                    "cohort_beta_se_hc3": np.nan,
                    "cohort_beta_ci_low": np.nan,
                    "cohort_beta_ci_high": np.nan,
                    "cohort_p_value": np.nan,
                    "n_observations": np.nan,
                    "n_parameters": np.nan,
                    "residual_df": np.nan,
                    "model_columns": "",
                    "status": "FAILED",
                    "message": str(exc),
                }
            )

    result_df = pd.DataFrame(rows)
    result_df["cohort_fdr"] = bh_adjust(
        result_df["cohort_p_value"].to_numpy(dtype=float)
    )

    return result_df


def radius_sensitivity_analysis(
    merged: pd.DataFrame,
    metrics: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """在全部候选radius下重复cohort效应估计。"""

    unadjusted_rows = []
    adjusted_rows = []

    for radius in EXPECTED_RADIUS_LIST:
        radius_df = merged[merged["radius"] == radius].copy()

        for metric in metrics:
            target = (
                pd.to_numeric(
                    radius_df.loc[
                        radius_df[COHORT_COLUMN] == TARGET_COHORT,
                        metric,
                    ],
                    errors="coerce",
                )
                .dropna()
                .to_numpy(dtype=float)
            )

            reference = (
                pd.to_numeric(
                    radius_df.loc[
                        radius_df[COHORT_COLUMN] == REFERENCE_COHORT,
                        metric,
                    ],
                    errors="coerce",
                )
                .dropna()
                .to_numpy(dtype=float)
            )

            unadjusted_rows.append(
                {
                    "radius": radius,
                    "metric": metric,
                    "cliffs_delta": cliffs_delta(target, reference),
                    "mann_whitney_p_value": mann_whitney_p(target, reference),
                    "n_target": len(target),
                    "n_reference": len(reference),
                }
            )

            try:
                y, x, column_names, transform_name, _ = build_adjusted_design_matrix(
                    radius_df, metric
                )
                cohort_index = column_names.index("cohort_target")
                model = fit_ols_hc3(y, x, cohort_index)

                adjusted_rows.append(
                    {
                        "radius": radius,
                        "metric": metric,
                        "outcome_transform": transform_name,
                        "cohort_beta_standardized": model.coefficient,
                        "cohort_beta_ci_low": model.confidence_low,
                        "cohort_beta_ci_high": model.confidence_high,
                        "cohort_p_value": model.p_value,
                        "n_observations": model.n_observations,
                        "status": "OK",
                        "message": "",
                    }
                )
            except Exception as exc:
                adjusted_rows.append(
                    {
                        "radius": radius,
                        "metric": metric,
                        "outcome_transform": "",
                        "cohort_beta_standardized": np.nan,
                        "cohort_beta_ci_low": np.nan,
                        "cohort_beta_ci_high": np.nan,
                        "cohort_p_value": np.nan,
                        "n_observations": np.nan,
                        "status": "FAILED",
                        "message": str(exc),
                    }
                )

    unadjusted = pd.DataFrame(unadjusted_rows)
    adjusted = pd.DataFrame(adjusted_rows)

    unadjusted["fdr_within_metric"] = np.nan
    adjusted["fdr_within_metric"] = np.nan

    for metric, index in unadjusted.groupby("metric").groups.items():
        unadjusted.loc[index, "fdr_within_metric"] = bh_adjust(
            unadjusted.loc[index, "mann_whitney_p_value"].to_numpy(dtype=float)
        )

    for metric, index in adjusted.groupby("metric").groups.items():
        adjusted.loc[index, "fdr_within_metric"] = bh_adjust(
            adjusted.loc[index, "cohort_p_value"].to_numpy(dtype=float)
        )

    return unadjusted, adjusted


def stratified_cohort_comparison(
    selected_df: pd.DataFrame,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """按material分层进行描述性cohort比较。"""

    rows = []

    if MATERIAL_COLUMN not in selected_df.columns:
        return pd.DataFrame(rows)

    for material, material_df in selected_df.groupby(MATERIAL_COLUMN):
        for metric in metrics:
            target = (
                pd.to_numeric(
                    material_df.loc[
                        material_df[COHORT_COLUMN] == TARGET_COHORT,
                        metric,
                    ],
                    errors="coerce",
                )
                .dropna()
                .to_numpy(dtype=float)
            )

            reference = (
                pd.to_numeric(
                    material_df.loc[
                        material_df[COHORT_COLUMN] == REFERENCE_COHORT,
                        metric,
                    ],
                    errors="coerce",
                )
                .dropna()
                .to_numpy(dtype=float)
            )

            rows.append(
                {
                    "material": material,
                    "metric": metric,
                    "n_target": len(target),
                    "n_reference": len(reference),
                    "target_median": (
                        float(np.median(target)) if len(target) else np.nan
                    ),
                    "reference_median": (
                        float(np.median(reference)) if len(reference) else np.nan
                    ),
                    "cliffs_delta": cliffs_delta(target, reference),
                    "mann_whitney_p_value": mann_whitney_p(target, reference),
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["fdr_within_material"] = np.nan
        for material, index in result.groupby("material").groups.items():
            result.loc[index, "fdr_within_material"] = bh_adjust(
                result.loc[index, "mann_whitney_p_value"].to_numpy(dtype=float)
            )

    return result


# ============================================================
# 9. 阶段2可视化
# ============================================================


def plot_selected_radius_boxplots(
    selected_df: pd.DataFrame,
    unadjusted: pd.DataFrame,
    metrics: Sequence[str],
    selected_radius: int,
) -> None:
    """每个指标绘制cohort箱线图和散点。"""

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    cohort_order = [REFERENCE_COHORT, TARGET_COHORT]

    for metric in metrics:
        groups = []
        for cohort in cohort_order:
            values = (
                pd.to_numeric(
                    selected_df.loc[selected_df[COHORT_COLUMN] == cohort, metric],
                    errors="coerce",
                )
                .dropna()
                .to_numpy(dtype=float)
            )
            groups.append(values)

        if any(len(values) == 0 for values in groups):
            continue

        result_row = unadjusted[unadjusted["metric"] == metric]
        if result_row.empty:
            subtitle = ""
        else:
            row = result_row.iloc[0]
            subtitle = (
                f"Cliff's delta={row['cliffs_delta']:.3f}; "
                f"FDR={row['mann_whitney_fdr']:.3g}"
            )

        fig, ax = plt.subplots(figsize=(6.0, 4.9))
        ax.boxplot(groups, labels=cohort_order, showfliers=False)

        for position, values in enumerate(groups, start=1):
            jitter = rng.normal(loc=0.0, scale=0.045, size=len(values))
            ax.scatter(
                np.full(len(values), position) + jitter,
                values,
                s=24,
                alpha=0.65,
            )

        ax.set_xlabel("Cohort")
        ax.set_ylabel(metric_label(metric))
        ax.set_title(f"Radius {selected_radius}: {metric_label(metric)}\n{subtitle}")
        ax.grid(axis="y", alpha=0.2)

        save_figure(
            fig,
            PHASE2_FIGURE_DIR
            / f"radius{selected_radius:02d}_cohort_{safe_filename(metric)}",
        )



def significance_label(fdr_value: float) -> str:
    """将FDR转换为常用显著性标记。"""

    if not np.isfinite(fdr_value):
        return "NA"
    if fdr_value < 0.001:
        return "***"
    if fdr_value < 0.01:
        return "**"
    if fdr_value < 0.05:
        return "*"
    return "ns"


def plot_selected_radius_boxplots_extra(
    selected_df: pd.DataFrame,
    boxplot_comparison: pd.DataFrame,
    metrics: Sequence[str],
    selected_radius: int,
) -> None:
    """
    绘制RA与ILD的四个TCRdist cluster指标箱线图。

    使用与旧脚本一致的箱线图风格，并叠加每位患者的散点。
    图中仅显示每个指标自身的原始Mann-Whitney U检验P值；
    主分析表中的FDR结果保持不变，可用于正式统计结论。
    """

    rng = np.random.default_rng(BOOTSTRAP_SEED + 1000)
    cohort_order = [REFERENCE_COHORT, TARGET_COHORT]

    for metric in metrics:
        if metric not in selected_df.columns:
            warnings.warn(f"箱线图指标不存在，已跳过: {metric}")
            continue

        groups = []
        x_labels = []

        for cohort in cohort_order:
            values = (
                pd.to_numeric(
                    selected_df.loc[selected_df[COHORT_COLUMN] == cohort, metric],
                    errors="coerce",
                )
                .dropna()
                .to_numpy(dtype=float)
            )
            groups.append(values)
            x_labels.append(f"{cohort}\n(n={len(values)})")

        if any(len(values) == 0 for values in groups):
            continue

        result_row = boxplot_comparison[boxplot_comparison["metric"] == metric]

        if result_row.empty:
            subtitle = ""
        else:
            row = result_row.iloc[0]
            p_value = float(row["mann_whitney_p_value"])
            delta_value = float(row["cliffs_delta"])
            subtitle = (
                f"Cliff's delta={delta_value:.3f}; "
                f"P={p_value:.3g} "
                f"({significance_label(p_value)})"
            )

        fig, ax = plt.subplots(figsize=(6.0, 5.1))
        ax.boxplot(groups, labels=x_labels, showfliers=False)

        for position, values in enumerate(groups, start=1):
            jitter = rng.normal(loc=0.0, scale=0.045, size=len(values))
            ax.scatter(
                np.full(len(values), position) + jitter,
                values,
                s=22,
                alpha=0.55,
            )

        ax.set_xlabel("Cohort")
        ax.set_ylabel(metric_label(metric))
        ax.set_title(
            f"Radius {selected_radius}: {metric_label(metric)}\n{subtitle}"
        )
        ax.grid(axis="y", alpha=0.2)

        save_figure(
            fig,
            PHASE2_FIGURE_DIR / f"radius{selected_radius:02d}_boxplot_{safe_filename(metric)}",
        )


def plot_selected_radius_boxplot_panel(
    selected_df: pd.DataFrame,
    boxplot_comparison: pd.DataFrame,
    metrics: Sequence[str],
    selected_radius: int,
) -> None:
    """将四个TCRdist cluster指标组合为2×2箱线图面板。"""

    available_metrics = [metric for metric in metrics if metric in selected_df.columns]

    if not available_metrics:
        return

    rng = np.random.default_rng(BOOTSTRAP_SEED + 2000)
    cohort_order = [REFERENCE_COHORT, TARGET_COHORT]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0))
    axes_flat = axes.ravel()

    for axis_index, ax in enumerate(axes_flat):
        if axis_index >= len(available_metrics):
            ax.axis("off")
            continue

        metric = available_metrics[axis_index]
        groups = []
        x_labels = []

        for cohort in cohort_order:
            values = (
                pd.to_numeric(
                    selected_df.loc[selected_df[COHORT_COLUMN] == cohort, metric],
                    errors="coerce",
                )
                .dropna()
                .to_numpy(dtype=float)
            )
            groups.append(values)
            x_labels.append(f"{cohort}\n(n={len(values)})")

        ax.boxplot(groups, labels=x_labels, showfliers=False)

        for position, values in enumerate(groups, start=1):
            jitter = rng.normal(loc=0.0, scale=0.045, size=len(values))
            ax.scatter(
                np.full(len(values), position) + jitter,
                values,
                s=17,
                alpha=0.45,
            )

        result_row = boxplot_comparison[boxplot_comparison["metric"] == metric]
        if result_row.empty:
            statistic_text = ""
        else:
            row = result_row.iloc[0]
            p_value = float(row["mann_whitney_p_value"])
            statistic_text = f"P={p_value:.3g} ({significance_label(p_value)})"

        ax.set_ylabel(metric_label(metric))
        ax.set_title(f"{metric_label(metric)}\n{statistic_text}")
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        f"RA versus ILD TCRdist cluster metrics at radius {selected_radius}",
        y=1.01,
    )

    save_figure(
        fig,
        PHASE2_FIGURE_DIR / f"radius{selected_radius:02d}_four_cluster_metrics_boxplot_panel",
    )

def plot_adjusted_forest(
    adjusted: pd.DataFrame,
    selected_radius: int,
) -> None:
    """绘制调整后标准化cohort效应森林图。"""

    plot_df = (
        adjusted[adjusted["status"] == "OK"]
        .dropna(
            subset=[
                "cohort_beta_standardized",
                "cohort_beta_ci_low",
                "cohort_beta_ci_high",
            ]
        )
        .copy()
    )

    if plot_df.empty:
        return

    plot_df = plot_df.sort_values("cohort_beta_standardized")
    y_positions = np.arange(len(plot_df))
    estimates = plot_df["cohort_beta_standardized"].to_numpy(dtype=float)
    lower = plot_df["cohort_beta_ci_low"].to_numpy(dtype=float)
    upper = plot_df["cohort_beta_ci_high"].to_numpy(dtype=float)
    xerr = np.vstack([estimates - lower, upper - estimates])

    fig_height = max(4.8, 0.45 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(8.0, fig_height))
    ax.errorbar(
        estimates,
        y_positions,
        xerr=xerr,
        fmt="o",
        capsize=3,
    )
    ax.axvline(0.0, linewidth=1.0, linestyle="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([metric_label(metric) for metric in plot_df["metric"]])
    ax.set_xlabel(
        f"Adjusted standardized effect ({TARGET_COHORT} minus {REFERENCE_COHORT})"
    )
    ax.set_title(f"Adjusted cohort effects at radius {selected_radius}")
    ax.grid(axis="x", alpha=0.2)

    save_figure(
        fig,
        PHASE2_FIGURE_DIR / f"radius{selected_radius:02d}_adjusted_forest",
    )


def plot_metric_correlation_heatmap(
    selected_df: pd.DataFrame,
    metrics: Sequence[str],
    selected_radius: int,
) -> None:
    """绘制固定radius下主要指标Spearman相关热图。"""

    available = [metric for metric in metrics if metric in selected_df.columns]
    correlation = selected_df[available].corr(method="spearman")

    if correlation.empty:
        return

    fig_size = max(7.0, 0.65 * len(available) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(correlation.to_numpy(dtype=float), vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(available)))
    ax.set_yticks(np.arange(len(available)))
    ax.set_xticklabels(
        [metric_label(metric) for metric in available],
        rotation=45,
        ha="right",
    )
    ax.set_yticklabels([metric_label(metric) for metric in available])
    ax.set_title(f"Spearman correlations at radius {selected_radius}")

    for row_index in range(len(available)):
        for column_index in range(len(available)):
            ax.text(
                column_index,
                row_index,
                f"{correlation.iloc[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Spearman rho")

    save_figure(
        fig,
        PHASE2_FIGURE_DIR / f"radius{selected_radius:02d}_metric_correlation_heatmap",
    )


def plot_effect_across_radius(
    adjusted_sensitivity: pd.DataFrame,
    metrics: Sequence[str],
) -> None:
    """绘制调整后cohort效应随radius变化。"""

    for metric in metrics:
        plot_df = adjusted_sensitivity[
            (adjusted_sensitivity["metric"] == metric)
            & (adjusted_sensitivity["status"] == "OK")
        ].sort_values("radius")

        if plot_df.empty:
            continue

        x = plot_df["radius"].to_numpy(dtype=float)
        estimate = plot_df["cohort_beta_standardized"].to_numpy(dtype=float)
        lower = plot_df["cohort_beta_ci_low"].to_numpy(dtype=float)
        upper = plot_df["cohort_beta_ci_high"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        ax.fill_between(x, lower, upper, alpha=0.20, label="95% CI")
        ax.plot(x, estimate, marker="o", linewidth=2.0, label="Adjusted effect")
        ax.axhline(0.0, linewidth=1.0, linestyle="--")
        ax.set_xticks(EXPECTED_RADIUS_LIST)
        ax.set_xlabel("TCRdist radius")
        ax.set_ylabel(f"Standardized effect ({TARGET_COHORT} minus {REFERENCE_COHORT})")
        ax.set_title(f"Cohort effect across radius: {metric_label(metric)}")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)

        save_figure(
            fig,
            PHASE2_FIGURE_DIR / f"effect_across_radius_{safe_filename(metric)}",
        )


# ============================================================
# 10. 主分析流程
# ============================================================


def run_phase1(
    all_radius: pd.DataFrame,
    merged: pd.DataFrame,
) -> Dict[str, Path]:
    """运行全样本radius选择阶段。"""

    print("[Phase 1] Running radius-result QC...", flush=True)
    sample_qc, global_qc = run_radius_qc(all_radius)

    all_radius_file = PHASE1_TABLE_DIR / "all_samples_all_radius_long.csv"
    merged_file = PHASE1_TABLE_DIR / "all_samples_all_radius_with_metadata.csv"
    sample_qc_file = PHASE1_TABLE_DIR / "radius_result_sample_qc.csv"
    global_qc_file = PHASE1_TABLE_DIR / "radius_result_global_qc.csv"

    all_radius.to_csv(all_radius_file, index=False)
    merged.to_csv(merged_file, index=False)
    sample_qc.to_csv(sample_qc_file, index=False)
    global_qc.to_csv(global_qc_file, index=False)

    print("[Phase 1] Summarizing radius distributions...", flush=True)
    radius_summary = summarize_radius_metrics(all_radius)
    giant_risk = calculate_giant_component_risk(all_radius)
    size_correlations = calculate_sample_size_correlations(all_radius)
    stratified = calculate_stratified_descriptives(merged)
    changes = calculate_adjacent_radius_changes(all_radius)
    change_summary = summarize_adjacent_radius_changes(changes)

    radius_summary_file = PHASE1_TABLE_DIR / "radius_distribution_summary.csv"
    giant_risk_file = PHASE1_TABLE_DIR / "radius_giant_component_risk.csv"
    size_correlation_file = PHASE1_TABLE_DIR / "radius_sample_size_correlations.csv"
    stratified_file = PHASE1_TABLE_DIR / "radius_batch_material_descriptives.csv"
    changes_file = PHASE1_TABLE_DIR / "adjacent_radius_patient_changes.csv"
    change_summary_file = PHASE1_TABLE_DIR / "adjacent_radius_change_summary.csv"

    radius_summary.to_csv(radius_summary_file, index=False)
    giant_risk.to_csv(giant_risk_file, index=False)
    size_correlations.to_csv(size_correlation_file, index=False)
    stratified.to_csv(stratified_file, index=False)
    changes.to_csv(changes_file, index=False)
    change_summary.to_csv(change_summary_file, index=False)

    print("[Phase 1] Creating radius-diagnostic figures...", flush=True)
    plot_radius_trends(radius_summary)
    plot_patient_trajectories(all_radius)
    plot_giant_component_risk(giant_risk)
    plot_radius_heatmaps(all_radius)
    plot_sample_size_scatter(all_radius)

    phase1_manifest = {
        "result_root": str(RESULT_ROOT),
        "metadata_file": str(METADATA_FILE),
        "n_result_files": int(all_radius["source_file"].nunique()),
        "n_samples": int(all_radius["sample_id"].nunique()),
        "n_rows": int(len(all_radius)),
        "candidate_radii": EXPECTED_RADIUS_LIST,
        "selected_radius": SELECTED_RADIUS,
        "sample_qc_pass": int((sample_qc["qc_status"] == "PASS").sum()),
        "sample_qc_fail": int((sample_qc["qc_status"] == "FAIL").sum()),
        "phase2_executed": SELECTED_RADIUS is not None,
    }
    write_json(phase1_manifest, PHASE1_DIR / "phase1_run_manifest.json")

    return {
        "all_radius_file": all_radius_file,
        "merged_file": merged_file,
        "sample_qc_file": sample_qc_file,
        "global_qc_file": global_qc_file,
        "radius_summary_file": radius_summary_file,
        "giant_risk_file": giant_risk_file,
        "size_correlation_file": size_correlation_file,
        "stratified_file": stratified_file,
        "change_summary_file": change_summary_file,
    }


def run_phase2(merged: pd.DataFrame, selected_radius: int) -> Dict[str, Path]:
    """运行固定主radius后的cohort比较。"""

    if selected_radius not in EXPECTED_RADIUS_LIST:
        raise ValueError(
            f"SELECTED_RADIUS={selected_radius}不在候选列表{EXPECTED_RADIUS_LIST}中"
        )

    cohort_levels = set(merged[COHORT_COLUMN].dropna().astype(str).unique())
    required_cohorts = {REFERENCE_COHORT, TARGET_COHORT}
    if not required_cohorts.issubset(cohort_levels):
        raise ValueError(
            f"cohort列未同时包含{sorted(required_cohorts)}，实际为{sorted(cohort_levels)}"
        )

    selected_df = merged[merged["radius"] == selected_radius].copy()

    if selected_df["sample_id"].duplicated().any():
        raise ValueError("固定radius数据中sample_id重复")

    detected_samples = selected_df["sample_id"].nunique()
    if EXPECTED_SAMPLE_COUNT is not None and detected_samples != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"固定radius检测到{detected_samples}例，预期{EXPECTED_SAMPLE_COUNT}例。"
            "请确认全量批处理完成后再执行阶段2。"
        )

    print(
        f"[Phase 2] Running selected-radius analysis: radius={selected_radius}...",
        flush=True,
    )

    selected_file = (
        PHASE2_TABLE_DIR / f"radius{selected_radius:02d}_selected_patient_metrics.csv"
    )
    selected_df.to_csv(selected_file, index=False)

    descriptives = cohort_descriptive_statistics(selected_df, ALL_COHORT_METRICS)
    unadjusted = unadjusted_cohort_comparison(
        selected_df,
        ALL_COHORT_METRICS,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
    )

    # 四个展示指标单独形成一个明确的FDR检验族，
    # 不改变主分析ALL_COHORT_METRICS的FDR结果。
    boxplot_comparison = unadjusted_cohort_comparison(
        selected_df,
        COHORT_BOXPLOT_METRICS,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
    )

    adjusted = adjusted_cohort_comparison(
        selected_df,
        ALL_COHORT_METRICS,
    )
    stratified = stratified_cohort_comparison(
        selected_df,
        PRIMARY_COHORT_METRICS,
    )

    sensitivity_unadjusted, sensitivity_adjusted = radius_sensitivity_analysis(
        merged,
        PRIMARY_COHORT_METRICS,
    )

    descriptive_file = (
        PHASE2_TABLE_DIR
        / f"radius{selected_radius:02d}_cohort_descriptive_statistics.csv"
    )
    unadjusted_file = (
        PHASE2_TABLE_DIR
        / f"radius{selected_radius:02d}_cohort_unadjusted_comparison.csv"
    )
    adjusted_file = (
        PHASE2_TABLE_DIR / f"radius{selected_radius:02d}_cohort_adjusted_comparison.csv"
    )
    boxplot_comparison_file = (
        PHASE2_TABLE_DIR
        / f"radius{selected_radius:02d}_four_boxplot_metrics_unadjusted_comparison.csv"
    )
    stratified_file = (
        PHASE2_TABLE_DIR
        / f"radius{selected_radius:02d}_material_stratified_comparison.csv"
    )
    sensitivity_unadjusted_file = PHASE2_TABLE_DIR / "radius_sensitivity_unadjusted.csv"
    sensitivity_adjusted_file = PHASE2_TABLE_DIR / "radius_sensitivity_adjusted.csv"

    descriptives.to_csv(descriptive_file, index=False)
    unadjusted.to_csv(unadjusted_file, index=False)
    adjusted.to_csv(adjusted_file, index=False)
    boxplot_comparison.to_csv(boxplot_comparison_file, index=False)
    stratified.to_csv(stratified_file, index=False)
    sensitivity_unadjusted.to_csv(sensitivity_unadjusted_file, index=False)
    sensitivity_adjusted.to_csv(sensitivity_adjusted_file, index=False)

    print("[Phase 2] Creating selected-radius figures...", flush=True)
    plot_selected_radius_boxplots(
        selected_df,
        unadjusted,
        PRIMARY_COHORT_METRICS,
        selected_radius,
    )
    plot_selected_radius_boxplots_extra(
        selected_df,
        boxplot_comparison,
        COHORT_BOXPLOT_METRICS,
        selected_radius,
    )
    plot_selected_radius_boxplot_panel(
        selected_df,
        boxplot_comparison,
        COHORT_BOXPLOT_METRICS,
        selected_radius,
    )
    plot_adjusted_forest(adjusted, selected_radius)
    plot_metric_correlation_heatmap(
        selected_df,
        ALL_COHORT_METRICS,
        selected_radius,
    )
    plot_effect_across_radius(
        sensitivity_adjusted,
        PRIMARY_COHORT_METRICS,
    )

    phase2_manifest = {
        "selected_radius": selected_radius,
        "reference_cohort": REFERENCE_COHORT,
        "target_cohort": TARGET_COHORT,
        "n_samples": int(selected_df["sample_id"].nunique()),
        "primary_metrics": PRIMARY_COHORT_METRICS,
        "secondary_metrics": SECONDARY_COHORT_METRICS,
        "cohort_boxplot_metrics": COHORT_BOXPLOT_METRICS,
        "boxplot_summary": "boxplot with individual patient points",
        "boxplot_fdr_family": "BH correction across the four COHORT_BOXPLOT_METRICS only",
        "adjustment_continuous": ADJUSTMENT_CONTINUOUS,
        "adjustment_categorical": ADJUSTMENT_CATEGORICAL,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "adjusted_effect_interpretation": (
            f"Standardized outcome difference: {TARGET_COHORT} minus {REFERENCE_COHORT}"
        ),
    }
    write_json(
        phase2_manifest,
        PHASE2_DIR / f"radius{selected_radius:02d}_phase2_run_manifest.json",
    )

    return {
        "selected_file": selected_file,
        "descriptive_file": descriptive_file,
        "unadjusted_file": unadjusted_file,
        "adjusted_file": adjusted_file,
        "boxplot_comparison_file": boxplot_comparison_file,
        "stratified_file": stratified_file,
        "sensitivity_unadjusted_file": sensitivity_unadjusted_file,
        "sensitivity_adjusted_file": sensitivity_adjusted_file,
    }


def main() -> None:
    """程序入口。"""

    ensure_directories()

    print("Discovering all-sample radius result files...", flush=True)
    result_files = discover_radius_summary_files()
    print(f"Detected radius summary files: {len(result_files)}", flush=True)

    print("Loading and merging radius results...", flush=True)
    all_radius = load_all_radius_results(result_files)

    print("Loading metadata...", flush=True)
    metadata = load_metadata()
    merged = merge_radius_with_metadata(all_radius, metadata)

    phase1_outputs = run_phase1(all_radius, merged)
    print(
        f"Phase 1 finished. Main summary: {phase1_outputs['radius_summary_file']}",
        flush=True,
    )

    if SELECTED_RADIUS is None:
        print(
            "SELECTED_RADIUS=None. Phase 2 was not executed. "
            "Review phase1_radius_selection outputs, set SELECTED_RADIUS, "
            "and rerun this script.",
            flush=True,
        )
        return

    phase2_outputs = run_phase2(merged, int(SELECTED_RADIUS))
    print(
        f"Phase 2 finished. Adjusted results: {phase2_outputs['adjusted_file']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
