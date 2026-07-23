import re
import pandas as pd
from tcrdist.repertoire import TCRrep
from pathlib import Path


clone_path = (
    "/data/users/chenhaisheng/RA-ILD/TRB/vdj/result/"
    "TRB_tcrdist_clone_table_all_samples.csv"
)

metadata_path = "/data/users/chenhaisheng/RA-ILD/metadata.csv"

# 1. 读取数据
clones = pd.read_csv(clone_path)
metadata = pd.read_csv(metadata_path)

# 2. 只保留连接所需的metadata列
meta_sub = metadata[["patient", "libraryid", "cohort"]].drop_duplicates()

# 检查一个libraryid是否错误地对应多个患者
if meta_sub["libraryid"].duplicated().any():
    duplicated_ids = meta_sub.loc[
        meta_sub["libraryid"].duplicated(keep=False), "libraryid"
    ].unique()

    raise ValueError(
        "metadata中存在重复libraryid: " + ", ".join(map(str, duplicated_ids[:20]))
    )

# 3. 连接患者与疾病分组
df = clones.merge(meta_sub, on="libraryid", how="left", validate="many_to_one")

df = df.rename(columns={"patient": "subject"})


# 4. 给V/J基因补充等位基因
def add_allele(gene):
    if pd.isna(gene):
        return gene

    gene = str(gene).strip()

    if "*" in gene:
        return gene

    return f"{gene}*01"


df["v_b_gene"] = df["v_b_gene"].map(add_allele)
df["j_b_gene"] = df["j_b_gene"].map(add_allele)

# 5. count必须为正整数
df["count"] = pd.to_numeric(df["count"], errors="raise")

if (df["count"] <= 0).any():
    raise ValueError("发现count <= 0的记录")

df["count"] = df["count"].astype(int)

# 6. 检查必要字段缺失
required_cols = [
    "subject",
    "count",
    "v_b_gene",
    "j_b_gene",
    "cdr3_b_aa",
    "clone_id",
]

missing_summary = df[required_cols].isna().sum()

print("Missing-value summary:")
print(missing_summary)

if missing_summary.sum() > 0:
    raise ValueError("必要字段存在缺失值，请先检查")

# 7. 检查CDR3-AA是否合法
valid_aa_pattern = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

invalid_aa = ~df["cdr3_b_aa"].astype(str).str.fullmatch(valid_aa_pattern)

print("Invalid CDR3-AA rows:", int(invalid_aa.sum()))

if invalid_aa.any():
    print(df.loc[invalid_aa, ["libraryid", "clone_id", "cdr3_b_aa"]].head(20))

# 8. 检查clone_id是否唯一
print("Duplicated clone_id:", int(df["clone_id"].duplicated().sum()))

if df["clone_id"].duplicated().any():
    raise ValueError("clone_id不是全局唯一")

# 9. 检查上游聚合后是否仍有重复clonotype
clone_key = [
    "subject",
    "v_b_gene",
    "j_b_gene",
    "cdr3_b_aa",
]

duplicated_clones = df.duplicated(subset=clone_key, keep=False)

print("Duplicated subject+V+J+CDR3-AA rows:", int(duplicated_clones.sum()))

""" #统计不同标准下过滤效果
if duplicated_clones.any():
    print(
        df.loc[duplicated_clones, clone_key + ["clone_id", "count"]]
        .sort_values(clone_key)
        .head(30)
    )

# 10. 过滤前统计
clone_key = ["v_b_gene", "j_b_gene", "cdr3_b_aa"]

# 统计每个clone出现在多少个不同患者中
clone_patient_count = df.groupby(clone_key)["subject"].nunique().reset_index()
clone_patient_count.columns = clone_key + ["n_patients"]

df_with_patient_count = df.merge(clone_patient_count, on=clone_key, how="left")

# 标准1: 至少在2个患者中出现
filter1 = df_with_patient_count["n_patients"] >= 2
# 标准2: count > 1
filter2 = df_with_patient_count["count"] > 1
# 标准3: 同时满足标准1和标准2
filter3 = filter1 & filter2
# 标准4：count > 2
filter4 = df_with_patient_count["count"] > 2

print(f"\n=== 过滤前总行数: {len(df)} ===")
print(f"标准1 (≥2个患者): {int(filter1.sum())} 行")
print(f"标准2 (count>1):   {int(filter2.sum())} 行")
print(f"标准3 (两者都满足): {int(filter3.sum())} 行")
print(f"标准4 (count>2):   {int(filter4.sum())} 行")

# 简单评估两个过滤条件的效果
clone_key = ["v_b_gene", "j_b_gene", "cdr3_b_aa"]

n_total = len(df)

# 条件1：同一 V+J+CDR3-AA 至少出现在两个患者中
mask_shared_ge2 = df.duplicated(subset=clone_key, keep=False)

# 条件2：患者内该 clone 的 count > 1
mask_count_gt1 = df["count"] > 1

# 两个条件同时满足
mask_combined = mask_shared_ge2 & mask_count_gt1

print(f"Original clones: {n_total:,}")

print(
    f"Shared by >=2 subjects: "
    f"{mask_shared_ge2.sum():,} "
    f"({mask_shared_ge2.mean() * 100:.2f}%)"
)

print(f"Count >1: {mask_count_gt1.sum():,} ({mask_count_gt1.mean() * 100:.2f}%)")

print(
    f"Shared by >=2 subjects AND count >1: "
    f"{mask_combined.sum():,} "
    f"({mask_combined.mean() * 100:.2f}%)"
)
"""

"""统计基于count大于2的分布情况
clone_n = df.loc[df["count"] > 2].groupby("subject").size()

print(clone_n.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

print("\nLargest samples:")
print(clone_n.sort_values(ascending=False).head(10))
"""

# 11. 使用 count > 2 过滤
# 即只保留 count >= 3 的患者内 clonotype
filtered_df = df.loc[
    df["count"] > 2,
    [
        "libraryid",
        "subject",
        "count",
        "v_b_gene",
        "j_b_gene",
        "cdr3_b_aa",
        "clone_id",
    ],
].copy()

print("")
print(f"过滤前总行数: {len(df):,}")
print(f"count > 2 后总行数: {len(filtered_df):,}")
print(f"样本数量: {filtered_df['libraryid'].nunique():,}")
print(f"患者数量: {filtered_df['subject'].nunique():,}")


# 12. 检查每个libraryid是否只对应一个患者
library_subject_n = filtered_df.groupby("libraryid")["subject"].nunique()

bad_libraryids = library_subject_n[library_subject_n > 1].index.tolist()

if bad_libraryids:
    raise ValueError(
        "以下libraryid对应多个subject: " + ", ".join(map(str, bad_libraryids[:20]))
    )


# 13. 创建单样本输出目录
single_table_dir = Path("/data/users/chenhaisheng/RA-ILD/TRB/vdj/result/single_table")

single_table_dir.mkdir(
    parents=True,
    exist_ok=True,
)

print("")
print("单样本输出目录:", single_table_dir)


# 14. 每个输出文件只保留tcrdist3需要的列
tcrdist_cols = [
    "subject",
    "count",
    "v_b_gene",
    "j_b_gene",
    "cdr3_b_aa",
    "clone_id",
]


# 15. 按libraryid拆分并输出
written_files = 0
written_rows = 0

for libraryid, sample_df in filtered_df.groupby(
    "libraryid",
    sort=True,
    observed=True,
):
    sample_df = sample_df[tcrdist_cols].copy()

    # 稳定排序，方便检查和复现
    sample_df = sample_df.sort_values(
        by=[
            "v_b_gene",
            "j_b_gene",
            "cdr3_b_aa",
        ]
    ).reset_index(drop=True)

    # 检查该样本内部是否仍有重复clone
    duplicate_mask = sample_df.duplicated(
        subset=[
            "subject",
            "v_b_gene",
            "j_b_gene",
            "cdr3_b_aa",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        raise ValueError(
            f"{libraryid} 中发现重复的 "
            "subject+V+J+CDR3-AA记录，"
            f"重复行数: {int(duplicate_mask.sum())}"
        )

    output_file = single_table_dir / f"{libraryid}_TRB_tcrdist_input.csv"

    sample_df.to_csv(
        output_file,
        index=False,
    )

    written_files += 1
    written_rows += len(sample_df)

    print(f"[{written_files:03d}] {libraryid}: {len(sample_df):,} clones")


# 16. 最终一致性检查
if written_rows != len(filtered_df):
    raise RuntimeError(
        "拆分后的总行数与过滤后数据不一致: "
        f"拆分后={written_rows:,}, "
        f"过滤后={len(filtered_df):,}"
    )

print("")
print("=" * 60)
print("单样本表拆分完成")
print(f"输出文件数: {written_files:,}")
print(f"输出总行数: {written_rows:,}")
print(f"输出目录: {single_table_dir}")
print("=" * 60)
