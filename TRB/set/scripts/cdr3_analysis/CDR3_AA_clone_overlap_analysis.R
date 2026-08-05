library(ggplot2)

setwd("/data/users/chenhaisheng/RA-ILD/")

# ====================================================================
# CDR3 AA Clone 重叠分析
# 条件：K = 12000（过滤前12000条clone），M = 10000（取前10000条统计）
# 统计：
#   1. 在所有样本（RA + ILD）中共有的 clone 数量
#   2. 只在所有 RA 患者中共有的 clone 数量（不在任何 ILD 样本中）
#   3. 只在所有 ILD 患者中共有的 clone 数量（不在任何 RA 样本中）
# ====================================================================

K <- 12000   # 过滤阈值：只保留每个样本的前 K 条 clone
M <- 8000    # 统计范围：在 K 内取前 M 条 clone

metadata <- read.csv("./TRB/metadata.csv")

aa_dir <- "./TRB/result/01_AA_clone_table/"
aa_files <- list.files(aa_dir, pattern = "_AA_clone_table\\.csv$")
aa_ids <- gsub("_AA_clone_table\\.csv$", "", aa_files)
aa_ids <- aa_ids[!grepl("^01_", aa_ids)]
aa_files_f <- paste0(aa_ids, "_AA_clone_table.csv")

cat("正在读取 AA clone table...\n")
cat("参数: K =", K, ", M =", M, "\n\n")

# 读取每个样本的 CDR3 AA 序列
all_cdr3 <- lapply(file.path(aa_dir, aa_files_f), function(f) {
  data <- read.csv(f, stringsAsFactors = FALSE)
  n_keep <- min(K, nrow(data))
  cdr3_seqs <- data$cdr3_aa[1:n_keep]
  list(
    top_K = cdr3_seqs,
    top_M = head(cdr3_seqs, min(M, n_keep))
  )
})
names(all_cdr3) <- aa_ids
cat("完成！共读取", length(all_cdr3), "个样本。\n\n")

# 分组
sample_info <- data.frame(libraryid = aa_ids, stringsAsFactors = FALSE)
sample_info <- merge(sample_info, metadata[, c("libraryid", "material", "cohort", "patient")],
                     by = "libraryid")
rownames(sample_info) <- sample_info$libraryid

ra_samples  <- sample_info$libraryid[sample_info$cohort == "RA"]
ild_samples <- sample_info$libraryid[sample_info$cohort == "ILD"]

cat("RA 样本数:", length(ra_samples), "\n")
cat("ILD 样本数:", length(ild_samples), "\n")
cat("总样本数:", length(aa_ids), "\n\n")

# ====================================================================
# 高效方法：构建 clone → 样本出现次数的映射
# ====================================================================

topM_sets <- lapply(all_cdr3, function(x) x$top_M)

# 将所有 clone 展开成向量，附带样本名
clone_vec <- unlist(lapply(names(topM_sets), function(sid) {
  topM_sets[[sid]]
}))
sample_vec <- rep(names(topM_sets), times = sapply(topM_sets, length))

# 用 table 统计每个 clone 出现在哪些样本中（构建 clone × sample 矩阵）
# 更快的方法：用 split 得到每个 clone 的样本列表
cat("正在构建 clone 出现统计...\n")
clone_to_samples <- split(sample_vec, clone_vec)
cat("共有", length(clone_to_samples), "个唯一 CDR3 AA clone（Top M 范围内）\n\n")

# 每个 clone 出现的样本数
clone_nsamples <- sapply(clone_to_samples, function(x) length(unique(x)))

# ====================================================================
# 核心统计
# ====================================================================

# 1. 所有样本共有的 clone
all_shared <- names(clone_nsamples)[clone_nsamples == length(aa_ids)]
cat("========================================================================\n")
cat("1. 在所有", length(aa_ids), "个样本（RA+ILD）中共有的 clone：",
    length(all_shared), "条\n")
if (length(all_shared) > 0) {
  for (s in all_shared) cat("   ", s, "\n")
}

# 2. 只在所有 RA 中共有的 clone
#    条件：出现在所有 RA 样本中 AND 不出现在任何 ILD 样本中
ra_shared <- names(clone_nsamples)[clone_nsamples == length(ra_samples)]
# 但是 clone_nsamples 统计的是总样本数，需要精确检查
# 用更精确的方法：
cat("正在计算 RA 专属共有 clone...\n")
ra_sample_sets <- lapply(ra_samples, function(sid) topM_sets[[sid]])
ra_shared_clones <- Reduce(intersect, ra_sample_sets)
ild_all_clones <- unique(unlist(lapply(ild_samples, function(sid) topM_sets[[sid]])))
ra_only_shared <- setdiff(ra_shared_clones, ild_all_clones)
cat("2. 只在所有 RA 患者中共有的 clone（所有 RA 共有，且不在任何 ILD 中）：",
    length(ra_only_shared), "条\n")
if (length(ra_only_shared) > 0) {
  for (s in ra_only_shared) cat("   ", s, "\n")
}

# 3. 只在所有 ILD 中共有的 clone
cat("正在计算 ILD 专属共有 clone...\n")
ild_sample_sets <- lapply(ild_samples, function(sid) topM_sets[[sid]])
ild_shared_clones <- Reduce(intersect, ild_sample_sets)
ra_all_clones <- unique(unlist(lapply(ra_samples, function(sid) topM_sets[[sid]])))
ild_only_shared <- setdiff(ild_shared_clones, ra_all_clones)
cat("3. 只在所有 ILD 患者中共有的 clone（所有 ILD 共有，且不在任何 RA 中）：",
    length(ild_only_shared), "条\n")
if (length(ild_only_shared) > 0) {
  for (s in ild_only_shared) cat("   ", s, "\n")
}

# ====================================================================
# 补充统计：基于 Top K
# ====================================================================
cat("\n========================================================================\n")
cat("补充：基于 Top K =", K, "的更广泛分析\n")
cat("========================================================================\n\n")

topK_sets <- lapply(all_cdr3, function(x) x$top_K)

# 1'. 所有样本共有
all_shared_K <- Reduce(intersect, topK_sets)
cat("1'. 在所有样本中共有的 clone（Top K）：", length(all_shared_K), "条\n")

# 2'. RA 专属共有
raK_shared <- Reduce(intersect, topK_sets[ra_samples])
ildK_all <- unique(unlist(topK_sets[ild_samples]))
ra_only_shared_K <- setdiff(raK_shared, ildK_all)
cat("2'. 只在所有 RA 中共有的 clone（Top K）：", length(ra_only_shared_K), "条\n")

# 3'. ILD 专属共有
ildK_shared <- Reduce(intersect, topK_sets[ild_samples])
raK_all <- unique(unlist(topK_sets[ra_samples]))
ild_only_shared_K <- setdiff(ildK_shared, raK_all)
cat("3'. 只在所有 ILD 中共有的 clone（Top K）：", length(ild_only_shared_K), "条\n")

# ====================================================================
# 更多维度统计
# ====================================================================
cat("\n========================================================================\n")
cat("更多维度统计\n")
cat("========================================================================\n\n")

# A/B/C
cat("A. 所有 RA 样本共有的 clone（Top M）：", length(ra_shared_clones), "条\n")
cat("B. 所有 ILD 样本共有的 clone（Top M）：", length(ild_shared_clones), "条\n")
ra_ild_common <- intersect(ra_shared_clones, ild_shared_clones)
cat("C. RA全共有 ∩ ILD全共有（Top M）：", length(ra_ild_common), "条\n")

# D. clone 出现频率分布
cat("\nD. Top M clone 在样本中的出现频率分布：\n")
count_table <- table(clone_nsamples)
for (n in sort(as.numeric(names(count_table)), decreasing = TRUE)) {
  cat(sprintf("   出现在 %3d 个样本中的 clone: %6d 条\n",
              n, count_table[as.character(n)]))
}

# E. cohort 层面统计
cat("\nE. 仅出现在 RA vs 仅出现在 ILD（Top M）：\n")
ra_union_M <- unique(unlist(lapply(ra_samples, function(sid) topM_sets[[sid]])))
ild_union_M <- unique(unlist(lapply(ild_samples, function(sid) topM_sets[[sid]])))
ra_unique_any <- setdiff(ra_union_M, ild_union_M)
ild_unique_any <- setdiff(ild_union_M, ra_union_M)
both_any <- intersect(ra_union_M, ild_union_M)
cat(sprintf("   仅 RA 中出现的 clone: %d 条\n", length(ra_unique_any)))
cat(sprintf("   仅 ILD 中出现的 clone: %d 条\n", length(ild_unique_any)))
cat(sprintf("   RA/ILD 共有的 clone: %d 条\n", length(both_any)))

# ====================================================================
# 按 material 分层统计
# ====================================================================
cat("\n========================================================================\n")
cat("按 material 分层统计（PBMC vs buffycoat）\n")
cat("========================================================================\n\n")

pbmc_samples <- sample_info$libraryid[sample_info$material == "PBMC"]
bc_samples   <- sample_info$libraryid[sample_info$material == "buffycoat"]

cat("PBMC 样本数:", length(pbmc_samples), "\n")
cat("buffycoat 样本数:", length(bc_samples), "\n")

if (length(pbmc_samples) > 1) {
  pbmc_shared <- Reduce(intersect, topM_sets[pbmc_samples])
  cat("所有 PBMC 样本共有的 clone（Top M）：", length(pbmc_shared), "条\n")
}
if (length(bc_samples) > 1) {
  bc_shared <- Reduce(intersect, topM_sets[bc_samples])
  cat("所有 buffycoat 样本共有的 clone（Top M）：", length(bc_shared), "条\n")
}

cat("\n========================================================================\n")
cat("分析完成！\n")
cat("========================================================================\n")

# 保存结果
result_dir <- "./TRB/result/"
out_file <- paste0(result_dir, "CDR3_AA_overlap_K", K, "_M", M, ".txt")
sink(out_file)
cat("CDR3 AA Clone 重叠分析结果\n")
cat("参数: K =", K, ", M =", M, "\n")
cat("分析日期:", Sys.Date(), "\n\n")
cat("RA 样本数:", length(ra_samples), "\n")
cat("ILD 样本数:", length(ild_samples), "\n")
cat("总样本数:", length(aa_ids), "\n\n")
cat("1. 所有样本共有的 clone（Top M）：", length(all_shared), "条\n")
cat("2. 只在所有 RA 中共有的 clone（Top M）：", length(ra_only_shared), "条\n")
cat("3. 只在所有 ILD 中共有的 clone（Top M）：", length(ild_only_shared), "条\n")
cat("\n--- 出现频率分布 ---\n")
for (n in sort(as.numeric(names(count_table)), decreasing = TRUE)) {
  cat(sprintf("出现在 %3d 个样本中的 clone: %6d 条\n", n, count_table[as.character(n)]))
}
sink()

cat("\n结果已保存至:", out_file, "\n")
