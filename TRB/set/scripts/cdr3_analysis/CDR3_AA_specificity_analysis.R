library(ggplot2)

setwd("/data/users/chenhaisheng/RA-ILD/")

# ====================================================================
# CDR3 AA Clone 特异性分析
# 条件：K=12000, M=10000, 阈值 ≥10%
# 筛选：clone 至少在 RA 或 ILD 中出现 ≥10%
# 分类：
#   RA-specific:  RA出现率 - ILD出现率 > 5%
#   ILD-specific: ILD出现率 - RA出现率 > 5%
#   Shared:       |RA出现率 - ILD出现率| ≤ 5%
# ====================================================================

K <- 12000
M <- 8000
THRESHOLD_PCT <- 15
SPECIFICITY_DELTAS <- c(5, 10)  # 两个差值阈值

metadata <- read.csv("./TRB/metadata.csv")

aa_dir <- "./TRB/result/01_AA_clone_table/"
aa_files <- list.files(aa_dir, pattern = "_AA_clone_table\\.csv$")
aa_ids <- gsub("_AA_clone_table\\.csv$", "", aa_files)
aa_ids <- aa_ids[!grepl("^01_", aa_ids)]
aa_files_f <- paste0(aa_ids, "_AA_clone_table.csv")

cat("正在读取数据... (K=", K, ", M=", M, ")\n", sep = "")

all_cdr3 <- lapply(file.path(aa_dir, aa_files_f), function(f) {
  data <- read.csv(f, stringsAsFactors = FALSE)
  n_keep <- min(K, nrow(data))
  head(data$cdr3_aa[1:n_keep], min(M, n_keep))
})
names(all_cdr3) <- aa_ids

sample_info <- data.frame(libraryid = aa_ids, stringsAsFactors = FALSE)
sample_info <- merge(sample_info, metadata[, c("libraryid", "material", "cohort", "patient")],
                     by = "libraryid")
rownames(sample_info) <- sample_info$libraryid

ra_samples  <- sample_info$libraryid[sample_info$cohort == "RA"]
ild_samples <- sample_info$libraryid[sample_info$cohort == "ILD"]

n_ra  <- length(ra_samples)
n_ild <- length(ild_samples)

cat("RA 样本数:", n_ra, "\n")
cat("ILD 样本数:", n_ild, "\n\n")

# ====================================================================
# 构建 clone × sample 出现矩阵
# ====================================================================
cat("构建 clone 出现矩阵...\n")

# RA 侧
ra_clone_vec <- unlist(all_cdr3[ra_samples])
ra_sample_vec <- rep(names(all_cdr3[ra_samples]),
                     times = sapply(all_cdr3[ra_samples], length))
ra_clone_to_samples <- split(ra_sample_vec, ra_clone_vec)
ra_clone_n <- sapply(ra_clone_to_samples, function(x) length(unique(x)))

# ILD 侧
ild_clone_vec <- unlist(all_cdr3[ild_samples])
ild_sample_vec <- rep(names(all_cdr3[ild_samples]),
                      times = sapply(all_cdr3[ild_samples], length))
ild_clone_to_samples <- split(ild_sample_vec, ild_clone_vec)
ild_clone_n <- sapply(ild_clone_to_samples, function(x) length(unique(x)))

# 阈值
ra_threshold  <- ceiling(n_ra  * THRESHOLD_PCT / 100)
ild_threshold <- ceiling(n_ild * THRESHOLD_PCT / 100)

cat("RA  阈值: ≥", ra_threshold,  " / ", n_ra,  " 样本\n", sep = "")
cat("ILD 阈值: ≥", ild_threshold, " / ", n_ild, " 样本\n\n", sep = "")

# ====================================================================
# 筛选：至少在 RA 或 ILD 中 ≥10%
# ====================================================================
all_clones <- union(names(ra_clone_n), names(ild_clone_n))

# 对每个候选 clone 计算 RA 出现次数和 ILD 出现次数
cat("计算所有 clone 的 RA/ILD 出现率... (共", scales::comma(length(all_clones)), "个候选)\n")

# 用 named vector 快速查找
ra_count <- setNames(as.numeric(ra_clone_n), names(ra_clone_n))
ild_count <- setNames(as.numeric(ild_clone_n), names(ild_clone_n))

# 筛选
candidate_clones <- all_clones[
  (!is.na(ra_count[all_clones]) & ra_count[all_clones] >= ra_threshold) |
  (!is.na(ild_count[all_clones]) & ild_count[all_clones] >= ild_threshold)
]

cat("至少在一个队列中 ≥", THRESHOLD_PCT, "% 的 clone: ", length(candidate_clones), "\n\n", sep = "")

# ====================================================================
# 计算基础数据（RA/ILD 出现率）
# ====================================================================
base <- data.frame(
  cdr3_aa = candidate_clones,
  ra_n = ifelse(is.na(ra_count[candidate_clones]), 0, ra_count[candidate_clones]),
  ild_n = ifelse(is.na(ild_count[candidate_clones]), 0, ild_count[candidate_clones]),
  stringsAsFactors = FALSE
)
base$ra_pct  <- base$ra_n  / n_ra  * 100
base$ild_pct <- base$ild_n / n_ild * 100
base$delta    <- base$ra_pct - base$ild_pct
base <- base[order(base$delta, decreasing = TRUE), ]

# ====================================================================
# 对每个 Delta 分别分类和展示
# ====================================================================
plot_dir <- "./TRB/gradient/specificity/"
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

for (d in SPECIFICITY_DELTAS) {
  cat("========================================================================\n")
  cat(" Δ =", d, "%\n")
  cat("========================================================================\n\n")

  base$category <- ifelse(base$delta > d, "RA-specific",
                   ifelse(base$delta < -d, "ILD-specific", "Shared"))

  n_ra_sp  <- sum(base$category == "RA-specific")
  n_ild_sp <- sum(base$category == "ILD-specific")
  n_shared <- sum(base$category == "Shared")

  cat(sprintf("  RA-specific  (RA%% - ILD%% > %d%%):  %d 条\n", d, n_ra_sp))
  cat(sprintf("  ILD-specific (ILD%% - RA%% > %d%%):  %d 条\n", d, n_ild_sp))
  cat(sprintf("  Shared       (|RA%% - ILD%%| ≤ %d%%): %d 条\n", d, n_shared))
  cat(sprintf("  ─────────────────────────────\n"))
  cat(sprintf("  合计: %d 条\n\n", nrow(base)))

  # RA-specific Top
  ra_sub <- base[base$category == "RA-specific", ]
  if (nrow(ra_sub) > 0) {
    cat(sprintf("  === RA-specific Top %d ===\n", min(15, nrow(ra_sub))))
    cat(sprintf("  %-25s %8s %8s %9s %9s %8s\n",
                "CDR3 AA", "RA(n)", "ILD(n)", "RA%", "ILD%", "Delta%"))
    for (i in 1:min(15, nrow(ra_sub))) {
      cat(sprintf("  %-25s %8d %8d %8.1f%% %8.1f%% %+7.1f%%\n",
                  ra_sub$cdr3_aa[i], ra_sub$ra_n[i], ra_sub$ild_n[i],
                  ra_sub$ra_pct[i], ra_sub$ild_pct[i], ra_sub$delta[i]))
    }
    cat("\n")
  }

  # ILD-specific Top
  ild_sub <- base[base$category == "ILD-specific", ]
  if (nrow(ild_sub) > 0) {
    cat(sprintf("  === ILD-specific Top %d ===\n", min(15, nrow(ild_sub))))
    cat(sprintf("  %-25s %8s %8s %9s %9s %8s\n",
                "CDR3 AA", "RA(n)", "ILD(n)", "RA%", "ILD%", "Delta%"))
    for (i in 1:min(15, nrow(ild_sub))) {
      cat(sprintf("  %-25s %8d %8d %8.1f%% %8.1f%% %+7.1f%%\n",
                  ild_sub$cdr3_aa[i], ild_sub$ra_n[i], ild_sub$ild_n[i],
                  ild_sub$ra_pct[i], ild_sub$ild_pct[i], ild_sub$delta[i]))
    }
    cat("\n")
  }

  # 图1：散点图
  base$cat_factor <- factor(base$category,
                             levels = c("RA-specific", "Shared", "ILD-specific"))

  p1 <- ggplot(base, aes(x = ra_pct, y = ild_pct, color = cat_factor)) +
    geom_point(alpha = 0.6, size = 2) +
    geom_abline(slope = 1, intercept = d,
                linetype = "dashed", color = "#E64B35", linewidth = 0.5) +
    geom_abline(slope = 1, intercept = -d,
                linetype = "dashed", color = "#4DBBD5", linewidth = 0.5) +
    geom_abline(slope = 1, intercept = 0,
                linetype = "dotted", color = "grey50", linewidth = 0.3) +
    scale_color_manual(values = c("RA-specific" = "#E64B35",
                                   "ILD-specific" = "#4DBBD5",
                                   "Shared" = "grey60"),
                       name = "类别") +
    labs(x = paste0("RA 出现率 (% , n=", n_ra, ")"),
         y = paste0("ILD 出现率 (% , n=", n_ild, ")"),
         title = paste0("CDR3 AA Clone 队列特异性 (K=", K, ", M=", M,
                        ", 阈值 ≥", THRESHOLD_PCT, "%)"),
         subtitle = paste0("RA-specific: ", n_ra_sp, " | Shared: ", n_shared,
                           " | ILD-specific: ", n_ild_sp, " | Δ > ", d, "%")) +
    theme_classic(base_size = 14) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
          plot.subtitle = element_text(hjust = 0.5, size = 10),
          axis.text = element_text(color = "black"),
          legend.position = "bottom")

  ggsave(paste0(plot_dir, "AA_specificity_scatter_K", K, "_M", M,
                "_T", THRESHOLD_PCT, "_D", d, ".png"),
         p1, width = 8, height = 8, dpi = 300)

  # 图2：柱状图
  count_df <- data.frame(
    category = factor(c("RA-specific", "Shared", "ILD-specific"),
                       levels = c("RA-specific", "Shared", "ILD-specific")),
    count = c(n_ra_sp, n_shared, n_ild_sp)
  )

  p2 <- ggplot(count_df, aes(x = category, y = count, fill = category)) +
    geom_bar(stat = "identity", width = 0.55) +
    geom_text(aes(label = count), vjust = -0.3, size = 5, fontface = "bold") +
    scale_fill_manual(values = c("RA-specific" = "#E64B35",
                                  "ILD-specific" = "#4DBBD5",
                                  "Shared" = "grey60"),
                      guide = "none") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
    labs(x = "", y = "Clone 数量",
         title = paste0("CDR3 AA Clone 特异性分类 (K=", K, ", M=", M, ")"),
         subtitle = paste0("筛选: ≥", THRESHOLD_PCT, "% 样本中出现 | Δ > ", d, "%")) +
    theme_classic(base_size = 14) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
          plot.subtitle = element_text(hjust = 0.5, size = 10),
          axis.text = element_text(color = "black", size = 12),
          axis.text.x = element_text(size = 14, face = "bold"))

  ggsave(paste0(plot_dir, "AA_specificity_bar_K", K, "_M", M,
                "_T", THRESHOLD_PCT, "_D", d, ".png"),
         p2, width = 7, height = 6, dpi = 300)

  cat(sprintf("  Δ=%d%% 图已保存\n", d))
}

cat("\n所有分析完成！图表保存至:", plot_dir, "\n")
