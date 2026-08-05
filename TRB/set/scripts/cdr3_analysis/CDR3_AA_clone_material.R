library(ggplot2)

setwd("/data/users/chenhaisheng/RA-ILD/")

# ====================================================================
# AA Clone material 间比较（AA 水平）
# 功能：基于 AA clone table 比较 PBMC vs buffycoat（含 cohort 分层）：
#       共享 clone 比例的差异曲线 + 按 top N 的梯度图
# 输入：TRB/result/01_AA_clone_table/*_AA_clone_table.csv + metadata
#       （与 TRB/origin_result/ NT 样本取交集）
# 输出：TRB/gradient/clone_material/AA_PBMC+buffycoat*.png、
#       AA_diff_curve_*.png
# ====================================================================

metadata <- read.csv("./TRB/metadata.csv")

# AA clone table 目录
aa_dir <- "./TRB/result/01_AA_clone_table/"
aa_files <- list.files(aa_dir, pattern = "_AA_clone_table\\.csv$")

cat("AA clone table 文件数:", length(aa_files), "\n")

# 提取 libraryid
aa_library_ids <- gsub("_AA_clone_table\\.csv$", "", aa_files)

# 与 NT 样本取交集（只保留两者共有的样本）
nt_dir <- "./TRB/origin_result/"
nt_files <- list.files(nt_dir, pattern = "_TRB_CDR3_NT_frequency_error_correct\\.csv$")
nt_library_ids <- gsub("_TRB_CDR3_NT_frequency_error_correct\\.csv$", "", nt_files)

common_ids <- intersect(aa_library_ids, nt_library_ids)
cat("NT 样本数:", length(nt_library_ids), "\n")
cat("AA 样本数:", length(aa_library_ids), "\n")
cat("交集样本数:", length(common_ids), "\n")

# 过滤：仅用交集样本
aa_files_filtered <- paste0(common_ids, "_AA_clone_table.csv")

# ====================================================================
# 读取所有 AA clone table，提取 frequency 列
# ====================================================================
cat("\n正在读取 AA clone table...\n")

all_aa_freqs <- lapply(file.path(aa_dir, aa_files_filtered), function(f) {
  data <- read.csv(f, stringsAsFactors = FALSE)
  # frequency 列是该 AA clone 的细胞频率（%），按降序排列
  sort(data$frequency, decreasing = TRUE)
})
names(all_aa_freqs) <- common_ids
cat("完成！共读取", length(all_aa_freqs), "个样本。\n")

# ====================================================================
# 基础统计
# ====================================================================
aa_clone_counts <- sapply(all_aa_freqs, length)

sample_info <- data.frame(
  libraryid = names(all_aa_freqs),
  aa_clone_count = aa_clone_counts,
  stringsAsFactors = FALSE
)
sample_info <- merge(sample_info, metadata[, c("libraryid", "material", "cohort")], by = "libraryid")
sample_info$group_two  <- sample_info$material
sample_info$group_four <- paste(sample_info$material, sample_info$cohort, sep = "-")

cat("\n========== AA Clone 数统计 ==========\n")
cat(sprintf("总体: 均值=%.0f, 范围=%d-%d\n",
            mean(aa_clone_counts), min(aa_clone_counts), max(aa_clone_counts)))

cat("\n--- 两组 ---\n")
print(aggregate(aa_clone_count ~ group_two, data = sample_info,
                FUN = function(x) c(mean = round(mean(x)), min = min(x), max = max(x))))

cat("\n--- 四组 ---\n")
print(aggregate(aa_clone_count ~ group_four, data = sample_info,
                FUN = function(x) c(mean = round(mean(x)), min = min(x), max = max(x))))

cat("\n========== AA 频率数据就绪，可进入后续分析 ==========\n")

# ====================================================================
# 第一部分：AA Clone 数分布箱线图
# ====================================================================
cat("\n\n########## AA Clone 数分布 ##########\n")

# 两组 stats
s2 <- aggregate(aa_clone_count ~ group_two, data = sample_info,
                FUN = function(x) c(mean = mean(x), min = min(x), max = max(x)))
s2 <- data.frame(
  group_two = s2$group_two,
  mean = s2$aa_clone_count[, "mean"],
  min  = s2$aa_clone_count[, "min"],
  max  = s2$aa_clone_count[, "max"]
)
s2$label <- paste0(round(s2$mean, 0), " (", round(s2$min, 0), "-", round(s2$max, 0), ")")

# 四组 stats
s4 <- aggregate(aa_clone_count ~ group_four, data = sample_info,
                FUN = function(x) c(mean = mean(x), min = min(x), max = max(x)))
s4 <- data.frame(
  group_four = s4$group_four,
  mean = s4$aa_clone_count[, "mean"],
  min  = s4$aa_clone_count[, "min"],
  max  = s4$aa_clone_count[, "max"]
)
s4$label <- paste0(round(s4$mean, 0), "\n(", round(s4$min, 0), "-", round(s4$max, 0), ")")

s2$y_pos <- s2$max + diff(range(aa_clone_counts)) * 0.08
s4$y_pos <- s4$max + diff(range(aa_clone_counts)) * 0.08

# 图1：两组
p1 <- ggplot(sample_info, aes(x = group_two, y = aa_clone_count, fill = group_two)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.6, width = 0.5) +
  geom_jitter(width = 0.1, size = 1.2, alpha = 0.5) +
  geom_text(data = s2, aes(x = group_two, y = y_pos, label = label),
            inherit.aes = FALSE, size = 3.5, fontface = "bold", color = "black") +
  scale_fill_manual(values = c("PBMC" = "#E64B35", "buffycoat" = "#4DBBD5")) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.18))) +
  labs(x = "Material", y = "Number of AA CDR3 Clones") +
  ggtitle("AA CDR3 Clone Count Distribution: PBMC vs Buffycoat") +
  theme_classic(base_size = 14) +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, face = "bold", size = 10),
    axis.text = element_text(color = "black"),
    axis.ticks = element_line(color = "black")
  )
ggsave("./TRB/gradient/clone_material/AA_PBMC+buffycoat.png", p1, width = 6, height = 5, dpi = 300)

# 图2：四组
p2 <- ggplot(sample_info, aes(x = group_four, y = aa_clone_count, fill = group_four)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.6, width = 0.5) +
  geom_jitter(width = 0.1, size = 1.2, alpha = 0.5) +
  geom_text(data = s4, aes(x = group_four, y = y_pos, label = label),
            inherit.aes = FALSE, size = 3.5, fontface = "bold", color = "black") +
  scale_fill_manual(values = c(
    "PBMC-RA" = "#E64B35", "PBMC-ILD" = "#F39B7F",
    "buffycoat-RA" = "#4DBBD5", "buffycoat-ILD" = "#7EC8E3"
  )) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.18))) +
  labs(x = "Group", y = "Number of AA CDR3 Clones") +
  ggtitle("AA CDR3 Clone Count Distribution by Material and Cohort") +
  theme_classic(base_size = 14) +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, face = "bold", size = 10),
    axis.text = element_text(color = "black"),
    axis.ticks = element_line(color = "black")
  )
ggsave("./TRB/gradient/clone_material/AA_PBMC+buffycoat+cohort.png", p2, width = 7, height = 5, dpi = 300)

cat("  -> 已保存: ./TRB/gradient/clone_material/AA_PBMC+buffycoat.png\n")
cat("  -> 已保存: ./TRB/gradient/clone_material/AA_PBMC+buffycoat+cohort.png\n")

# ====================================================================
# 第二部分：AA 梯度分析 + 差值曲线
# ====================================================================
cat("\n\n########## AA 梯度分析：Top N clone 累积频率 ##########\n")

gradient_dir <- "./TRB/gradient/clone_material/"
dir.create(gradient_dir, showWarnings = FALSE, recursive = TRUE)

gradient_values <- c(500, 1000, 1500, 2000, 3000, 5000, 8000, 12000, 16000)
cat("梯度值：", paste(gradient_values, collapse = ", "), "\n")

diff_data <- data.frame(
  top_n = integer(), PBMC_mean = numeric(), buffycoat_mean = numeric(),
  PBMC_RA_mean = numeric(), PBMC_ILD_mean = numeric(),
  buffycoat_RA_mean = numeric(), buffycoat_ILD_mean = numeric(),
  stringsAsFactors = FALSE
)

for (top_n in gradient_values) {
  cum_freq <- sapply(all_aa_freqs, function(freqs) {
    n <- min(top_n, length(freqs))
    sum(freqs[1:n])
  })
  top_data <- data.frame(
    libraryid = names(cum_freq),
    cum_frequency = as.numeric(cum_freq),
    stringsAsFactors = FALSE
  )
  top_data <- merge(top_data, metadata[, c("libraryid", "material", "cohort")], by = "libraryid")
  top_data$group_two  <- top_data$material
  top_data$group_four <- paste(top_data$material, top_data$cohort, sep = "-")

  means2 <- aggregate(cum_frequency ~ group_two, data = top_data, FUN = mean)
  means4 <- aggregate(cum_frequency ~ group_four, data = top_data, FUN = mean)

  diff_data <- rbind(diff_data, data.frame(
    top_n = top_n,
    PBMC_mean = means2$cum_frequency[means2$group_two == "PBMC"],
    buffycoat_mean = means2$cum_frequency[means2$group_two == "buffycoat"],
    PBMC_RA_mean = means4$cum_frequency[means4$group_four == "PBMC-RA"],
    PBMC_ILD_mean = means4$cum_frequency[means4$group_four == "PBMC-ILD"],
    buffycoat_RA_mean = means4$cum_frequency[means4$group_four == "buffycoat-RA"],
    buffycoat_ILD_mean = means4$cum_frequency[means4$group_four == "buffycoat-ILD"],
    stringsAsFactors = FALSE
  ))
  cat(sprintf("  Top %s 完成\n", top_n))
}

diff_data$diff_two <- diff_data$PBMC_mean - diff_data$buffycoat_mean
diff_data$diff_RA  <- diff_data$PBMC_RA_mean - diff_data$buffycoat_RA_mean
diff_data$diff_ILD <- diff_data$PBMC_ILD_mean - diff_data$buffycoat_ILD_mean

cat("\n--- AA 差值数据 ---\n")
cat("TopN\tPBMC-BC\tRA-diff\tILD-diff\n")
for (i in 1:nrow(diff_data)) {
  cat(sprintf("%d\t%.4f\t%.4f\t%.4f\n",
      diff_data$top_n[i], diff_data$diff_two[i],
      diff_data$diff_RA[i], diff_data$diff_ILD[i]))
}

# ====================================================================
# 差值曲线图1：单线 — PBMC - buffycoat
# ====================================================================
stagger_vjust <- function(i) ifelse(seq_along(i) %% 2 == 1, -2.2, 3.2)

p_diff1 <- ggplot(diff_data, aes(x = top_n, y = diff_two)) +
  geom_line(color = "#333333", linewidth = 1) +
  geom_point(color = "#333333", size = 2.5) +
  geom_text(aes(label = sprintf("%.2f", diff_two)),
            vjust = stagger_vjust(diff_data$diff_two),
            size = 3.2, color = "#333333", fontface = "bold") +
  scale_x_continuous(breaks = diff_data$top_n) +
  scale_y_continuous(expand = expansion(mult = c(0.15, 0.28))) +
  labs(x = "Top N Clones", y = "Mean Cumulative Frequency Difference (%)") +
  ggtitle("AA: PBMC - Buffycoat Cumulative Frequency Difference") +
  theme_classic(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 10),
    axis.text = element_text(color = "black"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.ticks = element_line(color = "black"),
    panel.grid.major.y = element_line(color = "grey90", linewidth = 0.3)
  )
ggsave(paste0(gradient_dir, "AA_diff_curve_PBMC_vs_buffycoat.png"),
       p_diff1, width = 8, height = 5, dpi = 300)
cat("  -> 已保存: AA_diff_curve_PBMC_vs_buffycoat.png\n")

# ====================================================================
# 差值曲线图2：双线 — RA + ILD
# ====================================================================
diff_long <- data.frame(
  top_n = rep(diff_data$top_n, 2),
  cohort_type = c(rep("RA", nrow(diff_data)), rep("ILD", nrow(diff_data))),
  diff_value = c(diff_data$diff_RA, diff_data$diff_ILD)
)
diff_long$label_vjust <- stagger_vjust(diff_data$diff_RA)

cohort_colors <- c("RA" = "#E64B35", "ILD" = "#4DBBD5")
cohort_linetypes <- c("RA" = "solid", "ILD" = "dashed")

p_diff2 <- ggplot(diff_long, aes(x = top_n, y = diff_value,
                                  color = cohort_type, linetype = cohort_type)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2.5) +
  geom_text(aes(label = sprintf("%.2f", diff_value),
                color = cohort_type, vjust = label_vjust),
            size = 3.2, fontface = "bold", show.legend = FALSE) +
  scale_x_continuous(breaks = diff_data$top_n) +
  scale_y_continuous(expand = expansion(mult = c(0.15, 0.28))) +
  scale_color_manual(values = cohort_colors) +
  scale_linetype_manual(values = cohort_linetypes) +
  labs(x = "Top N Clones", y = "Mean Cumulative Frequency Difference (%)",
       color = "Cohort", linetype = "Cohort") +
  ggtitle("AA: PBMC - Buffycoat Cumulative Frequency Difference by Cohort") +
  theme_classic(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 10),
    axis.text = element_text(color = "black"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.ticks = element_line(color = "black"),
    panel.grid.major.y = element_line(color = "grey90", linewidth = 0.3),
    legend.position = c(0.88, 0.88),
    legend.background = element_rect(color = "grey80", linewidth = 0.3)
  )
ggsave(paste0(gradient_dir, "AA_diff_curve_by_cohort.png"),
       p_diff2, width = 8, height = 5, dpi = 300)
cat("  -> 已保存: AA_diff_curve_by_cohort.png\n")

cat("\n========== AA 分析全部完成！==========\n")
