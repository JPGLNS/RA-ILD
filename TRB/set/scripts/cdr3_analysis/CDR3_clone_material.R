library(ggplot2)

setwd("/data/users/chenhaisheng/RA-ILD/")

# ====================================================================
# CDR3 Clone 数分布与 material 梯度分析（NT 水平，非 AA）
# 功能：统计各样本 clone 总数在 PBMC/buffycoat（×cohort）组的分布
#       与差异检验；按 top N 取 clone 绘制 material 间梯度对比图
# 输入：TRB/origin_result/*_TRB_CDR3_NT_frequency_error_correct.csv + metadata
# 输出：TRB/gradient/clone_material/PBMC+buffycoat*.png（含 top N 梯度系列）
# ====================================================================

# 读取metadata
metadata <- read.csv("./TRB/metadata.csv")

# 获取origin_result文件夹中所有CSV文件
result_dir <- "./TRB/origin_result/"
files <- list.files(result_dir, pattern = "_TRB_CDR3_NT_frequency_error_correct\\.csv$")

# 提取libraryid，统计每个文件的clone数（行数）
clone_counts <- data.frame(
  filename = files,
  libraryid = gsub("_TRB_CDR3_NT_frequency_error_correct\\.csv$", "", files),
  clone_count = sapply(file.path(result_dir, files), function(f) length(readLines(f))),
  stringsAsFactors = FALSE
)

# 与metadata合并，获取material和cohort信息
clone_counts <- merge(clone_counts, metadata[, c("libraryid", "material", "cohort")], by = "libraryid")

# 创建分组变量
clone_counts$group_two <- clone_counts$material
clone_counts$group_four <- paste(clone_counts$material, clone_counts$cohort, sep = "-")

cat("各组样本数:\n")
cat("\n--- 两组 ---\n")
print(table(clone_counts$group_two))
cat("\n--- 四组 ---\n")
print(table(clone_counts$group_four))

# 计算两组summary stats
stats_two <- aggregate(clone_count ~ group_two, data = clone_counts,
                       FUN = function(x) c(mean = mean(x), min = min(x), max = max(x)))
stats_two <- data.frame(
  group_two = stats_two$group_two,
  mean = stats_two$clone_count[, "mean"],
  min  = stats_two$clone_count[, "min"],
  max  = stats_two$clone_count[, "max"]
)
stats_two$label <- paste0(round(stats_two$mean, 0), " (", round(stats_two$min, 0), "-", round(stats_two$max, 0), ")")

# 计算四组summary stats
stats_four <- aggregate(clone_count ~ group_four, data = clone_counts,
                        FUN = function(x) c(mean = mean(x), min = min(x), max = max(x)))
stats_four <- data.frame(
  group_four = stats_four$group_four,
  mean = stats_four$clone_count[, "mean"],
  min  = stats_four$clone_count[, "min"],
  max  = stats_four$clone_count[, "max"]
)
stats_four$label <- paste0(round(stats_four$mean, 0), "\n(", round(stats_four$min, 0), "-", round(stats_four$max, 0), ")")

stats_two$y_pos  <- stats_two$max  + diff(range(clone_counts$clone_count)) * 0.08
stats_four$y_pos <- stats_four$max + diff(range(clone_counts$clone_count)) * 0.08

cat("\n--- 两组统计 ---\n")
print(stats_two[, c("group_two", "label")])
cat("\n--- 四组统计 ---\n")
print(stats_four[, c("group_four", "label")])

# ========== 图1：两组（PBMC vs buffycoat）==========
p1 <- ggplot(clone_counts, aes(x = group_two, y = clone_count, fill = group_two)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.6, width = 0.5) +
  geom_jitter(width = 0.1, size = 1.2, alpha = 0.5) +
  geom_text(data = stats_two, aes(x = group_two, y = y_pos, label = label),
            inherit.aes = FALSE, size = 3.5, fontface = "bold", color = "black") +
  scale_fill_manual(values = c("PBMC" = "#E64B35", "buffycoat" = "#4DBBD5")) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.18))) +
  labs(x = "Material", y = "Number of CDR3 Clones") +
  ggtitle("CDR3 Clone Count Distribution: PBMC vs Buffycoat") +
  theme_classic(base_size = 14) +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, face = "bold", size = 10),
    axis.text = element_text(color = "black"),
    axis.ticks = element_line(color = "black")
  )

ggsave("./TRB/gradient/clone_material/PBMC+buffycoat.png", p1, width = 6, height = 5, dpi = 300)

# ========== 图2：四组（PBMC-RA, PBMC-ILD, buffycoat-RA, buffycoat-ILD）==========
p2 <- ggplot(clone_counts, aes(x = group_four, y = clone_count, fill = group_four)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.6, width = 0.5) +
  geom_jitter(width = 0.1, size = 1.2, alpha = 0.5) +
  geom_text(data = stats_four, aes(x = group_four, y = y_pos, label = label),
            inherit.aes = FALSE, size = 3.5, fontface = "bold", color = "black") +
  scale_fill_manual(values = c(
    "PBMC-RA" = "#E64B35",
    "PBMC-ILD" = "#F39B7F",
    "buffycoat-RA" = "#4DBBD5",
    "buffycoat-ILD" = "#7EC8E3"
  )) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.18))) +
  labs(x = "Group", y = "Number of CDR3 Clones") +
  ggtitle("CDR3 Clone Count Distribution by Material and Cohort") +
  theme_classic(base_size = 14) +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, face = "bold", size = 10),
    axis.text = element_text(color = "black"),
    axis.ticks = element_line(color = "black")
  )

ggsave("./TRB/gradient/clone_material/PBMC+buffycoat+cohort.png", p2, width = 7, height = 5, dpi = 300)

cat("\n图片已保存至:\n")
cat("  ./TRB/gradient/clone_material/PBMC+buffycoat.png\n")
cat("  ./TRB/gradient/clone_material/PBMC+buffycoat+cohort.png\n")

# ====================================================================
# 第二部分：梯度分析 — Top N clone 累积频率
# ====================================================================
cat("\n\n########## 梯度分析：Top N clone 累积频率 ##########\n")

gradient_dir <- "./TRB/gradient/clone_material/"
dir.create(gradient_dir, showWarnings = FALSE, recursive = TRUE)

cat("\n正在读取所有文件的频率数据...\n")
all_freqs <- lapply(file.path(result_dir, files), function(f) {
  data <- read.table(f, sep = "\t", header = FALSE, stringsAsFactors = FALSE,
                     colClasses = c("NULL", "NULL", "NULL", "numeric"))
  sort(data[, 1], decreasing = TRUE)
})
names(all_freqs) <- gsub("_TRB_CDR3_NT_frequency_error_correct\\.csv$", "", files)
cat("完成！共读取", length(all_freqs), "个文件。\n")

# 梯度：前端密集，后端稀疏（方案C）
gradient_values <- c(500, 1000, 1500, 2000, 3000, 5000, 8000, 12000, 20000)
cat("梯度值：", paste(gradient_values, collapse = ", "), "\n")

# 用于收集各梯度下的组均值，供后续画差值曲线
diff_data <- data.frame(
  top_n = integer(),
  PBMC_mean = numeric(),
  buffycoat_mean = numeric(),
  PBMC_RA_mean = numeric(),
  PBMC_ILD_mean = numeric(),
  buffycoat_RA_mean = numeric(),
  buffycoat_ILD_mean = numeric(),
  stringsAsFactors = FALSE
)

for (top_n in gradient_values) {
  cat("\n========== 处理 Top", top_n, "==========\n")

  cum_freq <- sapply(all_freqs, function(freqs) {
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

  # --- 两组 stats ---
  s2 <- aggregate(cum_frequency ~ group_two, data = top_data,
                  FUN = function(x) c(mean = mean(x), min = min(x), max = max(x)))
  s2 <- data.frame(
    group_two = s2$group_two,
    mean = s2$cum_frequency[, "mean"],
    min  = s2$cum_frequency[, "min"],
    max  = s2$cum_frequency[, "max"]
  )
  s2$label <- paste0(round(s2$mean, 4), "\n(", round(s2$min, 4), "-", round(s2$max, 4), ")")
  s2$y_pos <- s2$max + diff(range(top_data$cum_frequency)) * 0.08

  # --- 四组 stats ---
  s4 <- aggregate(cum_frequency ~ group_four, data = top_data,
                  FUN = function(x) c(mean = mean(x), min = min(x), max = max(x)))
  s4 <- data.frame(
    group_four = s4$group_four,
    mean = s4$cum_frequency[, "mean"],
    min  = s4$cum_frequency[, "min"],
    max  = s4$cum_frequency[, "max"]
  )
  s4$label <- paste0(round(s4$mean, 4), "\n(", round(s4$min, 4), "-", round(s4$max, 4), ")")
  s4$y_pos <- s4$max + diff(range(top_data$cum_frequency)) * 0.08

  # 收集组均值
  diff_data <- rbind(diff_data, data.frame(
    top_n = top_n,
    PBMC_mean = s2$mean[s2$group_two == "PBMC"],
    buffycoat_mean = s2$mean[s2$group_two == "buffycoat"],
    PBMC_RA_mean = s4$mean[s4$group_four == "PBMC-RA"],
    PBMC_ILD_mean = s4$mean[s4$group_four == "PBMC-ILD"],
    buffycoat_RA_mean = s4$mean[s4$group_four == "buffycoat-RA"],
    buffycoat_ILD_mean = s4$mean[s4$group_four == "buffycoat-ILD"],
    stringsAsFactors = FALSE
  ))

  # --- 两组箱线图 ---
  p_top2 <- ggplot(top_data, aes(x = group_two, y = cum_frequency, fill = group_two)) +
    geom_boxplot(outlier.shape = NA, alpha = 0.6, width = 0.5) +
    geom_jitter(width = 0.1, size = 1.2, alpha = 0.5) +
    geom_text(data = s2, aes(x = group_two, y = y_pos, label = label),
              inherit.aes = FALSE, size = 3.5, fontface = "bold", color = "black") +
    scale_fill_manual(values = c("PBMC" = "#E64B35", "buffycoat" = "#4DBBD5")) +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.18))) +
    labs(x = "Material", y = paste0("Cumulative Frequency (Top ", top_n, " Clones)")) +
    ggtitle(paste0("Top ", top_n, " CDR3 Clone Cumulative Frequency: PBMC vs Buffycoat")) +
    theme_classic(base_size = 14) +
    theme(
      legend.position = "none",
      plot.title = element_text(hjust = 0.5, face = "bold", size = 10),
      axis.text = element_text(color = "black"),
      axis.ticks = element_line(color = "black")
    )
  fname2 <- paste0(gradient_dir, "PBMC+buffycoat_top", top_n, ".png")
  ggsave(fname2, p_top2, width = 6, height = 5, dpi = 300)

  # --- 四组箱线图 ---
  p_top4 <- ggplot(top_data, aes(x = group_four, y = cum_frequency, fill = group_four)) +
    geom_boxplot(outlier.shape = NA, alpha = 0.6, width = 0.5) +
    geom_jitter(width = 0.1, size = 1.2, alpha = 0.5) +
    geom_text(data = s4, aes(x = group_four, y = y_pos, label = label),
              inherit.aes = FALSE, size = 3.5, fontface = "bold", color = "black") +
    scale_fill_manual(values = c(
      "PBMC-RA" = "#E64B35",
      "PBMC-ILD" = "#F39B7F",
      "buffycoat-RA" = "#4DBBD5",
      "buffycoat-ILD" = "#7EC8E3"
    )) +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.18))) +
    labs(x = "Group", y = paste0("Cumulative Frequency (Top ", top_n, " Clones)")) +
    ggtitle(paste0("Top ", top_n, " CDR3 Clone Cumulative Frequency by Material and Cohort")) +
    theme_classic(base_size = 14) +
    theme(
      legend.position = "none",
      plot.title = element_text(hjust = 0.5, face = "bold", size = 10),
      axis.text = element_text(color = "black"),
      axis.ticks = element_line(color = "black")
    )
  fname4 <- paste0(gradient_dir, "PBMC+buffycoat+cohort_top", top_n, ".png")
  ggsave(fname4, p_top4, width = 7, height = 5, dpi = 300)

  cat("  -> 已保存:", fname2, "\n")
  cat("  -> 已保存:", fname4, "\n")
}

# ====================================================================
# 第三部分：差值曲线图 — PBMC均值 - buffycoat均值 随梯度变化
# ====================================================================
cat("\n\n########## 差值曲线图 ##########\n")

diff_data$diff_two <- diff_data$PBMC_mean - diff_data$buffycoat_mean
diff_data$diff_RA  <- diff_data$PBMC_RA_mean - diff_data$buffycoat_RA_mean
diff_data$diff_ILD <- diff_data$PBMC_ILD_mean - diff_data$buffycoat_ILD_mean

# 错开标签：偶数和奇数交替上下放置，避免前端密集区域重叠
stagger_vjust <- function(i) ifelse(seq_along(i) %% 2 == 1, -2.2, 3.2)

cat("\n--- 差值数据 ---\n")
cat("TopN\tPBMC-BC\tPBMC.RA-BC.RA\tPBMC.ILD-BC.ILD\n")
for (i in 1:nrow(diff_data)) {
  cat(sprintf("%d\t%.4f\t%.4f\t%.4f\n",
      diff_data$top_n[i], diff_data$diff_two[i],
      diff_data$diff_RA[i], diff_data$diff_ILD[i]))
}

# ===== 曲线图1：单线 — PBMC均值 - buffycoat均值 =====
p_diff1 <- ggplot(diff_data, aes(x = top_n, y = diff_two)) +
  geom_line(color = "#333333", linewidth = 1) +
  geom_point(color = "#333333", size = 2.5) +
  geom_text(aes(label = sprintf("%.2f", diff_two)),
            vjust = stagger_vjust(diff_data$diff_two),
            size = 3.2, color = "#333333", fontface = "bold") +
  scale_x_continuous(breaks = diff_data$top_n) +
  scale_y_continuous(expand = expansion(mult = c(0.15, 0.28))) +
  labs(x = "Top N Clones", y = "Mean Cumulative Frequency Difference") +
  ggtitle("PBMC - Buffycoat: Mean Cumulative Frequency Difference") +
  theme_classic(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 10),
    axis.text = element_text(color = "black"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.ticks = element_line(color = "black"),
    panel.grid.major.y = element_line(color = "grey90", linewidth = 0.3)
  )
ggsave(paste0(gradient_dir, "diff_curve_PBMC_vs_buffycoat.png"),
       p_diff1, width = 8, height = 5, dpi = 300)
cat("  -> 已保存:", paste0(gradient_dir, "diff_curve_PBMC_vs_buffycoat.png"), "\n")

# ===== 曲线图2：双线 — RA线和ILD线 =====
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
  labs(x = "Top N Clones", y = "Mean Cumulative Frequency Difference",
       color = "Cohort", linetype = "Cohort") +
  ggtitle("PBMC - Buffycoat: Mean Cumulative Frequency Difference by Cohort") +
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
ggsave(paste0(gradient_dir, "diff_curve_by_cohort.png"),
       p_diff2, width = 8, height = 5, dpi = 300)
cat("  -> 已保存:", paste0(gradient_dir, "diff_curve_by_cohort.png"), "\n")

cat("\n========== 全部完成！==========\n")
cat("梯度箱线图路径:", gradient_dir, "\n")
cat("差值曲线图路径:", gradient_dir, "\n")
