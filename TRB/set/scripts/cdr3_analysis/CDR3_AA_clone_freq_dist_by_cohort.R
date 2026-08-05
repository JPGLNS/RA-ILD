library(ggplot2)

setwd("/data/users/chenhaisheng/RA-ILD/")

# ====================================================================
# CDR3 AA Clone 出现频率分布分析（按队列）
# 功能：统计每个 clone 在 RA/ILD 队列内出现的样本数分布，
#       输出私有 clone、低频/中频/高频 clone 的比例汇总与分布图
# 输入：TRB/result/01_AA_clone_table/*_AA_clone_table.csv（按频率降序）
# 参数：K = 12000（每样本保留前 K 条），M = 8000（取前 M 条统计）
# 输出：TRB/gradient/clone_freq_dist/AA_clone_freq_dist_RA/ILD_*.png|pdf
# ====================================================================

K <- 12000
M <- 8000

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

# 分组
sample_info <- data.frame(libraryid = aa_ids, stringsAsFactors = FALSE)
sample_info <- merge(sample_info, metadata[, c("libraryid", "material", "cohort", "patient")],
                     by = "libraryid")
rownames(sample_info) <- sample_info$libraryid

ra_samples  <- sample_info$libraryid[sample_info$cohort == "RA"]
ild_samples <- sample_info$libraryid[sample_info$cohort == "ILD"]

cat("RA 样本数:", length(ra_samples), "\n")
cat("ILD 样本数:", length(ild_samples), "\n\n")

# ====================================================================
# 函数：统计 clone 在指定样本集内的出现频率分布
# ====================================================================
calc_freq_dist <- function(clone_sets, sample_ids, label) {
  # clone_sets: named list of CDR3 AA vectors (already subset to this cohort)
  sets <- clone_sets[sample_ids]
  n_total <- length(sets)

  # 展开
  clone_vec <- unlist(sets)
  sample_vec <- rep(names(sets), times = sapply(sets, length))
  clone_to_samples <- split(sample_vec, clone_vec)
  clone_n <- sapply(clone_to_samples, function(x) length(unique(x)))

  cat("========================================================================\n")
  cat(" ", label, "（共", n_total, "个样本）\n")
  cat("========================================================================\n")
  cat("  唯一 clone 总数:", length(clone_n), "\n")
  cat("  Top M 总 clone 次:", sum(sapply(sets, length)), "\n\n")

  # 频率分布表
  count_tab <- table(clone_n)
  max_n <- max(clone_n)

  cat("  出现样本数 | clone 数量 | 占该队列样本比例\n")
  cat("  -----------|-----------|----------------\n")
  for (n in sort(as.numeric(names(count_tab)), decreasing = TRUE)) {
    pct <- n / n_total * 100
    cat(sprintf("  %5d (%.1f%%) | %8d\n", n, pct, count_tab[as.character(n)]))
  }

  # 汇总区间
  cat("\n  --- 汇总区间 ---\n")

  # 仅1个样本 = 私有 clone
  n_private <- sum(clone_n == 1)
  cat(sprintf("  私有 clone（仅1个样本）: %d (%.1f%%)\n",
              n_private, n_private / length(clone_n) * 100))

  # 2-10% 样本
  lo10 <- floor(n_total * 0.10)
  n_low <- sum(clone_n >= 2 & clone_n <= lo10)
  cat(sprintf("  出现 2-%d 个样本 (≤10%%): %d (%.1f%%)\n",
              lo10, n_low, n_low / length(clone_n) * 100))

  # 10-25%
  lo25 <- floor(n_total * 0.25)
  n_mid1 <- sum(clone_n > lo10 & clone_n <= lo25)
  cat(sprintf("  出现 %d-%d 个样本 (10-25%%): %d (%.1f%%)\n",
              lo10 + 1, lo25, n_mid1, n_mid1 / length(clone_n) * 100))

  # 25-50%
  lo50 <- floor(n_total * 0.50)
  n_mid2 <- sum(clone_n > lo25 & clone_n <= lo50)
  cat(sprintf("  出现 %d-%d 个样本 (25-50%%): %d (%.1f%%)\n",
              lo25 + 1, lo50, n_mid2, n_mid2 / length(clone_n) * 100))

  # 50-75%
  lo75 <- floor(n_total * 0.75)
  n_high1 <- sum(clone_n > lo50 & clone_n <= lo75)
  cat(sprintf("  出现 %d-%d 个样本 (50-75%%): %d (%.1f%%)\n",
              lo50 + 1, lo75, n_high1, n_high1 / length(clone_n) * 100))

  # 75-100%
  n_high2 <- sum(clone_n > lo75 & clone_n < n_total)
  cat(sprintf("  出现 %d-%d 个样本 (75-100%%): %d (%.1f%%)\n",
              lo75 + 1, n_total - 1, n_high2, n_high2 / length(clone_n) * 100))

  # 100% = 所有样本共有
  n_all <- sum(clone_n == n_total)
  cat(sprintf("  公共 clone（所有 %d 样本）: %d (%.1f%%)\n",
              n_total, n_all, n_all / length(clone_n) * 100))

  invisible(list(
    clone_n = clone_n,
    count_tab = count_tab,
    n_total = n_total
  ))
}

# ====================================================================
# 执行分析
# ====================================================================

# RA 队列
ra_res <- calc_freq_dist(all_cdr3, ra_samples, "RA 队列")

# ILD 队列
ild_res <- calc_freq_dist(all_cdr3, ild_samples, "ILD 队列")

# ====================================================================
# 可视化：RA 和 ILD 各自一张图
#   横坐标 = 出现样本数区间（具体数值）
#   纵坐标 = clone 数 / 该队列唯一 clone 总数 × 100
# ====================================================================
cat("\n\n正在生成图表...\n")

plot_dir <- "./TRB/gradient/clone_freq_dist/"
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

# ---- 通用函数：按样本数区间汇总，返回画图数据 ----
build_plot_data <- function(clone_n, n_total, breaks_vec, label) {
  # breaks_vec: e.g. c(0,1,2,5,10,20,40,60,80,Inf)
  #   注意从 0 开始，(0,1]=1个样本, (1,2]=2个样本, (2,5]=3-5个样本 ...
  n_uniq <- length(clone_n)

  bin_labels <- c()
  for (i in 1:(length(breaks_vec) - 1)) {
    lo <- breaks_vec[i]
    hi <- breaks_vec[i + 1]
    if (is.infinite(hi)) {
      bin_labels <- c(bin_labels, paste0("≥", lo + 1))
    } else if (hi - lo == 1) {
      # 单个数值
      bin_labels <- c(bin_labels, as.character(hi))
    } else {
      bin_labels <- c(bin_labels, paste0(lo + 1, "-", hi))
    }
  }

  bin <- cut(clone_n, breaks = breaks_vec, labels = bin_labels,
             include.lowest = TRUE, right = TRUE)
  tab <- table(bin)

  df <- data.frame(
    bin  = factor(names(tab), levels = bin_labels),
    count = as.numeric(tab),
    pct   = as.numeric(tab) / n_uniq * 100,
    cohort = label,
    stringsAsFactors = FALSE
  )
  df
}

# ---- RA 分 bin（108 个样本，最大出现 79 次）----
ra_breaks <- c(0, 1, 2, 5, 10, 20, 40, 60, 80, Inf)
ra_plot <- build_plot_data(ra_res$clone_n, ra_res$n_total, ra_breaks, "RA")
cat("\nRA 分bin 数据：\n")
print(ra_plot)

# ---- ILD 分 bin（66 个样本，最大出现 48 次）----
ild_breaks <- c(0, 1, 2, 5, 10, 20, 30, 40, 50, Inf)
ild_plot <- build_plot_data(ild_res$clone_n, ild_res$n_total, ild_breaks, "ILD")
cat("\nILD 分bin 数据：\n")
print(ild_plot)

# ---- 通用画图函数 ----
make_plot <- function(df, n_total, n_uniq, cohort_label, bar_color) {
  ggplot(df, aes(x = bin, y = pct)) +
    geom_bar(stat = "identity", fill = bar_color, width = 0.65) +
    geom_text(aes(label = ifelse(count == 0, "0",
                                 paste0(scales::comma(count), "\n(", sprintf("%.2f", pct), "%)"))),
              vjust = -0.15, size = 3.2, lineheight = 0.9) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
    labs(x = "出现在多少个样本中",
         y = "占该队列唯一 Clone 总数的百分比 (%)",
         title = paste0(cohort_label, " 队列：CDR3 AA Clone 跨样本出现频率"),
         subtitle = paste0("K = ", K, "（过滤前", K, "条）, M = ", M, "（取前", M, "条统计） | ",
                           n_total, " 个样本 | 唯一 Clone 总数 = ", scales::comma(n_uniq))) +
    theme_classic(base_size = 14) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
          plot.subtitle = element_text(hjust = 0.5, size = 10),
          axis.text = element_text(color = "black"),
          axis.text.x = element_text(size = 12),
          axis.title = element_text(size = 13))
}

# ---- 生成 RA 图 ----
p_ra <- make_plot(ra_plot, ra_res$n_total, length(ra_res$clone_n),
                  "RA", "#4DBBD5")
ggsave(paste0(plot_dir, "AA_clone_freq_dist_RA_K", K, "_M", M, ".png"),
       p_ra, width = 10, height = 6, dpi = 300)
ggsave(paste0(plot_dir, "AA_clone_freq_dist_RA_K", K, "_M", M, ".pdf"),
       p_ra, width = 10, height = 6)
cat("RA 图已保存\n")

# ---- 生成 ILD 图 ----
p_ild <- make_plot(ild_plot, ild_res$n_total, length(ild_res$clone_n),
                   "ILD", "#E64B35")
ggsave(paste0(plot_dir, "AA_clone_freq_dist_ILD_K", K, "_M", M, ".png"),
       p_ild, width = 10, height = 6, dpi = 300)
ggsave(paste0(plot_dir, "AA_clone_freq_dist_ILD_K", K, "_M", M, ".pdf"),
       p_ild, width = 10, height = 6)
cat("ILD 图已保存\n")

cat("\n所有图表已保存至:", plot_dir, "\n")
cat("分析完成！\n")
