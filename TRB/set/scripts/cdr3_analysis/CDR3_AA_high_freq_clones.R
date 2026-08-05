library(ggplot2)

setwd("/data/users/chenhaisheng/RA-ILD/")

# ====================================================================
# CDR3 AA Clone 高频统计
# 条件：K=12000, M=10000
# 阈值：10% 和 20%
# ====================================================================

K <- 12000
M <- 10000
THRESHOLD_PCTS <- c(10, 20)

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

cat("RA 样本数:", length(ra_samples), "\n")
cat("ILD 样本数:", length(ild_samples), "\n\n")

# ====================================================================
# 分析函数（可指定阈值百分比）
# ====================================================================
analyze_high_freq_clones <- function(clone_sets, sample_ids, cohort_label, threshold_pct) {
  sets <- clone_sets[sample_ids]
  n_total <- length(sets)
  threshold_n <- ceiling(n_total * threshold_pct / 100)

  clone_vec <- unlist(sets)
  sample_vec <- rep(names(sets), times = sapply(sets, length))
  clone_to_samples <- split(sample_vec, clone_vec)
  clone_n <- sapply(clone_to_samples, function(x) length(unique(x)))

  high_freq <- clone_n[clone_n >= threshold_n]
  high_freq <- sort(high_freq, decreasing = TRUE)

  n_uniq <- length(clone_n)
  n_high <- length(high_freq)

  cat("    ", threshold_pct, "% 阈值: ≥", threshold_n, " 个样本 → ",
      n_high, " 条高频 Clone (", sprintf("%.4f", n_high / n_uniq * 100), "%)\n", sep = "")

  invisible(list(
    clone_n = clone_n,
    high_freq = high_freq,
    n_total = n_total,
    threshold_n = threshold_n,
    threshold_pct = threshold_pct,
    n_uniq = n_uniq,
    n_high = n_high
  ))
}

# ====================================================================
# 批量执行：每个队列 × 每个阈值
# ====================================================================
all_results <- list()

for (cohort_name in c("RA", "ILD")) {
  samples <- if (cohort_name == "RA") ra_samples else ild_samples
  cat("========================================================================\n")
  cat(" ", cohort_name, "队列（", length(samples), " 个样本）\n", sep = "")
  cat("========================================================================\n")

  # 先计算 clone_n（共享，避免重复计算）
  sets <- all_cdr3[samples]
  clone_vec <- unlist(sets)
  sample_vec <- rep(names(sets), times = sapply(sets, length))
  clone_to_samples <- split(sample_vec, clone_vec)
  clone_n <- sapply(clone_to_samples, function(x) length(unique(x)))
  n_uniq <- length(clone_n)
  cat("  唯一 Clone 总数:", scales::comma(n_uniq), "\n\n")

  for (tp in THRESHOLD_PCTS) {
    threshold_n <- ceiling(length(samples) * tp / 100)
    high_freq <- clone_n[clone_n >= threshold_n]
    high_freq <- sort(high_freq, decreasing = TRUE)

    key <- paste0(cohort_name, "_", tp, "pct")
    all_results[[key]] <- list(
      cohort = cohort_name,
      threshold_pct = tp,
      threshold_n = threshold_n,
      n_total = length(samples),
      n_uniq = n_uniq,
      n_high = length(high_freq),
      high_freq = high_freq
    )

    cat("   ", tp, "% 阈值: ≥", threshold_n, " 样本 → ",
        length(high_freq), " 条 (",
        sprintf("%.4f", length(high_freq) / n_uniq * 100), "%)\n", sep = "")
  }
  cat("\n")
}

# ====================================================================
# 汇总对比表
# ====================================================================
cat("========================================================================\n")
cat(" 汇总对比\n")
cat("========================================================================\n\n")

# 表头
header <- sprintf("%-12s %8s %8s %12s %12s %12s",
                  "队列", "阈值%", "阈值N", "唯一Clone", "高频Clone", "占比")
cat(header, "\n")
cat(paste(rep("-", nchar(header)), collapse = ""), "\n")

for (key in names(all_results)) {
  r <- all_results[[key]]
  cat(sprintf("%-12s %7d%% %8d %12s %12s %11.4f%%\n",
              r$cohort, r$threshold_pct, r$threshold_n,
              scales::comma(r$n_uniq), scales::comma(r$n_high),
              r$n_high / r$n_uniq * 100))
}

# 交集
cat("\n--- RA/ILD 高频 Clone 交集 ---\n")
for (tp in THRESHOLD_PCTS) {
  ra_key <- paste0("RA_", tp, "pct")
  ild_key <- paste0("ILD_", tp, "pct")
  ra_seqs <- names(all_results[[ra_key]]$high_freq)
  ild_seqs <- names(all_results[[ild_key]]$high_freq)
  shared <- intersect(ra_seqs, ild_seqs)
  cat(sprintf("  %d%% 阈值: RA %d 条, ILD %d 条, 交集 %d 条\n",
              tp, length(ra_seqs), length(ild_seqs), length(shared)))
}

# ====================================================================
# 20% 阈值详细列表
# ====================================================================
for (tp in THRESHOLD_PCTS) {
  for (cohort_name in c("RA", "ILD")) {
    key <- paste0(cohort_name, "_", tp, "pct")
    r <- all_results[[key]]
    hf <- r$high_freq
    if (length(hf) == 0) next
    cat("\n--- ", cohort_name, " ", tp, "% 阈值 (≥", r$threshold_n,
        " 样本) 高频 Clone: ", length(hf), " 条 ---\n", sep = "")
    cat(sprintf("  %-25s %8s %10s\n", "CDR3 AA", "样本数", "占队列%"))
    cat(sprintf("  %-25s %8s %10s\n", "-------------------------", "------", "------"))
    for (i in seq_along(hf)) {
      cat(sprintf("  %-25s %8d %9.1f%%\n",
                  names(hf)[i], hf[i], hf[i] / r$n_total * 100))
    }
  }
}

# ====================================================================
# 可视化：仅画 10% 阈值的 Top 30
# ====================================================================
cat("\n\n正在生成图表...\n")

plot_dir <- "./TRB/gradient/high_freq/"
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

plot_high_freq <- function(high_freq, n_total, threshold_n, threshold_pct,
                           cohort_label, bar_color) {
  if (length(high_freq) == 0) return(NULL)

  top_n <- min(30, length(high_freq))
  hf_sub <- high_freq[1:top_n]

  df <- data.frame(
    cdr3_aa = as.character(names(hf_sub)),
    n_samples = as.numeric(hf_sub),
    stringsAsFactors = FALSE
  )
  df <- df[order(df$n_samples, decreasing = TRUE), ]
  df$cdr3_aa <- factor(df$cdr3_aa, levels = df$cdr3_aa)
  df$pct_label <- paste0(df$n_samples, " (", sprintf("%.1f", df$n_samples / n_total * 100), "%)")
  df$label_short <- ifelse(nchar(as.character(df$cdr3_aa)) > 18,
                           paste0(substr(as.character(df$cdr3_aa), 1, 16), ".."),
                           as.character(df$cdr3_aa))

  subtitle_text <- paste0("K=", K, ", M=", M, " | ", n_total, " 样本 | 共 ",
                          length(high_freq), " 条 (≥", threshold_pct, "%) | 图中仅展示前 ", top_n)

  p <- ggplot(df, aes(x = label_short, y = n_samples)) +
    geom_bar(stat = "identity", fill = bar_color, width = 0.7) +
    geom_text(aes(label = pct_label), vjust = -0.3, size = 3.0) +
    geom_hline(yintercept = threshold_n, linetype = "dashed", color = "grey40", linewidth = 0.5) +
    annotate("text", x = nrow(df) * 0.85, y = threshold_n,
             label = paste0(threshold_pct, "% 阈值 (", threshold_n, " 样本)"),
             vjust = -0.6, size = 3.5, color = "grey40") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(x = "CDR3 AA 序列（前 30）",
         y = "出现的样本数",
         title = paste0(cohort_label, ": ≥", threshold_pct, "% 样本高频 CDR3 AA Clone (Top ", top_n, ")"),
         subtitle = subtitle_text) +
    theme_classic(base_size = 14) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
          plot.subtitle = element_text(hjust = 0.5, size = 10),
          axis.text = element_text(color = "black"),
          axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
          axis.title = element_text(size = 13))

  ggsave(paste0(plot_dir, "AA_high_freq_", cohort_label, "_", threshold_pct, "pct_K", K, "_M", M, ".png"),
         p, width = 12, height = 6, dpi = 300)
  ggsave(paste0(plot_dir, "AA_high_freq_", cohort_label, "_", threshold_pct, "pct_K", K, "_M", M, ".pdf"),
         p, width = 12, height = 6)

  cat("  ", cohort_label, " ", threshold_pct, "% 图已保存\n", sep = "")
}

for (tp in THRESHOLD_PCTS) {
  for (cohort_name in c("RA", "ILD")) {
    key <- paste0(cohort_name, "_", tp, "pct")
    r <- all_results[[key]]
    bar_color <- if (cohort_name == "RA") "#4DBBD5" else "#E64B35"
    plot_high_freq(r$high_freq, r$n_total, r$threshold_n, r$threshold_pct,
                   cohort_name, bar_color)
  }
}

cat("\n所有分析完成！图表保存至:", plot_dir, "\n")
