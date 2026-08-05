library(ggplot2)
library(gridExtra)
library(ggsignif)

setwd("/data/users/chenhaisheng/RA-ILD/")

# ====================================================================
# TRB AA Clone Shannon Diversity 分析
# 对每个 K：取 top K 条 AA clone，重新归一化后计算 Shannon Diversity
# 1 行 × 3 列 (Total / RA / ILD)，PBMC vs buffycoat boxplot + Wilcoxon
# ====================================================================

metadata <- read.csv("./TRB/metadata.csv")

aa_dir <- "./TRB/result/01_AA_clone_table/"
aa_files <- list.files(aa_dir, pattern = "_AA_clone_table\\.csv$")
aa_ids <- gsub("_AA_clone_table\\.csv$", "", aa_files)
aa_ids <- aa_ids[!grepl("^01_", aa_ids)]

cat("正在读取 AA clone table...\n")
all_freqs <- lapply(file.path(aa_dir, paste0(aa_ids, "_AA_clone_table.csv")), function(f) {
  data <- read.csv(f, stringsAsFactors = FALSE)
  sort(data$frequency, decreasing = TRUE)
})
names(all_freqs) <- aa_ids
cat("完成！共读取", length(all_freqs), "个样本。\n")

# 样本分组
sample_info <- data.frame(libraryid = aa_ids, stringsAsFactors = FALSE)
sample_info <- merge(sample_info, metadata[, c("libraryid", "material", "cohort")], by = "libraryid")

ra_ids  <- sample_info$libraryid[sample_info$cohort == "RA"]
ild_ids <- sample_info$libraryid[sample_info$cohort == "ILD"]

cat(sprintf("Total: %d (PBMC=%d, buffycoat=%d)\n",
            nrow(sample_info),
            sum(sample_info$material == "PBMC"),
            sum(sample_info$material == "buffycoat")))
cat(sprintf("RA: %d  ILD: %d\n", length(ra_ids), length(ild_ids)))

# ---- 参数 ----
K_values <- c(6000, 8000, 10000, 11000, 12000, 13000, 14000, 15000, 16000)

# ---- 输出目录 ----
out_dir <- "./TRB/gradient/diversity/"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---- 颜色 ----
material_colors <- c("PBMC" = "#E64B35", "buffycoat" = "#4DBBD5")

# ====================================================================
# 辅助函数
# ====================================================================
calc_shannon <- function(freqs, K) {
  n <- min(K, length(freqs))
  if (n == 0) return(NA_real_)
  pool <- freqs[1:n]
  p <- pool / sum(pool)
  -sum(p * log(p))
}

# ---- 通用 boxplot 函数（含 Wilcoxon 检验）----
make_boxplot <- function(df, group_label) {
  df$group <- df$material

  # summary stats
  stats <- aggregate(shannon ~ group, data = df,
                     FUN = function(x) c(mean = mean(x), min = min(x), max = max(x)))
  stats <- data.frame(
    group = stats$group,
    mean  = stats$shannon[, "mean"],
    min   = stats$shannon[, "min"],
    max   = stats$shannon[, "max"]
  )
  stats$label <- sprintf("%.4f\n(%.4f-%.4f)", stats$mean, stats$min, stats$max)
  stats$y_pos <- stats$max + diff(range(df$shannon, na.rm = TRUE)) * 0.08

  # Wilcoxon rank-sum test
  pb_vals <- df$shannon[df$group == "PBMC"]
  bc_vals <- df$shannon[df$group == "buffycoat"]
  wt <- wilcox.test(pb_vals, bc_vals, exact = FALSE)
  p_val <- wt$p.value

  if (p_val < 0.0001) {
    p_label <- "p < 0.0001"
  } else {
    p_label <- sprintf("p = %.4f", p_val)
  }

  y_range <- diff(range(df$shannon, na.rm = TRUE))
  bracket_y <- max(df$shannon, na.rm = TRUE) + y_range * 0.22

  ggplot(df, aes(x = group, y = shannon, fill = group)) +
    geom_boxplot(outlier.shape = NA, alpha = 0.6, width = 0.5) +
    geom_jitter(width = 0.1, size = 0.8, alpha = 0.4) +
    geom_text(data = stats, aes(x = group, y = y_pos, label = label),
              inherit.aes = FALSE, size = 2.8, fontface = "bold", color = "black") +
    geom_signif(comparisons = list(c("PBMC", "buffycoat")),
                annotations = p_label,
                y_position = bracket_y, tip_length = 0.01,
                textsize = 3, vjust = -0.2, color = "black") +
    scale_fill_manual(values = material_colors) +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.30))) +
    labs(x = "", y = "Shannon Diversity", title = group_label) +
    theme_classic(base_size = 12) +
    theme(
      legend.position = "none",
      plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
      axis.text = element_text(color = "black", size = 10),
      axis.title.y = element_text(size = 11)
    )
}

# ====================================================================
# 主循环
# ====================================================================
for (K in K_values) {
  cat(sprintf("\n========== K = %d ==========\n", K))

  metrics <- do.call(rbind, lapply(names(all_freqs), function(sid) {
    data.frame(libraryid = sid,
               shannon = calc_shannon(all_freqs[[sid]], K),
               stringsAsFactors = FALSE)
  }))
  metrics <- merge(metrics, sample_info, by = "libraryid")

  ra_metrics  <- metrics[metrics$libraryid %in% ra_ids, ]
  ild_metrics <- metrics[metrics$libraryid %in% ild_ids, ]

  p_tot <- make_boxplot(metrics,     "Total")
  p_ra  <- make_boxplot(ra_metrics,  "RA")
  p_ild <- make_boxplot(ild_metrics, "ILD")

  png(file.path(out_dir, sprintf("diversity_K%d.png", K)),
      width = 10, height = 4, units = "in", res = 300)

  grid.arrange(
    p_tot, p_ra, p_ild,
    ncol = 3, nrow = 1,
    top = grid::textGrob(
      sprintf("TRB AA CDR3 Shannon Diversity (K = %d)", K),
      gp = grid::gpar(fontsize = 14, fontface = "bold")
    )
  )

  dev.off()
  cat(sprintf("  -> diversity_K%d.png\n", K))
}

cat("\n========== 全部完成！", out_dir, "==========\n")
