library(ggplot2)

setwd("/data/users/chenhaisheng/RA-ILD/")

# ====================================================================
# CDR3 AA Clone 特异性分析 — 4 分类版本
# 条件：K=12000, M=10000
# 阈值 T ∈ {10%, 15%}，组间差异 Δ ∈ {5%, 10%}
#
# 4 分类规则：
#   RA-enrich:  p_RA ≥ T 且 p_RA - p_ILD ≥ Δ
#   ILD-enrich: p_ILD ≥ T 且 p_ILD - p_RA ≥ Δ
#   Shared:     p_RA ≥ T 且 p_ILD ≥ T 且 |p_RA - p_ILD| < Δ
#   Ambiguous:  仅一组 ≥ T，且 |p_RA - p_ILD| < Δ
# ====================================================================

K <- 12000
M <- 10000
THRESHOLD_PCTS <- c(10, 15)
DELTAS <- c(5, 10)

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

ra_samples  <- sample_info$libraryid[sample_info$cohort == "RA"]
ild_samples <- sample_info$libraryid[sample_info$cohort == "ILD"]
n_ra  <- length(ra_samples)
n_ild <- length(ild_samples)
cat("RA:", n_ra, "样本  ILD:", n_ild, "样本\n\n")

# ---- 构建 clone 出现矩阵 ----
ra_clone_vec <- unlist(all_cdr3[ra_samples])
ra_sample_vec <- rep(names(all_cdr3[ra_samples]),
                     times = sapply(all_cdr3[ra_samples], length))
ra_clone_to_samples <- split(ra_sample_vec, ra_clone_vec)
ra_clone_n <- sapply(ra_clone_to_samples, function(x) length(unique(x)))

ild_clone_vec <- unlist(all_cdr3[ild_samples])
ild_sample_vec <- rep(names(all_cdr3[ild_samples]),
                      times = sapply(all_cdr3[ild_samples], length))
ild_clone_to_samples <- split(ild_sample_vec, ild_clone_vec)
ild_clone_n <- sapply(ild_clone_to_samples, function(x) length(unique(x)))

ra_count <- setNames(as.numeric(ra_clone_n), names(ra_clone_n))
ild_count <- setNames(as.numeric(ild_clone_n), names(ild_clone_n))

# ---- 对每个 (T, Δ) 组合进行分析 ----
plot_dir <- "./TRB/gradient/4class/"
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

# 颜色方案（4 类）
four_colors <- c("RA-enrich"  = "#C62828",
                 "ILD-enrich" = "#1565C0",
                 "Shared"     = "#7CB342",
                 "Ambiguous"  = "#9E9E9E")

all_summaries <- list()

for (T_pct in THRESHOLD_PCTS) {
  ra_t  <- ceiling(n_ra  * T_pct / 100)
  ild_t <- ceiling(n_ild * T_pct / 100)

  for (d in DELTAS) {

    # 候选 clone：至少在一边 ≥ 阈值
    candidates <- union(
      names(ra_clone_n)[ra_clone_n >= ra_t],
      names(ild_clone_n)[ild_clone_n >= ild_t]
    )

    base <- data.frame(
      cdr3_aa = candidates,
      ra_n  = ifelse(is.na(ra_count[candidates]), 0, ra_count[candidates]),
      ild_n = ifelse(is.na(ild_count[candidates]), 0, ild_count[candidates]),
      stringsAsFactors = FALSE
    )
    base$ra_pct  <- base$ra_n  / n_ra  * 100
    base$ild_pct <- base$ild_n / n_ild * 100
    base$delta   <- base$ra_pct - base$ild_pct

    ra_ok  <- base$ra_pct  >= T_pct
    ild_ok <- base$ild_pct >= T_pct
    large_diff <- abs(base$delta) >= d

    # 5 分类
    base$category <- NA_character_

    # 1. RA-enrich: RA ≥ T 且 RA - ILD ≥ Δ
    ra_enrich_idx <- ra_ok & base$delta >= d
    base$category[ra_enrich_idx] <- "RA-enrich"

    # 2. ILD-enrich: ILD ≥ T 且 ILD - RA ≥ Δ
    ild_enrich_idx <- ild_ok & base$delta <= -d
    base$category[ild_enrich_idx] <- "ILD-enrich"

    # 3. Shared: 两组均 ≥ T 且 |diff| < Δ
    shared_idx <- ra_ok & ild_ok & abs(base$delta) < d & is.na(base$category)
    base$category[shared_idx] <- "Shared"

    # 4. Ambiguous: 仅一组 ≥ T，且 |diff| < Δ
    ambig_idx <- is.na(base$category) &
                 ((ra_ok & !ild_ok) | (!ra_ok & ild_ok)) &
                 abs(base$delta) < d
    base$category[ambig_idx] <- "Ambiguous"

    # 剩余未分类的（理论上不应有，除非两边都不达标但被 candidates 漏掉了）
    leftover <- is.na(base$category)
    if (any(leftover)) {
      cat("  WARNING:", sum(leftover), "unclassified clones\n")
      base$category[leftover] <- "Ambiguous"
    }

    # 排序
    base$category <- factor(base$category,
                            levels = c("RA-enrich", "ILD-enrich",
                                       "Shared", "Ambiguous"))
    base <- base[order(base$category, -base$delta), ]

    # ---- 汇总 ----
    counts <- table(base$category)
    key <- paste0("T", T_pct, "_D", d)
    all_summaries[[key]] <- list(
      T = T_pct, D = d,
      ra_t = ra_t, ild_t = ild_t,
      n_total = nrow(base),
      counts = counts
    )

    # ---- 输出 ----
    cat("========================================================================\n")
    cat(sprintf("  T = %d%%  (RA≥%d, ILD≥%d)  |  Δ = %d%%  |  候选: %d 条\n",
                T_pct, ra_t, ild_t, d, nrow(base)))
    cat("========================================================================\n\n")

    for (cat_name in levels(base$category)) {
      n_cat <- sum(base$category == cat_name)
      cat(sprintf("  %-15s %5d 条 (%5.1f%%)\n", cat_name, n_cat,
                  n_cat / nrow(base) * 100))
    }
    cat("\n")

    # Top 展示
    for (cat_name in c("RA-enrich", "ILD-enrich", "Shared")) {
      sub <- base[base$category == cat_name, ]
      n_show <- min(10, nrow(sub))
      if (n_show == 0) next
      cat(sprintf("  ── %s Top %d ──\n", cat_name, n_show))
      cat(sprintf("  %-25s %8s %8s %9s %9s %8s\n",
                  "CDR3 AA", "RA(n)", "ILD(n)", "RA%", "ILD%", "Delta%"))
      for (i in 1:n_show) {
        cat(sprintf("  %-25s %8d %8d %8.1f%% %8.1f%% %+7.1f%%\n",
                    sub$cdr3_aa[i], sub$ra_n[i], sub$ild_n[i],
                    sub$ra_pct[i], sub$ild_pct[i], sub$delta[i]))
      }
      cat("\n")
    }

    # ---- 散点图 ----
    p1 <- ggplot(base, aes(x = ra_pct, y = ild_pct, color = category)) +
      geom_point(alpha = 0.65, size = 2) +
      geom_abline(slope = 1, intercept = d,
                  linetype = "dashed", color = "#C62828", linewidth = 0.4) +
      geom_abline(slope = 1, intercept = -d,
                  linetype = "dashed", color = "#1565C0", linewidth = 0.4) +
      geom_abline(slope = 1, intercept = 0,
                  linetype = "dotted", color = "grey50", linewidth = 0.3) +
      geom_vline(xintercept = T_pct, linetype = "dotted", color = "grey70", linewidth = 0.3) +
      geom_hline(yintercept = T_pct, linetype = "dotted", color = "grey70", linewidth = 0.3) +
      scale_color_manual(values = four_colors, name = "类别",
                         drop = FALSE) +
      labs(x = paste0("RA 出现率 (% , n=", n_ra, ")"),
           y = paste0("ILD 出现率 (% , n=", n_ild, ")"),
           title = paste0("CDR3 AA 队列特异性 (K=", K, ", M=", M,
                          ", T=", T_pct, "%, Δ=", d, "%)"),
           subtitle = paste0("RA-enrich:", counts["RA-enrich"],
                             " | ILD-enrich:", counts["ILD-enrich"],
                             " | Shared:", counts["Shared"],
                             " | Ambiguous:", counts["Ambiguous"])) +
      theme_classic(base_size = 14) +
      theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
            plot.subtitle = element_text(hjust = 0.5, size = 9),
            axis.text = element_text(color = "black"),
            legend.position = "bottom")

    ggsave(paste0(plot_dir, "AA_4class_scatter_K", K, "_M", M,
                  "_T", T_pct, "_D", d, ".png"),
           p1, width = 9, height = 8.5, dpi = 300)

    # ---- 柱状图 ----
    count_df <- data.frame(
      category = factor(names(counts), levels = levels(base$category)),
      count = as.numeric(counts)
    )
    count_df <- count_df[count_df$count > 0, ]

    p2 <- ggplot(count_df, aes(x = category, y = count, fill = category)) +
      geom_bar(stat = "identity", width = 0.55) +
      geom_text(aes(label = count), vjust = -0.3, size = 4.5, fontface = "bold") +
      scale_fill_manual(values = four_colors, guide = "none") +
      scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
      labs(x = "", y = "Clone 数量",
           title = paste0("CDR3 AA 4分类统计 (K=", K, ", M=", M,
                          ", T=", T_pct, "%, Δ=", d, "%)")) +
      theme_classic(base_size = 14) +
      theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
            axis.text = element_text(color = "black", size = 11),
            axis.text.x = element_text(size = 12, face = "bold",
                                       angle = 20, hjust = 1))

    ggsave(paste0(plot_dir, "AA_4class_bar_K", K, "_M", M,
                  "_T", T_pct, "_D", d, ".png"),
           p2, width = 7.5, height = 5.5, dpi = 300)

    cat(sprintf("  T=%d%% Δ=%d%% 图已保存\n\n", T_pct, d))
  }
}

# ---- 最终汇总表格 ----
cat("================================================================================\n")
cat(" 全部组合汇总\n")
cat("================================================================================\n\n")

cat(sprintf("%-18s %10s %10s %10s %10s %8s\n",
            "参数", "RA-enrich", "ILD-enrich", "Shared", "Ambiguous", "合计"))
cat(sprintf("%-18s %10s %10s %10s %10s %8s\n",
            "------------------", "----------", "----------",
            "----------", "----------", "------"))

for (key in names(all_summaries)) {
  s <- all_summaries[[key]]
  label <- paste0("T=", s$T, "%, Δ=", s$D, "%")
  c <- s$counts
  cat(sprintf("%-18s %10d %10d %10d %10d %8d\n",
              label, c["RA-enrich"], c["ILD-enrich"],
              c["Shared"], c["Ambiguous"], s$n_total))
}

cat("\n所有分析完成！图表保存至:", plot_dir, "\n")
