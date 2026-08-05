library(ggplot2)
library(data.table)
library(pROC)

setwd("/data/users/chenhaisheng/RA-ILD/")

# ====================================================================
# pbmc / buffycoat 样本内（非留一法）可行性图：箱线图 + ROC
# 输入：TRB/result/CDR3_AA_sample_dict_hits_<part>_T20_D10.csv
# 输出：TRB/gradient/dict_feasibility/AA_dict_feasibility_boxplot/ROC_<part>_T20_D10.png
# ====================================================================

metrics <- c("RA_dict_clone_count", "ILD_dict_clone_count",
             "RA_dict_hit_rate", "ILD_dict_hit_rate",
             "RA_dict_read_fraction_sum", "ILD_dict_read_fraction_sum",
             "RA_minus_ILD_count", "RA_minus_ILD_freq")
metric_labels <- c("RA 字典命中数", "ILD 字典命中数",
                   "RA 字典命中率", "ILD 字典命中率",
                   "RA 命中 read_fraction 和", "ILD 命中 read_fraction 和",
                   "RA−ILD 命中数差", "RA−ILD read_fraction 和差")

skewness <- function(x) { m <- mean(x); s <- sd(x); mean((x - m)^3) / s^3 }

analyze_metrics <- function(hits_tbl) {
  ra  <- hits_tbl[cohort == "RA"]
  ild <- hits_tbl[cohort == "ILD"]
  out <- list()
  for (i in seq_along(metrics)) {
    m <- metrics[i]
    x_ra  <- ra[[m]]; x_ild <- ild[[m]]
    sw_ra  <- shapiro.test(x_ra); sw_ild <- shapiro.test(x_ild)
    sk_ra  <- skewness(x_ra); sk_ild <- skewness(x_ild)
    biased <- (sw_ra$p.value < 0.05 & abs(sk_ra) > 1) |
              (sw_ild$p.value < 0.05 & abs(sk_ild) > 1)
    main_test <- if (biased) "Mann-Whitney U" else "Welch t"
    t_res <- t.test(x_ra, x_ild)
    u_res <- wilcox.test(x_ra, x_ild)
    roc_obj <- roc(hits_tbl$cohort, hits_tbl[[m]], levels = c("ILD", "RA"),
                   direction = "<", quiet = TRUE)
    out[[i]] <- list(
      metric = m, label = metric_labels[i],
      main_test = main_test,
      t_p = t_res$p.value, u_p = u_res$p.value,
      auc = as.numeric(auc(roc_obj)),
      roc = roc_obj
    )
  }
  main_p <- sapply(out, function(r) if (r$main_test == "Mann-Whitney U") r$u_p else r$t_p)
  for (i in seq_along(out)) out[[i]]$main_p <- main_p[i]
  out
}

plot_dir <- "./TRB/gradient/dict_feasibility/"
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

for (part in c("pbmc", "buffycoat")) {
  hits <- fread(paste0("./TRB/result/CDR3_AA_sample_dict_hits_", part, "_T20_D10.csv"))
  res <- analyze_metrics(hits)
  n_ra  <- sum(hits$cohort == "RA")
  n_ild <- sum(hits$cohort == "ILD")

  # ---- 箱线图（2×4，标注方法与 p 值） ----
  box_df <- melt(hits, id.vars = c("libraryid", "cohort"),
                 measure.vars = metrics, variable.name = "metric", value.name = "value")
  box_df$metric <- factor(box_df$metric, levels = metrics, labels = metric_labels)
  ann_df <- data.frame(
    metric = factor(metric_labels, levels = metric_labels),
    p_label = sapply(res, function(r) {
      sprintf("%s\np = %.2g",
              if (r$main_test == "Mann-Whitney U") "Mann-Whitney U" else "Welch t",
              r$main_p)
    }),
    stringsAsFactors = FALSE
  )
  ann_df$y_pos <- sapply(metrics, function(m) {
    v <- hits[[m]]
    max(v) + (max(v) - min(v)) * 0.12
  })
  p_box <- ggplot(box_df, aes(x = cohort, y = value, fill = cohort)) +
    geom_boxplot(outlier.shape = NA, width = 0.5) +
    geom_jitter(width = 0.18, size = 0.8, alpha = 0.4, colour = "grey30") +
    geom_text(data = ann_df, inherit.aes = FALSE,
              aes(x = 1.5, y = y_pos, label = p_label), size = 3.4, fontface = "bold") +
    facet_wrap(~metric, scales = "free_y", ncol = 4) +
    scale_fill_manual(values = c("RA" = "#0072B2", "ILD" = "#D55E00")) +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.22))) +
    labs(x = NULL, y = "值",
         title = paste0("[", part, "] 样本×字典命中指标 队列分布（T=20%, Δ=10%, ", part, " 字典, 样本内）"),
         subtitle = paste0("蓝色=RA (n=", n_ra, ") | 朱红=ILD (n=", n_ild,
                           ") | 标注为主检验方法与 p 值（非留一法）")) +
    theme_classic(base_size = 13) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
          plot.subtitle = element_text(hjust = 0.5, size = 9.5),
          strip.text = element_text(size = 10))
  ggsave(paste0(plot_dir, "AA_dict_feasibility_boxplot_", part, "_T20_D10.png"), p_box,
         width = 16, height = 8, dpi = 300)

  # ---- ROC 曲线（8 指标一张） ----
  roc_df <- do.call(rbind, lapply(seq_along(metrics), function(i) {
    r <- res[[i]]
    # pROC 坐标反序存储（(1,1)→(0,0)），反转后才是曲线正序，避免并列 FPR 处 TPR 回折
    coord <- coords(r$roc, x = "all", ret = c("specificity", "sensitivity"), transpose = FALSE)
    coord <- coord[rev(seq_len(nrow(coord))), , drop = FALSE]
    data.frame(metric = paste0(r$label, " (AUC ", sprintf("%.3f", r$auc), ")"),
               fpr = 1 - coord$specificity,
               tpr = coord$sensitivity)
  }))
  roc_labels <- unique(roc_df$metric)
  roc_colors <- setNames(c("#E69F00", "#56B4E9", "#009E73", "#F0E442",
                           "#0072B2", "#D55E00", "#CC79A7", "#999999"), roc_labels)
  p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr, colour = metric)) +
    geom_path(aes(group = metric), linewidth = 0.9) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                colour = "grey60", linewidth = 0.4) +
    scale_color_manual(values = roc_colors) +
    coord_equal() +
    labs(x = "1 − 特异性", y = "敏感性", colour = NULL,
         title = paste0("[", part, "] 样本内 enrich 字典指标的队列判别 ROC（T=20%, Δ=10%）"),
         subtitle = "RA 为阳性类；曲线在对角线下方表示该指标在 ILD 中更高（非留一法）") +
    theme_classic(base_size = 14) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
          plot.subtitle = element_text(hjust = 0.5, size = 9),
          legend.position = "right", legend.text = element_text(size = 9))
  ggsave(paste0(plot_dir, "AA_dict_feasibility_ROC_", part, "_T20_D10.png"), p_roc,
         width = 9.5, height = 7, dpi = 300)

  cat(part, "图已生成\n")
}
cat("全部完成！\n")
