library(ggplot2)
library(data.table)
library(pROC)

setwd("/data/users/chenhaisheng/RA-ILD/")

# ====================================================================
# enrich 字典判别可行性分析（基于样本×字典命中表）
# ====================================================================
# 数据：TRB/result/CDR3_AA_sample_dict_hits_T20_D10.csv（174 样本）
# 指标（6 个）：4 原始 + 2 衍生差值（RA−ILD）
# 流程：描述统计 → 分布检查（Shapiro-Wilk + 偏度）
#       → 主检验选择（偏态则 Mann-Whitney U，否则 Welch t；另一检验作参考）
#       → BH-FDR 多重校正 → 效应量（Cohen's d / Cliff's delta）→ ROC/AUC（DeLong）
# 注意：字典基于全部样本构建，属样本内评估，AUC 偏乐观
# ====================================================================

hits <- fread("./TRB/result/CDR3_AA_sample_dict_hits_T20_D10.csv")
hits[, RA_minus_ILD_count := RA_dict_clone_count - ILD_dict_clone_count]
hits[, RA_minus_ILD_freq   := RA_dict_read_fraction_sum - ILD_dict_read_fraction_sum]

metrics <- c("RA_dict_clone_count", "ILD_dict_clone_count",
             "RA_dict_hit_rate", "ILD_dict_hit_rate",
             "RA_dict_read_fraction_sum", "ILD_dict_read_fraction_sum",
             "RA_minus_ILD_count", "RA_minus_ILD_freq")
metric_labels <- c("RA 字典命中数", "ILD 字典命中数",
                   "RA 字典命中率", "ILD 字典命中率",
                   "RA 命中 read_fraction 和", "ILD 命中 read_fraction 和",
                   "RA−ILD 命中数差", "RA−ILD read_fraction 和差")

ra  <- hits[cohort == "RA"]
ild <- hits[cohort == "ILD"]
n_ra <- nrow(ra); n_ild <- nrow(ild)
cat("RA:", n_ra, "样本 | ILD:", n_ild, "样本\n\n")

skewness <- function(x) {
  m <- mean(x); s <- sd(x)
  mean((x - m)^3) / s^3
}
cohen_d <- function(a, b) {
  s_pool <- sqrt(((length(a) - 1) * var(a) + (length(b) - 1) * var(b)) /
                   (length(a) + length(b) - 2))
  (mean(a) - mean(b)) / s_pool
}
cliff_delta <- function(a, b) {
  sum(sapply(a, function(x) sum(x > b) - sum(x < b))) / (length(a) * length(b))
}

results <- list()

for (i in seq_along(metrics)) {
  m <- metrics[i]
  x_ra  <- ra[[m]]
  x_ild <- ild[[m]]

  # ---- 分布检查 ----
  sw_ra  <- shapiro.test(x_ra)
  sw_ild <- shapiro.test(x_ild)
  sk_ra  <- skewness(x_ra)
  sk_ild <- skewness(x_ild)
  # 决策规则：任一队列显著偏离正态（SW p<0.05 且 |偏度|>1）→ 主检验用 MWU
  biased <- (sw_ra$p.value < 0.05 & abs(sk_ra) > 1) |
            (sw_ild$p.value < 0.05 & abs(sk_ild) > 1)
  main_test <- if (biased) "Mann-Whitney U" else "Welch t"

  # ---- 组间检验（双口径） ----
  t_res <- t.test(x_ra, x_ild)          # Welch t
  u_res <- wilcox.test(x_ra, x_ild)     # Mann-Whitney U

  # ---- ROC / AUC（DeLong 95% CI；RA 为阳性类，指标高值指向 RA） ----
  roc_obj <- roc(hits$cohort, hits[[m]], levels = c("ILD", "RA"),
                 direction = "<", quiet = TRUE)
  auc_val <- as.numeric(auc(roc_obj))
  auc_ci  <- as.numeric(ci.auc(roc_obj))

  results[[i]] <- list(
    metric = m, label = metric_labels[i],
    mean_ra = mean(x_ra), med_ra = median(x_ra),
    mean_ild = mean(x_ild), med_ild = median(x_ild),
    sw_p_ra = sw_ra$p.value, sw_p_ild = sw_ild$p.value,
    skew_ra = sk_ra, skew_ild = sk_ild,
    main_test = main_test,
    t_stat = as.numeric(t_res$statistic), t_p = t_res$p.value,
    u_stat = as.numeric(u_res$statistic), u_p = u_res$p.value,
    cohen_d = cohen_d(x_ra, x_ild), cliff_delta = cliff_delta(x_ra, x_ild),
    auc = auc_val, auc_lo = auc_ci[1], auc_hi = auc_ci[3],
    roc = roc_obj
  )
}

# ---- BH-FDR（主检验与参考检验分别校正） ----
main_p <- sapply(results, function(r) if (r$main_test == "Mann-Whitney U") r$u_p else r$t_p)
ref_p  <- sapply(results, function(r) if (r$main_test == "Mann-Whitney U") r$t_p else r$u_p)
main_q <- p.adjust(main_p, "BH")
ref_q  <- p.adjust(ref_p, "BH")
for (i in seq_along(results)) {
  results[[i]]$main_p <- main_p[i]
  results[[i]]$main_q <- main_q[i]
  results[[i]]$ref_p  <- ref_p[i]
  results[[i]]$ref_q  <- ref_q[i]
}

# ====================================================================
# 控制台汇总表
# ====================================================================
cat("========================================================================\n")
cat(" 判别可行性汇总（T=20%, Δ=10%, total 字典, 样本内评估）\n")
cat("========================================================================\n\n")
cat(sprintf("%-24s %10s %10s %10s %10s %9s %9s %7s %8s\n",
            "指标", "RA均值", "RA中位", "ILD均值", "ILD中位",
            "SW_p(RA)", "SW_p(ILD)", "主检验", "主检验p"))
cat(sprintf("%-24s %10s %10s %10s %10s %9s %9s %7s %8s\n",
            "------------------------", "----------", "----------",
            "----------", "----------", "---------", "---------", "-------", "--------"))
for (r in results) {
  cat(sprintf("%-24s %10.3g %10.3g %10.3g %10.3g %9.2g %9.2g %7s %8.2g\n",
              r$label, r$mean_ra, r$med_ra, r$mean_ild, r$med_ild,
              r$sw_p_ra, r$sw_p_ild,
              if (r$main_test == "Mann-Whitney U") "MWU" else "Welch",
              r$main_p))
}
cat("\n")
cat(sprintf("%-24s %12s %12s %12s %9s %14s\n",
            "指标", "主检验q(FDR)", "参考检验p", "效应量", "AUC", "AUC 95%CI"))
cat(sprintf("%-24s %12s %12s %12s %9s %14s\n",
            "------------------------", "------------", "------------",
            "------------", "---------", "--------------"))
for (r in results) {
  eff <- if (r$main_test == "Mann-Whitney U") sprintf("Cliff %+.3f", r$cliff_delta)
         else sprintf("d %+.3f", r$cohen_d)
  cat(sprintf("%-24s %12.2g %12.2g %12s %9.3f %7.3f-%5.3f\n",
              r$label, r$main_q, r$ref_p, eff, r$auc, r$auc_lo, r$auc_hi))
}
cat("\n注: AUC 以 RA 为阳性类；<0.5 表示该指标在 ILD 中更高（反向判别）\n")

# ---- 汇总表导出 CSV ----
summary_df <- data.frame(
  metric = metric_labels,
  RA_mean = sapply(results, `[[`, "mean_ra"), RA_median = sapply(results, `[[`, "med_ra"),
  ILD_mean = sapply(results, `[[`, "mean_ild"), ILD_median = sapply(results, `[[`, "med_ild"),
  SW_p_RA = sapply(results, `[[`, "sw_p_ra"), SW_p_ILD = sapply(results, `[[`, "sw_p_ild"),
  skew_RA = sapply(results, `[[`, "skew_ra"), skew_ILD = sapply(results, `[[`, "skew_ild"),
  main_test = sapply(results, `[[`, "main_test"),
  main_p = main_p, main_q = main_q, ref_p = ref_p, ref_q = ref_q,
  cohen_d = sapply(results, `[[`, "cohen_d"), cliff_delta = sapply(results, `[[`, "cliff_delta"),
  AUC = sapply(results, `[[`, "auc"),
  AUC_lo = sapply(results, `[[`, "auc_lo"), AUC_hi = sapply(results, `[[`, "auc_hi")
)
write.csv(summary_df, "./TRB/result/CDR3_AA_dict_feasibility_summary_T20_D10.csv", row.names = FALSE)

# ====================================================================
# 图形
# ====================================================================
plot_dir <- "./TRB/gradient/dict_feasibility/"
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

# ---- 箱线图 + 散点（2×3，面板标注主检验方法与 p 值） ----
box_df <- melt(hits, id.vars = c("libraryid", "cohort"),
               measure.vars = metrics, variable.name = "metric", value.name = "value")
box_df$metric <- factor(box_df$metric, levels = metrics, labels = metric_labels)

# 每面板的显著性标注（主检验方法 + p 值，位置在箱体上方）
ann_df <- data.frame(
  metric = factor(metric_labels, levels = metric_labels),
  p_label = sapply(results, function(r) {
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
            aes(x = 1.5, y = y_pos, label = p_label),
            size = 3.4, fontface = "bold") +
  facet_wrap(~metric, scales = "free_y", ncol = 4) +
  scale_fill_manual(values = c("RA" = "#0072B2", "ILD" = "#D55E00")) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.22))) +
  labs(x = NULL, y = "值",
       title = "样本×字典命中指标 队列分布（T=20%, Δ=10%, total 字典）",
       subtitle = "样本内评估 | 蓝色=RA (n=108) | 朱红=ILD (n=66) | 面板标注为主检验方法与 p 值") +
  theme_classic(base_size = 13) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
        plot.subtitle = element_text(hjust = 0.5, size = 10),
        strip.text = element_text(size = 10))
ggsave(paste0(plot_dir, "AA_dict_feasibility_boxplot_T20_D10.png"), p_box,
       width = 16, height = 8, dpi = 300)

# ---- ROC 曲线（6 指标一张） ----
roc_df <- do.call(rbind, lapply(seq_along(results), function(i) {
  r <- results[[i]]
  # pROC 的 specificities/sensitivities 按反序存储（(1,1)→(0,0)），
  # 直接交给 geom_line 会在并列 FPR 处产生 TPR 回折；反转后才是曲线正序（(0,0)→(1,1)）
  coord <- coords(r$roc, x = "all", ret = c("specificity", "sensitivity"), transpose = FALSE)
  coord <- coord[rev(seq_len(nrow(coord))), , drop = FALSE]
  data.frame(metric = paste0(r$label, " (AUC ", sprintf("%.3f", r$auc), ")"),
             fpr = 1 - coord$specificity,
             tpr = coord$sensitivity)
}))
roc_labels <- unique(roc_df$metric)
roc_colors <- setNames(c("#E69F00", "#56B4E9", "#009E73", "#F0E442",
                         "#0072B2", "#D55E00", "#CC79A7", "#999999"),
                       roc_labels)

p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr, colour = metric)) +
  geom_path(aes(group = metric), linewidth = 0.9) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed",
              colour = "grey60", linewidth = 0.4) +
  scale_color_manual(values = roc_colors) +
  coord_equal() +
  labs(x = "1 − 特异性", y = "敏感性", colour = NULL,
       title = "enrich 字典指标的队列判别 ROC（T=20%, Δ=10%, total 字典）",
       subtitle = "RA 为阳性类；曲线在对角线下方表示该指标在 ILD 中更高") +
  theme_classic(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
        plot.subtitle = element_text(hjust = 0.5, size = 9),
        legend.position = "right",
        legend.text = element_text(size = 9))
ggsave(paste0(plot_dir, "AA_dict_feasibility_ROC_T20_D10.png"), p_roc,
       width = 9.5, height = 7, dpi = 300)

cat("\n汇总表已保存: ./TRB/result/CDR3_AA_dict_feasibility_summary_T20_D10.csv\n")
cat("图已保存: ", plot_dir, "AA_dict_feasibility_*.png\n", sep = "")
