library(ggplot2)
library(data.table)
library(pROC)

setwd("/data/users/chenhaisheng/RA-ILD/")

# ====================================================================
# 留一法（LOOCV）验证：enrich 字典判别力
# ====================================================================
# 思路：174 轮，每轮藏起 1 个样本，用其余 173 个样本重建 T=20%、Δ=10%
#       的 RA/ILD 字典，再用"干净字典"计算被藏样本的 8 个指标。
# 输出：
#   1) TRB/result/CDR3_AA_sample_dict_hits_LOOCV_T20_D10.csv（公平分数表）
#   2) TRB/result/CDR3_AA_dict_LOOCV_summary_T20_D10.csv（对比汇总）
#   3) TRB/gradient/dict_LOOCV/：LOOCV 箱线图、LOOCV ROC、样本内 vs LOOCV AUC 对比图
#   4) 控制台：汇总对比 + 1000 次标签置换检验
# ====================================================================

THRESHOLD_PCT <- 20
DELTA_PCT <- 10

metadata <- read.csv("./TRB/metadata.csv")

aa_dir <- "./TRB/result/01_AA_clone_table/"
aa_files <- list.files(aa_dir, pattern = "_AA_clone_table\\.csv$")
aa_ids <- gsub("_AA_clone_table\\.csv$", "", aa_files)
aa_ids <- aa_ids[!grepl("^01_", aa_ids)]
aa_ids <- aa_ids[aa_ids %in% metadata$libraryid]
aa_files_f <- paste0(aa_ids, "_AA_clone_table.csv")

sample_info <- merge(data.frame(libraryid = aa_ids),
                     metadata[, c("libraryid", "material", "cohort")], by = "libraryid")
rownames(sample_info) <- sample_info$libraryid
ra_ids  <- sample_info$libraryid[sample_info$cohort == "RA"]
ild_ids <- sample_info$libraryid[sample_info$cohort == "ILD"]
n_ra  <- length(ra_ids)
n_ild <- length(ild_ids)
cat("RA:", n_ra, "样本 | ILD:", n_ild, "样本\n")

# ====================================================================
# 读取全部样本（clone 序列 + read_fraction），构建增量计数结构
# ====================================================================
cat("正在读取数据...\n")
all_clones <- list()   # 每样本唯一 clone 向量
all_rf     <- list()   # 每样本 read_fraction（与 clone 对齐）
for (i in seq_along(aa_ids)) {
  d <- fread(file.path(aa_dir, aa_files_f[i]), select = c("cdr3_aa", "read_fraction"))
  d <- d[!is.na(cdr3_aa)]
  all_clones[[aa_ids[i]]] <- d$cdr3_aa
  all_rf[[aa_ids[i]]]     <- d$read_fraction
}
rm(d)

# 全量计数（不重复构建，供每轮增量修正）
build_counts <- function(ids) {
  dt <- data.table(clone = unlist(all_clones[ids]))
  dt[, .N, by = clone]
}
ra_c  <- build_counts(ra_ids)
ild_c <- build_counts(ild_ids)
ra_counts  <- setNames(as.numeric(ra_c$N),  ra_c$clone)
ild_counts <- setNames(as.numeric(ild_c$N), ild_c$clone)
rm(ra_c, ild_c); gc()

# ====================================================================
# 174 轮留一法
# ====================================================================
cat("开始留一法（", length(aa_ids), " 轮）...\n", sep = "")
round_results <- vector("list", length(aa_ids))
dict_size_ra  <- integer(length(aa_ids))
dict_size_ild <- integer(length(aa_ids))

for (i in seq_along(aa_ids)) {
  sid  <- aa_ids[i]
  coh  <- sample_info[sid, "cohort"]
  cl_i <- all_clones[[sid]]
  rf_i <- all_rf[[sid]]

  if (coh == "RA") {
    n_ra_i  <- n_ra - 1;  n_ild_i <- n_ild
    idx <- match(cl_i, names(ra_counts))
    ra_counts[idx] <- ra_counts[idx] - 1
  } else {
    n_ra_i  <- n_ra;  n_ild_i <- n_ild - 1
    idx <- match(cl_i, names(ild_counts))
    ild_counts[idx] <- ild_counts[idx] - 1
  }
  ra_t  <- ceiling(n_ra_i  * THRESHOLD_PCT / 100)
  ild_t <- ceiling(n_ild_i * THRESHOLD_PCT / 100)

  # 候选 clone：任一队列达到阈值（全量扫描，向量化）
  cand_mask <- (ra_counts >= ra_t) | (ild_counts >= ild_t)
  cand <- names(ra_counts)[cand_mask]
  pct_ra  <- ra_counts[cand]  / n_ra_i  * 100
  pct_ild <- ild_counts[cand] / n_ild_i * 100
  delta <- pct_ra - pct_ild

  ra_dict  <- cand[pct_ra  >= THRESHOLD_PCT & delta >=  DELTA_PCT]
  ild_dict <- cand[pct_ild >= THRESHOLD_PCT & delta <= -DELTA_PCT]
  dict_size_ra[i]  <- length(ra_dict)
  dict_size_ild[i] <- length(ild_dict)

  # 恢复计数
  if (coh == "RA") ra_counts[idx]  <- ra_counts[idx] + 1
  else             ild_counts[idx] <- ild_counts[idx] + 1

  # 被藏样本的公平分数
  m_ra  <- cl_i %in% ra_dict
  m_ild <- cl_i %in% ild_dict
  ra_hits  <- sum(m_ra)
  ild_hits <- sum(m_ild)
  ra_fsum  <- sum(rf_i[m_ra])
  ild_fsum <- sum(rf_i[m_ild])
  round_results[[i]] <- data.table(
    libraryid = sid, cohort = coh, material = sample_info[sid, "material"],
    RA_dict_clone_count  = ra_hits,
    ILD_dict_clone_count = ild_hits,
    RA_dict_read_fraction_sum  = round(ra_fsum,  6),
    ILD_dict_read_fraction_sum = round(ild_fsum, 6),
    RA_dict_hit_rate  = round(ra_hits  / length(ra_dict),  4),
    ILD_dict_hit_rate = round(ild_hits / length(ild_dict), 4),
    RA_minus_ILD_count = ra_hits - ild_hits,
    RA_minus_ILD_freq  = round(ra_fsum - ild_fsum, 6)
  )
  if (i %% 20 == 0) cat("  已完成", i, "/", length(aa_ids), "轮\n")
}

loocv_hits <- rbindlist(round_results)
loocv_hits <- loocv_hits[order(cohort, libraryid)]
write.csv(loocv_hits, "./TRB/result/CDR3_AA_sample_dict_hits_LOOCV_T20_D10.csv",
          row.names = FALSE)

cat("每轮字典规模: RA 字典 ", min(dict_size_ra), "-", max(dict_size_ra),
    "（均值 ", round(mean(dict_size_ra)), "）条 | ILD 字典 ", min(dict_size_ild),
    "-", max(dict_size_ild), "（均值 ", round(mean(dict_size_ild)), "）条\n", sep = "")

# ====================================================================
# 指标定义与分析函数（与可行性脚本一致）
# ====================================================================
metrics <- c("RA_dict_clone_count", "ILD_dict_clone_count",
             "RA_dict_hit_rate", "ILD_dict_hit_rate",
             "RA_dict_read_fraction_sum", "ILD_dict_read_fraction_sum",
             "RA_minus_ILD_count", "RA_minus_ILD_freq")
metric_labels <- c("RA 字典命中数", "ILD 字典命中数",
                   "RA 字典命中率", "ILD 字典命中率",
                   "RA 命中 read_fraction 和", "ILD 命中 read_fraction 和",
                   "RA−ILD 命中数差", "RA−ILD read_fraction 和差")

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
    auc_ci <- as.numeric(ci.auc(roc_obj))
    out[[i]] <- list(
      metric = m, label = metric_labels[i],
      mean_ra = mean(x_ra), med_ra = median(x_ra),
      mean_ild = mean(x_ild), med_ild = median(x_ild),
      main_test = main_test,
      t_p = t_res$p.value, u_p = u_res$p.value,
      cohen_d = cohen_d(x_ra, x_ild), cliff_delta = cliff_delta(x_ra, x_ild),
      auc = as.numeric(auc(roc_obj)), auc_lo = auc_ci[1], auc_hi = auc_ci[3],
      roc = roc_obj
    )
  }
  main_p <- sapply(out, function(r) if (r$main_test == "Mann-Whitney U") r$u_p else r$t_p)
  main_q <- p.adjust(main_p, "BH")
  for (i in seq_along(out)) { out[[i]]$main_p <- main_p[i]; out[[i]]$main_q <- main_q[i] }
  out
}

cat("分析留一法公平分数...\n")
loocv_res <- analyze_metrics(loocv_hits)

# ---- 样本内结果（读取已有命中表） ----
insample_hits <- fread("./TRB/result/CDR3_AA_sample_dict_hits_T20_D10.csv")
cat("分析样本内分数...\n")
insample_res <- analyze_metrics(insample_hits)

# ====================================================================
# 置换检验：1000 次打乱队列标签，重算各指标 AUC（公平分数上）
# ====================================================================
cat("置换检验（1000 次）...\n")
set.seed(20260804)
labels_obs <- loocv_hits$cohort
y <- as.integer(labels_obs == "RA")
n_perm <- 1000
perm_p <- numeric(length(metrics))
for (k in seq_along(metrics)) {
  scores <- loocv_hits[[metrics[k]]]
  r_obs  <- rank(scores)
  auc_obs <- (sum(r_obs[y == 1]) - sum(y) * (sum(y) + 1) / 2) / (sum(y) * sum(!y))
  perm_auc <- numeric(n_perm)
  for (p in 1:n_perm) {
    yp <- sample(y)
    rp <- rank(scores)
    perm_auc[p] <- (sum(rp[yp == 1]) - sum(yp) * (sum(yp) + 1) / 2) / (sum(yp) * sum(!yp))
  }
  p_upper <- mean(perm_auc >= auc_obs)
  p_lower <- mean(perm_auc <= auc_obs)
  perm_p[k] <- min(1, 2 * min(p_upper, p_lower) + 1 / (n_perm + 1))
}
names(perm_p) <- metrics

# ====================================================================
# 汇总对比表（控制台 + CSV）
# ====================================================================
cat("\n========================================================================\n")
cat(" 留一法 vs 样本内（T=20%, Δ=10%, total 字典）\n")
cat("========================================================================\n\n")
cat(sprintf("%-22s %10s %10s %10s %10s %8s %10s %10s %8s\n",
            "指标", "样本内AUC", "LOOCV-AUC", "AUC降幅", "LOOCV p", "主检验",
            "效应量", "置换p", "LOOCV-AUC95%CI"))
cat(sprintf("%-22s %10s %10s %10s %10s %8s %10s %10s %8s\n",
            "----------------------", "----------", "----------", "----------",
            "--------", "--------", "----------", "----------", "----------"))
for (i in seq_along(metrics)) {
  r_in <- insample_res[[i]]
  r_lo <- loocv_res[[i]]
  eff <- if (r_lo$main_test == "Mann-Whitney U") sprintf("Cliff %+.2f", r_lo$cliff_delta)
         else sprintf("d %+.2f", r_lo$cohen_d)
  cat(sprintf("%-22s %10.3f %10.3f %10.3f %10.2g %8s %10s %10.2g %7.3f-%5.3f\n",
              r_lo$label, r_in$auc, r_lo$auc, r_in$auc - r_lo$auc,
              r_lo$main_p, if (r_lo$main_test == "Mann-Whitney U") "MWU" else "Welch",
              eff, perm_p[i], r_lo$auc_lo, r_lo$auc_hi))
}

summary_df <- data.frame(
  metric = metric_labels,
  insample_AUC = sapply(insample_res, `[[`, "auc"),
  insample_AUC_lo = sapply(insample_res, `[[`, "auc_lo"),
  insample_AUC_hi = sapply(insample_res, `[[`, "auc_hi"),
  LOOCV_AUC = sapply(loocv_res, `[[`, "auc"),
  LOOCV_AUC_lo = sapply(loocv_res, `[[`, "auc_lo"),
  LOOCV_AUC_hi = sapply(loocv_res, `[[`, "auc_hi"),
  AUC_drop = sapply(insample_res, `[[`, "auc") - sapply(loocv_res, `[[`, "auc"),
  LOOCV_RA_mean = sapply(loocv_res, `[[`, "mean_ra"),
  LOOCV_RA_median = sapply(loocv_res, `[[`, "med_ra"),
  LOOCV_ILD_mean = sapply(loocv_res, `[[`, "mean_ild"),
  LOOCV_ILD_median = sapply(loocv_res, `[[`, "med_ild"),
  LOOCV_main_test = sapply(loocv_res, `[[`, "main_test"),
  LOOCV_main_p = sapply(loocv_res, `[[`, "main_p"),
  LOOCV_main_q = sapply(loocv_res, `[[`, "main_q"),
  cohen_d = sapply(loocv_res, `[[`, "cohen_d"),
  cliff_delta = sapply(loocv_res, `[[`, "cliff_delta"),
  permutation_p = perm_p
)
write.csv(summary_df, "./TRB/result/CDR3_AA_dict_LOOCV_summary_T20_D10.csv", row.names = FALSE)

# ====================================================================
# 图
# ====================================================================
plot_dir <- "./TRB/gradient/dict_LOOCV/"
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

# ---- 1) LOOCV 公平分数箱线图（2×4，标注方法与 p 值） ----
box_df <- melt(loocv_hits, id.vars = c("libraryid", "cohort"),
               measure.vars = metrics, variable.name = "metric", value.name = "value")
box_df$metric <- factor(box_df$metric, levels = metrics, labels = metric_labels)
ann_df <- data.frame(
  metric = factor(metric_labels, levels = metric_labels),
  p_label = sapply(loocv_res, function(r) {
    sprintf("%s\np = %.2g",
            if (r$main_test == "Mann-Whitney U") "Mann-Whitney U" else "Welch t",
            r$main_p)
  }),
  stringsAsFactors = FALSE
)
ann_df$y_pos <- sapply(metrics, function(m) {
  v <- loocv_hits[[m]]
  max(v) + (max(v) - min(v)) * 0.12
})
p_box <- ggplot(box_df, aes(x = cohort, y = value, fill = cohort)) +
  geom_boxplot(outlier.shape = NA, width = 0.5) +
  geom_jitter(width = 0.18, size = 0.8, alpha = 0.4, colour = "grey30") +
  geom_text(data = ann_df, inherit.aes = FALSE, aes(x = 1.5, y = y_pos, label = p_label),
            size = 3.4, fontface = "bold") +
  facet_wrap(~metric, scales = "free_y", ncol = 4) +
  scale_fill_manual(values = c("RA" = "#0072B2", "ILD" = "#D55E00")) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.22))) +
  labs(x = NULL, y = "值",
       title = "留一法公平分数：样本×字典命中指标 队列分布（T=20%, Δ=10%）",
       subtitle = "每样本分数由不含该样本的字典计算 | 蓝色=RA (n=108) | 朱红=ILD (n=66) | 标注为主检验方法与 p 值") +
  theme_classic(base_size = 13) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
        plot.subtitle = element_text(hjust = 0.5, size = 9.5),
        strip.text = element_text(size = 10))
ggsave(paste0(plot_dir, "AA_dict_LOOCV_boxplot_T20_D10.png"), p_box,
       width = 16, height = 8, dpi = 300)

# ---- 2) LOOCV ROC 曲线 ----
roc_df <- do.call(rbind, lapply(seq_along(metrics), function(i) {
  r <- loocv_res[[i]]
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
       title = "留一法公平分数的队列判别 ROC（T=20%, Δ=10%）",
       subtitle = "RA 为阳性类；曲线在对角线下方表示该指标在 ILD 中更高") +
  theme_classic(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
        plot.subtitle = element_text(hjust = 0.5, size = 9),
        legend.position = "right", legend.text = element_text(size = 9))
ggsave(paste0(plot_dir, "AA_dict_LOOCV_ROC_T20_D10.png"), p_roc,
       width = 9.5, height = 7, dpi = 300)

# ---- 3) 样本内 vs LOOCV AUC 对比（哑铃图） ----
cmp_df <- data.frame(
  metric = factor(metric_labels, levels = metric_labels),
  insample = sapply(insample_res, `[[`, "auc"),
  loocv = sapply(loocv_res, `[[`, "auc")
)
p_cmp <- ggplot(cmp_df) +
  geom_segment(aes(x = insample, xend = loocv, y = metric, yend = metric),
               colour = "grey50", linewidth = 0.7) +
  geom_point(aes(x = insample, y = metric), colour = "#B0BEC5", size = 3.2) +
  geom_point(aes(x = loocv, y = metric), colour = "#0072B2", size = 3.2) +
  geom_text(aes(x = loocv, y = metric,
                label = sprintf("%.3f", loocv)),
            hjust = -0.4, size = 3.2, colour = "#0072B2") +
  geom_vline(xintercept = 0.5, linetype = "dashed", colour = "grey60", linewidth = 0.4) +
  scale_x_continuous(limits = c(0, 1.15), breaks = seq(0, 1, 0.2)) +
  labs(x = "AUC", y = NULL,
       title = "样本内 vs 留一法 AUC（T=20%, Δ=10%, total 字典）",
       subtitle = "浅灰点 = 样本内（乐观上限）| 蓝点 = 留一法（真实判别力）| 虚线 = 随机水平 0.5") +
  theme_classic(base_size = 13) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
        plot.subtitle = element_text(hjust = 0.5, size = 9.5),
        axis.text.y = element_text(size = 10))
ggsave(paste0(plot_dir, "AA_dict_LOOCV_vs_insample_T20_D10.png"), p_cmp,
       width = 9, height = 6, dpi = 300)

cat("\n全部完成！\n")
cat("表: ./TRB/result/CDR3_AA_sample_dict_hits_LOOCV_T20_D10.csv\n")
cat("    ./TRB/result/CDR3_AA_dict_LOOCV_summary_T20_D10.csv\n")
cat("图: ./TRB/gradient/dict_LOOCV/AA_dict_LOOCV_*.png\n")
