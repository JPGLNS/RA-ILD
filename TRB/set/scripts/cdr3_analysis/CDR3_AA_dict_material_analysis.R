library(ggplot2)
library(data.table)
library(pROC)

setwd("/data/users/chenhaisheng/RA-ILD/")

# ====================================================================
# material 拆分（pbmc / buffycoat）enrich 字典判别分析
# ====================================================================
# 口径：每部分字典在材料内部构建（RA vs ILD，T=20%、Δ=10%）
# 每部分输出两套结果：
#   1) 非留一法（样本内）：hits 表 + 8 指标检验/AUC 汇总
#   2) 留一法：fair scores 表 + 8 指标检验/AUC 汇总 + 置换检验 + 图
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

out_dir <- "./TRB/result/"
plot_dir <- "./TRB/gradient/dict_LOOCV/"
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

# 读取全部克隆数据一次（两部分 LOOCV 共用）
cat("正在读取数据...\n")
all_clones <- list(); all_rf <- list()
for (i in seq_along(aa_ids)) {
  d <- fread(file.path(aa_dir, aa_files_f[i]), select = c("cdr3_aa", "read_fraction"))
  d <- d[!is.na(cdr3_aa)]
  all_clones[[aa_ids[i]]] <- d$cdr3_aa
  all_rf[[aa_ids[i]]]     <- d$read_fraction
}
rm(d); gc()

# 每部分的 in-sample 字典（材料内构建，来自已有 enrich CSV）
dicts <- list(
  pbmc = list(
    RA  = fread(file.path(out_dir, "CDR3_AA_enrich_RA_pbmc_T20_D10.csv"),  select = "aa")$aa,
    ILD = fread(file.path(out_dir, "CDR3_AA_enrich_ILD_pbmc_T20_D10.csv"), select = "aa")$aa),
  buffycoat = list(
    RA  = fread(file.path(out_dir, "CDR3_AA_enrich_RA_buffycoat_T20_D10.csv"),  select = "aa")$aa,
    ILD = fread(file.path(out_dir, "CDR3_AA_enrich_ILD_buffycoat_T20_D10.csv"), select = "aa")$aa)
)

# ====================================================================
# 通用函数
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
cohen_d <- function(a, b) {
  s_pool <- sqrt(((length(a) - 1) * var(a) + (length(b) - 1) * var(b)) /
                   (length(a) + length(b) - 2))
  (mean(a) - mean(b)) / s_pool
}
cliff_delta <- function(a, b) {
  sum(sapply(a, function(x) sum(x > b) - sum(x < b))) / (length(a) * length(b))
}

score_sample <- function(cl, rf, ra_dict, ild_dict) {
  m_ra  <- cl %in% ra_dict
  m_ild <- cl %in% ild_dict
  ra_hits  <- sum(m_ra)
  ild_hits <- sum(m_ild)
  ra_fsum  <- sum(rf[m_ra])
  ild_fsum <- sum(rf[m_ild])
  data.table(
    RA_dict_clone_count  = ra_hits,
    ILD_dict_clone_count = ild_hits,
    RA_dict_read_fraction_sum  = round(ra_fsum,  6),
    ILD_dict_read_fraction_sum = round(ild_fsum, 6),
    RA_dict_hit_rate  = round(ra_hits  / length(ra_dict),  4),
    ILD_dict_hit_rate = round(ild_hits / length(ild_dict), 4),
    RA_minus_ILD_count = ra_hits - ild_hits,
    RA_minus_ILD_freq  = round(ra_fsum - ild_fsum, 6)
  )
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

perm_test <- function(hits_tbl, seed) {
  set.seed(seed)
  y <- as.integer(hits_tbl$cohort == "RA")
  n_perm <- 1000
  perm_p <- numeric(length(metrics))
  for (k in seq_along(metrics)) {
    scores <- hits_tbl[[metrics[k]]]
    r_obs <- rank(scores)
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
  perm_p
}

print_summary <- function(res, title, perm_p = NULL) {
  cat("\n========================================================================\n")
  cat(" ", title, "\n")
  cat("========================================================================\n")
  cat(sprintf("%-22s %10s %10s %10s %10s %8s %10s %8s\n",
              "指标", "RA中位", "ILD中位", "主检验", "p", "效应量", "AUC", "AUC 95%CI"))
  for (r in res) {
    eff <- if (r$main_test == "Mann-Whitney U") sprintf("Cliff %+.2f", r$cliff_delta)
           else sprintf("d %+.2f", r$cohen_d)
    cat(sprintf("%-22s %10.3g %10.3g %8s %8.2g %10s %8.3f %7.3f-%5.3f\n",
                r$label, r$med_ra, r$med_ild,
                if (r$main_test == "Mann-Whitney U") "MWU" else "Welch",
                r$main_p, eff, r$auc, r$auc_lo, r$auc_hi))
  }
  if (!is.null(perm_p)) {
    cat("  置换检验 p: ")
    cat(paste(sprintf("%s=%.2g", metric_labels, perm_p), collapse = "; "), "\n")
  }
}

# ====================================================================
# 主流程：pbmc / buffycoat 两部分
# ====================================================================
loocv_auc_all <- list()

for (part in c("pbmc", "buffycoat")) {
  mat <- if (part == "pbmc") "PBMC" else "buffycoat"
  part_ids <- sample_info$libraryid[sample_info$material == mat]
  ra_ids  <- part_ids[sample_info[part_ids, "cohort"] == "RA"]
  ild_ids <- part_ids[sample_info[part_ids, "cohort"] == "ILD"]
  n_ra  <- length(ra_ids); n_ild <- length(ild_ids)
  cat("\n################################################################\n")
  cat(" 部分：", part, "（", length(part_ids), " 样本，RA ", n_ra, " / ILD ", n_ild, "）\n", sep = "")
  cat("################################################################\n")

  # ============ 1) 非留一法（样本内） ============
  cat("计算样本内结果...\n")
  in_hits <- rbindlist(lapply(part_ids, function(sid) {
    h <- score_sample(all_clones[[sid]], all_rf[[sid]],
                      dicts[[part]]$RA, dicts[[part]]$ILD)
    h[, `:=`(libraryid = sid, cohort = sample_info[sid, "cohort"],
             material = sample_info[sid, "material"])]
    h[, .(libraryid, RA_dict_clone_count, ILD_dict_clone_count, cohort,
          RA_dict_read_fraction_sum, ILD_dict_read_fraction_sum,
          RA_dict_hit_rate, ILD_dict_hit_rate,
          RA_minus_ILD_count, RA_minus_ILD_freq, material)]
  }))
  in_hits <- in_hits[order(cohort, libraryid)]
  write.csv(in_hits, file.path(out_dir, paste0("CDR3_AA_sample_dict_hits_", part, "_T20_D10.csv")),
            row.names = FALSE)
  in_res <- analyze_metrics(in_hits)
  print_summary(in_res, paste0(part, " 样本内（非留一法，T=20%, Δ=10%）"))
  in_sum_df <- data.frame(
    metric = metric_labels,
    RA_mean = sapply(in_res, `[[`, "mean_ra"), RA_median = sapply(in_res, `[[`, "med_ra"),
    ILD_mean = sapply(in_res, `[[`, "mean_ild"), ILD_median = sapply(in_res, `[[`, "med_ild"),
    main_test = sapply(in_res, `[[`, "main_test"),
    main_p = sapply(in_res, `[[`, "main_p"), main_q = sapply(in_res, `[[`, "main_q"),
    cohen_d = sapply(in_res, `[[`, "cohen_d"), cliff_delta = sapply(in_res, `[[`, "cliff_delta"),
    AUC = sapply(in_res, `[[`, "auc"),
    AUC_lo = sapply(in_res, `[[`, "auc_lo"), AUC_hi = sapply(in_res, `[[`, "auc_hi")
  )
  write.csv(in_sum_df, file.path(out_dir, paste0("CDR3_AA_dict_insample_summary_", part, "_T20_D10.csv")),
            row.names = FALSE)

  # ============ 2) 留一法 ============
  cat("留一法（", length(part_ids), " 轮）...\n", sep = "")
  ra_c  <- data.table(clone = unlist(all_clones[ra_ids]))[, .N, by = clone]
  ild_c <- data.table(clone = unlist(all_clones[ild_ids]))[, .N, by = clone]
  ra_counts  <- setNames(as.numeric(ra_c$N),  ra_c$clone)
  ild_counts <- setNames(as.numeric(ild_c$N), ild_c$clone)
  rm(ra_c, ild_c); gc()

  round_results <- vector("list", length(part_ids))
  for (i in seq_along(part_ids)) {
    sid <- part_ids[i]
    coh <- sample_info[sid, "cohort"]
    cl_i <- all_clones[[sid]]; rf_i <- all_rf[[sid]]
    if (coh == "RA") {
      n_ra_i <- n_ra - 1; n_ild_i <- n_ild
      idx <- match(cl_i, names(ra_counts)); ra_counts[idx] <- ra_counts[idx] - 1
    } else {
      n_ra_i <- n_ra; n_ild_i <- n_ild - 1
      idx <- match(cl_i, names(ild_counts)); ild_counts[idx] <- ild_counts[idx] - 1
    }
    ra_t  <- ceiling(n_ra_i  * THRESHOLD_PCT / 100)
    ild_t <- ceiling(n_ild_i * THRESHOLD_PCT / 100)
    cand_mask <- (ra_counts >= ra_t) | (ild_counts >= ild_t)
    cand <- names(ra_counts)[cand_mask]
    pct_ra  <- ra_counts[cand]  / n_ra_i  * 100
    pct_ild <- ild_counts[cand] / n_ild_i * 100
    delta <- pct_ra - pct_ild
    ra_dict  <- cand[pct_ra  >= THRESHOLD_PCT & delta >=  DELTA_PCT]
    ild_dict <- cand[pct_ild >= THRESHOLD_PCT & delta <= -DELTA_PCT]
    if (coh == "RA") ra_counts[idx]  <- ra_counts[idx] + 1
    else             ild_counts[idx] <- ild_counts[idx] + 1

    h <- score_sample(cl_i, rf_i, ra_dict, ild_dict)
    h[, `:=`(libraryid = sid, cohort = coh, material = sample_info[sid, "material"])]
    h <- h[, .(libraryid, RA_dict_clone_count, ILD_dict_clone_count, cohort,
               RA_dict_read_fraction_sum, ILD_dict_read_fraction_sum,
               RA_dict_hit_rate, ILD_dict_hit_rate,
               RA_minus_ILD_count, RA_minus_ILD_freq, material)]
    round_results[[i]] <- h
    if (i %% 20 == 0) cat("  已完成", i, "/", length(part_ids), "轮\n")
  }
  loocv_hits <- rbindlist(round_results)
  loocv_hits <- loocv_hits[order(cohort, libraryid)]
  write.csv(loocv_hits,
            file.path(out_dir, paste0("CDR3_AA_sample_dict_hits_LOOCV_", part, "_T20_D10.csv")),
            row.names = FALSE)

  loocv_res <- analyze_metrics(loocv_hits)
  perm_p <- perm_test(loocv_hits, seed = 20260804)
  loocv_auc_all[[part]] <- sapply(loocv_res, `[[`, "auc")
  names(loocv_auc_all[[part]]) <- metrics
  print_summary(loocv_res, paste0(part, " 留一法（公平分数，T=20%, Δ=10%）"), perm_p)

  lo_sum_df <- data.frame(
    metric = metric_labels,
    insample_AUC = sapply(in_res, `[[`, "auc"),
    LOOCV_AUC = sapply(loocv_res, `[[`, "auc"),
    AUC_drop = sapply(in_res, `[[`, "auc") - sapply(loocv_res, `[[`, "auc"),
    LOOCV_AUC_lo = sapply(loocv_res, `[[`, "auc_lo"),
    LOOCV_AUC_hi = sapply(loocv_res, `[[`, "auc_hi"),
    LOOCV_RA_median = sapply(loocv_res, `[[`, "med_ra"),
    LOOCV_ILD_median = sapply(loocv_res, `[[`, "med_ild"),
    LOOCV_main_test = sapply(loocv_res, `[[`, "main_test"),
    LOOCV_main_p = sapply(loocv_res, `[[`, "main_p"),
    LOOCV_main_q = sapply(loocv_res, `[[`, "main_q"),
    cohen_d = sapply(loocv_res, `[[`, "cohen_d"),
    cliff_delta = sapply(loocv_res, `[[`, "cliff_delta"),
    permutation_p = perm_p
  )
  write.csv(lo_sum_df, file.path(out_dir, paste0("CDR3_AA_dict_LOOCV_summary_", part, "_T20_D10.csv")),
            row.names = FALSE)

  # ---- 图：LOOCV 箱线图 + ROC ----
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
    geom_text(data = ann_df, inherit.aes = FALSE,
              aes(x = 1.5, y = y_pos, label = p_label), size = 3.4, fontface = "bold") +
    facet_wrap(~metric, scales = "free_y", ncol = 4) +
    scale_fill_manual(values = c("RA" = "#0072B2", "ILD" = "#D55E00")) +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.22))) +
    labs(x = NULL, y = "值",
         title = paste0("[", part, "] 留一法公平分数 队列分布（T=20%, Δ=10%, 材料内字典）"),
         subtitle = paste0("每样本分数由不含该样本的", part, "字典计算 | RA n=", n_ra,
                           " | ILD n=", n_ild, " | 标注为主检验方法与 p 值")) +
    theme_classic(base_size = 13) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
          plot.subtitle = element_text(hjust = 0.5, size = 9.5),
          strip.text = element_text(size = 10))
  ggsave(paste0(plot_dir, "AA_dict_LOOCV_boxplot_", part, "_T20_D10.png"), p_box,
         width = 16, height = 8, dpi = 300)

  roc_df <- do.call(rbind, lapply(seq_along(metrics), function(i) {
    r <- loocv_res[[i]]
    # pROC 坐标反序存储（(1,1)→(0,0)），反转后才是曲线正序，避免并列 FPR 处 TPR 回折
    coord <- coords(r$roc, x = "all", ret = c("specificity", "sensitivity"), transpose = FALSE)
    coord <- coord[rev(seq_len(nrow(coord))), , drop = FALSE]
    data.frame(metric = paste0(r$label, " (AUC ", sprintf("%.3f", r$auc), ")"),
               fpr = 1 - coord$specificity, tpr = coord$sensitivity)
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
         title = paste0("[", part, "] 留一法公平分数 队列判别 ROC（T=20%, Δ=10%）"),
         subtitle = "RA 为阳性类；曲线在对角线下方表示该指标在 ILD 中更高") +
    theme_classic(base_size = 14) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
          plot.subtitle = element_text(hjust = 0.5, size = 9),
          legend.position = "right", legend.text = element_text(size = 9))
  ggsave(paste0(plot_dir, "AA_dict_LOOCV_ROC_", part, "_T20_D10.png"), p_roc,
         width = 9.5, height = 7, dpi = 300)

  rm(ra_counts, ild_counts); gc()
}

# ====================================================================
# 三部分 LOOCV AUC 对比（total 从已有汇总 CSV 读取，若已生成）
# ====================================================================
total_lo <- NULL
if (file.exists(file.path(out_dir, "CDR3_AA_dict_LOOCV_summary_T20_D10.csv"))) {
  total_lo <- fread(file.path(out_dir, "CDR3_AA_dict_LOOCV_summary_T20_D10.csv"))
}
cat("\n================================================================\n")
cat(" 三部分留一法 AUC 对比（材料内字典）\n")
cat("================================================================\n")
if (!is.null(total_lo)) {
  cat(sprintf("%-22s %10s %10s %10s\n", "指标", "total", "pbmc", "buffycoat"))
  for (i in seq_along(metrics)) {
    cat(sprintf("%-22s %10.3f %10.3f %10.3f\n", metric_labels[i],
                total_lo$LOOCV_AUC[i], loocv_auc_all$pbmc[i], loocv_auc_all$buffycoat[i]))
  }
} else {
  cat("（total 汇总尚未生成，跳过）\n")
  cat(sprintf("%-22s %10s %10s\n", "指标", "pbmc", "buffycoat"))
  for (i in seq_along(metrics)) {
    cat(sprintf("%-22s %10.3f %10.3f\n", metric_labels[i],
                loocv_auc_all$pbmc[i], loocv_auc_all$buffycoat[i]))
  }
}
cat("\n全部完成！\n")
