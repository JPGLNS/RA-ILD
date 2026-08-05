library(ggplot2)
library(data.table)
library(pROC)

setwd("/data/users/chenhaisheng/RA-ILD/")
# utils 以 GitHub 同步目录（set/）为正式版本
utils_file <- "TRB/set/scripts/cdr3_analysis/CDR3_AA_dict_utils.R"
stopifnot(file.exists(utils_file))
source(utils_file)

# ====================================================================
# material 拆分（pbmc / buffycoat）enrich 字典判别分析（GPT 审查修复版）
# ====================================================================
# 口径不变：每部分字典在材料内部构建（RA vs ILD，T=20%、Δ=10%），
# 每部分输出两套结果：
#   1) 非留一法（样本内）：hits 表 + 8 指标检验/AUC 汇总
#   2) 留一法（材料内）：fair scores 表 + 8 指标检验/AUC 汇总
#      + 完整流程置换检验 + 图
# GPT 审查修复（与 CDR3_AA_dict_LOOCV.R 相同）：
#   1) 整数编码 + 统一全集对齐计数，修复候选/字典构建对齐 bug；
#   2) 每轮记录 RA/ILD 字典规模，hit_rate 分母 = 当轮字典规模；
#   3) 重复克隆聚合、空字典/常数指标防御；
#   4) 置换升级为完整流程置换，且仅在各自 material 子集内进行
#      （默认整体置换；PERM_STRATA=batch 可改用 batch 内分层，此时
#       子集内 material 恒定，batch 分层不涉及 material 混杂）；
#      p = (1+sum(|perm_stat|>=|obs_stat|))/(n_valid+1)，分母为各
#      指标实际有效（有限）置换数，BH-FDR + p_maxT 控制多重比较
#      （maxT 仅用 8 指标全有限的完整置换行）。
# 环境变量：N_PERM（默认 1000）、PERM_SEED（默认 20260804）、
#           PERM_STRATA（默认 "none"，可选 "batch"）
# ====================================================================

THRESHOLD_PCT <- 20
DELTA_PCT <- 10

N_PERM <- as.integer(Sys.getenv("N_PERM", unset = NA_character_))
if (is.na(N_PERM)) N_PERM <- 1000L
PERM_SEED <- as.integer(Sys.getenv("PERM_SEED", unset = NA_character_))
if (is.na(PERM_SEED)) PERM_SEED <- 20260804L
PERM_STRATA <- Sys.getenv("PERM_STRATA", unset = "none")  # none | batch
stopifnot(N_PERM >= 0L, PERM_STRATA %in% c("none", "batch"))
cat(sprintf("置换配置: N_PERM=%d, PERM_SEED=%d, PERM_STRATA=%s\n",
            N_PERM, PERM_SEED, PERM_STRATA))

metrics <- c("RA_dict_clone_count", "ILD_dict_clone_count",
             "RA_dict_hit_rate", "ILD_dict_hit_rate",
             "RA_dict_read_fraction_sum", "ILD_dict_read_fraction_sum",
             "RA_minus_ILD_count", "RA_minus_ILD_freq")
metric_labels <- c("RA 字典命中数", "ILD 字典命中数",
                   "RA 字典命中率", "ILD 字典命中率",
                   "RA 命中 read_fraction 和", "ILD 命中 read_fraction 和",
                   "RA−ILD 命中数差", "RA−ILD read_fraction 和差")

metadata <- read.csv("./TRB/metadata.csv", stringsAsFactors = FALSE)
if (anyDuplicated(metadata$libraryid)) {
  stop("metadata 中存在重复 libraryid: ",
       paste(unique(metadata$libraryid[duplicated(metadata$libraryid)]),
             collapse = ", "))
}

aa_dir <- "./TRB/result/01_AA_clone_table/"

# libraryid → 文件名 显式映射；sample_info 按名匹配（不用 merge 隐式排序）
aa_file_map <- build_aa_file_map(aa_dir = aa_dir,
                                 metadata_ids = metadata$libraryid)
sample_ids_available <- names(aa_file_map)
meta_idx <- match(sample_ids_available, metadata$libraryid)
if (anyNA(meta_idx)) {
  stop("部分 AA clone table 样本无法匹配 metadata")
}
sample_info <- metadata[meta_idx, c("libraryid", "material", "cohort", "batch"),
                        drop = FALSE]
stopifnot(identical(sample_info$libraryid, sample_ids_available))
rownames(sample_info) <- sample_info$libraryid

out_dir <- "./TRB/result/"
plot_dir <- "./TRB/gradient/dict_LOOCV/"
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

# 裁剪到 keep 子集并重编码（trim_to_keep 见 utils，保留样本名）
# 汇总表输出（含 fullperm p 值列）
make_summary_df <- function(in_res, lo_res, p_raw, q_bh, p_maxT, fullperm_n_valid) {
  data.frame(
    metric = metric_labels,
    insample_AUC = sapply(in_res, `[[`, "auc"),
    LOOCV_AUC = sapply(lo_res, `[[`, "auc"),
    AUC_drop = sapply(in_res, `[[`, "auc") - sapply(lo_res, `[[`, "auc"),
    LOOCV_AUC_lo = sapply(lo_res, `[[`, "auc_lo"),
    LOOCV_AUC_hi = sapply(lo_res, `[[`, "auc_hi"),
    LOOCV_RA_median = sapply(lo_res, `[[`, "med_ra"),
    LOOCV_ILD_median = sapply(lo_res, `[[`, "med_ild"),
    LOOCV_main_test = sapply(lo_res, `[[`, "main_test"),
    LOOCV_main_p = sapply(lo_res, `[[`, "main_p"),
    LOOCV_main_q = sapply(lo_res, `[[`, "main_q"),
    cohen_d = sapply(lo_res, `[[`, "cohen_d"),
    cliff_delta = sapply(lo_res, `[[`, "cliff_delta"),
    permutation_p = p_raw,
    fullperm_p_raw = p_raw,
    fullperm_q_BH = q_bh,
    fullperm_p_maxT = p_maxT,
    fullperm_n = N_PERM,
    fullperm_n_valid = fullperm_n_valid,
    fullperm_seed = PERM_SEED,
    fullperm_scheme = PERM_STRATA
  )
}

print_summary <- function(res, title, p_raw = NULL) {
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
  if (!is.null(p_raw)) {
    cat("  完整流程置换 p: ")
    cat(paste(sprintf("%s=%.2g", metric_labels, p_raw), collapse = "; "), "\n")
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
  lab_part <- sample_info[part_ids, "cohort"]
  cat("\n################################################################\n")
  cat(" 部分：", part, "（", length(part_ids), " 样本，RA ", n_ra, " / ILD ", n_ild, "）\n", sep = "")
  cat("################################################################\n")

  # ---- 按 part_ids 显式解析文件（不依赖位置索引） ----
  part_files <- resolve_sample_files(aa_file_map = aa_file_map,
                                     sample_ids = part_ids,
                                     aa_dir = aa_dir)
  parsed_part_ids <- sub("_AA_clone_table\\.csv$", "", basename(part_files))
  stopifnot(length(part_files) == length(part_ids),
            identical(parsed_part_ids, part_ids))
  cat(sprintf("[%s] 样本—文件映射检查通过：%d 个样本全部一一对应\n",
              part, length(part_ids)))

  # ---- 读取该 part 全部样本（整数编码，全量供样本内打分） ----
  cat("正在读取数据...\n")
  dat <- load_sample_clone_data(aa_dir = aa_dir, aa_files_f = part_files,
                                sample_ids = part_ids)

  # ============ 1) 非留一法（样本内，字典来自 enrich CSV） ============
  dicts <- list(
    RA  = fread(file.path(out_dir, paste0("CDR3_AA_enrich_RA_", part, "_T20_D10.csv")),
                select = "aa")$aa,
    ILD = fread(file.path(out_dir, paste0("CDR3_AA_enrich_ILD_", part, "_T20_D10.csv")),
                select = "aa")$aa
  )
  ra_dict_ids  <- match(dicts$RA,  dat$clone_names); ra_dict_ids  <- ra_dict_ids[!is.na(ra_dict_ids)]
  ild_dict_ids <- match(dicts$ILD, dat$clone_names); ild_dict_ids <- ild_dict_ids[!is.na(ild_dict_ids)]
  cat("样本内字典规模: RA ", length(ra_dict_ids), " | ILD ", length(ild_dict_ids), " 条\n", sep = "")

  cat("计算样本内结果...\n")
  in_rows <- lapply(part_ids, function(sid) {
    sc <- score_sample_with_dict(dat$clone_ids[[sid]], dat$rf[[sid]],
                                 ra_dict_ids, ild_dict_ids)
    data.table(libraryid = sid,
               RA_dict_clone_count = sc$ra_count,
               ILD_dict_clone_count = sc$ild_count,
               cohort = sample_info[sid, "cohort"],
               RA_dict_read_fraction_sum = round(sc$ra_rf_sum, 6),
               ILD_dict_read_fraction_sum = round(sc$ild_rf_sum, 6),
               RA_dict_hit_rate = safe_hit_rate(sc$ra_count, sc$ra_dict_size),
               ILD_dict_hit_rate = safe_hit_rate(sc$ild_count, sc$ild_dict_size),
               RA_minus_ILD_count = sc$ra_count - sc$ild_count,
               RA_minus_ILD_freq = round(sc$ra_rf_sum - sc$ild_rf_sum, 6),
               material = sample_info[sid, "material"])
  })
  in_hits <- rbindlist(in_rows)[order(cohort, libraryid)]
  write.csv(in_hits, file.path(out_dir, paste0("CDR3_AA_sample_dict_hits_", part, "_T20_D10.csv")),
            row.names = FALSE)
  in_res <- safe_analyze_metrics(in_hits, metrics, metric_labels)
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

  # ---- 预筛选（子集内数学上界）→ 裁剪重编码，供 LOOCV/置换 ----
  keep <- prescreen_clones(dat$all_tab, n_ra, n_ild, THRESHOLD_PCT)
  keep_ids <- match(keep, dat$clone_names)
  stopifnot(!anyNA(keep_ids), length(keep_ids) >= 2)
  dat_lo <- trim_to_keep(dat$clone_ids, dat$rf, keep_ids)
  clone_ids_lo <- dat_lo$clone_ids
  rf_lo <- dat_lo$rf
  U_lo <- dat_lo$U
  rm(dat_lo, dat, in_rows); gc()

  # ============ 2) 留一法（材料内） ============
  cat("留一法（", length(part_ids), " 轮）...\n", sep = "")
  hits_obs <- run_dictionary_loocv(clone_ids_lo, rf_lo, part_ids, lab_part,
                                   n_ra, n_ild, U_lo,
                                   THRESHOLD_PCT, DELTA_PCT, verbose = TRUE)
  loocv_hits <- hits_obs[order(cohort, libraryid)]
  write.csv(loocv_hits,
            file.path(out_dir, paste0("CDR3_AA_sample_dict_hits_LOOCV_", part, "_T20_D10.csv")),
            row.names = FALSE)
  cat("每轮字典规模: RA 字典 ", min(loocv_hits$RA_dict_size), "-", max(loocv_hits$RA_dict_size),
      "（均值 ", round(mean(loocv_hits$RA_dict_size)), "）条 | ILD 字典 ",
      min(loocv_hits$ILD_dict_size), "-", max(loocv_hits$ILD_dict_size),
      "（均值 ", round(mean(loocv_hits$ILD_dict_size)), "）条\n", sep = "")

  loocv_res <- safe_analyze_metrics(loocv_hits, metrics, metric_labels)

  # ---- 完整流程置换（仅在当前 material 子集内） ----
  if (N_PERM > 0L) {
    cat(sprintf("完整流程置换检验（%d 次，seed=%d，strata=%s，仅 %s 子集内）...\n",
                N_PERM, PERM_SEED, PERM_STRATA, part))
    strata <- rep("all", length(part_ids))
    if (PERM_STRATA == "batch") {
      strata <- sample_info[part_ids, "batch"]
      check_strata(sample_info[part_ids, ], "batch")
    }
    fp <- run_full_pipeline_permutation(
      clone_ids_lo, rf_lo, part_ids, lab_part, strata,
      N_PERM, PERM_SEED, U_lo,
      THRESHOLD_PCT, DELTA_PCT,
      checkpoint_prefix = file.path(out_dir, paste0("CDR3_AA_dict_fullperm_auc_", part, "_T20_D10")),
      checkpoint_every = 25L)
    perm_auc <- fp$perm_auc

    n_na <- colSums(!is.finite(perm_auc))
    if (any(n_na > 0L)) {
      cat("置换 AUC 含 NA 的指标: ",
          paste(sprintf("%s=%d", metric_labels[n_na > 0L], n_na[n_na > 0L]),
                collapse = "; "), "\n")
    } else {
      cat("置换 AUC 全部有限（NA 数 = 0）\n")
    }

    obs_auc <- sapply(loocv_res, `[[`, "auc")
    dev_obs  <- abs(obs_auc - 0.5)
    dev_perm <- abs(perm_auc - 0.5)
    p_raw  <- vapply(seq_along(metrics), function(j)
      perm_p_two_sided(dev_obs[j], dev_perm[, j]), numeric(1))
    mt <- maxT_p_values(dev_obs, dev_perm)
    p_maxT <- mt$p_maxT
    n_complete <- mt$n_complete
    cat(sprintf("maxT 完整置换行数（8 指标全 finite）: %d/%d\n", n_complete, N_PERM))
    q_bh <- p.adjust(p_raw, "BH")
    names(p_raw) <- names(p_maxT) <- names(q_bh) <- metrics
    fullperm_n_valid <- colSums(is.finite(perm_auc))
    names(fullperm_n_valid) <- metrics

    fa <- as.data.frame(perm_auc)
    fa$perm <- paste0("perm_", seq_len(N_PERM))
    write.csv(fa, file.path(out_dir, paste0("CDR3_AA_dict_fullperm_auc_", part, "_T20_D10.csv")),
              row.names = FALSE)
  } else {
    cat("N_PERM=0，跳过置换检验\n")
    p_raw <- p_maxT <- q_bh <- setNames(rep(NA_real_, length(metrics)), metrics)
    fullperm_n_valid <- setNames(rep(0L, length(metrics)), metrics)
    n_complete <- 0L
  }

  loocv_auc_all[[part]] <- sapply(loocv_res, `[[`, "auc")
  names(loocv_auc_all[[part]]) <- metrics
  print_summary(loocv_res, paste0(part, " 留一法（公平分数，T=20%, Δ=10%）"), p_raw)

  lo_sum_df <- make_summary_df(in_res, loocv_res, p_raw, q_bh, p_maxT, fullperm_n_valid)
  write.csv(lo_sum_df, file.path(out_dir, paste0("CDR3_AA_dict_LOOCV_summary_", part, "_T20_D10.csv")),
            row.names = FALSE)

  # ---- 图：LOOCV 箱线图 + ROC ----
  perm_note <- "未做置换检验"
  if (N_PERM > 0L) {
    perm_note <- sprintf("完整流程置换 %d 次（%s 子集内, seed=%d）", N_PERM, part, PERM_SEED)
  }
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
    v <- v[is.finite(v)]
    if (length(v) == 0) return(1)
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
         title = paste0("[", part, "] 完整字典构建 + LOOCV 流程置换检验：留一法公平分数 队列分布（T=20%, Δ=10%, 材料内字典）"),
         subtitle = paste0("每样本分数由不含该样本的", part, "字典计算 | RA n=", n_ra,
                           " | ILD n=", n_ild, " | 标注为主检验方法与 p 值 | ", perm_note)) +
    theme_classic(base_size = 13) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
          plot.subtitle = element_text(hjust = 0.5, size = 9.5),
          strip.text = element_text(size = 10))
  ggsave(paste0(plot_dir, "AA_dict_LOOCV_boxplot_", part, "_T20_D10.png"), p_box,
         width = 16, height = 8, dpi = 300)

  roc_df <- do.call(rbind, lapply(seq_along(metrics), function(i) {
    r <- loocv_res[[i]]
    if (is.null(r$roc)) return(NULL)   # AUC 不可用时跳过该指标
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
         title = paste0("[", part, "] 完整字典构建 + LOOCV 流程置换检验：留一法公平分数 队列判别 ROC（T=20%, Δ=10%）"),
         subtitle = paste0("RA 为阳性类；曲线在对角线下方表示该指标在 ILD 中更高 | ", perm_note)) +
    theme_classic(base_size = 14) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
          plot.subtitle = element_text(hjust = 0.5, size = 9),
          legend.position = "right", legend.text = element_text(size = 9))
  ggsave(paste0(plot_dir, "AA_dict_LOOCV_ROC_", part, "_T20_D10.png"), p_roc,
         width = 9.5, height = 7, dpi = 300)

  rm(clone_ids_lo, rf_lo); gc()
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
