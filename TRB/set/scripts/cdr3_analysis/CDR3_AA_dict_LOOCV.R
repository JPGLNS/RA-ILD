library(ggplot2)
library(data.table)
library(pROC)

setwd("/data/users/chenhaisheng/RA-ILD/")
# utils 以 GitHub 同步目录（set/）为正式版本
utils_file <- "TRB/set/scripts/cdr3_analysis/CDR3_AA_dict_utils.R"
stopifnot(file.exists(utils_file))
source(utils_file)

# ====================================================================
# 留一法（LOOCV）验证：enrich 字典判别力（GPT 审查修复版）
# ====================================================================
# 分析设计不变：T=20%、Δ=10%，8 个指标，输出文件名不变。
# GPT 审查修复（2026-08-05）：
#   1) 修复候选/字典构建的对齐 bug（R 命名向量按位置回收比较，
#      不按 names 对齐）→ 整数编码 + 统一全集对齐计数；
#   2) LOOCV 每轮增量扣除留出样本 → 重建字典 → 恢复，并断言计数
#      恢复一致、非负；
#   3) 每轮记录 RA/ILD 字典规模，hit_rate 分母 = 当轮字典规模；
#   4) 重复克隆按 cdr3_aa 聚合、空字典 hit_rate=NA、常数指标防御；
#   5) 置换检验升级为完整流程置换：每轮置换重跑整个 LOOCV（字典
#      构建 + 打分），按 material 分层（保持层内 RA/ILD 数量）；
#      p = (1 + sum(|perm_stat| >= |obs_stat|)) / (n_valid + 1)，
#      分母为各指标实际有效（有限）置换数，BH-FDR + p_maxT 控制
#      8 指标多重比较（maxT 仅用 8 指标全有限的完整置换行）；
#   6) 环境变量 N_PERM / PERM_SEED / PERM_STRATA 控制置换，
#      PERM_STRATA ∈ {material, material_batch, none}，
#      material_batch = material×batch 联合分层。
# 输出：
#   1) TRB/result/CDR3_AA_sample_dict_hits_LOOCV_T20_D10.csv
#      （公平分数表，含 RA/ILD_dict_size 列）
#   2) TRB/result/CDR3_AA_dict_LOOCV_summary_T20_D10.csv
#      （对比汇总，新增 fullperm_p_raw/q_BH/p_maxT 等列）
#   3) TRB/result/CDR3_AA_dict_fullperm_auc_total_T20_D10.csv
#      （每次置换的 8 指标 AUC 矩阵）
#   4) TRB/gradient/dict_LOOCV/：LOOCV 箱线图、ROC、样本内 vs LOOCV
#      AUC 对比图（标题注明完整流程置换检验）
# ====================================================================

THRESHOLD_PCT <- 20
DELTA_PCT <- 10

N_PERM <- as.integer(Sys.getenv("N_PERM", unset = NA_character_))
if (is.na(N_PERM)) N_PERM <- 1000L
PERM_SEED <- as.integer(Sys.getenv("PERM_SEED", unset = NA_character_))
if (is.na(PERM_SEED)) PERM_SEED <- 20260804L
PERM_STRATA <- Sys.getenv("PERM_STRATA", unset = "material")  # material | material_batch | none
stopifnot(N_PERM >= 0L, PERM_STRATA %in% c("material", "material_batch", "none"))
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

# ====================================================================
# 元数据与样本划分（显式命名映射，不依赖 list.files/merge 顺序）
# ====================================================================
metadata <- read.csv("./TRB/metadata.csv", stringsAsFactors = FALSE)
if (anyDuplicated(metadata$libraryid)) {
  stop("metadata 中存在重复 libraryid: ",
       paste(unique(metadata$libraryid[duplicated(metadata$libraryid)]),
             collapse = ", "))
}

aa_dir <- "./TRB/result/01_AA_clone_table/"

# libraryid → 文件名 显式映射
aa_file_map <- build_aa_file_map(aa_dir = aa_dir,
                                 metadata_ids = metadata$libraryid)
sample_ids_available <- names(aa_file_map)

# 显式按名匹配 metadata（不用 merge 的隐式排序）
meta_idx <- match(sample_ids_available, metadata$libraryid)
if (anyNA(meta_idx)) {
  stop("部分 AA clone table 样本无法匹配 metadata")
}
sample_info <- metadata[meta_idx, c("libraryid", "material", "cohort", "batch"),
                        drop = FALSE]
stopifnot(identical(sample_info$libraryid, sample_ids_available))
rownames(sample_info) <- sample_info$libraryid
sample_ids <- sample_info$libraryid

# 显式解析每个样本的文件（严格按 sample_ids 顺序）
sample_files <- resolve_sample_files(aa_file_map = aa_file_map,
                                     sample_ids = sample_ids,
                                     aa_dir = aa_dir)
parsed_ids <- sub("_AA_clone_table\\.csv$", "", basename(sample_files))
stopifnot(length(sample_files) == length(sample_ids),
          identical(parsed_ids, sample_ids))
cat(sprintf("样本—文件映射检查通过：%d 个样本全部一一对应\n", length(sample_ids)))
lab_obs <- sample_info$cohort
ra_ids  <- sample_ids[lab_obs == "RA"]
ild_ids <- sample_ids[lab_obs == "ILD"]
n_ra  <- length(ra_ids)
n_ild <- length(ild_ids)
cat("RA:", n_ra, "样本 | ILD:", n_ild, "样本\n")
validate_metadata(metadata, sample_ids)
if (PERM_STRATA == "material_batch") {
  # material×batch 联合分层检查（interaction 因子无对应列，手写检查）
  strat_chk <- interaction(sample_info$material, sample_info$batch, drop = TRUE)
  tab <- table(strat_chk, sample_info$cohort)
  one_sided <- rownames(tab)[apply(tab, 1, function(r) any(r == 0))]
  if (length(one_sided) > 0) {
    warning("以下 material×batch stratum 只含单一 cohort，无法贡献标签交换: ",
            paste(one_sided, collapse = ", "))
  }
} else {
  check_strata(sample_info, "material")
  if (PERM_STRATA != "none") check_strata(sample_info, PERM_STRATA)
}

# ====================================================================
# 读取全部样本（整数编码）+ 标签无关安全预筛选
# ====================================================================
cat("正在读取数据...\n")
dat <- load_sample_clone_data(aa_dir, sample_files, sample_ids)
keep <- prescreen_clones(dat$all_tab, n_ra, n_ild, THRESHOLD_PCT)
keep_ids <- match(keep, dat$clone_names)
stopifnot(!anyNA(keep_ids), length(keep_ids) >= 2)

# 裁剪到 keep 子集并重编码（trim_to_keep 见 utils，保留样本名）
dat_lo <- trim_to_keep(dat$clone_ids, dat$rf, keep_ids)
clone_ids_lo <- dat_lo$clone_ids
rf_lo <- dat_lo$rf
U_lo <- dat_lo$U
rm(dat_lo, dat); gc()

# ====================================================================
# 留一法（observed 公平分数）
# ====================================================================
cat("开始留一法（", length(sample_ids), " 轮）...\n", sep = "")
hits_obs <- run_dictionary_loocv(clone_ids_lo, rf_lo, sample_ids, lab_obs,
                                 n_ra, n_ild, U_lo,
                                 THRESHOLD_PCT, DELTA_PCT, verbose = TRUE)
loocv_hits <- hits_obs[order(cohort, libraryid)]
write.csv(loocv_hits, "./TRB/result/CDR3_AA_sample_dict_hits_LOOCV_T20_D10.csv",
          row.names = FALSE)

cat("每轮字典规模: RA 字典 ", min(loocv_hits$RA_dict_size), "-", max(loocv_hits$RA_dict_size),
    "（均值 ", round(mean(loocv_hits$RA_dict_size)), "）条 | ILD 字典 ",
    min(loocv_hits$ILD_dict_size), "-", max(loocv_hits$ILD_dict_size),
    "（均值 ", round(mean(loocv_hits$ILD_dict_size)), "）条\n", sep = "")

# ====================================================================
# 指标分析（LOOCV + 样本内）
# ====================================================================
cat("分析留一法公平分数...\n")
loocv_res <- safe_analyze_metrics(loocv_hits, metrics, metric_labels)

insample_hits <- fread("./TRB/result/CDR3_AA_sample_dict_hits_T20_D10.csv")
cat("分析样本内分数...\n")
insample_res <- safe_analyze_metrics(insample_hits, metrics, metric_labels)

# ====================================================================
# 完整流程置换检验：每轮置换重跑整个 LOOCV（分层）
# ====================================================================
if (N_PERM > 0L) {
  cat(sprintf("完整流程置换检验（%d 次，seed=%d，strata=%s）...\n",
              N_PERM, PERM_SEED, PERM_STRATA))
  strata <- rep("all", length(sample_ids))
  if (PERM_STRATA == "material") strata <- sample_info$material
  if (PERM_STRATA == "material_batch") {
    strata <- interaction(sample_info$material, sample_info$batch, drop = TRUE)
  }
  fp <- run_full_pipeline_permutation(
    clone_ids_lo, rf_lo, sample_ids, lab_obs, strata,
    N_PERM, PERM_SEED, U_lo,
    THRESHOLD_PCT, DELTA_PCT,
    checkpoint_prefix = "./TRB/result/CDR3_AA_dict_fullperm_auc_total_T20_D10",
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
  dev_obs  <- abs(obs_auc - 0.5)                 # 统计量：偏离随机水平 0.5 的幅度
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

  # 置换 AUC 矩阵导出
  fa <- as.data.frame(perm_auc)
  fa$perm <- paste0("perm_", seq_len(N_PERM))
  write.csv(fa, "./TRB/result/CDR3_AA_dict_fullperm_auc_total_T20_D10.csv",
            row.names = FALSE)
} else {
  cat("N_PERM=0，跳过置换检验\n")
  p_raw <- p_maxT <- q_bh <- setNames(rep(NA_real_, length(metrics)), metrics)
  fullperm_n_valid <- setNames(rep(0L, length(metrics)), metrics)
  n_complete <- 0L
}

# ====================================================================
# 汇总对比表（控制台 + CSV）
# ====================================================================
cat("\n========================================================================\n")
cat(" 留一法 vs 样本内（T=20%, Δ=10%, total 字典）——完整字典构建 + LOOCV 流程置换\n")
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
              eff, p_raw[i], r_lo$auc_lo, r_lo$auc_hi))
}
if (N_PERM > 0L) {
  cat("\n完整流程置换检验 p 值（N_PERM=", N_PERM, ", seed=", PERM_SEED,
      ", 分层=", PERM_STRATA, "）：\n", sep = "")
  for (i in seq_along(metrics)) {
    cat(sprintf("  %-22s p_raw=%9.3g   q_BH=%9.3g   p_maxT=%9.3g\n",
                metric_labels[i], p_raw[i], q_bh[i], p_maxT[i]))
  }
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
  permutation_p = p_raw,
  fullperm_p_raw = p_raw,
  fullperm_q_BH = q_bh,
  fullperm_p_maxT = p_maxT,
  fullperm_n = N_PERM,
  fullperm_n_valid = fullperm_n_valid,
  fullperm_seed = PERM_SEED,
  fullperm_scheme = PERM_STRATA
)
write.csv(summary_df, "./TRB/result/CDR3_AA_dict_LOOCV_summary_T20_D10.csv",
          row.names = FALSE)

# ====================================================================
# 图
# ====================================================================
plot_dir <- "./TRB/gradient/dict_LOOCV/"
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)
perm_note <- "未做置换检验"
if (N_PERM > 0L) {
  scheme_desc <- switch(PERM_STRATA,
                        material = "material 内分层",
                        material_batch = "material×batch 联合分层",
                        none = "无分层（整体置换）")
  perm_note <- sprintf("完整流程置换 %d 次（%s, seed=%d）",
                       N_PERM, scheme_desc, PERM_SEED)
}

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
  v <- v[is.finite(v)]
  if (length(v) == 0) return(1)
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
       title = "完整字典构建 + LOOCV 流程置换检验：留一法公平分数 队列分布（T=20%, Δ=10%）",
       subtitle = paste0("每样本分数由不含该样本的字典计算 | 蓝色=RA (n=108) | 朱红=ILD (n=66) | ",
                         "标注为主检验方法与 p 值 | ", perm_note)) +
  theme_classic(base_size = 13) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
        plot.subtitle = element_text(hjust = 0.5, size = 9.5),
        strip.text = element_text(size = 10))
ggsave(paste0(plot_dir, "AA_dict_LOOCV_boxplot_T20_D10.png"), p_box,
       width = 16, height = 8, dpi = 300)

# ---- 2) LOOCV ROC 曲线 ----
roc_df <- do.call(rbind, lapply(seq_along(metrics), function(i) {
  r <- loocv_res[[i]]
  if (is.null(r$roc)) return(NULL)   # AUC 不可用时跳过该指标
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
       title = "完整字典构建 + LOOCV 流程置换检验：留一法公平分数 队列判别 ROC（T=20%, Δ=10%）",
       subtitle = paste0("RA 为阳性类；曲线在对角线下方表示该指标在 ILD 中更高 | ", perm_note)) +
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
       title = "完整字典构建 + LOOCV 流程置换检验：样本内 vs 留一法 AUC（T=20%, Δ=10%, total 字典）",
       subtitle = paste0("浅灰点 = 样本内（乐观上限）| 蓝点 = 留一法（真实判别力）| 虚线 = 随机水平 0.5 | ",
                         perm_note)) +
  theme_classic(base_size = 13) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
        plot.subtitle = element_text(hjust = 0.5, size = 9.5),
        axis.text.y = element_text(size = 10))
ggsave(paste0(plot_dir, "AA_dict_LOOCV_vs_insample_T20_D10.png"), p_cmp,
       width = 9, height = 6, dpi = 300)

cat("\n全部完成！\n")
cat("表: ./TRB/result/CDR3_AA_sample_dict_hits_LOOCV_T20_D10.csv\n")
cat("    ./TRB/result/CDR3_AA_dict_LOOCV_summary_T20_D10.csv\n")
cat("    ./TRB/result/CDR3_AA_dict_fullperm_auc_total_T20_D10.csv\n")
cat("图: ./TRB/gradient/dict_LOOCV/AA_dict_LOOCV_*.png\n")
