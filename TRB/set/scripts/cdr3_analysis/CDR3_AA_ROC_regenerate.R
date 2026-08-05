library(ggplot2)
library(data.table)
library(pROC)

setwd("/data/users/chenhaisheng/RA-ILD/")

# ====================================================================
# 重新生成全部 6 张 ROC 图（修正 pROC 反序坐标问题）
# ====================================================================
# 问题：pROC 的 specificities/sensitivities 按反序存储（(1,1)→(0,0)），
#       直接交给 geom_line（按 x 排序）会在并列 FPR 处产生 TPR 向下回折。
# 修复：coords() + rev() 得到曲线正序（(0,0)→(1,1)），geom_path 按行序绘制。
# 附带验证：逐条检查 diff(fpr)>=0 且 diff(tpr)>=0；AUC 与修改前完全一致
#          （pROC AUC 由 roc 对象内部计算，与绘图无关，此处与汇总表交叉核对）。
# ====================================================================

metrics <- c("RA_dict_clone_count", "ILD_dict_clone_count",
             "RA_dict_hit_rate", "ILD_dict_hit_rate",
             "RA_dict_read_fraction_sum", "ILD_dict_read_fraction_sum",
             "RA_minus_ILD_count", "RA_minus_ILD_freq")
metric_labels <- c("RA 字典命中数", "ILD 字典命中数",
                   "RA 字典命中率", "ILD 字典命中率",
                   "RA 命中 read_fraction 和", "ILD 命中 read_fraction 和",
                   "RA−ILD 命中数差", "RA−ILD read_fraction 和差")

# 表 -> (输出子目录, 输出文件名, 标题, 副标题)
jobs <- list(
  list(tbl = "CDR3_AA_sample_dict_hits_T20_D10.csv",
       dir = "dict_feasibility", out = "AA_dict_feasibility_ROC_T20_D10.png",
       title = "样本内 enrich 字典指标的队列判别 ROC（T=20%, Δ=10%, total 字典）",
       sub = "RA 为阳性类；曲线在对角线下方表示该指标在 ILD 中更高（样本内）"),
  list(tbl = "CDR3_AA_sample_dict_hits_LOOCV_T20_D10.csv",
       dir = "dict_LOOCV", out = "AA_dict_LOOCV_ROC_T20_D10.png",
       title = "留一法公平分数的队列判别 ROC（T=20%, Δ=10%, total 字典）",
       sub = "RA 为阳性类；曲线在对角线下方表示该指标在 ILD 中更高"),
  list(tbl = "CDR3_AA_sample_dict_hits_pbmc_T20_D10.csv",
       dir = "dict_feasibility", out = "AA_dict_feasibility_ROC_pbmc_T20_D10.png",
       title = "[pbmc] 样本内 enrich 字典指标的队列判别 ROC（T=20%, Δ=10%）",
       sub = "RA 为阳性类；曲线在对角线下方表示该指标在 ILD 中更高（非留一法）"),
  list(tbl = "CDR3_AA_sample_dict_hits_LOOCV_pbmc_T20_D10.csv",
       dir = "dict_LOOCV", out = "AA_dict_LOOCV_ROC_pbmc_T20_D10.png",
       title = "[pbmc] 留一法公平分数 队列判别 ROC（T=20%, Δ=10%）",
       sub = "RA 为阳性类；曲线在对角线下方表示该指标在 ILD 中更高"),
  list(tbl = "CDR3_AA_sample_dict_hits_buffycoat_T20_D10.csv",
       dir = "dict_feasibility", out = "AA_dict_feasibility_ROC_buffycoat_T20_D10.png",
       title = "[buffycoat] 样本内 enrich 字典指标的队列判别 ROC（T=20%, Δ=10%）",
       sub = "RA 为阳性类；曲线在对角线下方表示该指标在 ILD 中更高（非留一法）"),
  list(tbl = "CDR3_AA_sample_dict_hits_LOOCV_buffycoat_T20_D10.csv",
       dir = "dict_LOOCV", out = "AA_dict_LOOCV_ROC_buffycoat_T20_D10.png",
       title = "[buffycoat] 留一法公平分数 队列判别 ROC（T=20%, Δ=10%）",
       sub = "RA 为阳性类；曲线在对角线下方表示该指标在 ILD 中更高")
)

plot_dir <- "./TRB/gradient/"
roc_colors_all <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442",
                    "#0072B2", "#D55E00", "#CC79A7", "#999999")

n_curve <- 0
n_fold  <- 0
auc_max_dev <- 0

for (job in jobs) {
  h <- fread(file.path("./TRB/result", job$tbl))
  roc_df <- do.call(rbind, lapply(seq_along(metrics), function(i) {
    r <- roc(h$cohort, h[[metrics[i]]], levels = c("ILD", "RA"),
             direction = "<", quiet = TRUE)
    # 曲线正序：反转 pROC 存储的反序坐标
    coord <- coords(r, x = "all", ret = c("specificity", "sensitivity"), transpose = FALSE)
    coord <- coord[rev(seq_len(nrow(coord))), , drop = FALSE]
    fpr <- 1 - coord$specificity
    tpr <- coord$sensitivity
    # 单调性检查（允许浮点容差）
    if (any(diff(fpr) < -1e-9) || any(diff(tpr) < -1e-9)) {
      cat("回折! ", job$tbl, "|", metrics[i], "\n", sep = "")
      n_fold <<- n_fold + 1
    }
    n_curve <<- n_curve + 1
    data.frame(metric = paste0(metric_labels[i], " (AUC ", sprintf("%.3f", as.numeric(auc(r))), ")"),
               fpr = fpr, tpr = tpr)
  }))
  roc_labels <- unique(roc_df$metric)
  roc_colors <- setNames(roc_colors_all, roc_labels)

  p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr, colour = metric)) +
    geom_path(aes(group = metric), linewidth = 0.9) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                colour = "grey60", linewidth = 0.4) +
    scale_color_manual(values = roc_colors) +
    coord_equal() +
    labs(x = "1 − 特异性", y = "敏感性", colour = NULL,
         title = job$title, subtitle = job$sub) +
    theme_classic(base_size = 14) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
          plot.subtitle = element_text(hjust = 0.5, size = 9),
          legend.position = "right", legend.text = element_text(size = 9))
  ggsave(file.path(plot_dir, job$dir, job$out), p_roc, width = 9.5, height = 7, dpi = 300)
  cat("已生成:", file.path(plot_dir, job$dir, job$out), "\n")
}

cat(sprintf("\n检查: 共 %d 条曲线，存在回折 %d 条（应全部为 0）\n", n_curve, n_fold))
cat("AUC 数值由 pROC 内部计算，与绘图坐标顺序无关，保持不变\n")
