library(ggplot2)
library(data.table)

setwd("/data/users/chenhaisheng/RA-ILD/")

# ====================================================================
# RA-enrich / ILD-enrich Top10 差值条形图（不进行 K/M 两步操作）
# ====================================================================
# 数据口径（用户指定）：T=20%, Δ=10%
#   RA-enrich : RA 出现率 ≥ 20% 且 RA% − ILD% ≥ 10%
#   ILD-enrich: ILD 出现率 ≥ 20% 且 ILD% − RA% ≥ 10%
# 每个部分各出 2 张（RA-enrich / ILD-enrich），共 6 张
#
# bar 结构（差值可视化重点）：
#   bar 总高度 = 主队列出现率（0-100%）
#   顶部浅灰段 = 另一队列出现率（从 bar 顶部倒过来绘制）
#   下半强调色段 = 差值（视觉重点）
# 排序：按差值从大到小（左 → 右）
# ====================================================================

THRESHOLD_PCT <- 20
DELTA_PCT <- 10

metadata <- read.csv("./TRB/metadata.csv")

aa_dir <- "./TRB/result/01_AA_clone_table/"
aa_files <- list.files(aa_dir, pattern = "_AA_clone_table\\.csv$")
aa_ids <- gsub("_AA_clone_table\\.csv$", "", aa_files)
aa_ids <- aa_ids[!grepl("^01_", aa_ids)]
aa_files_f <- paste0(aa_ids, "_AA_clone_table.csv")

cat("正在读取数据...（不进行 K/M 截取，每样本全部 clone 参与）\n")

all_cdr3 <- lapply(file.path(aa_dir, aa_files_f), function(f) {
  d <- read.csv(f, stringsAsFactors = FALSE)
  unique(d$cdr3_aa)
})
names(all_cdr3) <- aa_ids

sample_info <- merge(data.frame(libraryid = aa_ids),
                     metadata[, c("libraryid", "material", "cohort")], by = "libraryid")
rownames(sample_info) <- sample_info$libraryid

parts <- list(
  total     = sample_info$libraryid,
  pbmc      = sample_info$libraryid[sample_info$material == "PBMC"],
  buffycoat = sample_info$libraryid[sample_info$material == "buffycoat"]
)

build_clone_n <- function(clone_sets, sample_ids) {
  dt <- data.table(clone = unlist(clone_sets[sample_ids]))
  dt <- dt[!is.na(clone)]
  counts <- dt[, .N, by = clone]
  setNames(as.numeric(counts$N), counts$clone)
}

plot_dir <- "./TRB/gradient/enrich_diff/"
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

# Okabe–Ito 配色（RA 蓝 / ILD 朱红，色觉安全），浅灰弱化非差值部分
ACCENT_RA   <- "#0072B2"   # RA-enrich 差值段（Okabe–Ito blue）
ACCENT_ILD  <- "#D55E00"   # ILD-enrich 差值段（Okabe–Ito vermillion）
OTHER_GRAY  <- "#D9DEE3"   # 另一队列频率段（浅中性灰）
INK_MAIN    <- "#212121"   # 柱顶总频率文字
INK_OTHER   <- "#455A64"   # 浅灰段内文字

# ---------------- 差值条形图 ----------------
# df 列：cdr3_aa, main_pct（主队列%）, other_pct（另一队列%）, diff（差值%）
plot_diff_bar <- function(df, part_label, side, n_ra, n_ild) {
  if (nrow(df) == 0) {
    cat("  [", part_label, "] ", side, "-enrich 无候选 clone，跳过\n", sep = "")
    return(NULL)
  }
  top_n <- nrow(df)

  df$label_short <- ifelse(nchar(df$cdr3_aa) > 16,
                           paste0(substr(df$cdr3_aa, 1, 14), ".."),
                           df$cdr3_aa)
  df$label_short <- factor(df$label_short, levels = df$label_short)

  accent <- if (side == "RA") ACCENT_RA else ACCENT_ILD
  main_label  <- if (side == "RA") "RA 频率" else "ILD 频率"
  other_label <- if (side == "RA") "ILD 频率" else "RA 频率"
  diff_label  <- if (side == "RA") "RA−ILD 差值" else "ILD−RA 差值"

  # 堆叠结构：下方 = 差值段（强调色），上方 = 另一队列段（浅灰），总高 = 主队列%
  seg <- rbind(
    data.frame(x = df$label_short, y = df$diff,   seg = diff_label),
    data.frame(x = df$label_short, y = df$other_pct, seg = other_label)
  )
  seg$seg <- factor(seg$seg, levels = c(diff_label, other_label))

  p <- ggplot(seg, aes(x = x, y = y)) +
    # ggplot2(>=4.0) 默认 position="stack" 会把第一个因子水平堆叠到顶部，
    # 与预期方向相反；用 position_stack(reverse = TRUE) 让第一个水平
    # （差值段）落在底部。保持因子水平顺序 c(diff_label, other_label) 不变，
    # 勿同时颠倒水平，否则会再次抵消方向调整。
    # colour="white" 给色块加白色边框：段间形成分割线，柱外围与白底融合。
    geom_col(aes(fill = seg), position = position_stack(reverse = TRUE),
             width = 0.68, colour = "white", linewidth = 0.3) +
    # 差值标注（重点，白字粗体，段内居中偏下）
    geom_text(data = df, aes(x = label_short, y = diff / 2,
                             label = sprintf("%+.1f%%", diff)),
              colour = "white", size = 3.6, fontface = "bold") +
    # 另一队列频率标注（浅灰段内，段较矮时跳过；保留 1 位小数）
    geom_text(data = df[df$other_pct >= 5, ],
              aes(x = label_short, y = diff + other_pct / 2,
                  label = sprintf("%.1f%%", other_pct)),
              colour = INK_OTHER, size = 2.9) +
    # 主队列频率标注（bar 顶部上方；保留 1 位小数）
    geom_text(data = df, aes(x = label_short, y = main_pct + 1.8,
                             label = sprintf("%.1f%%", main_pct)),
              colour = INK_MAIN, size = 3.2, fontface = "bold") +
    scale_fill_manual(values = setNames(c(accent, OTHER_GRAY), c(diff_label, other_label))) +
    scale_y_continuous(limits = c(0, NA), breaks = seq(0, 100, 20),
                       expand = expansion(mult = c(0, 0.08))) +
    labs(x = "CDR3 AA 序列（按差值从大到小排列）",
         y = "出现频率 (%)",
         title = paste0("[", part_label, "] ", side, "-enrich Top", top_n,
                        "：", diff_label, "（T=", THRESHOLD_PCT,
                        "%, Δ=", DELTA_PCT, "%）"),
         subtitle = paste0("不截取 K/M | bar 总高 = ", main_label,
                           " | 浅灰 = ", other_label,
                           " | 深色 = ", diff_label,
                           " | RA n=", n_ra, " / ILD n=", n_ild),
         fill = "") +
    theme_classic(base_size = 14) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
          plot.subtitle = element_text(hjust = 0.5, size = 9.5),
          axis.text = element_text(color = "black"),
          axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
          axis.title = element_text(size = 12),
          legend.position = "top")

  ggsave(paste0(plot_dir, "AA_diffbar_", side, "enrich_", part_label,
                "_T", THRESHOLD_PCT, "_D", DELTA_PCT, ".png"),
         p, width = 12, height = 7, dpi = 300)
  ggsave(paste0(plot_dir, "AA_diffbar_", side, "enrich_", part_label,
                "_T", THRESHOLD_PCT, "_D", DELTA_PCT, ".pdf"),
         p, width = 12, height = 7)

  cat("  [", part_label, "] ", side, "-enrich Top", top_n, " 图已保存\n", sep = "")
}

# ====================================================================
# 主流程：三个部分 × 两侧
# ====================================================================
for (part_name in names(parts)) {
  ids <- parts[[part_name]]
  ra_ids  <- ids[sample_info[ids, "cohort"] == "RA"]
  ild_ids <- ids[sample_info[ids, "cohort"] == "ILD"]
  n_ra  <- length(ra_ids)
  n_ild <- length(ild_ids)

  cat("========================================================================\n")
  cat(" 部分：", part_name, "（", length(ids), " 个样本，RA ", n_ra, " / ILD ", n_ild, "）\n", sep = "")
  cat("========================================================================\n")

  ra_c  <- build_clone_n(all_cdr3, ra_ids)
  ild_c <- build_clone_n(all_cdr3, ild_ids)

  cand <- union(names(ra_c)[ra_c >= ceiling(n_ra * THRESHOLD_PCT / 100)],
                names(ild_c)[ild_c >= ceiling(n_ild * THRESHOLD_PCT / 100)])
  ra_n  <- unname(ifelse(is.na(ra_c[cand]), 0, ra_c[cand]))
  ild_n <- unname(ifelse(is.na(ild_c[cand]), 0, ild_c[cand]))
  pct_ra  <- ra_n / n_ra * 100
  pct_ild <- ild_n / n_ild * 100
  delta <- pct_ra - pct_ild

  # ---- RA-enrich：RA% ≥ T 且 Δ ≥ ΔT，按差值从大到小取 Top10 ----
  ra_idx <- pct_ra >= THRESHOLD_PCT & delta >= DELTA_PCT
  ra_ord <- order(delta[ra_idx], decreasing = TRUE)
  n_show_ra <- min(10, length(ra_ord))
  df_ra <- data.frame(
    cdr3_aa   = cand[ra_idx][ra_ord][1:n_show_ra],
    main_pct  = pct_ra[ra_idx][ra_ord][1:n_show_ra],
    other_pct = pct_ild[ra_idx][ra_ord][1:n_show_ra],
    diff      = delta[ra_idx][ra_ord][1:n_show_ra]
  )

  # ---- ILD-enrich：ILD% ≥ T 且 Δ ≤ −ΔT，按 ILD−RA 差值从大到小取 Top10 ----
  ild_idx <- pct_ild >= THRESHOLD_PCT & delta <= -DELTA_PCT
  ild_ord <- order(delta[ild_idx], decreasing = FALSE)
  n_show_ild <- min(10, length(ild_ord))
  df_ild <- data.frame(
    cdr3_aa   = cand[ild_idx][ild_ord][1:n_show_ild],
    main_pct  = pct_ild[ild_idx][ild_ord][1:n_show_ild],
    other_pct = pct_ra[ild_idx][ild_ord][1:n_show_ild],
    diff      = -delta[ild_idx][ild_ord][1:n_show_ild]
  )

  cat("  RA-enrich 候选:", sum(ra_idx), "条 → Top", n_show_ra, "\n")
  for (i in seq_len(nrow(df_ra))) {
    cat(sprintf("    %-22s RA %5.1f%%  ILD %5.1f%%  Δ%+6.1f%%\n",
                df_ra$cdr3_aa[i], df_ra$main_pct[i],
                df_ra$other_pct[i], df_ra$diff[i]))
  }
  cat("  ILD-enrich 候选:", sum(ild_idx), "条 → Top", n_show_ild, "\n")
  for (i in seq_len(nrow(df_ild))) {
    cat(sprintf("    %-22s ILD %5.1f%%  RA %5.1f%%  Δ%+6.1f%%\n",
                df_ild$cdr3_aa[i], df_ild$main_pct[i],
                df_ild$other_pct[i], -df_ild$diff[i]))
  }

  plot_diff_bar(df_ra,  part_name, "RA",  n_ra, n_ild)
  plot_diff_bar(df_ild, part_name, "ILD", n_ra, n_ild)

  rm(ra_c, ild_c)
  gc()
}

cat("\n所有图形生成完毕！保存至:", plot_dir, "\n")
