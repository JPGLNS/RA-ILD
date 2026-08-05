library(ggplot2)
library(data.table)

setwd("/data/users/chenhaisheng/RA-ILD/")

# ====================================================================
# CDR3 AA Clone 高频统计 + delta 4分类（不进行 K/M 两步操作）
# ====================================================================
# 与旧脚本的差异：
#   - 不截取 K=12000 / M=10000，每个样本的全部 clone 都参与统计
# 参数（沿用旧脚本，不修改）：
#   - 高频统计阈值 T ∈ {10%, 20%}（同 CDR3_AA_high_freq_clones.R）
#   - 组间差异 Δ ∈ {5%, 10%}（同 CDR3_AA_4class_analysis.R）
# 三个部分（每部分内部按 RA vs ILD 比较）：
#   - total     ：全部样本（RA 108 / ILD 66）
#   - pbmc      ：material == PBMC（RA 67 / ILD 44）
#   - buffycoat ：material == buffycoat（RA 41 / ILD 22）
# 每部分输出：
#   1) 高频统计：汇总、RA/ILD 交集、明细列表、Top30 柱状图
#      → TRB/gradient/high_freq/
#   2) 4分类：T × Δ 组合计数汇总、Top 列表、散点图 + 柱状图
#      → TRB/gradient/4class/
# ====================================================================

THRESHOLD_PCTS <- c(10, 20)
DELTAS <- c(5, 10)

metadata <- read.csv("./TRB/metadata.csv")

aa_dir <- "./TRB/result/01_AA_clone_table/"
aa_files <- list.files(aa_dir, pattern = "_AA_clone_table\\.csv$")
aa_ids <- gsub("_AA_clone_table\\.csv$", "", aa_files)
aa_ids <- aa_ids[!grepl("^01_", aa_ids)]
aa_files_f <- paste0(aa_ids, "_AA_clone_table.csv")

cat("正在读取数据...（不进行 K/M 截取，每样本全部 clone 参与）\n")

all_cdr3 <- lapply(file.path(aa_dir, aa_files_f), function(f) {
  data <- read.csv(f, stringsAsFactors = FALSE)
  unique(data$cdr3_aa)   # 仅去重，不截取 top-K / top-M
})
names(all_cdr3) <- aa_ids

sample_info <- data.frame(libraryid = aa_ids, stringsAsFactors = FALSE)
sample_info <- merge(sample_info, metadata[, c("libraryid", "material", "cohort", "patient")],
                     by = "libraryid")
rownames(sample_info) <- sample_info$libraryid

# ---- 三个部分：total / pbmc / buffycoat ----
parts <- list(
  total     = sample_info$libraryid,
  pbmc      = sample_info$libraryid[sample_info$material == "PBMC"],
  buffycoat = sample_info$libraryid[sample_info$material == "buffycoat"]
)
for (p in names(parts)) {
  ids <- parts[[p]]
  cat(sprintf("  %-10s %3d 个样本 (RA %3d, ILD %3d)\n", p, length(ids),
              sum(sample_info[ids, "cohort"] == "RA"),
              sum(sample_info[ids, "cohort"] == "ILD")))
}

# ---- 统计组内每个 clone 出现的样本数（data.table 分组计数；每组样本内 clone 无重复，.N 即出现样本数）----
build_clone_n <- function(clone_sets, sample_ids) {
  dt <- data.table(clone = unlist(clone_sets[sample_ids]))
  dt <- dt[!is.na(clone)]          # 排除 NA 序列
  counts <- dt[, .N, by = clone]
  setNames(as.numeric(counts$N), counts$clone)
}

plot_dir <- "./TRB/gradient/high_freq/"
plot_dir_4class <- "./TRB/gradient/4class/"
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(plot_dir_4class, showWarnings = FALSE, recursive = TRUE)

ra_color  <- "#4DBBD5"
ild_color <- "#E64B35"
four_colors <- c("RA-enrich"  = "#C62828",
                 "ILD-enrich" = "#1565C0",
                 "Shared"     = "#7CB342",
                 "Ambiguous"  = "#9E9E9E")

# ---------------- 高频 Top30 柱状图 ----------------
plot_high_freq <- function(high_freq, n_total, threshold_n, threshold_pct,
                           part_label, cohort_label, bar_color) {
  if (length(high_freq) == 0) return(NULL)

  top_n <- min(30, length(high_freq))
  hf_sub <- high_freq[1:top_n]

  df <- data.frame(
    cdr3_aa = as.character(names(hf_sub)),
    n_samples = as.numeric(hf_sub),
    stringsAsFactors = FALSE
  )
  df <- df[order(df$n_samples, decreasing = TRUE), ]
  df$cdr3_aa <- factor(df$cdr3_aa, levels = df$cdr3_aa)
  df$pct_label <- paste0(df$n_samples, " (", sprintf("%.1f", df$n_samples / n_total * 100), "%)")
  df$label_short <- ifelse(nchar(as.character(df$cdr3_aa)) > 18,
                           paste0(substr(as.character(df$cdr3_aa), 1, 16), ".."),
                           as.character(df$cdr3_aa))

  subtitle_text <- paste0("不截取 K/M | ", n_total, " 样本 | 共 ",
                          length(high_freq), " 条 (≥", threshold_pct, "%) | 图中仅展示前 ", top_n)

  p <- ggplot(df, aes(x = label_short, y = n_samples)) +
    geom_bar(stat = "identity", fill = bar_color, width = 0.7) +
    geom_text(aes(label = pct_label), vjust = -0.3, size = 3.0) +
    geom_hline(yintercept = threshold_n, linetype = "dashed", color = "grey40", linewidth = 0.5) +
    annotate("text", x = nrow(df) * 0.85, y = threshold_n,
             label = paste0(threshold_pct, "% 阈值 (", threshold_n, " 样本)"),
             vjust = -0.6, size = 3.5, color = "grey40") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(x = "CDR3 AA 序列（前 30）",
         y = "出现的样本数",
         title = paste0("[", part_label, "] ", cohort_label, ": ≥", threshold_pct,
                        "% 样本高频 CDR3 AA Clone (Top ", top_n, ")"),
         subtitle = subtitle_text) +
    theme_classic(base_size = 14) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
          plot.subtitle = element_text(hjust = 0.5, size = 10),
          axis.text = element_text(color = "black"),
          axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
          axis.title = element_text(size = 13))

  ggsave(paste0(plot_dir, "AA_high_freq_", part_label, "_", cohort_label, "_",
                threshold_pct, "pct.png"), p, width = 12, height = 6, dpi = 300)
  ggsave(paste0(plot_dir, "AA_high_freq_", part_label, "_", cohort_label, "_",
                threshold_pct, "pct.pdf"), p, width = 12, height = 6)

  cat("  [", part_label, "] ", cohort_label, " ", threshold_pct, "% Top30 图已保存\n", sep = "")
}

all_hf_summary <- list()   # 高频统计汇总（含交集用序列）
all_4c_summary <- list()   # 4分类汇总

for (part_name in names(parts)) {
  ids <- parts[[part_name]]
  ra_ids  <- ids[sample_info[ids, "cohort"] == "RA"]
  ild_ids <- ids[sample_info[ids, "cohort"] == "ILD"]
  n_ra  <- length(ra_ids)
  n_ild <- length(ild_ids)

  cat("========================================================================\n")
  cat(" 部分：", part_name, "（", length(ids), " 个样本，RA ", n_ra, " / ILD ", n_ild, "）\n", sep = "")
  cat("========================================================================\n")

  # ---- 组内出现样本数（不 K/M）----
  ra_clone_n  <- build_clone_n(all_cdr3, ra_ids)
  ild_clone_n <- build_clone_n(all_cdr3, ild_ids)
  ra_count  <- ra_clone_n
  ild_count <- ild_clone_n
  cat("  RA 唯一 Clone:", scales::comma(length(ra_clone_n)),
      "  ILD 唯一 Clone:", scales::comma(length(ild_clone_n)), "\n\n")

  # ============ 1) 高频统计（阈值 10%/20%，沿用旧脚本逻辑） ============
  for (tp in THRESHOLD_PCTS) {
    ra_t  <- ceiling(n_ra  * tp / 100)
    ild_t <- ceiling(n_ild * tp / 100)
    ra_hf  <- sort(ra_clone_n[ra_clone_n >= ra_t], decreasing = TRUE)
    ild_hf <- sort(ild_clone_n[ild_clone_n >= ild_t], decreasing = TRUE)

    cat(sprintf("  [高频] %d%% 阈值 (RA≥%d, ILD≥%d)\n", tp, ra_t, ild_t))
    cat(sprintf("    RA : %s 条高频 (占 RA 唯一 %s 的 %.4f%%)\n",
                scales::comma(length(ra_hf)), scales::comma(length(ra_clone_n)),
                length(ra_hf) / length(ra_clone_n) * 100))
    cat(sprintf("    ILD: %s 条高频 (占 ILD 唯一 %s 的 %.4f%%)\n",
                scales::comma(length(ild_hf)), scales::comma(length(ild_clone_n)),
                length(ild_hf) / length(ild_clone_n) * 100))

    all_hf_summary[[paste(part_name, "RA", tp, sep = "_")]] <- list(
      part = part_name, cohort = "RA", threshold_pct = tp, threshold_n = ra_t,
      n_total = n_ra, n_uniq = length(ra_clone_n), n_high = length(ra_hf),
      hf_seq = names(ra_hf))
    all_hf_summary[[paste(part_name, "ILD", tp, sep = "_")]] <- list(
      part = part_name, cohort = "ILD", threshold_pct = tp, threshold_n = ild_t,
      n_total = n_ild, n_uniq = length(ild_clone_n), n_high = length(ild_hf),
      hf_seq = names(ild_hf))

    # 明细列表（沿用旧脚本格式，打印全部高频 clone）
    for (cohort_name in c("RA", "ILD")) {
      hf <- if (cohort_name == "RA") ra_hf else ild_hf
      thr_n <- if (cohort_name == "RA") ra_t else ild_t
      n_tot <- if (cohort_name == "RA") n_ra else n_ild
      if (length(hf) == 0) next
      cat(sprintf("  --- [%s] %s %d%% 阈值 (≥%d 样本) 高频 Clone: %d 条 ---\n",
                  part_name, cohort_name, tp, thr_n, length(hf)))
      cat(sprintf("  %-25s %8s %10s\n", "CDR3 AA", "样本数", "占队列%"))
      for (i in seq_along(hf)) {
        cat(sprintf("  %-25s %8d %9.1f%%\n", names(hf)[i], hf[i], hf[i] / n_tot * 100))
      }
      cat("\n")
    }

    # Top30 柱状图（RA/ILD 各一张）
    plot_high_freq(ra_hf, n_ra, ra_t, tp, part_name, "RA", ra_color)
    plot_high_freq(ild_hf, n_ild, ild_t, tp, part_name, "ILD", ild_color)
    cat("\n")
  }

  # ============ 2) delta 4分类（T × Δ，规则同 4class 脚本） ============
  for (T_pct in THRESHOLD_PCTS) {
    ra_t  <- ceiling(n_ra  * T_pct / 100)
    ild_t <- ceiling(n_ild * T_pct / 100)

    for (d in DELTAS) {
      # 候选 clone：至少在一边 ≥ 阈值
      candidates <- union(names(ra_clone_n)[ra_clone_n >= ra_t],
                          names(ild_clone_n)[ild_clone_n >= ild_t])

      if (length(candidates) == 0) {
        cat(sprintf("  [4分类] T=%d%% Δ=%d%%: 无候选 clone，跳过\n", T_pct, d))
        next
      }

      # unname(): 候选 clone 可能只存在于一侧，索引另一侧时产生 NA 名字，
      # 不去掉会被 data.frame 当作缺失行名报错
      base <- data.frame(
        cdr3_aa = candidates,
        ra_n  = unname(ifelse(is.na(ra_count[candidates]), 0, ra_count[candidates])),
        ild_n = unname(ifelse(is.na(ild_count[candidates]), 0, ild_count[candidates])),
        stringsAsFactors = FALSE
      )
      base$ra_pct  <- base$ra_n  / n_ra  * 100
      base$ild_pct <- base$ild_n / n_ild * 100
      base$delta   <- base$ra_pct - base$ild_pct

      ra_ok  <- base$ra_pct  >= T_pct
      ild_ok <- base$ild_pct >= T_pct

      # 4 分类规则（同 4class 脚本）：
      #   RA-enrich:  p_RA ≥ T 且 p_RA - p_ILD ≥ Δ
      #   ILD-enrich: p_ILD ≥ T 且 p_ILD - p_RA ≥ Δ
      #   Shared:     p_RA ≥ T 且 p_ILD ≥ T 且 |p_RA - p_ILD| < Δ
      #   Ambiguous:  仅一组 ≥ T，且 |p_RA - p_ILD| < Δ
      base$category <- NA_character_
      base$category[ra_ok & base$delta >= d] <- "RA-enrich"
      base$category[ild_ok & base$delta <= -d] <- "ILD-enrich"
      base$category[ra_ok & ild_ok & abs(base$delta) < d & is.na(base$category)] <- "Shared"
      base$category[is.na(base$category) &
                      ((ra_ok & !ild_ok) | (!ra_ok & ild_ok)) &
                      abs(base$delta) < d] <- "Ambiguous"
      leftover <- is.na(base$category)
      if (any(leftover)) {
        cat("  WARNING:", sum(leftover), "条未分类，归入 Ambiguous\n")
        base$category[leftover] <- "Ambiguous"
      }

      base$category <- factor(base$category,
                              levels = c("RA-enrich", "ILD-enrich", "Shared", "Ambiguous"))
      base <- base[order(base$category, -base$delta), ]

      counts <- table(base$category)
      all_4c_summary[[paste(part_name, T_pct, d, sep = "_")]] <- list(
        part = part_name, T = T_pct, D = d,
        ra_t = ra_t, ild_t = ild_t, n_total = nrow(base), counts = counts)

      cat(sprintf("  [4分类] T=%d%% (RA≥%d, ILD≥%d) | Δ=%d%% | 候选 %d 条\n",
                  T_pct, ra_t, ild_t, d, nrow(base)))
      for (cat_name in levels(base$category)) {
        n_cat <- sum(base$category == cat_name)
        cat(sprintf("    %-15s %5d 条 (%5.1f%%)\n", cat_name, n_cat, n_cat / nrow(base) * 100))
      }

      # Top 展示（RA-enrich / ILD-enrich / Shared 各 Top 10）
      for (cat_name in c("RA-enrich", "ILD-enrich", "Shared")) {
        sub <- base[base$category == cat_name, ]
        n_show <- min(10, nrow(sub))
        if (n_show == 0) next
        cat(sprintf("    ── %s Top %d ──\n", cat_name, n_show))
        cat(sprintf("    %-25s %8s %8s %9s %9s %8s\n",
                    "CDR3 AA", "RA(n)", "ILD(n)", "RA%", "ILD%", "Delta%"))
        for (i in 1:n_show) {
          cat(sprintf("    %-25s %8d %8d %8.1f%% %8.1f%% %+7.1f%%\n",
                      sub$cdr3_aa[i], sub$ra_n[i], sub$ild_n[i],
                      sub$ra_pct[i], sub$ild_pct[i], sub$delta[i]))
        }
        cat("\n")
      }

      # 散点图（候选过多时随机抽样显示）
      plot_base <- base
      if (nrow(base) > 50000) {
        set.seed(42)
        plot_base <- base[sample.int(nrow(base), 50000), ]
        cat(sprintf("    散点图点数过多，随机抽样 50000 / %d 条展示\n", nrow(base)))
      }
      p1 <- ggplot(plot_base, aes(x = ra_pct, y = ild_pct, color = category)) +
        geom_point(alpha = 0.65, size = 2) +
        geom_abline(slope = 1, intercept = d,
                    linetype = "dashed", color = "#C62828", linewidth = 0.4) +
        geom_abline(slope = 1, intercept = -d,
                    linetype = "dashed", color = "#1565C0", linewidth = 0.4) +
        geom_abline(slope = 1, intercept = 0,
                    linetype = "dotted", color = "grey50", linewidth = 0.3) +
        geom_vline(xintercept = T_pct, linetype = "dotted", color = "grey70", linewidth = 0.3) +
        geom_hline(yintercept = T_pct, linetype = "dotted", color = "grey70", linewidth = 0.3) +
        scale_color_manual(values = four_colors, name = "类别", drop = FALSE) +
        labs(x = paste0("RA 出现率 (% , n=", n_ra, ")"),
             y = paste0("ILD 出现率 (% , n=", n_ild, ")"),
             title = paste0("[", part_name, "] CDR3 AA 队列特异性 (不截取 K/M, T=",
                            T_pct, "%, Δ=", d, "%)"),
             subtitle = paste0("RA-enrich:", counts["RA-enrich"],
                               " | ILD-enrich:", counts["ILD-enrich"],
                               " | Shared:", counts["Shared"],
                               " | Ambiguous:", counts["Ambiguous"])) +
        theme_classic(base_size = 14) +
        theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
              plot.subtitle = element_text(hjust = 0.5, size = 9),
              axis.text = element_text(color = "black"),
              legend.position = "bottom")
      ggsave(paste0(plot_dir_4class, "AA_4class_scatter_", part_name, "_T", T_pct, "_D", d, ".png"),
             p1, width = 9, height = 8.5, dpi = 300)

      # 柱状图
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
             title = paste0("[", part_name, "] CDR3 AA 4分类 (不截取 K/M, T=",
                            T_pct, "%, Δ=", d, "%)")) +
        theme_classic(base_size = 14) +
        theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
              axis.text = element_text(color = "black", size = 11),
              axis.text.x = element_text(size = 12, face = "bold", angle = 20, hjust = 1))
      ggsave(paste0(plot_dir_4class, "AA_4class_bar_", part_name, "_T", T_pct, "_D", d, ".png"),
             p2, width = 7.5, height = 5.5, dpi = 300)
      cat(sprintf("  [4分类] T=%d%% Δ=%d%% 图已保存\n\n", T_pct, d))
    }
  }

  # 释放本部分的大对象
  rm(ra_clone_n, ild_clone_n, ra_count, ild_count)
  gc()
}

# ====================================================================
# 汇总对比表：高频统计
# ====================================================================
cat("================================================================================\n")
cat(" 汇总对比：高频统计（不截取 K/M）\n")
cat("================================================================================\n\n")

header <- sprintf("%-10s %-6s %8s %8s %12s %12s %10s",
                  "部分", "队列", "阈值%", "阈值N", "唯一Clone", "高频Clone", "占比")
cat(header, "\n")
cat(paste(rep("-", nchar(header)), collapse = ""), "\n")

for (key in names(all_hf_summary)) {
  r <- all_hf_summary[[key]]
  cat(sprintf("%-10s %-6s %7d%% %8d %12s %12s %11.4f%%\n",
              r$part, r$cohort, r$threshold_pct, r$threshold_n,
              scales::comma(r$n_uniq), scales::comma(r$n_high),
              r$n_high / r$n_uniq * 100))
}

# 交集
cat("\n--- 各部分 RA/ILD 高频 Clone 交集 ---\n")
for (part_name in names(parts)) {
  for (tp in THRESHOLD_PCTS) {
    ra_key  <- paste(part_name, "RA",  tp, sep = "_")
    ild_key <- paste(part_name, "ILD", tp, sep = "_")
    ra_seqs  <- all_hf_summary[[ra_key]]$hf_seq
    ild_seqs <- all_hf_summary[[ild_key]]$hf_seq
    shared <- intersect(ra_seqs, ild_seqs)
    cat(sprintf("  [%s] %d%% 阈值: RA %d 条, ILD %d 条, 交集 %d 条\n",
                part_name, tp, length(ra_seqs), length(ild_seqs), length(shared)))
  }
}

# ====================================================================
# 汇总对比表：delta 4分类
# ====================================================================
cat("\n================================================================================\n")
cat(" 汇总对比：delta 4分类（不截取 K/M）\n")
cat("================================================================================\n\n")

header <- sprintf("%-10s %8s %8s %12s %12s %12s %12s %10s",
                  "部分", "T%", "Δ%", "RA-enrich", "ILD-enrich", "Shared", "Ambiguous", "候选合计")
cat(header, "\n")
cat(paste(rep("-", nchar(header)), collapse = ""), "\n")

for (key in names(all_4c_summary)) {
  s <- all_4c_summary[[key]]
  c <- s$counts
  cat(sprintf("%-10s %7d%% %7d%% %12d %12d %12d %12d %10d\n",
              s$part, s$T, s$D, c["RA-enrich"], c["ILD-enrich"],
              c["Shared"], c["Ambiguous"], s$n_total))
}

cat("\n所有分析完成！图表保存至:", plot_dir, "\n")
