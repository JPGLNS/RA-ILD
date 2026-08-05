library(ggplot2)

setwd("/data/users/chenhaisheng/RA-ILD/")

# ====================================================================
# AA 归一化差异分析
# 对每个 K：用 top K 个 AA clone 的频率和做分母重新归一化
# 再在 K 窗口内取 top M 计算累积频率，比较 PBMC vs buffycoat
# ====================================================================

metadata <- read.csv("./TRB/metadata.csv")

# 读 AA clone table
aa_dir <- "./TRB/result/01_AA_clone_table/"
nt_dir <- "./TRB/origin_result/"
aa_files <- list.files(aa_dir, pattern = "_AA_clone_table\\.csv$")
nt_files <- list.files(nt_dir, pattern = "_TRB_CDR3_NT_frequency_error_correct\\.csv$")

aa_ids <- gsub("_AA_clone_table\\.csv$", "", aa_files)
nt_ids <- gsub("_TRB_CDR3_NT_frequency_error_correct\\.csv$", "", nt_files)
common_ids <- intersect(aa_ids, nt_ids)
aa_files_f <- paste0(common_ids, "_AA_clone_table.csv")

cat("正在读取 AA clone table...\n")
all_freqs <- lapply(file.path(aa_dir, aa_files_f), function(f) {
  data <- read.csv(f, stringsAsFactors = FALSE)
  sort(data$frequency, decreasing = TRUE)
})
names(all_freqs) <- common_ids
cat("完成！共读取", length(all_freqs), "个样本。\n")

# 参数
K_values <- c(3000, 5000, 8000, 10000, 12000, 14000, 16000)
M_values <- c(500, 1000, 1500, 2000, 3000, 5000, 8000, 10000)

# 分组
sample_info <- data.frame(
  libraryid = names(all_freqs),
  stringsAsFactors = FALSE
)
sample_info <- merge(sample_info, metadata[, c("libraryid", "material", "cohort")], by = "libraryid")
sample_info$group_four <- paste(sample_info$material, sample_info$cohort, sep = "-")
rownames(sample_info) <- sample_info$libraryid

# ====================================================================
# 主循环
# ====================================================================
results_two <- matrix(NA, nrow = length(K_values), ncol = length(M_values),
                      dimnames = list(K_values, M_values))
results_RA  <- matrix(NA, nrow = length(K_values), ncol = length(M_values),
                      dimnames = list(K_values, M_values))
results_ILD <- matrix(NA, nrow = length(K_values), ncol = length(M_values),
                      dimnames = list(K_values, M_values))

calc_norm_cumfreq <- function(freqs, K, M) {
  n_k <- min(K, length(freqs))
  if (n_k == 0) return(NA)
  pool <- freqs[1:n_k]
  pool_norm <- pool / sum(pool) * 100
  n_m <- min(M, n_k)
  sum(pool_norm[1:n_m])
}

valid_combos <- expand.grid(K = K_values, M = M_values)
valid_combos <- valid_combos[valid_combos$M < valid_combos$K, ]
cat("\n参数扫描：", nrow(valid_combos), "个 (K, M) 组合\n")

pb <- txtProgressBar(min = 0, max = nrow(valid_combos), style = 3)

for (i in 1:nrow(valid_combos)) {
  K <- valid_combos$K[i]
  M <- valid_combos$M[i]

  cumfreqs <- sapply(all_freqs, calc_norm_cumfreq, K = K, M = M)

  df <- data.frame(
    libraryid = names(cumfreqs),
    cumfreq   = as.numeric(cumfreqs),
    stringsAsFactors = FALSE
  )
  df <- merge(df, sample_info, by = "libraryid")

  means_two <- aggregate(cumfreq ~ material, data = df, FUN = mean)
  pb_mean <- means_two$cumfreq[means_two$material == "PBMC"]
  bc_mean <- means_two$cumfreq[means_two$material == "buffycoat"]
  results_two[as.character(K), as.character(M)] <- pb_mean - bc_mean

  means_four <- aggregate(cumfreq ~ group_four, data = df, FUN = mean)
  for (cohort in c("RA", "ILD")) {
    pb_val <- means_four$cumfreq[means_four$group_four == paste0("PBMC-", cohort)]
    bc_val <- means_four$cumfreq[means_four$group_four == paste0("buffycoat-", cohort)]
    if (length(pb_val) > 0 && length(bc_val) > 0) {
      mat <- if (cohort == "RA") results_RA else results_ILD
      mat[as.character(K), as.character(M)] <- pb_val - bc_val
      if (cohort == "RA") results_RA <- mat else results_ILD <- mat
    }
  }
  setTxtProgressBar(pb, i)
}
close(pb)

# ====================================================================
# 输出表格
# ====================================================================
cat("\n\n")
cat("########################################################################\n")
cat("# 表格1：PBMC均值 - buffycoat均值 (%) — AA\n")
cat("########################################################################\n")
cat("        ")
cat(paste(sprintf("%8s", M_values), collapse = "")); cat("\n")
for (K in K_values) {
  cat(sprintf("K=%-5d", K))
  for (M in M_values) {
    val <- results_two[as.character(K), as.character(M)]
    if (is.na(val)) cat(sprintf("%8s", "  -")) else cat(sprintf("%8.2f", val))
  }
  cat("\n")
}

cat("\n########################################################################\n")
cat("# 表格2：PBMC-RA - buffycoat-RA 差值 (%) — AA\n")
cat("########################################################################\n")
cat("        ")
cat(paste(sprintf("%8s", M_values), collapse = "")); cat("\n")
for (K in K_values) {
  cat(sprintf("K=%-5d", K))
  for (M in M_values) {
    val <- results_RA[as.character(K), as.character(M)]
    if (is.na(val)) cat(sprintf("%8s", "  -")) else cat(sprintf("%8.2f", val))
  }
  cat("\n")
}

cat("\n########################################################################\n")
cat("# 表格3：PBMC-ILD - buffycoat-ILD 差值 (%) — AA\n")
cat("########################################################################\n")
cat("        ")
cat(paste(sprintf("%8s", M_values), collapse = "")); cat("\n")
for (K in K_values) {
  cat(sprintf("K=%-5d", K))
  for (M in M_values) {
    val <- results_ILD[as.character(K), as.character(M)]
    if (is.na(val)) cat(sprintf("%8s", "  -")) else cat(sprintf("%8.2f", val))
  }
  cat("\n")
}

# ====================================================================
# 找最接近的组合
# ====================================================================
cat("\n# |PBMC - buffycoat| < 1% 的 (K, M)：\n")
close_two <- which(abs(results_two) < 1, arr.ind = TRUE)
if (nrow(close_two) > 0) {
  for (r in 1:nrow(close_two)) {
    K <- rownames(results_two)[close_two[r, 1]]
    M <- colnames(results_two)[close_two[r, 2]]
    val <- results_two[close_two[r, 1], close_two[r, 2]]
    cat(sprintf("  K=%s, M=%s: diff = %.4f%%\n", K, M, val))
  }
} else cat("  （无）\n")

cat("\n# |PBMC-RA - buffycoat-RA| < 1% 的 (K, M)：\n")
close_RA <- which(abs(results_RA) < 1, arr.ind = TRUE)
if (nrow(close_RA) > 0) {
  for (r in 1:nrow(close_RA)) {
    K <- rownames(results_RA)[close_RA[r, 1]]
    M <- colnames(results_RA)[close_RA[r, 2]]
    val <- results_RA[close_RA[r, 1], close_RA[r, 2]]
    cat(sprintf("  K=%s, M=%s: diff = %.4f%%\n", K, M, val))
  }
} else cat("  （无）\n")

cat("\n# |PBMC-ILD - buffycoat-ILD| < 1% 的 (K, M)：\n")
close_ILD <- which(abs(results_ILD) < 1, arr.ind = TRUE)
if (nrow(close_ILD) > 0) {
  for (r in 1:nrow(close_ILD)) {
    K <- rownames(results_ILD)[close_ILD[r, 1]]
    M <- colnames(results_ILD)[close_ILD[r, 2]]
    val <- results_ILD[close_ILD[r, 1], close_ILD[r, 2]]
    cat(sprintf("  K=%s, M=%s: diff = %.4f%%\n", K, M, val))
  }
} else cat("  （无）\n")

# ====================================================================
# 可视化
# ====================================================================
cat("\n\n########## 绘制可视化图 ##########\n")

gradient_dir <- "./TRB/gradient/normalized_diff/"
dir.create(gradient_dir, showWarnings = FALSE, recursive = TRUE)
stagger_vjust <- function(i) ifelse(seq_along(i) %% 2 == 1, -2.2, 3.2)

# --- 图1：RA K=16000 差值曲线 ---
K_RA <- 16000
M_RA <- M_values[M_values < K_RA]
cat("\nRA K=16000 差值曲线...\n")
ra_curve <- do.call(rbind, lapply(M_RA, function(M) {
  cf <- sapply(all_freqs, calc_norm_cumfreq, K = K_RA, M = M)
  df <- data.frame(libraryid = names(cf), cumfreq = as.numeric(cf))
  df <- merge(df, sample_info[, c("libraryid", "group_four")], by = "libraryid")
  means <- aggregate(cumfreq ~ group_four, data = df, FUN = mean)
  data.frame(M = M, diff = means$cumfreq[means$group_four == "PBMC-RA"] -
                           means$cumfreq[means$group_four == "buffycoat-RA"])
}))
for (i in 1:nrow(ra_curve)) cat(sprintf("  M=%s: %.4f%%\n", ra_curve$M[i], ra_curve$diff[i]))

p_ra <- ggplot(ra_curve, aes(x = M, y = diff)) +
  geom_hline(yintercept = 0, linetype = "dotted", color = "grey50", linewidth = 0.5) +
  geom_line(color = "#E64B35", linewidth = 1) +
  geom_point(color = "#E64B35", size = 2.5) +
  geom_text(aes(label = sprintf("%.2f", diff)), vjust = stagger_vjust(ra_curve$diff),
            size = 3.2, color = "#E64B35", fontface = "bold") +
  scale_x_continuous(breaks = M_RA) +
  scale_y_continuous(expand = expansion(mult = c(0.15, 0.28))) +
  labs(x = "Top M Clones", y = "Mean Cumulative Frequency Difference (%)") +
  ggtitle(paste0("AA RA: PBMC - Buffycoat (Normalized by Top ", K_RA, " Clones)")) +
  theme_classic(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 10),
        axis.text = element_text(color = "black"),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.ticks = element_line(color = "black"),
        panel.grid.major.y = element_line(color = "grey90", linewidth = 0.3))
ggsave(paste0(gradient_dir, "AA_norm_diff_RA_K", K_RA, ".png"), p_ra, width = 8, height = 5, dpi = 300)
ggsave(paste0(gradient_dir, "AA_norm_diff_RA_K", K_RA, ".pdf"), p_ra, width = 8, height = 5)

# --- 图2：ILD K=12000 差值曲线 ---
K_ILD <- 12000
M_ILD <- M_values[M_values < K_ILD]
cat("\nILD K=", K_ILD, " 差值曲线...\n", sep = "")
ild_curve <- do.call(rbind, lapply(M_ILD, function(M) {
  cf <- sapply(all_freqs, calc_norm_cumfreq, K = K_ILD, M = M)
  df <- data.frame(libraryid = names(cf), cumfreq = as.numeric(cf))
  df <- merge(df, sample_info[, c("libraryid", "group_four")], by = "libraryid")
  means <- aggregate(cumfreq ~ group_four, data = df, FUN = mean)
  data.frame(M = M, diff = means$cumfreq[means$group_four == "PBMC-ILD"] -
                           means$cumfreq[means$group_four == "buffycoat-ILD"])
}))
for (i in 1:nrow(ild_curve)) cat(sprintf("  M=%s: %.4f%%\n", ild_curve$M[i], ild_curve$diff[i]))

p_ild <- ggplot(ild_curve, aes(x = M, y = diff)) +
  geom_hline(yintercept = 0, linetype = "dotted", color = "grey50", linewidth = 0.5) +
  geom_line(color = "#4DBBD5", linewidth = 1) +
  geom_point(color = "#4DBBD5", size = 2.5) +
  geom_text(aes(label = sprintf("%.2f", diff)), vjust = stagger_vjust(ild_curve$diff),
            size = 3.2, color = "#4DBBD5", fontface = "bold") +
  scale_x_continuous(breaks = M_ILD) +
  scale_y_continuous(expand = expansion(mult = c(0.15, 0.28))) +
  labs(x = "Top M Clones", y = "Mean Cumulative Frequency Difference (%)") +
  ggtitle(paste0("AA ILD: PBMC - Buffycoat (Normalized by Top ", K_ILD, " Clones)")) +
  theme_classic(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 10),
        axis.text = element_text(color = "black"),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.ticks = element_line(color = "black"),
        panel.grid.major.y = element_line(color = "grey90", linewidth = 0.3))
ggsave(paste0(gradient_dir, "AA_norm_diff_ILD_K", K_ILD, ".png"), p_ild, width = 8, height = 5, dpi = 300)
ggsave(paste0(gradient_dir, "AA_norm_diff_ILD_K", K_ILD, ".pdf"), p_ild, width = 8, height = 5)

# --- 图3：整体热图 ---
cat("\n整体热图...\n")
heat_two <- data.frame(K = rep(K_values, each = length(M_values)),
                       M = rep(M_values, length(K_values)),
                       diff = as.vector(t(results_two)))
heat_two <- heat_two[!is.na(heat_two$diff), ]
heat_two$K <- factor(heat_two$K, levels = K_values)
heat_two$M <- factor(heat_two$M, levels = M_values)
heat_two$close_one <- abs(heat_two$diff) < 1
max_abs <- max(abs(heat_two$diff), na.rm = TRUE)

p_heat <- ggplot(heat_two, aes(x = M, y = K, fill = diff)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.2f", diff), color = close_one), size = 2.8) +
  scale_color_manual(values = c("TRUE" = "#B2182B", "FALSE" = "black"), guide = "none") +
  scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B",
                       midpoint = 0, limits = c(-max_abs, max_abs), name = "Diff (%)") +
  scale_x_discrete(position = "top") + coord_fixed() +
  labs(x = "M (Top M Clones)", y = "K (Normalization Window)") +
  ggtitle("AA: PBMC - Buffycoat Normalized Difference") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 11),
        axis.text = element_text(color = "black"), panel.grid = element_blank(),
        legend.position = "right", legend.title = element_text(size = 10),
        legend.text = element_text(size = 9))
ggsave(paste0(gradient_dir, "AA_norm_diff_heatmap_overall.png"), p_heat, width = 10, height = 6, dpi = 300)
ggsave(paste0(gradient_dir, "AA_norm_diff_heatmap_overall.pdf"), p_heat, width = 10, height = 6)

# --- 图4：RA 热图 ---
cat("RA 热图...\n")
heat_RA <- data.frame(K = rep(K_values, each = length(M_values)),
                      M = rep(M_values, length(K_values)),
                      diff = as.vector(t(results_RA)))
heat_RA <- heat_RA[!is.na(heat_RA$diff), ]
heat_RA$K <- factor(heat_RA$K, levels = K_values)
heat_RA$M <- factor(heat_RA$M, levels = M_values)
heat_RA$close_one <- abs(heat_RA$diff) < 1
max_abs_RA <- max(abs(heat_RA$diff), na.rm = TRUE)

p_heat_RA <- ggplot(heat_RA, aes(x = M, y = K, fill = diff)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.2f", diff), color = close_one), size = 2.8) +
  scale_color_manual(values = c("TRUE" = "#B2182B", "FALSE" = "black"), guide = "none") +
  scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B",
                       midpoint = 0, limits = c(-max_abs_RA, max_abs_RA), name = "Diff (%)") +
  scale_x_discrete(position = "top") + coord_fixed() +
  labs(x = "M (Top M Clones)", y = "K (Normalization Window)") +
  ggtitle("AA RA: PBMC - Buffycoat Normalized Difference") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 11),
        axis.text = element_text(color = "black"), panel.grid = element_blank(),
        legend.position = "right", legend.title = element_text(size = 10),
        legend.text = element_text(size = 9))
ggsave(paste0(gradient_dir, "AA_norm_diff_heatmap_RA.png"), p_heat_RA, width = 10, height = 6, dpi = 300)
ggsave(paste0(gradient_dir, "AA_norm_diff_heatmap_RA.pdf"), p_heat_RA, width = 10, height = 6)

# --- 图5：ILD 热图 ---
cat("ILD 热图...\n")
heat_ILD <- data.frame(K = rep(K_values, each = length(M_values)),
                       M = rep(M_values, length(K_values)),
                       diff = as.vector(t(results_ILD)))
heat_ILD <- heat_ILD[!is.na(heat_ILD$diff), ]
heat_ILD$K <- factor(heat_ILD$K, levels = K_values)
heat_ILD$M <- factor(heat_ILD$M, levels = M_values)
heat_ILD$close_one <- abs(heat_ILD$diff) < 1
max_abs_ILD <- max(abs(heat_ILD$diff), na.rm = TRUE)

p_heat_ILD <- ggplot(heat_ILD, aes(x = M, y = K, fill = diff)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.2f", diff), color = close_one), size = 2.8) +
  scale_color_manual(values = c("TRUE" = "#B2182B", "FALSE" = "black"), guide = "none") +
  scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B",
                       midpoint = 0, limits = c(-max_abs_ILD, max_abs_ILD), name = "Diff (%)") +
  scale_x_discrete(position = "top") + coord_fixed() +
  labs(x = "M (Top M Clones)", y = "K (Normalization Window)") +
  ggtitle("AA ILD: PBMC - Buffycoat Normalized Difference") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 11),
        axis.text = element_text(color = "black"), panel.grid = element_blank(),
        legend.position = "right", legend.title = element_text(size = 10),
        legend.text = element_text(size = 9))
ggsave(paste0(gradient_dir, "AA_norm_diff_heatmap_ILD.png"), p_heat_ILD, width = 10, height = 6, dpi = 300)
ggsave(paste0(gradient_dir, "AA_norm_diff_heatmap_ILD.pdf"), p_heat_ILD, width = 10, height = 6)

cat("\n========== AA 归一化分析全部完成！图片路径:", gradient_dir, " ==========\n")
