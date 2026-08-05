# ====================================================================
# CDR3 AA 字典分析共享函数库
# ====================================================================
# 供 CDR3_AA_dict_LOOCV.R 与 CDR3_AA_dict_material_analysis.R 共用。
# 核心修复：
#   1) RA/ILD 计数向量按统一克隆全集对齐（名称与顺序完全一致），
#      杜绝按位置比较造成的字典构建错误；
#   2) LOOCV 每轮从对齐计数中增量扣除留出样本，重建字典后恢复；
#   3) 完整流程置换检验：每轮置换都重跑整个 LOOCV 字典构建与打分；
#   4) 空字典 / 常数指标 / 重复克隆等边界情况的防御性处理。
# 性能：克隆以整数编码，计数为对齐的 integer 向量，置换前先做
#       标签无关的安全预筛选（数学上界，不引入真实标签信息）。
# ====================================================================

suppressMessages({library(data.table)})

# ------------------------------------------------------------------
# 基础工具
# ------------------------------------------------------------------

# 安全命中率：字典为空时返回 NA 而非 NaN
safe_hit_rate <- function(hit_count, dict_size) {
  if (is.na(dict_size) || dict_size == 0L) return(NA_real_)
  hit_count / dict_size
}

# 安全偏度：SD 为 0 时返回 NA
safe_skewness <- function(x) {
  s <- sd(x)
  if (is.na(s) || s == 0) return(NA_real_)
  m <- mean(x)
  mean((x - m)^3) / s^3
}

# 安全 Cohen's d：pooled SD 为 0 时返回 NA
safe_cohen_d <- function(a, b) {
  a <- a[is.finite(a)]; b <- b[is.finite(b)]
  if (length(a) < 2 || length(b) < 2) return(NA_real_)
  s_pool <- sqrt(((length(a) - 1) * var(a) + (length(b) - 1) * var(b)) /
                   (length(a) + length(b) - 2))
  if (is.na(s_pool) || s_pool == 0) return(NA_real_)
  (mean(a) - mean(b)) / s_pool
}

# 安全 Cliff's delta
safe_cliff_delta <- function(a, b) {
  a <- a[is.finite(a)]; b <- b[is.finite(b)]
  if (length(a) == 0 || length(b) == 0) return(NA_real_)
  sum(sapply(a, function(x) sum(x > b) - sum(x < b))) / (length(a) * length(b))
}

# 从分数与标签计算 AUC（Mann-Whitney U，ties 用平均秩）
auc_from_scores <- function(scores, y) {
  scores <- as.numeric(scores)
  y <- as.logical(y)
  ok <- is.finite(scores) & !is.na(y)
  scores <- scores[ok]; y <- y[ok]
  n1 <- sum(y); n2 <- sum(!y)
  if (n1 == 0 || n2 == 0) return(NA_real_)
  r <- rank(scores)
  (sum(r[y]) - n1 * (n1 + 1) / 2) / (n1 * n2)
}

# 双侧完整流程置换 p（标准公式）
perm_p_two_sided <- function(observed_stat, permuted_stats, n_perm) {
  (1 + sum(permuted_stats >= observed_stat)) / (n_perm + 1)
}

# ------------------------------------------------------------------
# 元数据与样本划分
# ------------------------------------------------------------------

# 校验 metadata 并输出 cohort × material × batch 列联表
validate_metadata <- function(metadata, sample_ids) {
  stopifnot(all(c("libraryid", "cohort", "material") %in% colnames(metadata)))
  meta <- metadata[metadata$libraryid %in% sample_ids, ]
  stopifnot(all(c("RA", "ILD") %in% unique(meta$cohort)))
  cat("cohort × material:\n")
  print(table(meta$cohort, meta$material))
  if ("batch" %in% colnames(metadata)) {
    cat("\ncohort × material × batch:\n")
    print(table(meta$cohort, meta$material, meta$batch))
  }
  invisible(meta)
}

# 按列联表检查某分层方案是否每个 stratum 都含两个 cohort
check_strata <- function(meta, strata_var) {
  tab <- table(meta[[strata_var]], meta$cohort)
  one_sided <- rownames(tab)[apply(tab, 1, function(r) any(r == 0))]
  if (length(one_sided) > 0) {
    warning("以下 stratum 只含单一 cohort，无法贡献标签交换: ",
            paste(one_sided, collapse = ", "))
  }
  invisible(tab)
}

# ------------------------------------------------------------------
# 数据读取（防御性）
# ------------------------------------------------------------------

# 读取单个样本 clone 表：过滤非法值、按 cdr3_aa 聚合重复行
read_sample_clones <- function(f, warn_dup = TRUE) {
  d <- fread(f, select = c("cdr3_aa", "read_fraction"))
  d <- d[!is.na(cdr3_aa) & nzchar(cdr3_aa) &
           is.finite(read_fraction) & read_fraction >= 0]
  n_dup <- d[, .N, by = cdr3_aa][N > 1]
  if (warn_dup && nrow(n_dup) > 0) {
    warning(sprintf("%s: 发现 %d 个重复 cdr3_aa（出现次数按 1 计，read_fraction 已求和）",
                    basename(f), nrow(n_dup)))
  }
  d <- d[, .(read_fraction = sum(read_fraction)), by = cdr3_aa]
  list(clone = d$cdr3_aa, rf = d$read_fraction)
}

# 读取全部样本并构建整数编码（可带预筛选 keep_clones）
load_sample_clone_data <- function(aa_dir, aa_files_f, sample_ids, keep_clones = NULL) {
  n <- length(sample_ids)
  clones_char <- vector("list", n)
  rf_list     <- vector("list", n)
  for (i in seq_len(n)) {
    d <- read_sample_clones(file.path(aa_dir, aa_files_f[i]))
    clones_char[[i]] <- d$clone
    rf_list[[i]]     <- d$rf
  }
  names(clones_char) <- sample_ids
  names(rf_list)     <- sample_ids

  # 全克隆集（用于计数与预筛选）
  all_tab <- data.table(clone = unlist(clones_char))
  all_tab <- all_tab[, .N, by = clone]
  if (!is.null(keep_clones)) {
    all_tab <- all_tab[clone %in% keep_clones]
  }
  all_tab[, id := seq_len(.N)]
  U <- nrow(all_tab)

  # 每样本映射到整数 id
  clone_ids <- lapply(clones_char, function(cl) {
    m <- chmatch(cl, all_tab$clone)
    m[!is.na(m)]
  })
  cat("克隆全集大小（预筛选后）:", U, "\n")
  list(
    clone_ids = clone_ids,
    rf = rf_list,
    clone_names = all_tab$clone,
    n_universe = U,
    all_tab = all_tab
  )
}

# 标签无关的安全预筛选：总出现次数 < 任一训练集最小 20% 阈值 → 永不可能入字典
prescreen_clones <- function(all_tab, n_ra, n_ild, threshold_pct) {
  ra_t_min  <- ceiling((n_ra  - 1) * threshold_pct / 100)
  ild_t_min <- ceiling((n_ild - 1) * threshold_pct / 100)
  min_t <- min(ra_t_min, ild_t_min)
  keep <- all_tab[N >= min_t, clone]
  cat(sprintf("预筛选：总出现次数 ≥ %d 的克隆 %d/%d 条（数学上界，标签无关）\n",
              min_t, length(keep), nrow(all_tab)))
  keep
}

# ------------------------------------------------------------------
# 对齐计数与字典构建（核心修复）
# ------------------------------------------------------------------

# 从整数克隆 id 构建"统一全集对齐"的计数向量
build_aligned_counts <- function(clone_ids, sample_ids, U) {
  counts <- integer(U)
  for (sid in sample_ids) {
    ids <- clone_ids[[sid]]
    if (length(ids) > 0) counts[ids] <- counts[ids] + 1L
  }
  counts
}

# 检查对齐与合法性的断言
assert_aligned <- function(ra_counts, ild_counts, n_ra, n_ild) {
  stopifnot(
    length(ra_counts) == length(ild_counts),
    is.integer(ra_counts), is.integer(ild_counts),
    !anyNA(ra_counts), !anyNA(ild_counts),
    all(ra_counts >= 0L), all(ild_counts >= 0L),
    max(ra_counts) <= n_ra,
    max(ild_counts) <= n_ild
  )
  invisible(TRUE)
}

# 从对齐计数构建字典（整数 id 向量）
# ra_counts/ild_counts 长度与顺序完全一致（整数编码天然对齐）
build_dictionary_from_counts <- function(ra_counts, ild_counts, n_ra, n_ild,
                                         threshold_pct, delta_pct) {
  ra_t  <- ceiling(n_ra  * threshold_pct / 100)
  ild_t <- ceiling(n_ild * threshold_pct / 100)
  cand_mask <- (ra_counts >= ra_t) | (ild_counts >= ild_t)   # 对齐后按位置比较即正确
  cand <- which(cand_mask)
  if (length(cand) == 0) {
    return(list(ra_dict = integer(0), ild_dict = integer(0), ra_t = ra_t, ild_t = ild_t))
  }
  pct_ra  <- ra_counts[cand] / n_ra  * 100
  pct_ild <- ild_counts[cand] / n_ild * 100
  delta <- pct_ra - pct_ild
  ra_dict  <- cand[pct_ra  >= threshold_pct & delta >=  delta_pct]
  ild_dict <- cand[pct_ild >= threshold_pct & delta <= -delta_pct]
  list(ra_dict = ra_dict, ild_dict = ild_dict, ra_t = ra_t, ild_t = ild_t)
}

# 用字典给单个样本打分（clone_ids/ra_dict 均为整数 id，须用 match 而非 chmatch）
score_sample_with_dict <- function(clone_ids, rf, ra_dict, ild_dict) {
  m_ra  <- match(clone_ids, ra_dict)
  m_ild <- match(clone_ids, ild_dict)
  ra_hit  <- !is.na(m_ra)
  ild_hit <- !is.na(m_ild)
  list(
    ra_count = sum(ra_hit),
    ild_count = sum(ild_hit),
    ra_rf_sum = sum(rf[ra_hit]),
    ild_rf_sum = sum(rf[ild_hit]),
    ra_dict_size = length(ra_dict),
    ild_dict_size = length(ild_dict)
  )
}

# 预筛选后裁剪：clone 集缩到 keep 子集并重编码到 1..K（保持 id/rf 对齐，
# 且保留样本名——下游 run_dictionary_loocv 按样本名索引）
trim_to_keep <- function(clone_ids, rf, keep_ids) {
  n <- length(clone_ids)
  map <- integer(max(keep_ids)); map[keep_ids] <- seq_along(keep_ids)
  ci <- vector("list", n); rf2 <- vector("list", n)
  names(ci) <- names(clone_ids); names(rf2) <- names(rf)
  for (i in seq_len(n)) {
    v <- clone_ids[[i]]
    mask <- v %in% keep_ids
    ci[[i]] <- map[v[mask]]
    rf2[[i]] <- rf[[i]][mask]
  }
  list(clone_ids = ci, rf = rf2, U = length(keep_ids))
}

# ------------------------------------------------------------------
# LOOCV（observed 与 permutation 共用同一实现）
# ------------------------------------------------------------------

# 对给定 cohort 标签向量执行完整 LOOCV，返回每样本 8 指标 + dict_size
# 标签向量 lab（character: "RA"/"ILD"）决定分组、留出样本所属组、
# 分母与字典构建——observed 与置换流程共用本函数，保证无泄漏语义一致。
run_dictionary_loocv <- function(clone_ids, rf, sample_ids, lab,
                                 n_ra_true, n_ild_true, U,
                                 threshold_pct = 20, delta_pct = 10,
                                 verbose = FALSE) {
  stopifnot(length(lab) == length(sample_ids))
  n <- length(sample_ids)
  ra_ids  <- sample_ids[lab == "RA"]
  ild_ids <- sample_ids[lab == "ILD"]
  n_ra  <- length(ra_ids)
  n_ild <- length(ild_ids)
  stopifnot(n_ra > 0, n_ild > 0)

  ra_counts  <- build_aligned_counts(clone_ids, ra_ids,  U)
  ild_counts <- build_aligned_counts(clone_ids, ild_ids, U)
  assert_aligned(ra_counts, ild_counts, n_ra, n_ild)
  ra_counts_initial  <- ra_counts
  ild_counts_initial <- ild_counts

  n_empty <- 0L
  rows <- vector("list", n)
  for (i in seq_len(n)) {
    sid <- sample_ids[i]
    coh <- lab[i]
    cl_i <- clone_ids[[sid]]
    rf_i <- rf[[sid]]

    if (coh == "RA") {
      n_ra_i  <- n_ra - 1L; n_ild_i <- n_ild
      if (length(cl_i) > 0) ra_counts[cl_i] <- ra_counts[cl_i] - 1L
    } else {
      n_ra_i  <- n_ra; n_ild_i <- n_ild - 1L
      if (length(cl_i) > 0) ild_counts[cl_i] <- ild_counts[cl_i] - 1L
    }
    stopifnot(all(ra_counts >= 0L), all(ild_counts >= 0L))

    dict <- build_dictionary_from_counts(ra_counts, ild_counts,
                                         n_ra_i, n_ild_i, threshold_pct, delta_pct)
    if (length(dict$ra_dict) == 0L || length(dict$ild_dict) == 0L) n_empty <- n_empty + 1L

    sc <- score_sample_with_dict(cl_i, rf_i, dict$ra_dict, dict$ild_dict)
    rows[[i]] <- data.table(
      libraryid = sid, cohort = coh,
      RA_dict_clone_count  = sc$ra_count,
      ILD_dict_clone_count = sc$ild_count,
      RA_dict_read_fraction_sum  = round(sc$ra_rf_sum,  6),
      ILD_dict_read_fraction_sum = round(sc$ild_rf_sum, 6),
      RA_dict_hit_rate  = safe_hit_rate(sc$ra_count,  sc$ra_dict_size),
      ILD_dict_hit_rate = safe_hit_rate(sc$ild_count, sc$ild_dict_size),
      RA_minus_ILD_count = sc$ra_count - sc$ild_count,
      RA_minus_ILD_freq  = round(sc$ra_rf_sum - sc$ild_rf_sum, 6),
      RA_dict_size  = sc$ra_dict_size,
      ILD_dict_size = sc$ild_dict_size
    )

    # 恢复计数
    if (coh == "RA") {
      if (length(cl_i) > 0) ra_counts[cl_i] <- ra_counts[cl_i] + 1L
    } else {
      if (length(cl_i) > 0) ild_counts[cl_i] <- ild_counts[cl_i] + 1L
    }
  }
  stopifnot(identical(ra_counts, ra_counts_initial),
            identical(ild_counts, ild_counts_initial))

  hits <- rbindlist(rows)
  if (verbose && n_empty > 0) {
    warning("LOOCV 中共有 ", n_empty, " 轮出现空字典（dict_size=0，hit_rate=NA）")
  }
  attr(hits, "n_empty_dict_rounds") <- n_empty
  hits
}

# ------------------------------------------------------------------
# 置换
# ------------------------------------------------------------------

# 分层标签置换：strata 内打乱 cohort（保持每层 RA/ILD 数量）
# strata: 与 sample_ids 等长的分组因子
permute_labels_within_strata <- function(lab, strata, seed = NULL) {
  n <- length(lab)
  out <- lab
  for (s in unique(strata)) {
    idx <- which(strata == s)
    n_ra <- sum(lab[idx] == "RA")
    n_ild <- length(idx) - n_ra
    if (n_ra == 0 || n_ild == 0) next   # 单 cohort stratum 无法交换，保持原样
    out[idx] <- sample(lab[idx])        # 仅在层内打乱
  }
  out
}

# 完整流程置换：每个置换重跑完整 LOOCV，返回 8 指标 AUC 矩阵
run_full_pipeline_permutation <- function(clone_ids, rf, sample_ids, lab_obs,
                                          strata, n_perm, seed, U,
                                          threshold_pct = 20, delta_pct = 10,
                                          checkpoint_prefix = NULL,
                                          checkpoint_every = 25L) {
  metrics <- c("RA_dict_clone_count", "ILD_dict_clone_count",
               "RA_dict_hit_rate", "ILD_dict_hit_rate",
               "RA_dict_read_fraction_sum", "ILD_dict_read_fraction_sum",
               "RA_minus_ILD_count", "RA_minus_ILD_freq")
  n_ra_true <- sum(lab_obs == "RA")
  n_ild_true <- sum(lab_obs == "ILD")

  # 预筛选（标签无关，数学上界）——在调用方完成并传入 clone_ids 已裁剪，
  # 此处不再重复。

  set.seed(seed)
  perm_auc <- matrix(NA_real_, nrow = n_perm, ncol = length(metrics),
                     dimnames = list(NULL, metrics))

  # observed baseline（真实标签完整 LOOCV）
  hits_obs <- run_dictionary_loocv(clone_ids, rf, sample_ids, lab_obs,
                                   n_ra_true, n_ild_true, U,
                                   threshold_pct, delta_pct, verbose = TRUE)
  for (p in seq_len(n_perm)) {
    lab_p <- permute_labels_within_strata(lab_obs, strata)
    hits_p <- run_dictionary_loocv(clone_ids, rf, sample_ids, lab_p,
                                   n_ra_true, n_ild_true, U,
                                   threshold_pct, delta_pct)
    for (j in seq_along(metrics)) {
      perm_auc[p, j] <- auc_from_scores(hits_p[[metrics[j]]], hits_p$cohort == "RA")
    }
    if (!is.null(checkpoint_prefix) && p %% checkpoint_every == 0L) {
      f <- sprintf("%s_checkpoint_perm%d.csv", checkpoint_prefix, p)
      write.csv(perm_auc[seq_len(p), , drop = FALSE], f, row.names = FALSE)
    }
  }
  list(perm_auc = perm_auc, hits_obs = hits_obs, metrics = metrics, n_perm = n_perm)
}

# ------------------------------------------------------------------
# 安全统计分析（对给定 hits 表的 8 指标）
# ------------------------------------------------------------------

safe_analyze_metrics <- function(hits_tbl, metrics = c("RA_dict_clone_count",
    "ILD_dict_clone_count", "RA_dict_hit_rate", "ILD_dict_hit_rate",
    "RA_dict_read_fraction_sum", "ILD_dict_read_fraction_sum",
    "RA_minus_ILD_count", "RA_minus_ILD_freq"),
    metric_labels = c("RA 字典命中数", "ILD 字典命中数",
                      "RA 字典命中率", "ILD 字典命中率",
                      "RA 命中 read_fraction 和", "ILD 命中 read_fraction 和",
                      "RA−ILD 命中数差", "RA−ILD read_fraction 和差")) {
  suppressMessages(require(pROC, quietly = TRUE))
  ra  <- hits_tbl[cohort == "RA"]
  ild <- hits_tbl[cohort == "ILD"]
  out <- vector("list", length(metrics))
  for (i in seq_along(metrics)) {
    m <- metrics[i]
    x_ra  <- as.numeric(ra[[m]]);  x_ild <- as.numeric(ild[[m]])
    x_ra  <- x_ra[is.finite(x_ra)]; x_ild <- x_ild[is.finite(x_ild)]
    n_ra_eff <- length(x_ra); n_ild_eff <- length(x_ild)
    note <- character(0)

    # Shapiro-Wilk（防全同值）
    sw_p_ra <- sw_p_ild <- NA_real_
    if (n_ra_eff >= 3 && length(unique(x_ra)) > 1) {
      sw_p_ra <- tryCatch(shapiro.test(x_ra)$p.value, error = function(e) NA_real_)
    } else if (n_ra_eff < 3) note <- c(note, "RA 有效样本过少")
    if (n_ild_eff >= 3 && length(unique(x_ild)) > 1) {
      sw_p_ild <- tryCatch(shapiro.test(x_ild)$p.value, error = function(e) NA_real_)
    } else if (n_ild_eff < 3) note <- c(note, "ILD 有效样本过少")

    biased <- (is.na(sw_p_ra) | (sw_p_ra < 0.05 & abs(safe_skewness(x_ra)) > 1)) |
              (is.na(sw_p_ild) | (sw_p_ild < 0.05 & abs(safe_skewness(x_ild)) > 1))
    main_test <- if (!is.na(biased) && biased) "Mann-Whitney U" else "Welch t"

    # 检验（防崩溃）
    t_p <- u_p <- NA_real_
    if (n_ra_eff >= 2 && n_ild_eff >= 2 && length(unique(x_ra)) > 1 &&
        length(unique(x_ild)) > 1) {
      t_p <- tryCatch(t.test(x_ra, x_ild)$p.value, error = function(e) NA_real_)
      u_p <- tryCatch(wilcox.test(x_ra, x_ild)$p.value, error = function(e) NA_real_)
    } else {
      note <- c(note, "检验无法执行（样本/变异不足）")
    }

    # AUC（常数预测变量 → 0.5，CI 为 NA；roc_obj 在不可用时为 NULL）
    auc_val <- auc_lo <- auc_hi <- NA_real_
    roc_obj <- NULL
    all_scores <- c(x_ra, x_ild)
    if (length(unique(all_scores)) > 1) {
      roc_obj <- tryCatch(
        roc(hits_tbl$cohort, hits_tbl[[m]], levels = c("ILD", "RA"),
            direction = "<", quiet = TRUE),
        error = function(e) NULL)
      if (!is.null(roc_obj)) {
        auc_val <- as.numeric(auc(roc_obj))
        ci <- tryCatch(as.numeric(ci.auc(roc_obj)), error = function(e) rep(NA_real_, 3))
        auc_lo <- ci[1]; auc_hi <- ci[3]
      }
    } else {
      auc_val <- 0.5
      note <- c(note, "预测变量为常数，AUC 记为 0.5")
    }

    out[[i]] <- list(
      metric = m, label = metric_labels[i],
      n_ra_eff = n_ra_eff, n_ild_eff = n_ild_eff,
      mean_ra = mean(x_ra), med_ra = median(x_ra),
      mean_ild = mean(x_ild), med_ild = median(x_ild),
      sw_p_ra = sw_p_ra, sw_p_ild = sw_p_ild,
      main_test = main_test,
      t_p = t_p, u_p = u_p,
      cohen_d = safe_cohen_d(x_ra, x_ild),
      cliff_delta = safe_cliff_delta(x_ra, x_ild),
      auc = auc_val, auc_lo = auc_lo, auc_hi = auc_hi,
      roc = roc_obj,   # 可能为 NULL（AUC 计算失败时），绘图需防御
      status = paste(note, collapse = "; ")
    )
  }
  main_p <- sapply(out, function(r) if (r$main_test == "Mann-Whitney U") r$u_p else r$t_p)
  main_q <- p.adjust(main_p, "BH")
  for (i in seq_along(out)) { out[[i]]$main_p <- main_p[i]; out[[i]]$main_q <- main_q[i] }
  out
}
