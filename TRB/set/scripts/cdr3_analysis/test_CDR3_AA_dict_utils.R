# ====================================================================
# CDR3_AA_dict_utils.R 单元测试（11 组，全部 synthetic 数据）
# ====================================================================
# 运行：Rscript TRB/set/scripts/cdr3_analysis/test_CDR3_AA_dict_utils.R
# 11 组：
#   T1  对齐字典构建（回归测试：命名向量按位置回收的 bug）
#   T2  留一法无泄漏（留出样本的私有 clone 不进入当轮字典）
#   T3  空字典 / 常数指标 / 全 NA 列的防御性处理
#   T4  重复 cdr3_aa 行聚合（rf 求和）+ 非法值过滤
#   T5  完整流程置换：置换 AUC 与"手动重演同一标签序列"严格对拍
#   T6  分层置换保持每层 RA/ILD 数量
#   T7  同一 seed 完全可复现
#   T8  两个主脚本实际加载 set/ 正式目录的 utils
#   T9  keep_clones 后 clone_id 与 rf 长度/顺序严格一致
#   T10 perm_p_two_sided 对部分/全 NA 置换统计量的处理
#   T11 maxT 仅使用完整置换行（分母 = 完整行数+1）
# ====================================================================

suppressMessages(library(data.table))

args <- commandArgs(trailingOnly = FALSE)
script_dir <- dirname(sub("--file=", "", args[grep("--file=", args)]))
source(file.path(script_dir, "CDR3_AA_dict_utils.R"))

n_pass <- 0L; n_fail <- 0L
ok <- function(cond, name) {
  if (isTRUE(cond)) {
    n_pass <<- n_pass + 1L
    cat(sprintf("[PASS] %s\n", name))
  } else {
    n_fail <<- n_fail + 1L
    cat(sprintf("[FAIL] %s\n", name))
  }
}

# ---------- 合成数据构造 ----------
# 每样本从 U 个 clone 中按概率 p 出现；返回整数编码 + rf
make_synth <- function(n_ra, n_ild, U, p, seed, n_strata = 2) {
  set.seed(seed)
  n <- n_ra + n_ild
  lab <- c(rep("RA", n_ra), rep("ILD", n_ild))
  strata <- rep(paste0("M", seq_len(n_strata)), length.out = n)
  mat <- matrix(runif(n * U) < p, nrow = n, ncol = U)
  for (i in seq_len(n)) {                       # 保底：每样本至少 1 个 clone
    if (!any(mat[i, ])) mat[i, sample.int(U, 1)] <- TRUE
  }
  ids <- paste0("s", seq_len(n))
  clone_ids <- setNames(lapply(seq_len(n), function(i) which(mat[i, ])), ids)
  set.seed(seed + 1)
  rf <- setNames(lapply(seq_len(n), function(i) runif(sum(mat[i, ]), 0.001, 1)), ids)
  list(sample_ids = ids, lab = lab, strata = strata,
       clone_ids = clone_ids, rf = rf, U = U,
       clone_names = paste0("clone", seq_len(U)))
}

# 整数 id 计数 → 名字计数向量（参考实现用）
to_name_counts <- function(ids_vec, U, clone_names) {
  v <- setNames(integer(U), as.character(seq_len(U)))
  t <- table(ids_vec)
  v[names(t)] <- as.integer(t)
  setNames(as.integer(v), clone_names)
}

# ====================================================================
# T1 对齐字典构建（核心回归测试）
# ====================================================================
cat("\n== T1 对齐字典构建（回归：命名向量位置回收 bug）==\n")
n_ra <- 10L; n_ild <- 12L; T <- 20; D <- 10
ra_v  <- c(5L, 2L, 1L, 4L, 0L, 3L, 1L, 0L)
ild_v <- c(0L, 0L, 3L, 0L, 4L, 2L, 5L, 6L)
cn <- paste0("clone", 1:8)
ref_ra <- setNames(ra_v, cn); ref_ild <- setNames(ild_v, cn)
ra_t_ref  <- ceiling(n_ra  * T / 100)
ild_t_ref <- ceiling(n_ild * T / 100)
cand_ref <- union(names(ref_ra)[ref_ra >= ra_t_ref],
                  names(ref_ild)[ref_ild >= ild_t_ref])
pct_ra_ref  <- ref_ra[cand_ref]  / n_ra  * 100
pct_ild_ref <- ref_ild[cand_ref] / n_ild * 100
delta_ref <- pct_ra_ref - pct_ild_ref
ra_dict_ref  <- cand_ref[pct_ra_ref  >= T & delta_ref >=  D]
ild_dict_ref <- cand_ref[pct_ild_ref >= T & delta_ref <= -D]
dict1 <- build_dictionary_from_counts(ra_v, ild_v, n_ra, n_ild, T, D)
ok(identical(cn[dict1$ra_dict],  ra_dict_ref) &&
     identical(cn[dict1$ild_dict], ild_dict_ref),
   "T1a 手动对齐计数 vs 按名参考实现（含交叉重叠 clone）")

# 合成数据全管线对拍（build_aligned_counts → build_dictionary_from_counts）
d <- make_synth(24, 16, 200, 0.25, 123)
ra_ids  <- d$sample_ids[d$lab == "RA"]
ild_ids <- d$sample_ids[d$lab == "ILD"]
ra_c  <- build_aligned_counts(d$clone_ids, ra_ids,  d$U)
ild_c <- build_aligned_counts(d$clone_ids, ild_ids, d$U)
dict2 <- build_dictionary_from_counts(ra_c, ild_c, 24, 16, T, D)
cnt_ra  <- to_name_counts(unlist(d$clone_ids[ra_ids]),  d$U, d$clone_names)
cnt_ild <- to_name_counts(unlist(d$clone_ids[ild_ids]), d$U, d$clone_names)
cand_r <- union(names(cnt_ra)[cnt_ra >= ra_t_ref],
                names(cnt_ild)[cnt_ild >= ild_t_ref])
pct_r  <- cnt_ra[cand_r] / 24 * 100
pct_i  <- cnt_ild[cand_r] / 16 * 100
dlt_r  <- pct_r - pct_i
ra_ref  <- cand_r[pct_r >= T & dlt_r >=  D]
ild_ref <- cand_r[pct_i >= T & dlt_r <= -D]
ok(length(dict2$ra_dict) > 0 && length(dict2$ild_dict) > 0,
   "T1b 合成数据字典非空（测试前提）")
ok(identical(sort(d$clone_names[dict2$ra_dict]),  sort(ra_ref)) &&
     identical(sort(d$clone_names[dict2$ild_dict]), sort(ild_ref)),
   "T1c 合成数据全管线 vs 按名参考实现")

# trim_to_keep：保留样本名 + 重编码正确（回归：曾丢 names 致 LOOCV 按名索引全 NULL）
keep_ids1 <- seq_len(40)                     # 只留前 40 个 clone
dt1 <- trim_to_keep(d$clone_ids, d$rf, keep_ids1)
ok(identical(names(dt1$clone_ids), names(d$clone_ids)) &&
     identical(names(dt1$rf), names(d$rf)),
   "T1d trim 后保留样本名")
ok(max(unlist(dt1$clone_ids)) <= 40L && min(unlist(dt1$clone_ids)) >= 1L &&
     identical(dt1$U, 40L),
   "T1e trim 重编码到 1..K")
ok(length(unlist(dt1$clone_ids)) == sum(unlist(lapply(d$clone_ids, function(v) sum(v %in% keep_ids1)))),
   "T1f trim 不丢 keep 内的 clone")

# ====================================================================
# T2 留一法无泄漏
# ====================================================================
cat("\n== T2 留一法无泄漏 ==\n")
ids2 <- c(paste0("RA", 1:8), paste0("ILD", 1:8))
lab2 <- c(rep("RA", 8), rep("ILD", 8))
clone_ids2 <- c(lapply(paste0("RA", 1:8), function(s) {
  if (s == "RA1") 1L else 2L                 # X 只在 RA1；Y 在其余 7 个 RA
}), lapply(paste0("ILD", 1:8), function(s) 3L))  # Z 在所有 ILD
names(clone_ids2) <- ids2
rf2 <- lapply(clone_ids2, function(v) rep(1, length(v)))
hits2 <- run_dictionary_loocv(clone_ids2, rf2, ids2, lab2, 8L, 8L, 3L, 20, 10)
row1 <- hits2[libraryid == "RA1"]
ok(row1$RA_dict_clone_count == 0 && row1$RA_dict_size > 0,
   "T2a 留出样本的私有 clone 未进入当轮字典（无泄漏）")
row2 <- hits2[libraryid == "RA2"]
ok(row2$RA_dict_clone_count >= 1,
   "T2b 共享 clone 正常计入其他样本的命中")
ok(all(hits2[grep("^ILD", libraryid)]$RA_dict_clone_count == 0),
   "T2c ILD 样本不命中 RA 字典（构造正确性）")

# ====================================================================
# T3 空字典 / 常数指标 / 全 NA 列
# ====================================================================
cat("\n== T3 空字典与常数指标防御 ==\n")
ok(is.na(safe_hit_rate(0, 0)), "T3a 空字典 hit_rate=NA（而非 NaN）")
ok(safe_hit_rate(3, 5) == 0.6, "T3b 正常 hit_rate")
ok(safe_hit_rate(0, 5) == 0,   "T3c 零命中 hit_rate=0")

hits_const <- data.table(cohort = c(rep("RA", 12), rep("ILD", 10)))
for (m in c("RA_dict_clone_count", "ILD_dict_clone_count", "RA_dict_hit_rate",
            "ILD_dict_hit_rate", "RA_dict_read_fraction_sum",
            "ILD_dict_read_fraction_sum", "RA_minus_ILD_count",
            "RA_minus_ILD_freq")) hits_const[[m]] <- 1
r_const <- safe_analyze_metrics(hits_const)
ok(r_const[[1]]$auc == 0.5 && !is.na(r_const[[1]]$status) &&
     grepl("常数", r_const[[1]]$status),
   "T3d 常数指标 AUC=0.5 且标注（不崩溃）")

hits_na <- data.table(cohort = c(rep("RA", 12), rep("ILD", 10)))
for (m in c("RA_dict_clone_count", "ILD_dict_clone_count", "RA_dict_hit_rate",
            "ILD_dict_hit_rate", "RA_dict_read_fraction_sum",
            "ILD_dict_read_fraction_sum", "RA_minus_ILD_count",
            "RA_minus_ILD_freq")) hits_na[[m]] <- NA_real_
r_na <- safe_analyze_metrics(hits_na)
ok(isTRUE(r_na[[1]]$auc == 0.5) || is.na(r_na[[1]]$auc),
   "T3e 全 NA 列不崩溃")

# AUC 可算时返回列表必须携带 roc 对象（回归：曾漏存 roc 致绘图 coords(NULL)）
hits_ok <- data.table(cohort = c(rep("RA", 12), rep("ILD", 10)))
set.seed(7)
for (m in c("RA_dict_clone_count", "ILD_dict_clone_count", "RA_dict_hit_rate",
            "ILD_dict_hit_rate", "RA_dict_read_fraction_sum",
            "ILD_dict_read_fraction_sum", "RA_minus_ILD_count",
            "RA_minus_ILD_freq")) hits_ok[[m]] <- c(rnorm(12, 5, 1), rnorm(10, 4, 1))
r_ok <- safe_analyze_metrics(hits_ok)
ok(!is.null(r_ok[[1]]$roc) && inherits(r_ok[[1]]$roc, "roc"),
   "T3f AUC 可算时返回 roc 对象（绘图依赖）")

# ====================================================================
# T4 重复 cdr3_aa 行聚合
# ====================================================================
cat("\n== T4 重复行聚合与非法值过滤 ==\n")
f4 <- tempfile(fileext = ".csv")
write.csv(data.frame(cdr3_aa = c("AAA", "AAA", "BBB", "", NA),
                     read_fraction = c(0.4, 0.3, 0.2, 0.5, 0.1)),
          f4, row.names = FALSE)
res4 <- suppressWarnings(read_sample_clones(f4))
ok(identical(sort(res4$clone), c("AAA", "BBB")),
   "T4a 重复/空串/NA 行处理后 clone 唯一且合法")
ok(isTRUE(all.equal(res4$rf[res4$clone == "AAA"], 0.7)),
   "T4b 重复行 read_fraction 求和")

# ====================================================================
# T5 完整流程置换：与手动重演严格对拍
# ====================================================================
cat("\n== T5 完整流程置换对拍 ==\n")
d5 <- make_synth(24, 16, 150, 0.3, 456)
seed5 <- 777L; n_perm5 <- 5L
fp <- run_full_pipeline_permutation(d5$clone_ids, d5$rf, d5$sample_ids, d5$lab,
                                    d5$strata, n_perm5, seed5, d5$U, 20, 10)
ok(identical(dim(fp$perm_auc), c(5L, 8L)) &&
     !anyNA(fp$perm_auc),
   "T5a 置换 AUC 矩阵维度 (5, 8) 且无 NA")
# 手动重演同一 RNG 序列（函数内 set.seed(seed) 后仅 permute 消耗 RNG）
set.seed(seed5)
labs_p <- lapply(seq_len(n_perm5), function(i)
  permute_labels_within_strata(d5$lab, d5$strata))
mt <- c("RA_dict_clone_count", "ILD_dict_clone_count",
        "RA_dict_hit_rate", "ILD_dict_hit_rate",
        "RA_dict_read_fraction_sum", "ILD_dict_read_fraction_sum",
        "RA_minus_ILD_count", "RA_minus_ILD_freq")
all_match <- TRUE
for (p in seq_len(n_perm5)) {
  hp <- run_dictionary_loocv(d5$clone_ids, d5$rf, d5$sample_ids, labs_p[[p]],
                             24L, 16L, d5$U, 20, 10)
  auc_manual <- sapply(mt, function(m) auc_from_scores(hp[[m]], hp$cohort == "RA"))
  if (!identical(as.numeric(fp$perm_auc[p, ]), as.numeric(auc_manual))) all_match <- FALSE
}
ok(all_match, "T5b 每次置换 AUC 与手动重演一致（完整流程重建）")

# ====================================================================
# T6 分层置换保持每层 RA/ILD 数量
# ====================================================================
cat("\n== T6 分层置换数量保持 ==\n")
d6 <- make_synth(24, 16, 100, 0.3, 789)
keep_counts <- TRUE
set.seed(99)
for (i in 1:50) {
  lab_p <- permute_labels_within_strata(d6$lab, d6$strata)
  for (s in unique(d6$strata)) {
    idx <- which(d6$strata == s)
    if (sum(d6$lab[idx] == "RA") != sum(lab_p[idx] == "RA")) keep_counts <- FALSE
  }
  if (!identical(sort(lab_p), sort(d6$lab))) keep_counts <- FALSE  # 整体是重排
}
ok(keep_counts, "T6 50 次置换每层 RA/ILD 数量不变、整体为重排")

# ====================================================================
# T7 同一 seed 完全可复现
# ====================================================================
cat("\n== T7 可重复性 ==\n")
d7 <- make_synth(24, 16, 100, 0.3, 111)
fp7a <- run_full_pipeline_permutation(d7$clone_ids, d7$rf, d7$sample_ids, d7$lab,
                                      d7$strata, 5L, 2026L, d7$U, 20, 10)
fp7b <- run_full_pipeline_permutation(d7$clone_ids, d7$rf, d7$sample_ids, d7$lab,
                                      d7$strata, 5L, 2026L, d7$U, 20, 10)
ok(identical(fp7a$perm_auc, fp7b$perm_auc) &&
     identical(fp7a$hits_obs, fp7b$hits_obs),
   "T7 同 seed 两次运行逐位一致")

# ====================================================================
# T8 主脚本实际加载的 utils 路径（GitHub 正式目录）
# ====================================================================
cat("\n== T8 utils 加载路径 ==\n")
src_lo <- readLines(file.path(script_dir, "CDR3_AA_dict_LOOCV.R"))
src_ma <- readLines(file.path(script_dir, "CDR3_AA_dict_material_analysis.R"))
pat <- "TRB/set/scripts/cdr3_analysis/CDR3_AA_dict_utils\\.R"
ok(any(grepl(pat, src_lo)) && any(grepl(pat, src_ma)),
   "T8a 两个主脚本引用 set/ 正式目录的 utils")
ok(file.exists("TRB/set/scripts/cdr3_analysis/CDR3_AA_dict_utils.R"),
   "T8b utils 文件可从项目根解析（相对路径有效）")

# ====================================================================
# T9 keep_clones 后 clone_id 与 rf 严格对齐
# ====================================================================
cat("\n== T9 keep_clones 后 clone_id 与 rf 严格对齐 ==\n")
td <- tempfile(); dir.create(td)
write.csv(data.frame(cdr3_aa = c("A1", "A2", "A3"),
                     read_fraction = c(0.5, 0.3, 0.2)),
          file.path(td, "s1_AA_clone_table.csv"), row.names = FALSE)
write.csv(data.frame(cdr3_aa = c("A2", "A3", "B1"),
                     read_fraction = c(0.4, 0.35, 0.25)),
          file.path(td, "s2_AA_clone_table.csv"), row.names = FALSE)
d9 <- load_sample_clone_data(td, c("s1_AA_clone_table.csv", "s2_AA_clone_table.csv"),
                             c("s1", "s2"), keep_clones = c("A2", "A3"))
ok(all(lengths(d9$clone_ids) == lengths(d9$rf)),
   "T9a 每样本 clone_id 与 rf 长度严格一致")
ok(d9$n_universe == 2L, "T9b universe 收缩到 keep 子集")
ok(identical(d9$clone_names[d9$clone_ids$s1], c("A2", "A3")) &&
     isTRUE(all.equal(d9$rf$s1, c(0.3, 0.2))),
   "T9c s1 的 clone 顺序与 rf 一一对应")
ok(identical(d9$clone_names[d9$clone_ids$s2], c("A2", "A3")) &&
     isTRUE(all.equal(d9$rf$s2, c(0.4, 0.35))),
   "T9d s2 的 clone 顺序与 rf 一一对应")

# ====================================================================
# T10 perm_p_two_sided 对 NA 置换统计量的处理
# ====================================================================
cat("\n== T10 perm_p_two_sided 对 NA 的处理 ==\n")
p10a <- perm_p_two_sided(0.2, c(0.1, NA, 0.3, NA, 0.15))
ok(is.finite(p10a) && isTRUE(all.equal(p10a, 0.5)),
   "T10a 部分 NA 时用有效置换数作分母（n_valid=3，(1+1)/4=0.5）")
ok(is.na(perm_p_two_sided(NA_real_, c(0.1, 0.2, 0.3))),
   "T10b observed 非有限 → NA")
ok(is.na(perm_p_two_sided(0.2, c(NA_real_, NA_real_, NA_real_))),
   "T10c 置换全 NA → NA（不崩溃）")
ok(isTRUE(all.equal(perm_p_two_sided(0.2, c(0.1, 0.3)), (1 + 1) / 3)),
   "T10d 无 NA 时与标准公式一致（n_valid = n_perm）")

# ====================================================================
# T11 maxT 仅使用完整置换行
# ====================================================================
cat("\n== T11 maxT 仅用完整置换行 ==\n")
m_names <- c("RA_dict_clone_count", "ILD_dict_clone_count", "RA_dict_hit_rate",
             "ILD_dict_hit_rate", "RA_dict_read_fraction_sum",
             "ILD_dict_read_fraction_sum", "RA_minus_ILD_count",
             "RA_minus_ILD_freq")
dev_obs11 <- setNames(c(0.10, 0.20, 0.05, 0.15, 0.10, 0.08, 0.12, 0.06), m_names)
dev_perm11 <- rbind(
  c(0.05, 0.02, 0.01, 0.03, 0.02, 0.01, 0.04, 0.01),   # 完整行 1（max=0.05）
  c(NA, 0.10, 0.05, 0.04, 0.03, 0.02, 0.06, 0.05),     # 不完整行
  c(0.30, 0.25, 0.20, 0.28, 0.22, 0.18, 0.26, 0.21),   # 完整行 2（max=0.30）
  c(0.02, NA, NA, 0.01, NA, NA, NA, NA)                # 不完整行
)
colnames(dev_perm11) <- m_names
mt11 <- maxT_p_values(dev_obs11, dev_perm11)
ok(mt11$n_complete == 2L, "T11a 完整行数正确（2/4）")
ok(isTRUE(all.equal(mt11$p_maxT[["RA_minus_ILD_count"]], (1 + 1) / 3)),
   "T11b p_maxT 分母 = 完整行数+1（max_dev 0.05/0.30，≥0.12 仅 0.30）")
ok(all(is.finite(mt11$p_maxT)), "T11c 完整行存在时全部指标 p_maxT 有限")
dev_perm11b <- matrix(NA_real_, nrow = 2, ncol = 8)
mt11b <- suppressWarnings(maxT_p_values(dev_obs11, dev_perm11b))
ok(mt11b$n_complete == 0L && all(is.na(mt11b$p_maxT)) &&
     !any(mt11b$p_maxT == -Inf, na.rm = TRUE),
   "T11d 无完整行 → 全 NA 且无 -Inf（warning 已抑制）")

# ====================================================================
cat(sprintf("\n结果：%d PASS / %d FAIL\n", n_pass, n_fail))
if (n_fail > 0L) stop("测试未全部通过")
cat("ALL TESTS PASSED\n")
