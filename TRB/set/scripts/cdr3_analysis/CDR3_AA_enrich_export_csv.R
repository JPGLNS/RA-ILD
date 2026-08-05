library(data.table)

setwd("/data/users/chenhaisheng/RA-ILD/")

# ====================================================================
# CDR3 AA RA/ILD-enrich clone 结果导出 CSV（不进行 K/M 两步操作）
# ====================================================================
# 条件：T=20%, Δ=10%（出现率口径，同 CDR3_AA_enrich_diff_barplot.R）
#   RA-enrich : RA 出现率 ≥ 20% 且 RA% − ILD% ≥ 10%
#   ILD-enrich: ILD 出现率 ≥ 20% 且 ILD% − RA% ≥ 10%
# 输出 6 个 CSV 至 TRB/result/：
#   CDR3_AA_enrich_RA_<part>_T20_D10.csv  与  CDR3_AA_enrich_ILD_<part>_T20_D10.csv
#   part ∈ {total, pbmc, buffycoat}
# 列（RA-enrich 文件）：
#   aa, RA_frequency, ILD_frequency, RA_minus_ILD,
#   RA_sample_count, ILD_sample_count, RA_read_count, ILD_read_count
# ILD-enrich 文件顺序镜像（ILD 在前），差值列为 ILD_minus_RA。
# 口径：
#   frequency（%）= 出现样本数 / 队列样本数 × 100，保留 2 位小数
#   read_count   = 队列内所有出现样本的 read_count 之和（整数）
# 排序：差值降序；输出全部符合条件 clone
# ====================================================================

THRESHOLD_PCT <- 20
DELTA_PCT <- 10

metadata <- read.csv("./TRB/metadata.csv")

aa_dir <- "./TRB/result/01_AA_clone_table/"
aa_files <- list.files(aa_dir, pattern = "_AA_clone_table\\.csv$")
aa_ids <- gsub("_AA_clone_table\\.csv$", "", aa_files)
aa_ids <- aa_ids[!grepl("^01_", aa_ids)]
aa_files_f <- paste0(aa_ids, "_AA_clone_table.csv")

cat("正在读取数据...（不进行 K/M 截取）\n")

# 每个样本保留 clone + read_count（供 read 总数统计）
all_data <- lapply(file.path(aa_dir, aa_files_f), function(f) {
  d <- fread(f, select = c("cdr3_aa", "read_count"))
  d <- d[!is.na(cdr3_aa)]
  d
})
names(all_data) <- aa_ids

sample_info <- merge(data.frame(libraryid = aa_ids),
                     metadata[, c("libraryid", "material", "cohort")], by = "libraryid")
rownames(sample_info) <- sample_info$libraryid

parts <- list(
  total     = sample_info$libraryid,
  pbmc      = sample_info$libraryid[sample_info$material == "PBMC"],
  buffycoat = sample_info$libraryid[sample_info$material == "buffycoat"]
)

out_dir <- "./TRB/result/"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# 组内统计：每个 clone 的出现样本数与 read 总和
group_stats <- function(sample_ids) {
  dt <- rbindlist(all_data[sample_ids])
  dt[, .(n_samples = .N, read_sum = sum(read_count)), by = cdr3_aa]
}

for (part_name in names(parts)) {
  ids <- parts[[part_name]]
  ra_ids  <- ids[sample_info[ids, "cohort"] == "RA"]
  ild_ids <- ids[sample_info[ids, "cohort"] == "ILD"]
  n_ra  <- length(ra_ids)
  n_ild <- length(ild_ids)

  cat("========================================================================\n")
  cat(" 部分：", part_name, "（", length(ids), " 个样本，RA ", n_ra, " / ILD ", n_ild, "）\n", sep = "")
  cat("========================================================================\n")

  ra_s  <- group_stats(ra_ids)
  ild_s <- group_stats(ild_ids)
  setnames(ra_s,  c("clone", "RA_sample_count", "RA_read_count"))
  setnames(ild_s, c("clone", "ILD_sample_count", "ILD_read_count"))

  base <- data.table(clone = union(ra_s$clone, ild_s$clone))
  base <- merge(base, ra_s,  by = "clone", all.x = TRUE)
  base <- merge(base, ild_s, by = "clone", all.x = TRUE)
  for (col in c("RA_sample_count", "ILD_sample_count", "RA_read_count", "ILD_read_count")) {
    set(base, which(is.na(base[[col]])), col, 0)
  }

  base[, RA_frequency := RA_sample_count / n_ra * 100]
  base[, ILD_frequency := ILD_sample_count / n_ild * 100]
  base[, delta := RA_frequency - ILD_frequency]

  # ---- RA-enrich：RA% ≥ T 且 RA% − ILD% ≥ Δ，按差值降序 ----
  ra_out <- base[RA_frequency >= THRESHOLD_PCT & delta >= DELTA_PCT][order(-delta)]
  ra_out[, `:=`(RA_frequency  = sprintf("%.2f", RA_frequency),
                ILD_frequency = sprintf("%.2f", ILD_frequency),
                delta         = sprintf("%.2f", delta))]
  ra_out <- ra_out[, .(aa = clone, RA_frequency, ILD_frequency, RA_minus_ILD = delta,
                       RA_sample_count, ILD_sample_count, RA_read_count, ILD_read_count)]
  write.csv(ra_out,
            paste0(out_dir, "CDR3_AA_enrich_RA_", part_name, "_T", THRESHOLD_PCT, "_D", DELTA_PCT, ".csv"),
            row.names = FALSE)

  # ---- ILD-enrich：ILD% ≥ T 且 ILD% − RA% ≥ Δ，按 ILD−RA 降序 ----
  ild_out <- base[ILD_frequency >= THRESHOLD_PCT & delta <= -DELTA_PCT]
  ild_out[, ild_delta := ILD_frequency - RA_frequency]
  ild_out <- ild_out[order(-ild_delta)]
  ild_out[, `:=`(ILD_frequency = sprintf("%.2f", ILD_frequency),
                 RA_frequency  = sprintf("%.2f", RA_frequency),
                 ild_delta     = sprintf("%.2f", ild_delta))]
  ild_out <- ild_out[, .(aa = clone, ILD_frequency, RA_frequency, ILD_minus_RA = ild_delta,
                         ILD_sample_count, RA_sample_count, ILD_read_count, RA_read_count)]
  write.csv(ild_out,
            paste0(out_dir, "CDR3_AA_enrich_ILD_", part_name, "_T", THRESHOLD_PCT, "_D", DELTA_PCT, ".csv"),
            row.names = FALSE)

  cat("  RA-enrich:", nrow(ra_out), "条 | ILD-enrich:", nrow(ild_out), "条 已导出\n")

  rm(ra_s, ild_s, base, ra_out, ild_out)
  gc()
}

cat("\n全部 6 个 CSV 导出完成！保存至:", out_dir, "\n")
