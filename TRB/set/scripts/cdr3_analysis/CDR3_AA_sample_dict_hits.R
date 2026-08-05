library(data.table)

setwd("/data/users/chenhaisheng/RA-ILD/")

# ====================================================================
# 样本 × enrich 字典命中统计（探索 enrich 结果的判别力）
# ====================================================================
# 字典：total 部分 T=20%、Δ=10% 的 RA-enrich / ILD-enrich clone
#       （TRB/result/CDR3_AA_enrich_RA_total_T20_D10.csv 等，全部符合条件 clone）
# 对每个样本（174 例，不 K/M）统计：
#   1. 命中 RA 字典的 clone 数量
#   2. 命中 ILD 字典的 clone 数量
#   3. 样本 cohort（RA/ILD）
#   4. 命中 RA 字典 clone 的 read_fraction 之和（样本内丰度占比）
#   5. 命中 ILD 字典 clone 的 read_fraction 之和
# 输出：TRB/result/CDR3_AA_sample_dict_hits_T20_D10.csv
# ====================================================================

out_dir <- "./TRB/result/"

ra_dict  <- fread(file.path(out_dir, "CDR3_AA_enrich_RA_total_T20_D10.csv"),  select = "aa")$aa
ild_dict <- fread(file.path(out_dir, "CDR3_AA_enrich_ILD_total_T20_D10.csv"), select = "aa")$aa
n_ra_dict  <- length(ra_dict)
n_ild_dict <- length(ild_dict)
cat("RA 字典:", n_ra_dict, "条 | ILD 字典:", n_ild_dict, "条\n")

metadata <- read.csv("./TRB/metadata.csv")

aa_dir <- "./TRB/result/01_AA_clone_table/"
aa_files <- list.files(aa_dir, pattern = "_AA_clone_table\\.csv$")
aa_ids <- gsub("_AA_clone_table\\.csv$", "", aa_files)
aa_ids <- aa_ids[!grepl("^01_", aa_ids)]
# 仅保留 metadata 中存在的样本（克隆表有 178 个，其中 4 个无 metadata 信息）
aa_ids <- aa_ids[aa_ids %in% metadata$libraryid]
aa_files_f <- paste0(aa_ids, "_AA_clone_table.csv")

sample_info <- merge(data.frame(libraryid = aa_ids),
                     metadata[, c("libraryid", "material", "cohort")], by = "libraryid")

cat("正在统计", length(aa_ids), "个样本...\n", sep = "")

res <- lapply(seq_along(aa_ids), function(i) {
  dt <- fread(file.path(aa_dir, aa_files_f[i]), select = c("cdr3_aa", "read_fraction"))
  dt <- dt[!is.na(cdr3_aa)]
  m_ra  <- dt[cdr3_aa %in% ra_dict]
  m_ild <- dt[cdr3_aa %in% ild_dict]
  data.table(
    libraryid = aa_ids[i],
    RA_dict_clone_count  = nrow(m_ra),
    ILD_dict_clone_count = nrow(m_ild),
    RA_dict_read_fraction_sum  = round(sum(m_ra$read_fraction),  6),
    ILD_dict_read_fraction_sum = round(sum(m_ild$read_fraction), 6)
  )
})
res <- rbindlist(res)
res <- merge(res, sample_info[, c("libraryid", "cohort", "material")], by = "libraryid")
res <- res[order(cohort, libraryid)]
# 命中率 = 命中 clone 数 / 对应字典总数（RA 字典 / ILD 字典）
res[, RA_dict_hit_rate  := round(RA_dict_clone_count  / n_ra_dict,  4)]
res[, ILD_dict_hit_rate := round(ILD_dict_clone_count / n_ild_dict, 4)]
# 衍生差值
res[, RA_minus_ILD_count := RA_dict_clone_count - ILD_dict_clone_count]
res[, RA_minus_ILD_freq   := round(RA_dict_read_fraction_sum - ILD_dict_read_fraction_sum, 6)]
# 列顺序：libraryid, RA/ILD_dict_clone_count, cohort, RA/ILD_dict_read_fraction_sum,
#         RA/ILD_dict_hit_rate, RA_minus_ILD_count, RA_minus_ILD_freq, material
res <- res[, .(libraryid, RA_dict_clone_count, ILD_dict_clone_count, cohort,
               RA_dict_read_fraction_sum, ILD_dict_read_fraction_sum,
               RA_dict_hit_rate, ILD_dict_hit_rate,
               RA_minus_ILD_count, RA_minus_ILD_freq, material)]

write.csv(res, file.path(out_dir, "CDR3_AA_sample_dict_hits_T20_D10.csv"), row.names = FALSE)

cat("已完成！输出: ", file.path(out_dir, "CDR3_AA_sample_dict_hits_T20_D10.csv"), "\n", sep = "")
cat("RA 队列:", sum(res$cohort == "RA"), "个样本 | ILD 队列:", sum(res$cohort == "ILD"), "个样本\n")
