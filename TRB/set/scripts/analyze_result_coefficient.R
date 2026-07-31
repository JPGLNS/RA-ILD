library(dplyr)
#此脚本用于分析TRB模型100次训练的结果中，参数的情况
metadata_path <- file.path(
  "./TRB/set/train/result/06_cv_result_summary/collected/06_all_model_coefficients.csv.gz"
)

metadata <- read.csv(metadata_path)
head(metadata)
table(metadata$model)
M2_metadata <- metadata[metadata$model == "M2_static_tcr_public", ]
dim(M2_metadata)
M2_metadata <- subset(M2_metadata, M2_metadata$nonzero == "True")
dim(M2_metadata)
M2_metadata0 <- M2_metadata
colnames(M2_metadata)
M2_metadata <- M2_metadata[, c(
  "model",
  "feature_name",
  "coefficient",
  "absolute_coefficient"
)]
M2_metadata <- M2_metadata[M2_metadata$feature_name != "__INTERCEPT__", ]
dim(M2_metadata)

table(M2_metadata$feature_name)

# ============================================================
# 筛选：频率 > 50% 且 方向性 > 80%
# ============================================================
# 计算每个 feature_name 的统计量
feature_stats <- M2_metadata %>%
  dplyr::group_by(feature_name) %>%
  dplyr::summarise(
    count = dplyr::n(),
    frequency = count / 100, # 出现次数/100
    prop_positive = sum(coefficient > 0) / count,
    prop_negative = sum(coefficient < 0) / count,
    directionality = pmax(prop_positive, prop_negative), # 正负一致性（取多数方向的比例）
    .groups = "drop"
  )

cat("\n========== 所有 feature 汇总 ==========\n")
cat("共", nrow(feature_stats), "个 feature_name\n")
print(summary(feature_stats))

# 筛选：频率 > 0.5 且 方向性 > 0.8
filtered_50 <- feature_stats %>%
  dplyr::filter(frequency > 0.5, directionality > 0.8) %>%
  dplyr::arrange(dplyr::desc(frequency), dplyr::desc(directionality))

cat("\n========== 筛选结果：频率 > 50% 且 方向性 > 80% ==========\n")
cat("符合条件的 feature_name 数量：", nrow(filtered_50), "\n\n")
print(as.data.frame(filtered_50))

# 筛选：频率 > 0.4 且 方向性 > 0.8
filtered_40 <- feature_stats %>%
  dplyr::filter(frequency > 0.4, directionality > 0.8) %>%
  dplyr::arrange(dplyr::desc(frequency), dplyr::desc(directionality))

cat("\n========== 筛选结果：频率 > 40% 且 方向性 > 80% ==========\n")
cat("符合条件的 feature_name 数量：", nrow(filtered_40), "\n\n")
print(as.data.frame(filtered_40))
