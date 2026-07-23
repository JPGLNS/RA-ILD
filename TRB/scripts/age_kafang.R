setwd("/data/users/chenhaisheng/RA-ILD/")

# ============================================================
# 脚本：TRB 队列的年龄与性别统计检验
# 目的：检验 ILD vs RA 两组在年龄和性别上是否存在显著差异
# ============================================================

library(car) # LeveneTest
library(ggplot2) # 绘图
library(dplyr) # 数据处理
library(ggpubr) # 出版级图形

# ------------------------------------------------------------
# 1. 读取数据
# ------------------------------------------------------------
metadata <- read.csv("./TRB/metadata.csv")
cat("\n========== 数据概览 ==========\n")
cat("总样本数:", nrow(metadata), "\n")
cat("变量:", paste(colnames(metadata), collapse = ", "), "\n\n")
str(metadata)

# ------------------------------------------------------------
# 2. 数据清洗 & 描述性统计
# ------------------------------------------------------------
# 确保 cohort 和 sex 为因子
metadata$cohort <- as.factor(metadata$cohort)
metadata$sex <- as.factor(metadata$sex)
metadata$age <- as.numeric(metadata$age)

# 剔除关键变量缺失的样本
metadata <- subset(metadata, !is.na(age) & !is.na(cohort) & !is.na(sex))
cat("\n清洗后样本数:", nrow(metadata), "\n")

# 分组描述统计
cat("\n========== 年龄描述统计（按 cohort） ==========\n")
cohort_stats <- metadata %>%
  group_by(cohort) %>%
  summarise(
    n = n(),
    mean = mean(age),
    sd = sd(age),
    median = median(age),
    min = min(age),
    max = max(age),
    skew = (sum((age - mean(age))^3) / length(age)) / (sd(age)^3),
    kurt = (sum((age - mean(age))^4) / length(age)) / (sd(age)^4) - 3
  )
print(as.data.frame(cohort_stats))

cat("\n========== 性别分布（按 cohort） ==========\n")
sex_table <- table(metadata$sex, metadata$cohort)
print(sex_table)
cat("\n行比例:\n")
print(round(prop.table(sex_table, margin = 2) * 100, 1))

# ------------------------------------------------------------
# 3. 年龄 ~ cohort 分析
# ------------------------------------------------------------
cat("\n\n========== 3. 年龄 vs Cohort 分析 ==========\n")

ild_age <- metadata$age[metadata$cohort == "ILD"]
ra_age <- metadata$age[metadata$cohort == "RA"]

# ----- 3a. 正态性检验（Shapiro-Wilk） -----
cat("\n--- 正态性检验 (Shapiro-Wilk) ---\n")
sw_ild <- shapiro.test(ild_age)
sw_ra <- shapiro.test(ra_age)
cat(sprintf("ILD 组: W = %.4f, p = %.4f\n", sw_ild$statistic, sw_ild$p.value))
cat(sprintf("RA  组: W = %.4f, p = %.4f\n", sw_ra$statistic, sw_ra$p.value))

normal_ild <- sw_ild$p.value > 0.05
normal_ra <- sw_ra$p.value > 0.05
cat(sprintf("ILD 组正态性: %s\n", ifelse(normal_ild, "通过 ✓", "不通过 ✗")))
cat(sprintf("RA  组正态性: %s\n", ifelse(normal_ra, "通过 ✓", "不通过 ✗")))

# ----- 3b. 方差齐性检验 (Levene) -----
cat("\n--- 方差齐性检验 (Levene's Test) ---\n")
levene_result <- leveneTest(age ~ cohort, data = metadata, center = median)
cat(sprintf(
  "Levene F = %.4f, df1 = %d, df2 = %d, p = %.4f\n",
  levene_result$`F value`[1],
  levene_result$Df[1],
  levene_result$Df[2],
  levene_result$`Pr(>F)`[1]
))
equal_var <- levene_result$`Pr(>F)`[1] > 0.05
cat(sprintf("方差齐性: %s\n", ifelse(equal_var, "通过 ✓", "不通过 ✗")))

# ----- 3c. 选择并执行检验 -----
cat("\n--- 检验选择 ---\n")
cat(sprintf(
  "正态性: ILD=%s, RA=%s; 方差齐性: %s\n",
  ifelse(normal_ild, "✓", "✗"),
  ifelse(normal_ra, "✓", "✗"),
  ifelse(equal_var, "✓", "✗")
))

if (normal_ild && normal_ra && equal_var) {
  cat("→ 使用 Student's t 检验（独立样本，等方差）\n\n")
  t_result <- t.test(age ~ cohort, data = metadata, var.equal = TRUE)
  test_name <- "Student's t-test"
} else if (normal_ild && normal_ra && !equal_var) {
  cat("→ 使用 Welch's t 检验（独立样本，不等方差）\n\n")
  t_result <- t.test(age ~ cohort, data = metadata, var.equal = FALSE)
  test_name <- "Welch's t-test"
} else {
  cat("→ 数据不正态，使用 Mann-Whitney U 检验（Wilcoxon 秩和）\n\n")
  t_result <- wilcox.test(
    age ~ cohort,
    data = metadata,
    exact = FALSE,
    conf.int = TRUE
  )
  test_name <- "Mann-Whitney U test"
}

cat(sprintf("========== 检验结果 (%s) ==========\n", test_name))
print(t_result)

# ----- 3e. 稳健性验证：同时报告 Welch's t 和 Mann-Whitney U -----
cat("\n--- 稳健性验证（同时报告其他方法结果） ---\n")
welch_result <- t.test(age ~ cohort, data = metadata, var.equal = FALSE)
cat(sprintf(
  "Welch's t-test:  t = %.4f, df = %.1f, p = %.6f\n",
  welch_result$statistic,
  welch_result$parameter,
  welch_result$p.value
))
wilcox_result <- wilcox.test(
  age ~ cohort,
  data = metadata,
  exact = FALSE,
  conf.int = TRUE
)
cat(sprintf(
  "Mann-Whitney U:  W = %.0f, p = %.6f\n",
  wilcox_result$statistic,
  wilcox_result$p.value
))
cat("（三种方法结论一致则结果稳健可信）\n")

# ----- 3d. 效应量 -----
cat("\n--- 效应量 ---\n")
# Cohen's d (仅正态适用)
if (normal_ild && normal_ra) {
  n1 <- length(ild_age)
  n2 <- length(ra_age)
  s1 <- sd(ild_age)
  s2 <- sd(ra_age)
  pooled_sd <- sqrt(((n1 - 1) * s1^2 + (n2 - 1) * s2^2) / (n1 + n2 - 2))
  cohens_d <- abs(mean(ild_age) - mean(ra_age)) / pooled_sd
  cat(sprintf(
    "Cohen's d = %.3f (%s)\n",
    cohens_d,
    ifelse(
      cohens_d < 0.2,
      "微小",
      ifelse(cohens_d < 0.5, "小", ifelse(cohens_d < 0.8, "中等", "大"))
    )
  ))
}

# Cliff's delta (非参数效应量，始终计算)
# 手动计算 Cliff's delta
cliff_delta <- function(x, y) {
  # x 和 y 的所有两两比较
  greater <- sum(outer(x, y, function(a, b) ifelse(a > b, 1, 0)))
  less <- sum(outer(x, y, function(a, b) ifelse(a < b, 1, 0)))
  (greater - less) / (length(x) * length(y))
}
delta <- cliff_delta(ild_age, ra_age)
cat(sprintf(
  "Cliff's delta = %.3f (%s)\n",
  delta,
  ifelse(
    abs(delta) < 0.147,
    "微小",
    ifelse(abs(delta) < 0.33, "小", ifelse(abs(delta) < 0.474, "中等", "大"))
  )
))

# ------------------------------------------------------------
# 4. Sex ~ Cohort 分析（卡方检验）
# ------------------------------------------------------------
cat("\n\n========== 4. 性别 vs Cohort 分析 ==========\n")

sex_contingency <- table(metadata$sex, metadata$cohort)
cat("\n列联表:\n")
print(sex_contingency)

# 检查期望频数
expected <- chisq.test(sex_contingency)$expected
cat("\n期望频数:\n")
print(round(expected, 2))

# 判断是否需要 Fisher 精确检验
min_expected <- min(expected)
cat(sprintf("\n最小期望频数: %.2f\n", min_expected))

if (min_expected < 5) {
  cat("→ 期望频数 < 5，使用 Fisher 精确检验\n\n")
  fisher_result <- fisher.test(sex_contingency)
  cat("========== Fisher's Exact Test ==========\n")
  print(fisher_result)
  sex_test_name <- "Fisher's exact test"
  sex_p <- fisher_result$p.value
} else {
  cat("→ 期望频数 ≥ 5，使用 Pearson 卡方检验\n\n")
  chisq_result <- chisq.test(sex_contingency, correct = FALSE)
  cat("========== Pearson's Chi-squared Test ==========\n")
  print(chisq_result)
  sex_test_name <- "Pearson's Chi-squared test"
  sex_p <- chisq_result$p.value
}

# Cramér's V 效应量
n_total <- sum(sex_contingency)
chi_stat <- if (min_expected < 5) {
  # 对 Fisher 也计算近似的 χ² 用于效应量
  chisq.test(sex_contingency, correct = FALSE)$statistic
} else {
  chisq_result$statistic
}
df_min <- min(nrow(sex_contingency) - 1, ncol(sex_contingency) - 1)
cramers_v <- sqrt(chi_stat / (n_total * df_min))
cat(sprintf(
  "\nCramér's V = %.3f (%s)\n",
  cramers_v,
  ifelse(
    cramers_v < 0.1,
    "微小",
    ifelse(cramers_v < 0.3, "小", ifelse(cramers_v < 0.5, "中等", "大"))
  )
))

# ------------------------------------------------------------
# 5. 可视化
# ------------------------------------------------------------
cat("\n\n========== 5. 绘制图形 ==========\n")

# 5a. 年龄箱线图 + 散点
p1 <- ggplot(metadata, aes(x = cohort, y = age, fill = cohort)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.5, width = 0.5) +
  geom_jitter(width = 0.12, size = 1.5, alpha = 0.6, color = "grey30") +
  stat_summary(
    fun = mean,
    geom = "point",
    shape = 18,
    size = 3,
    color = "red"
  ) +
  labs(
    title = "Age distribution by cohort",
    subtitle = sprintf(
      "%s: %s",
      test_name,
      ifelse(
        t_result$p.value < 0.0001,
        "p < 0.0001",
        sprintf("p = %.4f", t_result$p.value)
      )
    ),
    x = "Cohort",
    y = "Age (years)"
  ) +
  theme_classic(base_size = 13) +
  theme(legend.position = "none") +
  scale_fill_manual(values = c("ILD" = "#E64B35", "RA" = "#4DBBD5"))

# 5b. 性别柱状图
sex_df <- as.data.frame(sex_table)
colnames(sex_df) <- c("Sex", "Cohort", "Count")
sex_df <- sex_df %>%
  group_by(Cohort) %>%
  mutate(Proportion = Count / sum(Count) * 100)

p2 <- ggplot(sex_df, aes(x = Cohort, y = Proportion, fill = Sex)) +
  geom_col(position = "stack", width = 0.55, color = "white", linewidth = 0.3) +
  geom_text(
    aes(label = sprintf("%d\n(%.1f%%)", Count, Proportion)),
    position = position_stack(vjust = 0.5),
    size = 3.5,
    color = "white",
    fontface = "bold"
  ) +
  labs(
    title = "Sex distribution by cohort",
    subtitle = sprintf(
      "%s: %s",
      sex_test_name,
      ifelse(sex_p < 0.0001, "p < 0.0001", sprintf("p = %.4f", sex_p))
    ),
    x = "Cohort",
    y = "Percentage (%)"
  ) +
  theme_classic(base_size = 13) +
  scale_fill_manual(values = c("female" = "#F39B7F", "male" = "#8491B4"))

# 合并输出
combined <- ggarrange(p1, p2, ncol = 2, labels = c("A", "B"))
ggsave("./TRB/cohort_age_sex_test.pdf", combined, width = 10, height = 5)
cat("图形已保存至: ./TRB/cohort_age_sex_test.pdf\n")

# ------------------------------------------------------------
# 6. 综合结论
# ------------------------------------------------------------
cat("\n\n============================================\n")
cat("              综合结论\n")
cat("============================================\n\n")

cat(sprintf("1. 年龄 (age) ~ 队列 (cohort): %s\n", test_name))
if (t_result$p.value < 0.0001) {
  cat(sprintf("   检验统计量 p = %.2e (p < 0.0001)\n", t_result$p.value))
} else {
  cat(sprintf("   检验统计量 p = %.4f\n", t_result$p.value))
}
if (t_result$p.value < 0.05) {
  cat("   结论: ILD 和 RA 两组年龄存在显著差异 (p < 0.05)\n")
  mean_diff <- abs(mean(ild_age) - mean(ra_age))
  cat(sprintf("   平均年龄差: %.1f 岁\n", mean_diff))
  if (mean(ild_age) > mean(ra_age)) {
    cat("   ILD 组平均年龄高于 RA 组\n")
  } else {
    cat("   RA 组平均年龄高于 ILD 组\n")
  }
} else {
  cat("   结论: ILD 和 RA 两组年龄未见显著差异 (p >= 0.05)\n")
}

cat(sprintf("\n2. 性别 (sex) ~ 队列 (cohort): %s\n", sex_test_name))
if (sex_p < 0.0001) {
  cat(sprintf("   检验统计量 p = %.2e (p < 0.0001)\n", sex_p))
} else {
  cat(sprintf("   检验统计量 p = %.4f\n", sex_p))
}
if (sex_p < 0.05) {
  cat("   结论: ILD 和 RA 两组性别分布存在显著差异 (p < 0.05)\n")
} else {
  cat("   结论: ILD 和 RA 两组性别分布未见显著差异 (p >= 0.05)\n")
}

cat("\n============================================\n")
