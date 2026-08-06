# TRB Upgrade Batch 16 — Enrich-dictionary repeated 5-fold validation

## 1. 目的

Batch 16 为现有 CDR3-AA RA/RA-ILD enrich 字典分析增加一个**独立、可重复、无训练—验证泄漏的 repeated stratified 5-fold 验证流程**，用于回答：

> 由训练样本构建的 RA / RA-ILD enrich 字典，能否在未参与字典构建的验证样本中保持方向和区分能力？

该升级完全是增量功能，不覆盖现有：

- 样本内 enrich 字典分析；
- LOOCV 分析；
- 完整流程置换检验；
- Elastic Net / Linear SVM 建模框架；
- 既有结果和图片。

GitHub 参考版本：

```text
repository: JPGLNS/RA-ILD
branch: feature/trb-framework-v2
commit: c7e6ebda18ae78a5ee4396386f1595de62acfebf
```

## 2. 为什么增加 repeated 5-fold

现有非 LOO 分析使用全部样本构建字典，再评价同一批样本，因此属于样本内 apparent performance，可能明显乐观。

现有 LOOCV 虽然排除了当前样本，但每个样本都对应一套独立字典。对于 T=20%、Δ=10% 这种离散硬阈值规则，留出一个 RA 或 RA-ILD 样本可能改变整数阈值和字典组成，使不同样本的分数来自不同尺度的字典。

Batch 16 改为：

```text
每次重复生成 5 个分层 fold
            │
            ├─ fold 1：其余 4 fold 构建字典 → fold 1 全部样本共同评分
            ├─ fold 2：其余 4 fold 构建字典 → fold 2 全部样本共同评分
            ├─ ...
            └─ fold 5：其余 4 fold 构建字典 → fold 5 全部样本共同评分
```

因此，同一验证 fold 中所有 RA 和 RA-ILD 样本使用**同一套训练字典**，避免 LOOCV 中“样本类别不同导致训练组规模和整数门槛不同”的特殊不稳定性。

## 3. 与现有机器学习 5 折框架的关系

分折逻辑参考现有 `TRB/set/scripts/build_05_cv_splits.py`：

- cohort 为硬分层目标；
- fold 总样本数尽量均衡；
- total 分析额外平衡 batch、material、sex；
- PBMC / buffycoat 子集额外平衡 batch、sex；
- 从多个候选分折中选择平衡度更好的方案；
- seed 可复现；
- 每个 repeat 使用不同且可审计的 partition signature；
- 每个样本在每个 repeat 中恰好获得一次 out-of-fold 分数。

默认配置：

```text
5 folds × 100 repeats
300 candidate partitions / repeat
seed = 20260711
T = 20%
Δ = 10%
```

## 4. 严格的训练内字典构建

每个 fold 中仅使用训练样本完成：

1. RA / RA-ILD 样本出现计数；
2. T=20% 候选阈值；
3. Δ=10% enrich 方向判定；
4. RA 和 RA-ILD 字典构建。

验证样本不会参与：

- clone 筛选方向；
- 字典成员确定；
- fold 字典大小；
- 任何阈值选择。

全局预筛选仅使用 clone 在全部样本中的**总出现次数数学上界**，不读取 cohort 标签，不改变任何可能进入训练字典的 clone。

## 5. 输出指标

### 原有 8 个主要指标

```text
RA_dict_clone_count
ILD_dict_clone_count
RA_dict_hit_rate
ILD_dict_hit_rate
RA_dict_read_fraction_sum
ILD_dict_read_fraction_sum
RA_minus_ILD_count
RA_minus_ILD_read_fraction
```

### 新增诊断指标

```text
RA_minus_ILD_hit_rate
RA_hit_fraction_of_sample
ILD_hit_fraction_of_sample
RA_minus_ILD_sample_fraction
sample_unique_clone_count
```

其中：

- `dictionary hit rate` = 命中 clone 数 / 该 fold 字典 clone 数；
- `sample hit fraction` = 命中 clone 数 / 样本自身 unique AA clone 数；
- 两者分别用于诊断字典大小不平衡和样本 repertoire richness 影响。

## 6. AUC 方向

为与现有图保持一致：

```text
RA = ROC 阳性类
```

输出同时保留：

- `auc_RA_positive_raw`：原始方向 AUC；
- `auc_RAILD_positive_raw = 1 - raw AUC`；
- `directional_auc = max(AUC, 1-AUC)`；
- `higher_score_cohort`：明确指出分数更高的是 RA 还是 RA-ILD。

因此，RA-ILD 字典指标如果 raw AUC <0.5，并不表示无效，而是表示其值在 RA-ILD 中更高。

## 7. 诊断输出

每个 scope（total / pbmc / buffycoat）都会输出：

- 固定分折 assignments；
- 每个 repeat 的全部 OOF 分数；
- 每个样本跨 repeats 的平均 OOF 分数；
- 每个完整 repeat 将 5 个验证折合并后计算一个 pooled OOF AUC（主要结果）；
- 100 repeats 因而产生每指标 100 个主要 AUC；
- 同时保存 100×5=500 个 fold-level AUC 作为诊断，但不视为 500 个独立结果，也不以 5 个 fold AUC 的简单平均代替 pooled OOF AUC；
- repeat-level AUC 中位数、标准差和 2.5%–97.5% 分折敏感性范围；
- 每折字典大小和训练/验证样本数；
- fold balance；
- 同一 repeat 内不同 fold 字典的 pairwise Jaccard；
- 各指标与样本 unique clone 数的 Spearman 相关；
- 可选的 fold-specific 字典成员表。

注意：跨 repeat 的 2.5%–97.5% 范围用于描述**分折敏感性**，不是独立样本意义上的置信区间。

## 8. 安装

解压后执行：

```bash
cd /path/to/TRB_enrich_dictionary_5fold_batch16_v1.1.1

bash apply_upgrade.sh \
  /data/users/chenhaisheng/RA-ILD
```

安装器会：

1. 检查目标仓库的现有 5-fold 和字典分析基础文件；
2. 只新增 Batch 16 文件；
3. 检查 Python 依赖和语法；
4. 执行 25 项 focused tests；
5. 不修改任何已有结果。

若目标中已有同名但内容不同的 Batch 16 文件，默认拒绝覆盖。人工审阅后可使用：

```bash
bash apply_upgrade.sh \
  /data/users/chenhaisheng/RA-ILD \
  --force
```

`--force` 会先创建时间戳备份。

## 9. 快速 smoke test

```bash
cd /data/users/chenhaisheng/RA-ILD

bash TRB/set/scripts_v2/run_enrich_dictionary_5fold.sh \
  /data/users/chenhaisheng/RA-ILD \
  --smoke \
  --overwrite
```

smoke 配置为 5 folds × 1 repeat、最多 20 个候选分折，用于验证完整文件读取、建字典、OOF 评分、CSV 和绘图路径。

## 10. 正式运行

```bash
cd /data/users/chenhaisheng/RA-ILD

nohup bash TRB/set/scripts_v2/run_enrich_dictionary_5fold.sh \
  /data/users/chenhaisheng/RA-ILD \
  --folds 5 \
  --repeats 100 \
  --candidates 300 \
  --seed 20260711 \
  --threshold-pct 20 \
  --delta-pct 10 \
  --save-dictionaries \
  --overwrite \
  > TRB/result/enrich_dictionary_5fold_batch16.log 2>&1 &
```

查看进程：

```bash
ps -ef | grep run_enrich_dictionary_5fold | grep -v grep
```

查看日志：

```bash
tail -f TRB/result/enrich_dictionary_5fold_batch16.log
```

## 11. 输出位置

表格：

```text
TRB/result/enrich_dictionary_5fold_cv_v1/
├── enrich_5fold_run_manifest.json
├── enrich_5fold_summary.md
├── total/
├── pbmc/
└── buffycoat/
```

图片：

```text
TRB/gradient/enrich_dictionary_5fold_cv_v1/
├── total/
├── pbmc/
└── buffycoat/
```

每个 scope 默认生成 5 张图：

1. sample-averaged OOF boxplot；
2. raw-direction OOF ROC；
3. repeated split AUC distribution；
4. fold dictionary size + Jaccard diagnostics；
5. repertoire richness correlation。

## 12. 建议解读顺序

1. 先看 `enrich_5fold_metric_summary_*.csv` 中每个指标的 AUC 中位数与方向；
2. 再看 100 个完整 repeats 的 pooled OOF AUC 是否稳定；
3. 查看字典 Jaccard，判断 hard-threshold 字典是否稳定；
4. 查看与 `sample_unique_clone_count` 的相关性，判断两套字典同时在 RA 中升高是否由 richness 驱动；
5. 对比 total、PBMC、buffycoat；
6. 将非 LOO 结果仅作为 apparent performance，将 repeated 5-fold OOF 作为主要内部泛化评估。

## 13. 方法学边界

- repeated 5-fold 仍然是内部验证，不是独立外部验证；
- 本升级不进行参数调优，因此不需要 inner CV；
- T=20%、Δ=10% 应作为预先固定规则；
- 若未来比较多个 T/Δ 组合并选择最佳组合，阈值选择必须嵌套在训练 fold 内；
- 本版本不重复执行 1000 次完整流程置换；它主要用于评估分折稳定性、方向和泛化。正式显著性可以在确认 5-fold 设计后再单独扩展。
