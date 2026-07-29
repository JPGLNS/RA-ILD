# IGH Batch 08：repeat级3-mer与静态feature分组升级

## 1. 本补丁解决的问题

本补丁只处理以下三项需求，不改变Elastic Net、动态public、冻结3次123/51划分、inner CV、阈值选择和汇总算法。

1. **每个outer repeat重新选择3-mer词表**
   - Repeat 1只使用Train 1的123名患者生成词表；
   - Repeat 2、3同理；
   - 对应51名outer holdout只应用冻结词表；
   - 不在每个inner fold内重复选择。

2. **将原1083个静态预测变量拆成可组合feature组**
   - `core83`：83个非3-mer静态特征；
   - `repeat_3mer_weighted_topK`；
   - `repeat_3mer_unweighted_topK`；
   - `repeat_3mer_both_topK`；
   - 保留`static_igh_candidate_predictors`兼容别名，含本repeat的83+500+500。

3. **支持Top 50/100/200/500**
   - 每个repeat只计算一次完整ranking；
   - Top 50、100、200、500是同一ranking的嵌套前缀；
   - weighted与unweighted共享相同的3-mer序列名单。

## 2. 3-mer选择规则

沿用原始`build_02_sample_level_features.py`的无监督规则：

1. 当前outer训练集出现样本数至少5；
2. 当前outer训练集prevalence至少0.05；
3. 只有weighted和unweighted方差均低于`1e-12`时才删除；
4. 按训练集prevalence降序；
5. prevalence相同时按`variance_unweighted + variance_weighted`降序；
6. 仍相同时按3-mer字母顺序升序；
7. 保留前500，再截取Top 50/100/200/500。

整个词表选择不使用`cohort`、batch、material、sex或age。

## 3. 新增/替换文件

### 新增

- `IGH/set/src/ra_ild_igh/repeat_3mer_features.py`
- `IGH/set/scripts_v2/prepare_feature_ablation_scheme.py`
- `IGH/set/scripts_v2/test_batch08_repeat_3mer_feature_groups.py`
- `IGH/set/configs/schemes/igh_scheme_004_feature_ablation_repeat3.yaml`
- `IGH/set/scripts_v2/BATCH08_REPEAT_3MER_FEATURE_GROUPS_MANIFEST.json`

### 替换

- `IGH/set/scripts_v2/run_nested_cv_task.py`

替换后的任务脚本保持向后兼容：旧方案未配置`repeat_3mer_features.enabled: true`时，仍使用固定静态矩阵。

## 4. 安装

在仓库根目录执行：

```bash
bash /解压目录/apply_upgrade.sh /data/users/chenhaisheng/RA-ILD
```

安装脚本会：

- 检查仓库路径；
- 备份旧`run_nested_cv_task.py`；
- 复制新增/替换文件；
- 运行`py_compile`；
- 运行Batch 08 focused tests。

## 5. 准备feature消融方案

```bash
python IGH/set/scripts_v2/prepare_feature_ablation_scheme.py \
  --scheme IGH/set/configs/schemes/igh_scheme_004_feature_ablation_repeat3.yaml
```

输出配置：

```text
IGH/set/experiments/
igh_scheme_004_feature_ablation_repeat3/
00_config/resolved_config.yaml
```

该示例方案包含8个模型：

| 模型 | 特征 |
|---|---|
| A1 | 83 core |
| A2 | 83 + weighted Top 500 |
| A3 | 83 + unweighted Top 500 |
| A4 | 83 + weighted/unweighted Top 500，即原1000个3-mer版本 |
| A5 | 83 + 8 dynamic public |
| A6-50 | 83 + weighted/unweighted Top 50 |
| A6-100 | 83 + weighted/unweighted Top 100 |
| A6-200 | 83 + weighted/unweighted Top 200 |

A6 Top 500与A4完全相同，因此没有重复定义。A7需要先根据本轮结果冻结最佳Top-K/表示方式，再建立“最佳Top组合”和“最佳Top组合+public”方案。

## 6. 运行

### 启动3个outer repeat任务

```bash
python IGH/set/scripts_v2/run_all_outer_tasks.py \
  --config IGH/set/experiments/igh_scheme_004_feature_ablation_repeat3/00_config/resolved_config.yaml \
  --execute
```

当前默认参数下：

```text
8个模型 × 24组alpha/lambda × 5个inner folds × 3个repeat
= 2880次inner拟合
```

### 汇总原生holdout结果

```bash
python IGH/set/scripts_v2/aggregate_repeated_holdout_results.py \
  --config IGH/set/experiments/igh_scheme_004_feature_ablation_repeat3/00_config/resolved_config.yaml
```

正式比较使用每个repeat自己的51名原生holdout结果。

## 7. 每个repeat新增输出

每个`repeat_XX_fold_01`目录增加：

```text
06_repeat_3mer_vocabulary.csv
06_repeat_3mer_feature_matrix.csv.gz
06_repeat_3mer_audit.json
06_feature_group_manifest.csv
```

其中：

- `06_repeat_3mer_vocabulary.csv`：完整过滤、排序和Top 500信息；
- `06_repeat_3mer_feature_matrix.csv.gz`：174人按本repeat词表计算的500×2列；
- `06_repeat_3mer_audit.json`：训练ID哈希、词表哈希、参数和数量；
- `06_feature_group_manifest.csv`：每个命名feature组的具体列名和顺序。

## 8. 结果解释注意事项

- A2与A3使用同一组Top 500序列，仅表示方式不同；
- A4与A6 Top 500是同一个模型定义；
- 不同repeat的Top 500名单允许不同；
- feature stability汇总时，某个3-mer未进入某repeat候选词表等价于该repeat未选择；
- 本补丁未升级Batch 06的跨repeat冻结模型重放。feature优化阶段优先使用3次原生holdout；不同repeat词表的跨repeat测试需要后续单独适配。

## 9. 回滚

安装脚本会在原文件旁生成：

```text
IGH/set/scripts_v2/run_nested_cv_task.py.bak.batch08.<时间戳>
```

回滚时将该文件复制回原名，并删除本补丁新增文件即可。
