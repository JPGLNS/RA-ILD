# Batch 14：可配置的 inner-CV 调参指标

## 目标

Batch 14 将 TRB V2 框架中原本写死的 `pooled_inner_roc_auc` 调参规则改为方案级可配置规则。

支持的主指标：

- `roc_auc`：保持历史行为；
- `log_loss`：优先选择 pooled inner OOF log-loss 最低的 alpha/lambda。

本升级只改变“每个模型内部如何选择 alpha/lambda”，不改变：

- Elastic Net logistic regression；
- alpha/lambda 候选网格；
- inner/outer 划分；
- Youden threshold；
- outer holdout 评价指标；
- repeated-holdout 的最终描述性排名。

## 向后兼容

旧 YAML 中没有 `model_selection.tuning_primary_metric` 时：

```yaml
model_selection:
  tuning_primary_metric: roc_auc
```

将被作为默认行为。

因此此前完成的 M0–M2、Feature14 和其他 ROC 方案不需要重新训练，也不会被重新解释为 log-loss 方案。

## 方案配置

### ROC 模式

```yaml
overrides:
  model_selection:
    tuning_primary_metric: roc_auc
```

排序规则：

1. `pooled_inner_roc_auc`：降序；
2. `pooled_inner_pr_auc`：降序；
3. `lambda`：降序；
4. `l1_ratio_alpha`：降序。

### Log-loss 模式

```yaml
overrides:
  model_selection:
    tuning_primary_metric: log_loss
```

排序规则：

1. `pooled_inner_log_loss`：升序；
2. `pooled_inner_brier_score`：升序；
3. `pooled_inner_roc_auc`：降序；
4. `lambda`：降序；
5. `l1_ratio_alpha`：降序。

在 scheme YAML 中只需要指定 `tuning_primary_metric`。`prepare_analysis_scheme.py` 会自动同步：

- `model_selection.candidate_sort`；
- `nested_cv.candidate_selection_policy`。

不要手动编辑已经生成并开始运行的 `resolved_config.yaml`。

## 新增输出

### `06_inner_tuning_results.csv`

每个 alpha/lambda 候选新增：

- `pooled_inner_log_loss`
- `pooled_inner_brier_score`
- `mean_fold_log_loss`
- `sd_fold_log_loss`
- `mean_fold_brier_score`
- `sd_fold_brier_score`

### `06_inner_selected_oof_predictions.csv`

新增：

- `tuning_primary_metric`
- `candidate_selection_policy`

### `06_outer_validation_metrics.csv`

新增：

- `inner_selected_log_loss`
- `inner_selected_brier_score`
- `tuning_primary_metric`
- `candidate_selection_policy`

### `06_task_configuration.json`

记录本任务实际使用的主指标和完整候选排序策略。

### repeated-holdout 汇总

`08_selected_hyperparameters.csv` 和 `08_hyperparameter_frequency.csv` 会保留上述调参审计字段。

历史任务缺少这些字段时，汇总器自动填充：

```text
tuning_primary_metric = roc_auc
candidate_selection_policy = pooled_roc_pr_lambda_alpha
```

## 安装

在仓库外解压本升级包，然后运行：

```bash
/data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python3 \
  install_batch14_tuning_metric_selection.py \
  --repo-root /data/users/chenhaisheng/RA-ILD \
  --dry-run
```

dry-run 通过后正式安装：

```bash
/data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python3 \
  install_batch14_tuning_metric_selection.py \
  --repo-root /data/users/chenhaisheng/RA-ILD
```

## 验收

```bash
cd /data/users/chenhaisheng/RA-ILD

/data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python3 \
  TRB/set/scripts_v2/test_batch14_tuning_metric_selection.py
```

必须看到：

```text
TRB_BATCH14_ACCEPTANCE_PASS
```

建议再运行 TRB tests：

```bash
/data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python3 \
  -m unittest discover \
  -s TRB/set/tests \
  -p 'test_*.py' \
  -v
```

## Git 提交

只暂存 Batch 14 的以下文件：

```bash
git add -- \
  TRB/set/src/ra_ild_trb/config.py \
  TRB/set/src/ra_ild_trb/scheme_management.py \
  TRB/set/src/ra_ild_trb/specifications.py \
  TRB/set/src/ra_ild_trb/nested_cv.py \
  TRB/set/src/ra_ild_trb/repeated_holdout_summary.py \
  TRB/set/scripts_v2/run_nested_cv_task.py \
  TRB/set/scripts_v2/prepare_analysis_scheme.py \
  TRB/set/scripts_v2/BATCH14_TUNING_METRIC_SELECTION_MANIFEST.json \
  TRB/set/scripts_v2/README_upgrade_batch14_tuning_metric_selection.md \
  TRB/set/scripts_v2/test_batch14_tuning_metric_selection.py \
  TRB/set/tests/test_tuning_metric_selection.py
```

检查：

```bash
git diff --cached --check
git diff --cached --name-status
```

提交：

```bash
git commit -m "Add configurable ROC or log-loss tuning selection"
git push origin feature/trb-scheme-upgrade-batch01
```

不要使用 `git add .`，避免把实验输出和无关 YAML 一并加入提交。

## 后续创建 log-loss 训练方案

复制现有 scheme YAML，修改新的 scheme ID 和说明，并增加：

```yaml
overrides:
  model_selection:
    tuning_primary_metric: log_loss
```

然后正常执行：

```bash
python TRB/set/scripts_v2/prepare_analysis_scheme.py \
  --scheme <新方案YAML> \
  --dry-run
```

准备完成后，检查生成的：

```text
00_config/resolved_config.yaml
```

应明确包含：

```yaml
model_selection:
  tuning_primary_metric: log_loss
  candidate_sort:
  - field: pooled_inner_log_loss
    ascending: true
  - field: pooled_inner_brier_score
    ascending: true
  - field: pooled_inner_roc_auc
    ascending: false

nested_cv:
  candidate_selection_policy: pooled_log_loss_brier_roc_lambda_alpha
```
