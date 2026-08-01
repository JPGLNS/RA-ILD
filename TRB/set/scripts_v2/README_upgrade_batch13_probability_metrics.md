# TRB V2 Batch 13：Log-loss 与 Brier score

## 1. 升级目的

Batch 13 在不改变模型训练和超参数选择规则的前提下，为每个 outer holdout 增加两项概率预测指标：

- `log_loss`：越低越好；对过度自信的错误预测惩罚较强。
- `brier_score`：越低越好；衡量预测概率与真实 0/1 结局之间的均方误差。

原有指标继续保留：

- `roc_auc`
- `pr_auc`
- `accuracy`
- `sensitivity_recall`
- `specificity`
- `precision`
- `f1`

本次升级不会改变：

- alpha/lambda 网格；
- inner-CV 以 pooled ROC-AUC 为主、PR-AUC 为次的选择规则；
- Youden 阈值；
- 已拟合模型的系数；
- 已保存的 holdout probability。

## 2. 代码变化

### `metrics.py`

- 引入 `sklearn.metrics.log_loss`；
- `classification_metrics()` 新增 `include_log_loss`；
- bootstrap 默认同时支持 `log_loss` 和 `brier_score`。

### `nested_cv.py`

每个 outer holdout 的 `06_outer_validation_metrics.csv` 原生写入：

- `log_loss`
- `brier_score`

### `repeated_holdout_summary.py`

两项指标进入：

- `08_holdout_task_metrics.csv`
- `08_model_metric_summary.csv`
- `08_model_ranking.csv`
- `08_pairwise_model_metric_differences.csv`

模型的总描述性排序仍以 ROC-AUC → PR-AUC → F1 为主，不会因为本次升级改变原有模型排名规则。

配对比较新增方向字段：

- `metric_direction`
- `favorable_difference_sign`

对于 log-loss/Brier，`comparison - reference < 0` 才代表 comparison 更好，胜率按“更低”正确计算。

## 3. 已完成模型无需重新训练

现有 PBMC M0–M2 和 Feature14 已保存每个 split、每个模型、每名 holdout 患者的：

- 真实标签；
- `probability_ILD`；
- threshold；
- predicted label。

Log-loss 和 Brier 都只依赖真实标签与预测概率。因此，从这些冻结 prediction 回填得到的数值，与重新执行全部模型拟合后计算的数值完全相同。

`backfill_probability_metrics.py` 会：

1. 重新计算全部旧指标；
2. 核对旧指标与 prediction 一致；
3. 备份原始 metrics CSV；
4. 原子写入 `log_loss` 和 `brier_score`；
5. 不修改模型、prediction、系数或任务完成标记。

## 4. 安装后验收

```bash
/data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python3 \
  TRB/set/scripts_v2/test_batch13_probability_metrics.py
```

必须以以下标记结束：

```text
TRB_BATCH13_ACCEPTANCE_PASS
```

## 5. 回填现有 PBMC M0–M2

先 dry-run：

```bash
/data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python3 \
  TRB/set/scripts_v2/backfill_probability_metrics.py \
  --config \
  TRB/set/experiments/trb_scheme_pbmc_m0_m2_repeat100_fullgrid_v1/00_config/resolved_config.yaml \
  --dry-run
```

正式回填：

```bash
/data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python3 \
  TRB/set/scripts_v2/backfill_probability_metrics.py \
  --config \
  TRB/set/experiments/trb_scheme_pbmc_m0_m2_repeat100_fullgrid_v1/00_config/resolved_config.yaml
```

重新汇总：

```bash
/data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python3 \
  TRB/set/scripts_v2/aggregate_repeated_holdout_results.py \
  --config \
  TRB/set/experiments/trb_scheme_pbmc_m0_m2_repeat100_fullgrid_v1/00_config/resolved_config.yaml \
  --overwrite
```

## 6. 回填现有 PBMC Feature14

先 dry-run：

```bash
/data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python3 \
  TRB/set/scripts_v2/backfill_probability_metrics.py \
  --config \
  TRB/set/experiments/trb_scheme_pbmc_feature14_repeat100_fullgrid_v1/00_config/resolved_config.yaml \
  --dry-run
```

正式回填：

```bash
/data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python3 \
  TRB/set/scripts_v2/backfill_probability_metrics.py \
  --config \
  TRB/set/experiments/trb_scheme_pbmc_feature14_repeat100_fullgrid_v1/00_config/resolved_config.yaml
```

重新汇总：

```bash
/data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python3 \
  TRB/set/scripts_v2/aggregate_repeated_holdout_results.py \
  --config \
  TRB/set/experiments/trb_scheme_pbmc_feature14_repeat100_fullgrid_v1/00_config/resolved_config.yaml \
  --overwrite
```

## 7. 结果检查

M0–M2：

```bash
/data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python3 - <<'PY'
import pandas as pd
path = (
    "TRB/set/experiments/"
    "trb_scheme_pbmc_m0_m2_repeat100_fullgrid_v1/"
    "03_summary/08_model_ranking.csv"
)
df = pd.read_csv(path)
print(df[[
    "descriptive_rank", "model", "roc_auc", "pr_auc",
    "log_loss", "brier_score", "accuracy",
    "sensitivity_recall", "specificity", "precision", "f1",
    "rank_mean_log_loss", "rank_mean_brier_score",
]].round(4).to_string(index=False))
PY
```

Feature14：

```bash
/data/users/chenhaisheng/anaconda3/envs/ra-ild/bin/python3 - <<'PY'
import pandas as pd
path = (
    "TRB/set/experiments/"
    "trb_scheme_pbmc_feature14_repeat100_fullgrid_v1/"
    "03_summary/08_model_ranking.csv"
)
df = pd.read_csv(path)
print(df[[
    "descriptive_rank", "model", "roc_auc", "pr_auc",
    "log_loss", "brier_score", "accuracy",
    "sensitivity_recall", "specificity", "precision", "f1",
    "rank_mean_log_loss", "rank_mean_brier_score",
]].round(4).to_string(index=False))
PY
```

## 8. 运行结果与 Git

以下内容是运行产物，不提交：

- `TRB/set/experiments/`
- `TRB/set/training_bundles/`
- `TRB/set/split_sets/`
- 每个任务中的 `06_outer_validation_metrics.before_batch13.csv`
- `09_probability_metric_backfill_audit.csv`
- `03_summary/` 下重新生成的汇总文件

Git 仅提交 Batch 13 的源代码、测试、文档和 manifest。
