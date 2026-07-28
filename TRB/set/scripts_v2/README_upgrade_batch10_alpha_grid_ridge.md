# Batch 10：YAML参数网格与Ridge/Lasso端点升级

## 目标

本升级不负责自动选择前一轮最优feature组合。使用者人工将两个入选组合写入方案YAML；系统负责：

1. 从feature-ablation方案YAML读取并覆盖`model_engine.alpha_grid`；
2. 可选覆盖`model_engine.lambda_grid`；
3. 支持`alpha=0.0`、`0<alpha<1`和`alpha=1.0`；
4. 自动重算candidate数量和每个outer task的inner fit数量；
5. 继续使用既有prepare、outer-task和aggregation脚本。

## alpha含义

- `alpha=0.0`：L2/Ridge端点；
- `0<alpha<1`：Elastic Net；
- `alpha=1.0`：L1/Lasso端点。

三者仍使用同一个scikit-learn逻辑回归后端：`solver=saga`、`penalty=elasticnet`、`l1_ratio=alpha`、`C=1/lambda`。

## 安装后测试

```bash
python TRB/set/scripts_v2/test_batch10_alpha_grid_ridge.py
```

预期：

```text
PASS: test_alpha_grid_and_endpoints
PASS: test_config_validation_accepts_zero
PASS: test_scheme_override_and_count_recalculation
Batch 10 focused tests: 3/3 passed
```

## 新方案YAML

模板：

```text
TRB/set/configs/schemes/trb_scheme_005_parameter_tuning_template.yaml
```

先修改：

- `scheme.id`；
- `scheme.output_root`，末级目录必须等于`scheme.id`；
- `models`中的两个feature组合；
- `model_engine.alpha_grid`；
- 必要时添加或修改`model_engine.lambda_grid`。

示例：

```yaml
model_engine:
  alpha_grid: [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
```

只覆盖alpha时，lambda继续继承基础resolved config。

## 运行

```bash
python TRB/set/scripts_v2/prepare_feature_ablation_scheme.py \
  --scheme TRB/set/configs/schemes/<你的参数方案>.yaml
```

随后使用生成的resolved config：

```bash
python TRB/set/scripts_v2/run_all_outer_tasks.py \
  --config TRB/set/experiments/<experiment_id>/00_config/resolved_config.yaml \
  --status-only

python TRB/set/scripts_v2/run_all_outer_tasks.py \
  --config TRB/set/experiments/<experiment_id>/00_config/resolved_config.yaml \
  --workers 2 \
  --execute

python TRB/set/scripts_v2/aggregate_repeated_holdout_results.py \
  --config TRB/set/experiments/<experiment_id>/00_config/resolved_config.yaml
```

## 计算量示例

两个feature组合、7个alpha、8个lambda、5个inner folds：

```text
2 × 7 × 8 × 5 = 560 inner fits / outer repeat
```

三个repeat合计1680次inner fit，另加每个repeat最终模型重拟合。

## 兼容性

- 原Batch 09 Scheme 004不含`model_engine`覆盖，行为保持不变；
- 原alpha网格`[0.1, 0.5, 0.9]`仍然有效；
- 新增能力是可选的，不会改变既有实验输出。
