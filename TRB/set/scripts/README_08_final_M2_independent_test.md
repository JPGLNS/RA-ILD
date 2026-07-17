# Step 08：最终M2独立测试集一次性验证

## 已锁定设置

```text
Model     M2_static_tcr_public
alpha     0.5
lambda    30
threshold 0.509337
```

此步骤只应用已经拟合好的模型，不重新训练、不重新标准化、不选择特征、不调参、不调整阈值。

## 默认输入

```text
训练阶段保存的最终模型
set/train/result/07_final_M2_model/07_final_M2_model.joblib

训练阶段配置
set/train/result/07_final_M2_model/07_final_M2_configuration.json

完整训练reference masks
set/train/result/07_final_M2_model/07_full_training_public_reference_masks.npz

51例测试集最终特征矩阵
set/test/result/04_final_feature_matrix/04_test_final_feature_matrix.csv

测试集reference-public特征
set/test/result/03_public_features/03_test_reference_public_features.csv

测试集reference审计记录
set/test/result/03_public_features/03_reference_definition_used.json
```

## 脚本放置

将以下文件放入：

```text
/data/users/chenhaisheng/RA-ILD/TRB/set/scripts/
```

```text
run_08_final_M2_independent_test.py
run_08_final_M2_independent_test.sh
```

## 运行前语法检查

```bash
cd /data/users/chenhaisheng/RA-ILD/TRB

PYTHONPYCACHEPREFIX=/tmp/pycache_ra_ild \
python3 -m py_compile \
  set/scripts/run_08_final_M2_independent_test.py

bash -n set/scripts/run_08_final_M2_independent_test.sh
```

## 正式运行一次

该步骤耗时很短，不需要nohup：

```bash
cd /data/users/chenhaisheng/RA-ILD/TRB
bash set/scripts/run_08_final_M2_independent_test.sh
```

不要在正式运行命令中加入 `--overwrite`。脚本发现已有验证结果时会拒绝再次运行。

## 核心检查

脚本会确认：

- 测试样本正好51例；
- 训练集与测试集样本无重叠；
- 最终模型已收敛；
- 模型保存的alpha、lambda和阈值与锁定值完全一致；
- 测试集public特征来自固定训练reference；
- 构建测试reference时未使用测试标签；
- step-03 public特征与step-04合并矩阵数值一致；
- 测试数据只使用训练阶段保存的均值、标准差和类别编码；
- 测试集没有重新拟合、调参或阈值选择。

## 输出目录

```text
set/test/result/08_final_M2_validation/
```

主要文件：

```text
08_independent_test_predictions.csv
08_independent_test_metrics.csv
08_independent_test_bootstrap_CI.csv
08_independent_test_confusion_matrix.csv
08_independent_test_ROC_curve.csv
08_independent_test_PR_curve.csv
08_independent_test_evaluation.png
08_independent_test_evaluation.pdf
08_independent_test_configuration.json
08_independent_test_summary.md
08_VALIDATION_COMPLETE.json
```

其中bootstrap只用于估计固定模型性能的不确定性，不会改变模型或阈值。

## 运行后发送给ChatGPT

优先上传：

```text
08_independent_test_metrics.csv
08_independent_test_bootstrap_CI.csv
08_independent_test_predictions.csv
08_independent_test_summary.md
08_independent_test_evaluation.pdf
08_independent_test_configuration.json
```
