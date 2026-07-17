# 03 Public CDR3-AA 正式构建脚本

## 已确认的默认路径

- 训练 metadata：`/data/users/chenhaisheng/RA-ILD/TRB/set/train/metadata_train_70.csv`
- 训练 01 AA 表：`/data/users/chenhaisheng/RA-ILD/TRB/set/train/result/01_AA_clone_table/`
- 稳定性配置：`/data/users/chenhaisheng/RA-ILD/TRB/set/train/result/03_public_features/threshold_stability/03_threshold_stability_configuration.json`
- 稳定性 cache：`/data/users/chenhaisheng/RA-ILD/TRB/set/train/result/03_public_features/threshold_stability/cache/`
- 测试 metadata：`/data/users/chenhaisheng/RA-ILD/TRB/set/test/metadata_test_30.csv`
- 测试 01 AA 表：`/data/users/chenhaisheng/RA-ILD/TRB/set/test/result/01_AA_clone_table/`

脚本从稳定性配置中读取已经确认的 `main` 阈值，不重新手动定义阈值。

## 训练模式输出

目录：`set/train/result/03_public_features/`

- `03_public_aa_catalog.csv.gz`：至少在2个训练样本出现过的候选CDR3-AA目录，共约100万行；包含出现人数、prevalence、频率统计及public/reference分类。
- `03_final_reference_public_sets.csv.gz`：主阈值下的最终全局public reference集合，供独立测试集应用。
- `03_descriptive_public_features.csv`：123个训练样本的描述性public特征，26列。
- `03_reference_set_summary.csv`：各reference set大小和有效阈值。
- `03_reference_definition.json`：阈值、集合大小、reference文件SHA256和来源记录。
- `03_public_feature_build_log.csv`：样本级构建日志。
- `03_public_feature_build_summary.md`：构建总结。

训练集的固定描述性public特征具有self-inclusion，只用于描述和QC，不能直接作为普通交叉验证的固定模型输入。后续模型交叉验证必须每折动态重建reference sets。

## 测试模式输出

目录：`set/test/result/03_public_features/`

- `03_descriptive_public_features.csv`：51个测试样本与训练reference集合的描述性重叠特征，26列。
- `03_test_reference_public_features.csv`：用于最终独立测试的reference-public特征，19列。
- `03_reference_definition_used.json`：记录使用的训练reference文件和SHA256校验结果。
- `03_public_feature_build_log.csv`
- `03_public_feature_build_summary.md`

## 测试模型reference特征列

- `sample_id`
- `all_ref_public_clone_number`
- `all_ref_public_clone_ratio`
- `all_ref_public_frequency_sum`
- `all_ref_public_set_coverage`
- `RA_specific_ref_clone_number`
- `RA_specific_ref_clone_ratio`
- `RA_specific_ref_frequency_sum`
- `RA_specific_ref_set_coverage`
- `ILD_specific_ref_clone_number`
- `ILD_specific_ref_clone_ratio`
- `ILD_specific_ref_frequency_sum`
- `ILD_specific_ref_set_coverage`
- `shared_ref_clone_number`
- `shared_ref_clone_ratio`
- `shared_ref_frequency_sum`
- `shared_ref_set_coverage`
- `ILD_RA_specific_ref_frequency_delta`
- `ILD_RA_specific_ref_frequency_log_ratio`

## 执行

将以下文件放入 `set/scripts/`：

- `public_feature_utils.py`
- `build_03_public_features.py`
- `run_03_build_public_features.sh`

完整执行：

```bash
cd /data/users/chenhaisheng/RA-ILD/TRB
bash set/scripts/run_03_build_public_features.sh
```

也可以先单独运行训练模式，检查结果后再运行测试模式。
