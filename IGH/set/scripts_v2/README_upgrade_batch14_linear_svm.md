# IGH Upgrade Batch 14 — Configurable Linear SVM Engine

## 1. Purpose

Batch 14 adds an **additive Linear SVM analysis path** to the current IGH V2
framework. It does not remove, replace, or reinterpret the historical Elastic
Net logistic-regression path.

The new path reuses the existing validated IGH layers:

- M0–M3, Feature14 and arbitrary YAML model definitions;
- full cohort, PBMC-only and buffercoat-only inputs;
- frozen repeated-holdout assignments and configurable repeat counts;
- repeat-specific 3-mer feature groups;
- training-partition-only public-reference construction;
- exact leave-one-out public features for fitting samples;
- fold-local encoding, zero-variance filtering and z-score scaling;
- configurable worker count and isolated task directories.

## 2. Estimator and score contract

Estimator:

```text
sklearn.svm.LinearSVC
```

Continuous evaluation value:

```text
decision_function score for class ILD
```

The score is used for:

- pooled inner-OOF ROC-AUC tuning;
- pooled inner-OOF Average Precision as the secondary ranking metric;
- inner-OOF Youden threshold selection;
- outer ROC-AUC and Average Precision;
- ROC and PR-compatible downstream plots.

The uncalibrated SVM does **not** output probabilities. Therefore:

- log loss is not calculated;
- Brier score is not calculated;
- prediction figures are labelled as decision scores, not probabilities.

## 3. Candidate blocks

The framework supports conditional candidate blocks from the first version.
Each block can configure:

```yaml
penalty: l1 | l2
loss: hinge | squared_hinge
class_weight_grid: [balanced, none]
C_grid: [...]
dual: auto | true | false
priority: 1
```

Illegal combinations are rejected before training. In particular:

```text
penalty=l1 + loss=hinge
```

is not supported by `LinearSVC`.

The supplied main example fixes:

```text
penalty = l2
loss = squared_hinge
class_weight = balanced
dual = auto
```

and tunes only:

```text
C = 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000
```

## 4. Candidate ranking and convergence

For each model, candidates are ranked by:

1. pooled inner-OOF ROC-AUC, descending;
2. pooled inner-OOF Average Precision, descending;
3. C, ascending;
4. candidate-block priority, ascending;
5. candidate ID, ascending.

A candidate is eligible only when all its inner-fold fits converge. If all
candidates for one model are ineligible, the outer task fails instead of
silently selecting a non-converged model.

## 5. Thresholds and outer metrics

Primary operating point:

```text
threshold_source = inner_oof_youden
```

Tie-breaking:

```text
maximum Youden J
→ threshold closest to 0
→ larger threshold
```

The natural Linear SVM threshold `0` is retained as a secondary operating point.

Main outer metrics:

- ROC-AUC;
- Average Precision (`pr_auc` alias retained);
- Accuracy;
- Sensitivity/Recall;
- Specificity;
- Precision;
- F1;
- TN, FP, FN and TP.

## 6. Installed components

### Source modules

```text
IGH/set/src/ra_ild_igh/linear_svm.py
IGH/set/src/ra_ild_igh/linear_svm_config.py
IGH/set/src/ra_ild_igh/linear_svm_nested_cv.py
IGH/set/src/ra_ild_igh/linear_svm_summary.py
IGH/set/src/ra_ild_igh/linear_svm_visualization.py
```

### Commands

```text
IGH/set/scripts_v2/prepare_linear_svm_scheme.py
IGH/set/scripts_v2/run_linear_svm_nested_cv_task.py
IGH/set/scripts_v2/run_all_linear_svm_tasks.py
IGH/set/scripts_v2/aggregate_linear_svm_results.py
IGH/set/scripts_v2/plot_linear_svm_results.py
```

### Tests and examples

```text
IGH/set/tests/test_linear_svm.py
IGH/set/tests/test_linear_svm_config.py
IGH/set/scripts_v2/test_batch15_linear_svm.py
IGH/set/configs/linear_svm/igh_linear_svm_main_example.yaml
IGH/set/configs/linear_svm/igh_linear_svm_multi_block_example.yaml
IGH/set/configs/visualization/igh_linear_svm_visualization_example.yaml
```

## 7. Workflow

Prepare a scheme:

```bash
python3 IGH/set/scripts_v2/prepare_linear_svm_scheme.py \
  --scheme IGH/set/configs/linear_svm/igh_linear_svm_main_example.yaml \
  --repository-root /data/users/chenhaisheng/RA-ILD \
  --dry-run
```

After confirming the inherited base config and counts, run without `--dry-run`.

Inspect task status:

```bash
python3 IGH/set/scripts_v2/run_all_linear_svm_tasks.py \
  --config IGH/set/experiments/<scheme_id>/00_config/resolved_config.yaml \
  --repository-root /data/users/chenhaisheng/RA-ILD \
  --status-only
```

Execute with four outer-task workers:

```bash
python3 IGH/set/scripts_v2/run_all_linear_svm_tasks.py \
  --config IGH/set/experiments/<scheme_id>/00_config/resolved_config.yaml \
  --repository-root /data/users/chenhaisheng/RA-ILD \
  --workers 4 \
  --execute
```

Aggregate:

```bash
python3 IGH/set/scripts_v2/aggregate_linear_svm_results.py \
  --config IGH/set/experiments/<scheme_id>/00_config/resolved_config.yaml \
  --repository-root /data/users/chenhaisheng/RA-ILD
```

Visualize after copying and editing the example visualization YAML:

```bash
python3 IGH/set/scripts_v2/plot_linear_svm_results.py \
  --config IGH/set/configs/visualization/<your_svm_visualization>.yaml
```

## 8. Output compatibility

Each outer task uses the established `06_*` structure, but score fields are
engine-neutral:

```text
prediction_score
decision_score_ILD
score_type = decision_function
```

Repeated-holdout aggregation writes:

```text
03_summary/
├── 08_holdout_task_metrics.csv
├── 08_holdout_predictions.csv.gz
├── 08_final_model_coefficients.csv.gz
├── 08_metric_summary.csv
├── 08_selected_parameter_frequency.csv
├── 08_coefficient_stability_summary.csv
├── 08_aggregation_summary.json
└── 08_REPEATED_HOLDOUT_AGGREGATION_COMPLETE.json
```

Visualization writes four vector PDFs:

1. model-performance boxplots;
2. mean repeated-holdout ROC curves;
3. best observed repeat ROC curves;
4. best-repeat RA/ILD decision-score boxplots with Youden and zero thresholds.

## 9. Backward compatibility

Batch 14 is additive:

- no historical Elastic Net source file is overwritten;
- no frozen split or training bundle is modified;
- no existing experiment result is read or overwritten;
- existing Elastic Net commands and tests remain available;
- Linear SVM outputs use separate scheme IDs and directories.
