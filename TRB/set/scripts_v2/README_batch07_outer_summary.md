# TRB Framework V2 — Batch 07: Outer-task orchestration and stability analysis

## Purpose

Batch 07 extends the validated single-task nested-CV engine to the complete
20-repeat × 5-fold design. It adds three auditable layers:

1. a deterministic 100-task manifest and resumable task runner;
2. structural validation and status tracking for every task directory;
3. aggregation of repeated outer-validation performance, hyperparameter
   selection, coefficients, and sample-level predictions.

The independent test set is never read by this batch. All outputs are written
under `TRB/set/train/result_v2/` and do not modify the frozen V1 tree.

## New files

```text
TRB/set/src/ra_ild_trb/outer_cv.py
TRB/set/scripts_v2/run_all_outer_tasks.py
TRB/set/scripts_v2/aggregate_outer_results.py
TRB/set/scripts_v2/check_outer_orchestration.py
TRB/set/scripts_v2/README_batch07_outer_summary.md
TRB/set/scripts_v2/BATCH07_MANIFEST.json
TRB/set/tests/test_outer_cv.py
```

## Updated files

```text
TRB/set/configs/trb_baseline_m2_v1.yaml
TRB/set/src/ra_ild_trb/__init__.py
TRB/set/src/ra_ild_trb/config.py
TRB/set/scripts_v2/validate_experiment_config.py
TRB/set/tests/test_config.py
```

## Task manifest and safety rules

The manifest contains exactly 100 tasks in this order:

```text
repeat_01_fold_01
repeat_01_fold_02
...
repeat_20_fold_05
```

The task manager applies the following rules:

- a structurally valid task with `06_TASK_COMPLETE.json` is skipped;
- a missing task can be executed;
- an incomplete or invalid task is blocked by default;
- incomplete/invalid tasks require the explicit `--rerun-incomplete` option;
- the default worker count is 1;
- the configured maximum is 4;
- every subprocess writes an independent scheduler log;
- `--limit N` selects the first N pending tasks, not already complete tasks.

Each parallel worker loads the large sparse public-reference cache. Keep
`--workers 1` unless server memory has been checked carefully.

## Structural task validation

A task is marked complete only when all expected files exist and pass checks for:

- repeat/fold identity in configuration and completion marker;
- 123 unique sample-role rows;
- outer train/validation counts;
- exact M0–M3 model order;
- 96 inner tuning rows;
- selected inner OOF dimensions;
- outer prediction dimensions and uniqueness;
- four outer metric rows;
- coefficient model/feature uniqueness;
- the frozen 1,083-feature static list.

## Aggregated outputs

The default output directory is:

```text
TRB/set/train/result_v2/07_outer_summary/
```

Main files:

```text
07_outer_task_manifest.csv
07_outer_task_status.csv
07_all_outer_metrics.csv
07_all_outer_predictions.csv.gz
07_all_selected_hyperparameters.csv
07_all_coefficients.csv.gz
07_outer_task_metric_summary.csv
07_repeat_pooled_metrics.csv
07_hyperparameter_selection_frequency.csv
07_feature_stability.csv.gz
07_stable_features.csv
07_intercept_stability.csv
07_sample_prediction_stability.csv
07_pairwise_model_metric_differences.csv
07_aggregation_summary.json
07_AGGREGATION_COMPLETE.json
```

### Performance summaries

`07_outer_task_metric_summary.csv` summarizes the 100 task-level metrics for
M0–M3 using mean, SD, median, 2.5th/25th/75th/97.5th percentiles, minimum, and
maximum.

`07_repeat_pooled_metrics.csv` pools the five disjoint outer folds within each
repeat. It therefore contains:

```text
4 models × 20 repeats = 80 rows
```

Each repeat contains one out-of-fold prediction per training sample.

### Hyperparameter stability

`07_hyperparameter_selection_frequency.csv` reports, for each model and
alpha/lambda pair:

- selection count and frequency across outer tasks;
- mean selected inner ROC-AUC;
- mean selected inner PR-AUC.

### Feature stability

A coefficient is selected when:

```text
abs(coefficient) > 1e-12
```

For each model/feature, `07_feature_stability.csv.gz` reports:

- task availability;
- selection count and frequency across all completed model tasks;
- positive and negative selection counts;
- dominant sign and sign consistency;
- mean coefficient across all tasks, treating unavailable/dropped features as
  zero in the all-task denominator;
- selected-only coefficient summaries;
- whether the configured stability criteria are met.

The locked criteria are:

```text
selection frequency >= 0.50
sign consistency   >= 0.80
```

Stable non-intercept features are also written to `07_stable_features.csv`.

### Sample prediction stability

`07_sample_prediction_stability.csv` summarizes the 20 repeated out-of-fold
predictions for every model/sample combination, including probability mean/SD,
range, positive-classification frequency, and correct-classification frequency.

## Server validation before running all tasks

### 1. Configuration

```bash
python3 TRB/set/scripts_v2/validate_experiment_config.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml

python3 TRB/set/scripts_v2/validate_experiment_config.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --check-paths
```

Expected path count:

```text
Existing paths checked: 37
```

### 2. Unit tests

```bash
python3 -m unittest discover \
  -s TRB/set/tests \
  -p 'test_*.py' \
  -v
```

Expected:

```text
Ran 112 tests
OK
```

### 3. Current task tree

```bash
python3 TRB/set/scripts_v2/check_outer_orchestration.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml
```

After Batch 06 task 1/1 has been retained, the expected summary is:

```text
Checks=12 PASS=12 FAIL=0
complete=1, missing=99, incomplete=0, invalid=0
```

### 4. Generate manifest/status without execution

```bash
python3 TRB/set/scripts_v2/run_all_outer_tasks.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --status-only
```

Running the same command without `--execute` is also non-destructive.

## Controlled execution

### One pending task first

```bash
python3 TRB/set/scripts_v2/run_all_outer_tasks.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --execute \
  --limit 1 \
  --workers 1
```

Then inspect status again. The first already-complete task is skipped when the
pending-task limit is applied.

### All remaining tasks

```bash
python3 TRB/set/scripts_v2/run_all_outer_tasks.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --execute \
  --workers 1
```

The command is resumable. Re-running it skips all structurally complete tasks.
Do not use `--rerun-incomplete` until the relevant task log has been reviewed.

## Validation after all 100 tasks

```bash
python3 TRB/set/scripts_v2/check_outer_orchestration.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --require-all
```

Expected:

```text
Checks=13 PASS=13 FAIL=0
complete=100, missing=0, incomplete=0, invalid=0
```

## Final aggregation

```bash
python3 TRB/set/scripts_v2/aggregate_outer_results.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml
```

Expected fixed dimensions include:

```text
outer metric rows:       100 tasks × 4 models = 400
outer prediction rows:   123 samples × 20 repeats × 4 models = 9,840
selected parameter rows: 100 tasks × 4 models = 400
repeat metric rows:      20 repeats × 4 models = 80
```

`--allow-incomplete` is available only for a provisional preview. Without an
explicit `--output-dir`, partial results are written to
`TRB/set/train/result_v2/07_outer_summary_partial/`, so they cannot block the
final complete aggregation. A partial aggregation does not calculate
repeat-pooled metrics and must not be treated as the final repeated nested-CV
estimate.
