# TRB Framework V2 — Batch 06: Nested cross-validation engine

## Purpose

Batch 06 connects the independently regression-tested V2 modules into one
leakage-controlled nested-CV outer-task engine. It preserves the frozen V1
execution rules while separating the reusable in-memory engine from file I/O.

The independent test set is never read by this engine.

## New files

```text
TRB/set/src/ra_ild_trb/nested_cv.py
TRB/set/scripts_v2/run_nested_cv_task.py
TRB/set/scripts_v2/check_nested_cv_regression.py
TRB/set/scripts_v2/README_batch06_nested_cv.md
TRB/set/scripts_v2/BATCH06_MANIFEST.json
TRB/set/tests/test_nested_cv.py
```

## Updated files

```text
TRB/set/configs/trb_baseline_m2_v1.yaml
TRB/set/src/ra_ild_trb/__init__.py
TRB/set/src/ra_ild_trb/config.py
TRB/set/scripts_v2/validate_experiment_config.py
TRB/set/tests/test_config.py
```

The package initializer remains lightweight and does not eagerly import
scikit-learn.

## Frozen V1 orchestration reproduced

For one outer repeat/fold:

1. use the precomputed outer split;
2. require the inner assignment IDs to equal the outer-training IDs exactly;
3. exclude every outer-validation sample from all inner operations;
4. for each inner fold:
   - generate exact LOO public features for inner-training samples;
   - build the public reference from inner-training samples only and apply it to
     inner-validation samples;
   - fit categorical encoding, zero-variance filtering, means, and SDs on the
     inner-training partition only;
5. fit every alpha/lambda candidate using the V1 deterministic seed formula;
6. pool inner OOF predictions and rank candidates by ROC-AUC, PR-AUC, lambda,
   then alpha, all descending;
7. choose the Youden threshold from the selected inner OOF probabilities;
8. regenerate outer-training LOO and outer-validation public features;
9. refit each selected model on the complete outer-training partition;
10. evaluate the untouched outer-validation partition.

For the baseline configuration:

```text
4 models × 24 candidates × 5 inner folds = 480 inner model fits per outer task
20 repeats × 5 outer folds = 100 outer tasks
```

## Safe V2 runner

The runner writes only under `result_v2` by default and never replaces V1
outputs:

```bash
python3 TRB/set/scripts_v2/run_nested_cv_task.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --outer-repeat 1 \
  --outer-fold 1
```

Default output:

```text
TRB/set/train/result_v2/06_nested_cv/repeat_01_fold_01/
```

Do not use `--overwrite` unless a V2 technical rerun is intentional.

## Server validation

### 1. Configuration and paths

```bash
python3 TRB/set/scripts_v2/validate_experiment_config.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml

python3 TRB/set/scripts_v2/validate_experiment_config.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --check-paths
```

Expected path count:

```text
Existing paths checked: 36
```

### 2. Unit tests

```bash
python3 -m unittest discover \
  -s TRB/set/tests \
  -p 'test_*.py' \
  -v
```

Expected result:

```text
Ran 94 tests
OK
```

### 3. Quick real-data orchestration regression

This reconstructs the selected parameters from frozen V1 inner outputs and runs
only the four outer-final model fits:

```bash
python3 TRB/set/scripts_v2/check_nested_cv_regression.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml
```

Expected summary:

```text
Mode=quick Outer task=1/1 Checks=32 PASS=32 FAIL=0
```

### 4. Full nested-CV regression

This reruns all 480 inner fits for `repeat_01_fold_01`, followed by the four
outer-final fits. It may take substantially longer than previous regression
checks:

```bash
python3 TRB/set/scripts_v2/check_nested_cv_regression.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --full
```

Expected summary:

```text
Mode=full Outer task=1/1 Checks=43 PASS=43 FAIL=0
```

The full check compares all 96 tuning rows, selected inner OOF probabilities,
inner and outer preprocessing audits, public-reference/LOO audit counts, outer
probabilities, metrics, and coefficients.

### 5. Previous regression layers

```bash
python3 TRB/set/scripts_v2/check_specification_regression.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml

python3 TRB/set/scripts_v2/check_modeling_regression.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml

python3 TRB/set/scripts_v2/check_preprocessing_regression.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml

python3 TRB/set/scripts_v2/check_public_reference_regression.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml

python3 TRB/set/scripts_v2/check_v1_baseline_outputs.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --scope full
```

Expected summaries remain:

```text
Specifications:      19 PASS, 0 FAIL
Modeling:            28 PASS, 0 FAIL
Preprocessing:       18 PASS, 0 FAIL
Public reference:    15 PASS, 0 FAIL
V1 baseline:         36 PASS, 0 FAIL
```
