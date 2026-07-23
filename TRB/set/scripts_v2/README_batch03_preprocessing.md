# TRB Framework V2 — Batch 03: Fold-local preprocessing

This batch introduces a shared preprocessing component while leaving the V1
Step 05, Step 07, and Step 08 execution scripts unchanged.

## Added component

```text
TRB/set/src/ra_ild_trb/preprocessing.py
```

The module reproduces the V1 `build_design()` behavior:

1. numeric columns first, in the supplied order;
2. categorical variables next, in the supplied order;
3. categorical levels learned from the current training partition only;
4. lexicographically first training level used as the reference;
5. dummy names formatted as `column__category_vs_reference`;
6. population SD (`ddof=0`);
7. zero-variance filtering learned from training rows only;
8. training-derived mean/SD applied to validation or test rows;
9. unseen transform categories retain the V1 all-zero raw dummy code and are
   recorded in an audit count.

## New files

```text
TRB/set/src/ra_ild_trb/preprocessing.py
TRB/set/scripts_v2/check_preprocessing_regression.py
TRB/set/tests/test_preprocessing.py
TRB/set/scripts_v2/README_batch03_preprocessing.md
TRB/set/scripts_v2/BATCH03_MANIFEST.json
```

The package also updates:

```text
TRB/set/configs/trb_baseline_m2_v1.yaml
TRB/set/src/ra_ild_trb/__init__.py
TRB/set/src/ra_ild_trb/config.py
TRB/set/scripts_v2/validate_experiment_config.py
TRB/set/tests/test_config.py
```

## 1. Extract Batch 03 from the repository root

```bash
unzip -o TRB_framework_v2_batch03_preprocessing.zip
```

## 2. Validate configuration and paths

```bash
python3 TRB/set/scripts_v2/validate_experiment_config.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml
```

```bash
python3 TRB/set/scripts_v2/validate_experiment_config.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --check-paths
```

Batch 02 checked 19 paths. Batch 03 checks six additional frozen V1 files, so the
expected count is:

```text
Existing paths checked: 25
```

## 3. Run all unit tests

```bash
python3 -m unittest discover \
  -s TRB/set/tests \
  -p 'test_*.py' \
  -v
```

Expected result after extracting Batch 03 over Batches 01–02:

```text
Ran 34 tests
OK
```

## 4. Run the real-data preprocessing regression check

```bash
python3 TRB/set/scripts_v2/check_preprocessing_regression.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml
```

The checker reconstructs the outer-final preprocessing for
`repeat_01_fold_01` and verifies all four models:

```text
M0_clinical
M1_static_tcr
M2_static_tcr_public
M3_static_tcr_public_material
```

For each model it checks:

- feature order before filtering;
- source type (`numeric` or `categorical_dummy`);
- training mean;
- population training SD;
- zero-variance keep/drop flag;
- retained feature order against `05_final_model_coefficients.csv`;
- standardized training mean approximately zero;
- standardized training SD approximately one;
- finite validation values and unseen-category audit counts.

Expected summary:

```text
Outer task=1/1 Checks=18 PASS=18 FAIL=0
```

Small differences around `1e-15` are normal floating-point round-off. The
preprocessing audit and retained feature order must otherwise agree exactly.

## 5. Re-run earlier regression checks

```bash
python3 TRB/set/scripts_v2/check_public_reference_regression.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml
```

```bash
python3 TRB/set/scripts_v2/check_v1_baseline_outputs.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --scope full
```

Expected summaries remain:

```text
Checks=15 PASS=15 FAIL=0
Checks=36 PASS=36 FAIL=0
```

## 6. Suggested commit

```bash
git add \
  TRB/set/configs \
  TRB/set/src \
  TRB/set/scripts_v2 \
  TRB/set/tests

git diff --cached --stat
git commit -m "Add shared TRB preprocessing module"
git push
```
