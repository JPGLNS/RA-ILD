# TRB V2 Batch 06 Hotfix 01

## Problem

The full nested-CV regression checker attempted to compare V2 inner-fold
preprocessing audit rows with the frozen V1 file
`05_preprocessing_summary.csv`.

The V1 worker computes inner-fold preprocessing rows in memory, but its output
code writes only the preprocessing table returned by `fit_outer_models()`.
Consequently, the frozen V1 CSV contains only `stage=outer_final` rows and no
`stage=inner` rows. The original Batch 06 checker therefore reported:

```text
Full inner preprocessing audit equality:
key/order mismatch V2=16370, V1=0
```

This is a checker false negative, not a model or nested-CV mismatch.

## Fix

The replacement checker:

1. confirms that the frozen V1 preprocessing file contains zero inner rows;
2. validates the V2 inner preprocessing audit structurally;
3. requires all four models and all five inner folds;
4. checks the expected 16,370 rows;
5. checks unique `(model, inner_fold, feature_name)` keys;
6. checks per-model/per-fold row counts;
7. checks finite training means/SDs and valid source types.

All other V1 equality checks remain unchanged.

## Apply

```bash
unzip -o TRB_framework_v2_batch06_hotfix01_inner_preprocessing_audit.zip
```

Then run:

```bash
python3 -m unittest discover -s TRB/set/tests -p 'test_*.py' -v

python3 TRB/set/scripts_v2/check_nested_cv_regression.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml
```

The quick check remains expected to report:

```text
Mode=quick Outer task=1/1 Checks=32 PASS=32 FAIL=0
```

A repeated full run should report:

```text
Mode=full Outer task=1/1 Checks=43 PASS=43 FAIL=0
```

with the revised check named:

```text
Full inner preprocessing audit completeness
```
