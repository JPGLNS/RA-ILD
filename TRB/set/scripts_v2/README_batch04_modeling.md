# TRB Framework V2 — Batch 04: Elastic Net, thresholds, and metrics

Batch 04 extracts the V1 model-fitting and evaluation behavior into three shared
modules. It does not replace or edit the existing Step 05, Step 07, or Step 08
workers.

## Added modules

```text
TRB/set/src/ra_ild_trb/
├── modeling.py
├── thresholds.py
└── metrics.py
```

### `modeling.py`

Provides the V1-compatible Elastic Net logistic backend:

- `penalty="elasticnet"`;
- `solver="saga"`;
- `C = 1 / lambda`;
- `class_weight="balanced"` or no class weighting;
- deterministic seed derivation;
- convergence-warning capture;
- probability prediction;
- V1-compatible intercept/coefficient tables;
- non-zero coefficient counting with an absolute tolerance.

### `thresholds.py`

Provides:

- V1 Youden-J selection with closest-to-0.5 tie breaking;
- `probability >= threshold` classification;
- optional maximum-F1 threshold selection;
- optional sensitivity-target threshold selection.

Only the Youden rule is used by the frozen baseline.

### `metrics.py`

Provides:

- ROC-AUC and PR-AUC;
- accuracy, sensitivity, specificity, precision, and F1;
- TN, FP, FN, and TP counts;
- optional Brier score and cohort-size summary;
- stratified percentile bootstrap intervals matching Step 08.

## New regression check

```text
TRB/set/scripts_v2/check_modeling_regression.py
```

The script reconstructs the frozen `repeat_01_fold_01` task for all four models.
For every model it checks:

1. the selected inner-OOF Youden threshold;
2. metric calculation from frozen V1 probabilities;
3. threshold application and predicted labels;
4. refitted validation probabilities;
5. convergence, iteration count, feature count, and non-zero count;
6. intercept and every coefficient.

It also recalculates all saved Step 08 independent-test metrics from the frozen
independent predictions. It does not overwrite any existing result.

## 1. Extract Batch 04

From the repository root:

```bash
unzip -o TRB_framework_v2_batch04_modeling.zip
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

Expected path count after Batches 01–04:

```text
Existing paths checked: 31
```

## 3. Run all unit tests

```bash
python3 -m unittest discover \
  -s TRB/set/tests \
  -p 'test_*.py' \
  -v
```

Expected result:

```text
Ran 59 tests
OK
```

## 4. Run the inexpensive modeling check first

This validates thresholds, metrics, saved predictions, and configuration without
refitting the four models:

```bash
python3 TRB/set/scripts_v2/check_modeling_regression.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --skip-refit
```

Expected result:

```text
Outer task=1/1 Checks=16 PASS=16 FAIL=0
```

## 5. Run the complete modeling regression

```bash
python3 TRB/set/scripts_v2/check_modeling_regression.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml
```

Expected result:

```text
Outer task=1/1 Checks=28 PASS=28 FAIL=0
```

The complete check refits M0–M3 once on the 98-sample outer-training partition.
The high-dimensional M1–M3 fits may take several minutes, depending on the
server load. No nested CV or independent-test refitting is performed.

Probability and coefficient differences near machine precision, such as
`1e-15`, are normal. The default comparison tolerances are:

```text
rtol = 1e-10
atol = 1e-12
```

## 6. Confirm earlier batches remain unchanged

```bash
python3 TRB/set/scripts_v2/check_preprocessing_regression.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml
```

Expected:

```text
Checks=18 PASS=18 FAIL=0
```

```bash
python3 TRB/set/scripts_v2/check_public_reference_regression.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml
```

Expected:

```text
Checks=15 PASS=15 FAIL=0
```

```bash
python3 TRB/set/scripts_v2/check_v1_baseline_outputs.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --scope full
```

Expected:

```text
Checks=36 PASS=36 FAIL=0
```

## Suggested commit

After all real-data checks pass:

```bash
git add \
  TRB/set/configs \
  TRB/set/src \
  TRB/set/scripts_v2 \
  TRB/set/tests

git diff --cached --stat
git commit -m "Add shared TRB modeling and metric modules"
git push
```

## Final integrated status

Batch 04 Hotfix 01 is incorporated into the released framework. The package
initializer imports only lightweight configuration and path modules; importing
`ra_ild_trb.config` therefore does not require scikit-learn. Modeling and metric
modules continue to import their scientific dependencies explicitly.

The separate `README_batch04_hotfix01.md` and
`BATCH04_HOTFIX01_MANIFEST.json` files are obsolete after Batch 08 and should be
removed from the repository. `BATCH04_MANIFEST.json` is now a finalized
batch-owned-file manifest and intentionally excludes shared files that continued
to change in later batches.
