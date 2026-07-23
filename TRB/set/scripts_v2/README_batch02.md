# TRB Framework V2 — Batch 02

Batch 02 adds a shared, leakage-controlled public-reference module and regression
checks against the frozen V1 outputs. It does **not** modify the current V1
Step 03, Step 05, Step 07, or Step 08 execution scripts.

## Main additions

```text
TRB/set/src/ra_ild_trb/public_reference.py
TRB/set/scripts_v2/check_public_reference_regression.py
TRB/set/tests/test_public_reference.py
```

The baseline YAML is extended with the existing sparse-cache paths and one frozen
outer task used for regression testing.

## Install

Upload the ZIP to the repository root and overwrite the Batch 01 files that are
intentionally updated:

```bash
cd /data/users/chenhaisheng/RA-ILD
unzip -o TRB_framework_v2_batch02.zip
```

No new package is required beyond the current environment dependencies:

```text
numpy
pandas
scipy
PyYAML
```

## 1. Validate the updated configuration

```bash
python3 TRB/set/scripts_v2/validate_experiment_config.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml
```

Also verify all configured paths:

```bash
python3 TRB/set/scripts_v2/validate_experiment_config.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --check-paths
```

The number of checked existing paths increases from 15 to 19 because Batch 02
adds three sparse-cache files and one frozen Step 05 task directory.

## 2. Run all unit tests

```bash
python3 -m unittest discover \
  -s TRB/set/tests \
  -p 'test_*.py' \
  -v
```

Expected total after Batch 02:

```text
Ran 18 tests
OK
```

The new tests include:

- effective count thresholds;
- full public masks;
- count-based versus presence-based construction;
- all 18 public features;
- optimized exact LOO versus a brute-force leave-one-out calculation;
- disjoint external reference/target enforcement;
- public-feature-set ordering.

## 3. Check full-training masks first

This reads the existing sparse cache and locked Step 07 masks. It does not write
or overwrite analysis results.

```bash
python3 TRB/set/scripts_v2/check_public_reference_regression.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --skip-task-features
```

Expected checks:

- full reference sizes equal the frozen values;
- every element of the four V2 masks equals the Step 07 saved mask;
- effective thresholds and cohort reference counts are reported.

## 4. Run the complete public-reference regression check

```bash
python3 TRB/set/scripts_v2/check_public_reference_regression.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml
```

The default frozen task is:

```text
outer repeat 1, fold 1
```

The script reconstructs:

- outer-training exact LOO features;
- outer-validation external-reference features;
- all 18 public feature columns.

It then compares them numerically with:

```text
repeat_01_fold_01/05_dynamic_public_features_outer_train_loo.csv
repeat_01_fold_01/05_dynamic_public_features_outer_validation.csv
```

The default numerical tolerances are:

```text
rtol = 1e-10
atol = 1e-12
```

This calculation loads the 123 × 1,000,139 sparse cache and rebuilds LOO
references. It may take some time and memory, but it does not rerun model fitting.

Optional JSON report:

```bash
python3 TRB/set/scripts_v2/check_public_reference_regression.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --json-output TRB/set/train/result/v2_regression_checks/batch02_public_reference.json
```

The report path remains ignored under the current whitelist unless explicitly
added later.

## 5. Review and commit

```bash
git status --short
git diff --stat
git diff -- TRB/set/configs/trb_baseline_m2_v1.yaml
git diff -- TRB/set/src/ra_ild_trb/public_reference.py
```

After every check passes:

```bash
git add \
  TRB/set/configs/trb_baseline_m2_v1.yaml \
  TRB/set/src/ra_ild_trb \
  TRB/set/scripts_v2 \
  TRB/set/tests

git diff --cached --stat
git commit -m "Add shared TRB public reference module"
git push
```
