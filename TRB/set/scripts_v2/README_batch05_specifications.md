# TRB Framework V2 — Batch 05: Model specifications and candidate management

## Purpose

Batch 05 extracts the non-fitting modeling decisions that were embedded in the
V1 worker scripts. It adds a shared, auditable definition of:

1. static TRB feature selection from the step-04 feature manifest;
2. M0–M3 model specifications and concrete predictor-column expansion;
3. the Elastic Net alpha/lambda candidate grid;
4. the V1 inner-CV candidate ranking and tie-break rule;
5. final-model candidate-pair and locked-pair consistency.

This batch does **not** run the full nested cross-validation engine and does not
change any existing V1 result.

## New files

```text
TRB/set/src/ra_ild_trb/specifications.py
TRB/set/scripts_v2/check_specification_regression.py
TRB/set/scripts_v2/README_batch05_specifications.md
TRB/set/scripts_v2/BATCH05_MANIFEST.json
TRB/set/tests/test_specifications.py
```

## Updated files

```text
TRB/set/configs/trb_baseline_m2_v1.yaml
TRB/set/src/ra_ild_trb/__init__.py
TRB/set/src/ra_ild_trb/config.py
TRB/set/scripts_v2/validate_experiment_config.py
TRB/set/tests/test_config.py
```

The package initializer remains lightweight. Importing configuration utilities
does not require scikit-learn.

## Frozen V1 rules reproduced

### Static feature rule

A feature is selected when:

```text
present_in_train_base = true
feature_group starts with "tcr_"
feature_role = "candidate_predictor"
reference_drop = false
```

Manifest order is retained, and the selected feature must exist in the training
base matrix. The frozen baseline contains 1,083 static TRB predictors.

### Model expansion

```text
M0: age + sex
M1: age + 1,083 static TRB features + sex
M2: M1 + 8 dynamic public features
M3: M2 + material
```

Expected unresolved column counts are stored in the YAML and checked against the
shared specification resolver.

### Candidate grid

```text
alpha:  0.1, 0.5, 0.9
lambda: 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30
```

Candidates are generated in alpha-then-lambda order, giving 24 candidates per
model and 96 rows for four models.

### Inner-CV candidate ranking

The V1 ranking order is reproduced exactly:

```text
pooled_inner_roc_auc  descending
pooled_inner_pr_auc   descending
lambda                descending
l1_ratio_alpha        descending
```

The larger lambda tie-break favors stronger regularization when the two pooled
performance metrics are equal.

## Server validation

From the repository root:

```bash
unzip -o TRB_framework_v2_batch05_specifications.zip
```

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
Existing paths checked: 32
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
Ran 80 tests
OK
```

### 3. Real-data specification regression

```bash
python3 TRB/set/scripts_v2/check_specification_regression.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml
```

Expected summary:

```text
Checks=19 PASS=19 FAIL=0
```

The checker compares the shared V2 rules with:

```text
04_feature_manifest.csv
04_train_base_feature_matrix.csv
05_static_tcr_feature_list.csv
05_inner_tuning_results.csv
05_inner_selected_oof_predictions.csv
07_final_M2_configuration.json
```

It checks exact static-feature order, resolved M0–M3 column counts, all 96 tuning
rows, `C=1/lambda`, selected inner-CV pairs, final candidate-pair membership, and
the Step 07 locked alpha/lambda.

### 4. Previous regression layers

```bash
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
Modeling:          28 PASS, 0 FAIL
Preprocessing:     18 PASS, 0 FAIL
Public reference:  15 PASS, 0 FAIL
V1 baseline:       36 PASS, 0 FAIL
```
