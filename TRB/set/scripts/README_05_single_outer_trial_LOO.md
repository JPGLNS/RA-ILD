# Step 05.3 single outer-fold trial — efficient leave-one-out public features

## What changed

Only the model-fitting samples' dynamic public-feature generation changed:

```text
Old: 5-fold cross-fitting
New: exact efficient leave-one-out (LOO)
```

For a training partition containing `n` samples, every model-fitting sample is
scored against a reference built from the other `n-1` samples. Validation
samples continue to use the complete corresponding training partition.

This keeps self-inclusion out while making the reference sizes nearly equal:

```text
model-fitting sample: n-1 reference samples
validation sample:    n reference samples
```

Steps 01–04, the repeated nested-CV split files, and the existing sparse cache
do not need to be rebuilt.

## Confirmed project paths

Inputs:

```text
/data/users/chenhaisheng/RA-ILD/TRB/set/train/result/04_final_feature_matrix/04_train_base_feature_matrix.csv
/data/users/chenhaisheng/RA-ILD/TRB/set/train/result/04_final_feature_matrix/04_feature_manifest.csv

/data/users/chenhaisheng/RA-ILD/TRB/set/train/result/05_modeling/cv_splits/05_outer_fold_assignments.csv
/data/users/chenhaisheng/RA-ILD/TRB/set/train/result/05_modeling/cv_splits/05_inner_fold_assignments.csv.gz

/data/users/chenhaisheng/RA-ILD/TRB/set/train/result/03_public_features/03_public_aa_catalog.csv.gz
/data/users/chenhaisheng/RA-ILD/TRB/set/train/result/03_public_features/03_reference_definition.json

/data/users/chenhaisheng/RA-ILD/TRB/set/train/result/01_AA_clone_table/
```

Existing cache reused automatically:

```text
/data/users/chenhaisheng/RA-ILD/TRB/set/train/result/05_modeling/cache/
```

New output root:

```text
/data/users/chenhaisheng/RA-ILD/TRB/set/train/result/05_modeling/single_outer_trial_loo/
```

Using a new output root preserves the earlier 5-fold-cross-fitting trial for
method comparison.

## Install

Copy these files to:

```text
/data/users/chenhaisheng/RA-ILD/TRB/set/scripts/
```

Files:

```text
run_05_single_outer_trial_loo.py
run_05_single_outer_trial_loo.sh
```

Then:

```bash
cd /data/users/chenhaisheng/RA-ILD/TRB
chmod +x set/scripts/run_05_single_outer_trial_loo.py
chmod +x set/scripts/run_05_single_outer_trial_loo.sh
```

## Syntax check

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache_ra_ild \
python3 -m py_compile \
  set/scripts/run_05_single_outer_trial_loo.py
```

## Run

Recommended:

```bash
cd /data/users/chenhaisheng/RA-ILD/TRB
bash set/scripts/run_05_single_outer_trial_loo.sh
```

Or directly:

```bash
python3 set/scripts/run_05_single_outer_trial_loo.py \
  --outer-repeat 1 \
  --outer-fold 1 \
  --public-feature-set raw_bilateral \
  --overwrite
```

Do not add `--force-rebuild-cache`; the existing presence/frequency cache is
independent of cross-fitting versus LOO and remains valid.

## Expected preflight message

The script must first reproduce the step-03 full-training references exactly:

```text
[PASS] Full-train public reference sizes reproduce step-03:
global=394,235, RA_specific=41,649,
ILD_specific=28,947, shared=38,585
```

A mismatch stops the run before modeling.

## Output directory

```text
set/train/result/05_modeling/single_outer_trial_loo/repeat_01_fold_01/
```

Important files:

```text
05_trial_configuration.json
05_trial_sample_roles.csv

05_dynamic_public_features_outer_train_loo.csv
05_dynamic_public_features_outer_validation.csv
05_public_reference_build_summary.csv
05_public_loo_assignments.csv

05_inner_tuning_results.csv
05_inner_selected_oof_predictions.csv
05_outer_validation_predictions.csv
05_outer_validation_metrics.csv
05_final_model_coefficients.csv
05_preprocessing_summary.csv
05_static_tcr_feature_list.csv
05_single_outer_trial_summary.md
```

## LOO audit checks

`05_public_loo_assignments.csv` records each held-out training sample and must
satisfy:

```text
n_reference_samples = n_partition_samples - 1
```

The reference summary contains the sample-specific global, RA-specific,
ILD-specific and shared set sizes after removing that sample.

## Default model definitions

```text
M0: age + sex
M1: age + sex + static TCR
M2: age + sex + static TCR + dynamic public
M3: age + sex + material + static TCR + dynamic public
```

The default `raw_bilateral` public set contains eight ratio/frequency features.
The independent test set is not read.

## Validation completed before delivery

The script passed:

1. Python syntax/bytecode compilation.
2. Exact LOO equivalence test against brute-force rebuilding of the reference
   for every sample; all 18 public features and all reference-set sizes matched.
3. Synthetic end-to-end inner tuning and outer validation for all four models.
4. Model convergence and expected-output checks.
5. Negative leakage test: an outer-validation sample injected into the inner
   assignment caused the script to stop as intended.
