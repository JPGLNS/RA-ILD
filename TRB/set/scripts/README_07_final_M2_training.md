# Step 07: final full-training M2 tuning and fit

## Inputs

The script reads only the 123-sample training resources:

```text
/data/users/chenhaisheng/RA-ILD/TRB/set/train/result/04_final_feature_matrix/04_train_base_feature_matrix.csv
/data/users/chenhaisheng/RA-ILD/TRB/set/train/result/04_final_feature_matrix/04_feature_manifest.csv
/data/users/chenhaisheng/RA-ILD/TRB/set/train/result/05_modeling/cv_splits/05_outer_fold_assignments.csv
/data/users/chenhaisheng/RA-ILD/TRB/set/train/result/03_public_features/03_public_aa_catalog.csv.gz
/data/users/chenhaisheng/RA-ILD/TRB/set/train/result/03_public_features/03_reference_definition.json
/data/users/chenhaisheng/RA-ILD/TRB/set/train/result/01_AA_clone_table/
/data/users/chenhaisheng/RA-ILD/TRB/set/train/result/05_modeling/cache/
```

It imports the already validated LOO implementation from:

```text
/data/users/chenhaisheng/RA-ILD/TRB/set/scripts/run_05_single_outer_trial_loo.py
```

The independent 51-sample test set is not read.

## Default final-tuning design

```text
20 repeats × 5 folds
M2 = age + sex + static TCR + dynamic public
candidate 1: alpha=0.9, lambda=10
candidate 2: alpha=0.5, lambda=30
selection: one-standard-error rule
threshold: Youden J from 123 sample-level mean repeated OOF probabilities
```

The one-SE rule first finds the best mean repeat ROC-AUC. Any candidate within
one standard error is eligible; among eligible candidates, the script chooses
the one with fewer nonzero coefficients, then higher repeat PR-AUC.

## Install

Copy these files to:

```text
/data/users/chenhaisheng/RA-ILD/TRB/set/scripts/
```

```text
run_07_final_M2_training.py
start_07_final_M2_training.sh
```

## Syntax check

```bash
cd /data/users/chenhaisheng/RA-ILD/TRB

PYTHONPYCACHEPREFIX=/tmp/pycache_ra_ild \
python3 -m py_compile set/scripts/run_07_final_M2_training.py
```

## Recommended nohup run

```bash
cd /data/users/chenhaisheng/RA-ILD/TRB
chmod +x set/scripts/run_07_final_M2_training.py
chmod +x set/scripts/start_07_final_M2_training.sh
bash set/scripts/start_07_final_M2_training.sh
```

Monitor:

```bash
tail -f set/train/result/07_final_M2_model/07_final_M2_nohup.log
```

Check process:

```bash
PID=$(cat set/train/result/07_final_M2_model/07_final_M2_nohup.pid)
ps -p "${PID}" -o pid,etime,%cpu,%mem,cmd
```

## Direct foreground run

```bash
python3 set/scripts/run_07_final_M2_training.py \
  --tuning-mode stable_pairs \
  --candidate-pairs 0.9:10,0.5:30 \
  --selection-rule one_se \
  --threshold-rule youden \
  --overwrite
```

## Optional full original grid

This is supported but substantially slower:

```bash
python3 set/scripts/run_07_final_M2_training.py \
  --tuning-mode full_grid \
  --selection-rule one_se \
  --threshold-rule youden \
  --overwrite
```

## Main outputs

```text
set/train/result/07_final_M2_model/
├── 07_repeated_cv_candidate_predictions.csv.gz
├── 07_repeated_cv_fold_fit_summary.csv
├── 07_repeated_cv_repeat_metrics.csv
├── 07_repeated_cv_candidate_summary.csv
├── 07_selected_candidate_sample_mean_OOF.csv
├── 07_threshold_candidates.csv
├── 07_selected_candidate_repeat_thresholds.csv
├── 07_final_training_public_features_LOO.csv
├── 07_final_preprocessing.csv
├── 07_final_model_coefficients.csv
├── 07_final_training_fitted_predictions.csv
├── 07_full_training_public_reference_masks.npz
├── 07_full_training_public_reference_summary.csv
├── 07_public_reference_build_log.csv.gz
├── 07_public_LOO_assignments.csv.gz
├── 07_final_M2_model.joblib
├── 07_final_M2_configuration.json
└── 07_final_M2_summary.md
```

`07_final_training_fitted_predictions.csv` contains apparent in-sample fitted
results and must not be reported as unbiased model performance. The unbiased
training-stage estimates are the repeated OOF outputs.
