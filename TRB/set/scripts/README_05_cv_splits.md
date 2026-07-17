# Step 05 repeated nested CV split builder

## Purpose

This step freezes the cross-validation schedule before Elastic Net modeling:

```text
Outer CV: 5 folds × 20 repeats = 100 outer validation tasks
Inner CV: 5 folds inside every outer-training subset
```

All model variants must reuse these exact assignments.

## Input

```text
/data/users/chenhaisheng/RA-ILD/TRB/set/train/result/04_final_feature_matrix/04_train_base_feature_matrix.csv
```

Only these metadata columns are used:

```text
sample_id, patient, cohort, batch, material, sex
```

The independent test set and TCR feature values are not read.

## Output

```text
/data/users/chenhaisheng/RA-ILD/TRB/set/train/result/05_modeling/cv_splits/
```

Files:

```text
05_outer_fold_assignments.csv
05_outer_tasks_long.csv.gz
05_inner_fold_assignments.csv.gz
05_outer_fold_balance.csv
05_inner_fold_balance.csv
05_cv_split_configuration.json
05_cv_split_summary.md
```

## Main interpretation

For an outer task `(outer_repeat=r, outer_fold=f)`:

```text
outer_fold == f  -> outer validation
outer_fold != f  -> outer training
```

The matching rows in `05_inner_fold_assignments.csv.gz` are used only to tune
Elastic Net alpha/lambda inside that outer-training subset.

## Balance rules

1. RA/ILD is hard-stratified; counts differ by at most one across folds.
2. Among valid cohort-stratified candidates, the script favors better balance
   for batch, material and sex, including cohort×covariate distributions.
3. Twenty outer partitions are required to be distinct.
4. Total outer fold sizes differ by at most one (24–25 samples for 123 cases).

## Expected row counts

With 123 training samples:

```text
05_outer_fold_assignments.csv:  123 × 20     = 2,460 rows
05_outer_tasks_long.csv.gz:     123 × 20 × 5 = 12,300 rows
05_inner_fold_assignments.csv.gz:
                                123 × 20 × 4 = 9,840 rows
```

## Install and run

Copy both scripts to:

```text
/data/users/chenhaisheng/RA-ILD/TRB/set/scripts/
```

Then run:

```bash
cd /data/users/chenhaisheng/RA-ILD/TRB
chmod +x set/scripts/build_05_cv_splits.py
chmod +x set/scripts/run_05_build_cv_splits.sh
bash set/scripts/run_05_build_cv_splits.sh
```

Or directly:

```bash
python3 set/scripts/build_05_cv_splits.py --overwrite
```

This step does not fit a model. After it passes QC, the next task is to test one
outer task, such as `outer_repeat=1, outer_fold=1`, through the complete dynamic
public-feature, preprocessing and Elastic Net workflow.
