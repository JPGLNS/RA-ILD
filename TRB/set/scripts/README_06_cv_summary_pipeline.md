# Step 06 CV summary pipeline

## Confirmed input

```text
/data/users/chenhaisheng/RA-ILD/TRB/set/train/result/05_modeling/single_outer_trial_loo/
```

Expected directories: `repeat_01_fold_01` through `repeat_20_fold_05`.

## Output

```text
/data/users/chenhaisheng/RA-ILD/TRB/set/train/result/06_cv_result_summary/
├── collected/
├── visualization/
└── analysis/
```

## Scripts

```text
collect_06_cv_results.py
plot_06_cv_model_metrics.py
analyze_06_cv_results.py
run_06_cv_summary_pipeline.sh
```

Copy to:

```text
/data/users/chenhaisheng/RA-ILD/TRB/set/scripts/
```

## Script 1: collection and QC

```bash
python3 set/scripts/collect_06_cv_results.py --overwrite
```

Expected complete outputs include 100 passed tasks, 400 metric rows, 9,840 prediction rows and 400 parameter rows.

## Script 2: five-panel visualization

```bash
python3 set/scripts/plot_06_cv_model_metrics.py --overwrite
```

Panels: ROC-AUC, PR-AUC, Sensitivity, Specificity and F1. Colors:

```text
M0 #65c8cc
M1 #f0e94b
M2 #72c15a
M3 #f3793b
```

Boxes show the 100 outer-fold values. Because the same splits are used for all four models, p-values use paired analyses rather than ordinary one-way ANOVA:

```text
Global: Friedman test
Pairwise: paired Wilcoxon signed-rank test
Multiple testing: Holm correction
```

Tests use 20 pooled repeat-level OOF results. The figure annotates M0 vs M1, M1 vs M2 and M2 vs M3; all six comparisons are saved to CSV.

## Script 3: performance and stability

```bash
python3 set/scripts/analyze_06_cv_results.py --overwrite
```

Default stable-feature rule:

```text
selection frequency >= 50%
sign consistency >= 80%
```

## Run all

```bash
bash set/scripts/run_06_cv_summary_pipeline.sh
```

## Syntax check

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache_ra_ild \
python3 -m py_compile \
  set/scripts/collect_06_cv_results.py \
  set/scripts/plot_06_cv_model_metrics.py \
  set/scripts/analyze_06_cv_results.py
```
