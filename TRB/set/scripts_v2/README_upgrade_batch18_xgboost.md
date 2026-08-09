# TRB Upgrade Batch 18 — Configurable XGBoost Engine

## Purpose

Batch 18 adds an **additive XGBoost probability-modeling path** to the current TRB V2 framework. It does not replace or reinterpret the existing Elastic Net or Linear SVM paths.

The XGBoost path reuses the validated framework layers for M0–M3/Feature14/arbitrary YAML models, cohort/material subsets, frozen repeated holdout, repeat-specific 3-mer construction, training-only public references, exact fitting-sample leave-one-out public features, and fold-local preprocessing.

## Random Search contract

The scheme YAML defines `model_engine.hyperparameters` using a simple rule:

- scalar = fixed for all candidates;
- list = one dimension of the candidate pool.

`model_engine.search.strategy: random` samples `n_iter` combinations **once**, without replacement, from the full discrete pool using `search.random_state`. `prepare_xgboost_scheme.py` freezes those combinations into `00_config/xgboost_candidate_bank.csv` and stores its SHA256 in the resolved config. Every repeat/fold reads that same bank; candidates are never resampled per repeat.

The supplied main example implements the discussed pool of 972 combinations and samples 100 candidates. `n_estimators` is YAML-configurable and is fixed to 300 in the example; it can later be changed to a list without editing Python.

## Leakage control and tuning

For each outer task:

1. fixed outer train/validation assignment is loaded;
2. repeat-specific 3-mer features are constructed from outer-training data only;
3. each inner fold rebuilds training-only public references and exact leave-one-out features for fitting samples;
4. every frozen XGBoost candidate is evaluated on all inner folds;
5. pooled inner OOF predictions select the candidate by `roc_auc` or `log_loss`;
6. an inner-OOF Youden threshold is selected;
7. the selected candidate is refit on the full outer-training partition;
8. the untouched outer validation partition is evaluated.

Outer outputs include ROC-AUC, PR-AUC/Average Precision, log loss, Brier score, accuracy, sensitivity, specificity, precision and F1.

## Parallelism

Outer task parallelism remains controlled by `run_all_xgboost_tasks.py --workers`. Batch 18 allows up to 8 outer workers, matching the existing Linear SVM orchestration limit. XGBoost internal threads are separately controlled by:

```yaml
model_engine:
  runtime:
    n_jobs_per_fit: 1
```

The default/recommended combination for the planned run is `--workers 8` plus `n_jobs_per_fit: 1`, avoiding nested CPU oversubscription.

## Early stopping

Batch 18 v1 requires `early_stopping: false`. `n_estimators` is explicitly configured/tuned instead. This keeps candidate evaluation inside the existing leakage-controlled nested-CV contract. Early stopping can be considered in a later upgrade with a dedicated inner-fold design.

## Feature interpretation

XGBoost has no linear coefficient. Batch 18 therefore writes final-model tree importance for every retained predictor:

- gain;
- weight;
- cover;
- total gain;
- total cover.

Aggregation reports retained-task frequency, nonzero-gain frequency across all configured completed tasks, and gain/weight/cover summaries. Predictors removed by task-local preprocessing are not silently counted as retained. SHAP is intentionally not part of Batch 18 v1 and can be added later without changing this engine.

## Installed components

Source modules:

```text
TRB/set/src/ra_ild_trb/xgboost_model.py
TRB/set/src/ra_ild_trb/xgboost_config.py
TRB/set/src/ra_ild_trb/xgboost_nested_cv.py
TRB/set/src/ra_ild_trb/xgboost_summary.py
```

Commands:

```text
TRB/set/scripts_v2/prepare_xgboost_scheme.py
TRB/set/scripts_v2/run_xgboost_nested_cv_task.py
TRB/set/scripts_v2/run_all_xgboost_tasks.py
TRB/set/scripts_v2/aggregate_xgboost_results.py
```

Tests/examples:

```text
TRB/set/tests/test_xgboost_model.py
TRB/set/tests/test_xgboost_config.py
TRB/set/tests/test_xgboost_summary.py
TRB/set/tests/test_xgboost_nested_cv.py
TRB/set/scripts_v2/test_batch18_xgboost.py
TRB/set/scripts_v2/accept_batch18_xgboost.py
TRB/set/configs/xgboost/trb_xgboost_main_example.yaml
TRB/set/configs/xgboost/trb_xgboost_smoke_m0_v1.yaml
TRB/set/configs/visualization/trb_xgboost_visualization_example.yaml
```

## Workflow

Prepare/dry-run:

```bash
python3 TRB/set/scripts_v2/prepare_xgboost_scheme.py \
  --scheme TRB/set/configs/xgboost/trb_xgboost_main_example.yaml \
  --repository-root /data/users/chenhaisheng/RA-ILD \
  --dry-run
```

Prepare for real:

```bash
python3 TRB/set/scripts_v2/prepare_xgboost_scheme.py \
  --scheme TRB/set/configs/xgboost/trb_xgboost_main_example.yaml \
  --repository-root /data/users/chenhaisheng/RA-ILD
```

Check status:

```bash
python3 TRB/set/scripts_v2/run_all_xgboost_tasks.py \
  --config TRB/set/experiments/<scheme_id>/00_config/resolved_config.yaml \
  --repository-root /data/users/chenhaisheng/RA-ILD \
  --status-only
```

Run with eight outer workers:

```bash
python3 TRB/set/scripts_v2/run_all_xgboost_tasks.py \
  --config TRB/set/experiments/<scheme_id>/00_config/resolved_config.yaml \
  --repository-root /data/users/chenhaisheng/RA-ILD \
  --workers 8 --execute
```

Aggregate:

```bash
python3 TRB/set/scripts_v2/aggregate_xgboost_results.py \
  --config TRB/set/experiments/<scheme_id>/00_config/resolved_config.yaml \
  --repository-root /data/users/chenhaisheng/RA-ILD
```

The existing generic probability visualization command is reused:

```bash
python3 TRB/set/scripts_v2/plot_repeated_holdout_results.py \
  --config TRB/set/configs/visualization/<your_xgboost_visualization>.yaml \
  --repository-root /data/users/chenhaisheng/RA-ILD
```

## Backward compatibility

Batch 18 is additive. No historical Elastic Net or Linear SVM source/config/result is overwritten. New experiments use independent scheme IDs and result directories.
