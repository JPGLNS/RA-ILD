# IGH Upgrade Batch 15 — Configurable XGBoost Engine

## Purpose

Batch 15 adds an **additive XGBoost probability-modeling path** to the IGH V2 framework. Existing Elastic Net and Linear SVM paths are not replaced or reinterpreted.

The implementation is ported from TRB XGBoost source commit `d5ff373bf4b430f59b023fab8b2a189c37520a19` while preserving IGH-specific feature selection, model definitions, material subsets, frozen repeated holdout assignments, repeat-specific 3-mer construction, training-only public references, leave-one-out public features, and fold-local preprocessing.

## Important IGH adaptations

1. `experiment.receptor` is validated as `IGH`.
2. The single-task runner uses `select_static_igh_features()` and writes `06_static_igh_feature_list.csv`.
3. Formal XGBoost YAMLs inherit the **existing IGH Linear SVM `models` blocks and `base_resolved_config`** rather than copying TRB Feature14 definitions.
4. Formal experiments use the first 50 repeats of the inherited frozen repeat100 assignments.
5. Batch numbering is IGH Batch 15; the XGBoost path remains isolated from Batch 14 Linear SVM.

## Formal search contract

The current formal pool is 3,888 discrete combinations:

- max_depth: 1, 2, 3
- min_child_weight: 3, 5, 10
- learning_rate: 0.03, 0.05
- subsample: 0.70, 0.85
- colsample_bytree: 0.30, 0.50, 0.70, 1.00
- reg_alpha: 0.1, 0.5, 1.0
- reg_lambda: 3.0, 5.0, 10.0
- n_estimators: 200, 300, 500
- gamma: 0.0
- scale_pos_weight: 1.0

A deterministic random sample of 100 candidates is frozen once into `00_config/xgboost_candidate_bank.csv` and reused for all repeats/folds. Early stopping is disabled in v1. Use outer-task parallelism (up to 8 workers) with `n_jobs_per_fit: 1` to avoid nested oversubscription.

## Leakage control

For each outer task: fixed split → outer-training-only repeat 3-mer features → inner-fold training-only public references/LOO features → pooled inner OOF candidate selection → inner-OOF Youden threshold → full outer-training refit → untouched outer validation evaluation.

XGBoost outputs probability metrics (ROC-AUC, PR-AUC/AP, log loss, Brier, accuracy, sensitivity, specificity, precision, F1) and tree importance (gain, weight, cover, total gain, total cover). SHAP is intentionally not part of this first IGH port.

## Installed components

Source: `IGH/set/src/ra_ild_igh/xgboost_*.py`

Commands: `prepare_xgboost_scheme.py`, `run_xgboost_nested_cv_task.py`, `run_all_xgboost_tasks.py`, `aggregate_xgboost_results.py`, `test_batch15_xgboost.py`, `accept_batch15_xgboost.py`.

Formal configs are under `IGH/set/configs/xgboost/`; the generic repeated-holdout probability visualizer is reused through `IGH/set/scripts_v2/plot_repeated_holdout_results.py`.

## Recommended acceptance sequence

```bash
python3 IGH/set/scripts_v2/test_batch15_xgboost.py
python3 IGH/set/scripts_v2/prepare_xgboost_scheme.py \
  --scheme IGH/set/configs/xgboost/igh_xgboost_smoke_m0_v1.yaml \
  --repository-root "$PWD" --dry-run
python3 IGH/set/scripts_v2/prepare_xgboost_scheme.py \
  --scheme IGH/set/configs/xgboost/igh_xgboost_smoke_m0_v1.yaml \
  --repository-root "$PWD"
python3 IGH/set/scripts_v2/run_xgboost_nested_cv_task.py \
  --config IGH/set/experiments/igh_xgboost_smoke_m0_v1/00_config/resolved_config.yaml \
  --repository-root "$PWD"
python3 IGH/set/scripts_v2/aggregate_xgboost_results.py \
  --config IGH/set/experiments/igh_xgboost_smoke_m0_v1/00_config/resolved_config.yaml \
  --repository-root "$PWD"
python3 IGH/set/scripts_v2/accept_batch15_xgboost.py --repository-root "$PWD"
```
