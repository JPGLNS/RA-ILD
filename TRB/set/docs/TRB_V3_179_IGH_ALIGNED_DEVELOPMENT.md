# TRB V3 179 IGH-aligned development

Branch: `feature/trb-v3-179-igh-aligned-hierarchy`

This is a separate development line. Do not modify or reuse the historical Repeat-001 output directory or the previous five-algorithm manifest.

## Frozen model hierarchy

- Total: M0, M1, M2, M3.
- PBMC: M0, M1, M2.
- Buffycoat: M0, M1, M2.

| Model | Predictors |
|---|---|
| M0 | Age + Sex |
| M1 | M0 + Core83 + 500 unweighted + 500 weighted 3-mers |
| M2 | M1 + eight raw_bilateral Public features |
| M3 | M2 + Material |

The eight Public features are the existing raw_bilateral set. Enriched-dictionary predictors, QC/depth predictors, and exploratory feature groups remain excluded.

## Inner-fold 3-mer policy

The outer-training vocabulary is used only for the final outer-training refit and outer holdout. Each of the five inner folds independently selects its own vocabulary using only that fold's inner-training sample IDs. Its inner-validation samples are transformed using that frozen vocabulary.

The AA clone tables are preloaded once per outer task as unlabeled per-sample 3-mer maps. The ranking function receives explicit training IDs; validation and outer-holdout IDs are not used to construct the candidate universe, prevalence, variance, or ranking.

## Code changes

- `repeat_3mer_features.py`: supports `inner_fold_train_only`, preloads per-sample 3-mer maps once, and builds separate fold-specific vocabularies/matrices.
- `nested_cv.py`: rebuilds 3-mer columns for each inner fold and replaces the outer-repeat 3-mer predictors in each model specification with the current fold's predictors.
- Elastic Net, Linear SVM, and XGBoost runners pass the shared preloaded 3-mer inputs into their nested-CV engines.
- Inner-fold vocabulary and audit outputs are written to:
  - `06_inner_repeat_3mer_audit.csv`
  - `06_inner_repeat_3mer_vocabulary.csv`
- The existing outer-repeat vocabulary/matrix remains recorded separately for the final outer fit and holdout application.
- The five-algorithm preparation and scheduler now use dedicated IGH-aligned experiment roots and a dedicated manifest/status path.

## Server-side preparation

First update the checkout:

```bash
cd /data/users/chenhaisheng/RA-ILD
git fetch origin
git checkout feature/trb-v3-179-igh-aligned-hierarchy
git pull --ff-only
```

Run the fold-local 3-mer regression test and syntax checks:

```bash
python3 TRB/set/tests/test_trb_v3_inner_fold_3mer.py

python3 -m py_compile \
  TRB/set/src/ra_ild_trb/repeat_3mer_features.py \
  TRB/set/src/ra_ild_trb/nested_cv.py \
  TRB/set/src/ra_ild_trb/linear_svm_nested_cv.py \
  TRB/set/src/ra_ild_trb/xgboost_nested_cv.py \
  TRB/set/scripts_v2/run_nested_cv_task.py \
  TRB/set/scripts_v2/run_linear_svm_nested_cv_task.py \
  TRB/set/scripts_v2/run_xgboost_nested_cv_task.py \
  TRB/set/scripts_v2/prepare_trb_v3_179_igh_aligned_schemes.py \
  TRB/set/scripts_v2/prepare_hierarchical_five_algorithm_schemes.py
```

Create and validate the three corrected Elastic-Net base schemes:

```bash
python3 TRB/set/scripts_v2/prepare_trb_v3_179_igh_aligned_schemes.py
python3 TRB/set/scripts_v2/prepare_trb_v3_179_igh_aligned_schemes.py --apply
```

Then prepare Ridge, Lasso, Linear SVM, and XGBoost:

```bash
python3 TRB/set/scripts_v2/prepare_hierarchical_five_algorithm_schemes.py
python3 TRB/set/scripts_v2/prepare_hierarchical_five_algorithm_schemes.py --apply
```

Finally, validate the complete 15-entry / 1500-task scheduler plan without launching training:

```bash
python3 TRB/set/scripts_v2/run_hierarchical_five_algorithm_tasks.py
```

Do not add `--execute` yet. Do not start Repeat100 until the corrected resolved configurations, predictor audit, and new Repeat-001 smoke test have passed.

## Important implementation note

The fold-local 3-mer code and engine integration are committed, but they have not been executed in the user's neo3 environment. The above test, syntax, scheme-preparation, and scheduler dry-run results are still required. A Repeat-001 smoke-test command and full predictor/output audit are the next development gates before training.
