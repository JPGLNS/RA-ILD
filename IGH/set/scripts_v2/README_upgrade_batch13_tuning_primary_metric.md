# IGH Upgrade Batch 13 — Configurable Tuning Primary Metric

## Purpose

Batch 13 permits each new IGH experiment to choose the deterministic
hyperparameter-ranking objective through:

```yaml
model_selection:
  tuning_primary_metric: "roc_auc"
```

or:

```yaml
model_selection:
  tuning_primary_metric: "log_loss"
```

If the field is omitted, `roc_auc` is used. Existing resolved configurations
therefore retain their previous behavior.

## Ranking rules

### ROC-AUC mode

1. pooled inner ROC-AUC, descending;
2. pooled inner PR-AUC, descending;
3. lambda, descending;
4. alpha, descending.

### Log-loss mode

1. pooled inner log-loss, ascending;
2. pooled inner Brier score, ascending;
3. pooled inner ROC-AUC, descending;
4. lambda, descending;
5. alpha, descending.

Both modes continue to derive the classification threshold by Youden's rule
from the selected candidate's pooled inner OOF predictions.

## Added audit fields

Future task outputs record:

- `pooled_inner_log_loss`;
- `pooled_inner_brier_score`;
- `tuning_primary_metric`;
- `candidate_selection_policy`;
- `inner_selected_log_loss`;
- `inner_selected_brier_score`.

Both IGH repeated-holdout aggregators preserve these fields. Historical tasks
that lack them are interpreted as legacy ROC-AUC tuning and remain aggregatable.

## Feature-ablation schemes

`prepare_feature_ablation_scheme.py` now accepts:

```yaml
model_selection:
  tuning_primary_metric: "log_loss"
```

It writes the matching `candidate_sort` and
`nested_cv.candidate_selection_policy` into the resolved configuration.

## Acceptance

```bash
python3 IGH/set/scripts_v2/test_batch13_tuning_primary_metric.py
```

Expected ending:

```text
IGH Batch 13 focused tests: 7/7 PASS
IGH_BATCH13_ACCEPTANCE_PASS
```

The acceptance test creates and removes a temporary log-loss feature-ablation
configuration. It does not fit the full repeat100 experiment.

## Git

Commit the seven modified framework files and three new Batch 13 files only.
Do not commit `BATCH13_INSTALLATION.json`, temporary experiment directories,
upgrade archives, extracted package directories, existing task/aggregation
outputs, or `*.bak.batch13.*` files.
