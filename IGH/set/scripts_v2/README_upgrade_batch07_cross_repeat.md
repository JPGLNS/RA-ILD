# IGH Scheme Upgrade Batch 07 — Frozen-Model Cross-Repeat Evaluation

## Purpose

This upgrade ports the validated TRB Batch 06 v1.0.1 cross-repeat evaluation to
the independent IGH namespace.

For `R` frozen repeated-holdout splits, every repeat-specific final model is
applied to every frozen holdout:

```text
Model_1 -> Holdout_1, Holdout_2, ..., Holdout_R
Model_2 -> Holdout_1, Holdout_2, ..., Holdout_R
...
Model_R -> Holdout_1, Holdout_2, ..., Holdout_R
```

One model specification therefore produces `R²` evaluation cells. The current
IGH scheme has three repeats, three model specifications, and 49 samples per
holdout:

```text
model instances        = 3 repeats × 3 specifications = 9
full metric rows        = 3 × 3 × 3 = 27
unseen-only metric rows = 27
prediction rows         = 3 × 3 × 3 × 49 = 1,323
```

## Frozen-model contract

The evaluator does not rerun inner CV and does not refit Elastic Net. It replays:

- the saved final intercept;
- the saved final coefficient vector;
- the saved selected alpha/lambda as provenance;
- the saved inner-OOF-derived threshold;
- preprocessing reconstructed from the source-repeat training partition and
  verified against the saved preprocessing audit.

For public-reference features:

- source-training samples use their saved exact leave-one-out values;
- source-unseen samples are transformed using only the source-repeat training
  reference.

This includes the validated TRB v1.0.1 overlap hotfix: a target holdout is split
into source-training-seen and source-unseen rows before any external public
transformation.

## Outputs

Default output directory:

```text
IGH/set/experiments/igh_scheme_003_repeat3_no_clinical/
04_cross_repeat_evaluation/
```

Key files:

```text
04_cross_repeat_predictions.csv.gz
04_cross_repeat_full_holdout_metrics.csv
04_cross_repeat_unseen_only_metrics.csv
04_cross_repeat_overlap_audit.csv
04_preprocessing_reproduction_audit.csv
04_native_public_reproduction_audit.csv
04_native_prediction_reproduction_audit.csv
04_repeated_sample_prediction_consistency.csv
04_matrix_full_<model>_<metric>.csv
04_matrix_unseen_only_<model>_<metric>.csv
04_CROSS_REPEAT_EVALUATION_COMPLETE.json
```

Rows of each matrix are `fit_repeat`; columns are `evaluation_repeat`.

## Required audits

The run stops unless:

- all three repeated-holdout tasks are complete;
- the frozen split-set ID and assignment SHA256 remain valid;
- source-repeat preprocessing is reproduced;
- native diagonal public features are reproduced;
- native diagonal probabilities and thresholds are reproduced;
- the same frozen model gives the same probability to the same patient wherever
  that patient reappears;
- expected output row counts match the configured repeats, models, and holdout
  size.

## Interpretation

`full_requested_holdout` contains every sample in each requested holdout. An
off-diagonal cell may include patients used in the source model's training and is
therefore not an independent external test.

`unseen_only` contains only patients not used to fit the source-repeat model. Its
sample size can vary by cell. When both classes are not present, the row is kept
with `metric_status=not_estimable_both_classes_required`.

The n×n matrix is a correlated stability analysis. It does not automatically
select a final model and does not read the historical independent test set.

## Focused test

```bash
python3 IGH/set/scripts_v2/test_batch07_cross_repeat_evaluation.py
```

Expected:

```text
Batch 07 focused tests: 5/5 PASS
```

## Run all configured IGH models

```bash
python3 IGH/set/scripts_v2/evaluate_cross_repeat_holdouts.py \
  --config IGH/set/experiments/igh_scheme_003_repeat3_no_clinical/00_config/resolved_config.yaml \
  --repository-root /data/users/chenhaisheng/RA-ILD \
  --models all
```

Expected conceptual counts:

```text
Pairs per model:      3 x 3 = 9
Full metric rows:     27
Unseen metric rows:   27
Prediction rows:      1323
```
