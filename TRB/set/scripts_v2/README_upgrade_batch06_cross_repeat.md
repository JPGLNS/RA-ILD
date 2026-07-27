# TRB Scheme Upgrade Batch 06 — Frozen-Model Cross-Repeat Evaluation

> **Hotfix 01 (version 1.0.1):** The initial implementation passed the entire
> off-diagonal holdout to `external_public_features`, which correctly rejected
> samples overlapping the source training reference. The hotfix partitions each
> holdout first: saved exact LOO values are used for source-training samples, and
> only source-unseen samples are passed to the external reference transform.

## Purpose

This upgrade adds the approved `n × n` cross-repeat evaluation function.

For `n` repeated holdout splits, each repeat has one final fitted model instance
per configured model specification. Every final model is evaluated on every
repeated holdout:

```text
Model_1 -> Test_1, Test_2, ..., Test_n
Model_2 -> Test_1, Test_2, ..., Test_n
...
Model_n -> Test_1, Test_2, ..., Test_n
```

One model specification therefore produces `n²` complete test-set results.

For three repeats:

```text
3 final model instances × 3 test sets = 9 results
```

For all three current model specifications:

```text
3 model specifications × 3 fit repeats × 3 test repeats = 27 metric rows
```

## Frozen-model contract

The cross-repeat command does **not** rerun inner CV and does **not** refit
Elastic Net. It directly replays each repeat's saved final model using:

- saved final intercept;
- saved final coefficient vector;
- saved selected alpha/lambda as provenance;
- saved threshold;
- source-repeat training data to reconstruct preprocessing;
- saved preprocessing audit to verify exact reconstruction;
- saved exact LOO public values for samples used in source training;
- source-training-reference-only public values for unseen samples.

Prediction is calculated as:

```text
sigmoid(saved_intercept + transformed_X × saved_coefficients)
```

## Primary and supplementary results

Primary:

```text
full_requested_holdout
```

Every patient in each repeated test set is included, exactly as requested.

Supplementary:

```text
unseen_only
```

Only patients that did not participate in the source model's training are used.
When both classes are not present, the row remains in the table with
`metric_status=not_estimable_both_classes_required`.

## Mandatory audits

The run stops unless:

- every repeated-holdout task is complete;
- reconstructed preprocessing matches the saved outer-final audit;
- native public features reproduce the saved native public table;
- diagonal probabilities reproduce the original native predictions;
- the same frozen model gives the same probability to the same patient wherever
  that patient appears in multiple repeated test sets;
- expected row counts match `repeat × repeat × model × holdout`;
- frozen split-set ID and assignment SHA remain valid.

## Outputs

Default directory:

```text
<experiment root>/04_cross_repeat_evaluation/
```

Main outputs:

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

## Expected counts

For `R` repeats, `M` selected model specifications, and holdout size `H`:

```text
model instances            = R × M
full metric rows            = R × R × M
unseen-only metric rows     = R × R × M
prediction rows             = R × R × M × H
overlap audit rows          = R × R
```

Current repeat3 scheme, one selected model:

```text
model instances            = 3
full metric rows            = 9
unseen-only metric rows     = 9
prediction rows             = 459
```

Current repeat3 scheme, all three models:

```text
model instances            = 9
full metric rows            = 27
unseen-only metric rows     = 27
prediction rows             = 1,377
```

## Installation

From repository root:

```bash
unzip -o TRB_cross_repeat_evaluation_batch06.zip \
  -d /data/users/chenhaisheng/RA-ILD
```

Only new Batch 06 files are installed. Existing TRB source and completed task
results are not overwritten.

## Test

```bash
conda activate ra-ild
cd /data/users/chenhaisheng/RA-ILD

python3 \
  TRB/set/scripts_v2/test_batch06_cross_repeat_evaluation.py
```

Expected:

```text
PASS test_expected_counts
PASS test_evaluation_role
PASS test_overlap_audit
PASS test_frozen_probability_uses_saved_coefficients
Batch 06 focused tests: 4/4 PASS
```

## Run one selected model

```bash
python3 \
  TRB/set/scripts_v2/evaluate_cross_repeat_holdouts.py \
  --config \
  TRB/set/experiments/trb_scheme_003_repeat3_no_clinical/00_config/resolved_config.yaml \
  --models M2_static_tcr_public
```

Expected conceptual counts:

```text
Pairs per model:      3 x 3 = 9
Full metric rows:     9
Unseen metric rows:   9
Prediction rows:      459
```

## Run all configured models

```bash
python3 \
  TRB/set/scripts_v2/evaluate_cross_repeat_holdouts.py \
  --config \
  TRB/set/experiments/trb_scheme_003_repeat3_no_clinical/00_config/resolved_config.yaml \
  --models all
```

Expected conceptual counts:

```text
Pairs per model:      3 x 3 = 9
Full metric rows:     27
Unseen metric rows:   27
Prediction rows:      1377
```

## Interpretation

The complete `n × n` matrix is intentionally produced. Diagonal cells are the
original native holdouts. Off-diagonal cells may include patients that
participated in the source model's training; they remain in the requested full
result and are explicitly annotated.

The matrix is a correlated cross-repeat evaluation matrix, not `n²` independent
external validation cohorts. It does not automatically select a final model.
