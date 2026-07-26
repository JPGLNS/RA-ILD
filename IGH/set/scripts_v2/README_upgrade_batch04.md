# IGH Scheme Upgrade Batch 04

## Purpose

Batch 04 connects the three frozen 120/49 assignments from Batch 03 to the
existing leakage-controlled nested-CV engine. It does not replace or rewrite the
Elastic Net, public-reference, preprocessing, metric, or threshold algorithms.

The default generated scheme contains three models:

- `M1_static_igh`
- `M2_static_igh_public`
- `M3_static_igh_public_material`

It contains no M0, age, or sex predictors.

## Scientific execution contract

For each frozen split:

1. Use exactly 123 patients as training and 51 as holdout.
2. Generate fixed five-fold inner assignments inside the 123 training patients.
3. For every inner fold, construct public features using exact leave-one-out
   references for fitting samples and training-reference-only transforms for
   validation samples.
4. Fit all configured alpha/lambda candidates using fold-local preprocessing.
5. Rank candidates by pooled inner ROC-AUC, pooled inner PR-AUC, stronger lambda,
   and higher alpha, matching the existing V2 rule.
6. Select the threshold from the selected candidate's pooled inner OOF
   predictions using the existing Youden/closest-to-0.5 rule.
7. Refit on the full 120-person training partition.
8. Build the final public reference from those 123 patients only and apply it once
   to the 49-person holdout.

The full-cohort CDR3-AA catalog is only a sparse-column indexing universe. A
sequence receives membership in a public-reference mask only when it satisfies
training-partition thresholds. Sequences present only in holdout patients have
zero reference-mask membership and cannot influence fitting.

## Prepared training bundle

The heavy prepared inputs are written to:

`IGH/set/training_bundles/trb_repeat3_training_inputs_v1/`

This directory contains the full static matrix, CDR3-AA catalog, sparse matrices,
fixed outer/inner assignments, audits, and a frozen marker. It can be large and
should not be committed to Git.

A concrete scheme YAML is generated at:

`IGH/set/configs/schemes/generated/trb_scheme_003_repeat3_no_clinical.generated.yaml`

## Task count

Only validation outer fold 1 is executed for each frozen repeat:

- 3 repeats
- 1 executed outer fold per repeat
- 3 total tasks

Each task performs:

- 3 models
- 24 alpha/lambda candidates
- 5 inner folds
- 360 inner fits

## Safety

- Existing Batch 01–03 tests remain runnable.
- Existing V1 and earlier V2 result directories are not overwritten.
- A frozen training bundle cannot be regenerated under the same bundle ID.
- A prepared analysis scheme containing task results cannot be refreshed.
- Holdout labels are used only for final evaluation, not feature/reference
  construction, hyperparameter selection, or threshold selection.
