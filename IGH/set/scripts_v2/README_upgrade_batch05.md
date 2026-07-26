# IGH Scheme Upgrade Batch 05

## Scope

Batch 05 adds a repeated-holdout-specific aggregation layer for the three frozen
120/49 IGH tasks and a comparison utility for future schemes that reuse the exact
same frozen assignments.

The classical outer-CV aggregator is deliberately not used. In a repeated
holdout, each repeat evaluates only 49 patients; an individual patient may be
held out zero, one, or multiple times. These results must not be presented as a
complete out-of-fold prediction vector for all 169 patients.

## Scientific safeguards

- All three configured tasks must be structurally `complete` before aggregation.
- Every task's validation membership must exactly match its frozen holdout.
- The assignment SHA256 must match `SPLITS_FROZEN.json`.
- Metrics are summarized across the three paired splits.
- Hyperparameter and threshold selection frequencies are descriptive summaries.
- Feature stability is based on selection frequency and sign consistency across
  the completed splits.
- Model ranking is descriptive only and never locks a final model.
- Scheme comparison requires the exact same split-set ID and assignment SHA256.
- No historical independent-test result is read.

## Current scheme expected rows

For `igh_scheme_003_repeat3_no_clinical`:

- 3 completed splits
- 3 models
- 9 task-metric rows
- 441 prediction rows (`3 × 3 × 49`)
- 21 model-metric summary rows (`3 × 7`)
- 21 paired model-difference rows (`C(3,2) × 7`)
- 507 sample/model coverage rows (`169 × 3`)
