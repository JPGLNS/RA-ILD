# TRB Scheme Upgrade Batch 05

## Scope

Batch 05 adds a scientifically correct aggregation layer for the three frozen
123/51 repeated holdouts and a comparison utility for future schemes using the
same frozen assignments.

The classical outer-CV aggregator is not used for these experiments because it
expects every repeat to produce an out-of-fold prediction for every cohort
sample. A repeated holdout evaluates only 51 patients per repeat, and individual
patients can appear in zero, one, or multiple holdouts.

## Scientific safeguards

- Every task must be structurally `complete` before aggregation.
- Task validation membership must exactly match the corresponding frozen
  `split_01`, `split_02`, or `split_03` holdout membership.
- The frozen assignment file SHA256 is checked against `SPLITS_FROZEN.json`.
- Metrics are summarized as a distribution across the three paired splits.
- Model and scheme rankings are descriptive only; Batch 05 does not lock a final
  model and does not select a winner automatically.
- Scheme comparisons are allowed only when the exact assignment SHA256 and
  split-set ID match.
- No independent test dataset is read.

## Installed files

- `TRB/set/src/ra_ild_trb/repeated_holdout_summary.py`
- `TRB/set/scripts_v2/aggregate_repeated_holdout_results.py`
- `TRB/set/scripts_v2/compare_analysis_schemes.py`
- `TRB/set/scripts_v2/test_batch05_repeated_holdout_summary.py`
- this README and the Batch 05 manifest

## Current scheme expected rows

For `trb_scheme_003_repeat3_no_clinical`:

- 3 completed splits
- 3 models
- 9 task-metric rows
- 459 prediction rows (`3 × 3 × 51`)
- 21 model-metric summary rows (`3 × 7`)
