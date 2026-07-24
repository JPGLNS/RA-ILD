# TRB Scheme Upgrade Batch 03

## Scope

Batch 03 adds a frozen, reusable repeated-holdout split-set generator for the
single RA-ILD TRB project. It does not run models and does not modify the
Elastic Net, public-reference, preprocessing, or threshold-selection engine.

## Scientific rules

- Combine the old 123-person training metadata and 51-person test metadata only
  to reconstruct the complete 174-patient cohort. Their old roles are ignored.
- Split at the patient level; the current TRB cohort must have exactly one sample
  per patient.
- Generate three distinct 123/51 train/holdout assignments.
- Every candidate is stratified by the joint `cohort × batch` stratum.
- Candidate ranking uses only `material`, `sex`, and `age` balance plus a small
  prespecified overlap penalty.
- Model predictions, AUC, downstream features, and test results are never used.
- Once `SPLITS_FROZEN.json` exists, the split set is immutable. A changed split
  requires a new `split_set.id`.

## Installed files

- `TRB/set/src/ra_ild_trb/repeated_holdout.py`
- `TRB/set/scripts_v2/generate_repeated_holdout_splits.py`
- `TRB/set/scripts_v2/test_batch03_repeated_holdout.py`
- `TRB/set/configs/split_sets/ra_ild_repeat3_v1.yaml`
- Batch manifest and this README

## Expected generated output

`TRB/set/split_sets/ra_ild_repeat3_v1/`

- `repeated_holdout_assignments.csv`
- `split_counts.csv`
- `split_balance_audit.csv`
- `selected_split_candidates.csv`
- `pairwise_holdout_overlap.csv`
- `sample_holdout_frequency.csv`
- `combined_metadata_snapshot.csv`
- `split_set_spec_snapshot.yaml`
- `metadata_input_manifest.json`
- `SPLITS_FROZEN.json`

The assignment file contains patient/sample identifiers and should remain in the
protected server analysis area. Do not add the generated directory to Git unless
that data-handling decision has been explicitly approved.
