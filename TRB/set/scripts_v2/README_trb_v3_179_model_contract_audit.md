# TRB V3 179 model-contract audit

This is a read-only guard for the frozen five-algorithm experiment matrix.

## Scientific contract

- Total: M0-M3; 179 samples; 100 repeated holdouts; fold 1.
- PBMC: M0-M2; 116 samples; 100 repeated holdouts; fold 1.
- Buffycoat: M0-M2; 63 samples; 100 repeated holdouts; fold 1.
- M0: age + sex.
- M1: M0 + Core83 + weighted/unweighted Top-500 3-mers.
- M2: M1 + the eight frozen public-reference predictors.
- M3: M2 + material (Total only).
- 3-mer vocabulary selection remains `outer_repeat_train_only`; this audit deliberately rejects a change to inner-fold-local selection.

Raw predictor counts before categorical encoding and preprocessing:

| Cohort | Model | Numeric | Categorical |
|---|---|---:|---:|
| Total | M0 | 1 | 1 |
| Total | M1 | 1084 | 1 |
| Total | M2 | 1092 | 1 |
| Total | M3 | 1092 | 2 |
| PBMC/Buffycoat | M0 | 1 | 1 |
| PBMC/Buffycoat | M1 | 1084 | 1 |
| PBMC/Buffycoat | M2 | 1092 | 1 |

## Commands

Run the focused unit tests:

```bash
python3 TRB/set/tests/test_trb_v3_179_model_contract.py
```

Audit the prepared 15-scheme matrix:

```bash
python3 TRB/set/scripts_v2/audit_trb_v3_179_five_algorithm_contract.py \
  --repository-root "$PWD" \
  --output TRB/set/experiments/trb_v3_179_five_algorithm_global/00_audit/model_contract_audit.json
```

The audit requires the prepared five-algorithm manifest and all 15 resolved
configuration files. It checks exact model definitions, raw predictor counts,
3-mer settings, sample counts, repeat/fold contract, and equality of model,
data, public-reference, split and training-bundle definitions across algorithms
within each cohort.

It does not start training, alter inputs, modify frozen bundles, or overwrite
existing experiment outputs. The optional JSON report is write-once; choose a
new path for a subsequent audit.
