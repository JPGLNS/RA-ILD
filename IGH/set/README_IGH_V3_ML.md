# IGH V3 Machine Learning Experiment Freeze

Frozen: 2026-11-15
Data version: igh_v3_175
Repertoire input: Top10000 CDR3 AA clones
Positive class: RA-ILD (ILD)

## Cohorts

- Total: 175 samples
- PBMC: 113 samples
- Buffy coat: 62 samples

## Evaluation

- Elastic Net Logistic Regression
- 100 frozen repeated stratified holdouts per configuration
- 5-fold inner cross-validation
- 24 hyperparameter candidates
- Fold-local preprocessing and feature learning

## Experiments

| Route | Layer 1 | Layer 2 |
|---|---:|---:|
| IGH-only | 381 | 15 |
| Age+Sex+IGH | 384 | 18 |
| Total | 765 | 33 |

Total: 798 configurations, 79,800 outer holdout evaluations.

Clinical Layer 2 includes Age+Sex+3-mer-only controls.
F1 uses Top500 unweighted CDR3 AA 3-mer.
The enriched-reference dictionary is disabled.

## Frozen split SHA256 prefixes

- Total: 455eba17c74a
- PBMC: 4726d9b48301
- Buffy coat: da1e598559dc

Full assignment files and hashes are maintained in the local
experiment data directory.

## Main findings

Mean ROC-AUC of Age+Sex baseline:

- Total: 0.7218
- PBMC: 0.6680
- Buffy coat: 0.7815

The best Total static clinical combination was S007_PW
(mean ROC-AUC 0.7287).

Static IGH features showed limited additional predictive value
after clinical adjustment. Adding Top500 unweighted 3-mer
did not improve the evaluated clinical models.

These results are exploratory. Selection among many candidates
on repeated holdout scores can introduce selection bias.
They are not independent external-validation estimates.

## Repository scope

This freeze contains source code, experiment configurations,
and documentation.

Raw repertoire data, private metadata, frozen assignment CSVs,
model predictions, large analysis outputs, and run logs are not
included in this Git snapshot.
