# TRB V3 179-sample Layered Feature Experiment — Freeze Record

Freeze recorded: 2026-11-22. Branch: `feature/trb-v3-179-layer1-layer2`.

## Design and completion

- Three cohorts: total (179), PBMC (116), buffycoat (63).
- One modeling algorithm in the layered search: Elastic Net, with independent tuning within each outer repeat.
- Frozen repeated 70/30 holdouts: 100 per configuration; inner five-fold CV.
- Clinical covariates: age + sex always included.
- Layer 1: seven static Core83 feature groups, all 127 nonempty combinations per cohort = **381 configurations**, each with 100 outer repeats; **381/381 complete**.
- Layer 2: five selected Layer-1 combinations per cohort, each under F0 (static), F1 (+ fold-local top500 3-mer vocabulary with weighted/unweighted representations), F2 (+ fold-local RA-hit), F3 (+ both): **60 configurations**, each with 100 outer repeats; **60/60 complete**.
- The 15 Layer-2 F0 configurations reproduce the corresponding Layer-1 results for all **100/100** repeats.
- Across the two layers there are **426 unique feature/model configurations** (381 Layer 1 + 45 additional Layer 2).

## Best configurations, selected by mean outer ROC-AUC

| Cohort | Clinical + TCR feature combination | Mean ROC-AUC | Mean PR-AUC |
|---|---|---:|---:|
| total | Age + Sex + PW | 0.720079 | 0.621069 |
| pbmc | Age + Sex + D | 0.656467 | 0.585318 |
| buffycoat | Age + Sex + E + PU | 0.865595 | 0.806032 |

Codes: D=diversity; E=clonal expansion; PU=unweighted physicochemical; PW=weighted physicochemical.

Under mean PR-AUC ranking, the best static combinations instead are:
- total: Age + Sex + D + E (ROC-AUC 0.717208, PR-AUC 0.623484).
- pbmc: Age + Sex + PU (ROC-AUC 0.656317, PR-AUC 0.602954).
- buffycoat: Age + Sex + E + PU (ROC-AUC 0.865595, PR-AUC 0.806032).

For all 15 promoted Layer-1 combinations, Layer-2 F1, F2, and F3 showed no improvement in **mean ROC-AUC** relative to its F0 baseline in this dataset/experiment.

## Separate clinical-only M0 comparison (existing five-algorithm branch outputs)

The clinical-only Age + Sex M0 is **independently tuned** in the existing conventional Elastic Net repeated-holdout experiment. The following values were read from its 100 repeat outputs; delta is selected TCR model minus M0:

| Cohort | M0 ROC-AUC | M0 PR-AUC | Best TCR ROC-AUC | Best TCR PR-AUC | Delta ROC-AUC | Delta PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| total | 0.7250 | 0.6303 | 0.7201 | 0.6211 | -0.0049 | -0.0092 |
| pbmc | 0.6779 | 0.6219 | 0.6565 | 0.5853 | -0.0215 | -0.0366 |
| buffycoat | 0.7921 | 0.7211 | 0.8656 | 0.8060 | +0.0735 | +0.0850 |

**Interpretation limits:** These M0 values come from a separate conventional modeling workflow, not the Layer-1 unified runner. Treat differences as descriptive unless repeat-wise holdout identities and validation contracts are explicitly cross-audited. Because the best TCR configurations were selected using the same repeated evaluation outcomes, their apparent performance is subject to selection optimism. The 100 repeated holdouts are not 100 independent patient cohorts.

## Runtime result locations (not stored in Git)

All paths are relative to repository root:

- Layer-1 training: `TRB/result/feature_framework/layered_feature_experiments/layer1/`
- Layer-1 summary: `TRB/result/feature_framework/layered_feature_experiments/layer1_summary/`
- Layer-2 training: `TRB/result/feature_framework/layered_feature_experiments/layer2/`
- Layer-2 summary: `TRB/result/feature_framework/layered_feature_experiments/layer2_summary/`
- 426-model ranking: `TRB/result/feature_framework/layered_feature_experiments/final_feature_selection/`
- Existing five-algorithm M0 results: paths in `TRB/set/configs/schemes/generated/hierarchical_igh_aligned_five_algorithm/FIVE_ALGORITHM_RUN_MANIFEST.json`.

## Freeze and preservation rules

Commit the **generated YAML configurations and the two JSON config manifests** below from the neo3 working tree, in addition to the already versioned training scripts and this record:

- `TRB/set/configs/layered_feature_experiments/generated/layer1/`: 381 YAML + 1 JSON.
- `TRB/set/configs/layered_feature_experiments/generated/layer2/`: 60 YAML + 1 JSON.

These are generated locally and are **not** part of this remote documentation-only commit. Do not declare the Git freeze complete until the generated configurations have been committed and pushed.

Leave voluminous experiment outputs, logs, caches, and predictions on protected storage; do not `git add -f` them. Preserve data, split banks, summary CSVs, and the code revision used for the execution.
