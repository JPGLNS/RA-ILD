# Phase 6 — Unified ML + Leakage-Controlled CV

Phase 6 is the unified modeling layer for the TRB Feature Framework. It is
**additive**: Phase 1–5 outputs remain unchanged.

## Statistical design

`validation.repeats` means the number of frozen repeated holdouts to evaluate.
Within each outer-training set, `validation.folds` controls the inner CV used to
select hyperparameters and the decision threshold.

For the current project this preserves the established design:

- outer evaluation: deterministic metadata-balanced repeated holdout;
- inner tuning: optimized stratified 5-fold CV;
- ML positive class: `ILD`; negative class: `RA`;
- `repeats=100` is the usual setting; a smaller value such as `50` may be used
  for computationally expensive algorithms such as XGBoost;
- workers are configurable from 1–16 and default to 1 when omitted.

The existing frozen split bank is selected automatically by cohort scope:

- `total` → `TRB/set/split_sets/ra_ild_repeat100_v1/repeated_holdout_assignments.csv`
- `pbmc` → `TRB/set/split_sets/ra_ild_pbmc_repeat100_v1/repeated_holdout_assignments.csv`
- `buffycoat` → `TRB/set/split_sets/ra_ild_buffercoat_repeat100_v1/repeated_holdout_assignments.csv`

A custom assignment file can be supplied with `validation.split_assignments` or
`--split-assignments`. A requested repeat count must be available in the frozen
assignment bank; for the current standard banks any prefix from 1–100 is valid.

## Leakage contract

Phase 6 deliberately does **not** use a full-TRAIN preselected 3-mer matrix as
input to repeated-holdout CV. Instead, every current training partition rebuilds
all learned components.

### 3-mer

For every inner fold, the vocabulary is fitted only on that fold's training
samples. The outer-final vocabulary is fitted only on outer-training samples.
The ranking reproduces the existing Step02 rule:

1. minimum training-sample count;
2. minimum training prevalence;
3. reject only if both weighted and unweighted variances are below threshold;
4. prevalence descending;
5. weighted + unweighted variance descending;
6. amino-acid 3-mer lexical tie break;
7. keep configured Top-N.

### Enriched dictionary

The enriched dictionary always inherits the main `all / TopK` repertoire source.
Default parameters remain T=20% and Delta=10 percentage points, but are
configurable.

Validation/holdout samples are scored from a dictionary fitted on the current
training samples only. Training samples use **exact leave-one-out dictionary
scores**: each sample is scored from the dictionary obtained after excluding
that sample itself. The optimized implementation is regression-tested against
naive per-sample dictionary refitting.

### Preprocessing

For every current training partition only:

- numeric means and population SDs are fitted;
- zero-variance predictors are removed;
- categorical levels are learned;
- lexicographically first training level is the reference level;
- validation-only categories never change training coding/scaling.

### Model selection

Hyperparameters are selected from pooled inner-OOF predictions. The decision
threshold is then selected on the selected candidate's pooled inner-OOF scores
using Youden J. No outer-holdout information enters tuning or threshold fitting.

## Model backends

### Elastic Net logistic regression

Existing project defaults are retained:

- `alpha/l1_ratio`: 0.1, 0.5, 0.9
- `lambda`: 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30
- solver: `saga`
- penalty: `elasticnet`
- `C = 1 / lambda`
- class weight: balanced

### Ridge logistic regression

Same logistic backend and lambda grid, with `alpha=0` fixed.

### Lasso logistic regression

Same logistic backend and lambda grid, with `alpha=1` fixed.

### Linear SVM

Existing main configuration is retained:

- `LinearSVC`
- L2 penalty
- squared hinge loss
- balanced class weight
- dual auto
- C: 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000

The SVM uses `decision_function` scores; probability-only metrics such as
log-loss and Brier score are reported as unavailable.

### XGBoost

The existing deterministic random-search grid is retained. The standard candidate
bank contains 100 parameter combinations sampled without replacement with seed
20260807. `n_jobs_per_fit=1` is retained so repeat-level parallelism is controlled
by the framework `workers` setting rather than nested parallelism.

## Input contract

Phase 6 takes the **Phase-2 Step02 directories** for the original train/test
partitions. Only `02_core_sample_features.csv` is used as a precomputed model
feature source. Learned 3-mer and enriched-dictionary features are reconstructed
from the configured repertoire source inside each CV partition.

If a Step02 directory contains `02_resolved_feature_source.json`, its recorded
source must match the experiment repertoire source.

## Output files

- `06_outer_metrics.csv`
- `06_outer_predictions.csv.gz`
- `06_inner_tuning.csv.gz`
- `06_selected_hyperparameters.csv`
- `06_feature_explanations.csv.gz`
- `06_feature_fit_audit.csv.gz`
- `06_preprocessing_audit.csv.gz`
- `06_inner_assignment_audit.csv.gz`
- `06_run_manifest.json`
- `06_run_summary.md`

## Recommended validation sequence

1. `--dry-run` to validate source, sample coverage, static feature groups,
   candidate bank, split bank and leakage contract without fitting a model.
2. `--repeat 1 --workers 1` for a complete real-data smoke run.
3. Inspect the single-repeat outputs and audit tables.
4. Run the configured repeated-holdout experiment with the chosen workers.
