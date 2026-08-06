# TRB Upgrade Batch 17 — Enrich-Dictionary Threshold Grid

## Purpose

Batch 17 extends the installed Batch 16 repeated five-fold workflow with a
configurable prevalence-threshold grid and Linux process parallelism.

The default grid is:

```text
T:     10% to 30%, step 2%
Delta:  5% to 15%, step 1%
Rule: Delta <= T
Valid pairs: 112
```

The analysis is performed independently for:

```text
total
PBMC
buffycoat
```

No R script is installed or invoked.

## Validation design

For every scope and repeat:

1. Generate one optimized stratified five-fold partition.
2. Reuse this exact partition for all valid threshold pairs.
3. For each fold, build RA and RA-ILD dictionaries from the four training folds.
4. Score the held-out fold with the fold-specific dictionaries.
5. Pool all five out-of-fold parts and calculate one AUC for every threshold
   pair and metric.

Therefore 100 repeats produce 100 primary pooled-OOF AUC values per threshold
pair and metric. The 500 fold-level AUC values are retained only as diagnostics.

## Parallelism

Use:

```text
--workers N
```

Rules:

```text
default: 1
minimum: 1
maximum: 16
```

Each worker processes one complete repeat, including all five folds and all
valid threshold pairs. Serial and parallel results are deterministic and are
covered by focused regression tests.

Values above 16 are rejected. In particular, `--workers 18` is invalid because
the requested interface explicitly caps the worker count at 16.

## Installed files

```text
TRB/set/src/ra_ild_trb/enrich_dictionary_grid.py
TRB/set/scripts_v2/run_enrich_dictionary_threshold_grid.py
TRB/set/scripts_v2/run_enrich_dictionary_threshold_grid.sh
TRB/set/scripts_v2/test_batch17_enrich_dictionary_threshold_grid.py
TRB/set/scripts_v2/README_upgrade_batch17_enrich_dictionary_threshold_grid.md
TRB/set/scripts_v2/BATCH17_ENRICH_DICTIONARY_THRESHOLD_GRID_MANIFEST.json
TRB/set/configs/enrich_dictionary_cv/enrich_dictionary_threshold_grid_v1.json
```

Batch 16 single-threshold files are not replaced.

## Installation

```bash
cd /path/to/TRB_enrich_dictionary_threshold_grid_batch17_v1.0.0

bash apply_upgrade.sh /data/users/chenhaisheng/RA-ILD
```

## Smoke test

```bash
cd /data/users/chenhaisheng/RA-ILD

bash TRB/set/scripts_v2/run_enrich_dictionary_threshold_grid.sh \
  /data/users/chenhaisheng/RA-ILD \
  --smoke \
  --workers 2 \
  --overwrite
```

The smoke test uses one repeat, at most 20 split candidates, and three valid
threshold pairs.

## Requested formal run

The maximum supported worker count is 16, so the formal command is:

```bash
cd /data/users/chenhaisheng/RA-ILD

nohup bash TRB/set/scripts_v2/run_enrich_dictionary_threshold_grid.sh \
  /data/users/chenhaisheng/RA-ILD \
  --scopes total,pbmc,buffycoat \
  --folds 5 \
  --repeats 100 \
  --candidates 300 \
  --seed 20260711 \
  --threshold-start 10 \
  --threshold-stop 30 \
  --threshold-step 2 \
  --delta-start 5 \
  --delta-stop 15 \
  --delta-step 1 \
  --workers 16 \
  --overwrite \
  > TRB/result/enrich_dictionary_threshold_grid_batch17_v1.0.0.log 2>&1 &
```

Progress is printed as completed repeats within each scope.

## Main outputs

```text
TRB/result/enrich_dictionary_threshold_grid_cv_v1/
├── threshold_grid.csv
├── threshold_grid_run_manifest.json
├── threshold_grid_summary.md
├── total/
├── pbmc/
└── buffycoat/
```

Per scope:

```text
grid_auc_by_repeat_<scope>.csv.gz
grid_auc_by_fold_diagnostic_<scope>.csv.gz
grid_metric_threshold_summary_<scope>.csv
grid_best_thresholds_by_metric_<scope>.csv
grid_sample_averaged_scores_<scope>.csv.gz
grid_fold_dictionary_summary_<scope>.csv.gz
grid_dictionary_stability_<scope>.csv.gz
grid_dictionary_stability_summary_<scope>.csv
grid_richness_correlation_<scope>.csv
```

Plots are written under:

```text
TRB/gradient/enrich_dictionary_threshold_grid_cv_v1/<scope>/
```

Each scope receives eight primary-metric AUC heatmaps and three best-threshold
summary figures.

## Important statistical interpretation

The threshold grid is an exploratory screening analysis. Selecting the threshold
pair with the highest repeated-CV AUC and quoting that same AUC as final
performance introduces threshold-selection optimism. A final unbiased claim
requires nested cross-validation or an untouched validation set in which the
threshold is selected using training data only.
