# TRB V3 179 Layered Feature Experiments

This branch adds a free Core83 feature-combination search and its incremental
feature follow-up. It does not modify frozen V3 data, split banks, training
bundles, or completed M0-M3 results.

## Scientific contract

- Cohorts: Total (179), PBMC (116), Buffycoat (63).
- Algorithm for this free feature-combination search: **Elastic Net only**.
- The separate M0-M3 hierarchical model comparison remains a five-algorithm
  study; this branch does not change that study's algorithm design.
- Layer 1: all 127 non-empty combinations of the seven Core83 groups.
- Layer 2: Top 5 combinations selected separately within each cohort; each
  receives four paired states: F0 static, F1 3-mer, F2 RA-hit, F3 combined.
- RA-hit uses the established 20% prevalence threshold and 10 percentage-point
  between-class prevalence difference.
- Outer evaluation: 100 frozen repeated holdouts, outer fold 1; inner CV: 5-fold.

## Expected experiment size

- Layer 1: 3 cohorts × 127 combinations × Elastic Net = **381 configs**.
- Layer 2: 3 cohorts × Top 5 × 4 paired states = **60 configs**.
- Total: **441 configs**.

## Layer 1

Generate configs (dry-run first):

```bash
python3 TRB/set/scripts_v2/prepare_layer1_unified_configs.py
python3 TRB/set/scripts_v2/prepare_layer1_unified_configs.py --apply
python3 TRB/set/scripts_v2/audit_layer1_unified_configs.py
```

Age + sex remain fixed adjustment covariates; only the selected Core83 groups
vary. 3-mer and RA-hit are disabled in Layer 1. The generator writes an
Elastic-Net-only manifest and rejects overwriting existing configs.

Prepare the shared V3 Top10000 Step02 static-core cache once with the dedicated
all-Core83 cache-preparation config. Do NOT use an individual Layer-1 config
such as L1_001_D: Step02 follows the config's selected static groups and would
then materialize only a partial feature matrix.

```bash
python3 TRB/set/scripts/run_trb_experiment.py --config TRB/set/configs/layered_feature_experiments/trb179_top10000_core83_cache_prep.yaml --train-metadata TRB/set/data_versions/trb_v3_179_top10000/ml_compat/total/metadata_train_70.csv --test-metadata TRB/set/data_versions/trb_v3_179_top10000/ml_compat/total/metadata_test_30.csv --registry TRB/set/configs/feature_framework/trb_feature_registry_v3_179_top10000.yaml --prepare-only
```

Then verify all 83 Core83 columns, exact train/test sample coverage and source
provenance:

```bash
python3 TRB/set/scripts_v2/audit_layer1_unified_configs.py --check-cache
```

The Step02 train/test directories are only a storage partition for the 179
per-sample static features. Layer-1's cohort-specific frozen split bank defines
the actual train/holdout membership in every repeat.

Run one combination for Repeat001 acceptance:

```bash
python3 TRB/set/scripts_v2/run_layer1_unified_batch.py \
  --cohort total --combination-id L1_001_D \
  --repeat 1 --max-experiments 1 --execute
```

Smoke output is stored in a `smoke_repeat_001` child directory under each
combination. The full100 outputs are written into the combination directory
itself; the batch runner and ML entry point explicitly preserve and ignore
`smoke_repeat_*` child directories when checking for output collisions.
After acceptance, run the full Layer 1 in bounded batches. For example:

```bash
python3 TRB/set/scripts_v2/run_layer1_unified_batch.py \
  --cohort total --max-experiments 2 --execute
```

The runner rejects manifests containing non-Elastic-Net algorithms and checks
that the V3 registry and shared Step02 caches exist.

After all 381 combinations have exactly 100 repeats:

```bash
python3 TRB/set/scripts_v2/aggregate_layer1_unified_results.py
python3 TRB/set/scripts_v2/aggregate_layer1_unified_results.py --apply
```

Promotion is strict: each of the three cohorts must have all 127 combinations
complete with 100 repeats. Top 5 is selected separately within each cohort.

## Layer 2

Generate four paired states per promoted combination:

```bash
python3 TRB/set/scripts_v2/prepare_layer2_unified_configs.py
python3 TRB/set/scripts_v2/prepare_layer2_unified_configs.py --apply
```

- F0: selected Core83 groups only
- F1: same Core83 + both weighted and unweighted fold-local Top-500 3-mers
- F2: same Core83 + `RA_dict_hit_rate` from the enriched dictionary
- F3: same Core83 + both 3-mer representations + RA-hit

RA-hit remains fixed at T=20% and Delta=10 percentage points. Each Layer-2
state uses the exact same cohort split bank as its corresponding Layer-1
combination. Layer-2 config generation rejects non-Elastic-Net promotions and
requires exactly Top 5 from each of the three cohorts.

Run one paired combination through Repeat001 before full Layer 2:

```bash
python3 TRB/set/scripts_v2/run_layer2_unified_batch.py \
  --cohort total --feature-state F3_3mer_ra_hit \
  --limit 1 --repeat 1 --max-experiments 1 --execute
```

After all 60 paired states complete 100 repeats:

```bash
python3 TRB/set/scripts_v2/aggregate_layer2_incremental_results.py
python3 TRB/set/scripts_v2/aggregate_layer2_incremental_results.py --apply
```

The final paired report contains mean performance by state and per-repeat
paired deltas (F1/F2/F3 minus F0) for ROC-AUC and PR-AUC, including 95% t
confidence intervals and the fraction of repeats with positive improvement.

## Execution boundary

The code and manifests are committed, but training is not launched by this
development change. Run unit tests, syntax checks, input audits and Repeat001
acceptance on neo3 before any Repeat100 batch. Do not reuse old five-algorithm
Layer-1 manifests; regenerate the new 381-entry Elastic-Net-only manifest.
