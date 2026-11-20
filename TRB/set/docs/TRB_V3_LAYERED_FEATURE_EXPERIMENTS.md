# TRB V3 179 Layered Feature Experiments

This branch is an additive development branch based on
`feature/trb-v3-179-igh-aligned-hierarchy`. It does not modify frozen V3 data,
split banks, training bundles, or completed M0-M3 results.

## Scientific contract

- Cohorts: Total (179), PBMC (116), Buffycoat (63).
- Algorithms: Elastic Net, Ridge, Lasso, Linear SVM, XGBoost.
- Layer 1: all 127 non-empty combinations of the seven Core83 groups.
- Layer 2: top-N combinations are selected separately within each cohort ×
  algorithm; each receives four paired states: F0 static, F1 3-mer, F2 RA-hit,
  F3 3-mer + RA-hit.
- RA-hit uses the established 20% prevalence threshold and 10 percentage-point
  between-class prevalence difference.
- Outer evaluation target: 100 frozen repeated holdouts, fold 1.

## Current implementation slice

`TRB/set/src/ra_ild_trb/layered_experiments.py` defines the deterministic
127-combination search space, strict 100-repeat promotion contract, per-cohort
and per-algorithm top-N ranking, and paired Layer-2 manifest.

`TRB/set/scripts_v2/prepare_layered_feature_experiments.py` writes planning
manifests. It never starts training. Run a dry-run first:

```bash
python3 TRB/set/scripts_v2/prepare_layered_feature_experiments.py
```

Write Layer 1 manifest:

```bash
python3 TRB/set/scripts_v2/prepare_layered_feature_experiments.py --apply
```

After Layer 1 is trained and aggregated with exactly 100 outer repeats per
combination, generate promotion and Layer-2 manifests:

```bash
python3 TRB/set/scripts_v2/prepare_layered_feature_experiments.py \
  --layer1-results PATH_TO_LAYER1_AGGREGATED_METRICS.csv --top-n 5 --apply
```

## Important status

This commit establishes the experiment-space and promotion contract only. It
does **not** yet contain the Layer-1/Layer-2 model execution adapter, the
feature-group-aware nested-CV model specification, or the RA-hit fold-local
integration into all five engines. Do not launch Repeat100 based on these
planning manifests alone. Those execution components must pass Repeat001
acceptance and leakage audits first.

## Layer-1 executable pipeline

Generate the 1,905 YAMLs (dry-run first):

```bash
python3 TRB/set/scripts_v2/prepare_layer1_unified_configs.py
python3 TRB/set/scripts_v2/prepare_layer1_unified_configs.py --apply
```

The generated models keep age + sex as fixed adjustment covariates; only the
selected Core83 groups vary. Layer 1 has 3-mer and RA-hit disabled.

Prepare the shared V3 Top10000 Step02 static-core cache once, using the exact
V3 registry and 179-sample 70:30 compatibility metadata:

```bash
python3 TRB/set/scripts/run_trb_experiment.py \
  --config TRB/set/configs/layered_feature_experiments/generated/layer1/total/elastic_net/trb179_total_elastic_net_l1_001_d.yaml \
  --train-metadata TRB/set/data_versions/trb_v3_179_top10000/ml_compat/total/metadata_train_70.csv \
  --test-metadata TRB/set/data_versions/trb_v3_179_top10000/ml_compat/total/metadata_test_30.csv \
  --registry TRB/set/configs/feature_framework/trb_feature_registry_v3_179_top10000.yaml \
  --prepare-only
```

Run a single-combination Repeat001 acceptance test first:

```bash
python3 TRB/set/scripts_v2/run_layer1_unified_batch.py \
  --cohort total --algorithm elastic_net --combination-id L1_001_D \
  --repeat 1 --max-experiments 1 --execute
```

Smoke output is isolated under `smoke_repeat_001`; it cannot be mistaken for
the full 100-repeat result. After acceptance, run the full Layer 1 in bounded
batches. Example:

```bash
python3 TRB/set/scripts_v2/run_layer1_unified_batch.py \
  --cohort total --algorithm elastic_net --max-experiments 2 --execute
```

Omit cohort/algorithm filters only when deliberately scheduling the full 1,905
configurations. The runner checks that the shared Step02 caches and V3 registry
exist, and records per-combination logs/status.

After all 1,905 combinations have exactly 100 repeats:

```bash
python3 TRB/set/scripts_v2/aggregate_layer1_unified_results.py
python3 TRB/set/scripts_v2/aggregate_layer1_unified_results.py --apply
```

Promotion is strict: each cohort × algorithm must have all 127 combinations
with 100 repeats. Top-N (default 5) is selected separately in each of the 15
cohort × algorithm strata.

## Layer-2 executable pipeline

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
combination.

Run one paired combination through Repeat001 before the full Layer 2:

```bash
python3 TRB/set/scripts_v2/run_layer2_unified_batch.py \
  --cohort total --algorithm elastic_net --feature-state F3_3mer_ra_hit \
  --limit 1 --repeat 1 --max-experiments 1 --execute
```

After all paired states complete 100 repeats:

```bash
python3 TRB/set/scripts_v2/aggregate_layer2_incremental_results.py
python3 TRB/set/scripts_v2/aggregate_layer2_incremental_results.py --apply
```

The final paired report contains mean performance by state and per-repeat
paired deltas (F1/F2/F3 minus F0) for ROC-AUC and PR-AUC, including 95% t
confidence intervals and the fraction of repeats with positive improvement.

## Important execution boundary

The code and manifests are committed, but no training was launched from this
development environment. The repository's data and Step02 caches are on the
user's neo3 server. Run Repeat001 and inspect logs/audits there before any
Repeat100 batch. The 1,905 Layer-1 configurations are a substantial compute
job, especially XGBoost (100 candidates × 5 inner folds × 100 repeats per
combination); use the bounded runner and start with a small pilot.
