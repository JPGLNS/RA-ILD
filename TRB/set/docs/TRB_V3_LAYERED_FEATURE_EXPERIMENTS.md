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
