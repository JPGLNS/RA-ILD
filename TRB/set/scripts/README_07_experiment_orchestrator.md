# Phase 7 — Final Experiment Orchestrator

`run_trb_experiment.py` is the final user-facing entry point for the TRB feature/model framework.
It reuses the previously validated Phase-2/5/6 components rather than reimplementing feature mathematics or models.

## One-config contract

The same unified YAML controls:

- cohort scope: `total | pbmc | buffycoat`;
- repertoire source: `all | topk`, with arbitrary `top_k`;
- static TCR feature-group composition;
- fold-local AA 3-mer representation and `top_n`;
- enriched-dictionary T/Delta and selected features such as `RA_dict_hit_rate`;
- model: `elastic_net | ridge | lasso | linear_svm | xgboost`;
- repeated-holdout repeats and inner folds;
- workers (1–16).

## What Phase 7 automates

1. Resolve and validate the experiment YAML.
2. Validate the requested `all`/Top-K repertoire source; build Top-K when missing.
3. Render a dynamic feature registry/Phase-2 plan.
4. Reuse or build a source-level Step02 static-core/provenance cache.
5. Call Phase 6 for leakage-controlled repeated-holdout ML.
6. Write an orchestrator manifest/summary with paths and actions.

## Important leakage policy

The shared Step02 cache is **not** a source of model-time 3-mer vocabulary.  Phase 7 intentionally creates only a minimal one-kmer scaffold when it must build Step02 from scratch.  Phase 6 ignores those precomputed 3-mer matrices and rebuilds the configured Top-N vocabulary inside each current training partition.

Likewise, enriched-dictionary features, preprocessing, hyperparameter selection and decision-threshold selection remain Phase-6 fold-local operations.

## Canonical cache/output layout

```text
TRB/result/feature_framework/
├── cache/
│   └── step02/
│       └── <source_id>/
│           ├── train/
│           └── test/
└── experiments/
    └── <experiment_id>/
        ├── 00_resolved/
        ├── 06_unified_ml/
        │   ├── smoke_repeat1/   # when --repeat 1
        │   └── full/            # configured repeat bank
        ├── 07_orchestrator_manifest.json
        └── 07_orchestrator_summary.md
```

PBMC/buffycoat/total experiments can reuse the same full train/test Step02 static-core cache for a given repertoire source; Phase 6 performs scope filtering before modeling.

## Typical commands

Preflight only:

```bash
python3 TRB/set/scripts/run_trb_experiment.py \
  --config path/to/experiment.yaml \
  --dry-run
```

Real-data smoke validation using one frozen repeat:

```bash
python3 TRB/set/scripts/run_trb_experiment.py \
  --config path/to/experiment.yaml \
  --repeat 1 \
  --workers 1
```

Full configured experiment:

```bash
python3 TRB/set/scripts/run_trb_experiment.py \
  --config path/to/experiment.yaml
```

Prepare source/static prerequisites without model fitting:

```bash
python3 TRB/set/scripts/run_trb_experiment.py \
  --config path/to/experiment.yaml \
  --prepare-only
```

Reuse an already validated Step02 pair explicitly:

```bash
python3 TRB/set/scripts/run_trb_experiment.py \
  --config path/to/experiment.yaml \
  --train-step02-dir <train_step02> \
  --test-step02-dir <test_step02> \
  --repeat 1
```

`--rebuild-step02` is intentionally restricted to the canonical Phase-7 cache root.  Arbitrary overrides are never deleted automatically.
