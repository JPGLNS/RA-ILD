# RA/RA-ILD IGH Framework V2

**Batch 01 status: modular foundation and historical baseline compatibility**

This directory contains the independent IGH namespace migrated from the
validated TRB framework. Batch 01 establishes the reusable scientific engine,
configuration layer, historical IGH baseline configuration, regression runners,
unit tests, and output isolation.

## Batch 01 scope

Installed:

- `IGH/set/configs/igh_baseline_m2_v1.yaml`
- `IGH/set/src/ra_ild_igh/`
- `IGH/set/scripts_v2/`
- `IGH/set/tests/`
- `IGH/set/README_V2.md`

The original `IGH/set/scripts/`, historical train/test resources, historical
model bundle, and independent-test outputs are not modified.

## Scientific contracts retained

- independent test is not read by the nested-CV engine;
- public references are built from the current training partition only;
- samples used for fitting receive exact leave-one-out public features;
- validation samples use only the corresponding training reference;
- category encoding, zero-variance filtering, and scaling are fitted on the
  current training partition;
- alpha/lambda selection uses inner OOF predictions;
- the classification threshold comes from selected-candidate inner OOF
  predictions;
- positive class is `ILD`.

## IGH-specific contracts

- receptor must be `IGH`;
- runtime paths must remain under `IGH/`;
- clone-table suffix is `_IGH-without-DJ_CDR3_AA_clone_table.csv`;
- static manifest groups use the `igh_` prefix;
- historical IGH static candidate count is 1083 for regression only;
- somatic-hypermutation NT-to-AA aggregation remains in the original
  `build_01_AA_clone_table.py` and is covered by a synthetic Batch 01 test.

## Historical baseline

The baseline YAML records the already completed 120/49 IGH analysis. These
values are provenance and regression expectations, not defaults for a future
full-cohort repeated-holdout scheme.

## Validation

```bash
python3 IGH/set/scripts_v2/check_batch01_igh_installation.py \
  --repository-root /data/users/chenhaisheng/RA-ILD

python3 -m unittest discover -s IGH/set/tests -p 'test_*.py' -v

python3 IGH/set/scripts_v2/validate_experiment_config.py \
  --config IGH/set/configs/igh_baseline_m2_v1.yaml \
  --check-paths
```

Historical quick regression:

```bash
python3 IGH/set/scripts_v2/run_all_checks.py \
  --config IGH/set/configs/igh_baseline_m2_v1.yaml \
  --quick \
  --skip-outer-orchestration
```

Batch 01 does not generate new outer tasks or new model results.
