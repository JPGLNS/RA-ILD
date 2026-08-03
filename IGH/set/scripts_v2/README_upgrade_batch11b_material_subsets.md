# IGH Upgrade Batch 11B — Frozen PBMC and Buffercoat Cohorts

## Scope

Batch 11B converts the read-only Batch 11A audit into two reproducible source
cohorts. Observed IGH counts are used directly:

| cohort | samples | RA | ILD | train | holdout | exact strata |
|---|---:|---:|---:|---:|---:|---|
| PBMC | 108 | 63 | 45 | 76 | 32 | cohort + batch |
| buffercoat | 61 | 39 | 22 | 43 | 18 | cohort |

Buffercoat does not use `cohort + batch` as exact strata because Batch 11A found
two singleton cells. Batch and sex are balanced instead. For PBMC, batch is
already an exact-strata variable, so only sex remains in `balance_categorical`;
the split engine forbids the same variable appearing in both lists.

Filtering occurs before split creation, public-reference construction,
repeat-specific 3-mer selection, and model fitting. Material remains auditable
metadata but is not a predictor within a one-material cohort.

## Installed files

Fourteen source/config/test files are added. Existing full-cohort files and
frozen outputs are not replaced.

## Focused test

```bash
python3 IGH/set/scripts_v2/test_batch11b_material_subsets.py
```

Expected: `IGH_BATCH11B_ACCEPTANCE_PASS`.

## Step 1 — PBMC subset

```bash
python3 IGH/set/scripts_v2/prepare_material_subset.py \
  --spec IGH/set/configs/material_subsets/igh_pbmc_only_v1.yaml \
  --repository-root /data/users/chenhaisheng/RA-ILD \
  --dry-run

python3 IGH/set/scripts_v2/prepare_material_subset.py \
  --spec IGH/set/configs/material_subsets/igh_pbmc_only_v1.yaml \
  --repository-root /data/users/chenhaisheng/RA-ILD
```

Verify:

```bash
python3 IGH/set/scripts_v2/prepare_material_subset.py \
  --repository-root /data/users/chenhaisheng/RA-ILD \
  --verify-marker IGH/set/cohort_subsets/igh_pbmc_only_v1/SUBSET_FROZEN.json
```

Repeat with `igh_buffercoat_only_v1.yaml`.

## Step 2 — repeated holdouts

Run PBMC dry-run first, then write:

```bash
python3 IGH/set/scripts_v2/generate_repeated_holdout_splits.py \
  --spec IGH/set/configs/split_sets/igh_ra_ild_pbmc_repeat100_v1.yaml \
  --dry-run

python3 IGH/set/scripts_v2/generate_repeated_holdout_splits.py \
  --spec IGH/set/configs/split_sets/igh_ra_ild_pbmc_repeat100_v1.yaml
```

Use `igh_ra_ild_buffercoat_repeat100_v1.yaml` for buffercoat.

## Step 3 — training bundles

```bash
python3 IGH/set/scripts_v2/prepare_repeated_holdout_training.py \
  --spec IGH/set/configs/repeated_holdout_training/igh_pbmc_repeat100_training_v1.yaml \
  --dry-run

python3 IGH/set/scripts_v2/prepare_repeated_holdout_training.py \
  --spec IGH/set/configs/repeated_holdout_training/igh_pbmc_repeat100_training_v1.yaml
```

Use the buffercoat training YAML for the second cohort.

## Step 4 — optional A1–A6 feature schemes

```bash
python3 IGH/set/scripts_v2/prepare_feature_ablation_scheme.py \
  --scheme IGH/set/configs/schemes/igh_scheme_pbmc_a1_a6_repeat100_v1.yaml
```

The equivalent buffercoat YAML is also included. Preparing a scheme does not
select a final model automatically.

## Runtime outputs

Do not commit:

- `IGH/set/cohort_subsets/`
- `IGH/set/split_sets/`
- `IGH/set/training_bundles/`
- `IGH/set/experiments/`
- installer audit JSON or backups

Commit only the 14 files listed by the manifest/staging command supplied with
the package.
