# TRB V3 179-sample Top10000 data version

This is an additive analysis version derived from the frozen 180-sample V3
Step01 outputs. It excludes exactly one RA Buffycoat sample that has 5,199
AA clonotypes and therefore cannot satisfy the fixed Top10000 contract.

- Excluded library: `MGI260107R01-86`
- Excluded patient: `B1197705`
- Exclusion reason: `insufficient_AA_clonotypes_for_top10000`
- Analysis cohort: 179 (RA 110, ILD 69)
- PBMC: 116 (RA 69, ILD 47)
- Buffycoat: 63 (RA 41, ILD 22)
- TopK: exactly 10,000 AA clonotypes per retained sample

The original `trb_v3_180` metadata, Step01 tables, splits, feature matrices,
training bundles and results are not modified.

## 1. Prepare metadata and a 179-sample Step01 whitelist

Run from the repository root:

```bash
python3 TRB/set/scripts_v2/prepare_trb_v3_179_top10000.py \
  --repo /data/users/chenhaisheng/RA-ILD --mode prepare --dry-run
```

After reviewing the sample identity and cohort counts:

```bash
python3 TRB/set/scripts_v2/prepare_trb_v3_179_top10000.py \
  --repo /data/users/chenhaisheng/RA-ILD --mode prepare
```

The script writes metadata under
`TRB/set/data_versions/trb_v3_179_top10000/metadata/` and creates symlinks
under `source/01_AA_clone_table/` to the existing V3 180 Step01 tables.
It does not copy, rename, edit or delete any Step01 source table.

## 2. Build exact Top10000 tables

```bash
python3 TRB/set/scripts/build_01b_AA_clone_table_topk.py \
  --input-dir TRB/set/data_versions/trb_v3_179_top10000/source/01_AA_clone_table \
  --output-dir TRB/set/data_versions/trb_v3_179_top10000/result/01b_AA_clone_table_top10000 \
  --top-k 10000 --dry-run
```

Only after the full preflight passes:

```bash
python3 TRB/set/scripts/build_01b_AA_clone_table_topk.py \
  --input-dir TRB/set/data_versions/trb_v3_179_top10000/source/01_AA_clone_table \
  --output-dir TRB/set/data_versions/trb_v3_179_top10000/result/01b_AA_clone_table_top10000 \
  --top-k 10000
```

Audit the result:

```bash
python3 TRB/set/scripts_v2/prepare_trb_v3_179_top10000.py \
  --repo /data/users/chenhaisheng/RA-ILD --mode audit-topk
```

The audit requires exactly 179 sample files, matching summary/QC coverage,
all `qc_pass=True`, exactly 10,000 rows per table, and continuous ranks
1..10000. The low-clonotype sample must be absent.

## 3. Generate Total and Buffycoat repeated-holdout banks

PBMC is unchanged and reuses the existing 116-sample PBMC split bank. Generate
only the two new split banks, after metadata preparation:

```bash
python3 TRB/set/scripts_v2/generate_repeated_holdout_splits.py \
  --spec TRB/set/configs/repeated_holdout_splits/ra_ild_repeat100_v3_179_top10000.yaml --dry-run
python3 TRB/set/scripts_v2/generate_repeated_holdout_splits.py \
  --spec TRB/set/configs/repeated_holdout_splits/ra_ild_buffycoat_repeat100_v3_179_top10000.yaml --dry-run
```

Review balance diagnostics and then run without `--dry-run` to freeze the
100-repeat assignments. Expected per-repeat sizes:

- Total: 125 train / 54 holdout.
- Buffycoat: 44 train / 19 holdout.
- PBMC: existing 81 train / 35 holdout; no regeneration.

## 4. Materialize repeat-1 compatibility metadata

The existing Step02-Step04 train/test builders need a single compatibility
partition. After the Total split bank is frozen:

```bash
python3 TRB/set/scripts_v2/prepare_trb_v3_179_top10000.py \
  --repo /data/users/chenhaisheng/RA-ILD --mode materialize-repeat1 \
  --assignments TRB/set/split_sets/ra_ild_repeat100_v3_179_top10000/repeated_holdout_assignments.csv
```

This writes 125/54 metadata to the V3 179 data-version's `ml_compat/total`,
and matching `train/` and `test/` metadata locations. This is only a
compatibility partition; the 100 frozen split assignments remain the outer
evaluation design.

## 5. Step02-Step04 feature generation

Use the new registry and plan:

- Registry: `TRB/set/configs/feature_framework/trb_feature_registry_v3_179_top10000.yaml`
- Plan: `TRB/set/configs/feature_framework/trb_feature_plan_v3_179_top10000.yaml`

The repertoire source is Top10000. The legacy Step02 numerical kernel remains
unchanged. For canonical train/test metadata, keep train and test outputs
isolated under the new data-version directories. The Step02 TopK source summary
must be the new Step01b summary; its parent full-Step01 summary must be the
matching train/test Step01 summary from the original V3 180 source partition.
Do not use the old global 174-sample summaries.

Build Step03/Step04 with the existing scripts against the new 125/54
compatibility partition and keep all outputs inside
`TRB/set/data_versions/trb_v3_179_top10000/`. Review the resulting feature
manifest and matrix dimensions before subset preparation.

After Total Step04 matrices exist, use:

- `TRB/set/configs/material_subsets/trb_v3_179_top10000_pbmc_only.yaml`
- `TRB/set/configs/material_subsets/trb_v3_179_top10000_buffycoat_only.yaml`

Expected subsets: PBMC 116 and Buffycoat 63.

## 6. Training bundles

Dry-run all three new training bundles only after their source matrices, Top10000
tables and split markers exist:

```bash
python3 TRB/set/scripts_v2/prepare_repeated_holdout_training.py --spec TRB/set/configs/repeated_holdout_training/trb_v3_179_top10000_total_training_v1.yaml --dry-run
python3 TRB/set/scripts_v2/prepare_repeated_holdout_training.py --spec TRB/set/configs/repeated_holdout_training/trb_v3_179_top10000_pbmc_training_v1.yaml --dry-run
python3 TRB/set/scripts_v2/prepare_repeated_holdout_training.py --spec TRB/set/configs/repeated_holdout_training/trb_v3_179_top10000_buffycoat_training_v1.yaml --dry-run
```

Do not launch the five-algorithm / M0-M3 / feature-ablation runs until all
three bundles pass, input hashes are reviewed, and their generated scheme paths
have been connected to the existing experiment definitions.

## Important analysis rules

1. Do not change Top10000 to Top5000 for the primary analysis.
2. Do not include the excluded sample by retaining only 5,199 clones; the fixed-K
   contract requires exactly 10,000 for every included sample.
3. Keep the PBMC split bank unchanged.
4. Rebuild Total and Buffycoat split banks because their sample membership changed.
5. Never overwrite or mutate the original 180-sample version.
6. This branch prepares data and configuration only; it does not start model
   training or write scientific results.
