# TRB Upgrade Batch 12 — Configurable Material-Subset Cohorts

## 1. Purpose

Batch 12 adds reproducible material-specific analysis populations for the current
RA/RA-ILD TRB framework:

- PBMC-only: 111 patients, RA 67 and ILD 44;
- buffercoat-only: 63 patients, RA 41 and ILD 22.

The upgrade is **fully additive**. It does not replace or edit the existing:

- repeated-holdout split engine;
- repeated-holdout training-bundle engine;
- nested CV and Elastic Net implementation;
- dynamic public-reference implementation;
- repeat-specific 3-mer implementation;
- aggregation or Batch 11 visualization implementation.

Instead, Batch 12 prepares ordinary filtered metadata and old-train/old-test
matrix files that already satisfy the existing Batch 03/04 input contracts.
Consequently, all existing full-cohort YAML files and frozen outputs remain
unchanged.

The GitHub version reviewed while preparing this package was:

```text
repository: JPGLNS/RA-ILD
branch: feature/trb-scheme-upgrade-batch01
commit: 83ba7e4deae49cd1e4607500bb358052be1c3b26
```

## 2. Data flow

```text
Existing 174-patient metadata and Step-04 matrices
                    │
                    ▼
          Batch 12 material filter
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
   PBMC-only frozen     buffercoat-only frozen
   source cohort        source cohort
          │                    │
          ▼                    ▼
 Existing Batch 03 repeated-holdout generator
          │                    │
          ▼                    ▼
 Existing Batch 04 training-bundle preparation
          │                    │
          ▼                    ▼
 Existing nested CV / aggregation / visualization
```

Filtering occurs before split generation, sparse-cache construction,
repeat-specific 3-mer selection and dynamic public-reference construction.
Therefore buffercoat samples cannot influence a PBMC-only model and PBMC samples
cannot influence a buffercoat-only model.

## 3. Installed files

```text
TRB/set/src/ra_ild_trb/cohort_subset.py
TRB/set/scripts_v2/prepare_material_subset.py
TRB/set/scripts_v2/test_batch12_material_subsets.py
TRB/set/scripts_v2/README_upgrade_batch12_material_subsets.md
TRB/set/scripts_v2/BATCH12_MATERIAL_SUBSETS_MANIFEST.json

TRB/set/configs/material_subsets/trb_pbmc_only_v1.yaml
TRB/set/configs/material_subsets/trb_buffercoat_only_v1.yaml

TRB/set/configs/repeated_holdout_splits/ra_ild_pbmc_repeat100_v1.yaml
TRB/set/configs/repeated_holdout_splits/ra_ild_buffercoat_repeat100_v1.yaml

TRB/set/configs/repeated_holdout_training/trb_pbmc_repeat100_training_v1.yaml
TRB/set/configs/repeated_holdout_training/trb_buffercoat_repeat100_v1.yaml

TRB/set/configs/schemes/trb_scheme_material_homogeneous_no_clinical_template.yaml
TRB/set/configs/schemes/trb_scheme_pbmc_a1_a6_repeat100_v1.yaml
TRB/set/configs/schemes/trb_scheme_buffercoat_a1_a6_repeat100_v1.yaml
```

No existing repository file is replaced.

## 4. Backward compatibility

Existing full-cohort analysis remains:

```text
174 patients
123 training / 51 native holdout
PBMC + buffercoat
```

Batch 12 does not add required fields to any existing configuration schema. The
new split files continue to use:

```yaml
split_set_version: "1.0"
```

The new training-bundle files continue to use:

```yaml
training_bundle_version: "1.0"
```

They simply point to the Batch 12 filtered source files. Old Scheme 011–013,
existing training bundles and existing result directories are not read or
modified during subset preparation.

## 5. Material-specific split rules

### PBMC-only

Observed distribution:

```text
111 patients
RA  = 67
ILD = 44
```

Every PBMC `cohort × batch` cell contains at least five patients, so the outer
split can retain exact joint stratification:

```yaml
train_size: 78
holdout_size: 33
exact_strata: [cohort, batch]
balance_categorical: [sex]
balance_numeric: [age]
```

### Buffercoat-only

Observed distribution:

```text
63 patients
RA  = 41
ILD = 22
```

`batch4 × ILD` contains one patient. The existing split engine correctly rejects
singleton exact strata, so buffercoat uses:

```yaml
train_size: 44
holdout_size: 19
exact_strata: [cohort]
balance_categorical: [batch, sex]
balance_numeric: [age]
```

Batch is balanced during candidate ranking but is not forced as an exact joint
stratum.

## 6. Material as metadata, not predictor

The `material` column remains in metadata and matrices so every output can be
audited. It is not included in the PBMC-only or buffercoat-only model definitions
because it is constant within each subcohort.

The supplied material-homogeneous base template contains only M1 and M2 and
removes the inherited M3 material-sensitivity model. The supplied A1-A6
feature-ablation schemes also contain no `material` numeric or categorical
predictor.

## 7. Installation

From the extracted package directory:

```bash
cd /path/to/TRB_material_subset_batch12_v1.0.0

bash apply_upgrade.sh \
  /data/users/chenhaisheng/RA-ILD
```

The installer:

1. verifies that the target contains the current split and training-bundle code;
2. copies only the 14 new Batch 12 files;
3. runs `py_compile`;
4. runs the 10 focused tests.

A differing existing Batch 12 file is not overwritten by default. A reviewed
replacement can use:

```bash
bash apply_upgrade.sh \
  /data/users/chenhaisheng/RA-ILD \
  --force
```

`--force` creates timestamped backups before replacement. It still does not
modify any pre-Batch-12 file.

## 8. Focused validation

Run from the repository root:

```bash
cd /data/users/chenhaisheng/RA-ILD

python3 \
  TRB/set/scripts_v2/test_batch12_material_subsets.py
```

Expected:

```text
PASS test_pbmc_subset_realistic_174_by_1090
PASS test_buffercoat_subset_realistic_counts
PASS test_case_insensitive_and_whitespace_filter
PASS test_unknown_material_is_rejected
PASS test_matrix_metadata_mismatch_is_rejected
PASS test_frozen_marker_hash_verification
PASS test_frozen_subset_is_immutable
PASS test_existing_v1_contracts_are_reused_unchanged
PASS test_material_specific_stratification_rules
PASS test_material_is_not_a_subset_model_predictor
Batch 12 focused tests: 10/10 PASS
```

The focused test constructs a synthetic 174-row × 1090-column matrix matching the
observed full-cohort dimensions and material/cohort/batch counts.

The package was also checked with the Python 3.8 parser.

## 9. PBMC-only workflow

### Step 1 — inspect without writing

```bash
cd /data/users/chenhaisheng/RA-ILD

python3 \
  TRB/set/scripts_v2/prepare_material_subset.py \
  --spec TRB/set/configs/material_subsets/trb_pbmc_only_v1.yaml \
  --dry-run
```

Confirm:

```text
Selected samples: 111
RA: 67
ILD: 44
```

### Step 2 — freeze PBMC source inputs

```bash
python3 \
  TRB/set/scripts_v2/prepare_material_subset.py \
  --spec TRB/set/configs/material_subsets/trb_pbmc_only_v1.yaml
```

Output:

```text
TRB/set/cohort_subsets/trb_pbmc_only_v1/
├── subset_metadata.csv
├── subset_train_base_matrix.csv.gz
├── subset_test_final_matrix.csv.gz
├── subset_sample_audit.csv
├── subset_summary.json
├── subset_resolved_spec.yaml
└── SUBSET_FROZEN.json
```

Verify hashes:

```bash
python3 \
  TRB/set/scripts_v2/prepare_material_subset.py \
  --verify-marker \
  TRB/set/cohort_subsets/trb_pbmc_only_v1/SUBSET_FROZEN.json
```

### Step 3 — generate and freeze 100 PBMC holdouts

Dry run:

```bash
python3 \
  TRB/set/scripts_v2/generate_repeated_holdout_splits.py \
  --spec TRB/set/configs/repeated_holdout_splits/ra_ild_pbmc_repeat100_v1.yaml \
  --dry-run
```

Official write:

```bash
python3 \
  TRB/set/scripts_v2/generate_repeated_holdout_splits.py \
  --spec TRB/set/configs/repeated_holdout_splits/ra_ild_pbmc_repeat100_v1.yaml
```

### Step 4 — prepare the PBMC training bundle

Dry run:

```bash
python3 \
  TRB/set/scripts_v2/prepare_repeated_holdout_training.py \
  --spec TRB/set/configs/repeated_holdout_training/trb_pbmc_repeat100_training_v1.yaml \
  --dry-run
```

Official write:

```bash
python3 \
  TRB/set/scripts_v2/prepare_repeated_holdout_training.py \
  --spec TRB/set/configs/repeated_holdout_training/trb_pbmc_repeat100_training_v1.yaml
```

The generated PBMC base experiment is:

```text
TRB/set/experiments/trb_scheme_pbmc_repeat100_no_clinical/
```

### Step 5 — prepare the PBMC A1–A6 feature scheme

```bash
python3 \
  TRB/set/scripts_v2/prepare_feature_ablation_scheme.py \
  --scheme TRB/set/configs/schemes/trb_scheme_pbmc_a1_a6_repeat100_v1.yaml
```

### Step 6 — run and aggregate

```bash
python3 \
  TRB/set/scripts_v2/run_all_outer_tasks.py \
  --config TRB/set/experiments/trb_scheme_pbmc_a1_a6_repeat100_v1/00_config/resolved_config.yaml \
  --status-only

python3 \
  TRB/set/scripts_v2/run_all_outer_tasks.py \
  --config TRB/set/experiments/trb_scheme_pbmc_a1_a6_repeat100_v1/00_config/resolved_config.yaml \
  --workers 2 \
  --execute

python3 \
  TRB/set/scripts_v2/aggregate_repeated_holdout_results.py \
  --config TRB/set/experiments/trb_scheme_pbmc_a1_a6_repeat100_v1/00_config/resolved_config.yaml
```

Batch 11 visualization can then read the PBMC experiment's `03_summary`
directory without modification.

## 10. Buffercoat-only workflow

Use the same sequence with these replacements:

```text
Material subset:
TRB/set/configs/material_subsets/trb_buffercoat_only_v1.yaml

Split set:
TRB/set/configs/repeated_holdout_splits/ra_ild_buffercoat_repeat100_v1.yaml

Training bundle:
TRB/set/configs/repeated_holdout_training/trb_buffercoat_repeat100_training_v1.yaml

Feature scheme:
TRB/set/configs/schemes/trb_scheme_buffercoat_a1_a6_repeat100_v1.yaml

Resolved feature experiment:
TRB/set/experiments/trb_scheme_buffercoat_a1_a6_repeat100_v1/00_config/resolved_config.yaml
```

Run the subset preparation and split-generation dry runs before writing frozen
outputs. Pay particular attention to the batch-balance audit because buffercoat
contains only three batch4 patients.

## 11. Safety rules

Batch 12 rejects:

- an unobserved material value;
- missing or duplicated sample IDs;
- multiple samples for one patient when one-to-one is required;
- metadata/matrix cohort disagreement;
- metadata/matrix material disagreement;
- incomplete or extra source-matrix sample sets;
- unexpected PBMC or buffercoat cohort counts;
- replacement of a frozen subset;
- changed files recorded in `SUBSET_FROZEN.json`.

A partial, non-frozen output directory can only be replaced with
`--replace-incomplete` after review.

## 12. Git staging

From the repository root:

```bash
git add \
  TRB/set/src/ra_ild_trb/cohort_subset.py \
  TRB/set/scripts_v2/prepare_material_subset.py \
  TRB/set/scripts_v2/test_batch12_material_subsets.py \
  TRB/set/scripts_v2/README_upgrade_batch12_material_subsets.md \
  TRB/set/scripts_v2/BATCH12_MATERIAL_SUBSETS_MANIFEST.json \
  TRB/set/configs/material_subsets/trb_pbmc_only_v1.yaml \
  TRB/set/configs/material_subsets/trb_buffercoat_only_v1.yaml \
  TRB/set/configs/repeated_holdout_splits/ra_ild_pbmc_repeat100_v1.yaml \
  TRB/set/configs/repeated_holdout_splits/ra_ild_buffercoat_repeat100_v1.yaml \
  TRB/set/configs/repeated_holdout_training/trb_pbmc_repeat100_training_v1.yaml \
  TRB/set/configs/repeated_holdout_training/trb_buffercoat_repeat100_training_v1.yaml \
  TRB/set/configs/schemes/trb_scheme_material_homogeneous_no_clinical_template.yaml \
  TRB/set/configs/schemes/trb_scheme_pbmc_a1_a6_repeat100_v1.yaml \
  TRB/set/configs/schemes/trb_scheme_buffercoat_a1_a6_repeat100_v1.yaml
```

Suggested commit:

```text
Add configurable material-specific TRB cohorts
```

Generated cohort subsets, split sets, training bundles and experiments remain
server-side analysis artifacts and should not be added to Git.
