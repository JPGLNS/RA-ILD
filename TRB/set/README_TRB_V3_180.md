# TRB V3 Complete Cohort (180 samples)

This is an additive input-data adaptation. The original 174-sample data, split
banks, feature definitions, experiment schemes, and results are not overwritten.

## Frozen cohort contract

- Complete metadata: `TRB/metadata_TCR_with_batch.csv`
- Raw NT input directory: `TRB/完整数据/origin_result`
- Raw suffix: `_TRB_CDR3_NT_frequency_error_correct.csv`
- Data-version root: `TRB/set/data_versions/trb_v3_180`
- Cohort: 180 total (RA 111, ILD 69)
- Material: PBMC 116 (RA 69, ILD 47); Buffycoat 64 (RA 42, ILD 22)

## 1. Audit input inventory

Read-only by default:

```bash
python3 TRB/set/scripts_v2/audit_trb_v3_180_inputs.py \
  --repo /data/users/chenhaisheng/RA-ILD
```

After the audit passes, materialize normalized metadata copies:

```bash
python3 TRB/set/scripts_v2/audit_trb_v3_180_inputs.py \
  --repo /data/users/chenhaisheng/RA-ILD \
  --mode write-metadata
```

This writes only three metadata copies under
`TRB/set/data_versions/trb_v3_180/metadata/`. It does not copy or modify raw
repertoire files.

## 2. Generate the three repeated-holdout split banks

Run dry-runs first:

```bash
python3 TRB/set/scripts_v2/generate_repeated_holdout_splits.py --spec TRB/set/configs/repeated_holdout_splits/ra_ild_repeat100_v3_180.yaml --dry-run
python3 TRB/set/scripts_v2/generate_repeated_holdout_splits.py --spec TRB/set/configs/repeated_holdout_splits/ra_ild_pbmc_repeat100_v3_180.yaml --dry-run
python3 TRB/set/scripts_v2/generate_repeated_holdout_splits.py --spec TRB/set/configs/repeated_holdout_splits/ra_ild_buffycoat_repeat100_v3_180.yaml --dry-run
```

Then run the same commands without `--dry-run` to create and freeze the banks.

Split design follows the existing TRB repeated-holdout method:
- Total: 126/54; exact strata = cohort + batch.
- PBMC: 81/35; exact strata = cohort + batch.
- Buffycoat: 45/19; exact strata = cohort; batch and sex are balance variables.

The Buffycoat exception is required because ILD x batch4 has one sample.

## 3. Materialize the repeat-1 compatibility partition

The original Step02/Step03/Step04 builders use train/test input directories.
For this purpose only, repeat 1 of the Total split bank is materialized as a
126/54 compatibility partition:

```bash
python3 TRB/set/scripts_v2/audit_trb_v3_180_inputs.py \
  --repo /data/users/chenhaisheng/RA-ILD \
  --mode materialize-repeat1 \
  --split-assignments TRB/set/split_sets/ra_ild_repeat100_v3_180/repeated_holdout_assignments.csv
```

This compatibility partition is not a new independent validation design. The
100 frozen repeated-holdout assignments remain the actual outer evaluation
design used by the repeated-holdout training bundle.

## 4. Build the new repertoire data version

Step01:

```bash
python3 TRB/set/scripts/build_01_AA_clone_table.py \
  --input-dir TRB/完整数据/origin_result \
  --output-dir TRB/set/data_versions/trb_v3_180/result/01_AA_clone_table \
  --metadata TRB/set/data_versions/trb_v3_180/metadata/metadata_total.csv \
  --id-col libraryid \
  --input-suffix _TRB_CDR3_NT_frequency_error_correct.csv \
  --dry-run
```

After QC review, rerun without `--dry-run`.

Step01b Top10000:

```bash
python3 TRB/set/scripts/build_01b_AA_clone_table_topk.py \
  --input-dir TRB/set/data_versions/trb_v3_180/result/01_AA_clone_table \
  --output-dir TRB/set/data_versions/trb_v3_180/result/01b_AA_clone_table_top10000 \
  --top-k 10000 \
  --dry-run
```

After QC review, rerun without `--dry-run`.

Use the V3 registry and plan in the source-aware Step02 builder:
- Registry: `TRB/set/configs/feature_framework/trb_feature_registry_v3_180.yaml`
- Plan: `TRB/set/configs/feature_framework/trb_feature_plan_v3_180.yaml`

## 5. Feature matrices and subsets

The compatibility metadata files are under
`TRB/set/data_versions/trb_v3_180/ml_compat/total/`.

Build Step02 train/test static-core inputs, then Step03 public inputs and Step04
feature matrices in the corresponding V3 data-version directories. Step04
expected sample counts are 126 and 54; its existing feature-column contracts
remain unchanged unless the real Step02 output demonstrates a schema change.

Once Total Step04 outputs exist, create the PBMC and Buffycoat matrix subsets
using the two V3 material-subset configs:
- `TRB/set/configs/material_subsets/trb_v3_180_pbmc_only.yaml`
- `TRB/set/configs/material_subsets/trb_v3_180_buffycoat_only.yaml`

## 6. Prepare the repeated-holdout training bundles

Dry-run each V3 training bundle before writing:

```bash
python3 TRB/set/scripts_v2/prepare_repeated_holdout_training.py --spec TRB/set/configs/repeated_holdout_training/trb_v3_180_total_training_v1.yaml --dry-run
python3 TRB/set/scripts_v2/prepare_repeated_holdout_training.py --spec TRB/set/configs/repeated_holdout_training/trb_v3_180_pbmc_training_v1.yaml --dry-run
python3 TRB/set/scripts_v2/prepare_repeated_holdout_training.py --spec TRB/set/configs/repeated_holdout_training/trb_v3_180_buffycoat_training_v1.yaml --dry-run
```

Do not start model training until all three bundles pass, their frozen markers
and input hashes have been reviewed, and the M0-M3 / five-algorithm experiment
configuration paths have been adapted to the V3 generated schemes.

## Safety

- No original 174-sample files or configs are overwritten.
- No raw NT files or generated result matrices are committed to Git.
- The new split banks are separate from the original split banks.
- The read-only audit is the first required gate.
- No model training is launched by these configs or scripts.
