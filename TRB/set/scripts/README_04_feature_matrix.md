# TRB Feature Framework v1 — Phase 4 Matrix Assembler

## Purpose

Phase 4 converts the Phase-3 feature-module tables into a predictor matrix driven by a Feature Framework YAML plan.
It does **not** recalculate repertoire features, refit the training 3-mer vocabulary, standardize variables, encode categoricals, or fit a model.

Architecture:

```text
metadata + Phase-3 feature modules + feature plan
                    ↓
             Matrix Assembler
                    ↓
04_feature_matrix.csv        predictor matrix only
04_sample_context.csv        cohort / trace / auditing metadata
04_feature_matrix_manifest.csv
04_feature_selection_audit.csv
04_resolved_matrix_plan.json
04_feature_matrix_build_summary.md
```

## Outcome separation

`cohort` is intentionally not placed in `04_feature_matrix.csv`.
The predictor matrix contains only `sample_id` plus plan-selected predictor groups.
The outcome and trace fields are written to `04_sample_context.csv` and can later be joined one-to-one on `sample_id` by the modeling layer.

This separation reduces the risk of accidentally treating an outcome or technical field as a predictor.

## Clinical groups

Metadata-based feature groups are assembled directly from metadata:

- `clinical_age_sex` → `age`, `sex`
- `clinical_material` → `material`

They enter the predictor matrix only when explicitly selected by the feature plan.
Categorical variables remain categorical strings. Numeric variables remain numeric. Encoding/scaling must occur inside model-training folds.

## Repertoire modules

For sample-static and training-vocabulary groups, the assembler reads the Phase-3 module manifest and corresponding module CSV.
It requires:

- exact sample-set agreement with metadata;
- unique sample IDs and feature names;
- exact module-column agreement with the Phase-3 manifest;
- finite numeric repertoire features;
- repertoire-source agreement with the resolved feature plan.

Metadata order is authoritative. Module rows are explicitly reordered to metadata order after exact-coverage validation.

## Reference-column policy

Registry-defined compositional reference columns are dropped by default:

- `weighted_length_other_freq`
- `Y_frequency`
- `weighted_Y_frequency`
- `nonpolar_aa_ratio`
- `weighted_nonpolar_aa_ratio`

The raw Phase-3 modules remain unchanged. Phase 4 only excludes these columns from the assembled predictor matrix.
Use `--keep-reference-columns` only for an explicit sensitivity/debugging analysis.

## Example primary weighted plan

The package provides:

`TRB/set/configs/feature_framework/trb_feature_plan_top10000_primary_weighted_v1.yaml`

It selects:

- age + sex;
- diversity;
- clonal expansion;
- AA length;
- weighted and unweighted AA composition;
- weighted and unweighted physicochemical summaries;
- weighted 3-mer features.

It deliberately excludes QC/depth predictors and unweighted 3-mer features. Change the YAML plan to explore other combinations; do not edit assembler code.

## TRAIN example

```bash
PYTHONPATH=TRB/set/src \
python3 TRB/set/scripts/build_04_feature_matrix.py \
  --metadata TRB/set/train/metadata_train_70.csv \
  --module-dir TRB/set/train/result/feature_framework/top10000/02_sample_level_features/02_feature_modules \
  --plan TRB/set/configs/feature_framework/trb_feature_plan_top10000_primary_weighted_v1.yaml \
  --output-dir TRB/set/train/result/feature_framework/top10000/04_feature_matrix \
  --dry-run
```

Remove `--dry-run` to write outputs.

## TEST example with schema lock

After TRAIN is written:

```bash
PYTHONPATH=TRB/set/src \
python3 TRB/set/scripts/build_04_feature_matrix.py \
  --metadata TRB/set/test/metadata_test_30.csv \
  --module-dir TRB/set/test/result/feature_framework/top10000/02_sample_level_features/02_feature_modules \
  --plan TRB/set/configs/feature_framework/trb_feature_plan_top10000_primary_weighted_v1.yaml \
  --reference-manifest TRB/set/train/result/feature_framework/top10000/04_feature_matrix/04_feature_matrix_manifest.csv \
  --output-dir TRB/set/test/result/feature_framework/top10000/04_feature_matrix
```

`--reference-manifest` requires TEST to have the same ordered feature names/groups/types as TRAIN.

## Expected Top10000 primary-weighted shape for the current project

With the current 7,042 retained weighted 3-mers:

- clinical age/sex: 2
- non-QC static TCR candidates after five reference drops: 83
- weighted 3-mers: 7,042
- total predictors: 7,127

Therefore the expected matrix shape is:

- TRAIN: `123 × 7,128` including `sample_id`
- TEST: `51 × 7,128` including `sample_id`

The exact number of 3-mers remains training-derived and may change if Step02 vocabulary settings change.
