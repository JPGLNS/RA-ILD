# TRB Feature Framework v1 — Phase 3 modular Step02 outputs

## Scope

Phase 3 materializes the already-validated Phase-2 Step02 outputs as independent,
registry-defined feature-module tables. It intentionally does **not** recalculate
Shannon, Simpson, Gini, clonality, expansion, AA-length, composition,
physicochemical, or 3-mer values.

The numerical source of truth remains the Phase-2 source-aware Step02 artifacts:

- `02_core_sample_features.csv`
- `02_3mer_unweighted_features.csv`
- `02_3mer_weighted_features.csv`
- `02_resolved_feature_source.json`

This keeps the migration auditable: Phase 3 is a deterministic structural split,
not a new biological feature implementation.

## Generated modules

The 96 legacy core features are split into eight static modules:

- `tcr_qc_depth.csv` — 8 features
- `tcr_diversity.csv` — 5
- `tcr_clonal_expansion.csv` — 5
- `tcr_aa_length.csv` — 24
- `tcr_aa_composition_unweighted.csv` — 20
- `tcr_aa_composition_weighted.csv` — 20
- `tcr_physicochemical_unweighted.csv` — 7
- `tcr_physicochemical_weighted.csv` — 7

The existing training-derived 3-mer matrices become two additional modules:

- `tcr_3mer_unweighted.csv`
- `tcr_3mer_weighted.csv`

All module files retain `sample_id` as the first key column.

## Generation is not selection

Phase 3 deliberately writes all ten modules even when the Phase-2 feature plan did
not select every group. The manifest records `selected_in_phase2_plan`, but does
not use that flag to suppress generation.

This preserves the intended separation:

```text
repertoire representation
        ↓
feature generation
        ↓
module tables + manifest
        ↓
feature-plan / matrix assembly
        ↓
model-specific selection and fitting
```

For example, `tcr_qc_depth` is still materialized for audit/sensitivity analyses,
even though it is not selected by the default Top10000 exploratory plan.

## Strict compatibility checks

Before writing anything, the builder requires:

1. all 96 core feature columns to be assigned to exactly one of the eight static
   registry groups;
2. no extra or missing core columns;
3. exact sample-ID order agreement across core and both 3-mer matrices;
4. all feature values to be finite numeric values;
5. each 3-mer matrix to contain only the prefix registered for its group;
6. the eight static modules to recombine **exactly** to the original core table.

The legacy static contract remains:

```text
96 core features
- 8 QC/depth features
- 5 fixed compositional reference columns
= 83 default static candidate predictors
```

No scaling, filtering, standardization, prevalence selection, or model-specific
feature selection occurs in Phase 3.

## Manifest

`02_feature_module_manifest.csv` contains one row per feature and records:

- repertoire source and Phase-2 plan;
- feature group and module file;
- feature order and name;
- family, module type, and fit scope;
- leakage policy and producer;
- model role;
- whether the group was selected in the Phase-2 plan;
- whether the feature is a fixed reference-drop column;
- whether it is a default candidate predictor.

This manifest is intended to become the contract consumed by the Phase-4 matrix
assembler.

## Usage

Top10000 TRAIN example:

```bash
PYTHONPATH=TRB/set/src \
python3 TRB/set/scripts/build_02_feature_modules.py \
  --step02-dir TRB/set/train/result/feature_framework/top10000/02_sample_level_features \
  --dry-run
```

Formal write:

```bash
PYTHONPATH=TRB/set/src \
python3 TRB/set/scripts/build_02_feature_modules.py \
  --step02-dir TRB/set/train/result/feature_framework/top10000/02_sample_level_features
```

By default modules are written to:

```text
<step02-dir>/02_feature_modules/
```

The same command applies to TEST by changing `--step02-dir`.

## Compatibility artifacts

Phase 3 does not remove, rename, or rewrite the existing Phase-2 Step02 files.
Legacy Step04 and other existing code can therefore continue to consume the old
artifacts while the new module layer is validated and adopted incrementally.
