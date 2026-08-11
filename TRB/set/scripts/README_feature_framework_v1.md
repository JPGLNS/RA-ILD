# TRB Feature Framework v1 — Phase 1 contract layer

## Scope

This upgrade is intentionally **additive**. It does not change the numerical
implementation of existing Step02/03/04 features and does not change any ML
training code.

Phase 1 adds a contract layer for later migration:

1. logical repertoire-source selection (`all`, `top10000`);
2. modular feature-group registry;
3. explicit distinction between per-sample features and training-derived
   feature definitions;
4. feature-plan validation/resolution;
5. deterministic classification of legacy matrix columns by registry rules.

## Why this layer exists

The current project is still exploratory. A feature should therefore not be
hard-coded into every model. Repertoire representation, feature generation,
feature-group selection, matrix assembly, and ML model choice need to become
separate decisions.

The intended architecture is:

```text
all / top10000 repertoire
        ↓
feature modules
        ↓
feature registry
        ↓
experiment feature plan
        ↓
feature matrix assembly   (later upgrade)
        ↓
Elastic Net / Linear SVM / XGBoost  (later upgrade)
```

## Repertoire sources

The registry currently defines:

- `all`: full Step01 AA repertoire;
- `top10000`: exact Top10,000 Step01b AA repertoire.

The registry records the logical source layer, default project path, file glob,
required columns, canonical weight column, and Top-K value when applicable.

No data are copied or recalculated by this Phase-1 framework.

## Feature groups

Existing feature definitions are registered as small, independently selectable
modules instead of one fixed `TCR stat` block:

- `clinical_age_sex`
- `clinical_material`
- `tcr_qc_depth`
- `tcr_diversity`
- `tcr_clonal_expansion`
- `tcr_aa_length`
- `tcr_aa_composition_unweighted`
- `tcr_aa_composition_weighted`
- `tcr_physicochemical_unweighted`
- `tcr_physicochemical_weighted`
- `tcr_3mer_unweighted`
- `tcr_3mer_weighted`
- `public_legacy_descriptive` (legacy optional)
- `public_legacy_reference` (legacy optional)
- `enriched_dict` (planned framework-integration contract only)

All groups are opt-in at the feature-plan level. The registry does not force a
particular group into a model.

## Training-derived feature definitions

A critical distinction is recorded explicitly:

- diversity/expansion/length/composition/physicochemical features are
  `sample_static` and calculated independently within each sample;
- unweighted/weighted 3-mer groups are `training_vocabulary`: the vocabulary and
  unsupervised filters are fitted on training samples and then applied to held-out
  samples;
- legacy public reference and future enriched dictionary are
  `learned_reference`, so their reference/dictionary must be fitted on training
  data only and applied to held-out data.

This contract is designed to keep later CV implementations leakage controlled.

## Top terminology

`tcr_clonal_expansion` contains top1/top5/top10/top20/top50 cumulative clone
frequency features. This is distinct from **repertoire Top-K preprocessing**
(`top10000`) and from later **Top-N k-mer feature selection**.

## Feature plan

The supplied plan template selects `top10000` and the currently defined TCR
statistics plus both 3-mer representations. It intentionally excludes legacy
public features and the not-yet-implemented enriched dictionary.

Change only the plan to explore a different repertoire or feature-group
combination; the registry remains the contract.

Example switch:

```yaml
plan:
  repertoire_source: all   # or top10000
  feature_groups:
    - tcr_diversity
    - tcr_3mer_weighted
```

## Validation

```bash
python3 TRB/set/scripts/validate_feature_framework.py \
  --registry TRB/set/configs/feature_framework/trb_feature_registry_v1.yaml \
  --plan TRB/set/configs/feature_framework/trb_feature_plan_template_v1.yaml \
  --dry-run
```

Optional: classify the header of an existing matrix without changing it:

```bash
python3 TRB/set/scripts/validate_feature_framework.py \
  --registry TRB/set/configs/feature_framework/trb_feature_registry_v1.yaml \
  --columns-csv path/to/feature_matrix.csv \
  --dry-run
```

## Deliberately deferred to later upgrades

- routing Step02 to `all` versus `top10000`;
- replacing Step02 summary assumptions for TopK input;
- generating module-specific feature tables;
- matrix assembly from selected feature groups;
- binding/finalizing enriched-dictionary features (existing exploratory enrich-dictionary code is intentionally not wired here);
- replacing/removing legacy Step03 public features;
- ML config changes.

Keeping these out of Phase 1 makes this PR small and provides a stable contract
before numerical feature code is migrated.
