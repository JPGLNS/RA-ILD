# IGH Scheme Upgrade Batch 02

## Title

Flexible model definitions and additional-feature management.

## Scope

Batch 02 completes the configurable scheme layer established in Batch 01. It does
**not** change Elastic Net fitting, public-reference construction, fold-local
preprocessing, candidate ranking, or threshold selection.

New capabilities:

1. Replace the complete `models` mapping in a scheme, allowing models to be added
   or deleted and allowing each model to use its own feature inputs.
2. Recalculate model-dependent expected counts automatically.
3. Allow `final_model.status: not_selected` for development schemes.
4. Merge one or more additional CSV/CSV.GZ sample-level feature tables by sample ID.
5. Reject duplicate IDs, missing/extra samples, column collisions, NA/Inf, and bad
   declared numeric values.
6. Write feature-input and effective-model feature audits before model fitting.
7. Preserve the Batch 01 historical baseline-compatibility scheme.

## Scheme model replacement

Use `replacements.models`, not `overrides.models`, when a model needs to be deleted.
Recursive overrides cannot remove inherited keys.

```yaml
replacements:
  models:
    M1_static_igh:
      numeric: []
      categorical: []
      static_feature_groups: ["static_igh_candidate_predictors"]
      dynamic_public: []
```

The scheme preparer recalculates:

- `model_selection.expected_resolved_columns`;
- `nested_cv.expected_outer_tasks`;
- `nested_cv.expected_inner_fits_per_task`;
- `outer_tasks.expected_tasks`;
- aggregation metric and prediction row counts.

## Additional feature table

```yaml
overrides:
  data:
    train:
      additional_feature_tables:
        - path: "IGH/set/additional_features/my_features.csv"
          key: "sample_id"
          required_columns: ["feature_a", "feature_b"]
          numeric_columns: ["feature_a"]
          categorical_columns: ["feature_b"]
```

Every table must contain exactly the same configured training sample IDs as the
base matrix. Row order may differ; the base matrix order is restored after joining.
CSV and CSV.GZ are both supported by pandas.

Only ordinary sample-level features should be supplied this way. Features that use
labels, other patients, supervised screening, or public-reference information must
be constructed inside the CV partitions and are outside Batch 02.

## Included schemes

- `igh_scheme_001_baseline_compat.yaml`: four-model historical compatibility scheme;
- `igh_scheme_002_no_clinical.yaml`: removes M0, age, and sex;
- `igh_scheme_additional_features_template.yaml`: copy-and-edit template; it is
  intentionally not directly runnable until its placeholder feature file exists.

## Acceptance

Batch 02 acceptance prepares and audits configurations only. It does not launch the
100 historical outer tasks or generate new scientific model results.
