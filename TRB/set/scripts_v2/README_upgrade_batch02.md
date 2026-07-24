# TRB Scheme Upgrade Batch 02

## Title

Flexible model definitions and additional-feature management.

## Scope

Batch 02 extends the Batch 01 scheme layer. It does **not** change Elastic Net
fitting, public-reference construction, fold-local preprocessing, candidate ranking,
or threshold selection.

New capabilities:

1. Replace the complete `models` mapping in a scheme, allowing deletion of M0 and
   removal of age/sex.
2. Recalculate model-dependent expected counts automatically.
3. Allow `final_model.status: not_selected` for a new development scheme.
4. Merge one or more additional CSV/CSV.GZ feature tables by sample ID.
5. Reject duplicate IDs, missing/extra samples, column collisions, NA/Inf, and bad
   declared numeric values.
6. Write feature-input and effective-model feature audits before model fitting.
7. Preserve the Batch 01 baseline-compatibility scheme.

## Scheme model replacement

Use `replacements.models`, not `overrides.models`, when a model needs to be deleted.
Recursive overrides cannot remove inherited keys.

```yaml
replacements:
  models:
    M1_static_tcr:
      numeric: []
      categorical: []
      static_feature_groups: ["static_tcr_candidate_predictors"]
      dynamic_public: []
```

The preparer recalculates:

- `model_selection.expected_resolved_columns`
- `nested_cv.expected_outer_tasks`
- `nested_cv.expected_inner_fits_per_task`
- `outer_tasks.expected_tasks`
- aggregation metric and prediction row counts

## Additional feature table

```yaml
overrides:
  data:
    train:
      additional_feature_tables:
        - path: "TRB/set/additional_features/my_features.csv"
          key: "sample_id"
          required_columns: ["feature_a", "feature_b"]
          numeric_columns: ["feature_a"]
          categorical_columns: ["feature_b"]
```

Every additional table must contain exactly the same training sample IDs as the
base matrix. Row order may differ; the base matrix order is restored after joining.

Only ordinary sample-level features should be supplied this way. Features that use
labels, other patients, supervised screening, or public-reference information must
be constructed inside the CV partitions and are outside Batch 02.

## Included schemes

- `trb_scheme_002_no_clinical.yaml`: removes M0, age, and sex.
- `trb_scheme_additional_features_template.yaml`: copy-and-edit template; it is
  intentionally not directly runnable until its placeholder feature file exists.

## No model fitting in Batch 02 acceptance

Acceptance tests prepare and audit configurations only. Do not run the 100 outer
jobs for the new scheme during this batch.
