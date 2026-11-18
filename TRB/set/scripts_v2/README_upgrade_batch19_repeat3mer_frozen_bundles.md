# Batch 19 — Repeat-specific 3-mer adapter for frozen V3 bundles

## Purpose

This upgrade connects the historical Batch 09 outer-repeat-specific 3-mer implementation to the three frozen V3 Training Bundles:

- Total: 179 samples, 100 frozen 125/54 train/holdout splits.
- PBMC: 116 samples, 100 frozen 81/35 splits.
- Buffycoat: 63 samples, 100 frozen 44/19 splits.

It does **not** rebuild or modify any Training Bundle, split assignment, frozen marker, original generated scheme, static matrix, or sparse public-reference cache.

## Files

- \`TRB/set/scripts_v2/prepare_repeat3mer_bundle_schemes.py\`: validates frozen inputs and creates three isolated scheme copies.
- \`TRB/set/src/ra_ild_trb/repeat_3mer_features.py\`: existing Batch 09 implementation; reused without changing its numerical feature definition.
- \`TRB/set/src/ra_ild_trb/scheme_management.py\`: accepts the optional repeat-3mer section and includes its AA manifest in the prepared input manifest.
- \`TRB/set/tests/test_repeat3mer_bundle_scheme_adapter.py\`: adapter structural tests.

## Frozen-input validation

Before writing any scheme, the adapter verifies:

1. Training Bundle marker exists and reports \`FROZEN\`.
2. Bundle ID matches the bundle specification.
3. AA manifest SHA256 matches the frozen bundle marker.
4. AA manifest has one unique sample row per bundle sample and each referenced AA table exists.
5. Frozen split marker SHA256 matches the hash recorded in the Training Bundle marker.
6. Every one of the 100 split repeats has exact sample coverage, unique sample IDs, train/holdout roles and the train/holdout sizes recorded by the frozen marker.
7. Original generated scheme path, ID, output root and SHA256 match the frozen bundle marker.
8. The original generated scheme does not already contain a repeat-3mer section.

The adapter writes a separate manifest containing the source hashes and the new scheme hash. The original frozen files are read-only inputs.

## Batch 09 feature contract

Each outer repeat independently fits a 3-mer vocabulary using only that repeat's outer-training samples. The same vocabulary is then applied to all samples for that repeat.

- Core static features: 83.
- Unweighted 3-mer features: Top 500.
- Weighted 3-mer features: Top 500.
- Combined legacy alias: 83 + 500 + 500 = 1083 predictors.
- Minimum sample count: 5.
- Minimum prevalence: 0.05.
- Variance filter: remove only when both weighted and unweighted variance are below their respective thresholds.
- Ranking: prevalence descending, sum of weighted and unweighted variance descending, then 3-mer lexicographically ascending.
- Feature-ablation prefixes: Top 50, 100, 200 and 500.

The generated default M1–M3 models retain the legacy alias, which is backward-compatible and now resolves to the repeat-specific core + weighted/unweighted Top 500 features. No cohort labels, age, sex, batch or material are used to rank the 3-mers.

## Commands

From the repository root:

### 1. Dry-run the three bundles

\`\`\`bash
python3 TRB/set/scripts_v2/prepare_repeat3mer_bundle_schemes.py \\
  --repository-root /data/users/chenhaisheng/RA-ILD
\`\`\`

### 2. Write isolated scheme copies

Only after all three cohorts pass the dry-run:

\`\`\`bash
python3 TRB/set/scripts_v2/prepare_repeat3mer_bundle_schemes.py \\
  --repository-root /data/users/chenhaisheng/RA-ILD \\
  --apply
\`\`\`

Generated files:

\`\`\`text
TRB/set/configs/schemes/generated/repeat3mer/
├── total_trb_scheme_003_repeat100_no_clinical_v3_179_top10000_repeat3mer.yaml
├── pbmc_trb_scheme_pbmc_repeat100_no_clinical_v3_179_top10000_repeat3mer.yaml
├── buffycoat_trb_scheme_buffycoat_repeat100_no_clinical_v3_179_top10000_repeat3mer.yaml
└── REPEAT3MER_BUNDLE_SCHEME_MANIFEST.json
\`\`\`

Each scheme uses an independent experiment output root under \`TRB/set/experiments/\`.

### 3. Prepare the Total scheme without fitting

\`\`\`bash
python3 TRB/set/scripts_v2/prepare_analysis_scheme.py \\
  --scheme TRB/set/configs/schemes/generated/repeat3mer/total_trb_scheme_003_repeat100_no_clinical_v3_179_top10000_repeat3mer.yaml \\
  --repository-root /data/users/chenhaisheng/RA-ILD \\
  --dry-run
\`\`\`

Repeat for PBMC and Buffycoat. The prepared input manifest must include \`repeat_3mer_features.aa_manifest\`.

### 4. Run structural tests

\`\`\`bash
python3 TRB/set/scripts_v2/test_batch09_repeat_3mer_feature_groups.py
python3 TRB/set/tests/test_repeat3mer_bundle_scheme_adapter.py
python3 TRB/set/tests/test_batch01_scheme_management.py
\`\`\`

### 5. Smoke test

After scheme preparation succeeds, execute exactly one outer repeat for Total first. Inspect:

- \`06_repeat_3mer_vocabulary.csv\`
- \`06_repeat_3mer_feature_matrix.csv.gz\`
- \`06_repeat_3mer_audit.json\`
- \`06_feature_group_manifest.csv\`
- \`06_TASK_COMPLETE.json\`

Confirm the audit says \`outer_repeat_train_only\`, the training sample count is 125, the selected vocabulary has 500 unique 3-mers, and the outer-training sample hash matches the frozen assignment for that repeat. The holdout samples must not contribute to vocabulary selection.

Do not launch all 100 repeats or the five-algorithm production matrix until the smoke test passes and the algorithm runners are confirmed to consume this same prepared scheme and feature contract.

## Methodological scope

This adapter intentionally preserves Batch 09's historical **outer-repeat-level** vocabulary fitting. It does not silently change to inner-fold-level vocabulary fitting. The latter would be a separate methodological upgrade because it changes the feature-selection fit scope used during inner tuning.
