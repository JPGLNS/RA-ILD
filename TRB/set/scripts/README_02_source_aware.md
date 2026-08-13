# TRB Feature Framework v1 — Phase 2 source-aware Step02

## Purpose

Phase 2 makes Step02 repertoire-source aware without changing the established
biological feature mathematics.

The existing `build_02_sample_level_features.py` remains the numerical kernel.
A new entry point, `build_02_sample_level_features_source_aware.py`, resolves a
Feature Framework plan and routes either the full AA repertoire (`all`) or the
fixed-richness repertoire (`top10000`) into the same legacy calculation
functions.

This separation is deliberate:

```text
Feature plan
    ↓
repertoire source contract
    ├── all       → Step01 full summary
    └── top10000  → Step01b Top-K summary + parent Step01 summary
    ↓
source-specific QC / summary adapter
    ↓
unchanged legacy Step02 numerical kernel
    ↓
core + unweighted 3-mer + weighted 3-mer
```

## What changes in Phase 2

The repertoire registry now records, for every source:

- sample filename template;
- source summary kind;
- source summary path;
- optional parent source.

For example, `top10000` declares `all` as its parent. This lets future
`top5000`/`top20000` sources reuse the same logic without hard-coding filenames
inside Step02.

## Full repertoire (`all`) behavior

`all` uses the original Step01 AA tables and `01_AA_clone_table_summary.csv`.
The source adapter passes the original Step01 summary fields to the legacy
Step02 kernel unchanged.

Therefore Phase 2 has an explicit regression contract:

> For identical full-repertoire input, source-aware `all` mode must produce the
> same numerical feature values as the existing Step02 implementation.

The repository test verifies this against the real legacy Step02 script after
installation.

## Top10000 behavior

Top10000 must **not** compare the retained repertoire to full Step01 counts as if
they were expected to be equal.

Instead Phase 2 validates:

1. Step01b `qc_pass=True`;
2. exact `top_k` and retained AA clone count;
3. Top-K rank is exactly `1..K`;
4. Top-K `frequency_norm` and `read_fraction` each sum to 1;
5. retained read/frequency values agree with the Step01b summary;
6. `frequency_norm_full` / `read_fraction_full` retained mass agrees with the
   Step01b summary;
7. Step01b full-AA/read totals agree with the parent full Step01 summary.

For the legacy Step02 kernel, a source-specific summary row is constructed:

- `aa_clone_number` = current Top-K AA clone count;
- `total_reads_valid_nt` = read count retained in the analysis repertoire;
- `unique_valid_nt_clones` = NT clones contributing to retained AA clones;
- `valid_nt_row_ratio` = retained NT clones / full input NT rows;
- `total_reads_all_nt` and `input_nt_rows` retain the full Step01 technical
  denominators.

Thus diversity, expansion, weighted AA and weighted 3-mer features use the
Top-K-renormalized `frequency_norm`, while technical provenance is still
traceable to the parent Step01 data.

## Phase-2 output scope

Phase 2 still calculates the complete legacy Step02 output:

- `02_core_sample_features.csv`
- `02_3mer_unweighted_features.csv`
- `02_3mer_weighted_features.csv`
- training vocabulary / test vocabulary-used file
- build log and Markdown summary
- optional merged table

It additionally writes:

- `02_repertoire_source_audit.csv`
- `02_resolved_feature_source.json`

The feature plan is **not yet used to skip individual feature modules**. That is
Phase 3. In Phase 2 the plan selects the repertoire representation and records
provenance only.

## Recommended output isolation

Do not overwrite the old Step02 results. Use source-specific directories, e.g.

```text
TRB/set/train/result/feature_framework/all/02_sample_level_features/
TRB/set/train/result/feature_framework/top10000/02_sample_level_features/

TRB/set/test/result/feature_framework/all/02_sample_level_features/
TRB/set/test/result/feature_framework/top10000/02_sample_level_features/
```

## Dry-run example — Top10000 training subset

```bash
PYTHONPATH=TRB/set/src \
python3 TRB/set/scripts/build_02_sample_level_features_source_aware.py \
  --mode train \
  --plan TRB/set/configs/feature_framework/trb_feature_plan_template_v1.yaml \
  --metadata TRB/set/train/metadata_train_70.csv \
  --output-dir TRB/set/train/result/feature_framework/top10000/02_sample_level_features \
  --dry-run
```

The registry resolves the Top10000 AA directory and Step01b summary. For canonical
train/test metadata under `TRB/set/train` or `TRB/set/test`, the wrapper now
automatically binds the **parent full Step01 summary from the matching split**:

```text
TRB/set/train/result/01_AA_clone_table/01_AA_clone_table_summary.csv
TRB/set/test/result/01_AA_clone_table/01_AA_clone_table_summary.csv
```

This prevents historical global Step01 summaries from being mixed into current
train/test analysis. Explicit `--parent-summary-file` remains the highest-priority
override.

## Canonical split routing

For `repertoire_source: all`, canonical train/test metadata automatically routes
both the AA directory and Step01 summary to the matching split result directory.
For `top10000`, the global exact-TopK source remains unchanged, while its parent
full Step01 summary follows the current split. A mismatch such as `--mode train`
with metadata located under `TRB/set/test` is rejected instead of silently
routing data. Explicit `--aa-dir`, `--source-summary-file`, and
`--parent-summary-file` overrides still take precedence.

The legacy Step02 core contains **96 feature columns**. The written core table
therefore has **97 columns including `sample_id`**. The existing 96→83 contract
means 96 core features minus 8 QC/depth variables and 5 compositional reference
columns = 83 default candidate predictors; `sample_id` is not a feature.

## Dry-run example — full repertoire compatibility

```bash
PYTHONPATH=TRB/set/src \
python3 TRB/set/scripts/build_02_sample_level_features_source_aware.py \
  --mode train \
  --plan TRB/set/configs/feature_framework/trb_feature_plan_all_compat_v1.yaml \
  --metadata TRB/set/train/metadata_train_70.csv \
  --output-dir TRB/set/train/result/feature_framework/all/02_sample_level_features \
  --dry-run
```

## Test mode

Test mode must reuse the vocabulary produced by the matching training source.
Do not mix an `all` training vocabulary with a `top10000` test run, or vice
versa. Phase 4 will formalize this provenance check at matrix assembly; in Phase
2 the resolved-source JSON and directory isolation provide the audit trail.

## Deliberately deferred

- per-feature-group execution / module-specific output tables (Phase 3);
- config-driven matrix assembly (Phase 4);
- enriched-dictionary integration;
- legacy public replacement;
- ML input/config changes.
