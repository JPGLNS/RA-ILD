# RA-ILD TRB Framework Upgrade — Batch 01

## Title

**Multi-scheme configuration and isolated result directories**

## Scope

This batch adds scheme preparation around the existing V2 engine. It does **not** change:

- the 123/51 frozen baseline split;
- outer or inner CV assignments;
- public-reference thresholds or leakage-control rules;
- Elastic Net fitting;
- alpha/lambda candidate grids;
- threshold selection;
- M0–M3 definitions;
- feature selection rules.

It adds:

1. one compact YAML per RA-ILD adjustment scheme;
2. an isolated output root under `TRB/set/experiments/<scheme_id>`;
3. source/base/resolved configuration snapshots;
4. input file SHA256 and environment audit files;
5. a baseline-compatibility scheme for later regression testing;
6. removal of the hard-coded default nested-CV output directory in `run_nested_cv_task.py`.

## Important

Apply this batch on a new Git branch. The stopped V2 tasks and old V1/V2 results are not deleted or overwritten.

## 1. Create the upgrade branch

From `/data/users/chenhaisheng/RA-ILD`:

```bash
git status
git switch -c feature/trb-scheme-upgrade-batch01
```

A clean working tree is recommended before installation.

## 2. Upload and unzip

Upload the ZIP to the repository root, then run:

```bash
unzip TRB_framework_scheme_upgrade_batch01.zip -d /tmp/TRB_framework_scheme_upgrade_batch01
cd /tmp/TRB_framework_scheme_upgrade_batch01/trb_upgrade_batch01
```

## 3. Preflight

```bash
python3 apply_batch01.py \
  --repository-root /data/users/chenhaisheng/RA-ILD \
  --check-only
```

Expected ending:

```text
TRB_SCHEME_UPGRADE_BATCH01 preflight: PASS
Check only; no file was changed.
```

## 4. Apply

```bash
python3 apply_batch01.py \
  --repository-root /data/users/chenhaisheng/RA-ILD
```

## 5. Syntax tests

```bash
cd /data/users/chenhaisheng/RA-ILD

python3 -m py_compile \
  TRB/set/src/ra_ild_trb/scheme_management.py \
  TRB/set/scripts_v2/prepare_analysis_scheme.py \
  TRB/set/scripts_v2/test_batch01_scheme_management.py \
  TRB/set/scripts_v2/run_nested_cv_task.py
```

## 6. Fast Batch 01 tests

These tests do not fit a model:

```bash
python3 TRB/set/scripts_v2/test_batch01_scheme_management.py
```

Expected:

```text
Ran 5 tests
OK
```

## 7. Dry-run the baseline compatibility scheme

```bash
python3 TRB/set/scripts_v2/prepare_analysis_scheme.py \
  --scheme TRB/set/configs/schemes/trb_scheme_001_baseline_compat.yaml \
  --dry-run
```

Expected key fields:

```text
Scheme ID:       trb_scheme_001_baseline_compat
Output root:     .../TRB/set/experiments/trb_scheme_001_baseline_compat
Missing inputs:  0
Dry run only; no directory or snapshot was written.
```

If `Missing inputs` is not zero, stop and inspect the listed path. Do not use `--allow-missing-inputs` for the real baseline preparation.

## 8. Prepare the scheme

```bash
python3 TRB/set/scripts_v2/prepare_analysis_scheme.py \
  --scheme TRB/set/configs/schemes/trb_scheme_001_baseline_compat.yaml
```

Created structure:

```text
TRB/set/experiments/trb_scheme_001_baseline_compat/
├── 00_config/
│   ├── source_scheme.yaml
│   ├── base_config_snapshot.yaml
│   ├── resolved_config.yaml
│   ├── input_manifest.json
│   ├── environment.json
│   └── scheme_metadata.json
├── 01_splits/
├── 02_tasks/
├── 03_summary/
├── 04_final_model/
├── 05_independent_validation/
├── logs/
└── SCHEME_PREPARED.json
```

## 9. Check the resolved config without executing tasks

```bash
python3 TRB/set/scripts_v2/run_all_outer_tasks.py \
  --config TRB/set/experiments/trb_scheme_001_baseline_compat/00_config/resolved_config.yaml \
  --status-only
```

This may create only manifest/status files inside the new scheme directory. It must not write to the old `result_v2` directory.

## Acceptance criteria

Batch 01 passes when:

1. all five unit tests pass;
2. dry-run reports zero missing inputs;
3. the scheme is prepared under its own experiment directory;
4. resolved YAML preserves the frozen scientific sections;
5. `--status-only` writes only inside the scheme directory;
6. old V1 and V2 result directories are unchanged.

## Git review

```bash
git status --short
git diff -- TRB/set/scripts_v2/run_nested_cv_task.py
git diff --stat
```

Suggested commit after all checks pass:

```bash
git add \
  TRB/set/src/ra_ild_trb/scheme_management.py \
  TRB/set/scripts_v2/prepare_analysis_scheme.py \
  TRB/set/scripts_v2/test_batch01_scheme_management.py \
  TRB/set/scripts_v2/run_nested_cv_task.py \
  TRB/set/scripts_v2/README_upgrade_batch01.md \
  TRB/set/scripts_v2/BATCH01_SCHEME_UPGRADE_MANIFEST.json \
  TRB/set/configs/schemes/trb_scheme_001_baseline_compat.yaml \
  TRB/set/BATCH01_SCHEME_UPGRADE_APPLIED.json

git commit -m "Add TRB analysis scheme isolation"
```

Do not commit the generated `TRB/set/experiments/...` result directory until its repository policy is decided. It contains run-specific snapshots and later model outputs.
