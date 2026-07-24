# TRB Framework V2 — Batch 08: Final integration and release

## Purpose

Batch 08 is an engineering closeout. It does not change model definitions,
train a new model, rerun the independent test set, or require completion of the
remaining optional V2 outer-task mirror computation.

It provides:

1. `TRB/set/README_V2.md` as the unified framework entry point;
2. final integration of the Batch 04 and Batch 06 hotfix documentation;
3. finalized Batch 04 and Batch 06 manifests that track only batch-owned files;
4. an explicit Git whitelist for the unified README and completion marker;
5. `run_all_checks.py` for one-command quick or full validation;
6. framework version `1.0.0`;
7. server-generated `FRAMEWORK_V2_COMPLETE.json` release evidence.

## Files added

```text
TRB/set/README_V2.md
TRB/set/src/ra_ild_trb/release.py
TRB/set/scripts_v2/run_all_checks.py
TRB/set/scripts_v2/README_batch08_release.md
TRB/set/scripts_v2/BATCH08_MANIFEST.json
TRB/set/tests/test_release.py
```

## Files updated

```text
.gitignore
TRB/set/src/ra_ild_trb/__init__.py
TRB/set/scripts_v2/README_batch04_modeling.md
TRB/set/scripts_v2/BATCH04_MANIFEST.json
TRB/set/scripts_v2/README_batch06_nested_cv.md
TRB/set/scripts_v2/BATCH06_MANIFEST.json
```

## Obsolete hotfix metadata to remove

ZIP extraction cannot delete repository files. After extraction, remove:

```bash
rm -f \
  TRB/set/scripts_v2/README_batch04_hotfix01.md \
  TRB/set/scripts_v2/BATCH04_HOTFIX01_MANIFEST.json \
  TRB/set/scripts_v2/README_batch06_hotfix01.md \
  TRB/set/scripts_v2/BATCH06_HOTFIX01_MANIFEST.json
```

The fixes themselves remain in the final code and are documented in the main
Batch 04 and Batch 06 READMEs.

## Timing relative to an active tmux outer task

Batch 08 does not modify the active scheduler or nested-CV worker, but the
outer-orchestration check intentionally fails while a task directory is being
written and classified as `incomplete`.

While the optional background task is active, syntax/unit validation may be run
and the unified checker may use:

```bash
python3 TRB/set/scripts_v2/run_all_checks.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --quick \
  --skip-outer-orchestration
```

Do not create the completion marker in this mode. Wait until no task is active
and `incomplete=0`.

## Server validation

### Syntax

```bash
python3 -m py_compile \
  TRB/set/src/ra_ild_trb/release.py \
  TRB/set/scripts_v2/run_all_checks.py
```

### Unit tests

```bash
python3 -m unittest discover \
  -s TRB/set/tests \
  -p 'test_*.py' \
  -v
```

Expected after Batch 08:

```text
Ran 120 tests
OK
```

### Unified quick validation

After the optional scheduler is no longer writing an incomplete task:

```bash
python3 TRB/set/scripts_v2/run_all_checks.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --quick
```

Expected final summary:

```text
Unified validation summary: mode=quick checks=10 PASS=10 FAIL=0
```

### Release marker

Commit Batch 08 first. From the clean committed tree, run:

```bash
python3 TRB/set/scripts_v2/run_all_checks.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --quick \
  --write-completion-marker
```

Then commit only:

```text
TRB/set/FRAMEWORK_V2_COMPLETE.json
```

This two-commit sequence allows the marker to record the clean Batch 08 source
commit rather than the previous commit or an uncommitted working tree.
