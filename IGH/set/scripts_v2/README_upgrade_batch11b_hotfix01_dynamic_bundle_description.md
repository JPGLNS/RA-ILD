# IGH Batch 11B Hotfix 01 — Dynamic Training-Bundle Description

## Reason

The pre-existing IGH repeated-holdout training-bundle generator writes a
hard-coded generated-scheme description referring to three 120/49 holdouts.
Batch 11B introduces PBMC 76/32 and buffercoat 43/18 designs, each with 100
repeats. The hard-coded wording is scientifically misleading even though the
actual assignment files and computational parameters remain correct.

## Change

The generated scheme description is now resolved from the frozen assignments:

```text
{split_count} frozen {train_size}/{holdout_size} repeated holdouts ...
```

The command-line help/header is also made design-neutral.

## Scope

- no cohort membership changes;
- no split regeneration;
- no feature changes;
- no model fitting or tuning;
- no threshold changes;
- no existing frozen runtime output is overwritten.

Apply this hotfix before formally building the PBMC and buffercoat training
bundles so their generated YAML descriptions are correct on first creation.

## Acceptance test

```bash
python3 IGH/set/scripts_v2/test_batch11b_hotfix01_dynamic_bundle_description.py
```

Expected ending:

```text
IGH_BATCH11B_HOTFIX01_ACCEPTANCE_PASS
```
