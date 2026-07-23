# TRB Framework V2 — Batch 01

This batch adds configuration and frozen-baseline checks only. It does not change
any V1 feature, cross-validation, final-training, or independent-test algorithm.

## Included files

```text
TRB/set/
├── configs/trb_baseline_m2_v1.yaml
├── src/ra_ild_trb/
│   ├── __init__.py
│   ├── config.py
│   └── paths.py
├── scripts_v2/
│   ├── validate_experiment_config.py
│   ├── check_v1_baseline_outputs.py
│   ├── gitignore_TRB_v2_snippet.txt
│   └── README_batch01.md
└── tests/test_config.py
```

## Dependency

```bash
mamba install -c conda-forge pyyaml
```

The tests use Python's built-in `unittest`; `pytest` is not required for this batch.

## Installation into the repository

From the repository root:

```bash
unzip TRB_framework_v2_batch01.zip
```

Add the rules from:

```text
TRB/set/scripts_v2/gitignore_TRB_v2_snippet.txt
```

to the TRB whitelist section of the repository-root `.gitignore`. Do not replace
the existing `.gitignore`.

## Validate the YAML

```bash
python3 TRB/set/scripts_v2/validate_experiment_config.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml
```

Also require every configured V1 path to exist:

```bash
python3 TRB/set/scripts_v2/validate_experiment_config.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --check-paths
```

## Run unit tests

```bash
python3 -m unittest discover \
  -s TRB/set/tests \
  -p 'test_*.py' \
  -v
```

## Check the existing V1 baseline

Inputs and CV files only:

```bash
python3 TRB/set/scripts_v2/check_v1_baseline_outputs.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --scope inputs
```

Include the Step 07 final model:

```bash
python3 TRB/set/scripts_v2/check_v1_baseline_outputs.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --scope training
```

Include the one-time Step 08 validation:

```bash
python3 TRB/set/scripts_v2/check_v1_baseline_outputs.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --scope full \
  --json-output TRB/set/train/result/v2_regression_checks/batch01_v1_baseline.json
```

## Suggested commit

```bash
git add .gitignore \
  TRB/set/configs \
  TRB/set/src \
  TRB/set/scripts_v2 \
  TRB/set/tests

git diff --cached
git commit -m "Add TRB V2 configuration and baseline checks"
git push -u origin refactor/trb-framework-v2
```
