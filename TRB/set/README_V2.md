# RA/RA-ILD TRB Framework V2

**Framework version: 1.0.0**

This is the unified entry point for the refactored TRB analysis framework. The
V2 code reproduces the frozen V1 scientific workflow while separating reusable
configuration, public-reference construction, preprocessing, Elastic Net
fitting, nested cross-validation, outer-task orchestration, and stability
analysis into testable modules.

## Current scientific status

The original V1 analysis is already complete:

- training cohort: 123 samples;
- independent test cohort: 51 samples;
- final model: `M2_static_tcr_public`;
- locked alpha: `0.5`;
- locked lambda: `30`;
- locked probability threshold: `0.509337`;
- independent-test ROC-AUC: approximately `0.7796`.

The remaining unexecuted V2 outer tasks are optional mirror computation. They
are not required to establish the existing model result or to release the V2
framework.

## Core principles

- The independent test set is not read by the nested-CV engine.
- Public-reference construction uses training samples only.
- Model-fitting samples use exact leave-one-out public features.
- Encoding, zero-variance filtering, and standardization are fitted within the
  relevant training partition only.
- The V1 outer and inner assignments, candidate grid, deterministic seeds,
  candidate ranking, and Youden threshold rule are preserved.
- V2 outputs are isolated under `result_v2` and never overwrite frozen V1
  results.

## Directory map

```text
TRB/set/
├── configs/                         experiment configuration
├── src/ra_ild_trb/                 reusable framework modules
├── scripts_v2/                     command-line runners and regression checks
├── tests/                          synthetic unit tests
├── README_V2.md                    this entry document
└── FRAMEWORK_V2_COMPLETE.json      generated release-validation record
```

Important modules:

```text
public_reference.py   leakage-controlled public-sequence features
preprocessing.py      training-derived encoding, filtering, and scaling
modeling.py           Elastic Net logistic-regression backend
thresholds.py         Youden and optional threshold utilities
metrics.py            classification metrics and bootstrap helpers
specifications.py     M0–M3 definitions and candidate management
nested_cv.py          one outer-task nested-CV engine
outer_cv.py           100-task orchestration, aggregation, and stability
release.py            unified validation and release-marker helpers
```

## Environment check

Run the framework with the same validated conda environment:

```bash
conda activate ra-ild
which python3
python3 -c "import sys; print(sys.executable)"
```

The executable should point to the `ra-ild` environment. The unified checker
also prints the exact interpreter used for every child command.

## Unified validation

### Normal release check

Run after no outer task is actively writing an incomplete task directory:

```bash
python3 TRB/set/scripts_v2/run_all_checks.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --quick
```

This runs, in order:

1. configuration validation;
2. configured-path validation;
3. the complete unit-test suite;
4. frozen V1 baseline validation;
5. public-reference regression;
6. preprocessing regression;
7. modeling regression;
8. model-specification regression;
9. quick nested-CV regression;
10. outer-task tree validation.

### While an optional outer task is running

An active task is temporarily classified as `incomplete`. Use:

```bash
python3 TRB/set/scripts_v2/run_all_checks.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --quick \
  --skip-outer-orchestration
```

This mode is for development checks only and cannot write the completion
marker.

### Full nested-CV regression

The expensive validation repeats all 480 inner fits for the frozen
repeat-01/fold-01 task:

```bash
python3 TRB/set/scripts_v2/run_all_checks.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --full
```

A successful full regression was already recorded on 2026-07-24:

```text
Mode=full Outer task=1/1 Checks=43 PASS=43 FAIL=0
```

## Completion marker

The completion marker is deliberately generated on the server rather than
shipped pre-filled in the delivery archive. This avoids claiming validation
under an environment that did not actually run the checks.

Recommended release sequence:

1. apply Batch 08 and run the quick unified check;
2. commit the Batch 08 code and documentation;
3. from the clean committed tree, run:

```bash
python3 TRB/set/scripts_v2/run_all_checks.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --quick \
  --write-completion-marker
```

4. commit `TRB/set/FRAMEWORK_V2_COMPLETE.json` as the final release-validation
   record.

The marker records the Python executable, runtime package versions, Git branch
and commit, all check outcomes, the prior full nested-CV validation, the locked
model, and the independent-test policy.

## Optional V2 mirror computation

The 100-task outer runner is resumable:

```bash
python3 TRB/set/scripts_v2/run_all_outer_tasks.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml \
  --execute \
  --workers 1
```

This computation is optional for framework release. Do not start a second
scheduler while another scheduler is active. While a task is running, use
`--status-only` rather than treating the temporary `incomplete=1` state as a
framework failure.

After all 100 tasks are complete, aggregation is available through:

```bash
python3 TRB/set/scripts_v2/aggregate_outer_results.py \
  --config TRB/set/configs/trb_baseline_m2_v1.yaml
```

## Git policy

Generated outputs and delivery ZIP files are excluded. Track only source code,
configuration, tests, documentation, and the final completion marker. Avoid
`git add .`; stage the intended release files explicitly.
