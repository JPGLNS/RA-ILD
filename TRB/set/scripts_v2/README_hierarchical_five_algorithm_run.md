# TRB V3 179 — five-algorithm adaptation and global scheduler

This upgrade adds orchestration/configuration only. It does not edit the frozen
179-sample Training Bundles, split assignments, split markers, static feature
matrix, sparse public-reference cache, or the three already prepared Elastic
Net experiments/results.

## Branch

\`feature/trb-v3-179-five-algorithm-adapter\`

## Cohort/model contract

| Cohort | Samples | Models | Outer repeats | Train/holdout |
|---|---:|---:|---:|---:|
| Total | 179 | M0-M3 (4) | 100 | 125/54 |
| PBMC | 116 | M0-M2 (3) | 100 | 81/35 |
| Buffycoat | 63 | M0-M2 (3) | 100 | 44/19 |

The 15 cohort/algorithm schemes are 3 cohorts × 5 algorithms. Every algorithm
inherits its cohort's exact model definitions, frozen outer/inner assignments,
training bundle, public-reference cache, and repeat-specific outer-training-only
3-mer settings.

## Algorithm contract

- Elastic Net: existing resolved schemes; alpha=[0.1, 0.5, 0.9], lambda grid of 8.
- Ridge: alpha=0 endpoint of the same validated saga logistic backend; 8 lambdas.
- Lasso: alpha=1 endpoint of the same validated saga logistic backend; 8 lambdas.
- Linear SVM: LinearSVC, L2 squared hinge, balanced class weights, C grid of 8.
  Scores are decision_function values, not probabilities; log-loss and Brier
  score are unavailable.
- XGBoost: deterministic frozen random candidate bank of 100 candidates,
  n_jobs_per_fit=1, early_stopping=false. The configured candidate pool has
  3,888 possible combinations; the 100 selected candidates are frozen once
  per scheme and reused for all 100 repeats.

Ridge/Lasso are regularization endpoints of the existing Elastic Net logistic
engine; they are not separate estimators or a different feature pipeline.

## 1. Install branch on neo3

From the existing repository, first preserve the current working tree. Do not
run git clean/reset. The user-created generated hierarchical scheme YAMLs and
patch files may be untracked local files and must remain available.

\`\`\`bash
cd /data/users/chenhaisheng/RA-ILD

git status --short
git fetch origin
git switch --track origin/feature/trb-v3-179-five-algorithm-adapter
\`\`\`

If Git reports that the local branch already exists, use
\`git switch feature/trb-v3-179-five-algorithm-adapter\` instead.

## 2. Check prerequisites

The following local files must already exist from the previous completed stage:

\`\`\`text
TRB/set/configs/schemes/generated/hierarchical_repeat3mer/
  trb_v3_179_total_m0_m3_repeat100_top10000_r3_v1.yaml
  trb_v3_179_pbmc_m0_m2_repeat100_top10000_r3_v1.yaml
  trb_v3_179_buffycoat_m0_m2_repeat100_top10000_r3_v1.yaml
\`\`\`

The three base resolved configs must also exist under their existing experiment
directories. The preparation script verifies sample counts, exact model names,
100 repeats, fold 1, and enabled repeat-specific 3-mer features before writing.

Check algorithm dependencies:

\`\`\`bash
python3 - <<'PY'
import sklearn
import xgboost
import yaml
print("scikit-learn", sklearn.__version__)
print("xgboost", xgboost.__version__)
print("PyYAML", yaml.__version__)
PY
\`\`\`

## 3. Generate the 12 new schemes

First run the no-write plan:

\`\`\`bash
python3 TRB/set/scripts_v2/prepare_hierarchical_five_algorithm_schemes.py \\
  --repository-root "$PWD"
\`\`\`

Expected: 12 new scheme YAMLs (Ridge, Lasso, Linear SVM, XGBoost × 3 cohorts),
15 total algorithm/cohort entries including the existing Elastic Net configs,
and 1,500 outer-task slots.

Then apply:

\`\`\`bash
python3 TRB/set/scripts_v2/prepare_hierarchical_five_algorithm_schemes.py \\
  --repository-root "$PWD" \\
  --apply
\`\`\`

This writes isolated scheme configs and resolved experiment roots for the 12
new schemes. The existing three Elastic Net experiment directories are not
re-prepared or overwritten.

## 4. Global scheduler dry-run

\`\`\`bash
python3 TRB/set/scripts_v2/run_hierarchical_five_algorithm_tasks.py \\
  --repository-root "$PWD" \\
  --max-workers 50 \\
  --memory-budget-gb 120 \\
  --min-free-gb 12
\`\`\`

Expected task slots: 1,500. Existing complete Elastic Net repeat-1 tasks are
skipped after checking their completion marker, model rows, predictions, and
candidate rows. Incomplete/invalid existing task directories are blocked by
default rather than silently overwritten.

## 5. First execution wave: repeat 1 for all 15 schemes

The task order is round-robin across all 15 schemes by repeat. Therefore the
first 12 pending tasks are repeat 1 for the 12 new Ridge/Lasso/SVM/XGBoost
schemes; the three existing Elastic Net repeat-1 tasks are recognized as
complete and skipped.

\`\`\`bash
python3 TRB/set/scripts_v2/run_hierarchical_five_algorithm_tasks.py \\
  --repository-root "$PWD" \\
  --max-workers 50 \\
  --memory-budget-gb 120 \\
  --min-free-gb 12 \\
  --limit 12 \\
  --execute
\`\`\`

Inspect the 12 repeat-1 task outputs. Each must contain a COMPLETE marker,
outer metrics/predictions, and the expected number of tuning rows per model.

## 6. Launch all remaining tasks

After the first wave completes successfully, run the same command without
\`--limit 12\`:

\`\`\`bash
nohup python3 TRB/set/scripts_v2/run_hierarchical_five_algorithm_tasks.py \\
  --repository-root "$PWD" \\
  --max-workers 50 \\
  --memory-budget-gb 120 \\
  --min-free-gb 12 \\
  --execute \\
  > TRB/set/logs/five_algorithm_global_20261118.log 2>&1 &
echo $! > TRB/set/logs/five_algorithm_global.pid
\`\`\`

The scheduler automatically skips tasks already marked complete and executes
only missing tasks. It does not delete or overwrite completed tasks.

## Memory guard semantics

The scheduler applies one global reservation budget across all 15 schemes:

- maximum 50 task processes;
- maximum 120 GiB of nominal per-task reservations;
- default reservation: 3 GiB per Elastic Net/Ridge/Lasso/SVM task and 5 GiB
  per XGBoost task;
- additionally checks Linux \`/proc/meminfo:MemAvailable\` and keeps at least
  12 GiB available;
- XGBoost uses one internal thread per fit to avoid nested oversubscription.

This is a cooperative scheduler reservation plus a live free-memory guard, not
an OS-enforced cgroup/RSS hard limit. Other users/processes can consume memory
outside this scheduler; monitor system memory during the run.

## Status and logs

Each task writes to its engine's isolated experiment output root. The scheduler
writes a global task status report to:

\`\`\`text
TRB/set/experiments/trb_v3_179_five_algorithm_global/07_GLOBAL_TASK_STATUS.csv
\`\`\`

Per-task logs are written under each experiment's \`logs\` directory.

Do not aggregate results until all 100 repeats for the relevant cohort/algorithm
have completed. Use the corresponding engine-specific aggregation scripts:
\`aggregate_repeated_holdout_results.py\`,
\`aggregate_linear_svm_results.py\`, and
\`aggregate_xgboost_results.py\`.
