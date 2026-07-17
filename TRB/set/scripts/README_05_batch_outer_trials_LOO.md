# Step 05 LOO batch runner

## Purpose

Run all 100 outer validation tasks in the background:

```text
20 repeats × 5 outer folds = 100 tasks
```

Each task calls the already validated LOO worker:

```text
/data/users/chenhaisheng/RA-ILD/TRB/set/scripts/run_05_single_outer_trial_loo.py
```

The runner executes tasks sequentially by default. This is slower than parallel
execution but safer for memory use because each worker loads large sparse
public-repertoire matrices.

## Files

Copy these files to:

```text
/data/users/chenhaisheng/RA-ILD/TRB/set/scripts/
```

```text
run_05_batch_outer_trials_loo.py
start_05_batch_outer_trials_loo.sh
```

## Syntax and dry-run checks

```bash
cd /data/users/chenhaisheng/RA-ILD/TRB

PYTHONPYCACHEPREFIX=/tmp/pycache_ra_ild \
python3 -m py_compile \
  set/scripts/run_05_batch_outer_trials_loo.py

python3 set/scripts/run_05_batch_outer_trials_loo.py \
  --repeats 1 \
  --folds 1-2 \
  --dry-run
```

The dry run only prints commands and does not fit models.

## Recommended nohup start

```bash
cd /data/users/chenhaisheng/RA-ILD/TRB

chmod +x set/scripts/run_05_batch_outer_trials_loo.py
chmod +x set/scripts/start_05_batch_outer_trials_loo.sh

bash set/scripts/start_05_batch_outer_trials_loo.sh
```

This starts:

```bash
nohup python3 -u set/scripts/run_05_batch_outer_trials_loo.py \
  --repeats 1-20 \
  --folds 1-5 \
  --public-feature-set raw_bilateral \
  > set/train/result/05_modeling/batch_run_loo/05_batch_nohup.log \
  2>&1 &
```

The shell launcher stores the process ID in:

```text
set/train/result/05_modeling/batch_run_loo/05_batch_nohup.pid
```

## Monitor progress

Main progress log:

```bash
tail -f \
set/train/result/05_modeling/batch_run_loo/05_batch_nohup.log
```

Current process:

```bash
PID=$(cat \
set/train/result/05_modeling/batch_run_loo/05_batch_nohup.pid)

ps -p "${PID}" -o pid,etime,%cpu,%mem,cmd
```

Status table:

```bash
column -s, -t \
set/train/result/05_modeling/batch_run_loo/05_batch_run_status.csv \
| less -S
```

Count task states:

```bash
python3 - <<'PY'
import pandas as pd

f = (
    "set/train/result/05_modeling/batch_run_loo/"
    "05_batch_run_status.csv"
)
df = pd.read_csv(f)
print(df["status"].value_counts(dropna=False))
PY
```

Every task also has separate logs:

```text
set/train/result/05_modeling/batch_run_loo/task_logs/
```

For example:

```bash
tail -f \
set/train/result/05_modeling/batch_run_loo/task_logs/\
repeat_01_fold_02.stdout.log
```

## Resume after interruption

Run the same command again:

```bash
nohup python3 -u set/scripts/run_05_batch_outer_trials_loo.py \
  --repeats 1-20 \
  --folds 1-5 \
  --public-feature-set raw_bilateral \
  > set/train/result/05_modeling/batch_run_loo/05_batch_nohup.log \
  2>&1 &
```

The runner checks every task directory. Tasks are skipped only if:

- all expected output files exist;
- the metrics file contains exactly M0–M3;
- all four final fits converged;
- configuration repeat/fold values match the task.

Failed or incomplete tasks are rerun automatically.

## Existing single-fold result

Because `repeat_01_fold_01` already completed successfully, the batch runner
will normally report:

```text
repeat_01_fold_01: SKIP (already complete)
```

It then starts from the next incomplete task.

## Important outputs

Batch-level files:

```text
set/train/result/05_modeling/batch_run_loo/
├── 05_batch_nohup.log
├── 05_batch_nohup.pid
├── 05_batch_run_status.csv
├── 05_batch_run_summary.md
└── task_logs/
```

Per-task modeling results remain under:

```text
set/train/result/05_modeling/single_outer_trial_loo/
├── repeat_01_fold_01/
├── repeat_01_fold_02/
...
└── repeat_20_fold_05/
```

## Stop the batch

First inspect the process:

```bash
PID=$(cat \
set/train/result/05_modeling/batch_run_loo/05_batch_nohup.pid)

ps -p "${PID}" -o pid,etime,%cpu,%mem,cmd
```

Then request a normal stop:

```bash
kill "${PID}"
```

The currently running child worker may continue briefly. Check:

```bash
pgrep -af run_05_single_outer_trial_loo.py
```

Avoid `kill -9` unless the process cannot be stopped normally.

## Options

Run only selected tasks:

```bash
python3 set/scripts/run_05_batch_outer_trials_loo.py \
  --repeats 1-3 \
  --folds 1,3,5
```

Force rerun of already complete tasks:

```bash
python3 set/scripts/run_05_batch_outer_trials_loo.py \
  --rerun-completed
```

Stop on the first failure:

```bash
python3 set/scripts/run_05_batch_outer_trials_loo.py \
  --stop-on-error
```

By default, a failed task is logged and the runner continues with the next
task.
