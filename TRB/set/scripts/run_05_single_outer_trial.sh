#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/users/chenhaisheng/RA-ILD/TRB"

python3 -m py_compile \
  "${ROOT}/set/scripts/run_05_single_outer_trial.py"

python3 "${ROOT}/set/scripts/run_05_single_outer_trial.py" \
  --outer-repeat 1 \
  --outer-fold 1 \
  --public-feature-set raw_bilateral \
  --overwrite
