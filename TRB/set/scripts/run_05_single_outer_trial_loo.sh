#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/users/chenhaisheng/RA-ILD/TRB"

PYTHONPYCACHEPREFIX=/tmp/pycache_ra_ild \
python3 -m py_compile \
  "${ROOT}/set/scripts/run_05_single_outer_trial_loo.py"

python3 "${ROOT}/set/scripts/run_05_single_outer_trial_loo.py" \
  --outer-repeat 1 \
  --outer-fold 1 \
  --public-feature-set raw_bilateral \
  --overwrite
