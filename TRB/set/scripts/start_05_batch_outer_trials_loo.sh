#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/users/chenhaisheng/RA-ILD/TRB"
LOG_DIR="${ROOT}/set/train/result/05_modeling/batch_run_loo"
mkdir -p "${LOG_DIR}"

cd "${ROOT}"

nohup python3 -u set/scripts/run_05_batch_outer_trials_loo.py \
  --repeats 1-20 \
  --folds 1-5 \
  --public-feature-set raw_bilateral \
  > "${LOG_DIR}/05_batch_nohup.log" 2>&1 &

PID=$!
echo "${PID}" > "${LOG_DIR}/05_batch_nohup.pid"

echo "Batch started."
echo "PID: ${PID}"
echo "Main log: ${LOG_DIR}/05_batch_nohup.log"
echo "Status: ${LOG_DIR}/05_batch_run_status.csv"
echo
echo "Monitor with:"
echo "  tail -f ${LOG_DIR}/05_batch_nohup.log"
