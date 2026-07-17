#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/users/chenhaisheng/RA-ILD/TRB"
OUT="${ROOT}/set/train/result/07_final_M2_model"
mkdir -p "${OUT}"
cd "${ROOT}"

nohup python3 -u set/scripts/run_07_final_M2_training.py \
  --tuning-mode stable_pairs \
  --candidate-pairs 0.9:10,0.5:30 \
  --selection-rule one_se \
  --threshold-rule youden \
  --overwrite \
  > "${OUT}/07_final_M2_nohup.log" 2>&1 &

PID=$!
echo "${PID}" > "${OUT}/07_final_M2_nohup.pid"
echo "Started final M2 training."
echo "PID: ${PID}"
echo "Log: ${OUT}/07_final_M2_nohup.log"
