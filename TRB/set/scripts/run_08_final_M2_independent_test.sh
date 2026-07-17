#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/users/chenhaisheng/RA-ILD/TRB"
LOG="${ROOT}/set/test/result/08_final_M2_validation_run.log"

cd "${ROOT}"
mkdir -p "${ROOT}/set/test/result"

python3 -u set/scripts/run_08_final_M2_independent_test.py \
  --locked-alpha 0.5 \
  --locked-lambda 30 \
  --locked-threshold 0.509337 \
  2>&1 | tee "${LOG}"

echo
echo "Independent validation completed."
echo "Log: ${LOG}"
echo "Results: ${ROOT}/set/test/result/08_final_M2_validation"
