#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/users/chenhaisheng/RA-ILD/TRB"
SCRIPT_DIR="${ROOT}/set/scripts"
OUTPUT_DIR="${ROOT}/set/train/result/03_public_features/threshold_stability"
CACHE_MANIFEST="${OUTPUT_DIR}/cache/cache_manifest.json"

REUSE_CACHE=()
if [[ -f "${CACHE_MANIFEST}" ]]; then
    echo "[INFO] 检测到兼容性将由Python脚本验证的现有cache，尝试复用。"
    REUSE_CACHE=(--reuse-cache)
else
    echo "[INFO] 未检测到cache，首次运行将读取全部训练AA表并建立稀疏cache。"
fi

python3 "${SCRIPT_DIR}/evaluate_03_public_thresholds.py" \
  --aa-dir "${ROOT}/set/train/result/01_AA_clone_table" \
  --summary-file "${ROOT}/set/train/result/01_AA_clone_table/01_AA_clone_table_summary.csv" \
  --metadata "${ROOT}/set/train/metadata_train_70.csv" \
  --id-col libraryid \
  --cohort-col cohort \
  --batch-col batch \
  --strata-cols cohort,batch \
  --output-dir "${OUTPUT_DIR}" \
  --iterations 100 \
  --reference-fraction 0.80 \
  --seed 20260711 \
  --candidate-min-total-count 2 \
  --sort-memory 4G \
  --sort-parallel 4 \
  "${REUSE_CACHE[@]}" \
  --overwrite
