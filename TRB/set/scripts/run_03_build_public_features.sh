#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/users/chenhaisheng/RA-ILD/TRB"
SCRIPT_DIR="${ROOT}/set/scripts"

TRAIN_METADATA="${ROOT}/set/train/metadata_train_70.csv"
TRAIN_AA_DIR="${ROOT}/set/train/result/01_AA_clone_table"
TRAIN_SUMMARY="${TRAIN_AA_DIR}/01_AA_clone_table_summary.csv"
TRAIN_OUTPUT="${ROOT}/set/train/result/03_public_features"
STABILITY_CONFIG="${TRAIN_OUTPUT}/threshold_stability/03_threshold_stability_configuration.json"
STABILITY_CACHE="${TRAIN_OUTPUT}/threshold_stability/cache"

TEST_METADATA="${ROOT}/set/test/metadata_test_30.csv"
TEST_AA_DIR="${ROOT}/set/test/result/01_AA_clone_table"
TEST_SUMMARY="${TEST_AA_DIR}/01_AA_clone_table_summary.csv"
TEST_OUTPUT="${ROOT}/set/test/result/03_public_features"

REFERENCE_FILE="${TRAIN_OUTPUT}/03_final_reference_public_sets.csv.gz"
REFERENCE_DEFINITION="${TRAIN_OUTPUT}/03_reference_definition.json"

python3 -m py_compile \
  "${SCRIPT_DIR}/public_feature_utils.py" \
  "${SCRIPT_DIR}/build_03_public_features.py"

echo "[1/2] 构建训练集03 public catalog、描述性特征和最终reference sets"
python3 "${SCRIPT_DIR}/build_03_public_features.py" \
  --mode train \
  --metadata "${TRAIN_METADATA}" \
  --summary-file "${TRAIN_SUMMARY}" \
  --aa-dir "${TRAIN_AA_DIR}" \
  --output-dir "${TRAIN_OUTPUT}" \
  --stability-configuration "${STABILITY_CONFIG}" \
  --stability-cache-dir "${STABILITY_CACHE}" \
  --scheme-name main \
  --catalog-chunk-size 100000 \
  --overwrite

echo "[2/2] 使用固定训练reference sets构建测试集03特征"
python3 "${SCRIPT_DIR}/build_03_public_features.py" \
  --mode test \
  --metadata "${TEST_METADATA}" \
  --summary-file "${TEST_SUMMARY}" \
  --aa-dir "${TEST_AA_DIR}" \
  --output-dir "${TEST_OUTPUT}" \
  --reference-file "${REFERENCE_FILE}" \
  --reference-definition "${REFERENCE_DEFINITION}" \
  --scheme-name main \
  --overwrite

echo "[DONE] 03 public features构建完成。"
