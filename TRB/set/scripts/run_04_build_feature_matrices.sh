#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/users/chenhaisheng/RA-ILD/TRB"

python3 -m py_compile "${ROOT}/set/scripts/build_04_final_feature_matrix.py"

python3 "${ROOT}/set/scripts/build_04_final_feature_matrix.py" \
  --mode both \
  --train-metadata "${ROOT}/set/train/metadata_train_70.csv" \
  --test-metadata "${ROOT}/set/test/metadata_test_30.csv" \
  --train-02 "${ROOT}/set/train/result/02_sample_level_features/02_sample_level_features_merged.csv" \
  --test-02 "${ROOT}/set/test/result/02_sample_level_features/02_sample_level_features_merged.csv" \
  --train-03-descriptive "${ROOT}/set/train/result/03_public_features/03_descriptive_public_features.csv" \
  --test-03-descriptive "${ROOT}/set/test/result/03_public_features/03_descriptive_public_features.csv" \
  --test-03-reference "${ROOT}/set/test/result/03_public_features/03_test_reference_public_features.csv" \
  --train-output-dir "${ROOT}/set/train/result/04_final_feature_matrix" \
  --test-output-dir "${ROOT}/set/test/result/04_final_feature_matrix" \
  --metadata-id-col libraryid \
  --metadata-output-cols cohort,patient,age,sex,material,batch \
  --expected-train-samples 123 \
  --expected-test-samples 51 \
  --expected-02-columns 1097 \
  --expected-03-descriptive-columns 26 \
  --expected-test-reference-columns 19 \
  --overwrite
