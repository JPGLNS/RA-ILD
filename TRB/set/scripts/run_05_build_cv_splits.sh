#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/users/chenhaisheng/RA-ILD/TRB"

python3 -m py_compile "${ROOT}/set/scripts/build_05_cv_splits.py"

python3 "${ROOT}/set/scripts/build_05_cv_splits.py" \
  --input "${ROOT}/set/train/result/04_final_feature_matrix/04_train_base_feature_matrix.csv" \
  --output-dir "${ROOT}/set/train/result/05_modeling/cv_splits" \
  --sample-id-col sample_id \
  --label-col cohort \
  --patient-col patient \
  --balance-cols batch,material,sex \
  --outer-folds 5 \
  --outer-repeats 20 \
  --inner-folds 5 \
  --outer-candidates 300 \
  --inner-candidates 100 \
  --seed 20260711 \
  --expected-samples 123 \
  --overwrite
