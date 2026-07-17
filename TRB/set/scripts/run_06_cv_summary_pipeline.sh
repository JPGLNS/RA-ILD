#!/usr/bin/env bash
set -euo pipefail
ROOT="/data/users/chenhaisheng/RA-ILD/TRB"
cd "${ROOT}"
python3 set/scripts/collect_06_cv_results.py --overwrite
python3 set/scripts/plot_06_cv_model_metrics.py --overwrite
python3 set/scripts/analyze_06_cv_results.py --overwrite
