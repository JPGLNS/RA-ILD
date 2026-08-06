#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/data/users/chenhaisheng/RA-ILD}"
shift || true
cd "$ROOT"
python3 TRB/set/scripts_v2/run_enrich_dictionary_5fold.py \
  --project-root "$ROOT" \
  "$@"
