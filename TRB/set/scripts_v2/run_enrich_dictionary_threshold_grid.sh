#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash $0 /path/to/RA-ILD [grid options...]" >&2
  exit 2
fi

PROJECT_ROOT="${1%/}"
shift
exec python3 "$PROJECT_ROOT/TRB/set/scripts_v2/run_enrich_dictionary_threshold_grid.py" \
  --project-root "$PROJECT_ROOT" "$@"
