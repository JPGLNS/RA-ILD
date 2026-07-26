#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/users/chenhaisheng/RA-ILD}"
SCRIPT="${SCRIPT:-$PROJECT_ROOT/IGH/set/scripts/IGH_Batch00_server_audit.py}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/IGH/set/audit_batch00}"

cd "$PROJECT_ROOT"

python3 "$SCRIPT" \
  --project-root "$PROJECT_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  "$@"

echo
echo "Batch 00 audit finished."
echo "Please return these two files first:"
echo "  $OUTPUT_DIR/IGH_upgrade_audit.md"
echo "  $OUTPUT_DIR/IGH_upgrade_audit_raw.json"
