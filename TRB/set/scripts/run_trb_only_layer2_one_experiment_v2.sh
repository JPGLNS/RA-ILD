#!/usr/bin/env bash

set -Eeuo pipefail

COHORT="$1"
CONFIG="$2"

cd /data/users/chenhaisheng/RA-ILD

TRAIN_STEP02="TRB/set/train/result/feature_framework/top10000/advanced_unweighted500/02_sample_level_features"

TEST_STEP02="TRB/set/test/result/feature_framework/top10000/advanced_unweighted500/02_sample_level_features"

BATCH_ROOT="TRB/result/feature_framework/batch_logs/trb_only_layer2_top5_incremental_top10000_v1"

LOG_ROOT="$BATCH_ROOT/logs"
STATUS_ROOT="$BATCH_ROOT/status"

WORKERS="${WORKERS:-10}"

NAME="$(basename "$CONFIG" .yaml)"

LOG="$LOG_ROOT/$COHORT/${NAME}.log"
STATUS_FILE="$STATUS_ROOT/$COHORT/${NAME}.status"

mkdir -p "$LOG_ROOT/$COHORT"
mkdir -p "$STATUS_ROOT/$COHORT"

if [ -f "$STATUS_FILE" ] && \
   grep -qx 'PASS' "$STATUS_FILE"
then
    echo "[$(date '+%F %T')] SKIP $COHORT $NAME"
    exit 0
fi

echo "RUNNING" > "$STATUS_FILE"

on_interrupt() {
    echo "INTERRUPTED" > "$STATUS_FILE"
    exit 143
}

trap on_interrupt TERM INT HUP

echo "[$(date '+%F %T')] START $COHORT $NAME"

if python3 \
    TRB/set/scripts/run_trb_experiment.py \
    --config "$CONFIG" \
    --train-step02-dir "$TRAIN_STEP02" \
    --test-step02-dir "$TEST_STEP02" \
    --workers "$WORKERS" \
    > "$LOG" 2>&1
then

    echo "PASS" > "$STATUS_FILE"
    echo "[$(date '+%F %T')] PASS $COHORT $NAME"
    exit 0

else

    RC=$?

    echo "FAIL" > "$STATUS_FILE"
    echo "[$(date '+%F %T')] FAIL $COHORT $NAME RC=$RC"

    exit "$RC"
fi
