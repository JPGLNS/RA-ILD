#!/usr/bin/env bash

set -Eeuo pipefail

cd /data/users/chenhaisheng/RA-ILD

CONFIG_ROOT="TRB/set/configs/experiments/trb_only_layer1_all127_top10000_static_v1"

BATCH_ROOT="TRB/result/feature_framework/batch_logs/trb_only_layer1_all127_top10000_static_v1"

STATUS_ROOT="$BATCH_ROOT/status"

WORKER_SCRIPT="TRB/set/scripts/run_trb_only_one_experiment_v2.sh"

MAX_EXPERIMENTS="${MAX_EXPERIMENTS:-5}"
WORKERS="${WORKERS:-10}"

# 0 = no limit
# Used for safety pilot.
LAUNCH_LIMIT="${LAUNCH_LIMIT:-0}"

mkdir -p "$STATUS_ROOT"

# ------------------------------------------------------------
# Prevent nested BLAS/OpenMP oversubscription
# ------------------------------------------------------------

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Worker script reads this.
export WORKERS

# ------------------------------------------------------------
# IMPORTANT:
# Every experiment is launched via setsid.
#
# Therefore each experiment gets its own process group/session.
# If the master receives TERM/INT/HUP, we terminate ALL active
# experiment process groups, including run_06_unified_ml workers.
# ------------------------------------------------------------

stop_active_groups() {

    echo ""
    echo "[$(date '+%F %T')] MASTER CLEANUP START"

    ACTIVE_PIDS="$(jobs -pr || true)"

    if [ -n "$ACTIVE_PIDS" ]; then

        for pid in $ACTIVE_PIDS
        do
            echo \
                "TERM experiment process group PGID=$pid"

            kill -TERM -- "-$pid" \
                2>/dev/null || true
        done

        sleep 5

        ACTIVE_PIDS="$(jobs -pr || true)"

        if [ -n "$ACTIVE_PIDS" ]; then

            for pid in $ACTIVE_PIDS
            do
                echo \
                    "KILL experiment process group PGID=$pid"

                kill -KILL -- "-$pid" \
                    2>/dev/null || true
            done
        fi
    fi

    echo "[$(date '+%F %T')] MASTER CLEANUP END"
}

on_signal() {
    stop_active_groups
    exit 130
}

trap on_signal TERM INT HUP

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

echo "============================================================"
echo "TRB-only Layer1 SAFE MASTER V2"
echo "============================================================"
echo "Clinical predictors     : NONE"
echo "Repertoire              : Top10000"
echo "Algorithm               : Elastic Net"
echo "3-mer                   : OFF"
echo "RA dictionary           : OFF"
echo "Outer repeats           : 100"
echo "Inner CV                : 5"
echo "Max experiments         : $MAX_EXPERIMENTS"
echo "Workers / experiment    : $WORKERS"
echo "Launch limit            : $LAUNCH_LIMIT"
echo "Started                 : $(date '+%F %T')"
echo "============================================================"

ACTIVE=0
LAUNCHED=0
STOP_LAUNCH=0

for cohort in total pbmc buffycoat
do

    echo ""
    echo "===== COHORT: $cohort ====="

    for config in "$CONFIG_ROOT/$cohort"/S*.yaml
    do

        NAME="$(basename "$config" .yaml)"

        STATUS_FILE="$STATUS_ROOT/$cohort/${NAME}.status"

        # ----------------------------------------------------
        # Resume valid PASS jobs
        # ----------------------------------------------------

        if [ -f "$STATUS_FILE" ] && \
           grep -qx 'PASS' "$STATUS_FILE"
        then
            echo "SKIP PASS $cohort $NAME"
            continue
        fi

        # ----------------------------------------------------
        # Optional pilot launch limit
        # ----------------------------------------------------

        if [ "$LAUNCH_LIMIT" -gt 0 ] && \
           [ "$LAUNCHED" -ge "$LAUNCH_LIMIT" ]
        then
            STOP_LAUNCH=1
            break
        fi

        # ----------------------------------------------------
        # Hard concurrency gate
        #
        # wait -n waits for ONE existing child to terminate.
        # Therefore ACTIVE can never exceed MAX_EXPERIMENTS.
        # ----------------------------------------------------

        if [ "$ACTIVE" -ge "$MAX_EXPERIMENTS" ]
        then
            wait -n || true
            ACTIVE=$((ACTIVE - 1))
        fi

        echo \
            "LAUNCH $cohort $NAME | slot=$((ACTIVE + 1))/$MAX_EXPERIMENTS"

        # ----------------------------------------------------
        # Independent process group/session
        # ----------------------------------------------------

        setsid bash "$WORKER_SCRIPT" \
            "$cohort" \
            "$config" &

        ACTIVE=$((ACTIVE + 1))
        LAUNCHED=$((LAUNCHED + 1))

    done

    if [ "$STOP_LAUNCH" -eq 1 ]
    then
        break
    fi
done

# ------------------------------------------------------------
# Wait for remaining children
# ------------------------------------------------------------

while [ "$ACTIVE" -gt 0 ]
do
    wait -n || true
    ACTIVE=$((ACTIVE - 1))
done

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

PASS_COUNT=$(
    find "$STATUS_ROOT" \
        -type f \
        -name '*.status' \
        -exec grep -l '^PASS$' {} + \
        2>/dev/null |
    wc -l
)

FAIL_COUNT=$(
    find "$STATUS_ROOT" \
        -type f \
        -name '*.status' \
        -exec grep -l '^FAIL$' {} + \
        2>/dev/null |
    wc -l
)

RUNNING_COUNT=$(
    find "$STATUS_ROOT" \
        -type f \
        -name '*.status' \
        -exec grep -l '^RUNNING$' {} + \
        2>/dev/null |
    wc -l
)

INTERRUPTED_COUNT=$(
    find "$STATUS_ROOT" \
        -type f \
        -name '*.status' \
        -exec grep -l '^INTERRUPTED$' {} + \
        2>/dev/null |
    wc -l
)

echo ""
echo "============================================================"
echo "SAFE MASTER V2 FINISHED"
echo "============================================================"
echo "Launched    : $LAUNCHED"
echo "PASS        : $PASS_COUNT / 381"
echo "FAIL        : $FAIL_COUNT"
echo "RUNNING     : $RUNNING_COUNT"
echo "INTERRUPTED : $INTERRUPTED_COUNT"
echo "Finished    : $(date '+%F %T')"
echo "============================================================"
