#!/usr/bin/env bash

set -u

cd /data/users/chenhaisheng/RA-ILD || exit 1

CONFIG_ROOT="TRB/set/configs/experiments/layer2_incremental_top10000_v1"

TRAIN_STEP02="TRB/set/train/result/feature_framework/top10000/advanced_unweighted500/02_sample_level_features"

TEST_STEP02="TRB/set/test/result/feature_framework/top10000/advanced_unweighted500/02_sample_level_features"

BATCH_ROOT="TRB/result/feature_framework/batch_logs/layer2_incremental_top10000_v1"

LOG_ROOT="$BATCH_ROOT/logs"
STATUS_ROOT="$BATCH_ROOT/status"

MAX_EXPERIMENTS=5
WORKERS=10

mkdir -p "$LOG_ROOT"
mkdir -p "$STATUS_ROOT"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "============================================================"
echo "Layer 2: incremental repertoire feature analysis"
echo "Architectures            : 4"
echo "Feature states           : 4"
echo "Algorithms               : 2"
echo "Total experiments        : 32"
echo "Repeats / experiment     : 100"
echo "Inner CV folds           : 5"
echo "Concurrent experiments   : $MAX_EXPERIMENTS"
echo "Workers / experiment     : $WORKERS"
echo "Approx workers           : $((MAX_EXPERIMENTS * WORKERS))"
echo "Started                  : $(date '+%F %T')"
echo "============================================================"

run_one() {

    algorithm="$1"
    config="$2"

    name="$(basename "$config" .yaml)"

    mkdir -p "$LOG_ROOT/$algorithm"
    mkdir -p "$STATUS_ROOT/$algorithm"

    log="$LOG_ROOT/$algorithm/${name}.log"
    status_file="$STATUS_ROOT/$algorithm/${name}.status"

    # Resume support
    if [ -f "$status_file" ] && \
       grep -q '^PASS$' "$status_file"
    then
        echo "[$(date '+%F %T')] SKIP  $algorithm $name"
        return 0
    fi

    echo "[$(date '+%F %T')] START $algorithm $name"

    if python3 TRB/set/scripts/run_trb_experiment.py \
        --config "$config" \
        --train-step02-dir "$TRAIN_STEP02" \
        --test-step02-dir "$TEST_STEP02" \
        --workers "$WORKERS" \
        > "$log" 2>&1
    then

        echo "PASS" > "$status_file"

        echo \
        "[$(date '+%F %T')] PASS  $algorithm $name"

    else

        echo "FAIL" > "$status_file"

        echo \
        "[$(date '+%F %T')] FAIL  $algorithm $name"

    fi
}

for algorithm in elastic_net lasso
do

    echo ""
    echo "===== ALGORITHM: $algorithm ====="

    for config in "$CONFIG_ROOT/$algorithm"/*.yaml
    do

        while [ "$(jobs -rp | wc -l)" -ge "$MAX_EXPERIMENTS" ]
        do
            sleep 5
        done

        run_one "$algorithm" "$config" &

    done
done

wait

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

echo ""
echo "============================================================"
echo "LAYER2 BATCH FINISHED"
echo "Finished : $(date '+%F %T')"
echo "PASS     : $PASS_COUNT / 32"
echo "FAIL     : $FAIL_COUNT / 32"
echo "============================================================"
