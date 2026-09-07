#!/usr/bin/env bash

set -u

cd /data/users/chenhaisheng/RA-ILD || exit 1

CONFIG_DIR="TRB/set/configs/experiments/layer1_all128_top10000_static_v1"

TRAIN_STEP02="TRB/set/train/result/feature_framework/top10000/advanced_unweighted500/02_sample_level_features"

TEST_STEP02="TRB/set/test/result/feature_framework/top10000/advanced_unweighted500/02_sample_level_features"

LOG_DIR="TRB/result/feature_framework/batch_logs/layer1_all128_top10000_static_v1/full"

STATUS_DIR="TRB/result/feature_framework/batch_logs/layer1_all128_top10000_static_v1/status"

MAX_EXPERIMENTS=5
WORKERS=10

mkdir -p "$LOG_DIR"
mkdir -p "$STATUS_DIR"

rm -f "$STATUS_DIR"/*.status

# Prevent nested BLAS/OpenMP oversubscription
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "============================================================"
echo "Layer 1 complete static architecture search"
echo "Architectures             : 128"
echo "Repeats per architecture  : 100"
echo "Inner CV folds            : 5"
echo "Max concurrent experiments: $MAX_EXPERIMENTS"
echo "Workers per experiment    : $WORKERS"
echo "Approx max workers        : $((MAX_EXPERIMENTS * WORKERS))"
echo "Started                   : $(date '+%F %T')"
echo "============================================================"

run_one() {
    config="$1"

    name="$(basename "$config" .yaml)"
    log="$LOG_DIR/${name}.log"
    status_file="$STATUS_DIR/${name}.status"

    echo "[$(date '+%F %T')] START $name"

    if python3 TRB/set/scripts/run_trb_experiment.py \
        --config "$config" \
        --train-step02-dir "$TRAIN_STEP02" \
        --test-step02-dir "$TEST_STEP02" \
        --workers "$WORKERS" \
        > "$log" 2>&1
    then
        echo "PASS" > "$status_file"
        echo "[$(date '+%F %T')] PASS  $name"
    else
        echo "FAIL" > "$status_file"
        echo "[$(date '+%F %T')] FAIL  $name"
    fi
}

for config in "$CONFIG_DIR"/S*.yaml
do
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_EXPERIMENTS" ]
    do
        sleep 5
    done

    run_one "$config" &
done

wait

PASS_COUNT=$(grep -l '^PASS$' "$STATUS_DIR"/*.status 2>/dev/null | wc -l)
FAIL_COUNT=$(grep -l '^FAIL$' "$STATUS_DIR"/*.status 2>/dev/null | wc -l)

echo ""
echo "============================================================"
echo "LAYER1 ALL128 BATCH FINISHED"
echo "Finished: $(date '+%F %T')"
echo "PASS: $PASS_COUNT / 128"
echo "FAIL: $FAIL_COUNT / 128"
echo "============================================================"
