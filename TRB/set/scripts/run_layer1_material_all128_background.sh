#!/usr/bin/env bash

set -u

cd /data/users/chenhaisheng/RA-ILD || exit 1

CONFIG_ROOT="TRB/set/configs/experiments/layer1_material_all128_top10000_static_v1"

TRAIN_STEP02="TRB/set/train/result/feature_framework/top10000/advanced_unweighted500/02_sample_level_features"

TEST_STEP02="TRB/set/test/result/feature_framework/top10000/advanced_unweighted500/02_sample_level_features"

BATCH_ROOT="TRB/result/feature_framework/batch_logs/layer1_material_all128_top10000_static_v1"

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
echo "Layer 1 Material-Stratified Static Search"
echo "Cohorts                  : PBMC + Buffycoat"
echo "Architectures / cohort   : 128"
echo "Total experiments        : 256"
echo "Algorithm                : Elastic Net"
echo "Repertoire               : Top10000"
echo "Clinical                 : Age + Sex"
echo "3-mer                    : OFF"
echo "RA dictionary            : OFF"
echo "Repeats / experiment     : 100"
echo "Inner CV folds           : 5"
echo "Concurrent experiments   : $MAX_EXPERIMENTS"
echo "Workers / experiment     : $WORKERS"
echo "Approx workers           : $((MAX_EXPERIMENTS * WORKERS))"
echo "Started                  : $(date '+%F %T')"
echo "============================================================"

run_one() {

    cohort="$1"
    config="$2"

    name="$(basename "$config" .yaml)"

    mkdir -p "$LOG_ROOT/$cohort"
    mkdir -p "$STATUS_ROOT/$cohort"

    log="$LOG_ROOT/$cohort/${name}.log"
    status_file="$STATUS_ROOT/$cohort/${name}.status"

    # Resume support
    if [ -f "$status_file" ] && \
       grep -q '^PASS$' "$status_file"
    then
        echo "[$(date '+%F %T')] SKIP  $cohort $name"
        return 0
    fi

    echo "[$(date '+%F %T')] START $cohort $name"

    if python3 TRB/set/scripts/run_trb_experiment.py \
        --config "$config" \
        --train-step02-dir "$TRAIN_STEP02" \
        --test-step02-dir "$TEST_STEP02" \
        --workers "$WORKERS" \
        > "$log" 2>&1
    then

        echo "PASS" > "$status_file"

        echo \
        "[$(date '+%F %T')] PASS  $cohort $name"

    else

        echo "FAIL" > "$status_file"

        echo \
        "[$(date '+%F %T')] FAIL  $cohort $name"

    fi
}

for cohort in pbmc buffycoat
do

    echo ""
    echo "===== COHORT: $cohort ====="

    for config in "$CONFIG_ROOT/$cohort"/S*.yaml
    do

        while [ "$(jobs -rp | wc -l)" -ge "$MAX_EXPERIMENTS" ]
        do
            sleep 5
        done

        run_one "$cohort" "$config" &

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

PBMC_PASS=$(
    find "$STATUS_ROOT/pbmc" \
        -type f \
        -name '*.status' \
        -exec grep -l '^PASS$' {} + \
        2>/dev/null |
    wc -l
)

BUFFY_PASS=$(
    find "$STATUS_ROOT/buffycoat" \
        -type f \
        -name '*.status' \
        -exec grep -l '^PASS$' {} + \
        2>/dev/null |
    wc -l
)

echo ""
echo "============================================================"
echo "MATERIAL ALL128 BATCH FINISHED"
echo "Finished        : $(date '+%F %T')"
echo "PBMC PASS       : $PBMC_PASS / 128"
echo "Buffycoat PASS  : $BUFFY_PASS / 128"
echo "TOTAL PASS      : $PASS_COUNT / 256"
echo "TOTAL FAIL      : $FAIL_COUNT / 256"
echo "============================================================"
