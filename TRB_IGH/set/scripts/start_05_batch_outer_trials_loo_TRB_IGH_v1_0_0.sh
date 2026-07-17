#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/users/chenhaisheng/RA-ILD/TRB_IGH"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNNER="${ROOT}/set/scripts/run_05_batch_outer_trials_loo_TRB_IGH_v1_0_0.py"
BATCH_DIR="${ROOT}/set/train/result/05_modeling/batch_run_loo_TRB_IGH"
LOG_FILE="${BATCH_DIR}/05_batch_nohup.log"
PID_FILE="${BATCH_DIR}/05_batch_nohup.pid"

mkdir -p "${BATCH_DIR}"

if [[ ! -f "${RUNNER}" ]]; then
    echo "ERROR: batch runner not found: ${RUNNER}" >&2
    exit 1
fi

if [[ -f "${PID_FILE}" ]]; then
    old_pid="$(tr -d '[:space:]' < "${PID_FILE}" || true)"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
        echo "ERROR: an existing batch process is still running: PID=${old_pid}" >&2
        echo "Check with: ps -p ${old_pid} -o pid,etime,%cpu,%mem,cmd" >&2
        exit 1
    fi
    rm -f "${PID_FILE}"
fi

cd "${ROOT}"

nohup "${PYTHON_BIN}" -u "${RUNNER}" \
    --public-feature-set raw_bilateral \
    "$@" \
    > "${LOG_FILE}" 2>&1 &

pid=$!
echo "${pid}" > "${PID_FILE}"

# Catch immediate launch failures without waiting for the long batch job.
sleep 1
if ! kill -0 "${pid}" 2>/dev/null; then
    echo "ERROR: batch process exited immediately. Inspect: ${LOG_FILE}" >&2
    tail -n 50 "${LOG_FILE}" >&2 || true
    exit 1
fi

echo "Paired TRB+IGH batch run started."
echo "PID: ${pid}"
echo "PID file: ${PID_FILE}"
echo "Main log: ${LOG_FILE}"
echo
echo "Monitor process:"
echo "  ps -p ${pid} -o pid,etime,%cpu,%mem,rss,cmd"
echo "Monitor batch progress:"
echo "  tail -f ${LOG_FILE}"
echo "Inspect active task:"
echo "  cat ${BATCH_DIR}/05_active_task.json"
echo "Inspect current worker stdout:"
echo "  active_log=\$(python3 -c 'import json; print(json.load(open(\"${BATCH_DIR}/05_active_task.json\"))[\"stdout_log\"])' 2>/dev/null) && tail -f \"\${active_log}\""
