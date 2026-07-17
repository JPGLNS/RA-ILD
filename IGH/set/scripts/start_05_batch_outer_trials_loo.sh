#!/usr/bin/env bash
set -euo pipefail

# Environment variables may override these defaults for testing or alternate runs.
PROJECT_ROOT="${PROJECT_ROOT:-/data/users/chenhaisheng/RA-ILD/IGH}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REPEATS="${REPEATS:-1-20}"
FOLDS="${FOLDS:-1-5}"
PUBLIC_FEATURE_SET="${PUBLIC_FEATURE_SET:-raw_bilateral}"

BATCH_SCRIPT="${BATCH_SCRIPT:-${PROJECT_ROOT}/set/scripts/run_05_batch_outer_trials_loo.py}"
WORKER_SCRIPT="${WORKER_SCRIPT:-${PROJECT_ROOT}/set/scripts/run_05_single_outer_trial_loo.py}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/set/train/result/05_modeling/single_outer_trial_loo}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/set/train/result/05_modeling/batch_run_loo}"
MAIN_LOG="${LOG_DIR}/05_batch_nohup.log"
PID_FILE="${LOG_DIR}/05_batch_nohup.pid"
STATUS_FILE="${LOG_DIR}/05_batch_run_status.csv"
SUMMARY_FILE="${LOG_DIR}/05_batch_run_summary.md"

for required in "${BATCH_SCRIPT}" "${WORKER_SCRIPT}"; do
    if [[ ! -f "${required}" ]]; then
        echo "ERROR: required script not found: ${required}" >&2
        exit 1
    fi
done

mkdir -p "${LOG_DIR}"

# Refuse to launch a duplicate process recorded by this start script.
if [[ -s "${PID_FILE}" ]]; then
    OLD_PID="$(tr -d '[:space:]' < "${PID_FILE}")"
    if [[ "${OLD_PID}" =~ ^[0-9]+$ ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "ERROR: an existing batch process is still running (PID=${OLD_PID})." >&2
        echo "Monitor with:" >&2
        echo "  ps -p ${OLD_PID} -o pid,etime,%cpu,%mem,cmd" >&2
        echo "  tail -f ${MAIN_LOG}" >&2
        exit 1
    fi
    echo "Removing stale PID file: ${PID_FILE}"
    rm -f "${PID_FILE}"
fi

cd "${PROJECT_ROOT}"

nohup "${PYTHON_BIN}" -u "${BATCH_SCRIPT}" \
    --worker-script "${WORKER_SCRIPT}" \
    --result-root "${RESULT_ROOT}" \
    --batch-dir "${LOG_DIR}" \
    --repeats "${REPEATS}" \
    --folds "${FOLDS}" \
    --public-feature-set "${PUBLIC_FEATURE_SET}" \
    > "${MAIN_LOG}" 2>&1 &

PID=$!
echo "${PID}" > "${PID_FILE}"

# Detect immediate startup failures such as an existing scheduler lock.
sleep 1
if ! kill -0 "${PID}" 2>/dev/null; then
    set +e
    wait "${PID}"
    RC=$?
    set -e
    if [[ "${RC}" -ne 0 ]]; then
        echo "ERROR: batch process exited immediately with status ${RC}. Last log lines:" >&2
        tail -n 40 "${MAIN_LOG}" >&2 || true
        exit "${RC}"
    fi
    echo "IGH LOO batch finished before the startup check completed."
else
    echo "IGH LOO batch started."
fi
echo "PID: ${PID}"
echo "Tasks: repeats=${REPEATS}, folds=${FOLDS}"
echo "Public feature set: ${PUBLIC_FEATURE_SET}"
echo "Main log: ${MAIN_LOG}"
echo "Status: ${STATUS_FILE}"
echo "Summary: ${SUMMARY_FILE}"
echo
echo "Monitor with:"
echo "  ps -p ${PID} -o pid,etime,%cpu,%mem,cmd"
echo "  tail -f ${MAIN_LOG}"
echo "  column -s, -t < ${STATUS_FILE} | tail -n 20"
echo
echo "Recover PID later with:"
echo "  PID=\$(cat ${PID_FILE})"
