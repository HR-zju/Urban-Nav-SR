#!/usr/bin/env zsh
set -euo pipefail

# Run UrbanNav Module 1 external Doubao/ARK prediction generation in tmux,
# then evaluate the saved UrbanNavPrediction JSONL with the offline evaluator.
#
# Usage:
#   tools/evaluation/run_urbannav_doubao_external_benchmark.zsh start
#   tools/evaluation/run_urbannav_doubao_external_benchmark.zsh status
#   tools/evaluation/run_urbannav_doubao_external_benchmark.zsh tail
#   tools/evaluation/run_urbannav_doubao_external_benchmark.zsh stop
#
# Environment overrides:
#   RUN_NAME=... SESSION=... CONCURRENCY=16 KEY_FILE=/path/key.txt
#   PYTHON_BIN=/home/ubuntu/anaconda3/envs/UrbanNav/bin/python
#   MAX_CASES=100  # optional smoke cap
#   RESET_EVERY_CASES=500  # optional process restart interval for long LLM runs
#   RESTART=1     # kill existing tmux session before start

export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"

SCRIPT_PATH="${0:A}"
ROUTE_ROOT="${ROUTE_ROOT:-${SCRIPT_PATH:h:h:h}}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${ROUTE_ROOT}/../.." && pwd)}"

ACTION="${1:-start}"
SESSION="${SESSION:-urbannav_doubao_strict_10k}"
RUN_NAME="${RUN_NAME:-doubao_external_replay_strict_10k_urbannavenv_v2}"
RUN_ROOT="${RUN_ROOT:-${ROUTE_ROOT}/data/benchmark/urbannav_module1_runs/${RUN_NAME}}"

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/UrbanNav/bin/python}"
KEY_FILE="${KEY_FILE:-${PROJECT_ROOT}/key.txt}"
DATASET="${DATASET:-${ROUTE_ROOT}/data/benchmark/urbannav_global_osm_local_full_en_v1/task_gt.jsonl}"
POI_CACHE="${POI_CACHE:-${ROUTE_ROOT}/data/benchmark/urbannav_global_osm_local_full_en_v1/poi_cache.jsonl}"
ROUTE_CACHE="${ROUTE_CACHE:-${ROUTE_ROOT}/data/benchmark/urbannav_global_osm_local_full_en_v1/route_cache.jsonl}"

CONCURRENCY="${CONCURRENCY:-16}"
CANDIDATE_LIMIT="${CANDIDATE_LIMIT:-10}"
SELECTION_STRATEGY="${SELECTION_STRATEGY:-semantic_then_route}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-100}"
HEARTBEAT_INTERVAL_S="${HEARTBEAT_INTERVAL_S:-15}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
RESET_EVERY_CASES="${RESET_EVERY_CASES:-0}"
RESET_SLEEP_S="${RESET_SLEEP_S:-3}"
LLM_PROVIDER_NAME="${LLM_PROVIDER_NAME:-ark}"
LLM_MODEL_NAME="${LLM_MODEL_NAME:-doubao-seed-2-0-lite-260428}"
LLM_BASE_URL="${LLM_BASE_URL:-https://ark.cn-beijing.volces.com/api/v3}"
LLM_TIMEOUT_S="${LLM_TIMEOUT_S:-30}"
LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-1}"
MAX_CASES="${MAX_CASES:-}"
RESTART="${RESTART:-0}"

PREDICTIONS_JSONL="${RUN_ROOT}/raw_predictions.jsonl"
PREDICT_MANIFEST="${RUN_ROOT}/prediction_manifest.json"
PREDICT_LOG="${RUN_ROOT}/predict.log"
EVAL_DIR="${RUN_ROOT}/eval"
EVAL_LOG="${RUN_ROOT}/eval.log"
WORKER_SCRIPT="${RUN_ROOT}/run_worker.zsh"

_print_config() {
  print "route_root=${ROUTE_ROOT}"
  print "project_root=${PROJECT_ROOT}"
  print "session=${SESSION}"
  print "run_root=${RUN_ROOT}"
  print "python=${PYTHON_BIN}"
  print "key_file=${KEY_FILE}"
  print "dataset=${DATASET}"
  print "concurrency=${CONCURRENCY}"
  print "model=${LLM_MODEL_NAME}"
  print "llm_timeout_s=${LLM_TIMEOUT_S}"
  print "llm_max_retries=${LLM_MAX_RETRIES}"
  print "chunk_size=${CHUNK_SIZE}"
  print "reset_every_cases=${RESET_EVERY_CASES}"
  print "reset_sleep_s=${RESET_SLEEP_S}"
  if [[ -n "${MAX_CASES}" ]]; then
    print "max_cases=${MAX_CASES}"
  fi
}

_validate_inputs() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    print -u2 "python not executable: ${PYTHON_BIN}"
    exit 1
  fi
  for required_path in "${KEY_FILE}" "${DATASET}" "${POI_CACHE}" "${ROUTE_CACHE}"; do
    if [[ ! -f "${required_path}" ]]; then
      print -u2 "required file not found: ${required_path}"
      exit 1
    fi
  done
}

_write_worker_script() {
  mkdir -p "${RUN_ROOT}"
  cat > "${WORKER_SCRIPT}" <<EOF
#!/usr/bin/env zsh
set -euo pipefail
cd "${ROUTE_ROOT}"
trap 'status=\$?; print "[urbannav-doubao] worker exit status=\${status} at \$(date "+%Y-%m-%d %H:%M:%S")" 2>&1 | tee -a "${PREDICT_LOG}"' EXIT

{
print "[urbannav-doubao] worker started at \$(date '+%Y-%m-%d %H:%M:%S')"
print "[urbannav-doubao] run_root=${RUN_ROOT}"
print "[urbannav-doubao] python=${PYTHON_BIN}"
print "[urbannav-doubao] dataset=${DATASET}"
print "[urbannav-doubao] concurrency=${CONCURRENCY}"
print "[urbannav-doubao] progress_interval=${PROGRESS_INTERVAL}"
print "[urbannav-doubao] heartbeat_interval_s=${HEARTBEAT_INTERVAL_S}"
print "[urbannav-doubao] chunk_size=${CHUNK_SIZE}"
print "[urbannav-doubao] reset_every_cases=${RESET_EVERY_CASES}"
print "[urbannav-doubao] reset_sleep_s=${RESET_SLEEP_S}"
} 2>&1 | tee -a "${PREDICT_LOG}"

run_prediction_segment() {
  local extra_args=("\$@")
  PYTHONUNBUFFERED=1 PYTHONPATH=src "${PYTHON_BIN}" tools/evaluation/generate_urbannav_external_predictions.py \\
    --cases "${DATASET}" \\
    --backend replay \\
    --poi-cache "${POI_CACHE}" \\
    --route-cache "${ROUTE_CACHE}" \\
    --concurrency "${CONCURRENCY}" \\
    --candidate-limit "${CANDIDATE_LIMIT}" \\
    --selection-strategy "${SELECTION_STRATEGY}" \\
    --use-llm-intent \\
    --use-llm-reranker \\
    --strict-llm \\
    --api-keys-file "${KEY_FILE}" \\
    --llm-provider "${LLM_PROVIDER_NAME}" \\
    --llm-model "${LLM_MODEL_NAME}" \\
    --llm-base-url "${LLM_BASE_URL}" \\
    --llm-timeout-s "${LLM_TIMEOUT_S}" \\
    --llm-max-retries "${LLM_MAX_RETRIES}" \\
    --progress-interval "${PROGRESS_INTERVAL}" \\
    --heartbeat-interval-s "${HEARTBEAT_INTERVAL_S}" \\
    --chunk-size "${CHUNK_SIZE}" \\
    --resume \\
EOF
  if [[ -n "${MAX_CASES}" ]]; then
    print "  --max-cases \"${MAX_CASES}\" \\\\" >> "${WORKER_SCRIPT}"
  fi
  cat >> "${WORKER_SCRIPT}" <<EOF
    "\${extra_args[@]}" \\
    --output-jsonl "${PREDICTIONS_JSONL}" \\
    --manifest-json "${PREDICT_MANIFEST}" \\
    2>&1 | tee -a "${PREDICT_LOG}"
}

target_total=\$(wc -l < "${DATASET}" | tr -d ' ')
if [[ -n "${MAX_CASES}" ]]; then
  target_total="${MAX_CASES}"
fi

if (( ${RESET_EVERY_CASES} > 0 )); then
  while true; do
    current_rows=0
    if [[ -f "${PREDICTIONS_JSONL}" ]]; then
      current_rows=\$(wc -l < "${PREDICTIONS_JSONL}" | tr -d ' ')
    fi
    if (( current_rows >= target_total )); then
      print "[urbannav-doubao] prediction complete before segment: \${current_rows}/\${target_total}" 2>&1 | tee -a "${PREDICT_LOG}"
      break
    fi
    print "[urbannav-doubao] starting reset segment at \$(date '+%Y-%m-%d %H:%M:%S'), rows=\${current_rows}/\${target_total}, limit=${RESET_EVERY_CASES}" 2>&1 | tee -a "${PREDICT_LOG}"
    run_prediction_segment --stop-after-pending "${RESET_EVERY_CASES}"
    new_rows=0
    if [[ -f "${PREDICTIONS_JSONL}" ]]; then
      new_rows=\$(wc -l < "${PREDICTIONS_JSONL}" | tr -d ' ')
    fi
    print "[urbannav-doubao] reset segment finished at \$(date '+%Y-%m-%d %H:%M:%S'), rows=\${new_rows}/\${target_total}" 2>&1 | tee -a "${PREDICT_LOG}"
    if (( new_rows <= current_rows )); then
      print "[urbannav-doubao] no new predictions were written; stopping to avoid an infinite loop" 2>&1 | tee -a "${PREDICT_LOG}"
      exit 3
    fi
    sleep "${RESET_SLEEP_S}"
  done
else
  run_prediction_segment
fi

PYTHONPATH=src "${PYTHON_BIN}" tools/evaluation/run_urbannav_module1_benchmark.py \\
  --dataset "${DATASET}" \\
  --runner external \\
  --predictions "${PREDICTIONS_JSONL}" \\
  --output-dir "${EVAL_DIR}" \\
  --run-id "${RUN_NAME}" \\
  2>&1 | tee "${EVAL_LOG}"
EOF
  chmod +x "${WORKER_SCRIPT}"
}

_status() {
  print "== tmux =="
  tmux ls 2>/dev/null || true
  print "\n== process =="
  ps -eo pid,ppid,stat,etime,cmd | grep -E 'generate_urbannav_external_predictions|run_urbannav_module1_benchmark' | grep -v grep || true
  print "\n== files =="
  ls -lh "${RUN_ROOT}" 2>/dev/null || true
  print "\n== prediction rows =="
  wc -l "${PREDICTIONS_JSONL}" 2>/dev/null || true
  print "\n== latest predict log =="
  tail -20 "${PREDICT_LOG}" 2>/dev/null || true
  print "\n== latest eval log =="
  tail -20 "${EVAL_LOG}" 2>/dev/null || true
}

case "${ACTION}" in
  dry-run)
    _print_config
    _validate_inputs
    _write_worker_script
    print "worker_script=${WORKER_SCRIPT}"
    print "dry-run complete; no tmux session started."
    ;;
  start)
    _print_config
    _validate_inputs
    _write_worker_script
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
      if [[ "${RESTART}" == "1" ]]; then
        tmux kill-session -t "${SESSION}"
      else
        print -u2 "tmux session already exists: ${SESSION}"
        print -u2 "Use RESTART=1 ${SCRIPT_PATH} start, or run: ${SCRIPT_PATH} status"
        exit 1
      fi
    fi
    : > "${PREDICT_LOG}"
    tmux new-session -d -s "${SESSION}" "zsh '${WORKER_SCRIPT}'"
    print "started tmux session: ${SESSION}"
    print "tail log: tail -f '${PREDICT_LOG}'"
    ;;
  status)
    _status
    ;;
  tail)
    tail -f "${PREDICT_LOG}"
    ;;
  stop)
    tmux kill-session -t "${SESSION}" 2>/dev/null || true
    print "stopped tmux session if it existed: ${SESSION}"
    ;;
  *)
    print -u2 "unknown action: ${ACTION}"
    print -u2 "usage: ${SCRIPT_PATH} {start|status|tail|stop|dry-run}"
    exit 2
    ;;
esac
