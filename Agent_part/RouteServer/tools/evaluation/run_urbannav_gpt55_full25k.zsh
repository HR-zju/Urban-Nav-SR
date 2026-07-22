#!/usr/bin/env zsh
set -euo pipefail

# Run the complete UrbanNav v3 23-city 25K benchmark with Crazyrouter gpt-5.5.
# All API keys are loaded in memory by the existing prediction runner; key
# values are never copied into generated worker scripts or logs.
#
# Usage:
#   tools/evaluation/run_urbannav_gpt55_full25k.zsh preflight
#   tools/evaluation/run_urbannav_gpt55_full25k.zsh dry-run
#   tools/evaluation/run_urbannav_gpt55_full25k.zsh start
#   tools/evaluation/run_urbannav_gpt55_full25k.zsh status
#   tools/evaluation/run_urbannav_gpt55_full25k.zsh tail
#   tools/evaluation/run_urbannav_gpt55_full25k.zsh stop
#
# Useful overrides:
#   CONCURRENCY=16 RESET_EVERY_CASES=250 RESTART=1 ... start
#   MAX_CASES=20 RUN_NAME=gpt55_smoke20 ... start

export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"

SCRIPT_PATH="${0:A}"
ROUTE_ROOT="${ROUTE_ROOT:-${SCRIPT_PATH:h:h:h}}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${ROUTE_ROOT}/../../.." && pwd)}"

ACTION="${1:-start}"
MODEL="${MODEL:-gpt-5.5}"
SAFE_MODEL="${MODEL//[^A-Za-z0-9]/_}"
RUN_NAME="${RUN_NAME:-${SAFE_MODEL}_crazyrouter_strict_full25k_v3_23cities}"
SESSION="${SESSION:-urbannav_${SAFE_MODEL}_full25k}"
RUN_ROOT="${RUN_ROOT:-${ROUTE_ROOT}/data/benchmark/urbannav_module1_runs/${RUN_NAME}}"

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/UrbanNav/bin/python}"
KEY_FILE="${KEY_FILE:-${PROJECT_ROOT}/key.txt}"
DATA_ROOT="${DATA_ROOT:-${ROUTE_ROOT}/data/benchmark/urbannav_v3_osm_local_osrm_foot_final25k_v3_23cities}"
DATASET="${DATASET:-${DATA_ROOT}/task_gt.jsonl}"
POI_CACHE="${POI_CACHE:-${DATA_ROOT}/poi_cache.jsonl}"
ROUTE_CACHE="${ROUTE_CACHE:-${DATA_ROOT}/route_cache.jsonl}"

LLM_PROVIDER_NAME="${LLM_PROVIDER_NAME:-openai}"
LLM_BASE_URL="${LLM_BASE_URL:-https://cn.crazyrouter.com/v1}"
LLM_API_STYLE="${LLM_API_STYLE:-chat_completions}"
LLM_TIMEOUT_S="${LLM_TIMEOUT_S:-120}"
LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-3}"
LLM_REASONING_EFFORT="${LLM_REASONING_EFFORT:-low}"
LLM_RESPONSE_FORMAT_JSON="${LLM_RESPONSE_FORMAT_JSON:-1}"
CONCURRENCY="${CONCURRENCY:-32}"
CANDIDATE_LIMIT="${CANDIDATE_LIMIT:-10}"
SELECTION_STRATEGY="${SELECTION_STRATEGY:-semantic_then_route}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-100}"
HEARTBEAT_INTERVAL_S="${HEARTBEAT_INTERVAL_S:-20}"
CHUNK_SIZE="${CHUNK_SIZE:-25}"
RESET_EVERY_CASES="${RESET_EVERY_CASES:-500}"
RESET_SLEEP_S="${RESET_SLEEP_S:-3}"
MAX_PREDICTION_EXCEPTION_RATE="${MAX_PREDICTION_EXCEPTION_RATE:-0.05}"
EXCEPTION_RATE_MIN_SAMPLES="${EXCEPTION_RATE_MIN_SAMPLES:-100}"
MAX_CASES="${MAX_CASES:-}"
RESTART="${RESTART:-0}"

PREDICTIONS_JSONL="${RUN_ROOT}/raw_predictions.jsonl"
PREDICT_MANIFEST="${RUN_ROOT}/prediction_manifest.json"
PREDICT_LOG="${RUN_ROOT}/predict.log"
EVAL_DIR="${RUN_ROOT}/eval"
EVAL_LOG="${RUN_ROOT}/eval.log"
WORKER_SCRIPT="${RUN_ROOT}/run_worker.zsh"

_key_count() {
  "${PYTHON_BIN}" - "${KEY_FILE}" <<'PY'
from pathlib import Path
import sys
keys = []
seen = set()
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    key = raw.strip()
    if key and not key.startswith("#") and key not in seen:
        seen.add(key)
        keys.append(key)
print(len(keys))
PY
}

_print_config() {
  local key_count
  key_count="$(_key_count)"
  print "route_root=${ROUTE_ROOT}"
  print "run_root=${RUN_ROOT}"
  print "session=${SESSION}"
  print "dataset=${DATASET}"
  print "dataset_rows=$(wc -l < "${DATASET}" | tr -d ' ')"
  print "model=${MODEL}"
  print "llm_base_url=${LLM_BASE_URL}"
  print "llm_api_style=${LLM_API_STYLE}"
  print "key_count=${key_count}"
  print "concurrency=${CONCURRENCY}"
  print "approx_concurrency_per_key=$(( (${CONCURRENCY} + ${key_count} - 1) / ${key_count} ))"
  print "reasoning_effort=${LLM_REASONING_EFFORT}"
  print "chunk_size=${CHUNK_SIZE}"
  print "reset_every_cases=${RESET_EVERY_CASES}"
  print "max_prediction_exception_rate=${MAX_PREDICTION_EXCEPTION_RATE}"
  if [[ -n "${MAX_CASES}" ]]; then
    print "max_cases=${MAX_CASES}"
  fi
}

_validate_inputs() {
  [[ -x "${PYTHON_BIN}" ]] || { print -u2 "python not executable: ${PYTHON_BIN}"; exit 1; }
  for required_path in "${KEY_FILE}" "${DATASET}" "${POI_CACHE}" "${ROUTE_CACHE}"; do
    [[ -f "${required_path}" ]] || { print -u2 "required file not found: ${required_path}"; exit 1; }
  done
  local key_count dataset_rows
  key_count="$(_key_count)"
  dataset_rows="$(wc -l < "${DATASET}" | tr -d ' ')"
  (( key_count > 0 )) || { print -u2 "no API keys found in ${KEY_FILE}"; exit 1; }
  if [[ -z "${MAX_CASES}" ]] && (( dataset_rows != 25000 )); then
    print -u2 "full benchmark requires exactly 25000 dataset rows; found ${dataset_rows}"
    exit 1
  fi
  if (( CONCURRENCY < key_count )); then
    print -u2 "CONCURRENCY=${CONCURRENCY} is lower than key_count=${key_count}; not all keys can be active"
    exit 1
  fi
  local key_mode
  key_mode="$(stat -c '%a' "${KEY_FILE}" 2>/dev/null || true)"
  if [[ -n "${key_mode}" && "${key_mode}" != "600" && "${key_mode}" != "400" ]]; then
    print -u2 "warning: key file permissions are ${key_mode}; recommend: chmod 600 '${KEY_FILE}'"
  fi
}

_preflight() {
  _validate_inputs
  mkdir -p "${RUN_ROOT}"
  cd "${ROUTE_ROOT}"
  LLM_RESPONSE_FORMAT_JSON="${LLM_RESPONSE_FORMAT_JSON}" \
  LLM_REASONING_EFFORT="${LLM_REASONING_EFFORT}" \
  LLM_API_STYLE="${LLM_API_STYLE}" \
  LLM_API_STYLE_FALLBACK=0 \
  PYTHONPATH=src "${PYTHON_BIN}" tools/evaluation/generate_urbannav_external_predictions.py \
    --cases "${DATASET}" \
    --output-jsonl "${RUN_ROOT}/preflight_unused.jsonl" \
    --backend replay \
    --poi-cache "${POI_CACHE}" \
    --route-cache "${ROUTE_CACHE}" \
    --use-llm-intent \
    --use-llm-reranker \
    --strict-llm \
    --api-keys-file "${KEY_FILE}" \
    --llm-provider "${LLM_PROVIDER_NAME}" \
    --llm-model "${MODEL}" \
    --llm-base-url "${LLM_BASE_URL}" \
    --llm-timeout-s "${LLM_TIMEOUT_S}" \
    --llm-max-retries "${LLM_MAX_RETRIES}" \
    --preflight-only
}

_write_worker_script() {
  mkdir -p "${RUN_ROOT}"
  cat > "${WORKER_SCRIPT}" <<EOF
#!/usr/bin/env zsh
set -euo pipefail
cd "${ROUTE_ROOT}"
trap 'status=\$?; print "[urbannav-gpt55] worker exit status=\${status} at \$(date "+%Y-%m-%d %H:%M:%S")" 2>&1 | tee -a "${PREDICT_LOG}"' EXIT
export LLM_RESPONSE_FORMAT_JSON="${LLM_RESPONSE_FORMAT_JSON}"
export LLM_REASONING_EFFORT="${LLM_REASONING_EFFORT}"
export LLM_API_STYLE="${LLM_API_STYLE}"
export LLM_API_STYLE_FALLBACK=0

{
  print "[urbannav-gpt55] started at \$(date '+%Y-%m-%d %H:%M:%S')"
  print "[urbannav-gpt55] model=${MODEL}"
  print "[urbannav-gpt55] dataset=${DATASET}"
  print "[urbannav-gpt55] concurrency=${CONCURRENCY}"
  print "[urbannav-gpt55] key_count=\$(grep -cv '^[[:space:]]*\(#\|\$\)' '${KEY_FILE}')"
  print "[urbannav-gpt55] reset_every_cases=${RESET_EVERY_CASES}"
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
    --llm-model "${MODEL}" \\
    --llm-base-url "${LLM_BASE_URL}" \\
    --llm-timeout-s "${LLM_TIMEOUT_S}" \\
    --llm-max-retries "${LLM_MAX_RETRIES}" \\
    --progress-interval "${PROGRESS_INTERVAL}" \\
    --heartbeat-interval-s "${HEARTBEAT_INTERVAL_S}" \\
    --chunk-size "${CHUNK_SIZE}" \\
    --max-prediction-exception-rate "${MAX_PREDICTION_EXCEPTION_RATE}" \\
    --exception-rate-min-samples "${EXCEPTION_RATE_MIN_SAMPLES}" \\
    --resume \\
EOF
  if [[ -n "${MAX_CASES}" ]]; then
    print "    --max-cases \"${MAX_CASES}\" \\\\" >> "${WORKER_SCRIPT}"
  fi
  cat >> "${WORKER_SCRIPT}" <<EOF
    "\${extra_args[@]}" \\
    --output-jsonl "${PREDICTIONS_JSONL}" \\
    --manifest-json "${PREDICT_MANIFEST}" \\
    2>&1 | tee -a "${PREDICT_LOG}"
}

target_total=\$(wc -l < "${DATASET}" | tr -d ' ')
[[ -n "${MAX_CASES}" ]] && target_total="${MAX_CASES}"

while true; do
  current_rows=0
  [[ -f "${PREDICTIONS_JSONL}" ]] && current_rows=\$(wc -l < "${PREDICTIONS_JSONL}" | tr -d ' ')
  if (( current_rows >= target_total )); then
    print "[urbannav-gpt55] predictions complete: \${current_rows}/\${target_total}" 2>&1 | tee -a "${PREDICT_LOG}"
    break
  fi
  print "[urbannav-gpt55] segment start rows=\${current_rows}/\${target_total}" 2>&1 | tee -a "${PREDICT_LOG}"
  if (( ${RESET_EVERY_CASES} > 0 )); then
    run_prediction_segment --stop-after-pending "${RESET_EVERY_CASES}"
  else
    run_prediction_segment
  fi
  new_rows=\$(wc -l < "${PREDICTIONS_JSONL}" | tr -d ' ')
  if (( new_rows <= current_rows )); then
    print -u2 "[urbannav-gpt55] no progress: \${new_rows} <= \${current_rows}"
    exit 3
  fi
  (( new_rows >= target_total )) && break
  sleep "${RESET_SLEEP_S}"
done

PYTHONPATH=src "${PYTHON_BIN}" tools/evaluation/run_urbannav_module1_benchmark.py \\
  --dataset "${DATASET}" \\
  --runner external \\
  --predictions "${PREDICTIONS_JSONL}" \\
  --output-dir "${EVAL_DIR}" \\
  --run-id "${RUN_NAME}" \\
  --max-prediction-exception-rate "${MAX_PREDICTION_EXCEPTION_RATE}" \\
  --prediction-exception-min-samples "${EXCEPTION_RATE_MIN_SAMPLES}" \\
  2>&1 | tee "${EVAL_LOG}"
EOF
  chmod 700 "${WORKER_SCRIPT}"
}

_status() {
  local target_total current_rows
  target_total="$(wc -l < "${DATASET}" | tr -d ' ')"
  [[ -n "${MAX_CASES}" ]] && target_total="${MAX_CASES}"
  current_rows=0
  [[ -f "${PREDICTIONS_JSONL}" ]] && current_rows="$(wc -l < "${PREDICTIONS_JSONL}" | tr -d ' ')"
  print "== progress =="
  print "${current_rows}/${target_total} predictions"
  print "== tmux =="
  tmux ls 2>/dev/null | grep "${SESSION}" || true
  print "== process =="
  ps -eo pid,ppid,stat,etime,cmd | grep -E 'generate_urbannav_external_predictions|run_urbannav_module1_benchmark' | grep -v grep || true
  print "== latest prediction log =="
  tail -30 "${PREDICT_LOG}" 2>/dev/null || true
  print "== evaluation =="
  if [[ -f "${EVAL_DIR}/summary.json" ]]; then
    "${PYTHON_BIN}" - "${EVAL_DIR}/summary.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps(data.get("summary", data), ensure_ascii=False, indent=2, sort_keys=True))
PY
  else
    print "not finished"
  fi
}

case "${ACTION}" in
  preflight)
    _print_config
    _preflight
    ;;
  dry-run)
    _validate_inputs
    _print_config
    _write_worker_script
    zsh -n "${WORKER_SCRIPT}"
    print "worker_script=${WORKER_SCRIPT}"
    print "dry-run complete; no API call and no tmux session started"
    ;;
  start)
    _validate_inputs
    _print_config
    _write_worker_script
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
      if [[ "${RESTART}" == "1" ]]; then
        tmux kill-session -t "${SESSION}"
      else
        print -u2 "tmux session already exists: ${SESSION}"
        exit 1
      fi
    fi
    mkdir -p "${RUN_ROOT}"
    touch "${PREDICT_LOG}"
    tmux new-session -d -s "${SESSION}" "zsh '${WORKER_SCRIPT}'"
    print "started tmux session: ${SESSION}"
    print "status: ${SCRIPT_PATH} status"
    print "log:    ${SCRIPT_PATH} tail"
    ;;
  status)
    _status
    ;;
  tail)
    tail -f "${PREDICT_LOG}"
    ;;
  stop)
    tmux kill-session -t "${SESSION}" 2>/dev/null || true
    print "stopped ${SESSION}; predictions remain resumable"
    ;;
  *)
    print -u2 "usage: ${SCRIPT_PATH} {preflight|dry-run|start|status|tail|stop}"
    exit 2
    ;;
esac
