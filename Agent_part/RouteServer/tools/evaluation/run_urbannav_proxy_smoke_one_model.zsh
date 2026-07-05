#!/usr/bin/env zsh
set -euo pipefail

# Run one UrbanNav proxy smoke model in a tmux session.
#
# Usage:
#   MODEL=deepseek-v4-pro tools/evaluation/run_urbannav_proxy_smoke_one_model.zsh start
#   MODEL=deepseek-v4-pro tools/evaluation/run_urbannav_proxy_smoke_one_model.zsh status
#
# This script is intentionally separate from the sequential multi-model runner
# so we can increase throughput without mixing model outputs.

export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"

SCRIPT_PATH="${0:A}"
ROUTE_ROOT="${ROUTE_ROOT:-${SCRIPT_PATH:h:h:h}}"

ACTION="${1:-start}"
MODEL="${MODEL:?MODEL is required, for example MODEL=qwen3.7-max}"
SAFE_MODEL="${MODEL//[^A-Za-z0-9]/_}"
SESSION="${SESSION:-urbannav_proxy_smoke_${SAFE_MODEL}}"
RUN_GROUP="${RUN_GROUP:-proxy_strict_200_smoke_20260701}"
RUN_ROOT="${RUN_ROOT:-${ROUTE_ROOT}/data/benchmark/urbannav_module1_runs/${RUN_GROUP}/${SAFE_MODEL}_strict_${SAFE_MODEL}_200_parallel}"

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/UrbanNav/bin/python}"
KEY_FILE="${KEY_FILE:-${ROUTE_ROOT}/data/benchmark/urbannav_module1_runs/local_proxy_dummy_key.txt}"
DATASET="${DATASET:-${ROUTE_ROOT}/data/benchmark/urbannav_v2_osm_local_osrm_foot_final_15k/task_gt.jsonl}"
POI_CACHE="${POI_CACHE:-${ROUTE_ROOT}/data/benchmark/urbannav_v2_osm_local_osrm_foot_final_15k/poi_cache.jsonl}"
ROUTE_CACHE="${ROUTE_CACHE:-${ROUTE_ROOT}/data/benchmark/urbannav_v2_osm_local_osrm_foot_final_15k/route_cache.jsonl}"

MAX_CASES="${MAX_CASES:-200}"
CONCURRENCY="${CONCURRENCY:-4}"
CANDIDATE_LIMIT="${CANDIDATE_LIMIT:-10}"
SELECTION_STRATEGY="${SELECTION_STRATEGY:-semantic_then_route}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-20}"
HEARTBEAT_INTERVAL_S="${HEARTBEAT_INTERVAL_S:-20}"
CHUNK_SIZE="${CHUNK_SIZE:-20}"
LLM_PROVIDER_NAME="${LLM_PROVIDER_NAME:-openai}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:18088/v1}"
LLM_TIMEOUT_S="${LLM_TIMEOUT_S:-90}"
LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-1}"
RESTART="${RESTART:-0}"

PREDICTIONS_JSONL="${RUN_ROOT}/raw_predictions.jsonl"
PREDICT_MANIFEST="${RUN_ROOT}/prediction_manifest.json"
PREDICT_LOG="${RUN_ROOT}/predict.log"
EVAL_DIR="${RUN_ROOT}/eval"
EVAL_LOG="${RUN_ROOT}/eval.log"
WORKER_SCRIPT="${RUN_ROOT}/run_worker.zsh"

_print_config() {
  print "model=${MODEL}"
  print "session=${SESSION}"
  print "run_root=${RUN_ROOT}"
  print "dataset=${DATASET}"
  print "max_cases=${MAX_CASES}"
  print "concurrency=${CONCURRENCY}"
  print "llm_base_url=${LLM_BASE_URL}"
  print "api_style=/chat/completions"
  print "LLM_API_STYLE_FALLBACK=0"
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
export LLM_RESPONSE_FORMAT_JSON=0
export LLM_API_STYLE_FALLBACK=0

{
  print "[proxy-smoke-one] model=${MODEL} started at \$(date '+%Y-%m-%d %H:%M:%S')"
  print "[proxy-smoke-one] run_root=${RUN_ROOT}"
  print "[proxy-smoke-one] concurrency=${CONCURRENCY}"
} 2>&1 | tee -a "${PREDICT_LOG}"

LLM_RESPONSE_FORMAT_JSON=0 LLM_API_STYLE_FALLBACK=0 PYTHONUNBUFFERED=1 PYTHONPATH=src "${PYTHON_BIN}" \\
  tools/evaluation/generate_urbannav_external_predictions.py \\
  --cases "${DATASET}" \\
  --max-cases "${MAX_CASES}" \\
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
  --output-jsonl "${PREDICTIONS_JSONL}" \\
  --manifest-json "${PREDICT_MANIFEST}" \\
  2>&1 | tee -a "${PREDICT_LOG}"

PYTHONPATH=src "${PYTHON_BIN}" tools/evaluation/run_urbannav_module1_benchmark.py \\
  --dataset "${DATASET}" \\
  --runner external \\
  --predictions "${PREDICTIONS_JSONL}" \\
  --output-dir "${EVAL_DIR}" \\
  --max-cases "${MAX_CASES}" \\
  --run-id "${SAFE_MODEL}-proxy-strict-200-smoke-parallel" \\
  2>&1 | tee "${EVAL_LOG}"

"${PYTHON_BIN}" -c '
import json
from pathlib import Path
pred=Path("${PREDICTIONS_JSONL}")
rows=[json.loads(line) for line in pred.open()] if pred.exists() else []
non=sum(bool(row.get("chosen_destination") and row.get("route_polyline")) for row in rows)
summary_path=Path("${EVAL_DIR}") / "summary.json"
summary=json.load(summary_path.open()).get("summary", {}) if summary_path.exists() else {}
print(json.dumps({
    "model": "${MODEL}",
    "rows": len(rows),
    "non_empty": non,
    "non_empty_rate": round(non / len(rows), 4) if rows else 0.0,
    "destination_success": summary.get("destination_success"),
    "spl": summary.get("spl"),
    "prediction_exception_rate": summary.get("prediction_exception_rate"),
}, ensure_ascii=False, sort_keys=True))
' 2>&1 | tee -a "${PREDICT_LOG}"
EOF
  chmod +x "${WORKER_SCRIPT}"
}

_status() {
  print "== tmux =="
  tmux ls 2>/dev/null || true
  print "\n== process =="
  ps -eo pid,ppid,stat,etime,cmd | grep -E "generate_urbannav_external_predictions|run_urbannav_module1_benchmark|${WORKER_SCRIPT:t}" | grep -v grep || true
  print "\n== files =="
  ls -lh "${RUN_ROOT}" 2>/dev/null || true
  print "\n== prediction rows =="
  wc -l "${PREDICTIONS_JSONL}" 2>/dev/null || true
  print "\n== latest predict log =="
  tail -40 "${PREDICT_LOG}" 2>/dev/null || true
  print "\n== latest eval log =="
  tail -20 "${EVAL_LOG}" 2>/dev/null || true
}

case "${ACTION}" in
  dry-run)
    _print_config
    _validate_inputs
    _write_worker_script
    print "worker_script=${WORKER_SCRIPT}"
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
