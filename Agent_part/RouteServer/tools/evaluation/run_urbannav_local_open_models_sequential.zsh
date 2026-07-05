#!/usr/bin/env zsh
set -euo pipefail

# Sequentially run local OpenAI-compatible UrbanNav 15K benchmark jobs.
#
# This wrapper keeps only one model active at a time. Each model still uses the
# resumable sharded runner underneath, so rerunning this script skips completed
# predictions in each shard.
#
# Usage:
#   tools/evaluation/run_urbannav_local_open_models_sequential.zsh start
#   tools/evaluation/run_urbannav_local_open_models_sequential.zsh status
#   tools/evaluation/run_urbannav_local_open_models_sequential.zsh stop

export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"

SCRIPT_PATH="${0:A}"
ROUTE_ROOT="${ROUTE_ROOT:-${SCRIPT_PATH:h:h:h}}"
ACTION="${1:-start}"

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/UrbanNav/bin/python}"
RUNNER="${ROUTE_ROOT}/tools/evaluation/run_urbannav_proxy_smoke_sharded_model.zsh"
SESSION="${SESSION:-urbannav_local_open_models_seq_15k}"
RUN_GROUP="${RUN_GROUP:-local_open_models_final15k_20260705}"

MAX_CASES="${MAX_CASES:-15000}"
SHARDS="${SHARDS:-8}"
CONCURRENCY="${CONCURRENCY:-1}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-100}"
HEARTBEAT_INTERVAL_S="${HEARTBEAT_INTERVAL_S:-30}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
LLM_TIMEOUT_S="${LLM_TIMEOUT_S:-180}"
LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-1}"
MAX_PREDICTION_EXCEPTION_RATE="${MAX_PREDICTION_EXCEPTION_RATE:-0.5}"
EXCEPTION_RATE_MIN_SAMPLES="${EXCEPTION_RATE_MIN_SAMPLES:-100}"
RESTART="${RESTART:-1}"
RESUME="${RESUME:-1}"

RUN_ROOT_BASE="${RUN_ROOT_BASE:-${ROUTE_ROOT}/data/benchmark/urbannav_module1_runs}"
LOG_DIR="${RUN_ROOT_BASE}/${RUN_GROUP}/sequential_runner"
LOG_PATH="${LOG_DIR}/sequential_runner.log"

MODEL_SPECS=(
  "internvl3.5-8b|http://127.0.0.1:18090/v1|internvl35_8b_local_strict_final15k_s8c1_v1|urbannav_internvl35_8b_15k_s8c1"
  "qwen3.5-4b|http://127.0.0.1:18092/v1|qwen35_4b_local_strict_final15k_s8c1_v1|urbannav_qwen35_4b_15k_s8c1"
  "qwen3.5-9b|http://127.0.0.1:18093/v1|qwen35_9b_local_strict_final15k_s8c1_v1|urbannav_qwen35_9b_15k_s8c1"
)

_log() {
  mkdir -p "${LOG_DIR}"
  print -- "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_PATH}"
}

_kill_prefix() {
  local prefix="$1"
  for (( shard_id=0; shard_id<SHARDS; shard_id++ )); do
    local shard_name
    shard_name=$(printf "%02d" "${shard_id}")
    tmux kill-session -t "${prefix}_${shard_name}" 2>/dev/null || true
  done
}

_stop_model_shards() {
  local spec model base_url run_name prefix
  for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r model base_url run_name prefix <<< "${spec}"
    _kill_prefix "${prefix}"
  done
}

_wait_prefix_done() {
  local prefix="$1"
  while true; do
    if ! tmux ls 2>/dev/null | grep -q "^${prefix}_"; then
      break
    fi
    _log "waiting for ${prefix}_* shards"
    sleep 60
  done
}

_run_one_model() {
  local model="$1"
  local base_url="$2"
  local run_name="$3"
  local prefix="$4"
  local run_root="${RUN_ROOT_BASE}/${run_name}"

  _log "starting model=${model} run_root=${run_root}"
  MODEL="${model}" \
  RUN_ROOT="${run_root}" \
  SESSION_PREFIX="${prefix}" \
  MAX_CASES="${MAX_CASES}" \
  SHARDS="${SHARDS}" \
  CONCURRENCY="${CONCURRENCY}" \
  PROGRESS_INTERVAL="${PROGRESS_INTERVAL}" \
  HEARTBEAT_INTERVAL_S="${HEARTBEAT_INTERVAL_S}" \
  CHUNK_SIZE="${CHUNK_SIZE}" \
  LLM_TIMEOUT_S="${LLM_TIMEOUT_S}" \
  LLM_MAX_RETRIES="${LLM_MAX_RETRIES}" \
  LLM_BASE_URL="${base_url}" \
  MAX_PREDICTION_EXCEPTION_RATE="${MAX_PREDICTION_EXCEPTION_RATE}" \
  EXCEPTION_RATE_MIN_SAMPLES="${EXCEPTION_RATE_MIN_SAMPLES}" \
  RESTART="${RESTART}" \
  RESUME="${RESUME}" \
    "${RUNNER}" start 2>&1 | tee -a "${LOG_PATH}"

  _wait_prefix_done "${prefix}"

  _log "merging model=${model}"
  MODEL="${model}" \
  RUN_ROOT="${run_root}" \
  SESSION_PREFIX="${prefix}" \
  MAX_CASES="${MAX_CASES}" \
  SHARDS="${SHARDS}" \
  CONCURRENCY="${CONCURRENCY}" \
  LLM_BASE_URL="${base_url}" \
  MAX_PREDICTION_EXCEPTION_RATE="${MAX_PREDICTION_EXCEPTION_RATE}" \
  EXCEPTION_RATE_MIN_SAMPLES="${EXCEPTION_RATE_MIN_SAMPLES}" \
  RESUME="${RESUME}" \
    "${RUNNER}" merge 2>&1 | tee -a "${LOG_PATH}" || {
      _log "merge/eval failed for model=${model}; continuing to next model"
    }
}

_worker() {
  cd "${ROUTE_ROOT}"
  _log "sequential benchmark worker started"
  _stop_model_shards
  local spec model base_url run_name prefix
  for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r model base_url run_name prefix <<< "${spec}"
    _run_one_model "${model}" "${base_url}" "${run_name}" "${prefix}"
  done
  _log "sequential benchmark worker finished"
}

_status() {
  print "== master =="
  tmux ls 2>/dev/null | grep "${SESSION}" || true
  print "\n== model shards =="
  local spec model base_url run_name prefix run_root
  for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r model base_url run_name prefix <<< "${spec}"
    run_root="${RUN_ROOT_BASE}/${run_name}"
    print "\n## ${model}"
    tmux ls 2>/dev/null | grep "^${prefix}_" || true
    for (( shard_id=0; shard_id<SHARDS; shard_id++ )); do
      local shard_name pred_jsonl pred_log
      shard_name=$(printf "%02d" "${shard_id}")
      pred_jsonl="${run_root}/shards/shard_${shard_name}/raw_predictions.jsonl"
      pred_log="${run_root}/shards/shard_${shard_name}/predict.log"
      printf "shard_%s " "${shard_name}"
      wc -l "${pred_jsonl}" 2>/dev/null || true
      tail -2 "${pred_log}" 2>/dev/null || true
    done
  done
  print "\n== sequential log =="
  tail -60 "${LOG_PATH}" 2>/dev/null || true
}

case "${ACTION}" in
  start)
    mkdir -p "${LOG_DIR}"
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
      if [[ "${RESTART}" == "1" ]]; then
        tmux kill-session -t "${SESSION}"
      else
        print -u2 "tmux session already exists: ${SESSION}"
        exit 1
      fi
    fi
    _stop_model_shards
    tmux new-session -d -s "${SESSION}" "zsh '${SCRIPT_PATH}' worker"
    print "started sequential tmux session: ${SESSION}"
    print "tail log: tail -f '${LOG_PATH}'"
    ;;
  worker)
    _worker
    ;;
  status)
    _status
    ;;
  stop)
    tmux kill-session -t "${SESSION}" 2>/dev/null || true
    _stop_model_shards
    print "stopped sequential runner and model shards if they existed"
    ;;
  *)
    print -u2 "unknown action: ${ACTION}"
    print -u2 "usage: ${SCRIPT_PATH} {start|status|stop}"
    exit 2
    ;;
esac
