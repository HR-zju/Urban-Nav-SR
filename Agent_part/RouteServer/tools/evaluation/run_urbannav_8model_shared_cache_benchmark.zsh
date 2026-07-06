#!/usr/bin/env zsh
set -euo pipefail

# Start/status/stop the 8-model UrbanNav 15K benchmark using one shared replay
# cache service.  This avoids loading the large route_cache.jsonl in every
# shard process.
#
# Usage:
#   tools/evaluation/run_urbannav_8model_shared_cache_benchmark.zsh start
#   tools/evaluation/run_urbannav_8model_shared_cache_benchmark.zsh status
#   tools/evaluation/run_urbannav_8model_shared_cache_benchmark.zsh stop

SCRIPT_PATH="${0:A}"
ROUTE_ROOT="${ROUTE_ROOT:-${SCRIPT_PATH:h:h:h}}"
ACTION="${1:-status}"
REPLAY_CACHE_URL="${REPLAY_CACHE_URL:-http://127.0.0.1:18180}"
MAX_CASES="${MAX_CASES:-15000}"
RUN_SUFFIX="${RUN_SUFFIX:-sharedcache_s1_v1}"
BENCH_LLM_TIMEOUT_S="${BENCH_LLM_TIMEOUT_S:-240}"
BENCH_LLM_MAX_RETRIES="${BENCH_LLM_MAX_RETRIES:-4}"

typeset -a MODEL_ROWS
MODEL_ROWS=(
  "18100|qwen3.5-4b|qwen35_4b_local_strict_final15k_p18100_${RUN_SUFFIX}|12"
  "18101|qwen3-vl-30b-a3b|qwen3_vl_30b_a3b_local_strict_final15k_p18101_${RUN_SUFFIX}|4"
  "18102|internvl3.5-14b|internvl35_14b_local_strict_final15k_p18102_${RUN_SUFFIX}|4"
  "18103|qwen3.5-27b|qwen35_27b_local_strict_final15k_p18103_${RUN_SUFFIX}|6"
  "18104|internvl3.5-38b|internvl35_38b_local_strict_final15k_p18104_${RUN_SUFFIX}|2"
  "18105|qwen3.5-35b-a3b|qwen35_35b_a3b_local_strict_final15k_p18105_${RUN_SUFFIX}|2"
  "18106|qwen3.6-27b|qwen36_27b_local_strict_final15k_p18106_${RUN_SUFFIX}|6"
  "18107|qwen3.6-35b-a3b|qwen36_35b_a3b_local_strict_final15k_p18107_${RUN_SUFFIX}|2"
)

_check_endpoint() {
  local port="$1"
  curl --noproxy 127.0.0.1,localhost -sS -m 8 "http://127.0.0.1:${port}/v1/models" >/dev/null
}

_check_cache() {
  curl --noproxy 127.0.0.1,localhost -sS -m 8 "${REPLAY_CACHE_URL}/healthz" >/dev/null
}

_start_one() {
  local port="$1"
  local model="$2"
  local run_name="$3"
  local concurrency="$4"
  print "== start ${model} port=${port} concurrency=${concurrency} run=${run_name} =="
  MODEL="${model}" \
  RUN_ROOT="${ROUTE_ROOT}/data/benchmark/urbannav_module1_runs/${run_name}" \
  SESSION_PREFIX="urbannav_${run_name}_shard" \
  MAX_CASES="${MAX_CASES}" \
  SHARDS=1 \
  CONCURRENCY="${concurrency}" \
  PROGRESS_INTERVAL=100 \
  HEARTBEAT_INTERVAL_S=30 \
  CHUNK_SIZE=20 \
  LLM_TIMEOUT_S="${BENCH_LLM_TIMEOUT_S}" \
  LLM_MAX_RETRIES="${BENCH_LLM_MAX_RETRIES}" \
  LLM_BASE_URL="http://127.0.0.1:${port}/v1" \
  REPLAY_CACHE_URL="${REPLAY_CACHE_URL}" \
  MAX_PREDICTION_EXCEPTION_RATE=0.2 \
  EXCEPTION_RATE_MIN_SAMPLES=100 \
  RESUME=1 \
  RESTART=0 \
  "${ROUTE_ROOT}/tools/evaluation/run_urbannav_proxy_smoke_sharded_model.zsh" start
}

case "${ACTION}" in
  start)
    cd "${ROUTE_ROOT}"
    print "checking replay cache: ${REPLAY_CACHE_URL}"
    _check_cache
    for row in "${MODEL_ROWS[@]}"; do
      IFS="|" read -r port model run_name concurrency <<< "${row}"
      print "checking ${port} ${model}"
      _check_endpoint "${port}"
    done
    for row in "${MODEL_ROWS[@]}"; do
      IFS="|" read -r port model run_name concurrency <<< "${row}"
      _start_one "${port}" "${model}" "${run_name}" "${concurrency}"
    done
    ;;
  status)
    cd "${ROUTE_ROOT}"
    print "replay_cache=${REPLAY_CACHE_URL}"
    curl --noproxy 127.0.0.1,localhost -sS -m 8 "${REPLAY_CACHE_URL}/healthz" || true
    print
    for row in "${MODEL_ROWS[@]}"; do
      IFS="|" read -r port model run_name concurrency <<< "${row}"
      printf "%s %s " "${port}" "${model}"
      curl --noproxy 127.0.0.1,localhost -sS -m 5 "http://127.0.0.1:${port}/v1/models" >/dev/null && print "OK" || print "DOWN"
      local pred="${ROUTE_ROOT}/data/benchmark/urbannav_module1_runs/${run_name}/shards/shard_00/raw_predictions.jsonl"
      if [[ -f "${pred}" ]]; then
        print "  rows=$(wc -l < "${pred}") pred=${pred}"
      else
        print "  rows=0 pred=${pred}"
      fi
    done
    tmux ls 2>/dev/null | grep 'urbannav_.*sharedcache' || true
    ;;
  stop)
    for row in "${MODEL_ROWS[@]}"; do
      IFS="|" read -r port model run_name concurrency <<< "${row}"
      local session="urbannav_${run_name}_shard_00"
      if tmux has-session -t "${session}" 2>/dev/null; then
        tmux kill-session -t "${session}"
        print "stopped ${session}"
      fi
    done
    ;;
  *)
    print -u2 "unknown action: ${ACTION}"
    exit 2
    ;;
esac
