#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROUTE_SERVER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROUTE_SERVER_ROOT}"

PY="${PY:-/home/ubuntu/anaconda3/envs/UrbanNav/bin/python}"
DATASET="${DATASET:-${ROUTE_SERVER_ROOT}/data/benchmark/urbannav_v2_osm_local_osrm_foot_final_15k/task_gt.jsonl}"
POI_CACHE="${POI_CACHE:-${ROUTE_SERVER_ROOT}/data/benchmark/urbannav_v2_osm_local_osrm_foot_final_15k/poi_cache.jsonl}"
ROUTE_CACHE="${ROUTE_CACHE:-${ROUTE_SERVER_ROOT}/data/benchmark/urbannav_v2_osm_local_osrm_foot_final_15k/route_cache.jsonl}"

MODEL="${MODEL:-doubao-seed-2-0-lite-260428}"
RUN_ID="${RUN_ID:-tool_agent_${MODEL//[^A-Za-z0-9_]/_}_smoke200_v1}"
RUN_ROOT="${RUN_ROOT:-${ROUTE_SERVER_ROOT}/data/benchmark/urbannav_module1_runs/${RUN_ID}}"
MAX_CASES="${MAX_CASES:-200}"
CONCURRENCY="${CONCURRENCY:-4}"
CANDIDATE_LIMIT="${CANDIDATE_LIMIT:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
MAX_TURNS="${MAX_TURNS:-0}"
MAX_TOKENS="${MAX_TOKENS:-768}"
TEMPERATURE="${TEMPERATURE:-0}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-20}"
HEARTBEAT_INTERVAL_S="${HEARTBEAT_INTERVAL_S:-30}"
LLM_PROVIDER="${LLM_PROVIDER:-ark}"
LLM_BASE_URL="${LLM_API_BASE_URL:-${LLM_BASE_URL:-https://ark.cn-beijing.volces.com/api/v3}}"
LLM_TIMEOUT_S="${LLM_TIMEOUT_S:-180}"
LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-1}"
API_KEYS_FILE="${API_KEYS_FILE:-}"

if [[ -z "${API_KEYS_FILE}" && "${LLM_BASE_URL}" != http://127.0.0.1:* && "${LLM_BASE_URL}" != http://localhost:* ]]; then
  if [[ -z "${ARK_API_KEY:-}" && -z "${LLM_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: no API key found. Export ARK_API_KEY/LLM_API_KEY/OPENAI_API_KEY or set API_KEYS_FILE." >&2
    exit 2
  fi
fi

mkdir -p "${RUN_ROOT}"

echo "== UrbanNav tool-agent smoke =="
echo "route_server_root=${ROUTE_SERVER_ROOT}"
echo "dataset=${DATASET}"
echo "run_root=${RUN_ROOT}"
echo "model=${MODEL}"
echo "max_cases=${MAX_CASES}"
echo "concurrency=${CONCURRENCY}"

GEN_ARGS=(
  --cases "${DATASET}"
  --poi-cache "${POI_CACHE}"
  --route-cache "${ROUTE_CACHE}"
  --output-jsonl "${RUN_ROOT}/raw_predictions.jsonl"
  --manifest-json "${RUN_ROOT}/prediction_manifest.json"
  --max-cases "${MAX_CASES}"
  --concurrency "${CONCURRENCY}"
  --candidate-limit "${CANDIDATE_LIMIT}"
  --max-turns "${MAX_TURNS}"
  --max-tokens "${MAX_TOKENS}"
  --temperature "${TEMPERATURE}"
  --llm-provider "${LLM_PROVIDER}"
  --llm-model "${MODEL}"
  --llm-base-url "${LLM_BASE_URL}"
  --llm-timeout-s "${LLM_TIMEOUT_S}"
  --llm-max-retries "${LLM_MAX_RETRIES}"
  --chunk-size "${CHUNK_SIZE}"
  --progress-interval "${PROGRESS_INTERVAL}"
  --heartbeat-interval-s "${HEARTBEAT_INTERVAL_S}"
)

if [[ -n "${API_KEYS_FILE}" ]]; then
  GEN_ARGS+=(--api-keys-file "${API_KEYS_FILE}")
fi

PYTHONPATH=src "${PY}" tools/evaluation/generate_urbannav_tool_agent_predictions.py "${GEN_ARGS[@]}" \
  2>&1 | tee "${RUN_ROOT}/predict.log"

PYTHONPATH=src "${PY}" tools/evaluation/run_urbannav_module1_benchmark.py \
  --dataset "${DATASET}" \
  --runner external \
  --predictions "${RUN_ROOT}/raw_predictions.jsonl" \
  --output-dir "${RUN_ROOT}/eval" \
  --run-id "${RUN_ID}" \
  --max-cases "${MAX_CASES}" \
  --allow-invalid-predictions \
  2>&1 | tee "${RUN_ROOT}/eval.log"

PYTHONPATH=src "${PY}" tools/evaluation/analyze_urbannav_tool_agent_run.py \
  --predictions "${RUN_ROOT}/raw_predictions.jsonl" \
  --dataset "${DATASET}" \
  --eval-rows "${RUN_ROOT}/eval/eval_rows.jsonl" \
  --output-json "${RUN_ROOT}/tool_metrics_summary.json" \
  --output-md "${RUN_ROOT}/tool_metrics_summary.md" \
  --review-html "${RUN_ROOT}/trace_review/index.html" \
  --max-review-cases 160 \
  2>&1 | tee "${RUN_ROOT}/tool_metrics.log"

echo
echo "Done."
echo "Prediction: ${RUN_ROOT}/raw_predictions.jsonl"
echo "Eval summary: ${RUN_ROOT}/eval/summary.json"
echo "Tool metrics: ${RUN_ROOT}/tool_metrics_summary.json"
echo "Trace review: ${RUN_ROOT}/trace_review/index.html"
