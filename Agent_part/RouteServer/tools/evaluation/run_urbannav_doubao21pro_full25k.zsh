#!/usr/bin/env zsh
set -euo pipefail

# Run the complete UrbanNav v3 23-city 25K benchmark through ARK Responses API.
# The output provenance deliberately records the actual Doubao model even when
# this run is used as the GPT-5.5 comparison slot in downstream tables.

SCRIPT_PATH="${0:A}"
ROUTE_ROOT="${ROUTE_ROOT:-${SCRIPT_PATH:h:h:h}}"
REPO_ROOT="${REPO_ROOT:-$(cd "${ROUTE_ROOT}/../.." && pwd)}"
BASE_RUNNER="${ROUTE_ROOT}/tools/evaluation/run_urbannav_gpt55_full25k.zsh"

export MODEL="${MODEL:-doubao-seed-2-1-pro-260628}"
export RUN_NAME="${RUN_NAME:-gpt55_slot__actual_doubao_seed_2_1_pro_260628__full25k_v3_23cities}"
export SESSION="${SESSION:-urbannav_doubao21pro_full25k}"
export KEY_FILE="${KEY_FILE:-${REPO_ROOT}/key.txt}"
export LLM_PROVIDER_NAME="${LLM_PROVIDER_NAME:-ark}"
export LLM_BASE_URL="${LLM_BASE_URL:-https://ark.cn-beijing.volces.com/api/v3}"
export LLM_API_STYLE="${LLM_API_STYLE:-responses}"
export LLM_REASONING_EFFORT="${LLM_REASONING_EFFORT:-low}"
export LLM_TIMEOUT_S="${LLM_TIMEOUT_S:-120}"
export LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-3}"
export CONCURRENCY="${CONCURRENCY:-66}"
export CHUNK_SIZE="${CHUNK_SIZE:-200}"
export RESET_EVERY_CASES="${RESET_EVERY_CASES:-1000}"
export RESET_SLEEP_S="${RESET_SLEEP_S:-3}"
export PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-100}"
export HEARTBEAT_INTERVAL_S="${HEARTBEAT_INTERVAL_S:-20}"
export MAX_PREDICTION_EXCEPTION_RATE="${MAX_PREDICTION_EXCEPTION_RATE:-0.05}"
export EXCEPTION_RATE_MIN_SAMPLES="${EXCEPTION_RATE_MIN_SAMPLES:-100}"

exec "${BASE_RUNNER}" "${1:-start}"
