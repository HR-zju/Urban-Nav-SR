#!/usr/bin/env bash
set -Eeuo pipefail

# Compatibility entrypoint for one OpenAI-compatible Qwen endpoint.
# The canonical V4.5 orchestration and release gates live in the dual runner;
# both worker roles are pointed at the same endpoint here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUAL_RUNNER="${SCRIPT_DIR}/run_reasoning_v4_qwen122b_dual_annotation.sh"
MODE="${1:-help}"

BASE_URL="${LLM_API_BASE_URL:-http://127.0.0.1:18461/v1}"
MODEL="${LLM_MODEL:-qwen3.5-122b-a10b-behavior}"
CONCURRENCY="${CONCURRENCY_PER_SERVICE:-${SMOKE_CONCURRENCY:-2}}"

usage() {
  cat <<'EOF'
Usage:
  run_reasoning_v4_qwen122b_annotation.sh preflight
  run_reasoning_v4_qwen122b_annotation.sh smoke
  run_reasoning_v4_qwen122b_annotation.sh status-smoke
  run_reasoning_v4_qwen122b_annotation.sh tail-smoke
  SMOKE_APPROVED=1 SMOKE_RESOLUTION_JSONL=/path/to/resolutions.jsonl \
    run_reasoning_v4_qwen122b_annotation.sh launch-full
  run_reasoning_v4_qwen122b_annotation.sh status-full
  run_reasoning_v4_qwen122b_annotation.sh tail-full

This compatibility wrapper uses one service for both V4.5 worker roles.
Use run_reasoning_v4_qwen122b_dual_annotation.sh directly when two endpoints
are available.
EOF
}

case "${MODE}" in
  preflight|smoke|launch-smoke|status-smoke|tail-smoke|launch-full|status-full|tail-full)
    exec env \
      BEHAVIOR_BASE_URL="${BASE_URL}" \
      BEHAVIOR_MODEL="${MODEL}" \
      INSTRUCTION_BASE_URL="${BASE_URL}" \
      INSTRUCTION_MODEL="${MODEL}" \
      CONCURRENCY_PER_SERVICE="${CONCURRENCY}" \
      "${DUAL_RUNNER}" "${MODE}"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    echo "[error] unsupported mode: ${MODE}" >&2
    exit 2
    ;;
esac
