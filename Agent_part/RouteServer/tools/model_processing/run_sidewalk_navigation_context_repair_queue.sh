#!/usr/bin/env bash
set -Eeuo pipefail

# SideWalk 的原始 run 保持不可变；该队列只重标 producer contract 修复后的失败子集。

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/lpai/volumes/mind-eb-bd-ga/suh/UrbanNavigation}"
ROUTESERVER_ROOT="${ROUTESERVER_ROOT:-${WORKSPACE_ROOT}/Urban-Nav-SR/Agent_part/RouteServer}"
PYTHON_BIN="${PYTHON_BIN:-/lpai/volumes/mind-eb-bd-ga/suh/miniconda3/envs/qwen35_vllm_cuda128/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/lpai/volumes/mind-eb-bd-ga/suh/dataset/urbannav-sidewalk-agent-data-v1}"
INPUT_JSONL="${INPUT_JSONL:-${WORKSPACE_ROOT}/annotation_inputs/reasoning_v4_6/sidewalk_main_20way.jsonl}"
REGISTRY_JSONL="${REGISTRY_JSONL:-${WORKSPACE_ROOT}/annotation_inputs/reasoning_v4_6/sidewalk_bev_contract_registry_v1.jsonl}"
BASE_RUN_ROOT="${BASE_RUN_ROOT:-${WORKSPACE_ROOT}/annotation_runs/reasoning_v4_6_qwen122b_sidewalk_main_20way_routed_visual_full_v1}"
REPAIR_RUN_ROOT="${REPAIR_RUN_ROOT:-${WORKSPACE_ROOT}/annotation_runs/reasoning_v4_6_qwen122b_sidewalk_main_navigation_contract_repair_v1}"
EGOWALK_RUN_ROOT="${EGOWALK_RUN_ROOT:-${WORKSPACE_ROOT}/annotation_runs/reasoning_v4_6_qwen122b_egowalk_12way_routed_visual_full_v1}"
RUNNER="${RUNNER:-${ROUTESERVER_ROOT}/tools/model_processing/run_reasoning_v4_qwen122b_dual_annotation.sh}"
QUEUE_SESSION="${QUEUE_SESSION:-vla-annotation-fullcot-sidewalk-repair}"
EXPECTED_ROWS="${EXPECTED_ROWS:-10359}"
REGISTRY_SHA256="${REGISTRY_SHA256:-6877fffd374f12a5ef281703d2c71f9faf44da39660f86940fb20b6a64200607}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-120}"
CONCURRENCY_PER_SERVICE="${CONCURRENCY_PER_SERVICE:-8}"
BEHAVIOR_BASE_URL="${BEHAVIOR_BASE_URL:-http://127.0.0.1:18461/v1}"
BEHAVIOR_MODEL="${BEHAVIOR_MODEL:-qwen3.5-122b-a10b-behavior}"
INSTRUCTION_BASE_URL="${INSTRUCTION_BASE_URL:-http://127.0.0.1:18462/v1}"
INSTRUCTION_MODEL="${INSTRUCTION_MODEL:-qwen3.5-122b-a10b-instruction}"

SCRIPT_PATH="$(readlink -f "$0")"
REPAIRED_INPUT="${REPAIR_RUN_ROOT}/inputs/sidewalk_navigation_contract_repair.jsonl"
QUEUE_LOG="${REPAIR_RUN_ROOT}/logs/repair_queue.log"

usage() {
  cat <<'EOF'
Usage:
  run_sidewalk_navigation_context_repair_queue.sh launch
  run_sidewalk_navigation_context_repair_queue.sh run
  run_sidewalk_navigation_context_repair_queue.sh status

The queue waits for active VQA and EgoWalk FullCoT workers, verifies all 10,359
archived BEV digests against the producer registry, then starts a new immutable
SideWalk repair run. It does not overwrite the original 45,397-row run.
EOF
}

die() {
  echo "[repair-error] $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "required file is missing: $1"
}

tmux_session_exists_exact() {
  tmux list-sessions -F '#S' 2>/dev/null | grep -Fqx -- "$1"
}

assert_registry_identity() {
  local actual_rows
  local actual_sha
  require_file "${REGISTRY_JSONL}"
  actual_rows="$(wc -l <"${REGISTRY_JSONL}" | tr -d ' ')"
  actual_sha="$(sha256sum "${REGISTRY_JSONL}" | awk '{print $1}')"
  [[ "${actual_rows}" == "${EXPECTED_ROWS}" ]] \
    || die "registry row mismatch: ${actual_rows} != ${EXPECTED_ROWS}"
  [[ "${actual_sha}" == "${REGISTRY_SHA256}" ]] \
    || die "registry SHA mismatch: ${actual_sha} != ${REGISTRY_SHA256}"
  echo "[registry-ok] rows=${actual_rows} sha256=${actual_sha}"
}

annotation_workers_running() {
  pgrep -f "${ROUTESERVER_ROOT}/tools/model_processing/label_vqa_v2.py" \
    >/dev/null 2>&1 \
    || pgrep -f "${EGOWALK_RUN_ROOT}/sources/shards/service_" \
      >/dev/null 2>&1
}

wait_for_annotation_services() {
  while annotation_workers_running; do
    echo "[repair-wait] $(date --iso-8601=seconds) active VQA/EgoWalk worker"
    sleep "${POLL_INTERVAL_S}"
  done
  echo "[repair-wait-complete] $(date --iso-8601=seconds)"
}

prepare_repair_subset() {
  mkdir -p \
    "${REPAIR_RUN_ROOT}/inputs" \
    "${REPAIR_RUN_ROOT}/reports" \
    "${REPAIR_RUN_ROOT}/logs"
  assert_registry_identity
  require_file "${INPUT_JSONL}"
  PYTHONPATH="${ROUTESERVER_ROOT}/src:${ROUTESERVER_ROOT}" \
    "${PYTHON_BIN}" \
    "${ROUTESERVER_ROOT}/tools/model_processing/recover_sidewalk_bev_contract.py" \
    build-repair-subset \
    --input-jsonl "${INPUT_JSONL}" \
    --registry-jsonl "${REGISTRY_JSONL}" \
    --output-jsonl "${REPAIRED_INPUT}" \
    --report-json "${REPAIR_RUN_ROOT}/reports/source_contract_recovery.json" \
    --media-root "${DATASET_ROOT}" \
    --expected-rows "${EXPECTED_ROWS}"
}

run_repair() {
  wait_for_annotation_services
  prepare_repair_subset
  env \
    WORKSPACE_ROOT="${WORKSPACE_ROOT}" \
    ROUTESERVER_ROOT="${ROUTESERVER_ROOT}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    DATASET_ROOT="${DATASET_ROOT}" \
    DATASET_SOURCE=sidewalk \
    MEDIA_ROOT="${DATASET_ROOT}" \
    INPUT_JSONL="${REPAIRED_INPUT}" \
    RUN_ROOT="${REPAIR_RUN_ROOT}" \
    EXPECTED_ROWS="${EXPECTED_ROWS}" \
    DIRECTION_PROFILE_ID=fpv_degree10_terminal_centroid_v2 \
    BEHAVIOR_BASE_URL="${BEHAVIOR_BASE_URL}" \
    BEHAVIOR_MODEL="${BEHAVIOR_MODEL}" \
    INSTRUCTION_BASE_URL="${INSTRUCTION_BASE_URL}" \
    INSTRUCTION_MODEL="${INSTRUCTION_MODEL}" \
    CONCURRENCY_PER_SERVICE="${CONCURRENCY_PER_SERVICE}" \
    MAX_RETRIES=2 \
    IMAGE_DETAIL=low \
    NAVIGATION_CONTEXT_IMAGE_DETAIL=high \
    SECTION_IMAGE_POLICY_ID=reasoning_v4_section_images_routed_visual_v1 \
    RETRY_FAILED_ON_RESUME=1 \
    POSTPROCESS_FULL=0 \
    "${RUNNER}" run-full
  PYTHONPATH="${ROUTESERVER_ROOT}/src:${ROUTESERVER_ROOT}" \
    "${PYTHON_BIN}" \
    "${ROUTESERVER_ROOT}/tools/model_processing/audit_sidewalk_navigation_context_repair.py" \
    --source-jsonl \
    "${REPAIR_RUN_ROOT}/sources/reasoning_v4_5_sources.jsonl" \
    --annotation-jsonl \
    "${REPAIR_RUN_ROOT}/annotations/reasoning_v4_6_annotations.jsonl" \
    --expected-rows "${EXPECTED_ROWS}" \
    --report-json \
    "${REPAIR_RUN_ROOT}/reports/navigation_context_repair_audit.json" \
    --strict
  echo "[repair-complete] $(date --iso-8601=seconds) root=${REPAIR_RUN_ROOT}"
}

launch_repair() {
  tmux_session_exists_exact "${QUEUE_SESSION}" \
    && die "tmux session already exists: ${QUEUE_SESSION}"
  mkdir -p "${REPAIR_RUN_ROOT}/logs"
  local command
  local -a launch=(
    env
    "WORKSPACE_ROOT=${WORKSPACE_ROOT}"
    "ROUTESERVER_ROOT=${ROUTESERVER_ROOT}"
    "PYTHON_BIN=${PYTHON_BIN}"
    "DATASET_ROOT=${DATASET_ROOT}"
    "INPUT_JSONL=${INPUT_JSONL}"
    "REGISTRY_JSONL=${REGISTRY_JSONL}"
    "BASE_RUN_ROOT=${BASE_RUN_ROOT}"
    "REPAIR_RUN_ROOT=${REPAIR_RUN_ROOT}"
    "EGOWALK_RUN_ROOT=${EGOWALK_RUN_ROOT}"
    "EXPECTED_ROWS=${EXPECTED_ROWS}"
    "REGISTRY_SHA256=${REGISTRY_SHA256}"
    "POLL_INTERVAL_S=${POLL_INTERVAL_S}"
    "CONCURRENCY_PER_SERVICE=${CONCURRENCY_PER_SERVICE}"
    "BEHAVIOR_BASE_URL=${BEHAVIOR_BASE_URL}"
    "BEHAVIOR_MODEL=${BEHAVIOR_MODEL}"
    "INSTRUCTION_BASE_URL=${INSTRUCTION_BASE_URL}"
    "INSTRUCTION_MODEL=${INSTRUCTION_MODEL}"
    bash "${SCRIPT_PATH}" run
  )
  printf -v command '%q ' "${launch[@]}"
  tmux new-session -d -s "${QUEUE_SESSION}" \
    "exec ${command} >>${QUEUE_LOG} 2>&1"
  echo "[repair-launched] session=${QUEUE_SESSION} log=${QUEUE_LOG}"
}

status_repair() {
  if tmux_session_exists_exact "${QUEUE_SESSION}"; then
    echo "[repair-status] session=${QUEUE_SESSION} state=running"
  else
    echo "[repair-status] session=${QUEUE_SESSION} state=not-running"
  fi
  for path in \
    "${REPAIRED_INPUT}" \
    "${REPAIR_RUN_ROOT}/annotations/service_0.jsonl" \
    "${REPAIR_RUN_ROOT}/annotations/service_1.jsonl"; do
    if [[ -f "${path}" ]]; then
      echo "[repair-rows] $(wc -l <"${path}" | tr -d ' ') ${path}"
    fi
  done
  [[ -f "${QUEUE_LOG}" ]] && tail -n 20 "${QUEUE_LOG}"
}

case "${1:-help}" in
  launch)
    launch_repair
    ;;
  run)
    run_repair
    ;;
  status)
    status_repair
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
