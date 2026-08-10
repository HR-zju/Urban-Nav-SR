#!/usr/bin/env bash
# 双 122B 服务的全量语言目标标注。future V2 sidecar 仅用于 teacher 选目标，绝不进模型 prompt。
set -Eeuo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/lpai/volumes/mind-eb-bd-ga/suh/UrbanNavigation}"
ROUTESERVER_ROOT="${ROUTESERVER_ROOT:-${WORKSPACE_ROOT}/Urban-Nav-SR/Agent_part/RouteServer}"
PYTHON_BIN="${PYTHON_BIN:-/lpai/volumes/mind-eb-bd-ga/suh/miniconda3/envs/qwen35_vllm_cuda128/bin/python}"
RUN_ROOT="${RUN_ROOT:-${WORKSPACE_ROOT}/runs/language_waypoint_full_qwen122b_20260809}"

SIDEWALK_MAIN_SOURCE_JSONL="${SIDEWALK_MAIN_SOURCE_JSONL:-${WORKSPACE_ROOT}/annotation_runs/reasoning_v4_6_qwen122b_sidewalk_main_20way_routed_visual_full_v1/sources/reasoning_v4_5_sources.jsonl}"
SIDEWALK_EXTRA_SOURCE_JSONL="${SIDEWALK_EXTRA_SOURCE_JSONL:-${WORKSPACE_ROOT}/annotation_runs/reasoning_v4_6_qwen122b_sidewalk_extra_20way_routed_visual_full_v2/sources/reasoning_v4_5_sources.jsonl}"
FOXGLOVE_SOURCE_JSONL="${FOXGLOVE_SOURCE_JSONL:-${WORKSPACE_ROOT}/annotation_runs/reasoning_v4_6_qwen122b_foxglove_20way_routed_visual_full_v1/sources/reasoning_v4_5_sources.jsonl}"
EGOWALK_SOURCE_JSONL="${EGOWALK_SOURCE_JSONL:-${WORKSPACE_ROOT}/annotation_runs/reasoning_v4_6_qwen122b_egowalk_12way_routed_visual_full_v1/sources/reasoning_v4_5_sources.jsonl}"

# 冒号分隔的 train/validation/test V2 sidecar 路径；旧 2s/16 ledger 会被 typed parser 拒绝。
TRAJECTORY_SIDECAR_JSONLS="${TRAJECTORY_SIDECAR_JSONLS:?set TRAJECTORY_SIDECAR_JSONLS to colon-separated 4s/32 V2 sidecar JSONLs}"
LANE0_BASE_URL="${LANE0_BASE_URL:-http://127.0.0.1:18461/v1}"
LANE0_MODEL="${LANE0_MODEL:-qwen3.5-122b-a10b-behavior}"
LANE1_BASE_URL="${LANE1_BASE_URL:-http://127.0.0.1:18462/v1}"
LANE1_MODEL="${LANE1_MODEL:-qwen3.5-122b-a10b-instruction}"
CONCURRENCY_PER_SERVICE="${CONCURRENCY_PER_SERVICE:-8}"
MAX_RETRIES="${MAX_RETRIES:-1}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-100}"
CANDIDATE_TIME_S="${CANDIDATE_TIME_S:-4.0}"

for required in "${PYTHON_BIN}" "${SIDEWALK_MAIN_SOURCE_JSONL}" "${SIDEWALK_EXTRA_SOURCE_JSONL}" "${FOXGLOVE_SOURCE_JSONL}" "${EGOWALK_SOURCE_JSONL}"; do
  [[ -e "${required}" ]] || { echo "[error] missing required path: ${required}" >&2; exit 2; }
done
[[ "${CONCURRENCY_PER_SERVICE}" =~ ^[1-9][0-9]*$ ]] || { echo "[error] CONCURRENCY_PER_SERVICE must be positive" >&2; exit 2; }

mkdir -p "${RUN_ROOT}"/{inputs,logs,datasets,receipts,visualization}
export PYTHONPATH="${ROUTESERVER_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export LLM_PROVIDER=openai LLM_API_KEY=local-vllm LLM_API_STYLE=chat_completions
export LLM_TEMPERATURE=0 LLM_TIMEOUT_S=900 LLM_MAX_RETRIES=0
export LLM_CHAT_TEMPLATE_ENABLE_THINKING=false LLM_VISION_RESPONSE_FORMAT_JSON=true
export LLM_VISION_MAX_TOKENS="${LLM_VISION_MAX_TOKENS:-512}"

curl --noproxy 127.0.0.1,localhost -fsS --max-time 30 "${LANE0_BASE_URL}/models" >"${RUN_ROOT}/receipts/lane0_models.json"
curl --noproxy 127.0.0.1,localhost -fsS --max-time 30 "${LANE1_BASE_URL}/models" >"${RUN_ROOT}/receipts/lane1_models.json"

sidecar_args=()
IFS=':' read -r -a sidecar_paths <<< "${TRAJECTORY_SIDECAR_JSONLS}"
for sidecar in "${sidecar_paths[@]}"; do
  [[ -f "${sidecar}" ]] || { echo "[error] missing V2 sidecar: ${sidecar}" >&2; exit 2; }
  sidecar_args+=(--trajectory-sidecar-jsonl "${sidecar}")
done

"${PYTHON_BIN}" "${ROUTESERVER_ROOT}/tools/model_processing/materialize_language_waypoint_sidecars.py" \
  "${sidecar_args[@]}" \
  --output-root "${RUN_ROOT}/inputs/sidecars_by_dataset" \
  --report-json "${RUN_ROOT}/receipts/sidecar_materialization.json" \
  >"${RUN_ROOT}/logs/materialize_sidecars.log" 2>&1

cat >"${RUN_ROOT}/visualization/visualization_manifest.json" <<EOF
{"schema_version":"urbannav_visualization_manifest_v1","run_kind":"language_waypoint_annotation","source_media_unchanged":true,"planned_artifacts":["dataset merged reports","per-dataset language-goal review HTML generated after completion"]}
EOF

run_lane() {
  local dataset="$1"
  local lane="$2"
  local base_url="$3"
  local model="$4"
  local root="${RUN_ROOT}/datasets/${dataset}"
  env \
    LLM_API_BASE_URL="${base_url}" \
    LLM_MODEL="${model}" \
    "${PYTHON_BIN}" "${ROUTESERVER_ROOT}/tools/model_processing/label_language_waypoint_goals.py" \
      --source-jsonl "${root}/shards/lane_$(printf '%02d' "${lane}")/sources.jsonl" \
      --candidates-jsonl "${root}/shards/lane_$(printf '%02d' "${lane}")/candidates.jsonl" \
      --annotation-jsonl "${root}/lanes/lane_$(printf '%02d' "${lane}")/annotations.jsonl" \
      --decisions-jsonl "${root}/lanes/lane_$(printf '%02d' "${lane}")/decisions.jsonl" \
      --report-json "${root}/lanes/lane_$(printf '%02d' "${lane}")/report.json" \
      --concurrency "${CONCURRENCY_PER_SERVICE}" \
      --max-retries "${MAX_RETRIES}" \
      --progress-interval "${PROGRESS_INTERVAL}" \
      >"${root}/lanes/lane_$(printf '%02d' "${lane}")/run.log" 2>&1
}

run_dataset() {
  local dataset="$1"
  local source_jsonl="$2"
  local sidecar_dataset="$3"
  local root="${RUN_ROOT}/datasets/${dataset}"
  local sidecar_jsonl="${RUN_ROOT}/inputs/sidecars_by_dataset/${sidecar_dataset}_trajectory_sidecars_v2.jsonl"
  [[ -f "${sidecar_jsonl}" ]] || { echo "[error] no accepted V2 sidecars for dataset=${dataset}" >&2; exit 2; }
  mkdir -p "${root}"/{inputs,shards,lanes/lane_00,lanes/lane_01,merged,reports,review}
  echo "[language-waypoint] dataset=${dataset} source=${source_jsonl}" | tee -a "${RUN_ROOT}/logs/run.log"
  "${PYTHON_BIN}" "${ROUTESERVER_ROOT}/tools/evaluation/build_language_waypoint_benchmark.py" candidates \
    --source-jsonl "${source_jsonl}" \
    --trajectory-sidecar-jsonl "${sidecar_jsonl}" \
    --output-jsonl "${root}/inputs/candidates.jsonl" \
    --report-json "${root}/reports/candidates.json" \
    --candidate-time-s "${CANDIDATE_TIME_S}" \
    >"${root}/logs_candidates.log" 2>&1
  "${PYTHON_BIN}" "${ROUTESERVER_ROOT}/tools/model_processing/partition_language_waypoint_annotation_inputs.py" \
    --source-jsonl "${source_jsonl}" \
    --candidates-jsonl "${root}/inputs/candidates.jsonl" \
    --output-root "${root}/shards" \
    --report-json "${root}/reports/partition.json" \
    --lane-count 2 \
    >"${root}/logs_partition.log" 2>&1
  run_lane "${dataset}" 0 "${LANE0_BASE_URL}" "${LANE0_MODEL}" &
  local pid0=$!
  run_lane "${dataset}" 1 "${LANE1_BASE_URL}" "${LANE1_MODEL}" &
  local pid1=$!
  wait "${pid0}"
  wait "${pid1}"
  "${PYTHON_BIN}" "${ROUTESERVER_ROOT}/tools/model_processing/merge_language_waypoint_annotation_shards.py" \
    --candidates-jsonl "${root}/inputs/candidates.jsonl" \
    --annotation-jsonl "${root}/lanes/lane_00/annotations.jsonl" \
    --annotation-jsonl "${root}/lanes/lane_01/annotations.jsonl" \
    --decisions-jsonl "${root}/lanes/lane_00/decisions.jsonl" \
    --decisions-jsonl "${root}/lanes/lane_01/decisions.jsonl" \
    --output-annotations-jsonl "${root}/merged/language_goal_annotations.jsonl" \
    --output-decisions-jsonl "${root}/merged/language_goal_decisions.jsonl" \
    --report-json "${root}/reports/merge.json" \
    >"${root}/logs_merge.log" 2>&1
}

run_dataset sidewalk_main "${SIDEWALK_MAIN_SOURCE_JSONL}" sidewalk
run_dataset sidewalk_extra "${SIDEWALK_EXTRA_SOURCE_JSONL}" sidewalk
run_dataset foxglove "${FOXGLOVE_SOURCE_JSONL}" foxglove
run_dataset egowalk "${EGOWALK_SOURCE_JSONL}" egowalk

printf '%s\n' "[done] language-waypoint full annotation: ${RUN_ROOT}"
