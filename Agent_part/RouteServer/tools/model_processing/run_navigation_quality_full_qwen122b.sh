#!/usr/bin/env bash
# 全量数据质量过滤：只写 geometry/filter sidecar 与派生视图，不删除、移动或覆盖 source/media。
set -Eeuo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/lpai/volumes/mind-eb-bd-ga/suh/UrbanNavigation}"
ROUTESERVER_ROOT="${ROUTESERVER_ROOT:-${WORKSPACE_ROOT}/Urban-Nav-SR/Agent_part/RouteServer}"
PYTHON_BIN="${PYTHON_BIN:-/lpai/volumes/mind-eb-bd-ga/suh/miniconda3/envs/qwen35_vllm_cuda128/bin/python}"
RUN_ROOT="${RUN_ROOT:-${WORKSPACE_ROOT}/runs/navigation_quality_v1_qwen122b_full_20260809}"

SIDEWALK_MAIN_SOURCE_JSONL="${SIDEWALK_MAIN_SOURCE_JSONL:-${WORKSPACE_ROOT}/annotation_runs/reasoning_v4_6_qwen122b_sidewalk_main_20way_routed_visual_full_v1/sources/reasoning_v4_5_sources.jsonl}"
SIDEWALK_EXTRA_SOURCE_JSONL="${SIDEWALK_EXTRA_SOURCE_JSONL:-${WORKSPACE_ROOT}/annotation_runs/reasoning_v4_6_qwen122b_sidewalk_extra_20way_routed_visual_full_v2/sources/reasoning_v4_5_sources.jsonl}"
FOXGLOVE_SOURCE_JSONL="${FOXGLOVE_SOURCE_JSONL:-${WORKSPACE_ROOT}/annotation_runs/reasoning_v4_6_qwen122b_foxglove_20way_routed_visual_full_v1/sources/reasoning_v4_5_sources.jsonl}"
EGOWALK_SOURCE_JSONL="${EGOWALK_SOURCE_JSONL:-${WORKSPACE_ROOT}/annotation_runs/reasoning_v4_6_qwen122b_egowalk_12way_routed_visual_full_v1/sources/reasoning_v4_5_sources.jsonl}"
TRAJECTORY_SIDECAR_JSONLS="${TRAJECTORY_SIDECAR_JSONLS:?set TRAJECTORY_SIDECAR_JSONLS to colon-separated 4s/32 V2 sidecar JSONLs}"

BEHAVIOR_BASE_URL="${BEHAVIOR_BASE_URL:-http://127.0.0.1:18461/v1}"
BEHAVIOR_MODEL="${BEHAVIOR_MODEL:-qwen3.5-122b-a10b-behavior}"
INSTRUCTION_BASE_URL="${INSTRUCTION_BASE_URL:-http://127.0.0.1:18462/v1}"
INSTRUCTION_MODEL="${INSTRUCTION_MODEL:-qwen3.5-122b-a10b-instruction}"
GEOMETRY_CONCURRENCY_PER_DATASET="${GEOMETRY_CONCURRENCY_PER_DATASET:-4}"
SUITABILITY_CONCURRENCY_PER_DATASET="${SUITABILITY_CONCURRENCY_PER_DATASET:-4}"
MAX_RETRIES="${MAX_RETRIES:-1}"
MAX_TEMPORAL_FPV_FRAMES="${MAX_TEMPORAL_FPV_FRAMES:-5}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-250}"

for required in "${PYTHON_BIN}" "${SIDEWALK_MAIN_SOURCE_JSONL}" "${SIDEWALK_EXTRA_SOURCE_JSONL}" "${FOXGLOVE_SOURCE_JSONL}" "${EGOWALK_SOURCE_JSONL}"; do
  [[ -e "${required}" ]] || { echo "[error] missing required path: ${required}" >&2; exit 2; }
done
mkdir -p "${RUN_ROOT}"/{inputs,logs,receipts,visualization}
export PYTHONPATH="${ROUTESERVER_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export LLM_PROVIDER=openai LLM_API_KEY=local-vllm LLM_API_STYLE=chat_completions
export LLM_TEMPERATURE=0 LLM_TIMEOUT_S=900 LLM_MAX_RETRIES=0 LLM_CHAT_TEMPLATE_ENABLE_THINKING=false

curl --noproxy 127.0.0.1,localhost -fsS --max-time 30 "${BEHAVIOR_BASE_URL}/models" >"${RUN_ROOT}/receipts/behavior_models.json"
curl --noproxy 127.0.0.1,localhost -fsS --max-time 30 "${INSTRUCTION_BASE_URL}/models" >"${RUN_ROOT}/receipts/instruction_models.json"

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
{"schema_version":"urbannav_visualization_manifest_v1","run_kind":"navigation_quality_filter","source_media_unchanged":true,"planned_artifacts":["dataset quality report","per-dataset geometry and suitability review HTML"]}
EOF

register_source() {
  local dataset="$1"
  local source_jsonl="$2"
  local root="${RUN_ROOT}/${dataset}"
  mkdir -p "${root}"/{source,geometry,suitability,views,events,review}
  ln -sfn "${source_jsonl}" "${root}/source/source.jsonl"
  SOURCE_JSONL="${source_jsonl}" OUTPUT_JSON="${root}/source/source_selection.json" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

from mcp_server.data.io.jsonl import file_sha256, iter_jsonl

source = Path(os.environ["SOURCE_JSONL"]).resolve()
count = sum(1 for _ in iter_jsonl(source))
Path(os.environ["OUTPUT_JSON"]).write_text(
    json.dumps(
        {
            "selected_count": count,
            "selection_kind": "full_source_manifest",
            "source_jsonl": str(source),
            "source_jsonl_sha256": file_sha256(source),
        },
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
PY
}

run_geometry() {
  local dataset="$1"
  local root="${RUN_ROOT}/${dataset}"
  env LLM_API_BASE_URL="${BEHAVIOR_BASE_URL}" LLM_MODEL="${BEHAVIOR_MODEL}" \
    "${PYTHON_BIN}" "${ROUTESERVER_ROOT}/tools/model_processing/label_traversability_geometry.py" \
      --source-jsonl "${root}/source/source.jsonl" \
      --geometry-jsonl "${root}/geometry/geometry.jsonl" \
      --decisions-jsonl "${root}/geometry/decisions.jsonl" \
      --report-json "${root}/geometry/report.json" \
      --concurrency "${GEOMETRY_CONCURRENCY_PER_DATASET}" \
      --max-retries "${MAX_RETRIES}" --image-detail high \
      --progress-interval "${PROGRESS_INTERVAL}" \
      >"${root}/geometry/run.log" 2>&1
}

run_suitability() {
  local dataset="$1"
  local sidecar_dataset="$2"
  local root="${RUN_ROOT}/${dataset}"
  local sidecar_jsonl="${RUN_ROOT}/inputs/sidecars_by_dataset/${sidecar_dataset}_trajectory_sidecars_v2.jsonl"
  env LLM_API_BASE_URL="${INSTRUCTION_BASE_URL}" LLM_MODEL="${INSTRUCTION_MODEL}" \
    "${PYTHON_BIN}" "${ROUTESERVER_ROOT}/tools/model_processing/filter_navigation_trainability.py" \
      --source-jsonl "${root}/source/source.jsonl" \
      --trajectory-sidecar-jsonl "${sidecar_jsonl}" \
      --filter-jsonl "${root}/suitability/filter.jsonl" \
      --decisions-jsonl "${root}/suitability/decisions.jsonl" \
      --report-json "${root}/suitability/report.json" \
      --views-root "${root}/views" --views-report-json "${root}/views/report.json" \
      --concurrency "${SUITABILITY_CONCURRENCY_PER_DATASET}" \
      --max-retries "${MAX_RETRIES}" --max-temporal-fpv-frames "${MAX_TEMPORAL_FPV_FRAMES}" \
      --image-detail high --progress-interval "${PROGRESS_INTERVAL}" \
      >"${root}/suitability/run.log" 2>&1
}

register_source sidewalk_main "${SIDEWALK_MAIN_SOURCE_JSONL}"
register_source sidewalk_extra "${SIDEWALK_EXTRA_SOURCE_JSONL}"
register_source foxglove "${FOXGLOVE_SOURCE_JSONL}"
register_source egowalk "${EGOWALK_SOURCE_JSONL}"

# 两个 TP=4 服务并行工作；每个服务四数据集各四个 in-flight request，共 16 个请求上限。
for dataset in sidewalk_main sidewalk_extra foxglove egowalk; do run_geometry "${dataset}" & done
for pair in "sidewalk_main:sidewalk" "sidewalk_extra:sidewalk" "foxglove:foxglove" "egowalk:egowalk"; do
  IFS=':' read -r dataset sidecar_dataset <<< "${pair}"
  run_suitability "${dataset}" "${sidecar_dataset}" &
done
wait

# CityWalker-compatible slice tags are a deterministic read-only projection of
# completed sidecars; they never invoke another model or alter source/media.
for dataset in sidewalk_main sidewalk_extra foxglove egowalk; do
  root="${RUN_ROOT}/${dataset}"
  "${PYTHON_BIN}" "${ROUTESERVER_ROOT}/tools/model_processing/project_citywalker_scene_event_tags.py" \
    --source-jsonl "${root}/source/source.jsonl" \
    --filter-jsonl "${root}/suitability/filter.jsonl" \
    --geometry-jsonl "${root}/geometry/geometry.jsonl" \
    --output-jsonl "${root}/events/scene_event_tags.jsonl" \
    --report-json "${root}/events/report.json" \
    >"${root}/events/run.log" 2>&1
done

"${PYTHON_BIN}" "${ROUTESERVER_ROOT}/tools/evaluation/summarize_navigation_quality_run.py" \
  --run-root "${RUN_ROOT}" --output-json "${RUN_ROOT}/navigation_quality_run_report.json" \
  >"${RUN_ROOT}/logs/summarize.log" 2>&1

for dataset in sidewalk_main sidewalk_extra foxglove egowalk; do
  root="${RUN_ROOT}/${dataset}"
  "${PYTHON_BIN}" "${ROUTESERVER_ROOT}/tools/evaluation/build_traversability_geometry_review_html.py" \
    --source-jsonl "${root}/source/source.jsonl" --geometry-jsonl "${root}/geometry/geometry.jsonl" \
    --geometry-decisions-jsonl "${root}/geometry/decisions.jsonl" \
    --output-html "${root}/review/geometry_first25.html" --max-samples 25 --thumbnail-max-width-px 512 --jpeg-quality 72 \
    >"${root}/review/geometry_review.log" 2>&1
  "${PYTHON_BIN}" "${ROUTESERVER_ROOT}/tools/evaluation/build_navigation_trainability_review_html.py" \
    --source-jsonl "${root}/source/source.jsonl" --filter-jsonl "${root}/suitability/filter.jsonl" \
    --output-html "${root}/review/suitability_first25.html" --max-samples 25 --thumbnail-max-width-px 512 --jpeg-quality 72 \
    >"${root}/review/suitability_review.log" 2>&1
done
printf '%s\n' "[done] navigation quality full filter: ${RUN_ROOT}"
