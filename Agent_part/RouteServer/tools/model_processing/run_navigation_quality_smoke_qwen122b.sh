#!/usr/bin/env bash
# 在独立 run root 上执行四数据集质量过滤 smoke；不会覆盖任何 source 或旧标注。
set -euo pipefail

ROUTE_SERVER_ROOT="${ROUTE_SERVER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT to a new immutable smoke run root}"

BEHAVIOR_BASE_URL="${BEHAVIOR_BASE_URL:-http://127.0.0.1:18461/v1}"
BEHAVIOR_MODEL="${BEHAVIOR_MODEL:-qwen3.5-122b-a10b-behavior}"
INSTRUCTION_BASE_URL="${INSTRUCTION_BASE_URL:-http://127.0.0.1:18462/v1}"
INSTRUCTION_MODEL="${INSTRUCTION_MODEL:-qwen3.5-122b-a10b-instruction}"
MAX_ROWS="${MAX_ROWS:-150}"
TARGET_PER_STRATUM="${TARGET_PER_STRATUM:-25}"
GEOMETRY_CONCURRENCY="${GEOMETRY_CONCURRENCY:-4}"
SUITABILITY_CONCURRENCY="${SUITABILITY_CONCURRENCY:-4}"
MAX_RETRIES="${MAX_RETRIES:-1}"
MAX_TEMPORAL_FPV_FRAMES="${MAX_TEMPORAL_FPV_FRAMES:-5}"

: "${SIDEWALK_MAIN_SOURCE_JSONL:?set SIDEWALK_MAIN_SOURCE_JSONL}"
: "${SIDEWALK_EXTRA_SOURCE_JSONL:?set SIDEWALK_EXTRA_SOURCE_JSONL}"
: "${FOXGLOVE_SOURCE_JSONL:?set FOXGLOVE_SOURCE_JSONL}"
: "${EGOWALK_SOURCE_JSONL:?set EGOWALK_SOURCE_JSONL}"

export PYTHONPATH="${ROUTE_SERVER_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${RUN_ROOT}"

# 本地 vLLM 仍走 OpenAI-compatible client；占位 key 不会写入任何 ledger。
run_geometry() {
  local dataset="$1"
  local root="${RUN_ROOT}/${dataset}"
  env \
    LLM_PROVIDER=openai \
    LLM_API_KEY=local-vllm \
    LLM_API_STYLE=chat_completions \
    LLM_MODEL="${BEHAVIOR_MODEL}" \
    LLM_API_BASE_URL="${BEHAVIOR_BASE_URL}" \
    LLM_TEMPERATURE=0 \
    LLM_TIMEOUT_S=180 \
    LLM_MAX_RETRIES=0 \
    LLM_CHAT_TEMPLATE_ENABLE_THINKING=false \
    "${PYTHON_BIN}" "${ROUTE_SERVER_ROOT}/tools/model_processing/label_traversability_geometry.py" \
      --source-jsonl "${root}/source/smoke.jsonl" \
      --geometry-jsonl "${root}/geometry/geometry.jsonl" \
      --decisions-jsonl "${root}/geometry/decisions.jsonl" \
      --report-json "${root}/geometry/report.json" \
      --concurrency "${GEOMETRY_CONCURRENCY}" \
      --max-retries "${MAX_RETRIES}" \
      --image-detail high \
      --progress-interval 25 \
      >"${root}/geometry/run.log" 2>&1
}

run_suitability() {
  local dataset="$1"
  local sidecar_jsonl="$2"
  local root="${RUN_ROOT}/${dataset}"
  local -a sidecar_arg=()
  if [[ -n "${sidecar_jsonl}" ]]; then
    sidecar_arg=(--trajectory-sidecar-jsonl "${sidecar_jsonl}")
  fi
  env \
    LLM_PROVIDER=openai \
    LLM_API_KEY=local-vllm \
    LLM_API_STYLE=chat_completions \
    LLM_MODEL="${INSTRUCTION_MODEL}" \
    LLM_API_BASE_URL="${INSTRUCTION_BASE_URL}" \
    LLM_TEMPERATURE=0 \
    LLM_TIMEOUT_S=180 \
    LLM_MAX_RETRIES=0 \
    LLM_CHAT_TEMPLATE_ENABLE_THINKING=false \
    "${PYTHON_BIN}" "${ROUTE_SERVER_ROOT}/tools/model_processing/filter_navigation_trainability.py" \
      --source-jsonl "${root}/source/smoke.jsonl" \
      "${sidecar_arg[@]}" \
      --filter-jsonl "${root}/suitability/filter.jsonl" \
      --decisions-jsonl "${root}/suitability/decisions.jsonl" \
      --report-json "${root}/suitability/report.json" \
      --views-root "${root}/views" \
      --views-report-json "${root}/views/report.json" \
      --concurrency "${SUITABILITY_CONCURRENCY}" \
      --max-retries "${MAX_RETRIES}" \
      --max-temporal-fpv-frames "${MAX_TEMPORAL_FPV_FRAMES}" \
      --image-detail high \
      --progress-interval 25 \
      >"${root}/suitability/run.log" 2>&1
}

build_smoke() {
  local dataset="$1"
  local source_jsonl="$2"
  local dataset_source="$3"
  local root="${RUN_ROOT}/${dataset}"
  mkdir -p "${root}/source" "${root}/geometry" "${root}/suitability" "${root}/views" "${root}/events" "${root}/review"
  "${PYTHON_BIN}" "${ROUTE_SERVER_ROOT}/tools/model_processing/build_navigation_quality_smoke.py" \
    --source-jsonl "${source_jsonl}" \
    --dataset-source "${dataset_source}" \
    --output-jsonl "${root}/source/smoke.jsonl" \
    --report-json "${root}/source/smoke_selection.json" \
    --target-per-stratum "${TARGET_PER_STRATUM}" \
    --max-rows "${MAX_ROWS}" \
    >"${root}/source/build.log" 2>&1
}

build_smoke sidewalk_main "${SIDEWALK_MAIN_SOURCE_JSONL}" sidewalk
build_smoke sidewalk_extra "${SIDEWALK_EXTRA_SOURCE_JSONL}" sidewalk
build_smoke foxglove "${FOXGLOVE_SOURCE_JSONL}" foxglove
build_smoke egowalk "${EGOWALK_SOURCE_JSONL}" egowalk

# 两路 TP=4 服务分别承担 geometry 与 suitability，避免单个服务的多模态 prefill 相互阻塞。
for dataset in sidewalk_main sidewalk_extra foxglove egowalk; do
  run_geometry "${dataset}" &
done
wait

for dataset in sidewalk_main sidewalk_extra foxglove egowalk; do
  sidecar_var="${dataset^^}_TRAJECTORY_SIDECAR_JSONL"
  run_suitability "${dataset}" "${!sidecar_var:-}" &
done
wait

# 事件标签仅从已经完成的 sidecar 读取，不增加 LVLM 调用或修改 smoke source。
for dataset in sidewalk_main sidewalk_extra foxglove egowalk; do
  root="${RUN_ROOT}/${dataset}"
  "${PYTHON_BIN}" "${ROUTE_SERVER_ROOT}/tools/model_processing/project_citywalker_scene_event_tags.py" \
    --source-jsonl "${root}/source/smoke.jsonl" \
    --filter-jsonl "${root}/suitability/filter.jsonl" \
    --geometry-jsonl "${root}/geometry/geometry.jsonl" \
    --output-jsonl "${root}/events/scene_event_tags.jsonl" \
    --report-json "${root}/events/report.json" \
    >"${root}/events/run.log" 2>&1
done

"${PYTHON_BIN}" "${ROUTE_SERVER_ROOT}/tools/evaluation/summarize_navigation_quality_run.py" \
  --run-root "${RUN_ROOT}" --output-json "${RUN_ROOT}/navigation_quality_run_report.json" \
  >"${RUN_ROOT}/navigation_quality_summary.log" 2>&1

for dataset in sidewalk_main sidewalk_extra foxglove egowalk; do
  root="${RUN_ROOT}/${dataset}"
  "${PYTHON_BIN}" "${ROUTE_SERVER_ROOT}/tools/evaluation/build_traversability_geometry_review_html.py" \
    --source-jsonl "${root}/source/smoke.jsonl" \
    --geometry-jsonl "${root}/geometry/geometry.jsonl" \
    --geometry-decisions-jsonl "${root}/geometry/decisions.jsonl" \
    --output-html "${root}/review/geometry_review.html" \
    >"${root}/review/geometry_review.log" 2>&1
  "${PYTHON_BIN}" "${ROUTE_SERVER_ROOT}/tools/evaluation/build_navigation_trainability_review_html.py" \
    --source-jsonl "${root}/source/smoke.jsonl" \
    --filter-jsonl "${root}/suitability/filter.jsonl" \
    --output-html "${root}/review/suitability_review.html" \
    >"${root}/review/suitability_review.log" 2>&1
done

printf '%s\n' "[done] navigation quality smoke: ${RUN_ROOT}"
