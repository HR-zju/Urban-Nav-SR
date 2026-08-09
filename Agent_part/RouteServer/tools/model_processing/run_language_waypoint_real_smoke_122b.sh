#!/usr/bin/env bash
set -Eeuo pipefail

# 真实数据语言目标 smoke：只标注当前 FPV 可见的 instruction，不向模型暴露 future 轨迹坐标。
# 这个脚本刻意只接受已经构造好的 V4 source 和 V2 trajectory sidecar；原始数据集的
# planner adapter、waypoint guidance 和四 arm package 仍必须走各自独立的离线步骤。

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/lpai/volumes/mind-eb-bd-ga/suh/UrbanNavigation}"
ROUTESERVER_ROOT="${ROUTESERVER_ROOT:-${WORKSPACE_ROOT}/Urban-Nav-SR/Agent_part/RouteServer}"
PYTHON_BIN="${PYTHON_BIN:-/lpai/volumes/mind-eb-bd-ga/suh/miniconda3/envs/qwen35_vllm_cuda128/bin/python}"
SOURCE_JSONL="${SOURCE_JSONL:?set SOURCE_JSONL to a canonical Reasoning V4 source JSONL}"
TRAJECTORY_SIDECAR_JSONL="${TRAJECTORY_SIDECAR_JSONL:?set TRAJECTORY_SIDECAR_JSONL to an accepted Stage-II V2 sidecar JSONL}"
RUN_ROOT="${RUN_ROOT:-${WORKSPACE_ROOT}/runs/language_waypoint_real_smoke_20260809}"
LIMIT="${LIMIT:-64}"
CONCURRENCY="${CONCURRENCY:-2}"
MAX_RETRIES="${MAX_RETRIES:-1}"
CANDIDATE_TIME_S="${CANDIDATE_TIME_S:-4.0}"
LLM_API_BASE_URL="${LLM_API_BASE_URL:-http://127.0.0.1:18462/v1}"
LLM_MODEL="${LLM_MODEL:-qwen3.5-122b-a10b-instruction}"

usage() {
  cat <<'EOF'
Usage:
  SOURCE_JSONL=/path/to/reasoning_v4.jsonl \
  TRAJECTORY_SIDECAR_JSONL=/path/to/trajectory_sidecars_v2.jsonl \
  tools/model_processing/run_language_waypoint_real_smoke_122b.sh

Optional overrides:
  RUN_ROOT, LIMIT (1-100, default 64), CONCURRENCY (default 2), MAX_RETRIES,
  CANDIDATE_TIME_S, LLM_API_BASE_URL and LLM_MODEL.

The script first creates teacher-only candidates, then calls the LVLM with
exactly one current FPV image. It never launches a full annotation job.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

[[ "${LIMIT}" =~ ^[0-9]+$ ]] && ((LIMIT >= 1 && LIMIT <= 100)) \
  || { echo "[error] LIMIT must be an integer in [1, 100]" >&2; exit 2; }
[[ "${CONCURRENCY}" =~ ^[0-9]+$ ]] && ((CONCURRENCY >= 1)) \
  || { echo "[error] CONCURRENCY must be a positive integer" >&2; exit 2; }
[[ -x "${PYTHON_BIN}" ]] || { echo "[error] missing Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -f "${SOURCE_JSONL}" ]] || { echo "[error] missing source: ${SOURCE_JSONL}" >&2; exit 2; }
[[ -f "${TRAJECTORY_SIDECAR_JSONL}" ]] || { echo "[error] missing sidecar: ${TRAJECTORY_SIDECAR_JSONL}" >&2; exit 2; }

mkdir -p "${RUN_ROOT}"/{inputs,annotations,reports,logs,review}

export PYTHONPATH="${ROUTESERVER_ROOT}/src:${ROUTESERVER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LLM_PROVIDER="${LLM_PROVIDER:-openai}"
export LLM_API_STYLE="${LLM_API_STYLE:-chat_completions}"
export LLM_API_KEY="${LLM_API_KEY:-local}"
export LLM_API_BASE_URL
export LLM_MODEL
export LLM_TEMPERATURE="${LLM_TEMPERATURE:-0}"
export LLM_CHAT_TEMPLATE_ENABLE_THINKING="${LLM_CHAT_TEMPLATE_ENABLE_THINKING:-false}"
export LLM_VISION_RESPONSE_FORMAT_JSON="${LLM_VISION_RESPONSE_FORMAT_JSON:-true}"
export LLM_VISION_TIMEOUT_S="${LLM_VISION_TIMEOUT_S:-900}"
export LLM_VISION_MAX_TOKENS="${LLM_VISION_MAX_TOKENS:-512}"

curl --noproxy 127.0.0.1,localhost -fsS --max-time 30 \
  "${LLM_API_BASE_URL}/models" >"${RUN_ROOT}/reports/model_listing.json"

# 先按 sidecar identity 截出少量真实 source，避免 smoke 意外扫描或标注整个数据集。
SOURCE_JSONL="${SOURCE_JSONL}" \
TRAJECTORY_SIDECAR_JSONL="${TRAJECTORY_SIDECAR_JSONL}" \
OUTPUT_JSONL="${RUN_ROOT}/inputs/reasoning_sources_smoke.jsonl" \
LIMIT="${LIMIT}" \
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

from mcp_server.data.io.jsonl import iter_jsonl

sidecar_ids = {str(row["sample_id"]) for row in iter_jsonl(os.environ["TRAJECTORY_SIDECAR_JSONL"])}
selected = []
for row in iter_jsonl(os.environ["SOURCE_JSONL"]):
    observation = row.get("observation") or {}
    parent_id = str(observation.get("source_group_id") or "")
    images = observation.get("images") or []
    roles = observation.get("image_roles") or []
    current_images = [image for image, role in zip(images, roles) if role == "fpv.current"]
    if parent_id not in sidecar_ids or not current_images or not Path(current_images[0]).is_file():
        continue
    selected.append(row)
    if len(selected) >= int(os.environ["LIMIT"]):
        break
target = Path(os.environ["OUTPUT_JSONL"])
target.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in selected), encoding="utf-8")
if not selected:
    raise SystemExit("no source rows with a matching sidecar and fpv.current image")
print(json.dumps({"selected_source_rows": len(selected)}, ensure_ascii=False))
PY

"${PYTHON_BIN}" "${ROUTESERVER_ROOT}/tools/evaluation/build_language_waypoint_benchmark.py" \
  candidates \
  --source-jsonl "${RUN_ROOT}/inputs/reasoning_sources_smoke.jsonl" \
  --trajectory-sidecar-jsonl "${TRAJECTORY_SIDECAR_JSONL}" \
  --output-jsonl "${RUN_ROOT}/inputs/language_goal_candidates_smoke.jsonl" \
  --report-json "${RUN_ROOT}/reports/candidate_build_report.json" \
  --candidate-time-s "${CANDIDATE_TIME_S}"

"${PYTHON_BIN}" "${ROUTESERVER_ROOT}/tools/model_processing/label_language_waypoint_goals.py" \
  --source-jsonl "${RUN_ROOT}/inputs/reasoning_sources_smoke.jsonl" \
  --candidates-jsonl "${RUN_ROOT}/inputs/language_goal_candidates_smoke.jsonl" \
  --annotation-jsonl "${RUN_ROOT}/annotations/language_goal_annotations.jsonl" \
  --decisions-jsonl "${RUN_ROOT}/annotations/language_goal_decisions.jsonl" \
  --report-json "${RUN_ROOT}/reports/language_goal_annotation_report.json" \
  --limit "${LIMIT}" \
  --concurrency "${CONCURRENCY}" \
  --max-retries "${MAX_RETRIES}" \
  --progress-interval 8 \
  | tee "${RUN_ROOT}/logs/annotation.log"

echo "[done] run_root=${RUN_ROOT}"
