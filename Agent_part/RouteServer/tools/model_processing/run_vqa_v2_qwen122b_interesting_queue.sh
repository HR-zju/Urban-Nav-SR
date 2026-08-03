#!/usr/bin/env bash
set -Eeuo pipefail

# 使用两套本地 Qwen3.5-122B 服务，对已完成 behavior filter 的 interesting
# 样本执行 VQA V2 标注；所有输入、断点、报告和日志均保存在共享工作盘。

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/lpai/volumes/mind-eb-bd-ga/suh/UrbanNavigation}"
ROUTESERVER_ROOT="${ROUTESERVER_ROOT:-${WORKSPACE_ROOT}/Urban-Nav-SR/Agent_part/RouteServer}"
PYTHON_BIN="${PYTHON_BIN:-/lpai/volumes/mind-eb-bd-ga/suh/miniconda3/envs/qwen35_vllm_cuda128/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/lpai/volumes/mind-eb-bd-ga/suh/dataset/urbannav-sidewalk-agent-data-v1}"
FILTER_ROOT="${FILTER_ROOT:-${WORKSPACE_ROOT}/runs/qwen35_122b_a10b_20260716/behavior_filter}"
RUN_ROOT="${RUN_ROOT:-${WORKSPACE_ROOT}/annotation_runs/vqa_v2_0_qwen122b_interesting_full_v1}"

BEHAVIOR_BASE_URL="${BEHAVIOR_BASE_URL:-http://127.0.0.1:18461/v1}"
BEHAVIOR_MODEL="${BEHAVIOR_MODEL:-qwen3.5-122b-a10b-behavior}"
INSTRUCTION_BASE_URL="${INSTRUCTION_BASE_URL:-http://127.0.0.1:18462/v1}"
INSTRUCTION_MODEL="${INSTRUCTION_MODEL:-qwen3.5-122b-a10b-instruction}"
CONCURRENCY_PER_SERVICE="${CONCURRENCY_PER_SERVICE:-16}"
MAX_RETRIES="${MAX_RETRIES:-2}"
IMAGE_DETAIL="${IMAGE_DETAIL:-low}"
LLM_VISION_TIMEOUT_S="${LLM_VISION_TIMEOUT_S:-900}"
LLM_VISION_MAX_TOKENS="${LLM_VISION_MAX_TOKENS:-512}"

FPV_PROFILE="fpv_degree10_terminal_centroid_v2"
EGO_PROFILE="ego_degree10_12way_terminal_centroid_v2"
CAPABILITY_JSON="${ROUTESERVER_ROOT}/config/vqa_v2_stage_a_capabilities.json"
QUEUE_STATE="${RUN_ROOT}/reports/queue_state.txt"

FOXGLOVE_INPUT="${FILTER_ROOT}/foxglove/interesting.jsonl"
SIDEWALK_MAIN_INPUT="${FILTER_ROOT}/sidewalk_main/interesting.jsonl"
SIDEWALK_EXTRA_INPUT="${FILTER_ROOT}/sidewalk_extra/interesting.jsonl"
EGOWALK_A_INPUT="${FILTER_ROOT}/egowalk_part_a/interesting.jsonl"
EGOWALK_B_INPUT="${FILTER_ROOT}/egowalk_part_b/interesting.jsonl"

FOXGLOVE_CANONICAL_INPUT="${FOXGLOVE_CANONICAL_INPUT:-${WORKSPACE_ROOT}/annotation_runs/reasoning_v4_6_qwen122b_foxglove_20way_routed_visual_full_v1/inputs/stamped_input.jsonl}"
SIDEWALK_MAIN_CANONICAL_INPUT="${SIDEWALK_MAIN_CANONICAL_INPUT:-${WORKSPACE_ROOT}/annotation_runs/reasoning_v4_6_qwen122b_sidewalk_main_20way_routed_visual_full_v1/inputs/stamped_input.jsonl}"
SIDEWALK_EXTRA_CANONICAL_INPUT="${SIDEWALK_EXTRA_CANONICAL_INPUT:-${WORKSPACE_ROOT}/annotation_inputs/reasoning_v4_6/sidewalk_extra_unique_20way.jsonl}"
EGOWALK_CANONICAL_INPUT="${EGOWALK_CANONICAL_INPUT:-${WORKSPACE_ROOT}/annotation_inputs/reasoning_v4_6/egowalk_ego_12way.jsonl}"

EGOWALK_COT_RUN_ROOT="${EGOWALK_COT_RUN_ROOT:-${WORKSPACE_ROOT}/annotation_runs/reasoning_v4_6_qwen122b_egowalk_12way_routed_visual_full_v1}"
EGOWALK_COT_INPUT="${EGOWALK_COT_INPUT:-${WORKSPACE_ROOT}/annotation_inputs/reasoning_v4_6/egowalk_ego_12way.jsonl}"
RESUME_EGOWALK_COT="${RESUME_EGOWALK_COT:-1}"

log() {
  printf '[vqa-interesting] %s %s\n' "$(date --iso-8601=seconds)" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

require_file() {
  [[ -s "$1" ]] || die "required file is missing or empty: $1"
}

write_state() {
  mkdir -p "$(dirname "${QUEUE_STATE}")"
  printf '%s\n' "$1" >"${QUEUE_STATE}"
  log "stage=$1"
}

python_tool() {
  PYTHONPATH="${ROUTESERVER_ROOT}/src:${ROUTESERVER_ROOT}" \
    "${PYTHON_BIN}" "$@"
}

strict_chat_smoke() {
  local base_url="$1"
  local model="$2"
  local response
  response="$(
    curl -fsS --max-time 120 \
      "${base_url}/chat/completions" \
      -H 'Content-Type: application/json' \
      --data-binary \
      "{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"Return exactly one JSON object: {\\\"status\\\":\\\"OK\\\"}\"}],\"response_format\":{\"type\":\"json_object\"},\"temperature\":0,\"max_tokens\":16}"
  )"
  RESPONSE="${response}" "${PYTHON_BIN}" -c \
    'import json, os; p=json.loads(os.environ["RESPONSE"]); assert json.loads(p["choices"][0]["message"]["content"]) == {"status":"OK"}'
  log "service_ok model=${model} base_url=${base_url}"
}

preflight() {
  local file
  [[ -x "${PYTHON_BIN}" ]] || die "Python is unavailable: ${PYTHON_BIN}"
  for file in \
    "${CAPABILITY_JSON}" \
    "${ROUTESERVER_ROOT}/tools/model_processing/stamp_reasoning_v4_directions.py" \
    "${ROUTESERVER_ROOT}/tools/model_processing/build_reasoning_v4_sources.py" \
    "${ROUTESERVER_ROOT}/tools/model_processing/build_vqa_v2_sources.py" \
    "${ROUTESERVER_ROOT}/tools/model_processing/label_vqa_v2.py" \
    "${ROUTESERVER_ROOT}/tools/model_processing/export_vqa_v2_views.py" \
    "${ROUTESERVER_ROOT}/tools/model_processing/audit_vqa_v2_artifacts.py" \
    "${FOXGLOVE_INPUT}" \
    "${SIDEWALK_MAIN_INPUT}" \
    "${SIDEWALK_EXTRA_INPUT}" \
    "${EGOWALK_A_INPUT}" \
    "${EGOWALK_B_INPUT}" \
    "${FOXGLOVE_CANONICAL_INPUT}" \
    "${SIDEWALK_MAIN_CANONICAL_INPUT}" \
    "${SIDEWALK_EXTRA_CANONICAL_INPUT}" \
    "${EGOWALK_CANONICAL_INPUT}"; do
    require_file "${file}"
  done
  [[ "${CONCURRENCY_PER_SERVICE}" =~ ^[1-9][0-9]*$ ]] \
    || die "CONCURRENCY_PER_SERVICE must be positive"
  strict_chat_smoke "${BEHAVIOR_BASE_URL}" "${BEHAVIOR_MODEL}"
  strict_chat_smoke "${INSTRUCTION_BASE_URL}" "${INSTRUCTION_MODEL}"
  python_tool -c \
    'from mcp_server.data.vqa_v2_eligibility import load_vqa_capabilities_v2; c=load_vqa_capabilities_v2(); assert sum(t.training_export_enabled for t in c.tasks)==18'
}

select_canonical_interesting() {
  local family="$1"
  local output_jsonl="$2"
  local report_json="$3"
  shift 3
  mkdir -p "$(dirname "${output_jsonl}")" "$(dirname "${report_json}")"

  # behavior filter 只决定成员资格；方向和私有监督始终取 canonical 输入。
  python_tool - "${family}" "${output_jsonl}" "${report_json}" "$@" <<'PY'
import json
import os
import sys
from pathlib import Path

family = sys.argv[1]
output = Path(sys.argv[2])
report_path = Path(sys.argv[3])
arguments = sys.argv[4:]
try:
    separator = arguments.index("--interesting")
except ValueError as exc:
    raise ValueError("missing --interesting separator") from exc
canonical_paths = [Path(value) for value in arguments[:separator]]
interesting_paths = [Path(value) for value in arguments[separator + 1:]]
if not canonical_paths or not interesting_paths:
    raise ValueError("canonical and interesting path lists must be non-empty")

def stable_key(row):
    if family in {"foxglove", "egowalk"}:
        value = row.get("id") or row.get("sample_id") or row.get("source_record_id")
        return f"{family}:id:{value}" if value else None
    metadata = row.get("sidewalk_meta")
    if not isinstance(metadata, dict):
        metadata = {}
    clip = metadata.get("clip_id") or row.get("clip_id")
    frame = metadata.get("frame_name")
    if frame is None:
        frame = metadata.get("frame_idx")
    if frame is None:
        frame = row.get("frame_idx")
    if clip and frame is not None:
        return f"sidewalk:clip_frame:{clip}:{frame}"
    return None

wanted = set()
interesting_rows = 0
missing_interesting_keys = 0
for path in interesting_paths:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            interesting_rows += 1
            value = stable_key(json.loads(line))
            if value is None:
                missing_interesting_keys += 1
            else:
                wanted.add(value)

tmp = output.with_name(output.name + f".tmp.{os.getpid()}")
selected = set()
duplicate_canonical_matches = 0
with tmp.open("w", encoding="utf-8") as target:
    for path in canonical_paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                value = stable_key(row)
                if value not in wanted:
                    continue
                if value in selected:
                    duplicate_canonical_matches += 1
                    continue
                selected.add(value)
                target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

unmatched = sorted(wanted - selected)
report = {
    "schema_version": "urbannav_vqa_v2_interesting_canonical_selection_v1",
    "family": family,
    "canonical_inputs": [str(path) for path in canonical_paths],
    "interesting_inputs": [str(path) for path in interesting_paths],
    "interesting_rows": interesting_rows,
    "interesting_unique_keys": len(wanted),
    "missing_interesting_keys": missing_interesting_keys,
    "selected_canonical_rows": len(selected),
    "duplicate_canonical_matches": duplicate_canonical_matches,
    "unmatched_interesting_keys": len(unmatched),
    "unmatched_examples": unmatched[:20],
    "passed": missing_interesting_keys == 0 and not unmatched,
}
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
if not report["passed"]:
    tmp.unlink(missing_ok=True)
    raise ValueError(f"canonical interesting selection failed: {report}")
tmp.replace(output)
print(json.dumps(report, ensure_ascii=False))
PY
}

prepare_partition() {
  local name="$1"
  local dataset_source="$2"
  local profile_id="$3"
  local input_jsonl="$4"
  local part_root="${RUN_ROOT}/partitions/${name}"
  local stamped="${part_root}/stamped.jsonl"
  local reasoning_sources="${part_root}/reasoning_sources.jsonl"
  local vqa_sources="${part_root}/vqa_sources.jsonl"
  local marker="${part_root}/prepare.complete"
  local input_sha code_digest cache_key expected_rows

  mkdir -p "${part_root}/reports"
  input_sha="$(sha256sum "${input_jsonl}" | awk '{print $1}')"
  code_digest="$(
    sha256sum \
      "${ROUTESERVER_ROOT}/src/mcp_server/contracts/direction_profile.py" \
      "${ROUTESERVER_ROOT}/src/mcp_server/contracts/vqa_v2.py" \
      "${ROUTESERVER_ROOT}/src/mcp_server/contracts/vqa_dataset_v2.py" \
      "${ROUTESERVER_ROOT}/src/mcp_server/data/reasoning_v4_source.py" \
      "${ROUTESERVER_ROOT}/src/mcp_server/data/vqa_v2_source.py" \
      "${ROUTESERVER_ROOT}/src/mcp_server/teacher/vqa_v2_profiles.py" \
      | sha256sum | awk '{print $1}'
  )"
  cache_key="$(printf '%s\n' "${input_sha}" "${code_digest}" "${dataset_source}" "${profile_id}" | sha256sum | awk '{print $1}')"
  if [[ -s "${marker}" ]] \
    && [[ "$(cat "${marker}")" == "${cache_key}" ]] \
    && [[ -s "${vqa_sources}" ]]; then
    log "reuse_prepared_partition name=${name} rows=$(wc -l <"${vqa_sources}")"
    return
  fi

  rm -f "${marker}"
  expected_rows="$(wc -l <"${input_jsonl}" | tr -d ' ')"
  log "prepare_partition name=${name} dataset=${dataset_source} rows=${expected_rows}"
  python_tool \
    "${ROUTESERVER_ROOT}/tools/model_processing/stamp_reasoning_v4_directions.py" \
    --input-jsonl "${input_jsonl}" \
    --output-jsonl "${stamped}" \
    --report-json "${part_root}/reports/direction_stamp.json" \
    --dataset-source "${dataset_source}" \
    --profile-id "${profile_id}" \
    --selection sequential
  python_tool \
    "${ROUTESERVER_ROOT}/tools/model_processing/build_reasoning_v4_sources.py" \
    --input-jsonl "${stamped}" \
    --output-jsonl "${reasoning_sources}" \
    --report-json "${part_root}/reports/reasoning_source_build.json" \
    --dataset-source "${dataset_source}" \
    --travel-mode pedestrian_assist \
    --max-fpv-frames 5 \
    --media-root "${DATASET_ROOT}"
  python_tool \
    "${ROUTESERVER_ROOT}/tools/model_processing/build_vqa_v2_sources.py" \
    --input-jsonl "${reasoning_sources}" \
    --output-jsonl "${vqa_sources}"
  [[ "$(wc -l <"${vqa_sources}" | tr -d ' ')" == "${expected_rows}" ]] \
    || die "partition row count mismatch: ${name}"
  printf '%s\n' "${cache_key}" >"${marker}"
}

merge_vqa_sources() {
  local dataset_name="$1"
  shift
  local dataset_root="${RUN_ROOT}/datasets/${dataset_name}"
  local output="${dataset_root}/vqa_sources.jsonl"
  local input
  mkdir -p "${dataset_root}/sources/shards" "${dataset_root}/annotations" \
    "${dataset_root}/views" "${dataset_root}/reports" "${dataset_root}/logs"

  # 相同 source_group_id 只有 fingerprint 完全一致时才允许去重。
  python_tool - "${output}" "$@" <<'PY'
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
inputs = [Path(value) for value in sys.argv[2:]]
tmp = output.with_name(output.name + f".tmp.{os.getpid()}")
seen = {}
rows = 0
duplicates = 0
with tmp.open("w", encoding="utf-8") as target:
    for source_path in inputs:
        with source_path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                group_id = row["observation"]["source_group_id"]
                fingerprint = row["source_fingerprint"]
                previous = seen.get(group_id)
                if previous is not None:
                    if previous != fingerprint:
                        raise ValueError(
                            f"conflicting duplicate source_group_id: {group_id}"
                        )
                    duplicates += 1
                    continue
                seen[group_id] = fingerprint
                target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                rows += 1
tmp.replace(output)
print(json.dumps({"rows": rows, "duplicates_removed": duplicates, "inputs": [str(p) for p in inputs]}))
PY
  awk 'NR % 2 == 1 {print}' "${output}" \
    >"${dataset_root}/sources/shards/service_0.jsonl"
  awk 'NR % 2 == 0 {print}' "${output}" \
    >"${dataset_root}/sources/shards/service_1.jsonl"
  log "dataset_ready name=${dataset_name} rows=$(wc -l <"${output}")"
}

run_vqa_worker() {
  local dataset_name="$1"
  local worker_id="$2"
  local base_url="$3"
  local model="$4"
  local dataset_root="${RUN_ROOT}/datasets/${dataset_name}"
  env \
    PYTHONPATH="${ROUTESERVER_ROOT}/src:${ROUTESERVER_ROOT}" \
    TMPDIR="${RUN_ROOT}/tmp" \
    NO_PROXY="127.0.0.1,localhost" \
    no_proxy="127.0.0.1,localhost" \
    LLM_PROVIDER=openai \
    LLM_API_STYLE=chat_completions \
    LLM_API_STYLE_FALLBACK=0 \
    LLM_API_KEY=local-vllm \
    LLM_API_BASE_URL="${base_url}" \
    LLM_MODEL="${model}" \
    LLM_TEMPERATURE=0 \
    LLM_MAX_RETRIES=0 \
    LLM_CHAT_TEMPLATE_ENABLE_THINKING=false \
    LLM_RESPONSE_FORMAT_JSON=true \
    LLM_VISION_RESPONSE_FORMAT_JSON=1 \
    LLM_VISION_TIMEOUT_S="${LLM_VISION_TIMEOUT_S}" \
    LLM_VISION_MAX_TOKENS="${LLM_VISION_MAX_TOKENS}" \
    "${PYTHON_BIN}" \
    "${ROUTESERVER_ROOT}/tools/model_processing/label_vqa_v2.py" \
    --source-jsonl "${dataset_root}/sources/shards/service_${worker_id}.jsonl" \
    --annotation-jsonl "${dataset_root}/annotations/service_${worker_id}.jsonl" \
    --report-json "${dataset_root}/reports/annotation_service_${worker_id}.json" \
    --capability-json "${CAPABILITY_JSON}" \
    --max-retries "${MAX_RETRIES}" \
    --concurrency "${CONCURRENCY_PER_SERVICE}" \
    --image-detail "${IMAGE_DETAIL}" \
    2>&1 | tee -a "${dataset_root}/logs/service_${worker_id}.log"
}

postprocess_dataset() {
  local dataset_name="$1"
  local dataset_root="${RUN_ROOT}/datasets/${dataset_name}"
  local merged_annotations="${dataset_root}/annotations/vqa_annotations.jsonl"
  cat \
    "${dataset_root}/annotations/service_0.jsonl" \
    "${dataset_root}/annotations/service_1.jsonl" \
    >"${merged_annotations}.tmp"
  mv "${merged_annotations}.tmp" "${merged_annotations}"
  python_tool \
    "${ROUTESERVER_ROOT}/tools/model_processing/export_vqa_v2_views.py" \
    --source-jsonl "${dataset_root}/vqa_sources.jsonl" \
    --annotation-jsonl "${merged_annotations}" \
    --output-jsonl "${dataset_root}/views/vqa_training_views.jsonl" \
    --capability-json "${CAPABILITY_JSON}"
  python_tool \
    "${ROUTESERVER_ROOT}/tools/model_processing/audit_vqa_v2_artifacts.py" \
    --source-jsonl "${dataset_root}/vqa_sources.jsonl" \
    --annotation-jsonl "${merged_annotations}" \
    --training-view-jsonl "${dataset_root}/views/vqa_training_views.jsonl" \
    --capability-json "${CAPABILITY_JSON}" \
    --output-json "${dataset_root}/reports/vqa_audit.json"
}

label_dataset() {
  local dataset_name="$1"
  local pid0 pid1 status0 status1
  write_state "vqa_label_${dataset_name}"
  run_vqa_worker "${dataset_name}" 0 "${BEHAVIOR_BASE_URL}" "${BEHAVIOR_MODEL}" &
  pid0=$!
  run_vqa_worker "${dataset_name}" 1 "${INSTRUCTION_BASE_URL}" "${INSTRUCTION_MODEL}" &
  pid1=$!
  set +e
  wait "${pid0}"
  status0=$?
  wait "${pid1}"
  status1=$?
  set -e
  [[ "${status0}" == 0 && "${status1}" == 0 ]] \
    || die "VQA worker failed: dataset=${dataset_name} service_0=${status0} service_1=${status1}"
  write_state "vqa_postprocess_${dataset_name}"
  postprocess_dataset "${dataset_name}"
}

prepare_all() {
  local selected_root="${RUN_ROOT}/selected_inputs"
  write_state "prepare_sources"
  select_canonical_interesting \
    foxglove \
    "${selected_root}/foxglove.jsonl" \
    "${RUN_ROOT}/reports/selection_foxglove.json" \
    "${FOXGLOVE_CANONICAL_INPUT}" \
    --interesting "${FOXGLOVE_INPUT}"
  select_canonical_interesting \
    sidewalk \
    "${selected_root}/sidewalk.jsonl" \
    "${RUN_ROOT}/reports/selection_sidewalk.json" \
    "${SIDEWALK_MAIN_CANONICAL_INPUT}" \
    "${SIDEWALK_EXTRA_CANONICAL_INPUT}" \
    --interesting "${SIDEWALK_MAIN_INPUT}" "${SIDEWALK_EXTRA_INPUT}"
  select_canonical_interesting \
    egowalk \
    "${selected_root}/egowalk.jsonl" \
    "${RUN_ROOT}/reports/selection_egowalk.json" \
    "${EGOWALK_CANONICAL_INPUT}" \
    --interesting "${EGOWALK_A_INPUT}" "${EGOWALK_B_INPUT}"

  prepare_partition foxglove foxglove "${FPV_PROFILE}" \
    "${selected_root}/foxglove.jsonl"
  prepare_partition sidewalk sidewalk "${FPV_PROFILE}" \
    "${selected_root}/sidewalk.jsonl"
  prepare_partition egowalk egowalk "${EGO_PROFILE}" \
    "${selected_root}/egowalk.jsonl"

  merge_vqa_sources foxglove "${RUN_ROOT}/partitions/foxglove/vqa_sources.jsonl"
  merge_vqa_sources sidewalk "${RUN_ROOT}/partitions/sidewalk/vqa_sources.jsonl"
  merge_vqa_sources egowalk "${RUN_ROOT}/partitions/egowalk/vqa_sources.jsonl"
}

write_provenance() {
  local destination="${RUN_ROOT}/reports/run_provenance.txt"
  mkdir -p "${RUN_ROOT}/reports" "${RUN_ROOT}/tmp"
  {
    echo "schema_version=urbannav_vqa_v2_interesting_qwen122b_run_v1"
    echo "created_at=$(date --iso-8601=seconds)"
    echo "code_root=${ROUTESERVER_ROOT}"
    echo "code_commit=$(git -C "${ROUTESERVER_ROOT}" rev-parse HEAD)"
    echo "code_dirty_entry_count=$(git -C "${ROUTESERVER_ROOT}" status --porcelain | wc -l | tr -d ' ')"
    echo "filter_root=${FILTER_ROOT}"
    echo "canonical_selection=stable_id_or_sidewalk_clip_frame"
    echo "foxglove_canonical_input=${FOXGLOVE_CANONICAL_INPUT}"
    echo "sidewalk_main_canonical_input=${SIDEWALK_MAIN_CANONICAL_INPUT}"
    echo "sidewalk_extra_canonical_input=${SIDEWALK_EXTRA_CANONICAL_INPUT}"
    echo "egowalk_canonical_input=${EGOWALK_CANONICAL_INPUT}"
    echo "capability_json=${CAPABILITY_JSON}"
    echo "capability_boundary=C3_C4_C6_and_other_disabled_tasks_remain_disabled"
    echo "behavior_service=${BEHAVIOR_BASE_URL}|${BEHAVIOR_MODEL}"
    echo "instruction_service=${INSTRUCTION_BASE_URL}|${INSTRUCTION_MODEL}"
    echo "concurrency_per_service=${CONCURRENCY_PER_SERVICE}"
    echo "max_retries=${MAX_RETRIES}"
    echo "image_detail=${IMAGE_DETAIL}"
    echo "vision_max_tokens=${LLM_VISION_MAX_TOKENS}"
    echo "post_vqa_queue=resume_egowalk_cot"
    echo "sidewalk_extra_cot_status=deferred_by_current_queue_priority"
  } >"${destination}"
}

resume_egowalk_cot() {
  [[ "${RESUME_EGOWALK_COT}" == 1 ]] || {
    write_state "complete_vqa_only"
    return
  }
  require_file "${EGOWALK_COT_INPUT}"
  write_state "resume_egowalk_cot"
  env \
    WORKSPACE_ROOT="${WORKSPACE_ROOT}" \
    ROUTESERVER_ROOT="${ROUTESERVER_ROOT}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    DATASET_ROOT="${DATASET_ROOT}" \
    DATASET_SOURCE=egowalk \
    MEDIA_ROOT="${DATASET_ROOT}" \
    BEHAVIOR_BASE_URL="${BEHAVIOR_BASE_URL}" \
    BEHAVIOR_MODEL="${BEHAVIOR_MODEL}" \
    INSTRUCTION_BASE_URL="${INSTRUCTION_BASE_URL}" \
    INSTRUCTION_MODEL="${INSTRUCTION_MODEL}" \
    DIRECTION_PROFILE_ID="${EGO_PROFILE}" \
    CONCURRENCY_PER_SERVICE=8 \
    MAX_RETRIES=2 \
    IMAGE_DETAIL=low \
    NAVIGATION_CONTEXT_IMAGE_DETAIL=high \
    SECTION_IMAGE_POLICY_ID=reasoning_v4_section_images_routed_visual_v1 \
    RETRY_FAILED_ON_RESUME=1 \
    LLM_VISION_TIMEOUT_S=900 \
    LLM_VISION_MAX_TOKENS=4096 \
    INPUT_JSONL="${EGOWALK_COT_INPUT}" \
    RUN_ROOT="${EGOWALK_COT_RUN_ROOT}" \
    EXPECTED_ROWS=284470 \
    "${ROUTESERVER_ROOT}/tools/model_processing/run_reasoning_v4_qwen122b_dual_annotation.sh" \
    run-full
  write_state "complete"
}

main() {
  trap 'status=$?; write_state "failed_exit_${status}"; exit ${status}' ERR
  mkdir -p "${RUN_ROOT}/reports" "${RUN_ROOT}/tmp"
  write_provenance
  write_state "preflight"
  preflight
  prepare_all
  label_dataset foxglove
  label_dataset sidewalk
  label_dataset egowalk
  resume_egowalk_cot
}

main "$@"
