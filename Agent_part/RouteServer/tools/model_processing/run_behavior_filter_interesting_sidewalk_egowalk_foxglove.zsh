#!/usr/bin/env zsh
set -euo pipefail

ROUTE_ROOT="${ROUTE_ROOT:-/home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer}"
cd "${ROUTE_ROOT}"

export PYTHONPATH="${ROUTE_ROOT}/src:${PYTHONPATH:-}"

KEY_FILE="${KEY_FILE:-/home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/key.txt}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

MODEL="${MODEL:-gemini-3_1-pro}"
BASE_URL="${BASE_URL:-http://127.0.0.1:18088/v1}"
MODEL_SLUG="${MODEL//[^A-Za-z0-9_]/_}"
RUN_ROOT="${RUN_ROOT:-/media/ubuntu/data2/SR-Nav Dataset/value_filter_interesting_${MODEL_SLUG}_${RUN_TAG}}"
CONCURRENCY="${CONCURRENCY:-24}"
MAX_FRAMES_PER_CALL="${MAX_FRAMES_PER_CALL:-5}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-200}"
IMAGE_DETAIL="${IMAGE_DETAIL:-auto}"
FAILURE_POLICY="${FAILURE_POLICY:-unknown}"
RETRY_INVALID_CACHED="${RETRY_INVALID_CACHED:-0}"

export LLM_PROVIDER="${LLM_PROVIDER:-openai}"
export LLM_MODEL="${LLM_MODEL:-${MODEL}}"
export LLM_API_BASE_URL="${LLM_API_BASE_URL:-${BASE_URL}}"
export LLM_API_STYLE="${LLM_API_STYLE:-chat_completions}"
export LLM_TIMEOUT_S="${LLM_TIMEOUT_S:-90}"
export LLM_VISION_TIMEOUT_S="${LLM_VISION_TIMEOUT_S:-180}"
export LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-1}"
export LLM_TEMPERATURE="${LLM_TEMPERATURE:-0}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

typeset -A SOURCES
SOURCES=(
  sidewalk_main "/home/ubuntu/WorkSpace/dataset/sidewalk_phase1_all_chunks0_26_29_archive/manifests/train_all_chunks0_26_29_selfcontained_direction_choice_cot_hybrid_12way_full_choice.jsonl"
  sidewalk_extra "/home/ubuntu/WorkSpace/dataset/sidewalk_phase1_chunks26_29_nonstraight_slight_stop_vlm_manual_filtered_archive/manifests/train_chunks26_29_nonstraight_slight_stop_vlm_manual_filtered_direction_choice_cot_hybrid_12way_full_choice.jsonl"
  foxglove "/media/ubuntu/data2/SR-Nav Dataset/phase1_cot_teacher_frodobots_foxglove_lite_final/foxglove_yaw_p90_clean/manifests/phase1_cot_foxglove_yaw_p90_clean_lite_cot_final_direction_choice_cot_12way_full_choice.jsonl"
  egowalk "/media/ubuntu/data2/EgoWalk/prepared/fpv_only_multiview_ms_swift/egowalk_fpv_only_source.jsonl"
)

if [[ -n "${ORDER_OVERRIDE:-}" ]]; then
  ORDER=(${=ORDER_OVERRIDE})
else
  ORDER=(sidewalk_main sidewalk_extra foxglove egowalk)
fi

mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${RUN_ROOT}/run.log") 2>&1
trap 'status=$?; print "[behavior-interesting] ERROR line=${LINENO} status=${status} finished_at=$(date -Is)" >&2; exit ${status}' ERR
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

print "[behavior-interesting] started_at=$(date -Is)"
print "[behavior-interesting] run_root=${RUN_ROOT}"
print "[behavior-interesting] key_file=${KEY_FILE}"
print "[behavior-interesting] model=${LLM_MODEL}"
print "[behavior-interesting] base_url=${LLM_API_BASE_URL}"
print "[behavior-interesting] concurrency=${CONCURRENCY}"
print "[behavior-interesting] max_frames_per_call=${MAX_FRAMES_PER_CALL}"
print "[behavior-interesting] image_detail=${IMAGE_DETAIL}"
print "[behavior-interesting] keep_labels=1"
print "[behavior-interesting] retry_invalid_cached=${RETRY_INVALID_CACHED}"
print "[behavior-interesting] order=${ORDER[*]}"

for name in "${ORDER[@]}"; do
  source_jsonl="${SOURCES[$name]}"
  out_dir="${RUN_ROOT}/${name}"
  mkdir -p "${out_dir}"

  if [[ ! -f "${source_jsonl}" ]]; then
    print "[behavior-interesting] ERROR source missing name=${name} path=${source_jsonl}" >&2
    exit 2
  fi

  print ""
  print "[behavior-interesting] === ${name} start $(date -Is) ==="
  print "[behavior-interesting] source=${source_jsonl}"
  print "[behavior-interesting] source_rows=$(wc -l < "${source_jsonl}")"

  retry_args=()
  if [[ "${RETRY_INVALID_CACHED}" == "1" || "${RETRY_INVALID_CACHED:l}" == "true" ]]; then
    retry_args=(--retry-invalid-cached)
  fi

  python tools/model_processing/sidewalk_phase1_behavior_filter_vlm.py \
    --source-jsonl "${source_jsonl}" \
    --output-jsonl "${out_dir}/interesting.jsonl" \
    --rejected-jsonl "${out_dir}/rejected.jsonl" \
    --decisions-jsonl "${out_dir}/decisions.jsonl" \
    --report-json "${out_dir}/report.json" \
    --api-key-file "${KEY_FILE}" \
    --concurrency "${CONCURRENCY}" \
    --max-frames-per-call "${MAX_FRAMES_PER_CALL}" \
    --image-detail "${IMAGE_DETAIL}" \
    --keep-labels 1 \
    --failure-policy "${FAILURE_POLICY}" \
    --progress-interval "${PROGRESS_INTERVAL}" \
    "${retry_args[@]}"

  print "[behavior-interesting] === ${name} done $(date -Is) ==="
  print "[behavior-interesting] interesting_rows=$(wc -l < "${out_dir}/interesting.jsonl")"
done

combined_dir="${RUN_ROOT}/combined"
mkdir -p "${combined_dir}"
: > "${combined_dir}/interesting_all.jsonl"
for name in "${ORDER[@]}"; do
  cat "${RUN_ROOT}/${name}/interesting.jsonl" >> "${combined_dir}/interesting_all.jsonl"
done

print ""
print "[behavior-interesting] combined_interesting_rows=$(wc -l < "${combined_dir}/interesting_all.jsonl")"
print "[behavior-interesting] finished_at=$(date -Is)"
