#!/usr/bin/env zsh
set -euo pipefail

export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
if [[ ":${PATH}:" != *":/usr/bin:"* ]]; then
  export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH}"
fi

# Sequential full CoT teacher launch for the QC-clean FrodoBots and Foxglove
# Phase1 sources.  This intentionally keeps dataset order: label FrodoBots
# first, then Foxglove.  Within each dataset, the teacher can run multiple
# concurrent workers and rotates the loaded keys by worker.
#
# Default key lookup:
#   1. KEY_FILE if set
#   2. ${PROJECT_ROOT}/keys.txt
#   3. ${PROJECT_ROOT}/key.txt
#
SCRIPT_DIR="$(cd "$(dirname "${0}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${REPO_ROOT}/../.." && pwd)}"

RUN_ROOT="${RUN_ROOT:-/media/ubuntu/data2/SR-Nav Dataset/phase1_cot_teacher_frodobots_foxglove_lite}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/frodobots_foxglove_cot_sequential_$(date +%Y%m%d_%H%M%S).log}"
PID_FILE="${PID_FILE:-${RUN_ROOT}/frodobots_foxglove_cot_sequential.pid}"

FRODOBOTS_SOURCE_JSONL="${FRODOBOTS_SOURCE_JSONL:-/media/ubuntu/data2/SR-Nav Dataset/frodobots_phase1_merged_1000_gs10pct_calibK_multiframe_5f_5hz/manifests/phase1_source_frodobots_merged_1000_gs10pct_calibK_multiframe_5f_5hz_fpv_bev_pairs.jsonl}"
FOXGLOVE_SOURCE_JSONL="${FOXGLOVE_SOURCE_JSONL:-/media/ubuntu/Extreme SSD/foxglove/processed_batch_yaw_p90/phase1_labeling_full_multiframe_5f_5hz/manifests/foxglove_yaw_p90_full_multiframe_5f_5hz_osm_qc_repaired_anomaly_only_clean.jsonl}"

FRODOBOTS_SPLIT="${FRODOBOTS_SPLIT:-frodobots_qc_lite_cot}"
FOXGLOVE_SPLIT="${FOXGLOVE_SPLIT:-foxglove_yaw_p90_clean_lite_cot}"
FRODOBOTS_OUTPUT_ROOT="${FRODOBOTS_OUTPUT_ROOT:-${RUN_ROOT}/frodobots_qc}"
FOXGLOVE_OUTPUT_ROOT="${FOXGLOVE_OUTPUT_ROOT:-${RUN_ROOT}/foxglove_yaw_p90_clean}"

LLM_PROVIDER_NAME="${LLM_PROVIDER_NAME:-ark}"
LLM_MODEL_NAME="${LLM_MODEL:-${MODEL_NAME:-doubao-seed-2-0-lite-260428}}"
MAX_RETRIES="${MAX_RETRIES:-1}"
FALLBACK_POLICY="${FALLBACK_POLICY:-drop}"
IMAGE_DETAIL="${IMAGE_DETAIL:-auto}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-50}"
NUM_POINTS="${NUM_POINTS:-16}"
TRAJECTORY_TOLERANCE="${TRAJECTORY_TOLERANCE:-0.025}"
THETA_TOLERANCE_DEG="${THETA_TOLERANCE_DEG:-2.0}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/UrbanNav/bin/python}"
DETACH="${DETACH:-1}"
DRY_RUN="${DRY_RUN:-0}"
CONCURRENCY="${CONCURRENCY:-}"

# Archive is disabled by default because the legacy archive writer creates a
# single-FPV + single-BEV row.  Keep CoT source first; export multi-frame rows
# explicitly after labels are complete.
ARCHIVE_AFTER_COT="${ARCHIVE_AFTER_COT:-0}"
ARCHIVE_MEDIA_MODE="${ARCHIVE_MEDIA_MODE:-reference}"
ARCHIVE_PATH_MODE="${ARCHIVE_PATH_MODE:-absolute}"
ARCHIVE_REVIEW_LIMIT="${ARCHIVE_REVIEW_LIMIT:-200}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  print -u2 "python not executable: ${PYTHON_BIN}"
  print -u2 "Set PYTHON_BIN=/path/to/python if your UrbanNav environment is elsewhere."
  exit 1
fi

if [[ -z "${KEY_FILE:-}" ]]; then
  KEY_FILE="${PROJECT_ROOT}/keys.txt"
  if [[ ! -f "${KEY_FILE}" && -f "${PROJECT_ROOT}/key.txt" ]]; then
    KEY_FILE="${PROJECT_ROOT}/key.txt"
  fi
fi

if [[ ! -f "${KEY_FILE}" ]]; then
  print -u2 "KEY_FILE not found: ${KEY_FILE}"
  print -u2 "Create ${PROJECT_ROOT}/keys.txt or set KEY_FILE=/path/to/keys.txt."
  exit 1
fi

typeset -a keys
while IFS= read -r line; do
  key="${line#"${line%%[![:space:]]*}"}"
  key="${key%"${key##*[![:space:]]}"}"
  if [[ -z "${key}" || "${key}" == \#* ]]; then
    continue
  fi
  keys+=("${key}")
done < "${KEY_FILE}"

if (( ${#keys[@]} < 1 )); then
  print -u2 "no API keys loaded from KEY_FILE: ${KEY_FILE}"
  exit 1
fi

if [[ -z "${CONCURRENCY}" ]]; then
  CONCURRENCY="${#keys[@]}"
fi
if (( CONCURRENCY < 1 )); then
  CONCURRENCY=1
fi

pick_key() {
  local requested="$1"
  local count="${#keys[@]}"
  local idx="${requested}"
  if (( idx < 1 )); then
    idx=1
  fi
  if (( idx > count )); then
    idx=$(( ((idx - 1) % count) + 1 ))
  fi
  print -r -- "${keys[${idx}]}"
}

row_count() {
  local jsonl_path="$1"
  wc -l < "${jsonl_path}" | tr -d '[:space:]'
}

print_config() {
  print "[phase1-cot-sequential] repo_root=${REPO_ROOT}"
  print "[phase1-cot-sequential] run_root=${RUN_ROOT}"
  print "[phase1-cot-sequential] key_file=${KEY_FILE}"
  print "[phase1-cot-sequential] key_count=${#keys[@]}"
  print "[phase1-cot-sequential] concurrency=${CONCURRENCY}"
  print "[phase1-cot-sequential] model=${LLM_MODEL_NAME}"
  print "[phase1-cot-sequential] fallback_policy=${FALLBACK_POLICY}"
  print "[phase1-cot-sequential] max_retries=${MAX_RETRIES}"
  print "[phase1-cot-sequential] archive_after_cot=${ARCHIVE_AFTER_COT}"
  print "[phase1-cot-sequential] frodobots_source=${FRODOBOTS_SOURCE_JSONL}"
  print "[phase1-cot-sequential] foxglove_source=${FOXGLOVE_SOURCE_JSONL}"
}

run_teacher_one() {
  local name="$1"
  local source_jsonl="$2"
  local output_root="$3"
  local split="$4"

  if [[ ! -f "${source_jsonl}" ]]; then
    print -u2 "[phase1-cot-sequential] missing source for ${name}: ${source_jsonl}"
    exit 1
  fi

  local manifest_dir="${output_root}/manifests"
  local output_jsonl="${manifest_dir}/phase1_cot_${split}.jsonl"
  local report_json="${manifest_dir}/phase1_cot_report_${split}.json"
  local source_rows
  source_rows="$(row_count "${source_jsonl}")"

  mkdir -p "${manifest_dir}"

  print
  print "[phase1-cot-sequential] === ${name} start $(date '+%Y-%m-%d %H:%M:%S') ==="
  print "[phase1-cot-sequential] source_rows=${source_rows}"
  print "[phase1-cot-sequential] output_jsonl=${output_jsonl}"
  print "[phase1-cot-sequential] report_json=${report_json}"

  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
    print "[phase1-cot-sequential] dry-run: skip ${name}"
    return 0
  fi

  local first_key
  first_key="$(pick_key 1)"

  LLM_API_KEY= \
  LLM_PROVIDER="${LLM_PROVIDER_NAME}" \
  LLM_MODEL="${LLM_MODEL_NAME}" \
  ARK_API_KEY="${first_key}" \
  "${PYTHON_BIN}" "${REPO_ROOT}/tools/model_processing/sidewalk_phase1_cot_teacher.py" \
    --source-jsonl "${source_jsonl}" \
    --output-jsonl "${output_jsonl}" \
    --report-json "${report_json}" \
    --expected-points "${NUM_POINTS}" \
    --max-retries "${MAX_RETRIES}" \
    --trajectory-tolerance "${TRAJECTORY_TOLERANCE}" \
    --theta-tolerance-deg "${THETA_TOLERANCE_DEG}" \
    --image-detail "${IMAGE_DETAIL}" \
    --fallback-policy "${FALLBACK_POLICY}" \
    --progress-interval "${PROGRESS_INTERVAL}" \
    --api-key-file "${KEY_FILE}" \
    --concurrency "${CONCURRENCY}"

  if [[ "${ARCHIVE_AFTER_COT}" == "1" || "${ARCHIVE_AFTER_COT}" == "true" ]]; then
    local archive_root="${output_root}_archive"
    print "[phase1-cot-sequential] archiving ${name}: ${archive_root}"
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/model_processing/sidewalk_phase1_archive.py" \
      --cot-jsonl "${output_jsonl}" \
      --output-root "${archive_root}" \
      --split "${split}" \
      --media-mode "${ARCHIVE_MEDIA_MODE}" \
      --path-mode "${ARCHIVE_PATH_MODE}" \
      --expected-points "${NUM_POINTS}" \
      --trajectory-tolerance "${TRAJECTORY_TOLERANCE}" \
      --theta-tolerance-deg "${THETA_TOLERANCE_DEG}" \
      --write-review-html \
      --review-limit "${ARCHIVE_REVIEW_LIMIT}"
  fi

  print "[phase1-cot-sequential] === ${name} done $(date '+%Y-%m-%d %H:%M:%S') ==="
}

if [[ "${DETACH}" == "1" || "${DETACH}" == "true" ]]; then
  if [[ "${RUN_IN_WORKER:-0}" != "1" ]]; then
    mkdir -p "${LOG_DIR}" "$(dirname "${PID_FILE}")"
    print_config
    print "[phase1-cot-sequential] launching detached worker"
    print "[phase1-cot-sequential] log_file=${LOG_FILE}"
    RUN_IN_WORKER=1 DETACH=0 nohup "${0}" > "${LOG_FILE}" 2>&1 &
    pid=$!
    print "${pid}" > "${PID_FILE}"
    print "[phase1-cot-sequential] pid=${pid}"
    print "[phase1-cot-sequential] pid_file=${PID_FILE}"
    print "[phase1-cot-sequential] monitor:"
    print "  tail -f '${LOG_FILE}'"
    print "  ps -fp ${pid}"
    exit 0
  fi
fi

mkdir -p "${LOG_DIR}" "$(dirname "${PID_FILE}")"
print_config

run_teacher_one \
  "frodobots_qc" \
  "${FRODOBOTS_SOURCE_JSONL}" \
  "${FRODOBOTS_OUTPUT_ROOT}" \
  "${FRODOBOTS_SPLIT}"

run_teacher_one \
  "foxglove_yaw_p90_clean" \
  "${FOXGLOVE_SOURCE_JSONL}" \
  "${FOXGLOVE_OUTPUT_ROOT}" \
  "${FOXGLOVE_SPLIT}"

print
print "[phase1-cot-sequential] all done $(date '+%Y-%m-%d %H:%M:%S')"
print "[phase1-cot-sequential] outputs:"
print "  ${FRODOBOTS_OUTPUT_ROOT}/manifests/phase1_cot_${FRODOBOTS_SPLIT}.jsonl"
print "  ${FOXGLOVE_OUTPUT_ROOT}/manifests/phase1_cot_${FOXGLOVE_SPLIT}.jsonl"
