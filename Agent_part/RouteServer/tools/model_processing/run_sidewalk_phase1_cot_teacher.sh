#!/usr/bin/env bash
set -euo pipefail

# Generate validated single-shot CoT targets for Phase1 SideWalk SFT.
#
# This stage requires the Agent LLM/VLM provider to be configured, for example:
#   LLM_PROVIDER=ark ARK_API_KEY=... LLM_MODEL=...
#
# Example:
#   OUTPUT_ROOT="/media/ubuntu/data1/SR-Nav Dataset/sidewalk_phase1" \
#   ./tools/model_processing/run_sidewalk_phase1_cot_teacher.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OUTPUT_ROOT="${OUTPUT_ROOT:-/media/ubuntu/data1/SR-Nav Dataset/sidewalk_phase1}"
SPLIT_NAME="${SPLIT_NAME:-train}"
NUM_POINTS="${NUM_POINTS:-16}"
LIMIT="${LIMIT:-}"
SOURCE_JSONL="${SOURCE_JSONL:-${OUTPUT_ROOT}/manifests/phase1_source_${SPLIT_NAME}.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_ROOT}/manifests/phase1_cot_${SPLIT_NAME}.jsonl}"
REPORT_JSON="${REPORT_JSON:-${OUTPUT_ROOT}/manifests/phase1_cot_report_${SPLIT_NAME}.json}"
MAX_RETRIES="${MAX_RETRIES:-2}"
TRAJECTORY_TOLERANCE="${TRAJECTORY_TOLERANCE:-0.025}"
THETA_TOLERANCE_DEG="${THETA_TOLERANCE_DEG:-2.0}"
IMAGE_DETAIL="${IMAGE_DETAIL:-auto}"
FALLBACK_POLICY="${FALLBACK_POLICY:-drop}"
ARCHIVE_AFTER_COT="${ARCHIVE_AFTER_COT:-1}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-${OUTPUT_ROOT}_archive}"
ARCHIVE_SPLIT_NAME="${ARCHIVE_SPLIT_NAME:-${SPLIT_NAME}}"
MEDIA_MODE="${MEDIA_MODE:-hardlink}"
PATH_MODE="${PATH_MODE:-absolute}"
WRITE_REVIEW_HTML="${WRITE_REVIEW_HTML:-1}"
REVIEW_LIMIT="${REVIEW_LIMIT:-200}"
VALIDATE_IMAGES="${VALIDATE_IMAGES:-0}"

cmd=(
  python "${REPO_ROOT}/tools/model_processing/sidewalk_phase1_cot_teacher.py"
  --source-jsonl "${SOURCE_JSONL}"
  --output-jsonl "${OUTPUT_JSONL}"
  --report-json "${REPORT_JSON}"
  --expected-points "${NUM_POINTS}"
  --max-retries "${MAX_RETRIES}"
  --trajectory-tolerance "${TRAJECTORY_TOLERANCE}"
  --theta-tolerance-deg "${THETA_TOLERANCE_DEG}"
  --image-detail "${IMAGE_DETAIL}"
  --fallback-policy "${FALLBACK_POLICY}"
)

if [[ -n "${LIMIT}" ]]; then
  cmd+=(--limit "${LIMIT}")
fi

echo "[sidewalk-phase1-cot] generating validated teacher CoT"
printf '  %q' "${cmd[@]}"
echo
"${cmd[@]}"

if [[ "${ARCHIVE_AFTER_COT}" == "1" || "${ARCHIVE_AFTER_COT}" == "true" ]]; then
  archive_cmd=(
    python "${REPO_ROOT}/tools/model_processing/sidewalk_phase1_archive.py"
    --cot-jsonl "${OUTPUT_JSONL}"
    --output-root "${ARCHIVE_ROOT}"
    --split "${ARCHIVE_SPLIT_NAME}"
    --media-mode "${MEDIA_MODE}"
    --path-mode "${PATH_MODE}"
    --expected-points "${NUM_POINTS}"
    --trajectory-tolerance "${TRAJECTORY_TOLERANCE}"
    --theta-tolerance-deg "${THETA_TOLERANCE_DEG}"
    --review-limit "${REVIEW_LIMIT}"
  )
  if [[ -n "${LIMIT}" ]]; then
    archive_cmd+=(--limit "${LIMIT}")
  fi
  if [[ "${WRITE_REVIEW_HTML}" == "1" || "${WRITE_REVIEW_HTML}" == "true" ]]; then
    archive_cmd+=(--write-review-html)
  fi
  if [[ "${VALIDATE_IMAGES}" == "1" || "${VALIDATE_IMAGES}" == "true" ]]; then
    archive_cmd+=(--validate-images)
  fi

  echo "[sidewalk-phase1-cot] archiving validated CoT for ms-swift"
  printf '  %q' "${archive_cmd[@]}"
  echo
  "${archive_cmd[@]}"
fi

cat <<EOF

[sidewalk-phase1-cot] done
cot_jsonl: ${OUTPUT_JSONL}
report_json: ${REPORT_JSON}
archive_root: ${ARCHIVE_ROOT}
train_jsonl: ${ARCHIVE_ROOT}/manifests/${ARCHIVE_SPLIT_NAME}.jsonl

Set ARCHIVE_AFTER_COT=0 to skip automatic archive generation.
EOF
