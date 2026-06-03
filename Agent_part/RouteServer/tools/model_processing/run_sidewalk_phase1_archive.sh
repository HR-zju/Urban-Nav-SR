#!/usr/bin/env bash
set -euo pipefail

# Validate Phase1 CoT rows and archive accepted samples as a ms-swift-ready JSONL.
#
# Typical local training archive:
#   OUTPUT_ROOT="/media/ubuntu/data1/SR-Nav Dataset/sidewalk_phase1" \
#   ARCHIVE_ROOT="/media/ubuntu/data1/SR-Nav Dataset/sidewalk_phase1_archive" \
#   MEDIA_MODE=hardlink PATH_MODE=absolute \
#   ./tools/model_processing/run_sidewalk_phase1_archive.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OUTPUT_ROOT="${OUTPUT_ROOT:-/media/ubuntu/data1/SR-Nav Dataset/sidewalk_phase1}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-${OUTPUT_ROOT}_archive}"
SPLIT_NAME="${SPLIT_NAME:-train}"
COT_JSONL="${COT_JSONL:-${OUTPUT_ROOT}/manifests/phase1_cot_${SPLIT_NAME}.jsonl}"
MEDIA_MODE="${MEDIA_MODE:-hardlink}"
PATH_MODE="${PATH_MODE:-absolute}"
NUM_POINTS="${NUM_POINTS:-16}"
LIMIT="${LIMIT:-}"
TRAJECTORY_TOLERANCE="${TRAJECTORY_TOLERANCE:-0.025}"
THETA_TOLERANCE_DEG="${THETA_TOLERANCE_DEG:-2.0}"
WRITE_REVIEW_HTML="${WRITE_REVIEW_HTML:-0}"
REVIEW_LIMIT="${REVIEW_LIMIT:-200}"
VALIDATE_IMAGES="${VALIDATE_IMAGES:-0}"

cmd=(
  python "${REPO_ROOT}/tools/model_processing/sidewalk_phase1_archive.py"
  --cot-jsonl "${COT_JSONL}"
  --output-root "${ARCHIVE_ROOT}"
  --split "${SPLIT_NAME}"
  --media-mode "${MEDIA_MODE}"
  --path-mode "${PATH_MODE}"
  --expected-points "${NUM_POINTS}"
  --trajectory-tolerance "${TRAJECTORY_TOLERANCE}"
  --theta-tolerance-deg "${THETA_TOLERANCE_DEG}"
  --review-limit "${REVIEW_LIMIT}"
)

if [[ -n "${LIMIT}" ]]; then
  cmd+=(--limit "${LIMIT}")
fi
if [[ "${WRITE_REVIEW_HTML}" == "1" || "${WRITE_REVIEW_HTML}" == "true" ]]; then
  cmd+=(--write-review-html)
fi
if [[ "${VALIDATE_IMAGES}" == "1" || "${VALIDATE_IMAGES}" == "true" ]]; then
  cmd+=(--validate-images)
fi

echo "[sidewalk-phase1-archive] validating and archiving CoT data"
printf '  %q' "${cmd[@]}"
echo
"${cmd[@]}"

cat <<EOF

[sidewalk-phase1-archive] done
archive_root: ${ARCHIVE_ROOT}
train_jsonl: ${ARCHIVE_ROOT}/manifests/${SPLIT_NAME}.jsonl
review_csv: ${ARCHIVE_ROOT}/manifests/${SPLIT_NAME}.review.csv
stats_json: ${ARCHIVE_ROOT}/manifests/${SPLIT_NAME}.stats.json

ms-swift training input:
  ${ARCHIVE_ROOT}/manifests/${SPLIT_NAME}.jsonl
EOF
