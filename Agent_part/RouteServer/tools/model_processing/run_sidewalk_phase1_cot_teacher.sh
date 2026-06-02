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

cat <<EOF

[sidewalk-phase1-cot] done
cot_jsonl: ${OUTPUT_JSONL}
report_json: ${REPORT_JSON}

Next ms-swift step:
  python -m swift.sidewalk_nav.dataset_builder \\
    --input-jsonl "${OUTPUT_JSONL}" \\
    --output-jsonl "<ms-swift-output>/phase1_cot_train.jsonl" \\
    --input-format sidewalk \\
    --num-points "${NUM_POINTS}"
EOF
