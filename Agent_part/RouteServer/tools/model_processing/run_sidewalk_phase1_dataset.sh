#!/usr/bin/env bash
set -euo pipefail

# Build Phase1 SideWalk System2 data artifacts without requiring SAM/SoM.
#
# Example:
#   SIDEWALK_ROOT="/media/ubuntu/data1/SR-Nav Dataset/sidewalk/chunk_0" \
#   OUTPUT_ROOT="/media/ubuntu/data1/SR-Nav Dataset/sidewalk_phase1" \
#   ./tools/model_processing/run_sidewalk_phase1_dataset.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SIDEWALK_ROOT="${SIDEWALK_ROOT:-/media/ubuntu/data1/SR-Nav Dataset/sidewalk/chunk_0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/ubuntu/data1/SR-Nav Dataset/sidewalk_phase1}"
SPLIT_NAME="${SPLIT_NAME:-train}"
FRAME_STEP="${FRAME_STEP:-10}"
FUTURE_FRAMES="${FUTURE_FRAMES:-50}"
FUTURE_HORIZON_SEC="${FUTURE_HORIZON_SEC:-2.5}"
NUM_POINTS="${NUM_POINTS:-16}"
LIMIT="${LIMIT:-}"
MAX_CLIPS="${MAX_CLIPS:-}"
WRITE_OVERLAYS="${WRITE_OVERLAYS:-0}"
ALLOW_MISSING_BEV="${ALLOW_MISSING_BEV:-0}"
ALLOW_HEURISTIC_TRAJECTORY="${ALLOW_HEURISTIC_TRAJECTORY:-0}"
MIN_FUTURE_DISTANCE_M="${MIN_FUTURE_DISTANCE_M:-0.3}"
TRAJECTORY_MIN_Y_STEP="${TRAJECTORY_MIN_Y_STEP:-0.005}"
MAX_BEV_TIME_DELTA_MS="${MAX_BEV_TIME_DELTA_MS:-300.0}"
FRAME_RATE_HZ="${FRAME_RATE_HZ:-20.0}"

cmd=(
  python "${REPO_ROOT}/tools/model_processing/sidewalk_phase1_dataset.py"
  --sidewalk-root "${SIDEWALK_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --split-name "${SPLIT_NAME}"
  --frame-step "${FRAME_STEP}"
  --future-frames "${FUTURE_FRAMES}"
  --future-horizon-sec "${FUTURE_HORIZON_SEC}"
  --num-points "${NUM_POINTS}"
  --min-future-distance-m "${MIN_FUTURE_DISTANCE_M}"
  --trajectory-min-y-step "${TRAJECTORY_MIN_Y_STEP}"
  --max-bev-time-delta-ms "${MAX_BEV_TIME_DELTA_MS}"
  --frame-rate-hz "${FRAME_RATE_HZ}"
)

if [[ -n "${LIMIT}" ]]; then
  cmd+=(--limit "${LIMIT}")
fi
if [[ -n "${MAX_CLIPS}" ]]; then
  cmd+=(--max-clips "${MAX_CLIPS}")
fi
if [[ "${WRITE_OVERLAYS}" == "1" || "${WRITE_OVERLAYS}" == "true" ]]; then
  cmd+=(--write-overlays)
fi
if [[ "${ALLOW_MISSING_BEV}" == "1" || "${ALLOW_MISSING_BEV}" == "true" ]]; then
  cmd+=(--allow-missing-bev)
fi
if [[ "${ALLOW_HEURISTIC_TRAJECTORY}" == "1" || "${ALLOW_HEURISTIC_TRAJECTORY}" == "true" ]]; then
  cmd+=(--allow-heuristic-trajectory)
fi

echo "[sidewalk-phase1] building dataset"
printf '  %q' "${cmd[@]}"
echo
"${cmd[@]}"

SOURCE_JSONL="${OUTPUT_ROOT}/manifests/phase1_source_${SPLIT_NAME}.jsonl"
SUMMARY_JSON="${OUTPUT_ROOT}/manifests/phase1_eval_${SPLIT_NAME}.json"

echo "[sidewalk-phase1] evaluating artifacts"
python "${REPO_ROOT}/tools/model_processing/eval_sidewalk_phase1_artifacts.py" \
  --source-jsonl "${SOURCE_JSONL}" \
  --qa-root "${OUTPUT_ROOT}/clips" \
  --summary-json "${SUMMARY_JSON}" \
  --expected-points "${NUM_POINTS}"

cat <<EOF

[sidewalk-phase1] done
source_jsonl: ${SOURCE_JSONL}
summary_json: ${SUMMARY_JSON}

Next ms-swift step:
  python -m swift.sidewalk_nav.dataset_builder \\
    --input-jsonl "${SOURCE_JSONL}" \\
    --output-jsonl "<ms-swift-output>/phase1_train.jsonl" \\
    --input-format sidewalk \\
    --num-points "${NUM_POINTS}"
EOF
