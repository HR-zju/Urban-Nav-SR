#!/usr/bin/env bash
set -euo pipefail

# Build one FrodoBots clip from the curated longest subset into Phase1 CoT format.
#
# Defaults:
#   - source subset: /media/ubuntu/data2/SR-Nav Dataset/AD Dataset_longest20_of_top10
#   - clip: longest clip in batch_selection_top10_longest20.json
#   - sampling: 0.5fps for BEV and Phase1 source rows
#   - teacher fallback: template, so smoke runs without an LLM key
#
# For production teacher labels, set:
#   LLM_PROVIDER=ark ARK_API_KEY=... LLM_MODEL=... FALLBACK_POLICY=drop

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROUTESERVER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
URBANNAV_ROOT="$(cd "${ROUTESERVER_ROOT}/../.." && pwd)"

SUBSET_ROOT="${SUBSET_ROOT:-/media/ubuntu/data2/SR-Nav Dataset/AD Dataset_longest20_of_top10}"
SUBSET_REPORT="${SUBSET_REPORT:-${SUBSET_ROOT}/batch_selection_top10_longest20.json}"
CLIP_RANK="${CLIP_RANK:-0}"
TARGET_SAMPLE_FPS="${TARGET_SAMPLE_FPS:-0.5}"
BEV_SAMPLE_FPS="${BEV_SAMPLE_FPS:-${TARGET_SAMPLE_FPS}}"
MAP_PROVIDER="${MAP_PROVIDER:-osm}"
BEV_MAP_TILE_OFFLINE="${BEV_MAP_TILE_OFFLINE:-1}"
USE_POSE_EKF="${USE_POSE_EKF:-1}"
WRITE_OVERLAYS="${WRITE_OVERLAYS:-1}"
NUM_POINTS="${NUM_POINTS:-16}"
FUTURE_HORIZON_SEC="${FUTURE_HORIZON_SEC:-2.5}"
FALLBACK_POLICY="${FALLBACK_POLICY:-template}"
ARCHIVE_AFTER_COT="${ARCHIVE_AFTER_COT:-1}"
MEDIA_MODE="${MEDIA_MODE:-hardlink}"
PATH_MODE="${PATH_MODE:-absolute}"
VALIDATE_IMAGES="${VALIDATE_IMAGES:-0}"
LIMIT="${LIMIT:-}"

if [[ -z "${CLIP_DIR:-}" ]]; then
  CLIP_DIR="$(
    python - "${SUBSET_REPORT}" "${CLIP_RANK}" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1]).expanduser()
rank = int(sys.argv[2])
with report_path.open() as f:
    report = json.load(f)
selected = report.get("selected") or []
if not selected:
    raise SystemExit(f"No selected clips in {report_path}")
if rank < 0 or rank >= len(selected):
    raise SystemExit(f"CLIP_RANK {rank} out of range for {len(selected)} clips")
print(selected[rank]["clip_dir"])
PY
  )"
fi

CLIP_DIR="$(python - "${CLIP_DIR}" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
CLIP_ID="$(basename "${CLIP_DIR}")"
RUN_NAME="${RUN_NAME:-${CLIP_ID}_phase1_cot_0p5fps}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${URBANNAV_ROOT}/data/frodobots_phase1_cot_smoke/${RUN_NAME}/phase1}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-${URBANNAV_ROOT}/data/frodobots_phase1_cot_smoke/${RUN_NAME}/phase1_archive}"
SPLIT_NAME="${SPLIT_NAME:-${CLIP_ID}_smoke}"

if [[ -z "${FRAME_RATE_HZ:-}" || -z "${FRAME_STEP:-}" ]]; then
  read -r AUTO_FRAME_RATE_HZ AUTO_FRAME_STEP < <(
    python - "${CLIP_DIR}" "${TARGET_SAMPLE_FPS}" <<'PY'
import sys
from pathlib import Path
import numpy as np

clip_dir = Path(sys.argv[1])
target_fps = float(sys.argv[2])
times_path = clip_dir / "frame_times.npy"
fps = 10.0
if times_path.is_file():
    times = np.load(times_path)
    if len(times) > 1:
        diffs = np.diff(times.astype("float64")) / 1e9
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if len(diffs):
            fps = 1.0 / float(np.median(diffs))
step = max(1, int(round(fps / max(target_fps, 1e-6))))
print(f"{fps:.6f} {step}")
PY
  )
  FRAME_RATE_HZ="${FRAME_RATE_HZ:-${AUTO_FRAME_RATE_HZ}}"
  FRAME_STEP="${FRAME_STEP:-${AUTO_FRAME_STEP}}"
fi

echo "[frodobots-phase1-cot-smoke] config"
echo "  clip_dir: ${CLIP_DIR}"
echo "  output_root: ${OUTPUT_ROOT}"
echo "  archive_root: ${ARCHIVE_ROOT}"
echo "  split_name: ${SPLIT_NAME}"
echo "  target_sample_fps: ${TARGET_SAMPLE_FPS}"
echo "  estimated_frame_rate_hz: ${FRAME_RATE_HZ}"
echo "  frame_step: ${FRAME_STEP}"
echo "  bev_sample_fps: ${BEV_SAMPLE_FPS}"
echo "  map_provider: ${MAP_PROVIDER}"
echo "  bev_map_tile_offline: ${BEV_MAP_TILE_OFFLINE}"
echo "  fallback_policy: ${FALLBACK_POLICY}"

cd "${ROUTESERVER_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${ARCHIVE_ROOT}"

process_cmd=(
  python tools/dataset_processing/process_sidewalk_dataset.py
  --sidewalk-root "${CLIP_DIR}"
  --bev-sample-fps "${BEV_SAMPLE_FPS}"
  --map-provider "${MAP_PROVIDER}"
)
if [[ "${USE_POSE_EKF}" == "1" || "${USE_POSE_EKF}" == "true" ]]; then
  process_cmd+=(--use-pose-ekf)
fi

echo "[frodobots-phase1-cot-smoke] generating trajectory + BEV"
BEV_MAP_TILE_OFFLINE="${BEV_MAP_TILE_OFFLINE}" "${process_cmd[@]}"

echo "[frodobots-phase1-cot-smoke] building Phase1 source"
phase1_env=(
  SIDEWALK_ROOT="${CLIP_DIR}"
  OUTPUT_ROOT="${OUTPUT_ROOT}"
  SPLIT_NAME="${SPLIT_NAME}"
  FRAME_STEP="${FRAME_STEP}"
  FRAME_RATE_HZ="${FRAME_RATE_HZ}"
  FUTURE_HORIZON_SEC="${FUTURE_HORIZON_SEC}"
  NUM_POINTS="${NUM_POINTS}"
  WRITE_OVERLAYS="${WRITE_OVERLAYS}"
)
if [[ -n "${LIMIT}" ]]; then
  phase1_env+=(LIMIT="${LIMIT}")
fi
env "${phase1_env[@]}" ./tools/model_processing/run_sidewalk_phase1_dataset.sh

echo "[frodobots-phase1-cot-smoke] generating CoT + archive"
cot_env=(
  OUTPUT_ROOT="${OUTPUT_ROOT}"
  SPLIT_NAME="${SPLIT_NAME}"
  NUM_POINTS="${NUM_POINTS}"
  FALLBACK_POLICY="${FALLBACK_POLICY}"
  ARCHIVE_AFTER_COT="${ARCHIVE_AFTER_COT}"
  ARCHIVE_ROOT="${ARCHIVE_ROOT}"
  ARCHIVE_SPLIT_NAME="${SPLIT_NAME}"
  MEDIA_MODE="${MEDIA_MODE}"
  PATH_MODE="${PATH_MODE}"
  VALIDATE_IMAGES="${VALIDATE_IMAGES}"
)
if [[ -n "${LIMIT}" ]]; then
  cot_env+=(LIMIT="${LIMIT}")
fi
env "${cot_env[@]}" ./tools/model_processing/run_sidewalk_phase1_cot_teacher.sh

cat <<EOF

[frodobots-phase1-cot-smoke] done
source_jsonl: ${OUTPUT_ROOT}/manifests/phase1_source_${SPLIT_NAME}.jsonl
cot_jsonl: ${OUTPUT_ROOT}/manifests/phase1_cot_${SPLIT_NAME}.jsonl
archive_jsonl: ${ARCHIVE_ROOT}/manifests/${SPLIT_NAME}.jsonl
review_csv: ${ARCHIVE_ROOT}/manifests/${SPLIT_NAME}.review.csv
EOF
