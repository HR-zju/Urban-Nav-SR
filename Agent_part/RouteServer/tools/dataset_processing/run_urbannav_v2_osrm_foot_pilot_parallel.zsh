#!/usr/bin/env zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PY="${URBANNAV_PYTHON:-/home/ubuntu/anaconda3/envs/UrbanNav/bin/python}"
RUN_ROOT="${URBANNAV_OSRM_FOOT_PILOT_OUT:-data/benchmark/urbannav_v2_osm_osrm_foot_pilot_parallel}"
CONFIG="${URBANNAV_PILOT_CONFIG:-data/benchmark/urbannav_instruction_set_full_en_v2/collection_config_full_single_stop.json}"
INSTRUCTION_SET="${URBANNAV_INSTRUCTION_SET:-data/benchmark/urbannav_instruction_set_full_en_v2/instruction_set.jsonl}"
TARGET_MAP="${URBANNAV_TARGET_MAP:-config/urbannav_target_type_provider_map_mined_v2.json}"
ROUTER_CONFIG="${URBANNAV_ROUTER_CONFIG:-config/urbannav_osm_online_osrm_foot_sources.json}"
POI_INDEX_ROOT="${URBANNAV_POI_INDEX_ROOT:-data/osm/index}"

SINGLE_SHARDS="${URBANNAV_SINGLE_SHARDS:-4}"
COMPLEX_SHARDS="${URBANNAV_COMPLEX_SHARDS:-2}"
SINGLE_TARGET_PER_SHARD="${URBANNAV_SINGLE_TARGET_PER_SHARD:-8}"
MULTISTOP_TARGET_PER_SHARD="${URBANNAV_MULTISTOP_TARGET_PER_SHARD:-2}"
UPDATE_TARGET_PER_SHARD="${URBANNAV_UPDATE_TARGET_PER_SHARD:-1}"
RESUME="${URBANNAV_RESUME:-0}"
OVERWRITE="${URBANNAV_OVERWRITE:-1}"

COLLECT_MODE_ARGS=()
if [[ "$RESUME" == "1" || "$RESUME" == "true" ]]; then
  COLLECT_MODE_ARGS+=(--resume)
elif [[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" ]]; then
  COLLECT_MODE_ARGS+=(--overwrite)
fi

if [[ ! -x "$PY" ]]; then
  echo "UrbanNav python not found or not executable: $PY" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT/shards" "$RUN_ROOT/logs"

echo "[1/3] split pilot instruction sets"
PYTHONPATH=src "$PY" - "$INSTRUCTION_SET" "$RUN_ROOT/shards" "$SINGLE_SHARDS" "$COMPLEX_SHARDS" <<'PY'
import json
import sys
from pathlib import Path

instruction_set = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
single_shards = int(sys.argv[3])
complex_shards = int(sys.argv[4])
out_dir.mkdir(parents=True, exist_ok=True)

def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def write_shards(rows, prefix, count):
    buckets = [[] for _ in range(count)]
    for index, row in enumerate(rows):
        buckets[index % count].append(row)
    for shard_id, bucket in enumerate(buckets):
        path = out_dir / f"{prefix}_shard_{shard_id:02d}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in bucket), encoding="utf-8")

rows = load(instruction_set)
single_rows = [
    row for row in rows
    if row.get("category") in {"poi_navigation", "implicit_need_navigation", "public_poi_navigation"}
]
complex_rows = [
    row for row in rows
    if row.get("category") in {"multi_stop_route", "route_update_add_via"} or str(row.get("category") or "").startswith("route_update")
]
write_shards(single_rows, "single", single_shards)
write_shards(complex_rows, "complex", complex_shards)
print(json.dumps({
    "instruction_set": str(instruction_set),
    "single_rows": len(single_rows),
    "complex_rows": len(complex_rows),
    "single_shards": single_shards,
    "complex_shards": complex_shards,
}, ensure_ascii=False))
PY

echo "[2/3] launch single-stop shards"
for shard in $(seq 0 $((SINGLE_SHARDS - 1))); do
  shard_id="$(printf '%02d' "$shard")"
  out="$RUN_ROOT/single_shard_$shard_id"
  log="$RUN_ROOT/logs/single_shard_$shard_id.log"
  (
    PYTHONPATH=src "$PY" tools/dataset_processing/collect_urbannav_global_gt.py \
      --output-root "$out" \
      --config "$CONFIG" \
      --instruction-set "$RUN_ROOT/shards/single_shard_$shard_id.jsonl" \
      --target-cases "$SINGLE_TARGET_PER_SHARD" \
      --provider osm_local \
      --poi-index-root "$POI_INDEX_ROOT" \
      --local-router-config "$ROUTER_CONFIG" \
      --origins-per-region 16 \
      --candidate-limit 5 \
      --search-radius-m 1500 \
      --acceptable-route-ratio 1.15 \
      --case-concurrency 1 \
      --route-concurrency 1 \
      --retry 1 \
      --throttle-s 0.2 \
      "${COLLECT_MODE_ARGS[@]}"
  ) > "$log" 2>&1 &
  echo "single shard $shard_id pid=$! log=$log"
done

echo "[3/3] launch complex shards"
for shard in $(seq 0 $((COMPLEX_SHARDS - 1))); do
  shard_id="$(printf '%02d' "$shard")"
  out="$RUN_ROOT/complex_shard_$shard_id"
  log="$RUN_ROOT/logs/complex_shard_$shard_id.log"
  (
    PYTHONPATH=src "$PY" tools/dataset_processing/collect_urbannav_complex_gt.py \
      --output-root "$out" \
      --config "$CONFIG" \
      --instruction-set "$RUN_ROOT/shards/complex_shard_$shard_id.jsonl" \
      --target-type-map "$TARGET_MAP" \
      --target-multistop "$MULTISTOP_TARGET_PER_SHARD" \
      --target-route-update "$UPDATE_TARGET_PER_SHARD" \
      --provider osm_local \
      --poi-index-root "$POI_INDEX_ROOT" \
      --local-router-config "$ROUTER_CONFIG" \
      --origins-per-region 16 \
      --candidate-limit 5 \
      --search-radius-m 1500 \
      --acceptable-route-ratio 1.15 \
      --case-concurrency 1 \
      --route-concurrency 1 \
      --retry 1 \
      --throttle-s 0.2 \
      "${COLLECT_MODE_ARGS[@]}"
  ) > "$log" 2>&1 &
  echo "complex shard $shard_id pid=$! log=$log"
done

wait
echo "done"
