#!/usr/bin/env zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

PY="${URBANNAV_PYTHON:-/home/ubuntu/anaconda3/envs/UrbanNav/bin/python}"
MODE="${1:-prepare}"
POOL_PLAN="${URBANNAV_POOL_PLAN:-config/urbannav_25k_v2_collection_pool_plan.json}"
SUPPLEMENT_PLAN="${URBANNAV_SUPPLEMENT_PLAN:-config/urbannav_25k_v2_supplement_pool_plan.json}"
FINAL_PLAN="${URBANNAV_FINAL_PLAN:-config/urbannav_25k_v2_distribution_plan.json}"
QUALITY_POLICY="${URBANNAV_QUALITY_POLICY:-config/urbannav_25k_v2_quality_policy.json}"
REGIONS_CONFIG="${URBANNAV_REGIONS_CONFIG:-config/urbannav_global_gt_collection_full_en.json}"
ROUTER_CONFIG="${URBANNAV_ROUTER_CONFIG:-config/urbannav_osm_local_sources_full.json}"
POI_INDEX_ROOT="${URBANNAV_POI_INDEX_ROOT:-data/osm/index}"
POOL_ROOT="${URBANNAV_POOL_ROOT:-data/benchmark/urbannav_instruction_pool_final25k_v2_40k}"
SUPPLEMENT_POOL_ROOT="${URBANNAV_SUPPLEMENT_POOL_ROOT:-data/benchmark/urbannav_instruction_pool_final25k_v2_supp1_4k}"
COLLECTION_ROOT="${URBANNAV_COLLECTION_ROOT:-data/benchmark/urbannav_v2_osm_local_osrm_foot_25k_v2_collection_v2}"
SUPPLEMENT_COLLECTION_ROOT="${URBANNAV_SUPPLEMENT_COLLECTION_ROOT:-data/benchmark/urbannav_v2_osm_local_osrm_foot_25k_v2_supp1_collection}"
FINAL_ROOT="${URBANNAV_FINAL_ROOT:-data/benchmark/urbannav_v2_osm_local_osrm_foot_final25k_v2}"
LEGACY_FINAL15K="${URBANNAV_LEGACY_FINAL15K:-data/benchmark/urbannav_v2_osm_local_osrm_foot_final_15k}"
SINGLE_SHARDS="${URBANNAV_SINGLE_SHARDS:-8}"
COMPLEX_SHARDS="${URBANNAV_COMPLEX_SHARDS:-4}"
CASE_CONCURRENCY="${URBANNAV_CASE_CONCURRENCY:-4}"
COMPLEX_CASE_CONCURRENCY="${URBANNAV_COMPLEX_CASE_CONCURRENCY:-2}"
ROUTE_CONCURRENCY="${URBANNAV_ROUTE_CONCURRENCY:-8}"

if [[ ! -x "$PY" ]]; then
  print -u2 "UrbanNav Python is not executable: $PY"
  exit 1
fi

prepare_pool() {
  PYTHONPATH=src "$PY" tools/dataset_processing/build_urbannav_instruction_set.py \
    --distribution "$POOL_PLAN" \
    --quality-policy "$QUALITY_POLICY" \
    --city-plan-mode exact \
    --output-root "$POOL_ROOT" \
    --instruction-mode case_variant \
    --seed 20260715 \
    --overwrite
  PYTHONPATH=src "$PY" tools/dataset_processing/split_urbannav_instruction_pool.py \
    --instruction-set "$POOL_ROOT/instruction_set.jsonl" \
    --output-root "$POOL_ROOT/shards" \
    --single-shards "$SINGLE_SHARDS" \
    --complex-shards "$COMPLEX_SHARDS"
}

prepare_supplement_pool() {
  PYTHONPATH=src "$PY" tools/dataset_processing/build_urbannav_instruction_set.py \
    --distribution "$SUPPLEMENT_PLAN" \
    --quality-policy "$QUALITY_POLICY" \
    --city-plan-mode exact \
    --output-root "$SUPPLEMENT_POOL_ROOT" \
    --instruction-mode case_variant \
    --seed 20260716 \
    --overwrite
  PYTHONPATH=src "$PY" tools/dataset_processing/split_urbannav_instruction_pool.py \
    --instruction-set "$SUPPLEMENT_POOL_ROOT/instruction_set.jsonl" \
    --output-root "$SUPPLEMENT_POOL_ROOT/shards" \
    --single-shards 1 \
    --complex-shards 2
}

collection_mode_args() {
  local output_root="$1"
  if [[ -s "$output_root/task_gt.jsonl" || -s "$output_root/task_rejects.jsonl" ]]; then
    print -- --resume
  else
    print -- --overwrite
  fi
}

run_collection() {
  mkdir -p "$COLLECTION_ROOT/logs"
  local -a pids
  for shard in $(seq 0 $((SINGLE_SHARDS - 1))); do
    local shard_id="$(printf '%02d' "$shard")"
    local output="$COLLECTION_ROOT/single_shard_$shard_id"
    local mode_arg="$(collection_mode_args "$output")"
    (
      PYTHONPATH=src "$PY" tools/dataset_processing/collect_urbannav_global_gt.py \
        --output-root "$output" \
        --config "$POOL_ROOT/collection_config_full_single_stop.json" \
        --instruction-set "$POOL_ROOT/shards/single_shard_$shard_id.jsonl" \
        --target-cases 3750 \
        --provider osm_local \
        --poi-index-root "$POI_INDEX_ROOT" \
        --local-router-config "$ROUTER_CONFIG" \
        --origins-per-region 96 \
        --candidate-limit 5 \
        --search-radius-m 2000 \
        --case-concurrency "$CASE_CONCURRENCY" \
        --route-concurrency "$ROUTE_CONCURRENCY" \
        --retry 2 \
        --throttle-s 0 \
        "$mode_arg"
    ) > "$COLLECTION_ROOT/logs/single_shard_$shard_id.log" 2>&1 &
    pids+=("$!")
  done
  for shard in $(seq 0 $((COMPLEX_SHARDS - 1))); do
    local shard_id="$(printf '%02d' "$shard")"
    local output="$COLLECTION_ROOT/complex_shard_$shard_id"
    local mode_arg="$(collection_mode_args "$output")"
    (
      PYTHONPATH=src "$PY" tools/dataset_processing/collect_urbannav_complex_gt.py \
        --output-root "$output" \
        --config "$POOL_ROOT/collection_config_full_single_stop.json" \
        --instruction-set "$POOL_ROOT/shards/complex_shard_$shard_id.jsonl" \
        --target-type-map config/urbannav_target_type_provider_map_mined_v2.json \
        --quality-policy "$QUALITY_POLICY" \
        --multi-stop-cache-policy full_route_matrix \
        --target-multistop 1500 \
        --target-route-update 1000 \
        --provider osm_local \
        --poi-index-root "$POI_INDEX_ROOT" \
        --local-router-config "$ROUTER_CONFIG" \
        --origins-per-region 96 \
        --candidate-limit 5 \
        --search-radius-m 2000 \
        --case-concurrency "$COMPLEX_CASE_CONCURRENCY" \
        --route-concurrency "$ROUTE_CONCURRENCY" \
        --retry 2 \
        --throttle-s 0 \
        "$mode_arg"
    ) > "$COLLECTION_ROOT/logs/complex_shard_$shard_id.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  return "$failed"
}

run_supplement_collection() {
  mkdir -p "$SUPPLEMENT_COLLECTION_ROOT/logs"
  local -a pids
  for shard in 0 1; do
    local shard_id="$(printf '%02d' "$shard")"
    local output="$SUPPLEMENT_COLLECTION_ROOT/complex_shard_$shard_id"
    local mode_arg="$(collection_mode_args "$output")"
    (
      PYTHONPATH=src "$PY" tools/dataset_processing/collect_urbannav_complex_gt.py \
        --output-root "$output" \
        --config "$POOL_ROOT/collection_config_full_single_stop.json" \
        --instruction-set "$SUPPLEMENT_POOL_ROOT/shards/complex_shard_$shard_id.jsonl" \
        --target-type-map config/urbannav_target_type_provider_map_mined_v2.json \
        --quality-policy "$QUALITY_POLICY" \
        --multi-stop-cache-policy full_route_matrix \
        --target-multistop 1250 \
        --target-route-update 750 \
        --provider osm_local \
        --poi-index-root "$POI_INDEX_ROOT" \
        --local-router-config "$ROUTER_CONFIG" \
        --origins-per-region 96 \
        --candidate-limit 5 \
        --search-radius-m 2000 \
        --case-concurrency "$COMPLEX_CASE_CONCURRENCY" \
        --route-concurrency "$ROUTE_CONCURRENCY" \
        --retry 2 \
        --throttle-s 0 \
        "$mode_arg"
    ) > "$SUPPLEMENT_COLLECTION_ROOT/logs/complex_shard_$shard_id.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  return "$failed"
}

run_smoke() {
  local smoke_root="${COLLECTION_ROOT}_smoke"
  PYTHONPATH=src "$PY" tools/dataset_processing/collect_urbannav_global_gt.py \
    --output-root "$smoke_root/single" \
    --config "$POOL_ROOT/collection_config_full_single_stop.json" \
    --instruction-set "$POOL_ROOT/shards/single_shard_00.jsonl" \
    --target-cases 4 \
    --max-attempts 20 \
    --provider osm_local \
    --poi-index-root "$POI_INDEX_ROOT" \
    --local-router-config "$ROUTER_CONFIG" \
    --origins-per-region 8 \
    --candidate-limit 5 \
    --search-radius-m 2000 \
    --case-concurrency 2 \
    --route-concurrency 4 \
    --retry 1 \
    --overwrite
  PYTHONPATH=src "$PY" tools/dataset_processing/collect_urbannav_complex_gt.py \
    --output-root "$smoke_root/complex" \
    --config "$POOL_ROOT/collection_config_full_single_stop.json" \
    --instruction-set "$POOL_ROOT/shards/complex_shard_00.jsonl" \
    --target-type-map config/urbannav_target_type_provider_map_mined_v2.json \
    --quality-policy "$QUALITY_POLICY" \
    --multi-stop-cache-policy full_route_matrix \
    --target-multistop 2 \
    --target-route-update 2 \
    --max-attempts 30 \
    --provider osm_local \
    --poi-index-root "$POI_INDEX_ROOT" \
    --local-router-config "$ROUTER_CONFIG" \
    --origins-per-region 8 \
    --candidate-limit 5 \
    --search-radius-m 2000 \
    --case-concurrency 1 \
    --route-concurrency 4 \
    --retry 1 \
    --overwrite
}

finalize_package() {
  local -a supplement_args
  if [[ -d "$SUPPLEMENT_COLLECTION_ROOT" ]]; then
    supplement_args=(--input-root "$SUPPLEMENT_COLLECTION_ROOT")
  else
    supplement_args=()
  fi
  PYTHONPATH=src "$PY" tools/dataset_processing/finalize_urbannav_benchmark_package.py \
    --input-root "$LEGACY_FINAL15K" \
    --input-root "$COLLECTION_ROOT" \
    "${supplement_args[@]}" \
    --output-root "$FINAL_ROOT" \
    --distribution-plan "$FINAL_PLAN" \
    --quality-policy "$QUALITY_POLICY" \
    --target-cases 25000 \
    --review-cases 500 \
    --seed 20260715 \
    --overwrite
  PYTHONPATH=src "$PY" tools/dataset_processing/validate_urbannav_benchmark.py \
    --ground-truth "$FINAL_ROOT/task_gt.jsonl" \
    --poi-cache "$FINAL_ROOT/poi_cache.jsonl" \
    --route-cache "$FINAL_ROOT/route_cache.jsonl" \
    --rejects "$FINAL_ROOT/task_rejects.jsonl" \
    --multi-stop-continuity-tolerance-m 5 \
    --min-inter-stop-distance-m 20 \
    --require-disjoint-multistop-target-types \
    --require-full-multistop-route-matrix \
    --output-json "$FINAL_ROOT/validation_report.json"
}

case "$MODE" in
  prepare)
    prepare_pool
    ;;
  smoke)
    [[ -s "$POOL_ROOT/shards/split_manifest.json" ]] || prepare_pool
    run_smoke
    ;;
  collect)
    [[ -s "$POOL_ROOT/shards/split_manifest.json" ]] || prepare_pool
    run_collection
    ;;
  supplement-prepare)
    prepare_supplement_pool
    ;;
  supplement-collect)
    [[ -s "$SUPPLEMENT_POOL_ROOT/shards/split_manifest.json" ]] || prepare_supplement_pool
    run_supplement_collection
    ;;
  finalize)
    finalize_package
    ;;
  all)
    prepare_pool
    run_smoke
    run_collection
    prepare_supplement_pool
    run_supplement_collection
    finalize_package
    ;;
  *)
    print -u2 "Usage: $0 {prepare|smoke|collect|supplement-prepare|supplement-collect|finalize|all}"
    exit 2
    ;;
esac
