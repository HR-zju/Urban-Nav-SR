#!/usr/bin/env zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

PY="${URBANNAV_PYTHON:-/home/ubuntu/anaconda3/envs/UrbanNav/bin/python}"
MODE="${1:-prepare}"
QUALITY_POLICY="${URBANNAV_QUALITY_POLICY:-config/urbannav_25k_v2_quality_policy.json}"
FINAL_PLAN="${URBANNAV_FINAL_PLAN:-config/urbannav_25k_v3_23cities_distribution_plan.json}"
POOL_PLAN="${URBANNAV_POOL_PLAN:-config/urbannav_25k_v3_new_cities_pool_plan_36k.json}"
ASSET_REGIONS_CONFIG="${URBANNAV_ASSET_REGIONS_CONFIG:-config/urbannav_25k_v3_new_city_regions_expanded.json}"
REGIONS_CONFIG="${URBANNAV_REGIONS_CONFIG:-config/urbannav_25k_v3_new_city_hotspots.json}"
SOURCE_CONFIG="${URBANNAV_SOURCE_CONFIG:-config/urbannav_25k_v3_new_city_osm_sources.json}"
BASE_ROUTER_CONFIG="${URBANNAV_BASE_ROUTER_CONFIG:-config/urbannav_25k_v3_new_city_router.json}"
ROUTER_CONFIG="${URBANNAV_ROUTER_CONFIG:-config/urbannav_25k_v3_new_city_router_hotspots.json}"

ASSET_ROOT="${URBANNAV_ASSET_ROOT:-data/osm_expanded_23cities_v3}"
POOL_ROOT="${URBANNAV_POOL_ROOT:-data/benchmark/urbannav_instruction_pool_final25k_v3_new6_hotspots_36k}"
COLLECTION_ROOT="${URBANNAV_COLLECTION_ROOT:-data/benchmark/urbannav_v3_osm_local_osrm_foot_25k_v3_new6_hotspots_collection}"
FINAL_ROOT="${URBANNAV_FINAL_ROOT:-data/benchmark/urbannav_v3_osm_local_osrm_foot_final25k_v3_23cities}"
HOTSPOT_REPORT="${URBANNAV_HOTSPOT_REPORT:-$ASSET_ROOT/poi_hotspot_report.json}"

LEGACY_FINAL15K="${URBANNAV_LEGACY_FINAL15K:-data/benchmark/urbannav_v2_osm_local_osrm_foot_final_15k}"
V4_COLLECTION="${URBANNAV_V4_COLLECTION:-data/benchmark/urbannav_v2_osm_local_osrm_foot_25k_v2_collection_v2}"
V4_SUPPLEMENT="${URBANNAV_V4_SUPPLEMENT:-data/benchmark/urbannav_v2_osm_local_osrm_foot_25k_v2_supp1_collection}"
V6_SUPPLEMENT="${URBANNAV_V6_SUPPLEMENT:-data/benchmark/urbannav_v3_osm_local_osrm_foot_25k_v3_shenzhen_supplement_collection}"
V6_UPDATE_SUPPLEMENT="${URBANNAV_V6_UPDATE_SUPPLEMENT:-data/benchmark/urbannav_v3_osm_local_osrm_foot_25k_v3_shenzhen_update_supplement_collection}"

SINGLE_SHARDS="${URBANNAV_SINGLE_SHARDS:-6}"
COMPLEX_SHARDS="${URBANNAV_COMPLEX_SHARDS:-3}"
CASE_CONCURRENCY="${URBANNAV_CASE_CONCURRENCY:-4}"
COMPLEX_CASE_CONCURRENCY="${URBANNAV_COMPLEX_CASE_CONCURRENCY:-2}"
ROUTE_CONCURRENCY="${URBANNAV_ROUTE_CONCURRENCY:-8}"
SEED="${URBANNAV_SEED:-20260716}"

if [[ ! -x "$PY" ]]; then
  print -u2 "UrbanNav Python is not executable: $PY"
  exit 1
fi

prepare_asset_plan() {
  PYTHONPATH=src "$PY" tools/dataset_processing/prepare_urbannav_osm_local_assets.py \
    --global-config "$ASSET_REGIONS_CONFIG" \
    --source-config "$SOURCE_CONFIG" \
    --output-root "$ASSET_ROOT" \
    --router-config-output "$BASE_ROUTER_CONFIG" \
    --bbox-buffer-m 3000 \
    --write-docker-osrm-scripts \
    --write-router-config \
    --overwrite
}

prepare_assets() {
  prepare_asset_plan
  PYTHONPATH=src "$PY" tools/dataset_processing/prepare_urbannav_osm_local_assets.py \
    --global-config "$ASSET_REGIONS_CONFIG" \
    --source-config "$SOURCE_CONFIG" \
    --output-root "$ASSET_ROOT" \
    --router-config-output "$BASE_ROUTER_CONFIG" \
    --bbox-buffer-m 3000 \
    --download
  "$ASSET_ROOT/osrm/extract_city_pbf_all_docker.sh"
  PYTHONPATH=src "$PY" tools/dataset_processing/prepare_urbannav_osm_local_assets.py \
    --global-config "$ASSET_REGIONS_CONFIG" \
    --source-config "$SOURCE_CONFIG" \
    --output-root "$ASSET_ROOT" \
    --router-config-output "$BASE_ROUTER_CONFIG" \
    --bbox-buffer-m 3000 \
    --build-poi-index \
    --overwrite
  prepare_hotspots
  if [[ -x "$ASSET_ROOT/osrm/stop_osrm_all_docker.sh" ]]; then
    "$ASSET_ROOT/osrm/stop_osrm_all_docker.sh"
  fi
  "$ASSET_ROOT/osrm/build_osrm_all_docker.sh"
  "$ASSET_ROOT/osrm/run_osrm_all_docker.sh"
  wait_for_routers
}

prepare_hotspots() {
  PYTHONPATH=src "$PY" tools/dataset_processing/plan_urbannav_poi_hotspots.py \
    --regions-config "$ASSET_REGIONS_CONFIG" \
    --router-config "$BASE_ROUTER_CONFIG" \
    --poi-index-root "$ASSET_ROOT/index" \
    --output-regions-config "$REGIONS_CONFIG" \
    --output-router-config "$ROUTER_CONFIG" \
    --output-report "$HOTSPOT_REPORT" \
    --hotspots-per-city 5 \
    --hotspot-radius-m 800 \
    --search-radius-m 2500 \
    --grid-size-m 1000 \
    --min-separation-m 1800 \
    --overwrite
}

wait_for_routers() {
  local failed=0
  for port in 5031 5032 5033 5034 5035 5036; do
    local coordinates
    case "$port" in
      5031) coordinates="116.4110,39.9149;116.4120,39.9155" ;;
      5032) coordinates="121.4737,31.2304;121.4747,31.2310" ;;
      5033) coordinates="114.0570,22.5430;114.0580,22.5436" ;;
      5034) coordinates="120.1634,30.2587;120.1644,30.2593" ;;
      5035) coordinates="55.2744,25.1972;55.2754,25.1978" ;;
      5036) coordinates="37.6173,55.7558;37.6183,55.7564" ;;
    esac
    local ready=0
    for _ in $(seq 1 30); do
      if curl --noproxy 127.0.0.1,localhost -fsS \
        "http://127.0.0.1:$port/route/v1/foot/$coordinates?overview=false" >/dev/null 2>&1; then
        ready=1
        break
      fi
      sleep 2
    done
    if (( ready )); then
      print "router_port=$port ready"
    else
      print -u2 "router_port=$port not_ready"
      failed=1
    fi
  done
  return "$failed"
}

ensure_routers() {
  if ! wait_for_routers; then
    "$ASSET_ROOT/osrm/run_osrm_all_docker.sh"
    wait_for_routers
  fi
}

prepare_pool() {
  prepare_hotspots
  PYTHONPATH=src "$PY" tools/dataset_processing/build_urbannav_instruction_set.py \
    --distribution "$POOL_PLAN" \
    --regions-config "$REGIONS_CONFIG" \
    --quality-policy "$QUALITY_POLICY" \
    --city-plan-mode exact \
    --output-root "$POOL_ROOT" \
    --instruction-mode case_variant \
    --seed "$SEED" \
    --overwrite
  PYTHONPATH=src "$PY" tools/dataset_processing/filter_urbannav_instruction_coverage.py \
    --instruction-set "$POOL_ROOT/instruction_set.jsonl" \
    --regions-config "$REGIONS_CONFIG" \
    --router-config "$ROUTER_CONFIG" \
    --poi-index-root "$ASSET_ROOT/index" \
    --target-type-map config/urbannav_target_type_provider_map_mined_v2.json \
    --output-instruction-set "$POOL_ROOT/instruction_set_supported.jsonl" \
    --output-rejects "$POOL_ROOT/instruction_set_coverage_rejects.jsonl" \
    --output-report "$POOL_ROOT/city_target_type_coverage.json" \
    --search-radius-m 2500 \
    --overwrite
  PYTHONPATH=src "$PY" tools/dataset_processing/split_urbannav_instruction_pool.py \
    --instruction-set "$POOL_ROOT/instruction_set_supported.jsonl" \
    --output-root "$POOL_ROOT/shards" \
    --single-shards "$SINGLE_SHARDS" \
    --complex-shards "$COMPLEX_SHARDS"
}

collection_mode_args() {
  local output_root="$1"
  if [[ -s "$output_root/task_gt.jsonl" || -s "$output_root/task_rejects.jsonl" ]]; then
    print -- --resume
  else
    print -- --overwrite
  fi
}

run_smoke() {
  ensure_routers
  local smoke_root="${COLLECTION_ROOT}_smoke"
  PYTHONPATH=src "$PY" tools/dataset_processing/collect_urbannav_global_gt.py \
    --output-root "$smoke_root/single" \
    --config "$POOL_ROOT/collection_config_full_single_stop.json" \
    --instruction-set "$POOL_ROOT/shards/single_shard_00.jsonl" \
    --target-cases 12 \
    --seed "$SEED" \
    --max-attempts 60 \
    --provider osm_local \
    --poi-index-root "$ASSET_ROOT/index" \
    --local-router-config "$ROUTER_CONFIG" \
    --origins-per-region 12 \
    --candidate-limit 5 \
    --search-radius-m 2500 \
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
    --target-multistop 6 \
    --target-route-update 6 \
    --seed "$SEED" \
    --max-attempts 80 \
    --provider osm_local \
    --poi-index-root "$ASSET_ROOT/index" \
    --local-router-config "$ROUTER_CONFIG" \
    --origins-per-region 12 \
    --candidate-limit 5 \
    --search-radius-m 2500 \
    --case-concurrency 1 \
    --route-concurrency 4 \
    --retry 1 \
    --overwrite
}

run_collection() {
  ensure_routers
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
        --target-cases 4500 \
        --seed "$SEED" \
        --provider osm_local \
        --poi-index-root "$ASSET_ROOT/index" \
        --local-router-config "$ROUTER_CONFIG" \
        --origins-per-region 128 \
        --candidate-limit 5 \
        --search-radius-m 2500 \
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
        --target-multistop 1800 \
        --target-route-update 1200 \
        --seed "$SEED" \
        --provider osm_local \
        --poi-index-root "$ASSET_ROOT/index" \
        --local-router-config "$ROUTER_CONFIG" \
        --origins-per-region 128 \
        --candidate-limit 5 \
        --search-radius-m 2500 \
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

finalize_package() {
  PYTHONPATH=src "$PY" tools/dataset_processing/finalize_urbannav_benchmark_package.py \
    --input-root "$LEGACY_FINAL15K" \
    --input-root "$V4_COLLECTION" \
    --input-root "$V4_SUPPLEMENT" \
    --input-root "$COLLECTION_ROOT" \
    --input-root "$V6_SUPPLEMENT" \
    --input-root "$V6_UPDATE_SUPPLEMENT" \
    --output-root "$FINAL_ROOT" \
    --distribution-plan "$FINAL_PLAN" \
    --quality-policy "$QUALITY_POLICY" \
    --target-cases 25000 \
    --review-cases 500 \
    --seed "$SEED" \
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
  asset-plan)
    prepare_asset_plan
    ;;
  assets)
    prepare_assets
    ;;
  hotspots)
    prepare_hotspots
    ;;
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
  finalize)
    finalize_package
    ;;
  all)
    prepare_assets
    prepare_pool
    run_smoke
    run_collection
    finalize_package
    ;;
  *)
    print -u2 "Usage: $0 {asset-plan|assets|hotspots|prepare|smoke|collect|finalize|all}"
    exit 2
    ;;
esac
