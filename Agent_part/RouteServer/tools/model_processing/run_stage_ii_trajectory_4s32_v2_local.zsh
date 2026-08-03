#!/usr/bin/env zsh

# 从本地原始位姿生成严格 4 秒、32 点的 Stage II 轨迹数据资产。
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
ROUTE_ROOT="${SCRIPT_DIR:h:h}"
CODE_ROOT="${ROUTE_ROOT:h:h:h}"
MS_SWIFT_ROOT="${MS_SWIFT_ROOT:-${CODE_ROOT}/ms-swift}"
CONDA_ENV="${CONDA_ENV:-UrbanNav}"

RUN_ROOT="${RUN_ROOT:-/media/ubuntu/Extreme SSD/urbannav_stage_ii_metric_flu_4s_32_speed_audit_v2_1}"
QUANTILE="${QUANTILE:-0.995}"
RAW_CACHE_SIZE="${RAW_CACHE_SIZE:-16}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-1000}"
SMOKE_LIMIT_PER_MANIFEST="${SMOKE_LIMIT_PER_MANIFEST:-}"

SIDEWALK_MAIN_MANIFEST="${SIDEWALK_MAIN_MANIFEST:-/media/ubuntu/data2/SR-Nav Dataset/sidewalk_foxglove_direction12_v4_multiframe/manifests/sidewalk_main_multiframe_5hz_5f_direction_choice_12way_full.jsonl}"
SIDEWALK_EXTRA_MANIFEST="${SIDEWALK_EXTRA_MANIFEST:-/media/ubuntu/data2/SR-Nav Dataset/sidewalk_foxglove_direction12_v4_multiframe/manifests/sidewalk_extra_multiframe_5hz_5f_direction_choice_12way_full_abs.jsonl}"
FOXGLOVE_MANIFEST="${FOXGLOVE_MANIFEST:-/media/ubuntu/Extreme SSD/foxglove/processed_batch_yaw_p90/phase1_labeling_full_multiframe_5f_5hz/manifests/foxglove_yaw_p90_full_multiframe_5f_5hz_osm_qc_repaired_anomaly_only_clean.jsonl}"
EGOWALK_MANIFEST="${EGOWALK_MANIFEST:-/media/ubuntu/data2/EgoWalk/prepared/fpv_only_multiview_ms_swift/egowalk_fpv_only_source.jsonl}"

SIDEWALK_RAW_ROOT="${SIDEWALK_RAW_ROOT:-/media/ubuntu/data2/SR-Nav Dataset/sidewalk}"
FOXGLOVE_RAW_ROOT="${FOXGLOVE_RAW_ROOT:-/media/ubuntu/Extreme SSD/foxglove/processed_batch_yaw_p90/sidewalk/chunk_foxglove}"
EGOWALK_RAW_ROOT="${EGOWALK_RAW_ROOT:-/media/ubuntu/data2/EgoWalk}"
IDENTITY_REGISTRY_JSON="${IDENTITY_REGISTRY_JSON:-${ROUTE_ROOT}/config/stage_ii_cfm/trajectory_source_registry_v1.json}"

for required_path in \
  "${MS_SWIFT_ROOT}" \
  "${SIDEWALK_MAIN_MANIFEST}" \
  "${SIDEWALK_EXTRA_MANIFEST}" \
  "${FOXGLOVE_MANIFEST}" \
  "${EGOWALK_MANIFEST}" \
  "${SIDEWALK_RAW_ROOT}" \
  "${FOXGLOVE_RAW_ROOT}" \
  "${EGOWALK_RAW_ROOT}" \
  "${IDENTITY_REGISTRY_JSON}"; do
  if [[ ! -e "${required_path}" ]]; then
    print -u2 -- "[stage-ii-v2] missing required path: ${required_path}"
    exit 2
  fi
done

mkdir -p "${RUN_ROOT}"

build_args=(
  --dataset-manifest "sidewalk=${SIDEWALK_MAIN_MANIFEST}"
  --dataset-manifest "sidewalk=${SIDEWALK_EXTRA_MANIFEST}"
  --dataset-manifest "foxglove=${FOXGLOVE_MANIFEST}"
  --dataset-manifest "egowalk=${EGOWALK_MANIFEST}"
  --raw-root "sidewalk=${SIDEWALK_RAW_ROOT}"
  --raw-root "foxglove=${FOXGLOVE_RAW_ROOT}"
  --raw-root "egowalk=${EGOWALK_RAW_ROOT}"
  --identity-registry-json "${IDENTITY_REGISTRY_JSON}"
  --output-dir "${RUN_ROOT}"
  --raw-cache-size "${RAW_CACHE_SIZE}"
  --progress-interval "${PROGRESS_INTERVAL}"
)
if [[ -n "${SMOKE_LIMIT_PER_MANIFEST}" ]]; then
  build_args+=(--limit-per-manifest "${SMOKE_LIMIT_PER_MANIFEST}")
fi

print -- "[stage-ii-v2] run_root=${RUN_ROOT}"
print -- "[stage-ii-v2] conda_env=${CONDA_ENV}"
print -- "[stage-ii-v2] smoke_limit_per_manifest=${SMOKE_LIMIT_PER_MANIFEST:-none}"

conda run --no-capture-output -n "${CONDA_ENV}" \
  env PYTHONPATH="${ROUTE_ROOT}/src" \
  python "${SCRIPT_DIR}/build_stage_ii_trajectory_4s32.py" "${build_args[@]}"

conda run --no-capture-output -n "${CONDA_ENV}" \
  env PYTHONPATH="${MS_SWIFT_ROOT}" \
  python -m swift.sidewalk_nav.cli.fit_stage_ii_normalization_v2 \
  --train-sidecar "${RUN_ROOT}/trajectory_sidecars_v2_train.jsonl" \
  --output-json "${RUN_ROOT}/trajectory_normalization_v2.json" \
  --quantile "${QUANTILE}"

print -- "[stage-ii-v2] complete"
print -- "[stage-ii-v2] report=${RUN_ROOT}/trajectory_build_report_v2.json"
print -- "[stage-ii-v2] normalization=${RUN_ROOT}/trajectory_normalization_v2.json"
