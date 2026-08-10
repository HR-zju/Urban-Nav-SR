#!/usr/bin/env bash
# 单一断续队列：语言目标完成后，再执行只读数据质量过滤。
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/lpai/volumes/mind-eb-bd-ga/suh/UrbanNavigation}"
PIPELINE_RUN_ROOT="${PIPELINE_RUN_ROOT:-${WORKSPACE_ROOT}/runs/language_waypoint_then_navigation_quality_20260809}"
LANGUAGE_RUN_ROOT="${LANGUAGE_RUN_ROOT:-${PIPELINE_RUN_ROOT}/language_waypoint}"
QUALITY_RUN_ROOT="${QUALITY_RUN_ROOT:-${PIPELINE_RUN_ROOT}/navigation_quality}"

mkdir -p "${PIPELINE_RUN_ROOT}/logs"
printf '%s\n' "[pipeline] language run root=${LANGUAGE_RUN_ROOT}" | tee -a "${PIPELINE_RUN_ROOT}/logs/pipeline.log"
RUN_ROOT="${LANGUAGE_RUN_ROOT}" "${SCRIPT_DIR}/run_language_waypoint_full_dual_122b.sh" \
  2>&1 | tee -a "${PIPELINE_RUN_ROOT}/logs/language_waypoint.log"
printf '%s\n' "[pipeline] language stage completed; starting read-only quality filter=${QUALITY_RUN_ROOT}" | tee -a "${PIPELINE_RUN_ROOT}/logs/pipeline.log"
RUN_ROOT="${QUALITY_RUN_ROOT}" "${SCRIPT_DIR}/run_navigation_quality_full_qwen122b.sh" \
  2>&1 | tee -a "${PIPELINE_RUN_ROOT}/logs/navigation_quality.log"
printf '%s\n' "[done] pipeline=${PIPELINE_RUN_ROOT}" | tee -a "${PIPELINE_RUN_ROOT}/logs/pipeline.log"
