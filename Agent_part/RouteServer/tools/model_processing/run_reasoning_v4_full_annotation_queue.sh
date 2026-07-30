#!/usr/bin/env bash
set -Eeuo pipefail

# Persistent, resumable V4.6 full-annotation queue.
# Admission remains fail closed: launch requires the explicit pilot approval
# latch, and each later dataset starts only after the preceding completeness
# report and exact annotation row count pass.

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/lpai/volumes/mind-eb-bd-ga/suh/UrbanNavigation}"
ROUTESERVER_ROOT="${ROUTESERVER_ROOT:-${WORKSPACE_ROOT}/Urban-Nav-SR/Agent_part/RouteServer}"
PYTHON_BIN="${PYTHON_BIN:-/lpai/volumes/mind-eb-bd-ga/suh/miniconda3/envs/qwen35_vllm_cuda128/bin/python}"
RUNNER="${RUNNER:-${ROUTESERVER_ROOT}/tools/model_processing/run_reasoning_v4_qwen122b_dual_annotation.sh}"
DATASET_ROOT="${DATASET_ROOT:-/lpai/volumes/mind-eb-bd-ga/suh/dataset/urbannav-sidewalk-agent-data-v1}"
INPUT_ROOT="${INPUT_ROOT:-${WORKSPACE_ROOT}/annotation_inputs/reasoning_v4_6}"
QUEUE_ROOT="${QUEUE_ROOT:-${WORKSPACE_ROOT}/annotation_runs/reasoning_v4_6_qwen122b_full_queue_v1}"
QUEUE_SESSION="${QUEUE_SESSION:-reasoning-v4-full-queue}"
FOXGLOVE_SESSION="${FOXGLOVE_SESSION:-reasoning-v4-full}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-300}"

FOXGLOVE_INPUT="${FOXGLOVE_INPUT:-${DATASET_ROOT}/manifests/train/phase1_cot_teacher_frodobots_foxglove_lite_final/foxglove_yaw_p90_clean/manifests/phase1_cot_foxglove_yaw_p90_clean_lite_cot_final_direction_choice_cot_full_choice.jsonl}"
FOXGLOVE_ROWS=47249
FOXGLOVE_SHA256="79ba1fd7084d5364492f36b6e67ef2d0481e7cefe7c3869c02e5d1a44f5591cf"
FOXGLOVE_RUN_ROOT="${FOXGLOVE_RUN_ROOT:-${WORKSPACE_ROOT}/annotation_runs/reasoning_v4_6_qwen122b_foxglove_20way_full_v1}"

SIDEWALK_MAIN_INPUT="${SIDEWALK_MAIN_INPUT:-${INPUT_ROOT}/sidewalk_main_20way.jsonl}"
SIDEWALK_MAIN_ROWS=45397
SIDEWALK_MAIN_SHA256="32e1b2bbe6e68a89177d0b04f9114ce593e99aacc00d4511a149687dc14f069f"
SIDEWALK_MAIN_RUN_ROOT="${SIDEWALK_MAIN_RUN_ROOT:-${WORKSPACE_ROOT}/annotation_runs/reasoning_v4_6_qwen122b_sidewalk_main_20way_full_v1}"

SIDEWALK_EXTRA_INPUT="${SIDEWALK_EXTRA_INPUT:-${INPUT_ROOT}/sidewalk_extra_unique_20way.jsonl}"
SIDEWALK_EXTRA_ROWS=2611
SIDEWALK_EXTRA_SHA256="13d9a650cc17964e52e3aafe68bb9acd672ea6c898248f0bd2a00830ca3597e4"
SIDEWALK_EXTRA_RUN_ROOT="${SIDEWALK_EXTRA_RUN_ROOT:-${WORKSPACE_ROOT}/annotation_runs/reasoning_v4_6_qwen122b_sidewalk_extra_20way_full_v1}"

usage() {
  cat <<'EOF'
Usage:
  PILOT_APPROVED=1 run_reasoning_v4_full_annotation_queue.sh launch
  PILOT_APPROVED=1 run_reasoning_v4_full_annotation_queue.sh run
  run_reasoning_v4_full_annotation_queue.sh status

The queue is strictly sequential:
  Foxglove 47,249 -> Sidewalk main 45,397 -> Sidewalk extra-exclusive 2,611

`PILOT_APPROVED=1` may be set only after inspecting the 256-row natural
pilot report and review bundle. Full-run postprocessing is intentionally
disabled; each stage must still pass its complete source/annotation row gate.
EOF
}

die() {
  echo "[queue-error] $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "required file is missing: $1"
}

tmux_session_exists_exact() {
  local session="$1"
  tmux list-sessions -F '#S' 2>/dev/null \
    | grep -Fqx -- "${session}"
}

assert_positive_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] \
    || die "${name} must be a positive integer: ${value}"
}

assert_input_identity() {
  local path="$1"
  local expected_rows="$2"
  local expected_sha="$3"
  local actual_rows
  local actual_sha
  require_file "${path}"
  actual_rows="$(wc -l <"${path}" | tr -d ' ')"
  [[ "${actual_rows}" == "${expected_rows}" ]] \
    || die "input row mismatch: path=${path} expected=${expected_rows} actual=${actual_rows}"
  actual_sha="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual_sha}" == "${expected_sha}" ]] \
    || die "input SHA256 mismatch: path=${path} expected=${expected_sha} actual=${actual_sha}"
  echo "[input-ok] rows=${actual_rows} sha256=${actual_sha} path=${path}"
}

stage_complete() {
  local run_root="$1"
  local expected_rows="$2"
  local report="${run_root}/reports/annotation_completeness_report.json"
  local annotations="${run_root}/annotations/reasoning_v4_6_annotations.jsonl"
  [[ -f "${report}" && -f "${annotations}" ]] || return 1
  REPORT="${report}" ANNOTATIONS="${annotations}" EXPECTED="${expected_rows}" \
    "${PYTHON_BIN}" -c \
    'import json, os, pathlib; expected=int(os.environ["EXPECTED"]); report=json.load(open(os.environ["REPORT"], encoding="utf-8")); rows=sum(1 for line in pathlib.Path(os.environ["ANNOTATIONS"]).open(encoding="utf-8") if line.strip()); assert report.get("passed") is True, report; assert rows==expected,(rows,expected)' \
    >/dev/null 2>&1
}

assert_stage_complete() {
  local name="$1"
  local run_root="$2"
  local expected_rows="$3"
  stage_complete "${run_root}" "${expected_rows}" \
    || die "${name} did not pass its completeness gate: ${run_root}"
  echo "[stage-complete] name=${name} rows=${expected_rows} root=${run_root}"
}

status_stage() {
  local name="$1"
  local run_root="$2"
  local expected_rows="$3"
  local annotations="${run_root}/annotations/reasoning_v4_6_annotations.jsonl"
  local service_0="${run_root}/annotations/service_0.jsonl"
  local service_1="${run_root}/annotations/service_1.jsonl"
  local current=0
  if [[ -f "${annotations}" ]]; then
    current="$(wc -l <"${annotations}" | tr -d ' ')"
  else
    if [[ -f "${service_0}" ]]; then
      current=$((current + $(wc -l <"${service_0}")))
    fi
    if [[ -f "${service_1}" ]]; then
      current=$((current + $(wc -l <"${service_1}")))
    fi
  fi
  if stage_complete "${run_root}" "${expected_rows}"; then
    echo "[stage-status] name=${name} state=complete rows=${current}/${expected_rows}"
  else
    echo "[stage-status] name=${name} state=incomplete rows=${current}/${expected_rows}"
  fi
}

wait_for_foxglove() {
  while tmux_session_exists_exact "${FOXGLOVE_SESSION}"; do
    echo "[queue-wait] $(date --iso-8601=seconds) stage=foxglove"
    status_stage foxglove "${FOXGLOVE_RUN_ROOT}" "${FOXGLOVE_ROWS}"
    sleep "${POLL_INTERVAL_S}"
  done
  assert_stage_complete foxglove "${FOXGLOVE_RUN_ROOT}" "${FOXGLOVE_ROWS}"
}

run_sidewalk_stage() {
  local name="$1"
  local input_jsonl="$2"
  local run_root="$3"
  local expected_rows="$4"
  if stage_complete "${run_root}" "${expected_rows}"; then
    echo "[queue-skip] name=${name} reason=already_complete"
    return
  fi
  env \
    WORKSPACE_ROOT="${WORKSPACE_ROOT}" \
    ROUTESERVER_ROOT="${ROUTESERVER_ROOT}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    DATASET_ROOT="${DATASET_ROOT}" \
    DATASET_SOURCE=sidewalk \
    MEDIA_ROOT="${DATASET_ROOT}" \
    INPUT_JSONL="${input_jsonl}" \
    RUN_ROOT="${run_root}" \
    EXPECTED_ROWS="${expected_rows}" \
    POSTPROCESS_FULL=0 \
    bash "${RUNNER}" preflight
  env \
    WORKSPACE_ROOT="${WORKSPACE_ROOT}" \
    ROUTESERVER_ROOT="${ROUTESERVER_ROOT}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    DATASET_ROOT="${DATASET_ROOT}" \
    DATASET_SOURCE=sidewalk \
    MEDIA_ROOT="${DATASET_ROOT}" \
    INPUT_JSONL="${input_jsonl}" \
    RUN_ROOT="${run_root}" \
    EXPECTED_ROWS="${expected_rows}" \
    POSTPROCESS_FULL=0 \
    bash "${RUNNER}" run-full
  assert_stage_complete "${name}" "${run_root}" "${expected_rows}"
}

run_queue() {
  [[ "${PILOT_APPROVED:-0}" == "1" ]] \
    || die "set PILOT_APPROVED=1 only after inspecting the natural pilot"
  assert_positive_integer POLL_INTERVAL_S "${POLL_INTERVAL_S}"
  require_file "${RUNNER}"
  assert_input_identity \
    "${FOXGLOVE_INPUT}" "${FOXGLOVE_ROWS}" "${FOXGLOVE_SHA256}"
  assert_input_identity \
    "${SIDEWALK_MAIN_INPUT}" "${SIDEWALK_MAIN_ROWS}" "${SIDEWALK_MAIN_SHA256}"
  assert_input_identity \
    "${SIDEWALK_EXTRA_INPUT}" "${SIDEWALK_EXTRA_ROWS}" "${SIDEWALK_EXTRA_SHA256}"

  if stage_complete "${FOXGLOVE_RUN_ROOT}" "${FOXGLOVE_ROWS}"; then
    echo "[queue-skip] name=foxglove reason=already_complete"
  elif tmux_session_exists_exact "${FOXGLOVE_SESSION}"; then
    echo "[queue-resume-wait] name=foxglove session=${FOXGLOVE_SESSION}"
    wait_for_foxglove
  else
    env \
      PILOT_APPROVED=1 \
      WORKSPACE_ROOT="${WORKSPACE_ROOT}" \
      ROUTESERVER_ROOT="${ROUTESERVER_ROOT}" \
      PYTHON_BIN="${PYTHON_BIN}" \
      DATASET_ROOT="${DATASET_ROOT}" \
      DATASET_SOURCE=foxglove \
      MEDIA_ROOT="${DATASET_ROOT}" \
      FULL_INPUT_JSONL="${FOXGLOVE_INPUT}" \
      FULL_RUN_ROOT="${FOXGLOVE_RUN_ROOT}" \
      FULL_EXPECTED_ROWS="${FOXGLOVE_ROWS}" \
      FULL_TMUX_SESSION="${FOXGLOVE_SESSION}" \
      POSTPROCESS_FULL=0 \
      bash "${RUNNER}" launch-full
    wait_for_foxglove
  fi

  run_sidewalk_stage \
    sidewalk_main \
    "${SIDEWALK_MAIN_INPUT}" \
    "${SIDEWALK_MAIN_RUN_ROOT}" \
    "${SIDEWALK_MAIN_ROWS}"
  run_sidewalk_stage \
    sidewalk_extra_exclusive \
    "${SIDEWALK_EXTRA_INPUT}" \
    "${SIDEWALK_EXTRA_RUN_ROOT}" \
    "${SIDEWALK_EXTRA_ROWS}"
  echo "[queue-complete] $(date --iso-8601=seconds) all stages passed"
}

launch_queue() {
  [[ "${PILOT_APPROVED:-0}" == "1" ]] \
    || die "set PILOT_APPROVED=1 only after inspecting the natural pilot"
  tmux_session_exists_exact "${QUEUE_SESSION}" \
    && die "tmux session already exists: ${QUEUE_SESSION}"
  mkdir -p "${QUEUE_ROOT}/logs"
  local command
  local -a launch=(
    env
    "PILOT_APPROVED=1"
    "WORKSPACE_ROOT=${WORKSPACE_ROOT}"
    "ROUTESERVER_ROOT=${ROUTESERVER_ROOT}"
    "PYTHON_BIN=${PYTHON_BIN}"
    "DATASET_ROOT=${DATASET_ROOT}"
    "INPUT_ROOT=${INPUT_ROOT}"
    "QUEUE_ROOT=${QUEUE_ROOT}"
    "QUEUE_SESSION=${QUEUE_SESSION}"
    "FOXGLOVE_SESSION=${FOXGLOVE_SESSION}"
    "POLL_INTERVAL_S=${POLL_INTERVAL_S}"
    bash "$0" run
  )
  printf -v command '%q ' "${launch[@]}"
  tmux new-session -d -s "${QUEUE_SESSION}" \
    "exec ${command} >>${QUEUE_ROOT}/logs/queue.log 2>&1"
  echo "[queue-launched] session=${QUEUE_SESSION} root=${QUEUE_ROOT}"
}

status_queue() {
  if tmux_session_exists_exact "${QUEUE_SESSION}"; then
    echo "[queue] session=${QUEUE_SESSION} state=running"
  else
    echo "[queue] session=${QUEUE_SESSION} state=not-running"
  fi
  status_stage foxglove "${FOXGLOVE_RUN_ROOT}" "${FOXGLOVE_ROWS}"
  status_stage sidewalk_main "${SIDEWALK_MAIN_RUN_ROOT}" "${SIDEWALK_MAIN_ROWS}"
  status_stage \
    sidewalk_extra_exclusive \
    "${SIDEWALK_EXTRA_RUN_ROOT}" \
    "${SIDEWALK_EXTRA_ROWS}"
  echo "[queue-log] ${QUEUE_ROOT}/logs/queue.log"
}

case "${1:-help}" in
  launch)
    launch_queue
    ;;
  run)
    run_queue
    ;;
  status)
    status_queue
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
