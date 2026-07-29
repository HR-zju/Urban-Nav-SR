#!/usr/bin/env bash
set -Eeuo pipefail

# Reasoning V4.6 annotation runner for two local Qwen3.5-122B-A10B services.
# All generated artifacts live under the shared /lpai volume by default.

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/lpai/volumes/mind-eb-bd-ga/suh/UrbanNavigation}"
ROUTESERVER_ROOT="${ROUTESERVER_ROOT:-${WORKSPACE_ROOT}/Urban-Nav-SR/Agent_part/RouteServer}"
PYTHON_BIN="${PYTHON_BIN:-/lpai/volumes/mind-eb-bd-ga/suh/miniconda3/envs/qwen35_vllm_cuda128/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/lpai/volumes/mind-eb-bd-ga/suh/dataset/urbannav-sidewalk-agent-data-v1}"

BEHAVIOR_BASE_URL="${BEHAVIOR_BASE_URL:-http://127.0.0.1:18461/v1}"
BEHAVIOR_MODEL="${BEHAVIOR_MODEL:-qwen3.5-122b-a10b-behavior}"
INSTRUCTION_BASE_URL="${INSTRUCTION_BASE_URL:-http://127.0.0.1:18462/v1}"
INSTRUCTION_MODEL="${INSTRUCTION_MODEL:-qwen3.5-122b-a10b-instruction}"

DIRECTION_PROFILE_ID="${DIRECTION_PROFILE_ID:-fpv_degree10_terminal_centroid_v2}"
CONCURRENCY_PER_SERVICE="${CONCURRENCY_PER_SERVICE:-2}"
MAX_RETRIES="${MAX_RETRIES:-2}"
IMAGE_DETAIL="${IMAGE_DETAIL:-low}"
NAVIGATION_CONTEXT_IMAGE_DETAIL="${NAVIGATION_CONTEXT_IMAGE_DETAIL:-high}"
RETRY_FAILED_ON_RESUME="${RETRY_FAILED_ON_RESUME:-1}"
LLM_VISION_TIMEOUT_S="${LLM_VISION_TIMEOUT_S:-900}"
LLM_VISION_MAX_TOKENS="${LLM_VISION_MAX_TOKENS:-4096}"
POSTPROCESS_FULL="${POSTPROCESS_FULL:-0}"

FULL_INPUT_JSONL_DEFAULT="${DATASET_ROOT}/manifests/train/phase1_cot_teacher_frodobots_foxglove_lite_final/foxglove_yaw_p90_clean/manifests/phase1_cot_foxglove_yaw_p90_clean_lite_cot_final_direction_choice_cot_full_choice.jsonl"
FULL_RUN_ROOT_DEFAULT="${WORKSPACE_ROOT}/annotation_runs/reasoning_v4_6_qwen122b_foxglove_20way_full_v1"
FULL_EXPECTED_ROWS="${FULL_EXPECTED_ROWS:-47249}"
FULL_TMUX_SESSION="${FULL_TMUX_SESSION:-reasoning-v4-full}"

SMOKE_INPUT_JSONL_DEFAULT="${FULL_INPUT_JSONL_DEFAULT}"
SMOKE_RUN_ROOT_DEFAULT="${WORKSPACE_ROOT}/annotation_runs/reasoning_v4_6_qwen122b_dual_smoke_v1"
SMOKE_EXPECTED_ROWS="${SMOKE_EXPECTED_ROWS:-64}"
SMOKE_TMUX_SESSION="${SMOKE_TMUX_SESSION:-reasoning-v4-smoke}"
SMOKE_INTERESTING_JSONLS_DEFAULT="${WORKSPACE_ROOT}/runs/qwen35_122b_a10b_20260716/behavior_filter/foxglove/interesting.jsonl:${DATASET_ROOT}/manifests/value_filter/qwen35_retry_unknown_20260713_1415/foxglove/interesting.jsonl:${DATASET_ROOT}/manifests/value_filter/doubao_lite_20260709_184920/foxglove/interesting.jsonl"
SMOKE_INTERESTING_JSONLS="${SMOKE_INTERESTING_JSONLS:-${SMOKE_INTERESTING_JSONLS_DEFAULT}}"
SMOKE_RANDOM_SEED="${SMOKE_RANDOM_SEED:-20260728}"
SMOKE_CONSENSUS_MIN_VOTES="${SMOKE_CONSENSUS_MIN_VOTES:-2}"
SMOKE_RESOLUTION_JSONL="${SMOKE_RESOLUTION_JSONL:-}"

SCRIPT_PATH="$(readlink -f "$0")"

usage() {
  cat <<'EOF'
Usage:
  run_reasoning_v4_qwen122b_dual_annotation.sh preflight
  run_reasoning_v4_qwen122b_dual_annotation.sh smoke
  run_reasoning_v4_qwen122b_dual_annotation.sh launch-smoke
  run_reasoning_v4_qwen122b_dual_annotation.sh status-smoke
  run_reasoning_v4_qwen122b_dual_annotation.sh tail-smoke

  SMOKE_APPROVED=1 SMOKE_RESOLUTION_JSONL=/path/to/resolutions.jsonl \
    run_reasoning_v4_qwen122b_dual_annotation.sh launch-full
  run_reasoning_v4_qwen122b_dual_annotation.sh status-full
  run_reasoning_v4_qwen122b_dual_annotation.sh tail-full

Important overrides:
  WORKSPACE_ROOT, ROUTESERVER_ROOT, PYTHON_BIN, DATASET_ROOT
  SMOKE_INPUT_JSONL, SMOKE_RUN_ROOT, SMOKE_EXPECTED_ROWS
  SMOKE_INTERESTING_JSONLS, SMOKE_RANDOM_SEED
  SMOKE_CONSENSUS_MIN_VOTES
  FULL_INPUT_JSONL, FULL_RUN_ROOT, FULL_EXPECTED_ROWS
  CONCURRENCY_PER_SERVICE, MAX_RETRIES, IMAGE_DETAIL
  NAVIGATION_CONTEXT_IMAGE_DETAIL
  POSTPROCESS_FULL=1  # Not recommended until the full exporter is streaming.

Safety:
  - The default full input is the 47,249-row Foxglove 20-way manifest.
  - The legacy direction is validated and stamped; it is never re-derived.
  - Smoke uses historical interesting labels, a fixed seed, four direction
    buckets, and at most one sample per clip.
  - The 96,986-row legacy 12-way merged manifest is intentionally not a
    default because it does not satisfy the current Foxglove V4.6 profile.
  - launch-full requires exact V4.6 machine-readable smoke gates, completed
    append-only resolutions for triggered rows, and SMOKE_APPROVED=1.
EOF
}

die() {
  echo "[error] $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "required file is missing: $1"
}

validate_positive_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] \
    || die "${name} must be a positive integer: ${value}"
}

strict_chat_smoke() {
  local base_url="$1"
  local model="$2"
  local response
  response="$(
    curl -fsS --max-time 120 \
      "${base_url}/chat/completions" \
      -H 'Content-Type: application/json' \
      --data-binary \
      "{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"Return exactly one JSON object: {\\\"status\\\":\\\"OK\\\"}\"}],\"response_format\":{\"type\":\"json_object\"},\"temperature\":0,\"max_tokens\":16}"
  )"
  RESPONSE="${response}" "${PYTHON_BIN}" -c \
    'import json, os; payload=json.loads(os.environ["RESPONSE"]); content=json.loads(payload["choices"][0]["message"].get("content", "")); assert content == {"status":"OK"}, content'
  echo "[chat-smoke-ok] model=${model} base_url=${base_url}"
}

preflight() {
  local code_commit
  local interesting_path
  local -a interesting_paths
  require_file "${ROUTESERVER_ROOT}/tools/model_processing/label_reasoning_v4.py"
  require_file "${ROUTESERVER_ROOT}/tools/model_processing/audit_reasoning_v4_annotation_completeness.py"
  require_file "${ROUTESERVER_ROOT}/tools/model_processing/stamp_reasoning_v4_directions.py"
  [[ -x "${PYTHON_BIN}" ]] || die "annotation Python is unavailable: ${PYTHON_BIN}"
  command -v curl >/dev/null 2>&1 || die "curl is unavailable"
  command -v awk >/dev/null 2>&1 || die "awk is unavailable"
  command -v tmux >/dev/null 2>&1 || die "tmux is unavailable"
  validate_positive_integer CONCURRENCY_PER_SERVICE "${CONCURRENCY_PER_SERVICE}"
  validate_positive_integer LLM_VISION_MAX_TOKENS "${LLM_VISION_MAX_TOKENS}"
  validate_positive_integer SMOKE_RANDOM_SEED "${SMOKE_RANDOM_SEED}"
  validate_positive_integer \
    SMOKE_CONSENSUS_MIN_VOTES "${SMOKE_CONSENSUS_MIN_VOTES}"
  IFS=: read -r -a interesting_paths <<<"${SMOKE_INTERESTING_JSONLS}"
  ((${#interesting_paths[@]} > 0)) \
    || die "SMOKE_INTERESTING_JSONLS must not be empty"
  for interesting_path in "${interesting_paths[@]}"; do
    require_file "${interesting_path}"
  done
  [[ "${IMAGE_DETAIL}" =~ ^(auto|low|high)$ ]] \
    || die "IMAGE_DETAIL must be auto, low, or high"
  [[ "${NAVIGATION_CONTEXT_IMAGE_DETAIL}" =~ ^(auto|low|high)$ ]] \
    || die "NAVIGATION_CONTEXT_IMAGE_DETAIL must be auto, low, or high"
  PYTHONPATH="${ROUTESERVER_ROOT}/src:${ROUTESERVER_ROOT}" \
    "${PYTHON_BIN}" -c \
    'import openai, PIL, pydantic; import mcp_server; from mcp_server.contracts.reasoning_dataset_v4 import ReasoningSourceV4'
  strict_chat_smoke "${BEHAVIOR_BASE_URL}" "${BEHAVIOR_MODEL}"
  strict_chat_smoke "${INSTRUCTION_BASE_URL}" "${INSTRUCTION_MODEL}"
  code_commit="$(git -C "${ROUTESERVER_ROOT}" rev-parse HEAD)"
  echo "[preflight-ok] code_commit=${code_commit}"
}

prepare_run_dirs() {
  local run_root="$1"
  mkdir -p \
    "${run_root}/inputs" \
    "${run_root}/sources/shards" \
    "${run_root}/annotations" \
    "${run_root}/views" \
    "${run_root}/reports" \
    "${run_root}/review" \
    "${run_root}/logs" \
    "${run_root}/tmp"
}

record_provenance() {
  local input_jsonl="$1"
  local run_root="$2"
  local input_sha
  local code_commit
  local dirty_count
  input_sha="$(sha256sum "${input_jsonl}" | awk '{print $1}')"
  code_commit="$(git -C "${ROUTESERVER_ROOT}" rev-parse HEAD)"
  dirty_count="$(
    git -C "${ROUTESERVER_ROOT}" status --porcelain | wc -l | tr -d ' '
  )"
  {
    echo "schema_version=urbannav_reasoning_v4_6_qwen122b_run_provenance_v1"
    echo "created_at=$(date --iso-8601=seconds)"
    echo "code_root=${ROUTESERVER_ROOT}"
    echo "code_commit=${code_commit}"
    echo "code_dirty_entry_count=${dirty_count}"
    echo "input_jsonl=${input_jsonl}"
    echo "input_sha256=${input_sha}"
    echo "direction_profile_id=${DIRECTION_PROFILE_ID}"
    echo "behavior_base_url=${BEHAVIOR_BASE_URL}"
    echo "behavior_model=${BEHAVIOR_MODEL}"
    echo "instruction_base_url=${INSTRUCTION_BASE_URL}"
    echo "instruction_model=${INSTRUCTION_MODEL}"
    echo "concurrency_per_service=${CONCURRENCY_PER_SERVICE}"
    echo "max_retries=${MAX_RETRIES}"
    echo "image_detail=${IMAGE_DETAIL}"
    echo "navigation_context_image_detail=${NAVIGATION_CONTEXT_IMAGE_DETAIL}"
    echo "vision_timeout_s=${LLM_VISION_TIMEOUT_S}"
    echo "vision_max_tokens=${LLM_VISION_MAX_TOKENS}"
    echo "smoke_interesting_jsonls=${SMOKE_INTERESTING_JSONLS}"
    echo "smoke_random_seed=${SMOKE_RANDOM_SEED}"
    echo "smoke_consensus_min_votes=${SMOKE_CONSENSUS_MIN_VOTES}"
  } >"${run_root}/reports/run_provenance.txt"
}

stamp_and_build_sources() {
  local input_jsonl="$1"
  local run_root="$2"
  local expected_rows="$3"
  local selection="$4"
  local stamped="${run_root}/inputs/stamped_input.jsonl"
  local sources="${run_root}/sources/reasoning_v4_5_sources.jsonl"
  local interesting_path
  local -a interesting_paths
  local -a stamp_cmd=(
    "${PYTHON_BIN}"
    "${ROUTESERVER_ROOT}/tools/model_processing/stamp_reasoning_v4_directions.py"
    --input-jsonl "${input_jsonl}"
    --output-jsonl "${stamped}"
    --report-json "${run_root}/reports/direction_stamp_report.json"
    --dataset-source foxglove
    --profile-id "${DIRECTION_PROFILE_ID}"
    --selection "${selection}"
  )
  validate_positive_integer EXPECTED_ROWS "${expected_rows}"
  if [[ "${selection}" != "sequential" ]]; then
    stamp_cmd+=(--limit "${expected_rows}")
  fi
  if [[ "${selection}" == "interesting_diverse" ]]; then
    stamp_cmd+=(
      --random-seed "${SMOKE_RANDOM_SEED}"
      --consensus-min-votes "${SMOKE_CONSENSUS_MIN_VOTES}"
    )
    IFS=: read -r -a interesting_paths <<<"${SMOKE_INTERESTING_JSONLS}"
    for interesting_path in "${interesting_paths[@]}"; do
      stamp_cmd+=(--interesting-jsonl "${interesting_path}")
    done
  fi
  PYTHONPATH="${ROUTESERVER_ROOT}/src:${ROUTESERVER_ROOT}" \
    "${stamp_cmd[@]}"
  PYTHONPATH="${ROUTESERVER_ROOT}/src:${ROUTESERVER_ROOT}" \
    "${PYTHON_BIN}" \
    "${ROUTESERVER_ROOT}/tools/model_processing/build_reasoning_v4_sources.py" \
    --input-jsonl "${stamped}" \
    --output-jsonl "${sources}" \
    --report-json "${run_root}/reports/source_build_report.json" \
    --dataset-source foxglove \
    --travel-mode pedestrian_assist \
    --max-fpv-frames 5 \
    --media-root "${DATASET_ROOT}"
  PYTHONPATH="${ROUTESERVER_ROOT}/src:${ROUTESERVER_ROOT}" \
    "${PYTHON_BIN}" \
    "${ROUTESERVER_ROOT}/tools/model_processing/audit_reasoning_v4_source_coverage.py" \
    --source-jsonl "${sources}" \
    --require-dataset foxglove \
    --min-source-rows "${expected_rows}" \
    --report-json "${run_root}/reports/source_coverage_report.json" \
    --strict
  EXPECTED_ROWS="${expected_rows}" RUN_ROOT_VALUE="${run_root}" \
    SELECTION="${selection}" \
    "${PYTHON_BIN}" -c \
    'import json, os, pathlib; root=pathlib.Path(os.environ["RUN_ROOT_VALUE"]); expected=int(os.environ["EXPECTED_ROWS"]); selection=os.environ["SELECTION"]; stamp=json.loads((root/"reports/direction_stamp_report.json").read_text()); source=json.loads((root/"reports/source_build_report.json").read_text()); sc=stamp["counts"]; rc=source["counts"]; assert sc.get("output_rows")==expected, sc; assert sc.get("invalid_rows", 0)==0, sc; assert selection != "sequential" or sc.get("input_rows")==expected, sc; assert rc.get("input_rows")==expected, rc; assert rc.get("output_rows")==expected, rc; assert rc.get("direction_rows")==expected, rc'
  split_sources "${sources}" "${run_root}"
}

split_sources() {
  local sources="$1"
  local run_root="$2"
  local shard0="${run_root}/sources/shards/service_0.jsonl"
  local shard1="${run_root}/sources/shards/service_1.jsonl"
  local tmp0="${shard0}.tmp.$$"
  local tmp1="${shard1}.tmp.$$"
  awk 'NR % 2 == 1 {print}' "${sources}" >"${tmp0}"
  awk 'NR % 2 == 0 {print}' "${sources}" >"${tmp1}"
  mv "${tmp0}" "${shard0}"
  mv "${tmp1}" "${shard1}"
  [[ -s "${shard0}" && -s "${shard1}" ]] \
    || die "source sharding produced an empty shard"
  echo "[sharded] service_0=$(wc -l <"${shard0}") service_1=$(wc -l <"${shard1}")"
}

run_label_worker() {
  local worker_id="$1"
  local base_url="$2"
  local model="$3"
  local run_root="$4"
  local shard="${run_root}/sources/shards/service_${worker_id}.jsonl"
  local annotation="${run_root}/annotations/service_${worker_id}.jsonl"
  local rejected="${run_root}/annotations/service_${worker_id}.rejected.jsonl"
  local report="${run_root}/reports/annotation_service_${worker_id}.json"
  local -a retry_args=()
  if [[ "${RETRY_FAILED_ON_RESUME}" == "1" ]]; then
    retry_args+=(--retry-failed-on-resume)
  fi
  env \
    PYTHONPATH="${ROUTESERVER_ROOT}/src:${ROUTESERVER_ROOT}" \
    TMPDIR="${run_root}/tmp" \
    NO_PROXY="127.0.0.1,localhost" \
    no_proxy="127.0.0.1,localhost" \
    LLM_PROVIDER=openai \
    LLM_API_STYLE=chat_completions \
    LLM_API_STYLE_FALLBACK=0 \
    LLM_API_KEY=local-vllm \
    LLM_API_BASE_URL="${base_url}" \
    LLM_MODEL="${model}" \
    LLM_TEMPERATURE=0 \
    LLM_MAX_RETRIES=0 \
    LLM_CHAT_TEMPLATE_ENABLE_THINKING=false \
    LLM_RESPONSE_FORMAT_JSON=true \
    LLM_VISION_RESPONSE_FORMAT_JSON=1 \
    LLM_VISION_TIMEOUT_S="${LLM_VISION_TIMEOUT_S}" \
    LLM_VISION_MAX_TOKENS="${LLM_VISION_MAX_TOKENS}" \
    "${PYTHON_BIN}" \
    "${ROUTESERVER_ROOT}/tools/model_processing/label_reasoning_v4.py" \
    --source-jsonl "${shard}" \
    --annotation-jsonl "${annotation}" \
    --rejected-jsonl "${rejected}" \
    --report-json "${report}" \
    --concurrency "${CONCURRENCY_PER_SERVICE}" \
    --max-retries "${MAX_RETRIES}" \
    --image-detail "${IMAGE_DETAIL}" \
    --navigation-context-image-detail "${NAVIGATION_CONTEXT_IMAGE_DETAIL}" \
    --progress-interval 10 \
    "${retry_args[@]}"
}

label_with_both_services() {
  local run_root="$1"
  local pid0
  local pid1
  local failed=0
  run_label_worker \
    0 "${BEHAVIOR_BASE_URL}" "${BEHAVIOR_MODEL}" "${run_root}" \
    >"${run_root}/logs/annotation_service_0.log" 2>&1 &
  pid0=$!
  run_label_worker \
    1 "${INSTRUCTION_BASE_URL}" "${INSTRUCTION_MODEL}" "${run_root}" \
    >"${run_root}/logs/annotation_service_1.log" 2>&1 &
  pid1=$!
  echo "[annotation-started] service_0_pid=${pid0} service_1_pid=${pid1}"
  wait "${pid0}" || failed=1
  wait "${pid1}" || failed=1
  ((failed == 0)) || die "one or both annotation workers failed; rerun to resume"
}

merge_annotations() {
  local run_root="$1"
  local merged="${run_root}/annotations/reasoning_v4_6_annotations.jsonl"
  local temporary="${merged}.tmp.$$"
  require_file "${run_root}/annotations/service_0.jsonl"
  require_file "${run_root}/annotations/service_1.jsonl"
  awk 'NF {print}' \
    "${run_root}/annotations/service_0.jsonl" \
    "${run_root}/annotations/service_1.jsonl" \
    >"${temporary}"
  mv "${temporary}" "${merged}"
}

build_review_run() {
  local run_root="$1"
  local sources="${run_root}/sources/reasoning_v4_5_sources.jsonl"
  local annotations="${run_root}/annotations/reasoning_v4_6_annotations.jsonl"
  local views="${run_root}/views/reasoning_v4_reconciliation_preview_views.jsonl"
  PYTHONPATH="${ROUTESERVER_ROOT}/src:${ROUTESERVER_ROOT}" \
    "${PYTHON_BIN}" \
    "${ROUTESERVER_ROOT}/tools/model_processing/export_reasoning_v4_views.py" \
    --source-jsonl "${sources}" \
    --annotation-jsonl "${annotations}" \
    --release-track review_preview \
    --output-jsonl "${views}" \
    --report-json "${run_root}/reports/view_export_report.json" \
    --skip-errors
  PYTHONPATH="${ROUTESERVER_ROOT}/src:${ROUTESERVER_ROOT}" \
    "${PYTHON_BIN}" \
    "${ROUTESERVER_ROOT}/tools/model_processing/export_reasoning_v4_reconciliation_review.py" \
    --source-jsonl "${sources}" \
    --annotation-jsonl "${annotations}" \
    --audit-jsonl \
    "${run_root}/annotations/action_reconciliation_audits.jsonl" \
    --queue-jsonl \
    "${run_root}/review/action_reconciliation_queue.jsonl" \
    --report-json \
    "${run_root}/reports/action_reconciliation_report.json" \
    --strict
  PYTHONPATH="${ROUTESERVER_ROOT}/src:${ROUTESERVER_ROOT}" \
    "${PYTHON_BIN}" \
    "${ROUTESERVER_ROOT}/tools/model_processing/build_reasoning_v4_review_html.py" \
    --source-jsonl "${sources}" \
    --annotation-jsonl "${annotations}" \
    --views-jsonl "${views}" \
    --output-html "${run_root}/review/action_reconciliation_review.complete.html" \
    --report-json "${run_root}/reports/review_report.json" \
    --title "UrbanNav Reasoning V4.6 Qwen3.5-122B Review" \
    --max-samples 100
}

audit_postprocessed_run() {
  local run_root="$1"
  local sources="${run_root}/sources/reasoning_v4_5_sources.jsonl"
  local annotations="${run_root}/annotations/reasoning_v4_6_annotations.jsonl"
  local views="${run_root}/views/reasoning_v4_reconciliation_preview_views.jsonl"
  PYTHONPATH="${ROUTESERVER_ROOT}/src:${ROUTESERVER_ROOT}" \
    "${PYTHON_BIN}" \
    "${ROUTESERVER_ROOT}/tools/model_processing/audit_reasoning_v4_artifacts.py" \
    --source-jsonl "${sources}" \
    --annotation-jsonl "${annotations}" \
    --view-jsonl "${views}" \
    --report-json "${run_root}/reports/artifact_audit_report.json" \
    --require-dataset foxglove \
    --strict
}

assert_annotation_completeness() {
  local sources="$1"
  local annotations="$2"
  local expected_rows="$3"
  local run_root="$4"
  PYTHONPATH="${ROUTESERVER_ROOT}/src:${ROUTESERVER_ROOT}" \
    "${PYTHON_BIN}" \
    "${ROUTESERVER_ROOT}/tools/model_processing/audit_reasoning_v4_annotation_completeness.py" \
    --source-jsonl "${sources}" \
    --annotation-jsonl "${annotations}" \
    --expected-rows "${expected_rows}" \
    --report-json \
    "${run_root}/reports/annotation_completeness_report.json" \
    --strict
}

run_pipeline() {
  local input_jsonl="$1"
  local run_root="$2"
  local expected_rows="$3"
  local postprocess="$4"
  local selection="$5"
  prepare_run_dirs "${run_root}"
  exec > >(tee -a "${run_root}/logs/controller.log") 2>&1
  echo "[run-start] root=${run_root} input=${input_jsonl}"
  preflight
  require_file "${input_jsonl}"
  record_provenance "${input_jsonl}" "${run_root}"
  stamp_and_build_sources \
    "${input_jsonl}" "${run_root}" "${expected_rows}" "${selection}"
  label_with_both_services "${run_root}"
  merge_annotations "${run_root}"
  if [[ "${postprocess}" != "1" ]]; then
    PYTHONPATH="${ROUTESERVER_ROOT}/src:${ROUTESERVER_ROOT}" \
      "${PYTHON_BIN}" \
      "${ROUTESERVER_ROOT}/tools/model_processing/export_reasoning_v4_reconciliation_review.py" \
      --source-jsonl \
      "${run_root}/sources/reasoning_v4_5_sources.jsonl" \
      --annotation-jsonl \
      "${run_root}/annotations/reasoning_v4_6_annotations.jsonl" \
      --audit-jsonl \
      "${run_root}/annotations/action_reconciliation_audits.jsonl" \
      --queue-jsonl \
      "${run_root}/review/action_reconciliation_queue.jsonl" \
      --report-json \
      "${run_root}/reports/action_reconciliation_report.json" \
      --strict
  fi
  if [[ "${postprocess}" == "1" ]]; then
    build_review_run "${run_root}"
    echo "[review-ready-before-gate] ${run_root}/review/action_reconciliation_review.complete.html"
  fi
  assert_annotation_completeness \
    "${run_root}/sources/reasoning_v4_5_sources.jsonl" \
    "${run_root}/annotations/reasoning_v4_6_annotations.jsonl" \
    "${expected_rows}" \
    "${run_root}"
  if [[ "${postprocess}" == "1" ]]; then
    audit_postprocessed_run "${run_root}"
    echo "[run-complete] review=${run_root}/review/action_reconciliation_review.complete.html"
  else
    echo "[run-complete] annotation=${run_root}/annotations/reasoning_v4_6_annotations.jsonl"
    echo "[run-complete] full view export skipped because the current exporter is not streaming"
  fi
}

validate_smoke_gate() {
  local smoke_root="${SMOKE_RUN_ROOT:-${SMOKE_RUN_ROOT_DEFAULT}}"
  local completeness="${smoke_root}/reports/annotation_completeness_report.json"
  local sources="${smoke_root}/sources/reasoning_v4_5_sources.jsonl"
  local annotations="${smoke_root}/annotations/reasoning_v4_6_annotations.jsonl"
  local formal_views="${smoke_root}/views/reasoning_v4_6_formal_gate.jsonl"
  local review_report="${smoke_root}/reports/reasoning_v4_6_release_gate.json"
  local artifact_report="${smoke_root}/reports/reasoning_v4_6_formal_artifact_gate.json"
  local -a resolution_args=()
  [[ "${SMOKE_APPROVED:-0}" == "1" ]] \
    || die "set SMOKE_APPROVED=1 only after manually reviewing the smoke HTML"
  require_file "${sources}"
  require_file "${annotations}"
  require_file "${completeness}"
  if [[ -n "${SMOKE_RESOLUTION_JSONL}" ]]; then
    require_file "${SMOKE_RESOLUTION_JSONL}"
    resolution_args=(
      --resolution-jsonl "${SMOKE_RESOLUTION_JSONL}"
    )
  fi
  PYTHONPATH="${ROUTESERVER_ROOT}/src:${ROUTESERVER_ROOT}" \
    "${PYTHON_BIN}" \
    "${ROUTESERVER_ROOT}/tools/model_processing/export_reasoning_v4_reconciliation_review.py" \
    --source-jsonl "${sources}" \
    --annotation-jsonl "${annotations}" \
    "${resolution_args[@]}" \
    --audit-jsonl \
    "${smoke_root}/annotations/action_reconciliation_audits.gate.jsonl" \
    --queue-jsonl \
    "${smoke_root}/review/action_reconciliation_queue.gate.jsonl" \
    --report-json "${review_report}" \
    --strict
  PYTHONPATH="${ROUTESERVER_ROOT}/src:${ROUTESERVER_ROOT}" \
    "${PYTHON_BIN}" \
    "${ROUTESERVER_ROOT}/tools/model_processing/export_reasoning_v4_views.py" \
    --source-jsonl "${sources}" \
    --annotation-jsonl "${annotations}" \
    "${resolution_args[@]}" \
    --release-track formal_training \
    --output-jsonl "${formal_views}" \
    --report-json "${smoke_root}/reports/formal_view_export_gate.json"
  PYTHONPATH="${ROUTESERVER_ROOT}/src:${ROUTESERVER_ROOT}" \
    "${PYTHON_BIN}" \
    "${ROUTESERVER_ROOT}/tools/model_processing/audit_reasoning_v4_artifacts.py" \
    --source-jsonl "${sources}" \
    --annotation-jsonl "${annotations}" \
    "${resolution_args[@]}" \
    --view-jsonl "${formal_views}" \
    --report-json "${artifact_report}" \
    --require-dataset foxglove \
    --strict
  ARTIFACT="${artifact_report}" COMPLETENESS="${completeness}" \
    REVIEW="${review_report}" EXPECTED="${SMOKE_EXPECTED_ROWS}" \
    FORMAL_VIEWS="${formal_views}" \
    "${PYTHON_BIN}" -c \
    'import json, os, pathlib; expected=int(os.environ["EXPECTED"]); audit=json.load(open(os.environ["ARTIFACT"], encoding="utf-8")); complete=json.load(open(os.environ["COMPLETENESS"], encoding="utf-8")); review=json.load(open(os.environ["REVIEW"], encoding="utf-8")); metrics=audit["metrics"]; required={"sources":expected,"record_complete":expected,"all_sections_accepted":expected,"candidate_structurally_valid":expected,"pending_human":0,"formal_training_eligible":expected,"task_canonicalization_drift":0,"probe_tie_loss":0,"preview_training_leak":0,"formal_ineligible_artifact_leak":0}; assert audit.get("passed") is True, audit; assert complete.get("passed") is True, complete; assert review.get("formal_release_ready") is True, review; assert all(metrics.get(key)==value for key,value in required.items()), (metrics,required); rows=sum(1 for line in pathlib.Path(os.environ["FORMAL_VIEWS"]).open(encoding="utf-8") if line.strip()); assert rows==expected,(rows,expected)'
  echo "[smoke-gate-ok] root=${smoke_root}"
}

launch_job() {
  local action="$1"
  local session="$2"
  local input_jsonl="$3"
  local run_root="$4"
  local expected_rows="$5"
  local command
  local -a launch
  prepare_run_dirs "${run_root}"
  tmux has-session -t "${session}" 2>/dev/null \
    && die "tmux session already exists: ${session}"
  launch=(
    env
    "WORKSPACE_ROOT=${WORKSPACE_ROOT}"
    "ROUTESERVER_ROOT=${ROUTESERVER_ROOT}"
    "PYTHON_BIN=${PYTHON_BIN}"
    "DATASET_ROOT=${DATASET_ROOT}"
    "BEHAVIOR_BASE_URL=${BEHAVIOR_BASE_URL}"
    "BEHAVIOR_MODEL=${BEHAVIOR_MODEL}"
    "INSTRUCTION_BASE_URL=${INSTRUCTION_BASE_URL}"
    "INSTRUCTION_MODEL=${INSTRUCTION_MODEL}"
    "DIRECTION_PROFILE_ID=${DIRECTION_PROFILE_ID}"
    "CONCURRENCY_PER_SERVICE=${CONCURRENCY_PER_SERVICE}"
    "MAX_RETRIES=${MAX_RETRIES}"
    "IMAGE_DETAIL=${IMAGE_DETAIL}"
    "NAVIGATION_CONTEXT_IMAGE_DETAIL=${NAVIGATION_CONTEXT_IMAGE_DETAIL}"
    "RETRY_FAILED_ON_RESUME=${RETRY_FAILED_ON_RESUME}"
    "LLM_VISION_TIMEOUT_S=${LLM_VISION_TIMEOUT_S}"
    "LLM_VISION_MAX_TOKENS=${LLM_VISION_MAX_TOKENS}"
    "POSTPROCESS_FULL=${POSTPROCESS_FULL}"
    "SMOKE_INTERESTING_JSONLS=${SMOKE_INTERESTING_JSONLS}"
    "SMOKE_RANDOM_SEED=${SMOKE_RANDOM_SEED}"
    "SMOKE_CONSENSUS_MIN_VOTES=${SMOKE_CONSENSUS_MIN_VOTES}"
    "INPUT_JSONL=${input_jsonl}"
    "RUN_ROOT=${run_root}"
    "EXPECTED_ROWS=${expected_rows}"
    bash "${SCRIPT_PATH}" "${action}"
  )
  printf -v command '%q ' "${launch[@]}"
  tmux new-session -d -s "${session}" "exec ${command}"
  echo "[launched] session=${session} root=${run_root}"
  echo "[follow] tmux attach -t ${session}"
}

status_job() {
  local session="$1"
  local run_root="$2"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[tmux] session=${session} state=running"
  else
    echo "[tmux] session=${session} state=not-running"
  fi
  for path in \
    "${run_root}/sources/reasoning_v4_5_sources.jsonl" \
    "${run_root}/annotations/service_0.jsonl" \
    "${run_root}/annotations/service_1.jsonl" \
    "${run_root}/annotations/reasoning_v4_6_annotations.jsonl" \
    "${run_root}/views/reasoning_v4_reconciliation_preview_views.jsonl"; do
    if [[ -f "${path}" ]]; then
      echo "[rows] $(wc -l <"${path}") ${path}"
    fi
  done
  for report in \
    "${run_root}/reports/annotation_service_0.json" \
    "${run_root}/reports/annotation_service_1.json" \
    "${run_root}/reports/annotation_completeness_report.json"; do
    if [[ -f "${report}" ]]; then
      REPORT="${report}" "${PYTHON_BIN}" -c \
        'import json, os; p=json.load(open(os.environ["REPORT"], encoding="utf-8")); print("[report]", os.environ["REPORT"], {k:p.get(k) for k in ("status","input_rows","pending_rows","processed_rows","record_complete_rows","all_sections_accepted_rows","candidate_structurally_valid_rows","pending_human_rows","rate_rows_per_hour","passed") if k in p})'
    fi
  done
  echo "[log] ${run_root}/logs/controller.log"
}

tail_job() {
  local run_root="$1"
  local log="${run_root}/logs/controller.log"
  require_file "${log}"
  tail -n 120 -f "${log}"
}

action="${1:-help}"
case "${action}" in
  preflight)
    preflight
    ;;
  smoke)
    run_pipeline \
      "${SMOKE_INPUT_JSONL:-${SMOKE_INPUT_JSONL_DEFAULT}}" \
      "${SMOKE_RUN_ROOT:-${SMOKE_RUN_ROOT_DEFAULT}}" \
      "${SMOKE_EXPECTED_ROWS}" \
      1 \
      interesting_diverse
    ;;
  launch-smoke)
    preflight
    launch_job \
      run-smoke "${SMOKE_TMUX_SESSION}" \
      "${SMOKE_INPUT_JSONL:-${SMOKE_INPUT_JSONL_DEFAULT}}" \
      "${SMOKE_RUN_ROOT:-${SMOKE_RUN_ROOT_DEFAULT}}" \
      "${SMOKE_EXPECTED_ROWS}"
    ;;
  run-smoke)
    run_pipeline \
      "${INPUT_JSONL:?INPUT_JSONL is required}" \
      "${RUN_ROOT:?RUN_ROOT is required}" \
      "${EXPECTED_ROWS:?EXPECTED_ROWS is required}" \
      1 \
      interesting_diverse
    ;;
  status-smoke)
    status_job \
      "${SMOKE_TMUX_SESSION}" \
      "${SMOKE_RUN_ROOT:-${SMOKE_RUN_ROOT_DEFAULT}}"
    ;;
  tail-smoke)
    tail_job "${SMOKE_RUN_ROOT:-${SMOKE_RUN_ROOT_DEFAULT}}"
    ;;
  launch-full)
    validate_smoke_gate
    preflight
    launch_job \
      run-full "${FULL_TMUX_SESSION}" \
      "${FULL_INPUT_JSONL:-${FULL_INPUT_JSONL_DEFAULT}}" \
      "${FULL_RUN_ROOT:-${FULL_RUN_ROOT_DEFAULT}}" \
      "${FULL_EXPECTED_ROWS}"
    ;;
  run-full)
    run_pipeline \
      "${INPUT_JSONL:?INPUT_JSONL is required}" \
      "${RUN_ROOT:?RUN_ROOT is required}" \
      "${EXPECTED_ROWS:?EXPECTED_ROWS is required}" \
      "${POSTPROCESS_FULL}" \
      sequential
    ;;
  status-full)
    status_job \
      "${FULL_TMUX_SESSION}" \
      "${FULL_RUN_ROOT:-${FULL_RUN_ROOT_DEFAULT}}"
    ;;
  tail-full)
    tail_job "${FULL_RUN_ROOT:-${FULL_RUN_ROOT_DEFAULT}}"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
