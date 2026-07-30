#!/usr/bin/env bash
# Run paired all-visible versus section-routed Reasoning V4 annotation.

set -euo pipefail

ROUTESERVER_ROOT="${ROUTESERVER_ROOT:?set ROUTESERVER_ROOT}"
PYTHON_BIN="${PYTHON_BIN:?set PYTHON_BIN}"
SOURCE_JSONL="${SOURCE_JSONL:?set SOURCE_JSONL}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT}"
BASELINE_BASE_URL="${BASELINE_BASE_URL:-http://127.0.0.1:18461/v1}"
BASELINE_MODEL="${BASELINE_MODEL:-qwen3.5-122b-a10b-behavior}"
ROUTED_BASE_URL="${ROUTED_BASE_URL:-http://127.0.0.1:18462/v1}"
ROUTED_MODEL="${ROUTED_MODEL:-qwen3.5-122b-a10b-instruction}"
CONCURRENCY_PER_ARM="${CONCURRENCY_PER_ARM:-8}"
MAX_RETRIES="${MAX_RETRIES:-2}"
EXPECTED_ROWS="${EXPECTED_ROWS:-256}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-10}"
IMAGE_DETAIL="${IMAGE_DETAIL:-low}"
NAVIGATION_CONTEXT_IMAGE_DETAIL="${NAVIGATION_CONTEXT_IMAGE_DETAIL:-high}"
MINIMUM_THROUGHPUT_GAIN="${MINIMUM_THROUGHPUT_GAIN:-0.10}"

BASELINE_POLICY="reasoning_v4_section_images_all_visible_v1"
ROUTED_POLICY="reasoning_v4_section_images_routed_visual_v1"
BASELINE_ROOT="${RUN_ROOT}/all_visible"
ROUTED_ROOT="${RUN_ROOT}/section_routed_visual"

for path in \
  "${SOURCE_JSONL}" \
  "${ROUTESERVER_ROOT}/tools/model_processing/label_reasoning_v4.py" \
  "${ROUTESERVER_ROOT}/tools/model_processing/compare_reasoning_v4_image_policy_runs.py"
do
  if [[ ! -f "${path}" ]]; then
    echo "[error] missing file: ${path}" >&2
    exit 2
  fi
done

mkdir -p \
  "${RUN_ROOT}/reports" \
  "${RUN_ROOT}/logs" \
  "${BASELINE_ROOT}/annotations" \
  "${BASELINE_ROOT}/reports" \
  "${BASELINE_ROOT}/review" \
  "${ROUTED_ROOT}/annotations" \
  "${ROUTED_ROOT}/reports" \
  "${ROUTED_ROOT}/review"

PYTHONPATH="${ROUTESERVER_ROOT}/src:${ROUTESERVER_ROOT}" \
  "${PYTHON_BIN}" - "${SOURCE_JSONL}" "${EXPECTED_ROWS}" <<'PY'
import json
import sys

source_path = sys.argv[1]
expected = int(sys.argv[2])
ids = []
with open(source_path, encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        ids.append(row["observation"]["source_group_id"])
if len(ids) != expected or len(set(ids)) != expected:
    raise SystemExit(
        f"paired source gate failed: rows={len(ids)} unique={len(set(ids))} "
        f"expected={expected}"
    )
print(f"[source-ok] rows={len(ids)} unique={len(set(ids))}")
PY

curl -fsS --max-time 5 "${BASELINE_BASE_URL}/models" >/dev/null
curl -fsS --max-time 5 "${ROUTED_BASE_URL}/models" >/dev/null

PYTHONPATH="${ROUTESERVER_ROOT}/src:${ROUTESERVER_ROOT}" \
  "${PYTHON_BIN}" - \
  "${RUN_ROOT}/reports/run_provenance.json" \
  "${SOURCE_JSONL}" \
  "${BASELINE_BASE_URL}" \
  "${BASELINE_MODEL}" \
  "${ROUTED_BASE_URL}" \
  "${ROUTED_MODEL}" \
  "${CONCURRENCY_PER_ARM}" \
  "${BASELINE_POLICY}" \
  "${ROUTED_POLICY}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from mcp_server.teacher.reasoning_v4_image_policy import (
    get_reasoning_v4_section_image_policy,
)

(
    output_path,
    source_path,
    baseline_url,
    baseline_model,
    routed_url,
    routed_model,
    concurrency,
    baseline_policy_id,
    routed_policy_id,
) = sys.argv[1:]
policies = {
    name: get_reasoning_v4_section_image_policy(policy_id)
    for name, policy_id in (
        ("baseline", baseline_policy_id),
        ("routed", routed_policy_id),
    )
}
payload = {
    "schema_version": "urbannav_reasoning_v4_image_policy_ab_run_v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "source_jsonl": source_path,
    "source_sha256": hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
    "baseline": {
        "base_url": baseline_url,
        "model": baseline_model,
        "policy_id": policies["baseline"].policy_id,
        "policy_digest": policies["baseline"].policy_digest,
        "selectors": {
            name: list(values)
            for name, values in policies["baseline"].selectors.items()
        },
    },
    "routed": {
        "base_url": routed_url,
        "model": routed_model,
        "policy_id": policies["routed"].policy_id,
        "policy_digest": policies["routed"].policy_digest,
        "selectors": {
            name: list(values)
            for name, values in policies["routed"].selectors.items()
        },
    },
    "concurrency_per_arm": int(concurrency),
}
Path(output_path).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

run_arm() {
  local arm_name="$1"
  local arm_root="$2"
  local base_url="$3"
  local model="$4"
  local policy_id="$5"
  local log_path="${RUN_ROOT}/logs/${arm_name}.log"

  echo "[arm-start] ${arm_name} policy=${policy_id} model=${model}"
  (
    set -o pipefail
    NO_PROXY=127.0.0.1,localhost \
    no_proxy=127.0.0.1,localhost \
    LLM_PROVIDER=openai \
    LLM_API_STYLE=chat_completions \
    LLM_API_STYLE_FALLBACK=0 \
    LLM_API_KEY=EMPTY \
    LLM_API_BASE_URL="${base_url}" \
    LLM_MODEL="${model}" \
    LLM_TEMPERATURE=0 \
    LLM_MAX_RETRIES=0 \
    LLM_CHAT_TEMPLATE_ENABLE_THINKING=false \
    LLM_RESPONSE_FORMAT_JSON=true \
    LLM_VISION_RESPONSE_FORMAT_JSON=1 \
    LLM_VISION_TIMEOUT_S=900 \
    LLM_VISION_MAX_TOKENS=4096 \
    PYTHONPATH="${ROUTESERVER_ROOT}/src:${ROUTESERVER_ROOT}" \
      "${PYTHON_BIN}" \
      "${ROUTESERVER_ROOT}/tools/model_processing/label_reasoning_v4.py" \
      --source-jsonl "${SOURCE_JSONL}" \
      --annotation-jsonl \
      "${arm_root}/annotations/reasoning_v4_annotations.jsonl" \
      --rejected-jsonl \
      "${arm_root}/annotations/reasoning_v4_annotations.rejected.jsonl" \
      --report-json "${arm_root}/reports/annotation_report.json" \
      --concurrency "${CONCURRENCY_PER_ARM}" \
      --max-retries "${MAX_RETRIES}" \
      --image-detail "${IMAGE_DETAIL}" \
      --navigation-context-image-detail \
      "${NAVIGATION_CONTEXT_IMAGE_DETAIL}" \
      --section-image-policy-id "${policy_id}" \
      --progress-interval "${PROGRESS_INTERVAL}" \
      2>&1 | tee -a "${log_path}"
  )
}

run_arm \
  "baseline_all_visible" \
  "${BASELINE_ROOT}" \
  "${BASELINE_BASE_URL}" \
  "${BASELINE_MODEL}" \
  "${BASELINE_POLICY}" &
baseline_pid=$!

run_arm \
  "routed_visual" \
  "${ROUTED_ROOT}" \
  "${ROUTED_BASE_URL}" \
  "${ROUTED_MODEL}" \
  "${ROUTED_POLICY}" &
routed_pid=$!

set +e
wait "${baseline_pid}"
baseline_status=$?
wait "${routed_pid}"
routed_status=$?
set -e
if [[ "${baseline_status}" -ne 0 || "${routed_status}" -ne 0 ]]; then
  echo "[error] arm failures baseline=${baseline_status} routed=${routed_status}" >&2
  exit 3
fi

PYTHONPATH="${ROUTESERVER_ROOT}/src:${ROUTESERVER_ROOT}" \
  "${PYTHON_BIN}" \
  "${ROUTESERVER_ROOT}/tools/model_processing/compare_reasoning_v4_image_policy_runs.py" \
  --source-jsonl "${SOURCE_JSONL}" \
  --baseline-annotation-jsonl \
  "${BASELINE_ROOT}/annotations/reasoning_v4_annotations.jsonl" \
  --routed-annotation-jsonl \
  "${ROUTED_ROOT}/annotations/reasoning_v4_annotations.jsonl" \
  --baseline-report-json "${BASELINE_ROOT}/reports/annotation_report.json" \
  --routed-report-json "${ROUTED_ROOT}/reports/annotation_report.json" \
  --output-json "${RUN_ROOT}/reports/paired_comparison.json" \
  --expected-rows "${EXPECTED_ROWS}" \
  --minimum-throughput-gain "${MINIMUM_THROUGHPUT_GAIN}"

for specification in \
  "all_visible:${BASELINE_ROOT}" \
  "section_routed_visual:${ROUTED_ROOT}"
do
  arm_name="${specification%%:*}"
  arm_root="${specification#*:}"
  PYTHONPATH="${ROUTESERVER_ROOT}/src:${ROUTESERVER_ROOT}" \
    "${PYTHON_BIN}" \
    "${ROUTESERVER_ROOT}/tools/model_processing/build_reasoning_v4_review_html.py" \
    --source-jsonl "${SOURCE_JSONL}" \
    --annotation-jsonl \
    "${arm_root}/annotations/reasoning_v4_annotations.jsonl" \
    --output-html "${arm_root}/review/reasoning_v4_review.linked.html" \
    --report-json "${arm_root}/reports/review_report.json" \
    --title "Reasoning V4 Image Policy A/B: ${arm_name}" \
    --max-samples 64 \
    --link-images
done

echo "[done] paired comparison: ${RUN_ROOT}/reports/paired_comparison.json"
