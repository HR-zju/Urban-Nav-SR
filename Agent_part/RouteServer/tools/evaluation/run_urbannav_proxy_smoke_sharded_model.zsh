#!/usr/bin/env zsh
set -euo pipefail

# Run one OpenAI-compatible proxy model as multiple shard processes.
#
# Usage:
#   MODEL=glm-5.2 tools/evaluation/run_urbannav_proxy_smoke_sharded_model.zsh start
#   MODEL=glm-5.2 tools/evaluation/run_urbannav_proxy_smoke_sharded_model.zsh status
#   MODEL=glm-5.2 tools/evaluation/run_urbannav_proxy_smoke_sharded_model.zsh merge
#   MODEL=glm-5.2 tools/evaluation/run_urbannav_proxy_smoke_sharded_model.zsh stop
#
# The script writes one JSONL per shard and merges only after all shards finish.

export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"

SCRIPT_PATH="${0:A}"
ROUTE_ROOT="${ROUTE_ROOT:-${SCRIPT_PATH:h:h:h}}"

ACTION="${1:-start}"
MODEL="${MODEL:-glm-5.2}"
SAFE_MODEL="${MODEL//[^A-Za-z0-9]/_}"
RUN_GROUP="${RUN_GROUP:-proxy_strict_200_smoke_20260701}"
RUN_ROOT="${RUN_ROOT:-${ROUTE_ROOT}/data/benchmark/urbannav_module1_runs/${RUN_GROUP}/${SAFE_MODEL}_strict_${SAFE_MODEL}_200_sharded}"
SESSION_PREFIX="${SESSION_PREFIX:-urbannav_proxy_smoke_${SAFE_MODEL}_shard}"

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/UrbanNav/bin/python}"
KEY_FILE="${KEY_FILE:-${ROUTE_ROOT}/data/benchmark/urbannav_module1_runs/local_proxy_dummy_key.txt}"
DATASET="${DATASET:-${ROUTE_ROOT}/data/benchmark/urbannav_v2_osm_local_osrm_foot_final_15k/task_gt.jsonl}"
POI_CACHE="${POI_CACHE:-${ROUTE_ROOT}/data/benchmark/urbannav_v2_osm_local_osrm_foot_final_15k/poi_cache.jsonl}"
ROUTE_CACHE="${ROUTE_CACHE:-${ROUTE_ROOT}/data/benchmark/urbannav_v2_osm_local_osrm_foot_final_15k/route_cache.jsonl}"

MAX_CASES="${MAX_CASES:-200}"
SHARDS="${SHARDS:-4}"
CONCURRENCY="${CONCURRENCY:-2}"
CANDIDATE_LIMIT="${CANDIDATE_LIMIT:-10}"
SELECTION_STRATEGY="${SELECTION_STRATEGY:-semantic_then_route}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-10}"
HEARTBEAT_INTERVAL_S="${HEARTBEAT_INTERVAL_S:-20}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
LLM_PROVIDER_NAME="${LLM_PROVIDER_NAME:-openai}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:18088/v1}"
LLM_TIMEOUT_S="${LLM_TIMEOUT_S:-90}"
LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-1}"
LLM_RESPONSE_FORMAT_JSON="${LLM_RESPONSE_FORMAT_JSON:-1}"
SKIP_LLM_PREFLIGHT="${SKIP_LLM_PREFLIGHT:-0}"
MAX_PREDICTION_EXCEPTION_RATE="${MAX_PREDICTION_EXCEPTION_RATE:-0.1}"
EXCEPTION_RATE_MIN_SAMPLES="${EXCEPTION_RATE_MIN_SAMPLES:-20}"
RESUME="${RESUME:-0}"
STOP_AFTER_PENDING="${STOP_AFTER_PENDING:-}"
RESTART="${RESTART:-0}"

SHARD_DIR="${RUN_ROOT}/shards"
MERGED_DIR="${RUN_ROOT}/merged_eval"
MERGED_CASES="${RUN_ROOT}/cases_merged.jsonl"
MERGED_PREDICTIONS="${RUN_ROOT}/raw_predictions_merged.jsonl"
GROUP_LOG="${RUN_ROOT}/sharded_smoke.log"

_print_config() {
  print "model=${MODEL}"
  print "run_root=${RUN_ROOT}"
  print "dataset=${DATASET}"
  print "max_cases=${MAX_CASES}"
  print "shards=${SHARDS}"
  print "concurrency_per_shard=${CONCURRENCY}"
  print "llm_base_url=${LLM_BASE_URL}"
  print "api_style=/chat/completions"
  print "LLM_API_STYLE_FALLBACK=0"
  print "skip_llm_preflight=${SKIP_LLM_PREFLIGHT}"
  print "max_prediction_exception_rate=${MAX_PREDICTION_EXCEPTION_RATE}"
  print "resume=${RESUME}"
  print "stop_after_pending=${STOP_AFTER_PENDING:-none}"
}

_validate_inputs() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    print -u2 "python not executable: ${PYTHON_BIN}"
    exit 1
  fi
  for required_path in "${KEY_FILE}" "${DATASET}" "${POI_CACHE}" "${ROUTE_CACHE}"; do
    if [[ ! -f "${required_path}" ]]; then
      print -u2 "required file not found: ${required_path}"
      exit 1
    fi
  done
}

_export_local_no_proxy_if_needed() {
  case "${LLM_BASE_URL}" in
    http://127.*|https://127.*|http://localhost*|https://localhost*|http://[::1]*|https://[::1]*)
      export NO_PROXY="127.0.0.1,localhost,::1,${NO_PROXY:-}"
      export no_proxy="${NO_PROXY}"
      unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
      ;;
  esac
}

_prepare_shards() {
  mkdir -p "${SHARD_DIR}"
  "${PYTHON_BIN}" -c '
import json
from pathlib import Path
dataset = Path("'"${DATASET}"'")
out_dir = Path("'"${SHARD_DIR}"'")
merged = Path("'"${MERGED_CASES}"'")
max_cases = int("'"${MAX_CASES}"'")
shards = int("'"${SHARDS}"'")
rows = []
with dataset.open() as f:
    for i, line in enumerate(f):
        if i >= max_cases:
            break
        rows.append(line.rstrip("\n"))
out_dir.mkdir(parents=True, exist_ok=True)
for shard_id in range(shards):
    shard_rows = rows[shard_id::shards]
    path = out_dir / f"cases_shard_{shard_id:02d}.jsonl"
    path.write_text("\n".join(shard_rows) + ("\n" if shard_rows else ""), encoding="utf-8")
merged.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
print(json.dumps({"max_cases": max_cases, "rows": len(rows), "shards": shards}, sort_keys=True))
'
}

_preflight_llm() {
  if [[ "${SKIP_LLM_PREFLIGHT}" == "1" ]]; then
    print "[proxy-preflight] skipped" | tee -a "${GROUP_LOG}"
    return
  fi
  _export_local_no_proxy_if_needed
  mkdir -p "${RUN_ROOT}"
  cd "${ROUTE_ROOT}"
  print "[proxy-preflight] checking model=${MODEL} base_url=${LLM_BASE_URL}" | tee -a "${GROUP_LOG}"
  "${PYTHON_BIN}" - "${LLM_BASE_URL}" "${MODEL}" "${KEY_FILE}" "${LLM_TIMEOUT_S}" <<'PY' 2>&1 | tee -a "${GROUP_LOG}"
import json
import sys
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

base_url, model, key_file, timeout_s = sys.argv[1:5]
key = ""
for line in Path(key_file).read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#"):
        key = line
        break
url = base_url.rstrip("/") + "/chat/completions"
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Only output OK."}],
    "temperature": 0,
    "max_tokens": 8,
}
headers = {"Content-Type": "application/json"}
if key:
    headers["Authorization"] = f"Bearer {key}"
request = urlrequest.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers=headers,
    method="POST",
)
try:
    with urlrequest.urlopen(request, timeout=max(1.0, min(float(timeout_s), 15.0))) as response:
        status = int(response.status)
        body = response.read(4096).decode("utf-8", errors="replace")
except urlerror.HTTPError as exc:
    detail = exc.read(1024).decode("utf-8", errors="replace")
    raise SystemExit(f"proxy preflight failed: HTTP {exc.code}: {detail}")
except Exception as exc:
    raise SystemExit(f"proxy preflight failed: {type(exc).__name__}: {exc}")
if not 200 <= status < 300:
    raise SystemExit(f"proxy preflight failed: HTTP {status}: {body[:200]}")
if not body.strip():
    raise SystemExit("proxy preflight failed: empty response")
print(json.dumps({"event": "proxy_preflight_ok", "model": model, "status": status}, sort_keys=True))
PY
}

_write_worker_script() {
  local shard_id="$1"
  local session_name="${SESSION_PREFIX}_${shard_id}"
  local shard_case="${SHARD_DIR}/cases_shard_${shard_id}.jsonl"
  local shard_root="${SHARD_DIR}/shard_${shard_id}"
  local pred_jsonl="${shard_root}/raw_predictions.jsonl"
  local pred_manifest="${shard_root}/prediction_manifest.json"
  local pred_log="${shard_root}/predict.log"
  local eval_dir="${shard_root}/eval"
  local eval_log="${shard_root}/eval.log"
  local worker_script="${shard_root}/run_worker.zsh"
  mkdir -p "${shard_root}"
  cat > "${worker_script}" <<EOF
#!/usr/bin/env zsh
set -euo pipefail
cd "${ROUTE_ROOT}"
export LLM_RESPONSE_FORMAT_JSON="${LLM_RESPONSE_FORMAT_JSON}"
export LLM_API_STYLE_FALLBACK=0
case "${LLM_BASE_URL}" in
  http://127.*|https://127.*|http://localhost*|https://localhost*|http://[::1]*|https://[::1]*)
    export NO_PROXY="127.0.0.1,localhost,::1,\${NO_PROXY:-}"
    export no_proxy="\${NO_PROXY}"
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
    ;;
esac

extra_prediction_args=()
if [[ "${RESUME}" == "1" ]]; then
  extra_prediction_args+=(--resume)
fi
if [[ -n "${STOP_AFTER_PENDING}" ]]; then
  extra_prediction_args+=(--stop-after-pending "${STOP_AFTER_PENDING}")
fi

{
  print "[proxy-shard] model=${MODEL} shard=${shard_id} started at \$(date '+%Y-%m-%d %H:%M:%S')"
  print "[proxy-shard] shard_case=${shard_case}"
  print "[proxy-shard] concurrency=${CONCURRENCY}"
} 2>&1 | tee -a "${pred_log}" | tee -a "${GROUP_LOG}"

LLM_RESPONSE_FORMAT_JSON="${LLM_RESPONSE_FORMAT_JSON}" LLM_API_STYLE_FALLBACK=0 PYTHONUNBUFFERED=1 PYTHONPATH=src "${PYTHON_BIN}" \\
  tools/evaluation/generate_urbannav_external_predictions.py \\
  --cases "${shard_case}" \\
  --backend replay \\
  --poi-cache "${POI_CACHE}" \\
  --route-cache "${ROUTE_CACHE}" \\
  --concurrency "${CONCURRENCY}" \\
  --candidate-limit "${CANDIDATE_LIMIT}" \\
  --selection-strategy "${SELECTION_STRATEGY}" \\
  --use-llm-intent \\
  --use-llm-reranker \\
  --strict-llm \\
  --api-keys-file "${KEY_FILE}" \\
  --llm-provider "${LLM_PROVIDER_NAME}" \\
  --llm-model "${MODEL}" \\
  --llm-base-url "${LLM_BASE_URL}" \\
  --llm-timeout-s "${LLM_TIMEOUT_S}" \\
  --llm-max-retries "${LLM_MAX_RETRIES}" \\
  --progress-interval "${PROGRESS_INTERVAL}" \\
  --heartbeat-interval-s "${HEARTBEAT_INTERVAL_S}" \\
  --chunk-size "${CHUNK_SIZE}" \\
  --max-prediction-exception-rate "${MAX_PREDICTION_EXCEPTION_RATE}" \\
  --exception-rate-min-samples "${EXCEPTION_RATE_MIN_SAMPLES}" \\
  "\${extra_prediction_args[@]}" \\
  --output-jsonl "${pred_jsonl}" \\
  --manifest-json "${pred_manifest}" \\
  2>&1 | tee -a "${pred_log}"

PYTHONPATH=src "${PYTHON_BIN}" tools/evaluation/run_urbannav_module1_benchmark.py \\
  --dataset "${shard_case}" \\
  --runner external \\
  --predictions "${pred_jsonl}" \\
  --output-dir "${eval_dir}" \\
  --run-id "${SAFE_MODEL}-proxy-strict-200-shard-${shard_id}" \\
  2>&1 | tee "${eval_log}"

print "[proxy-shard] model=${MODEL} shard=${shard_id} finished at \$(date '+%Y-%m-%d %H:%M:%S')" 2>&1 | tee -a "${pred_log}" | tee -a "${GROUP_LOG}"
EOF
  chmod +x "${worker_script}"
  print "${session_name}|${worker_script}"
}

_start() {
  _print_config
  _validate_inputs
  mkdir -p "${RUN_ROOT}"
  : > "${GROUP_LOG}"
  _preflight_llm
  _prepare_shards | tee -a "${GROUP_LOG}"
  local shard_count=$((SHARDS))
  for (( shard_id=0; shard_id<shard_count; shard_id++ )); do
    local shard_name
    shard_name=$(printf "%02d" "${shard_id}")
    local pair session_name worker_script
    pair=$(_write_worker_script "${shard_name}")
    session_name="${pair%%|*}"
    worker_script="${pair#*|}"
    if tmux has-session -t "${session_name}" 2>/dev/null; then
      if [[ "${RESTART}" == "1" ]]; then
        tmux kill-session -t "${session_name}"
      else
        print -u2 "tmux session already exists: ${session_name}"
        exit 1
      fi
    fi
    tmux new-session -d -s "${session_name}" "zsh '${worker_script}'"
    print "started ${session_name}" | tee -a "${GROUP_LOG}"
  done
}

_stop() {
  for (( shard_id=0; shard_id<SHARDS; shard_id++ )); do
    local shard_name
    shard_name=$(printf "%02d" "${shard_id}")
    tmux kill-session -t "${SESSION_PREFIX}_${shard_name}" 2>/dev/null || true
  done
  print "stopped shard sessions for ${MODEL}"
}

_merge() {
  mkdir -p "${MERGED_DIR}"
  : > "${MERGED_PREDICTIONS}"
  for (( shard_id=0; shard_id<SHARDS; shard_id++ )); do
    local shard_name pred_jsonl
    shard_name=$(printf "%02d" "${shard_id}")
    pred_jsonl="${SHARD_DIR}/shard_${shard_name}/raw_predictions.jsonl"
    if [[ -f "${pred_jsonl}" ]]; then
      cat "${pred_jsonl}" >> "${MERGED_PREDICTIONS}"
    fi
  done
  "${PYTHON_BIN}" - "${MERGED_PREDICTIONS}" "${MAX_PREDICTION_EXCEPTION_RATE}" "${EXCEPTION_RATE_MIN_SAMPLES}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
max_rate = float(sys.argv[2])
min_samples = int(sys.argv[3])
rows = 0
exceptions = 0
if path.exists():
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                exceptions += 1
                continue
            trace = payload.get("trace") if isinstance(payload.get("trace"), dict) else {}
            if trace.get("error") == "prediction_exception":
                exceptions += 1
rate = exceptions / max(rows, 1)
print(json.dumps({
    "event": "merged_prediction_exception_rate_check",
    "rows": rows,
    "prediction_exception_count": exceptions,
    "prediction_exception_rate": round(rate, 6),
    "max_prediction_exception_rate": max_rate,
}, sort_keys=True))
if rows >= min_samples and rate > max_rate:
    raise SystemExit(
        "merged prediction_exception_rate exceeded threshold; "
        "run is invalid and evaluator will not be executed"
    )
PY
  cd "${ROUTE_ROOT}"
  PYTHONPATH=src "${PYTHON_BIN}" tools/evaluation/run_urbannav_module1_benchmark.py \
    --dataset "${MERGED_CASES}" \
    --runner external \
    --predictions "${MERGED_PREDICTIONS}" \
    --output-dir "${MERGED_DIR}" \
    --run-id "${SAFE_MODEL}-proxy-strict-200-sharded-merged"
}

_status() {
  print "== tmux =="
  tmux ls 2>/dev/null | grep "${SESSION_PREFIX}" || true
  print "\n== shards =="
  for (( shard_id=0; shard_id<SHARDS; shard_id++ )); do
    local shard_name shard_root pred_jsonl pred_log
    shard_name=$(printf "%02d" "${shard_id}")
    shard_root="${SHARD_DIR}/shard_${shard_name}"
    pred_jsonl="${shard_root}/raw_predictions.jsonl"
    pred_log="${shard_root}/predict.log"
    print -- "-- shard ${shard_name}"
    wc -l "${pred_jsonl}" 2>/dev/null || true
    tail -6 "${pred_log}" 2>/dev/null || true
  done
  print "\n== merged =="
  wc -l "${MERGED_PREDICTIONS}" 2>/dev/null || true
  tail -20 "${MERGED_DIR}/summary.json" 2>/dev/null || true
}

case "${ACTION}" in
  start)
    _start
    ;;
  status)
    _status
    ;;
  stop)
    _stop
    ;;
  merge)
    _merge
    ;;
  dry-run)
    _print_config
    _validate_inputs
    _prepare_shards
    ;;
  *)
    print -u2 "unknown action: ${ACTION}"
    print -u2 "usage: ${SCRIPT_PATH} {start|status|stop|merge|dry-run}"
    exit 2
    ;;
esac
