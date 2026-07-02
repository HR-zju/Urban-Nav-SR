#!/usr/bin/env zsh
set -euo pipefail

# Run low-concurrency UrbanNav Module 1 proxy smoke tests for a list of
# OpenAI-compatible models.  Each model is evaluated on the same first N cases
# of the final 15K benchmark split and written to an isolated run directory.
#
# Usage:
#   tools/evaluation/run_urbannav_proxy_smoke_models.zsh start
#   tools/evaluation/run_urbannav_proxy_smoke_models.zsh status
#   tools/evaluation/run_urbannav_proxy_smoke_models.zsh tail
#   tools/evaluation/run_urbannav_proxy_smoke_models.zsh stop
#
# Environment overrides:
#   MODELS="qwen3.7-max deepseek-v4-pro glm-5.2 kimi-k2.6"
#   MAX_CASES=200 CONCURRENCY=2 LLM_BASE_URL=http://127.0.0.1:18088/v1

export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"

SCRIPT_PATH="${0:A}"
ROUTE_ROOT="${ROUTE_ROOT:-${SCRIPT_PATH:h:h:h}}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${ROUTE_ROOT}/../.." && pwd)}"

ACTION="${1:-start}"
SESSION="${SESSION:-urbannav_proxy_smoke_models}"
RUN_GROUP="${RUN_GROUP:-proxy_strict_200_smoke_20260701}"
RUN_BASE="${RUN_BASE:-${ROUTE_ROOT}/data/benchmark/urbannav_module1_runs/${RUN_GROUP}}"

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/UrbanNav/bin/python}"
KEY_FILE="${KEY_FILE:-${ROUTE_ROOT}/data/benchmark/urbannav_module1_runs/local_proxy_dummy_key.txt}"
DATASET="${DATASET:-${ROUTE_ROOT}/data/benchmark/urbannav_v2_osm_local_osrm_foot_final_15k/task_gt.jsonl}"
POI_CACHE="${POI_CACHE:-${ROUTE_ROOT}/data/benchmark/urbannav_v2_osm_local_osrm_foot_final_15k/poi_cache.jsonl}"
ROUTE_CACHE="${ROUTE_CACHE:-${ROUTE_ROOT}/data/benchmark/urbannav_v2_osm_local_osrm_foot_final_15k/route_cache.jsonl}"

MODELS="${MODELS:-qwen3.7-max deepseek-v4-pro glm-5.2 kimi-k2.6}"
MAX_CASES="${MAX_CASES:-200}"
CONCURRENCY="${CONCURRENCY:-2}"
CANDIDATE_LIMIT="${CANDIDATE_LIMIT:-10}"
SELECTION_STRATEGY="${SELECTION_STRATEGY:-semantic_then_route}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-20}"
HEARTBEAT_INTERVAL_S="${HEARTBEAT_INTERVAL_S:-20}"
CHUNK_SIZE="${CHUNK_SIZE:-20}"
LLM_PROVIDER_NAME="${LLM_PROVIDER_NAME:-openai}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:18088/v1}"
LLM_TIMEOUT_S="${LLM_TIMEOUT_S:-90}"
LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-1}"
RESTART="${RESTART:-0}"

GROUP_LOG="${RUN_BASE}/proxy_smoke.log"
WORKER_SCRIPT="${RUN_BASE}/run_proxy_smoke_models_worker.zsh"

_safe_model_name() {
  local model="$1"
  print "${model//[^A-Za-z0-9]/_}"
}

_print_config() {
  print "route_root=${ROUTE_ROOT}"
  print "session=${SESSION}"
  print "run_base=${RUN_BASE}"
  print "python=${PYTHON_BIN}"
  print "dataset=${DATASET}"
  print "models=${MODELS}"
  print "max_cases=${MAX_CASES}"
  print "concurrency=${CONCURRENCY}"
  print "llm_base_url=${LLM_BASE_URL}"
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

_write_worker_script() {
  mkdir -p "${RUN_BASE}"
  cat > "${WORKER_SCRIPT}" <<'EOF'
#!/usr/bin/env zsh
set -euo pipefail

cd "__ROUTE_ROOT__"

export LLM_RESPONSE_FORMAT_JSON="${LLM_RESPONSE_FORMAT_JSON:-0}"
export LLM_API_STYLE_FALLBACK="${LLM_API_STYLE_FALLBACK:-1}"

print "[proxy-smoke] started at $(date '+%Y-%m-%d %H:%M:%S')" 2>&1 | tee -a "__GROUP_LOG__"
print "[proxy-smoke] models=__MODELS__" 2>&1 | tee -a "__GROUP_LOG__"

for model in __MODELS__; do
  safe="${model//[^A-Za-z0-9]/_}"
  run_root="__RUN_BASE__/${safe}_strict_${model//[^A-Za-z0-9]/_}_200"
  pred_jsonl="${run_root}/raw_predictions.jsonl"
  pred_manifest="${run_root}/prediction_manifest.json"
  pred_log="${run_root}/predict.log"
  eval_dir="${run_root}/eval"
  eval_log="${run_root}/eval.log"
  mkdir -p "${run_root}"
  : > "${pred_log}"

  {
    print "[proxy-smoke] model=${model} started at $(date '+%Y-%m-%d %H:%M:%S')"
    print "[proxy-smoke] run_root=${run_root}"
  } 2>&1 | tee -a "__GROUP_LOG__" | tee -a "${pred_log}"

  LLM_RESPONSE_FORMAT_JSON=0 LLM_API_STYLE_FALLBACK=1 PYTHONUNBUFFERED=1 PYTHONPATH=src "__PYTHON_BIN__" \
    tools/evaluation/generate_urbannav_external_predictions.py \
    --cases "__DATASET__" \
    --max-cases "__MAX_CASES__" \
    --backend replay \
    --poi-cache "__POI_CACHE__" \
    --route-cache "__ROUTE_CACHE__" \
    --concurrency "__CONCURRENCY__" \
    --candidate-limit "__CANDIDATE_LIMIT__" \
    --selection-strategy "__SELECTION_STRATEGY__" \
    --use-llm-intent \
    --use-llm-reranker \
    --strict-llm \
    --api-keys-file "__KEY_FILE__" \
    --llm-provider "__LLM_PROVIDER_NAME__" \
    --llm-model "${model}" \
    --llm-base-url "__LLM_BASE_URL__" \
    --llm-timeout-s "__LLM_TIMEOUT_S__" \
    --llm-max-retries "__LLM_MAX_RETRIES__" \
    --progress-interval "__PROGRESS_INTERVAL__" \
    --heartbeat-interval-s "__HEARTBEAT_INTERVAL_S__" \
    --chunk-size "__CHUNK_SIZE__" \
    --output-jsonl "${pred_jsonl}" \
    --manifest-json "${pred_manifest}" \
    2>&1 | tee -a "${pred_log}"

  PYTHONPATH=src "__PYTHON_BIN__" tools/evaluation/run_urbannav_module1_benchmark.py \
    --dataset "__DATASET__" \
    --runner external \
    --predictions "${pred_jsonl}" \
    --output-dir "${eval_dir}" \
    --max-cases "__MAX_CASES__" \
    --run-id "${safe}-proxy-strict-200-smoke" \
    2>&1 | tee "${eval_log}"

  "__PYTHON_BIN__" -c '
import json
from pathlib import Path
pred=Path("'"${pred_jsonl}"'")
rows=[json.loads(line) for line in pred.open()] if pred.exists() else []
non=sum(bool(row.get("chosen_destination") and row.get("route_polyline")) for row in rows)
summary_path=Path("'"${eval_dir}"'") / "summary.json"
summary=json.load(summary_path.open()).get("summary", {}) if summary_path.exists() else {}
print(json.dumps({
    "model": "'"${model}"'",
    "rows": len(rows),
    "non_empty": non,
    "non_empty_rate": round(non / len(rows), 4) if rows else 0.0,
    "destination_success": summary.get("destination_success"),
    "spl": summary.get("spl"),
    "prediction_exception_rate": summary.get("prediction_exception_rate"),
}, ensure_ascii=False, sort_keys=True))
' 2>&1 | tee -a "__GROUP_LOG__"

  print "[proxy-smoke] model=${model} finished at $(date '+%Y-%m-%d %H:%M:%S')" 2>&1 | tee -a "__GROUP_LOG__"
done

print "[proxy-smoke] all models finished at $(date '+%Y-%m-%d %H:%M:%S')" 2>&1 | tee -a "__GROUP_LOG__"
EOF

  perl -0pi -e "s#__ROUTE_ROOT__#${ROUTE_ROOT}#g; s#__GROUP_LOG__#${GROUP_LOG}#g; s#__MODELS__#${MODELS}#g; s#__RUN_BASE__#${RUN_BASE}#g; s#__PYTHON_BIN__#${PYTHON_BIN}#g; s#__DATASET__#${DATASET}#g; s#__POI_CACHE__#${POI_CACHE}#g; s#__ROUTE_CACHE__#${ROUTE_CACHE}#g; s#__MAX_CASES__#${MAX_CASES}#g; s#__CONCURRENCY__#${CONCURRENCY}#g; s#__CANDIDATE_LIMIT__#${CANDIDATE_LIMIT}#g; s#__SELECTION_STRATEGY__#${SELECTION_STRATEGY}#g; s#__KEY_FILE__#${KEY_FILE}#g; s#__LLM_PROVIDER_NAME__#${LLM_PROVIDER_NAME}#g; s#__LLM_BASE_URL__#${LLM_BASE_URL}#g; s#__LLM_TIMEOUT_S__#${LLM_TIMEOUT_S}#g; s#__LLM_MAX_RETRIES__#${LLM_MAX_RETRIES}#g; s#__PROGRESS_INTERVAL__#${PROGRESS_INTERVAL}#g; s#__HEARTBEAT_INTERVAL_S__#${HEARTBEAT_INTERVAL_S}#g; s#__CHUNK_SIZE__#${CHUNK_SIZE}#g" "${WORKER_SCRIPT}"
  chmod +x "${WORKER_SCRIPT}"
}

_status() {
  print "== tmux =="
  tmux ls 2>/dev/null || true
  print "\n== process =="
  ps -eo pid,ppid,stat,etime,cmd | grep -E 'generate_urbannav_external_predictions|run_urbannav_module1_benchmark|run_proxy_smoke_models_worker' | grep -v grep || true
  print "\n== run base =="
  find "${RUN_BASE}" -maxdepth 2 -type f \( -name 'summary.json' -o -name 'raw_predictions.jsonl' -o -name '*.log' \) -print 2>/dev/null | sort || true
  print "\n== group log =="
  tail -40 "${GROUP_LOG}" 2>/dev/null || true
}

case "${ACTION}" in
  dry-run)
    _print_config
    _validate_inputs
    _write_worker_script
    print "worker_script=${WORKER_SCRIPT}"
    ;;
  start)
    _print_config
    _validate_inputs
    _write_worker_script
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
      if [[ "${RESTART}" == "1" ]]; then
        tmux kill-session -t "${SESSION}"
      else
        print -u2 "tmux session already exists: ${SESSION}"
        exit 1
      fi
    fi
    : > "${GROUP_LOG}"
    tmux new-session -d -s "${SESSION}" "zsh '${WORKER_SCRIPT}'"
    print "started tmux session: ${SESSION}"
    print "tail log: tail -f '${GROUP_LOG}'"
    ;;
  status)
    _status
    ;;
  tail)
    tail -f "${GROUP_LOG}"
    ;;
  stop)
    tmux kill-session -t "${SESSION}" 2>/dev/null || true
    print "stopped tmux session if it existed: ${SESSION}"
    ;;
  *)
    print -u2 "unknown action: ${ACTION}"
    print -u2 "usage: ${SCRIPT_PATH} {start|status|tail|stop|dry-run}"
    exit 2
    ;;
esac
