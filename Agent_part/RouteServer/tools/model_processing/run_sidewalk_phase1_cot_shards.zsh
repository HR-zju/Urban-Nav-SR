#!/usr/bin/env zsh
set -euo pipefail

# Launch Phase1 CoT teacher generation over N shards.
#
# Provide keys as environment variables before running:
#   export ARK_KEY_1=...
#   export ARK_KEY_2=...
#   ...
#   export ARK_KEY_8=...
#   ./tools/model_processing/run_sidewalk_phase1_cot_shards.zsh
#
# Each shard writes its own output JSONL and log. The script starts workers with
# nohup and exits after writing a PID file, so it is safe for long runs.

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer}"
OUT="${OUT:-/media/ubuntu/data1/SR-Nav Dataset/sidewalk_phase1_v4}"
SPLIT="${SPLIT:-train_v4}"
SHARD_COUNT="${SHARD_COUNT:-8}"
KEY_COUNT="${KEY_COUNT:-${SHARD_COUNT}}"

SOURCE="${SOURCE:-${OUT}/manifests/phase1_source_${SPLIT}.jsonl}"
SHARD_DIR="${SHARD_DIR:-${OUT}/manifests/cot_shards_${SPLIT}}"
LOG_DIR="${LOG_DIR:-/tmp}"
PID_FILE="${PID_FILE:-${SHARD_DIR}/cot_${SPLIT}.pids}"

MODEL_PROVIDER="${MODEL_PROVIDER:-ark}"
IMAGE_DETAIL="${IMAGE_DETAIL:-auto}"
MAX_RETRIES="${MAX_RETRIES:-2}"
FALLBACK_POLICY="${FALLBACK_POLICY:-template}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/UrbanNav/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  print -u2 "python not executable: ${PYTHON_BIN}"
  print -u2 "Set PYTHON_BIN=/path/to/python if your UrbanNav environment is elsewhere."
  exit 1
fi

if [[ ! -f "${SOURCE}" ]]; then
  print -u2 "source JSONL not found: ${SOURCE}"
  exit 1
fi

if (( SHARD_COUNT < 1 )); then
  print -u2 "SHARD_COUNT must be >= 1"
  exit 1
fi
if (( KEY_COUNT < 1 )); then
  print -u2 "KEY_COUNT must be >= 1"
  exit 1
fi

mkdir -p "${SHARD_DIR}"

print "[cot-shards] source: ${SOURCE}"
print "[cot-shards] shard_dir: ${SHARD_DIR}"
print "[cot-shards] shard_count: ${SHARD_COUNT}"
print "[cot-shards] key_count: ${KEY_COUNT}"

split -d -n "l/${SHARD_COUNT}" "${SOURCE}" "${SHARD_DIR}/source_"

: > "${PID_FILE}"

for idx in {1..${SHARD_COUNT}}; do
  shard_num=$((idx - 1))
  key_idx=$(( (shard_num % KEY_COUNT) + 1 ))
  key_var="ARK_KEY_${key_idx}"
  key="${(P)key_var}"
  if [[ -z "${key}" ]]; then
    print -u2 "missing ${key_var}; export ${key_var}=... before running"
    exit 1
  fi
  if [[ "${key}" == key_* || "${key}" == "你的_key"* ]]; then
    print -u2 "${key_var} still looks like a placeholder"
    exit 1
  fi

  shard="$(printf "%s/source_%02d" "${SHARD_DIR}" "${shard_num}")"
  output_jsonl="$(printf "%s/phase1_cot_%s_part_%d.jsonl" "${SHARD_DIR}" "${SPLIT}" "${shard_num}")"
  report_json="$(printf "%s/phase1_cot_report_%s_part_%d.json" "${SHARD_DIR}" "${SPLIT}" "${shard_num}")"
  log="$(printf "%s/sidewalk_cot_%s_shard_%02d.log" "${LOG_DIR}" "${SPLIT}" "${shard_num}")"

  if [[ ! -s "${shard}" ]]; then
    print -u2 "empty shard file: ${shard}"
    exit 1
  fi

  print "[cot-shards] starting shard ${shard_num} with ${key_var}: ${log}"
  nohup env \
    LLM_API_KEY= \
    LLM_PROVIDER="${MODEL_PROVIDER}" \
    ARK_API_KEY="${key}" \
    FALLBACK_POLICY="${FALLBACK_POLICY}" \
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/model_processing/sidewalk_phase1_cot_teacher.py" \
      --source-jsonl "${shard}" \
      --output-jsonl "${output_jsonl}" \
      --report-json "${report_json}" \
      --expected-points 16 \
      --max-retries "${MAX_RETRIES}" \
      --trajectory-tolerance 0.025 \
      --theta-tolerance-deg 2.0 \
      --image-detail "${IMAGE_DETAIL}" \
      --fallback-policy "${FALLBACK_POLICY}" \
    > "${log}" 2>&1 &

  pid=$!
  print "${pid} ${shard_num} ${log}" >> "${PID_FILE}"
done

print
print "[cot-shards] launched. pid_file: ${PID_FILE}"
print "[cot-shards] monitor:"
print "  tail -f ${LOG_DIR}/sidewalk_cot_${SPLIT}_shard_00.log"
print "  ps -fp \$(awk '{print \$1}' '${PID_FILE}' | tr '\\n' ',')"
