#!/usr/bin/env zsh
set -euo pipefail

# Launch Phase1 CoT teacher generation over N shards.
#
# Provide keys with KEY_FILE, one key per line:
#   KEY_FILE=/home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/key.txt \
#   ./tools/model_processing/run_sidewalk_phase1_cot_shards.zsh
#
# Each shard writes its own output JSONL and log. The script starts workers with
# nohup and exits after writing a PID file, so it is safe for long runs.

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${REPO_ROOT}/../.." && pwd)}"
OUT="${OUT:-${OUTPUT_ROOT:-/media/ubuntu/data1/SR-Nav Dataset/sidewalk_phase1_v4}}"
SPLIT="${SPLIT:-${SPLIT_NAME:-train_v4}}"
SHARD_COUNT="${SHARD_COUNT:-8}"
KEY_COUNT="${KEY_COUNT:-${SHARD_COUNT}}"
KEY_FILE="${KEY_FILE:-${PROJECT_ROOT}/key.txt}"

SOURCE="${SOURCE:-${SOURCE_JSONL:-${OUT}/manifests/phase1_source_${SPLIT}.jsonl}}"
SHARD_DIR="${SHARD_DIR:-${OUT}/manifests/cot_shards_${SPLIT}}"
LOG_DIR="${LOG_DIR:-/tmp}"
PID_FILE="${PID_FILE:-${SHARD_DIR}/cot_${SPLIT}.pids}"

MODEL_PROVIDER="${MODEL_PROVIDER:-ark}"
IMAGE_DETAIL="${IMAGE_DETAIL:-auto}"
MAX_RETRIES="${MAX_RETRIES:-2}"
FALLBACK_POLICY="${FALLBACK_POLICY:-template}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-25}"
LLM_MODEL_NAME="${LLM_MODEL:-${MODEL_NAME:-doubao-seed-2-0-pro-260215}}"
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
if [[ ! -f "${KEY_FILE}" ]]; then
  print -u2 "KEY_FILE not found: ${KEY_FILE}"
  exit 1
fi

keys=()
while IFS= read -r line; do
  key="${line#"${line%%[![:space:]]*}"}"
  key="${key%"${key##*[![:space:]]}"}"
  if [[ -z "${key}" || "${key}" == \#* ]]; then
    continue
  fi
  keys+=("${key}")
done < "${KEY_FILE}"

if (( ${#keys[@]} < 1 )); then
  print -u2 "no API keys loaded from KEY_FILE: ${KEY_FILE}"
  exit 1
fi
if (( KEY_COUNT > ${#keys[@]} )); then
  KEY_COUNT="${#keys[@]}"
fi

mkdir -p "${SHARD_DIR}"

print "[cot-shards] source: ${SOURCE}"
print "[cot-shards] shard_dir: ${SHARD_DIR}"
print "[cot-shards] shard_count: ${SHARD_COUNT}"
print "[cot-shards] key_count: ${KEY_COUNT}"
print "[cot-shards] key_file: ${KEY_FILE}"
print "[cot-shards] model: ${LLM_MODEL_NAME}"
print "[cot-shards] progress_interval: ${PROGRESS_INTERVAL}"

split -d -n "l/${SHARD_COUNT}" "${SOURCE}" "${SHARD_DIR}/source_"

: > "${PID_FILE}"

for idx in {1..${SHARD_COUNT}}; do
  shard_num=$((idx - 1))
  key_idx=$(( (shard_num % KEY_COUNT) + 1 ))
  key="${keys[${key_idx}]}"
  if [[ -z "${key}" ]]; then
    print -u2 "missing key index ${key_idx} from ${KEY_FILE}"
    exit 1
  fi
  if [[ "${key}" == key_* || "${key}" == "你的_key"* ]]; then
    print -u2 "key index ${key_idx} still looks like a placeholder"
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

  print "[cot-shards] starting shard ${shard_num} with key_${key_idx}: ${log}"
  : > "${log}"
  nohup zsh -c '
    set -uo pipefail
    shard_num="$1"
    provider="$2"
    model="$3"
    api_key="$4"
    fallback_policy="$5"
    python_bin="$6"
    repo_root="$7"
    progress_interval="$8"
    shard_path="$9"
    output_jsonl="${10}"
    report_json="${11}"
    max_retries="${12}"
    image_detail="${13}"
    print "[cot-shards-worker] start shard=${shard_num} pid=$$ at $(date "+%Y-%m-%d %H:%M:%S")"
    LLM_API_KEY= \
    LLM_PROVIDER="${provider}" \
    LLM_MODEL="${model}" \
    ARK_API_KEY="${api_key}" \
    FALLBACK_POLICY="${fallback_policy}" \
    "${python_bin}" "${repo_root}/tools/model_processing/sidewalk_phase1_cot_teacher.py" \
      --source-jsonl "${shard_path}" \
      --output-jsonl "${output_jsonl}" \
      --report-json "${report_json}" \
      --expected-points 16 \
      --max-retries "${max_retries}" \
      --trajectory-tolerance 0.025 \
      --theta-tolerance-deg 2.0 \
      --image-detail "${image_detail}" \
      --fallback-policy "${fallback_policy}" \
      --progress-interval "${progress_interval}"
    status=$?
    print "[cot-shards-worker] exit shard=${shard_num} status=${status} at $(date "+%Y-%m-%d %H:%M:%S")"
    exit ${status}
  ' -- "${shard_num}" "${MODEL_PROVIDER}" "${LLM_MODEL_NAME}" "${key}" "${FALLBACK_POLICY}" "${PYTHON_BIN}" "${REPO_ROOT}" "${PROGRESS_INTERVAL}" "${shard}" "${output_jsonl}" "${report_json}" "${MAX_RETRIES}" "${IMAGE_DETAIL}" \
    > "${log}" 2>&1 &

  pid=$!
  print "${pid} ${shard_num} ${log}" >> "${PID_FILE}"
done

print
print "[cot-shards] launched. pid_file: ${PID_FILE}"
print "[cot-shards] monitor:"
print "  tail -f ${LOG_DIR}/sidewalk_cot_${SPLIT}_shard_00.log"
print "  ps -fp \$(awk '{print \$1}' '${PID_FILE}' | tr '\\n' ',')"
print "[cot-shards] merge after all shards exit:"
print "  python ${REPO_ROOT}/tools/model_processing/sidewalk_phase1_cot_merge_by_id.py --source-jsonl '${SOURCE}' --cot-glob '${SHARD_DIR}/phase1_cot_${SPLIT}_part_*.jsonl' --output-jsonl '${OUT}/manifests/phase1_cot_${SPLIT}.jsonl' --report-json '${OUT}/manifests/phase1_cot_merge_${SPLIT}.json' --allow-missing"
