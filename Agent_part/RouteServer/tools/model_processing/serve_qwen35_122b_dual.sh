#!/usr/bin/env bash
set -Eeuo pipefail

# 在 8 x A800 80GB 上部署两个相互隔离的 Qwen3.5-122B-A10B TP=4 服务。
# 两个副本分别服务 Behavior 和 Instruction 标注，避免一个 TP=8 服务成为吞吐瓶颈。

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/lpai/volumes/mind-eb-bd-ga/suh/UrbanNavigation}"
ENV_DIR="${ENV_DIR:-/lpai/volumes/mind-eb-bd-ga/suh/miniconda3/envs/qwen35_vllm_cuda128}"
MODEL_DIR="${MODEL_DIR:-}"
SERVICE_ROOT="${SERVICE_ROOT:-${WORKSPACE_ROOT}/model_services/qwen35_122b_a10b_dual_v33}"

BEHAVIOR_GPUS="${BEHAVIOR_GPUS:-0,1,2,3}"
INSTRUCTION_GPUS="${INSTRUCTION_GPUS:-4,5,6,7}"
BEHAVIOR_PORT="${BEHAVIOR_PORT:-18461}"
INSTRUCTION_PORT="${INSTRUCTION_PORT:-18462}"
BEHAVIOR_MODEL_NAME="${BEHAVIOR_MODEL_NAME:-qwen3.5-122b-a10b-behavior}"
INSTRUCTION_MODEL_NAME="${INSTRUCTION_MODEL_NAME:-qwen3.5-122b-a10b-instruction}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
MAX_IMAGES_PER_PROMPT="${MAX_IMAGES_PER_PROMPT:-6}"
STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-1800}"
STOP_TIMEOUT_SEC="${STOP_TIMEOUT_SEC:-120}"
HEALTH_INTERVAL_SEC="${HEALTH_INTERVAL_SEC:-10}"
GPU_PREFLIGHT_MAX_USED_MIB="${GPU_PREFLIGHT_MAX_USED_MIB:-2048}"
GPU_PREFLIGHT_MAX_UTIL="${GPU_PREFLIGHT_MAX_UTIL:-10}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}"
FORCE_START="${FORCE_START:-0}"

SCRIPT_PATH="$(readlink -f "$0")"
ROLES=(behavior instruction)

usage() {
  cat <<'EOF'
Usage:
  serve_qwen35_122b_dual.sh preflight [all|behavior|instruction]
  serve_qwen35_122b_dual.sh start     [all|behavior|instruction]
  serve_qwen35_122b_dual.sh stop      [all|behavior|instruction]
  serve_qwen35_122b_dual.sh restart   [all|behavior|instruction]
  serve_qwen35_122b_dual.sh status    [all|behavior|instruction]
  serve_qwen35_122b_dual.sh smoke     [all|behavior|instruction]
  serve_qwen35_122b_dual.sh tail      [behavior|instruction]
  serve_qwen35_122b_dual.sh print-config

Important overrides:
  MODEL_DIR, ENV_DIR, WORKSPACE_ROOT, SERVICE_ROOT
  BEHAVIOR_GPUS, INSTRUCTION_GPUS, BEHAVIOR_PORT, INSTRUCTION_PORT
  GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN, MAX_NUM_SEQS, MAX_BATCHED_TOKENS
  FORCE_START=1                    # Bypass only GPU usage thresholds.
  DISABLE_CUSTOM_ALL_REDUCE=0     # Benchmark before changing the stable default.
EOF
}

die() {
  echo "[error] $*" >&2
  exit 1
}

resolve_model_dir() {
  local candidate
  if [[ -n "${MODEL_DIR}" ]]; then
    return 0
  fi
  for candidate in \
    /lpai/inputs/models/Qwen__Qwen3_5-122B-A10B-26-02-24-1554 \
    /lpai/inputs/models/Qwen__Qwen3_5-122B-A10B/26-02-24-1554 \
    /lpai/models/Qwen__Qwen3_5-122B-A10B-26-02-24-1554 \
    /lpai/models/Qwen__Qwen3_5-122B-A10B/26-02-24-1554; do
    if [[ -f "${candidate}/config.json" ]]; then
      MODEL_DIR="${candidate}"
      return 0
    fi
  done
  die "cannot locate Qwen3.5-122B-A10B; set MODEL_DIR explicitly"
}

role_gpus() {
  case "$1" in
    behavior) echo "${BEHAVIOR_GPUS}" ;;
    instruction) echo "${INSTRUCTION_GPUS}" ;;
    *) die "unknown role: $1" ;;
  esac
}

role_port() {
  case "$1" in
    behavior) echo "${BEHAVIOR_PORT}" ;;
    instruction) echo "${INSTRUCTION_PORT}" ;;
    *) die "unknown role: $1" ;;
  esac
}

role_model_name() {
  case "$1" in
    behavior) echo "${BEHAVIOR_MODEL_NAME}" ;;
    instruction) echo "${INSTRUCTION_MODEL_NAME}" ;;
    *) die "unknown role: $1" ;;
  esac
}

role_session() {
  case "$1" in
    behavior) echo "vla-eval-0" ;;
    instruction) echo "vla-eval-1" ;;
    *) die "unknown role: $1" ;;
  esac
}

role_root() {
  echo "${SERVICE_ROOT}/$1"
}

role_log() {
  echo "$(role_root "$1")/logs/vllm_$(role_port "$1").log"
}

role_pid_file() {
  echo "$(role_root "$1")/server.pid"
}

selected_roles() {
  case "${1:-all}" in
    all) printf '%s\n' "${ROLES[@]}" ;;
    behavior|instruction) printf '%s\n' "$1" ;;
    *) die "expected all, behavior, or instruction; got: ${1:-}" ;;
  esac
}

gpu_array() {
  local gpu_csv="$1"
  local -a values
  local value
  IFS=',' read -r -a values <<<"${gpu_csv}"
  ((${#values[@]} > 0)) || die "empty GPU set"
  for value in "${values[@]}"; do
    [[ "${value}" =~ ^[0-9]+$ ]] || die "invalid GPU id in '${gpu_csv}': ${value}"
    printf '%s\n' "${value}"
  done
}

tp_size() {
  gpu_array "$(role_gpus "$1")" | wc -l | tr -d ' '
}

validate_gpu_sets() {
  local role gpu
  local -A owners=()
  for role in "${ROLES[@]}"; do
    while IFS= read -r gpu; do
      if [[ -n "${owners[${gpu}]:-}" ]]; then
        die "GPU ${gpu} is assigned to both ${owners[${gpu}]} and ${role}"
      fi
      owners["${gpu}"]="${role}"
    done < <(gpu_array "$(role_gpus "${role}")")
    [[ "$(tp_size "${role}")" -eq 4 ]] || die "${role} must use exactly four GPUs"
  done
}

prepare_role_dirs() {
  local root
  root="$(role_root "$1")"
  mkdir -p \
    "${root}/logs" \
    "${root}/cache/home" \
    "${root}/cache/xdg" \
    "${root}/cache/hf" \
    "${root}/cache/torchinductor" \
    "${root}/cache/triton" \
    "${root}/cache/vllm" \
    "${root}/tmp"
}

validate_common() {
  local command
  resolve_model_dir
  for command in tmux curl nvidia-smi setsid; do
    command -v "${command}" >/dev/null 2>&1 || die "required command not found: ${command}"
  done
  [[ -x "${ENV_DIR}/bin/python" ]] || die "Python environment is unavailable: ${ENV_DIR}"
  [[ -f "${MODEL_DIR}/config.json" ]] || die "model config is unavailable: ${MODEL_DIR}/config.json"
  "${ENV_DIR}/bin/python" -c 'import torch, vllm' >/dev/null \
    || die "torch/vllm import failed in ${ENV_DIR}"
  validate_gpu_sets
}

port_is_listening() {
  local port="$1"
  "${ENV_DIR}/bin/python" -c \
    'import socket, sys; sock = socket.socket(); sock.settimeout(0.5); sys.exit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)' \
    "${port}"
}

check_gpu_available() {
  local role="$1"
  local gpu raw used util
  while IFS= read -r gpu; do
    raw="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)" \
      || die "cannot query GPU ${gpu}"
    IFS=',' read -r used util <<<"${raw}"
    used="${used//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    echo "[gpu] role=${role} id=${gpu} used_mib=${used} util_pct=${util}"
    if ((used > GPU_PREFLIGHT_MAX_USED_MIB || util > GPU_PREFLIGHT_MAX_UTIL)); then
      if [[ "${FORCE_START}" == "1" ]]; then
        echo "[warning] FORCE_START=1 bypasses busy GPU ${gpu}" >&2
      else
        die "GPU ${gpu} is busy; set FORCE_START=1 only after confirming ownership"
      fi
    fi
  done < <(gpu_array "$(role_gpus "${role}")")
}

preflight_role() {
  local role="$1"
  local port session
  port="$(role_port "${role}")"
  session="$(role_session "${role}")"
  check_gpu_available "${role}"
  tmux has-session -t "${session}" 2>/dev/null && die "tmux session already exists: ${session}"
  port_is_listening "${port}" && die "port is already listening: ${port}"
  prepare_role_dirs "${role}"
  echo "[preflight-ok] role=${role} gpus=$(role_gpus "${role}") tp=$(tp_size "${role}") port=${port}"
}

run_preflight() {
  local role
  validate_common
  while IFS= read -r role; do
    preflight_role "${role}"
  done < <(selected_roles "${1:-all}")
}

serve_role() {
  local role="$1"
  local root cache_dir tmp_link log_file pid_file port model_name
  local -a command
  resolve_model_dir
  root="$(role_root "${role}")"
  cache_dir="${root}/cache"
  tmp_link="/lpai/volumes/mind-eb-bd-ga/suh/q122b_v33_${role}_$(role_port "${role}")"
  log_file="$(role_log "${role}")"
  pid_file="$(role_pid_file "${role}")"
  port="$(role_port "${role}")"
  model_name="$(role_model_name "${role}")"
  prepare_role_dirs "${role}"

  # ZMQ Unix socket 路径有长度限制，短链接只缩短路径，真实文件仍写入持久化工作区。
  ln -sfn "${root}/tmp" "${tmp_link}"
  export HOME="${cache_dir}/home"
  export XDG_CACHE_HOME="${cache_dir}/xdg"
  export HF_HOME="${cache_dir}/hf"
  export TRANSFORMERS_CACHE="${cache_dir}/hf/transformers"
  export TORCHINDUCTOR_CACHE_DIR="${cache_dir}/torchinductor"
  export TRITON_CACHE_DIR="${cache_dir}/triton"
  export VLLM_CACHE_ROOT="${cache_dir}/vllm"
  export TMPDIR="${tmp_link}"
  export TEMP="${tmp_link}"
  export TMP="${tmp_link}"
  export CUDA_VISIBLE_DEVICES="$(role_gpus "${role}")"
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export VLLM_WORKER_MULTIPROC_METHOD="spawn"
  export TOKENIZERS_PARALLELISM="false"
  export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
  export no_proxy="${NO_PROXY}"

  echo "$$" >"${pid_file}"
  exec > >(tee -a "${log_file}") 2>&1
  echo "[serve] role=${role} pid=$$ pgid=$(ps -o pgid= -p $$ | tr -d ' ')"
  echo "[serve] gpus=$(role_gpus "${role}") tp=$(tp_size "${role}") port=${port}"
  echo "[serve] model=${MODEL_DIR} served_model_name=${model_name}"
  echo "[serve] started_at=$(date --iso-8601=seconds)"

  command=(
    "${ENV_DIR}/bin/python" -m vllm.entrypoints.openai.api_server
    --host 127.0.0.1
    --port "${port}"
    --model "${MODEL_DIR}"
    --served-model-name "${model_name}"
    --tensor-parallel-size "$(tp_size "${role}")"
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS}"
    --dtype bfloat16
    --trust-remote-code
    --attention-config '{"backend":"TRITON_ATTN"}'
    --mm-encoder-attn-backend TORCH_SDPA
    --mm-processor-cache-gb 0
    --limit-mm-per-prompt "{\"image\":${MAX_IMAGES_PER_PROMPT}}"
    --default-chat-template-kwargs '{"enable_thinking":false}'
    --generation-config vllm
    --enable-prefix-caching
    --no-enable-log-requests
  )
  if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "1" ]]; then
    command+=(--disable-custom-all-reduce)
  fi
  exec "${command[@]}"
}

tmux_start_role() {
  local role="$1"
  local session command
  local -a launch
  session="$(role_session "${role}")"
  launch=(
    setsid --wait env
    "WORKSPACE_ROOT=${WORKSPACE_ROOT}"
    "ENV_DIR=${ENV_DIR}"
    "MODEL_DIR=${MODEL_DIR}"
    "SERVICE_ROOT=${SERVICE_ROOT}"
    "BEHAVIOR_GPUS=${BEHAVIOR_GPUS}"
    "INSTRUCTION_GPUS=${INSTRUCTION_GPUS}"
    "BEHAVIOR_PORT=${BEHAVIOR_PORT}"
    "INSTRUCTION_PORT=${INSTRUCTION_PORT}"
    "BEHAVIOR_MODEL_NAME=${BEHAVIOR_MODEL_NAME}"
    "INSTRUCTION_MODEL_NAME=${INSTRUCTION_MODEL_NAME}"
    "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
    "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
    "MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS}"
    "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
    "MAX_IMAGES_PER_PROMPT=${MAX_IMAGES_PER_PROMPT}"
    "DISABLE_CUSTOM_ALL_REDUCE=${DISABLE_CUSTOM_ALL_REDUCE}"
    bash "${SCRIPT_PATH}" serve "${role}"
  )
  printf -v command '%q ' "${launch[@]}"
  tmux new-session -d -s "${session}" "exec ${command}"
  echo "[started] role=${role} session=${session} log=$(role_log "${role}")"
}

wait_for_health() {
  local role="$1"
  local port model_name session deadline response
  port="$(role_port "${role}")"
  model_name="$(role_model_name "${role}")"
  session="$(role_session "${role}")"
  deadline=$((SECONDS + STARTUP_TIMEOUT_SEC))
  while ((SECONDS < deadline)); do
    if response="$(curl -fsS --max-time 5 "http://127.0.0.1:${port}/v1/models" 2>/dev/null)" \
      && grep -Fq "${model_name}" <<<"${response}"; then
      echo "[healthy] role=${role} url=http://127.0.0.1:${port}/v1 model=${model_name}"
      return 0
    fi
    if ! tmux has-session -t "${session}" 2>/dev/null; then
      echo "[startup-failed] role=${role}; recent log:" >&2
      tail -80 "$(role_log "${role}")" >&2 || true
      return 1
    fi
    sleep "${HEALTH_INTERVAL_SEC}"
  done
  echo "[startup-timeout] role=${role}; recent log:" >&2
  tail -80 "$(role_log "${role}")" >&2 || true
  return 1
}

start_services() {
  local target="${1:-all}"
  local role failed=0
  validate_common
  while IFS= read -r role; do
    preflight_role "${role}"
  done < <(selected_roles "${target}")
  # 两个副本并行加载，避免串行加载把启动时间翻倍。
  while IFS= read -r role; do
    tmux_start_role "${role}"
  done < <(selected_roles "${target}")
  while IFS= read -r role; do
    wait_for_health "${role}" || failed=1
  done < <(selected_roles "${target}")
  return "${failed}"
}

process_matches_role() {
  local role="$1" pid="$2"
  local args
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  args="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
  [[ "${args}" == *"vllm.entrypoints.openai.api_server"* \
    && "${args}" == *"--port $(role_port "${role}")"* ]]
}

stop_role() {
  local role="$1"
  local session pid_file pid pgid deadline
  session="$(role_session "${role}")"
  pid_file="$(role_pid_file "${role}")"
  if [[ -s "${pid_file}" ]]; then
    read -r pid <"${pid_file}"
    if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null \
      && process_matches_role "${role}" "${pid}"; then
      pgid="$(ps -o pgid= -p "${pid}" | tr -d ' ')"
      echo "[stopping] role=${role} pid=${pid} pgid=${pgid}"
      kill -TERM -- "-${pgid}" 2>/dev/null || true
      deadline=$((SECONDS + STOP_TIMEOUT_SEC))
      while kill -0 "${pid}" 2>/dev/null && ((SECONDS < deadline)); do
        sleep 2
      done
      if kill -0 "${pid}" 2>/dev/null; then
        echo "[force-kill] role=${role} pgid=${pgid}" >&2
        kill -KILL -- "-${pgid}" 2>/dev/null || true
      fi
    else
      echo "[warning] ignoring stale or mismatched PID file: ${pid_file}" >&2
    fi
  fi
  tmux kill-session -t "${session}" 2>/dev/null || true
  rm -f "${pid_file}"
  echo "[stopped] role=${role}"
}

stop_services() {
  local role
  while IFS= read -r role; do
    stop_role "${role}"
  done < <(selected_roles "${1:-all}")
}

status_role() {
  local role="$1"
  local session port pid_file state="stopped" health="down" pid="-"
  session="$(role_session "${role}")"
  port="$(role_port "${role}")"
  pid_file="$(role_pid_file "${role}")"
  tmux has-session -t "${session}" 2>/dev/null && state="running"
  [[ -s "${pid_file}" ]] && read -r pid <"${pid_file}"
  curl -fsS --max-time 3 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && health="healthy"
  echo "[status] role=${role} tmux=${state} health=${health} pid=${pid} port=${port} gpus=$(role_gpus "${role}")"
}

status_services() {
  local role
  while IFS= read -r role; do
    status_role "${role}"
  done < <(selected_roles "${1:-all}")
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits
}

smoke_role() {
  local role="$1"
  local port model_name response
  port="$(role_port "${role}")"
  model_name="$(role_model_name "${role}")"
  response="$(curl -fsS --max-time 120 "http://127.0.0.1:${port}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --data-binary "{\"model\":\"${model_name}\",\"messages\":[{\"role\":\"user\",\"content\":\"Return only OK\"}],\"temperature\":0,\"max_tokens\":8}")"
  grep -Fq '"content"' <<<"${response}" || die "chat smoke returned no content for ${role}"
  echo "[smoke-ok] role=${role} response=${response}"
}

smoke_services() {
  local role
  while IFS= read -r role; do
    smoke_role "${role}"
  done < <(selected_roles "${1:-all}")
}

print_config() {
  resolve_model_dir
  cat <<EOF
WORKSPACE_ROOT=${WORKSPACE_ROOT}
ENV_DIR=${ENV_DIR}
MODEL_DIR=${MODEL_DIR}
SERVICE_ROOT=${SERVICE_ROOT}
BEHAVIOR_GPUS=${BEHAVIOR_GPUS}
BEHAVIOR_PORT=${BEHAVIOR_PORT}
BEHAVIOR_MODEL_NAME=${BEHAVIOR_MODEL_NAME}
INSTRUCTION_GPUS=${INSTRUCTION_GPUS}
INSTRUCTION_PORT=${INSTRUCTION_PORT}
INSTRUCTION_MODEL_NAME=${INSTRUCTION_MODEL_NAME}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
MAX_NUM_SEQS=${MAX_NUM_SEQS}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
MAX_IMAGES_PER_PROMPT=${MAX_IMAGES_PER_PROMPT}
DISABLE_CUSTOM_ALL_REDUCE=${DISABLE_CUSTOM_ALL_REDUCE}
EOF
}

action="${1:-status}"
target="${2:-all}"
case "${action}" in
  preflight) run_preflight "${target}" ;;
  start) start_services "${target}" ;;
  stop) stop_services "${target}" ;;
  restart) stop_services "${target}"; start_services "${target}" ;;
  status) status_services "${target}" ;;
  smoke) smoke_services "${target}" ;;
  tail)
    [[ "${target}" != "all" ]] || die "tail requires behavior or instruction"
    tail -n 120 -f "$(role_log "${target}")"
    ;;
  print-config) print_config ;;
  serve)
    [[ "${target}" != "all" ]] || die "serve requires behavior or instruction"
    serve_role "${target}"
    ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
