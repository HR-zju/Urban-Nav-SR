#!/usr/bin/env bash
set -Eeuo pipefail

# V4 五帧、独立 section prompt 可能超过 10K token，服务端保留 16K 上下文。
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/lpai/volumes/mind-eb-bd-ga/suh/UrbanNavigation}"
ENV_DIR="${ENV_DIR:-/lpai/volumes/mind-eb-bd-ga/suh/miniconda3/envs/qwen35_vllm_cuda128}"
MODEL_DIR="${MODEL_DIR:-/lpai/inputs/models/Qwen__Qwen3_5-27B-26-04-24-0254}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-27b-v4}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-18103}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
SERVICE_ROOT="${SERVICE_ROOT:-${WORKSPACE_ROOT}/model_services/qwen35_27b_v4_gpu${GPU_ID}_${PORT}}"
SESSION_NAME="${SESSION_NAME:-qwen35_27b_v4_gpu${GPU_ID}_${PORT}}"
SCRIPT_PATH="$(readlink -f "$0")"
LOG_FILE="${SERVICE_ROOT}/logs/server.log"
TMP_LINK="/lpai/volumes/mind-eb-bd-ga/suh/q35v4_tmp_${GPU_ID}_${PORT}"

prepare_dirs() {
  mkdir -p \
    "${SERVICE_ROOT}/logs" \
    "${SERVICE_ROOT}/cache/home" \
    "${SERVICE_ROOT}/cache/xdg" \
    "${SERVICE_ROOT}/cache/hf" \
    "${SERVICE_ROOT}/cache/torchinductor" \
    "${SERVICE_ROOT}/cache/triton" \
    "${SERVICE_ROOT}/cache/vllm" \
    "${SERVICE_ROOT}/tmp"
}

serve() {
  prepare_dirs
  # ZMQ Unix socket 有 107 字节上限，使用短链接而不把临时文件写到系统盘。
  ln -sfn "${SERVICE_ROOT}/tmp" "${TMP_LINK}"
  export HOME="${SERVICE_ROOT}/cache/home"
  export XDG_CACHE_HOME="${SERVICE_ROOT}/cache/xdg"
  export HF_HOME="${SERVICE_ROOT}/cache/hf"
  export TRANSFORMERS_CACHE="${SERVICE_ROOT}/cache/hf/transformers"
  export TORCHINDUCTOR_CACHE_DIR="${SERVICE_ROOT}/cache/torchinductor"
  export TRITON_CACHE_DIR="${SERVICE_ROOT}/cache/triton"
  export VLLM_CACHE_ROOT="${SERVICE_ROOT}/cache/vllm"
  export TMPDIR="${TMP_LINK}"
  export TEMP="${TMP_LINK}"
  export TMP="${TMP_LINK}"
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
  export no_proxy="${NO_PROXY}"
  exec > >(tee -a "${LOG_FILE}") 2>&1
  exec "${ENV_DIR}/bin/python" -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --model "${MODEL_DIR}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size 1 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
    --dtype bfloat16 \
    --trust-remote-code \
    --attention-config '{"backend":"TRITON_ATTN"}' \
    --mm-encoder-attn-backend TORCH_SDPA \
    --mm-processor-cache-gb 0 \
    --limit-mm-per-prompt '{"image":6}' \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    --generation-config vllm \
    --enable-prefix-caching \
    --no-enable-log-requests
}

start() {
  prepare_dirs
  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "[already-running] ${SESSION_NAME}"
    return 0
  fi
  tmux new-session -d -s "${SESSION_NAME}" \
    "env WORKSPACE_ROOT='${WORKSPACE_ROOT}' ENV_DIR='${ENV_DIR}' MODEL_DIR='${MODEL_DIR}' SERVED_MODEL_NAME='${SERVED_MODEL_NAME}' GPU_ID='${GPU_ID}' PORT='${PORT}' MAX_MODEL_LEN='${MAX_MODEL_LEN}' MAX_NUM_SEQS='${MAX_NUM_SEQS}' MAX_BATCHED_TOKENS='${MAX_BATCHED_TOKENS}' GPU_MEMORY_UTILIZATION='${GPU_MEMORY_UTILIZATION}' SERVICE_ROOT='${SERVICE_ROOT}' SESSION_NAME='${SESSION_NAME}' bash '${SCRIPT_PATH}' serve"
  echo "[started] session=${SESSION_NAME} port=${PORT} gpu=${GPU_ID} model=${SERVED_MODEL_NAME}"
  echo "[log] ${LOG_FILE}"
}

stop() {
  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    tmux kill-session -t "${SESSION_NAME}"
    echo "[stopped] ${SESSION_NAME}"
  else
    echo "[not-running] ${SESSION_NAME}"
  fi
}

status() {
  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "[tmux-running] ${SESSION_NAME}"
  else
    echo "[tmux-stopped] ${SESSION_NAME}"
  fi
  curl -sS --max-time 3 "http://127.0.0.1:${PORT}/v1/models" || true
}

case "${1:-status}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  serve) serve ;;
  *) echo "Usage: $0 {start|stop|restart|status|serve}" >&2; exit 2 ;;
esac
