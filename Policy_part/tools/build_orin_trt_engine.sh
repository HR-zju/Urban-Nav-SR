#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
ONNX="${1:-onnx_ckpts/vint_goal_conditioned_orin_shape.onnx}"
ENGINE="${2:-onnx_ckpts/vint_goal_conditioned_orin_fp16_fast.engine}"
CACHE="${TRT_TIMING_CACHE:-onnx_ckpts/navflow_trt_fp16_fast.timing.cache}"

export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/nvidia:${LD_LIBRARY_PATH:-}"

"${TRTEXEC}" \
  --onnx="${ONNX}" \
  --saveEngine="${ENGINE}" \
  --fp16 \
  --memPoolSize=workspace:2048 \
  --builderOptimizationLevel="${TRT_BUILDER_OPT_LEVEL:-0}" \
  --avgTiming="${TRT_AVG_TIMING:-1}" \
  --timingCacheFile="${CACHE}" \
  --skipInference
