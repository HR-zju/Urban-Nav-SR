#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
ENGINE="${1:-onnx_ckpts/vint_goal_conditioned_orin_fp16_fast.engine}"

export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/nvidia:${LD_LIBRARY_PATH:-}"

"${TRTEXEC}" \
  --loadEngine="${ENGINE}" \
  --duration="${TRT_BENCH_DURATION:-15}" \
  --warmUp="${TRT_BENCH_WARMUP_MS:-1000}" \
  --avgRuns="${TRT_BENCH_AVG_RUNS:-100}" \
  --useCudaGraph \
  --useSpinWait \
  --noDataTransfers
