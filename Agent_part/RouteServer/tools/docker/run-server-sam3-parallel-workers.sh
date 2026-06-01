#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/mnt/nvme0/haosu/sidewalk/chunk_0}"
LOG_ROOT="${LOG_ROOT:-/mnt/nvme0/haosu/sam3_logs}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/nvme0/haosu/sam3_cache}"
TEMP_ROOT="${TEMP_ROOT:-/mnt/nvme0/haosu/temp}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/mnt/nvme0/haosu/checkpoints}"
IMAGE="${IMAGE:-urbannav-sam3-processor:latest}"
GPUS_CSV="${GPUS_CSV:-0,1,2,3}"
SAMPLE_FPS="${SAMPLE_FPS:-20.0}"
MIN_SCORE="${MIN_SCORE:-0.3}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p \
  "$LOG_ROOT" \
  "$CACHE_ROOT/home" \
  "$CACHE_ROOT/mpl" \
  "$CACHE_ROOT/xdg" \
  "$CACHE_ROOT/hf" \
  "$CACHE_ROOT/torch" \
  "$CACHE_ROOT/torchinductor" \
  "$TEMP_ROOT"

uid="$(id -u)"
gid="$(id -g)"

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
NUM_WORKERS="${#GPUS[@]}"

if [[ "$NUM_WORKERS" -lt 1 ]]; then
  echo "No GPUs configured in GPUS_CSV" >&2
  exit 2
fi

for worker in "${!GPUS[@]}"; do
  gpu="${GPUS[$worker]}"
  name="sam3_chunk0_w${worker}_${STAMP}"
  log="${LOG_ROOT}/${name}.log"

  nohup docker run --rm \
    --name "$name" \
    --entrypoint /bin/bash \
    --device /dev/nvidia0 \
    --device /dev/nvidia1 \
    --device /dev/nvidia2 \
    --device /dev/nvidia3 \
    --device /dev/nvidia4 \
    --device /dev/nvidia5 \
    --device /dev/nvidia6 \
    --device /dev/nvidia7 \
    --device /dev/nvidiactl \
    --device /dev/nvidia-uvm \
    --device /dev/nvidia-modeset \
    -v /lib/x86_64-linux-gnu/libcuda.so.1:/usr/local/nvidia/lib64/libcuda.so.1:ro \
    -v /lib/x86_64-linux-gnu/libcuda.so:/usr/local/nvidia/lib64/libcuda.so:ro \
    -v /lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/local/nvidia/lib64/libnvidia-ml.so.1:ro \
    -v /lib/x86_64-linux-gnu/libnvidia-ml.so:/usr/local/nvidia/lib64/libnvidia-ml.so:ro \
    -e USER=haosu \
    -e LOGNAME=haosu \
    -e HOME=/data/cache/home \
    -e LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/home/ubuntu/anaconda3/envs/sam3/lib \
    -e CUDA_VISIBLE_DEVICES="$gpu" \
    -e MPLCONFIGDIR=/data/cache/mpl \
    -e XDG_CACHE_HOME=/data/cache/xdg \
    -e HF_HOME=/data/cache/hf \
    -e TORCH_HOME=/data/cache/torch \
    -e TORCHINDUCTOR_CACHE_DIR=/data/cache/torchinductor \
    -e WORKER_INDEX="$worker" \
    -e NUM_WORKERS="$NUM_WORKERS" \
    -v "${DATA_ROOT}:/data/chunk_0" \
    -v "${CHECKPOINT_ROOT}:/data/checkpoints:ro" \
    -v "${CACHE_ROOT}:/data/cache" \
    -v "${TEMP_ROOT}:/data/temp" \
    -v /mnt/nvme0/haosu/sam3_tools:/tools:ro \
    -u "${uid}:${gid}" \
    "$IMAGE" \
    -lc '
      set -euo pipefail
      mapfile -t clips < <(find /data/chunk_0 -mindepth 2 -maxdepth 2 -name front.mp4 -printf "%h\n" | sort)
      echo "[INFO] worker=${WORKER_INDEX}/${NUM_WORKERS} visible_gpu=${CUDA_VISIBLE_DEVICES} total_clips=${#clips[@]}"
      for idx in "${!clips[@]}"; do
        if (( idx % NUM_WORKERS != WORKER_INDEX )); then
          continue
        fi

        clip="${clips[$idx]}"
        status="$(python -c '"'"'
import sys
from pathlib import Path

clip = Path(sys.argv[1])
images = list((clip / "images").glob("*.jpg"))
masks = list((clip / "semantic_masks").glob("*.png"))
if images and len(masks) >= len(images):
    print(f"complete:{len(masks)}/{len(images)}")
else:
    print(f"incomplete:{len(masks)}/{len(images)}")
'"'"' "$clip")"

        if [[ "$status" == complete:* ]]; then
          echo "[SKIP] $(basename "$clip") manifest complete"
          continue
        fi

        echo "[RUN] idx=${idx} status=${status} clip=$(basename "$clip")"
        python /tools/model_processing/run_sam3_som_phase.py \
          --clip-dir "$clip" \
          --run-sam3 \
          --sam3-inference-script /home/ubuntu/WorkSpace/project/SR/Code/sam3/scripts/batch_sam3_demo_260124.py \
          --checkpoint-path /data/checkpoints/checkpoint_35_merged.pt \
          --sample-fps '"$SAMPLE_FPS"' \
          --gpus 0 \
          --min-score '"$MIN_SCORE"' \
          --disable-route-support \
          --compact-semantic-only \
          --cleanup-sam3-artifacts
      done
      echo "[DONE] worker=${WORKER_INDEX}"
    ' > "$log" 2>&1 &

  echo "STARTED worker=${worker} gpu=${gpu} name=${name} log=${log} pid=$!"
done
