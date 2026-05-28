#!/bin/bash
# 处理单个 clip - Phase2 格式

# 配置参数
CHECKPOINT_PATH="/path/to/your/checkpoint_35_merged.pt"
CLIP_DIR="/media/ubuntu/data1/SR-Nav Dataset/sidewalk/chunk_0/2025-07-13__C10003__9c6c72d9-fe02-4a37-816b-15c16effe13e__15"

# 脚本路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRIPT_PATH="$SCRIPT_DIR/run_sam3_som_phase.py"

echo "========================================="
echo "  Phase2 格式 - 单个 clip"
echo "========================================="
echo ""
echo "Clip 目录: $CLIP_DIR"
echo "Checkpoint: $CHECKPOINT_PATH"
echo ""

if [ ! -d "$CLIP_DIR" ]; then
    echo "错误: Clip 目录不存在: $CLIP_DIR"
    exit 1
fi

echo "开始处理..."
python "$SCRIPT_PATH" \
    --clip-dir "$CLIP_DIR" \
    --run-sam3 \
    --sam3-env-name "sam3" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --gpus 0 \
    --min-score 0.3 \
    --sample-fps 2.0 \
    --generate-video \
    --video-fps 10 \
    --cleanup-temp

echo ""
