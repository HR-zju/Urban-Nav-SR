#!/bin/bash
# 批量处理 sidewalk 目录下所有 clips - Phase2 格式

# 配置参数
CHECKPOINT_PATH="/media/ubuntu/data1/checkpoints/checkpoint_35_merged.pt"
SIDEWALK_DIR="/media/ubuntu/data1/SR-Nav Dataset/sidewalk"

# 脚本路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRIPT_PATH="$SCRIPT_DIR/run_sam3_som_phase.py"

echo "========================================="
echo "  批量处理 Sidewalk - Phase2 格式"
echo "========================================="
echo ""
echo "Sidewalk 目录: $SIDEWALK_DIR"
echo "Checkpoint: $CHECKPOINT_PATH"
echo ""

if [ ! -d "$SIDEWALK_DIR" ]; then
    echo "错误: Sidewalk 目录不存在: $SIDEWALK_DIR"
    exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "错误: Checkpoint 文件不存在: $CHECKPOINT_PATH"
    exit 1
fi

echo "开始处理..."
python "$SCRIPT_PATH" \
    --sidewalk-dir "$SIDEWALK_DIR" \
    --run-sam3 \
    --sam3-env-name "sam3" \
    --finetuned \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --gpus 0 \
    --min-score 0.3 \
    --mode "foot" \
    --sample-fps 2.0 \
    --generate-video \
    --video-fps 10 \
    --skip-existing

echo ""
