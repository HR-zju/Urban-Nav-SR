#!/bin/bash
# 批量处理 sidewalk 下所有 clips - Phase2 格式

# 配置参数
CHECKPOINT_PATH="/path/to/your/checkpoint_35_merged.pt"
SIDEWALK_ROOT="/media/ubuntu/data1/SR-Nav Dataset/sidewalk/chunk_0"

# 脚本路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRIPT_PATH="$SCRIPT_DIR/run_sam3_som_phase.py"

echo "========================================="
echo "  批量处理 - Phase2 格式"
echo "========================================="
echo ""
echo "Sidewalk 根目录: $SIDEWALK_ROOT"
echo "Checkpoint: $CHECKPOINT_PATH"
echo ""

if [ ! -d "$SIDEWALK_ROOT" ]; then
    echo "错误: Sidewalk 根目录不存在: $SIDEWALK_ROOT"
    exit 1
fi

echo "开始处理..."
python "$SCRIPT_PATH" \
    --sidewalk-root "$SIDEWALK_ROOT" \
    --run-sam3 \
    --sam3-env-name "sam3" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --gpus 0 \
    --min-score 0.3 \
    --sample-fps 2.0 \
    --generate-video \
    --video-fps 10 \
    --skip-existing \
    --cleanup-temp

echo ""
