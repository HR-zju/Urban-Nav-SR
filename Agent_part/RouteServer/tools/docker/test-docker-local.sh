#!/bin/bash
# 本地测试 Docker 镜像

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROUTE_SERVER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROUTE_SERVER_DIR"

echo "========================================"
echo "  本地测试 UrbanNav Docker 镜像"
echo "========================================"
echo ""

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo "错误: Docker 未安装"
    echo "请先安装 Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# 构建镜像
echo "[1/3] 构建 Docker 镜像..."
docker build -f tools/docker/Dockerfile -t urbannav-processor:latest .

# 测试运行帮助
echo ""
echo "[2/3] 测试运行 --help..."
docker run --rm urbannav-processor:latest dataset_processing/process_chunks_26_29.py --help

# 测试导入所有依赖
echo ""
echo "[3/3] 测试 Python 依赖..."
docker run --rm --entrypoint python urbannav-processor:latest -c "
import numpy
import PIL
import lz4
import tqdm
import cv2
import subprocess
import json
import shutil
from pathlib import Path

print('✓ numpy:', numpy.__version__)
print('✓ PIL/Pillow:', PIL.__version__)
print('✓ lz4:', lz4.__version__)
print('✓ tqdm:', tqdm.__version__)
print('✓ opencv:', cv2.__version__)
print()
print('检查 ffmpeg...')
ffmpeg_ver = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
if ffmpeg_ver.returncode == 0:
    print('✓ ffmpeg:', ffmpeg_ver.stdout.split('\n')[0])
else:
    print('✗ ffmpeg 未找到')
"

echo ""
echo "========================================"
echo "  本地测试完成！"
echo "========================================"
echo ""
echo "现在可以运行 tools/docker/deploy-to-server.sh 部署到服务器"
echo ""
