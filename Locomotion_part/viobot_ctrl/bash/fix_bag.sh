#!/bin/bash

# 检查是否传入了目标文件夹参数
if [ -z "$1" ]; then
    echo "错误: 缺少参数"
    echo "用法: $0 <bag文件夹路径>"
    echo "示例: $0 /home/symc/data/real_data/bag_20230320_223210"
    exit 1
fi

# 获取绝对路径并去掉末尾可能多余的斜杠
BAG_DIR=$(realpath "$1")

# 检查目录是否存在
if [ ! -d "$BAG_DIR" ]; then
    echo "错误: 目录 $BAG_DIR 不存在！"
    exit 1
fi

echo "=================================================="
echo "开始修复 ROS 2 Bag 数据: $BAG_DIR"
echo "=================================================="

# 1. 检查 zstd 依赖
if ! command -v zstd &> /dev/null; then
    echo "错误: 系统未安装 zstd。"
    echo "=请先执行命令安装: sudo apt install zstd"
    exit 1
fi

# 进入目标目录
cd "$BAG_DIR" || exit

# 2. 解压 .zstd 文件
# 检查是否存在 .zstd 文件，避免命令报错
ZSTD_FILES=(*.mcap.zstd)
if [ -e "${ZSTD_FILES[0]}" ]; then
    echo "[1/4] 正在解压 .mcap.zstd 文件..."
    zstd -d -T0 *.mcap.zstd
else
    echo "[1/4] 当前目录下没有发现 .mcap.zstd 文件，跳过解压。"
fi

# 3. 删除残缺的 metadata.yaml
if [ -f "metadata.yaml" ]; then
    echo "[2/4] 清理残留的 metadata.yaml..."
    rm metadata.yaml
else
    echo "[2/4] 未发现 metadata.yaml，无需清理。"
fi

# 4. 隔离/备份原始压缩包
if [ -e "${ZSTD_FILES[0]}" ]; then
    echo "[3/4] 隔离原始的 .zstd 文件至 backup_zstd 目录..."
    mkdir -p backup_zstd
    mv *.mcap.zstd backup_zstd/
else
    echo "[3/4] 跳过备份步骤。"
fi

# 返回上级目录准备执行 ros2 bag 相关的命令
cd ..

# 5. 重新生成全局索引
echo "[4/4] 正在调用 ROS 2 重建完整数据索引..."
# 注意：这里需要确保你的环境已经 source 了 ROS 2
if ! command -v ros2 &> /dev/null; then
    echo "警告: 未检测到 ros2 命令！尝试加载默认 Humble 环境..."
    source /opt/ros/humble/setup.bash
fi

ros2 bag reindex "$BAG_DIR"

echo ""
echo "================= 修复完成 ================="
echo "正在输出修复后的 Bag 信息验证结果："
echo "--------------------------------------------------"
ros2 bag info "$BAG_DIR"
