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
    echo "[1/5] 正在解压 .mcap.zstd 文件..."
    zstd -d -T0 *.mcap.zstd
else
    echo "[1/5] 当前目录下没有发现 .mcap.zstd 文件，跳过解压。"
fi

# 3. 删除残缺的 metadata.yaml
if [ -f "metadata.yaml" ]; then
    echo "[2/5] 清理残留的 metadata.yaml..."
    mv metadata.yaml metadata.yaml.bak
else
    echo "[2/5] 未发现 metadata.yaml，无需清理。"
fi

# 4. 隔离/备份原始压缩包
if [ -e "${ZSTD_FILES[0]}" ]; then
    echo "[3/5] 隔离原始的 .zstd 文件至 backup_zstd 目录..."
    mkdir -p backup_zstd
    mv *.mcap.zstd backup_zstd/
else
    echo "[3/5] 跳过备份步骤。"
fi

# 5. 自动隔离尾部损坏的极小文件
echo "[4/5] 检查并隔离可能损坏的尾部分块文件..."
# 设定大小阈值为 5KB (异常中断的尾部文件通常 < 100KB 或为 0)
SMALL_FILES=$(find . -maxdepth 1 -type f \( -name "*.mcap" -o -name "*.db3" \) -size -5k)

if [ -n "$SMALL_FILES" ]; then
    mkdir -p corrupted_chunks
    echo "发现体积过小的分块文件(大概率已损坏)，已将其隔离至 corrupted_chunks/ 目录:"
    for file in $SMALL_FILES; do
        echo "   -> 被隔离文件: $file"
        mv "$file" corrupted_chunks/
    done
else
    echo "[4/5] 未发现异常小体积文件，数据块大小正常。"
fi

# 返回上级目录准备执行 ros2 bag 相关的命令
cd ..

# 5. 重新生成全局索引
echo "[5/5] 正在调用 ROS 2 重建完整数据索引..."
# 注意：这里需要确保你的环境已经 source 了 ROS 2
if ! command -v ros2 &> /dev/null; then
    echo "警告: 未检测到 ros2 命令！尝试加载默认 Humble 环境..."
    source /opt/ros/humble/setup.bash
fi

# ros2 bag reindex "$BAG_DIR"
# 判断目录下存在哪种格式的数据文件，从而指定对应的 storage_id
if ls "$BAG_DIR"/*.mcap 1> /dev/null 2>&1; then
    ros2 bag reindex "$BAG_DIR" --storage mcap
elif ls "$BAG_DIR"/*.db3 1> /dev/null 2>&1; then
    ros2 bag reindex "$BAG_DIR" --storage sqlite3
else
    echo "错误: 在 $BAG_DIR 中没有找到 .mcap 或 .db3 数据文件！请检查数据是否完整。"
    exit 1
fi

echo ""
echo "================= 修复完成 ================="
echo "正在输出修复后的 Bag 信息验证结果："
echo "--------------------------------------------------"
ros2 bag info "$BAG_DIR"
