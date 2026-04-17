#!/bin/bash
export LD_LIBRARY_PATH=/root/Baton/install/baton/lib/baton/:$LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash
source /root/Baton/install/local_setup.bash  --extend
$(grep "export ROS_DOMAIN_ID" /home/PRR/.bashrc)
source /root/viobot_ctrl/install/local_setup.bash


# 等待核心驱动节点上线
echo "[1/3] 等待核心驱动节点上线..."
sleep 25

# 启动stereo3算法
echo "[2/3] 发送算法启动指令..."
timeout 3 ros2 topic pub /baton/stereo3_ctrl system_ctrl/AlgoCtrl "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, algo_enable: true, algo_reboot: false, algo_reset: false}" > /dev/null 2>&1


echo "[3/3] 等待算法状态反馈 (algo_status)..."
WAIT_ALGO=0
while [ $WAIT_ALGO -lt 15 ]; do
    # 匹配 "algo_status: stereo3_running"
    STATUS=$(timeout 2 ros2 topic echo --once /baton/algo_status | grep "algo_status: stereo3_running" | wc -l)
    
    if [ "$STATUS" -gt 0 ]; then
        echo "算法成功运行 (stereo3_running)！开始录包..."
        break
    fi
    echo "等待算法反馈中... (${WAIT_ALGO}s)"
    let WAIT_ALGO+=2
done

RECORD_DIR="/root/record_data"
# 确保父目录存在
mkdir -p "$RECORD_DIR"

# 获取当前时间作为基础名称
BASE_BAG_NAME="${RECORD_DIR}/bag_$(date +%Y%m%d_%H%M%S)"
BAG_NAME="$BASE_BAG_NAME"
SUFFIX=1

# 如果文件夹已存在，则自动添加 _1, _2 等后缀
while [ -d "$BAG_NAME" ]; do
    echo "[WARN] 目录 $BAG_NAME 已存在，自动追加后缀..."
    BAG_NAME="${BASE_BAG_NAME}_${SUFFIX}"
    let SUFFIX+=1
done

exec ros2 bag record \
    /baton/gnss/ephem /baton/gnss/meas /baton/stereo3/odometry /baton/algo_status \
    --storage mcap \
    --compression-mode file \
    --compression-format zstd \
    --max-bag-duration 60 \
    -o $BAG_NAME
