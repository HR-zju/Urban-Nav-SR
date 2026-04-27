#!/bin/bash
sleep 1
ros_domain_id_line=$(grep "export ROS_DOMAIN_ID" /home/PRR/.bashrc)
export ROS_HOME=/root
source /opt/ros/humble/setup.bash
source /root/Baton/install/local_setup.bash  --extend
source /root/viobot_ctrl/install/local_setup.bash --extend
eval $ros_domain_id_line
export LD_LIBRARY_PATH=/root/Baton/install/baton/lib/baton/:/usr/lib/aarch64-linux-gnu/libv4l/:$LD_LIBRARY_PATH

# 等待核心驱动节点上线
echo "[1/3] 等待核心驱动节点上线..."
sleep 25

# 启动定位算法
echo "[2/3] 启动定位算法..."
timeout 3 ros2 topic pub /baton/stereo3_ctrl system_ctrl/AlgoCtrl "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, algo_enable: true, algo_reboot: false, algo_reset: false}" > /dev/null 2>&1
sleep 1
ros2 launch viobot_ctrl start_gnss_then_stereo3.launch.py &

echo "[3/3] 等待算法状态反馈 (algo_status)..."
# 设置最大超时时间为 15 秒
# 启动一次 echo，持续监听，遇到第一个匹配项自动退出
timeout 15 bash -c "ros2 topic echo /baton/algo_status | grep -m 1 'algo_status: stereo3_running'" > /dev/null 2>&1

# 检查上一条命令（即 timeout/grep）的退出状态码
if [ $? -eq 0 ]; then
    echo "算法成功运行 (stereo3_running)！开始录包..."
else
    echo "[错误] 等待算法反馈超时 (15s) 或未获取到正确状态！"
    exit 1 # 状态不对，建议直接退出脚本，不执行后续的录包
fi

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
    /baton/gnss/ephem /baton/gnss/meas /baton/stereo3/odometry /baton/algo_status /camera_x001/image_raw/compressed /baton/gnss/status /baton/gnss/spp_fix_raw /baton/gnss/spp_fix /baton/gnss/heading\
    --storage mcap \
    --compression-mode file \
    --compression-format zstd \
    --max-bag-duration 60 \
    -o $BAG_NAME
