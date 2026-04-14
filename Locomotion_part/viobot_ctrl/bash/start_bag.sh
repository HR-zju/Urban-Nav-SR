#!/bin/bash
export LD_LIBRARY_PATH=/root/Baton/install/baton/lib/baton/:$LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash
source /root/Baton/install/local_setup.bash  --extend
$(grep "export ROS_DOMAIN_ID" /home/PRR/.bashrc)
source /root/viobot_ctrl/install/local_setup.bash

# 等待 5 秒，确保系统环境完全加载
sleep 5

# 使用 --once 参数发送一次性指令，启动stereo3算法
ros2 topic pub --once /baton/stereo3_ctrl system_ctrl/AlgoCtrl "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, algo_enable: true, algo_reboot: false, algo_reset: false}"

# 等待 3 秒，确保指令生效后再开始录包
sleep 3

# 使用时间戳确保每次重启都会创建新文件夹
BAG_NAME="/root/record_data/bag_$(date +%Y%m%d_%H%M%S)"

exec ros2 bag record \
    /baton/gnss/ephem /baton/gnss/meas /baton/stereo3/odometry /baton/algo_status \
    --storage mcap \
    --compression-mode file \
    --compression-format zstd \
    --max-bag-duration 60 \
    -o $BAG_NAME
