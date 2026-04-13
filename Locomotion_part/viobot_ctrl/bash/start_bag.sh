#!/bin/bash
export LD_LIBRARY_PATH=/root/Baton/install/baton/lib/baton/:$LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash
source /root/Baton/install/local_setup.bash  --extend
$(grep "export ROS_DOMAIN_ID" /home/PRR/.bashrc)
source /root/viobot_ctrl/install/local_setup.bash

# 使用时间戳命名 bag 文件夹，并限制单文件大小（强烈建议在边缘设备上使用）
BAG_DIR="/root/record_data/bag_$(date +%Y%m%d_%H%M%S)"
ros2 bag record /baton/gnss/ephem /baton/gnss/meas /baton/stereo3/odometry /baton/algo_status --storage mcap --compression-mode file --compression-format zstd
