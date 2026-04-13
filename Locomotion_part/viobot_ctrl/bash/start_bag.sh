#!/bin/bash
export LD_LIBRARY_PATH=/root/Baton/install/baton/lib/baton/:$LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash
source /root/Baton/install/local_setup.bash  --extend
$(grep "export ROS_DOMAIN_ID" /home/PRR/.bashrc)
source /root/viobot_ctrl/install/local_setup.bash

# 使用时间戳确保每次重启都会创建新文件夹
BAG_NAME="/root/record_data/bag_$(date +%Y%m%d_%H%M%S)"

exec ros2 bag record \
    /baton/gnss/ephem /baton/gnss/meas /baton/stereo3/odometry /baton/algo_status \
    --storage mcap \
    --compression-mode file \
    --compression-format zstd \
    --max-bag-duration 60 \
    -o $BAG_NAME