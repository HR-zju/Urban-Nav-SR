#!/bin/bash
sleep 1
ros_domain_id_line=$(grep "export ROS_DOMAIN_ID" /home/PRR/.bashrc)
export ROS_HOME=/root
source /opt/ros/humble/setup.bash
source /root/Baton/install/local_setup.bash  --extend
source /root/viobot_ctrl/install/local_setup.bash  --extend
eval $ros_domain_id_line

ros2 run usb_cam usb_cam_node_exe --ros-args \
-r camera_info:=/camera_x001/camera_info \
-r image_raw:=/camera_x001/image_raw \
-r image_raw/compressed:=/camera_x001/image_raw/compressed \
-r image_raw/compressedDepth:=/camera_x001/image_raw/compressedDepth \
-r image_raw/theora:=/camera_x001/image_raw/theora \
-p video_device:="/dev/video40" \
-p pixel_format:="raw_mjpeg" \
-p image_width:=1920 \
-p image_height:=1080 \
-p framerate:=30.0
