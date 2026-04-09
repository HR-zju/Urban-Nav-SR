# !/bin/bash



# 环境变量
# source /home/mayukai/miniconda3/etc/profile.d/conda.sh
# source /opt/ros/humble/setup.bash
# bash kill_tmux.sh
# bash run_ros_pub.sh go2
# bash run_ros_pub.sh coco


# conda activate vint_deployment
# cd visualnav_ros/deployment/src
python tools/create_topomap_ros2.py -distance-threshold 2.0

# conda activate vint_deployment
# cd CityWalker/
# python create_topomap_ros2.py --distance-threshold 1

# # 根据模型名称选择不同的启动命令
# if [ "$ROBOT" == "coco" ]; then
#     echo "Starting {$ROBOT} robot rosnode..."
#     cd zed_sdk2
#     bash run_rosnode.sh


# elif [ "$ROBOT" == "go2" ]; then
#     echo "Starting {$ROBOT} robot rosnode..."
#     cd unitree_sdk2_python
#     bash run_rosnode.sh

# else
#     echo "Error: Unknown robot '$ROBOT'"
#     exit 1
# fi
