#!/bin/zsh
# notice running this script out of docker container, and running ./record_bag.sh <bag_name> 
# Create a new tmux session
session_name="LLM_foundation_policy"
tmux kill-session -t $session_name 2>/dev/null
tmux new-session -d -s $session_name


for i in {1..7}; do
    tmux splitw
    tmux select-layout tiled
done


# [ 0 ] [ 1 ] [ 2 ] [ 3 ]
# [ 4 ] [ 5 ] [ 6 ] [ 7 ]


# Run the realsense2 launch command in pane 0
tmux select-pane -t 0
tmux send-keys "tmux set -g mouse on" Enter
# tmux send-keys "sudo nvidia-smi -pm 1" Enter
# tmux send-keys "sudo nvidia-smi -pl 100" Enter
tmux send-keys "docker start ros1toros2" Enter
sleep 1
tmux send-keys "docker start vins_fusion" Enter
sleep 1
tmux send-keys "docker exec -it ros1toros2 zsh" Enter
sleep 1
tmux send-keys "ros2 launch realsense2_camera rs_launch.py \
    rgb_camera.color_profile:=640,360,30 \
    depth_module.depth_profile:=640,360,30 \
    align_depth.enable:=true" Enter

# gps
tmux select-pane -t 1
tmux send-keys "cd /home/nuvo/SR/foundation-policy/Policy_part/ROS_RTK/catkin_ws" Enter
tmux send-keys "source devel/setup.zsh" Enter
tmux send-keys "rosrun gps_ntrip_py gps_ntrip_node.py" Enter


# control global fusion node in pane 2
tmux select-pane -t 2
tmux send-keys "docker exec -it vins_fusion bash" Enter
sleep 1
tmux send-keys "cd /home/nuvo/SR/foundation-policy/Policy_part/vins_fusion_ws" Enter
tmux send-keys "source devel/setup.bash" Enter
tmux send-keys "rosrun global_fusion global_fusion_node" Enter

# run the robot by joy teleopic in pane 3
tmux select-pane -t 3
tmux send-keys "docker exec -it unitree bash" Enter
sleep 1
tmux send-keys "cd driver_ws/" Enter
tmux send-keys "source devel/setup.bash" Enter
tmux send-keys "roslaunch joystick_control seekur_joy.launch" Enter 

# run ros bridge in pane 4
tmux select-pane -t 4
tmux send-keys "docker exec -it ros1toros2 zsh" Enter
sleep 1
tmux send-keys "source /ros-humble-ros1-bridge/install/setup.zsh
ros2 run ros1_bridge dynamic_bridge" Enter # change topic if necessary

# Run nav flow in pane 5
tmux select-pane -t 5
tmux send-keys "docker exec -it ros1toros2 zsh" Enter
sleep 1
tmux send-keys "mamba activate foundation_policy" Enter
tmux send-keys "cd /home/nuvo/SR/foundation-policy/Policy_part" Enter
# 接收上位机LLM发来的目标点，并导航到目标点
tmux send-keys "python run_navflow_realsense_vis.py --onnx onnx_ckpts/vint_goal_less.onnx --targets ./topomaps/LLM_Policy/LLM_UTM_odom_poses.txt --no_record_video" Enter


# run pd controller in pane 6
tmux select-pane -t 6
tmux send-keys "docker exec -it ros1toros2 zsh" Enter
sleep 1
tmux send-keys "mamba activate foundation_policy" Enter
tmux send-keys "cd /home/nuvo/SR/foundation-policy/Policy_part" Enter
tmux send-keys "python pd_controller_ros2_new2.py" Enter 

# record bag in pane 7
tmux select-pane -t 7
tmux send-keys "docker exec -it ros1toros2 zsh" Enter
sleep 1
tmux send-keys "cd /home/nuvo/SR/foundation-policy/Policy_part/bags/" Enter
# 录制需要手动确认
tmux send-keys "ros2 bag record /camera/camera/color/image_raw/compressed  /globalEstimator/global_odometry  /tf_static -o $1" 



# Attach to the tmux session
tmux -2 attach-session -t $session_name