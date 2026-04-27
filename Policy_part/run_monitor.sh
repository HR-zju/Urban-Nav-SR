#!/bin/zsh
# notice running this script out of docker container, and running ./record_bag.sh <bag_name> 
# Create a new tmux session
session_name="test_monitor"
tmux kill-session -t $session_name 2>/dev/null
tmux new-session -d -s $session_name


for i in {1..4}; do
    tmux splitw
    tmux select-layout tiled
done


# [ 0 ] [ 1 ] [ 2 ] [ 3 ]
# [ 4 ] [ 5 ] [ 6 ] [ 7 ]


# Run the realsense2 launch command in pane 0
tmux select-pane -t 0
tmux send-keys "tmux set -g mouse on" Enter
# tmux send-keys "docker start vins_fusion" Enter
tmux send-keys "docker exec -it ros1toros2 zsh" Enter
sleep 1
# cmd_vel
tmux send-keys "ros2 topic echo /cmd_vel" Enter

# global odometry
tmux select-pane -t 1
tmux send-keys "docker exec -it ros1toros2 zsh" Enter
sleep 1
tmux send-keys "ros2 topic echo /globalEstimator/global_odometry" Enter


# waypoints
tmux select-pane -t 2
tmux send-keys "docker exec -it ros1toros2 zsh" Enter
sleep 1
tmux send-keys "ros2 topic echo /waypoint" Enter

# run the robot by joy teleopic in pane 3
# tmux select-pane -t 3
# tmux send-keys "docker exec -it unitree bash" Enter
# sleep 1
# tmux send-keys "cd driver_ws/" Enter
# tmux send-keys "source devel/setup.bash" Enter
# tmux send-keys "roslaunch joystick_control seekur_joy.launch" Enter 

# # run ros bridge in pane 4
# tmux select-pane -t 4
# tmux send-keys "docker exec -it ros1toros2 zsh" Enter
# sleep 1
# tmux send-keys "source /ros-humble-ros1-bridge/install/setup.zsh
# ros2 run ros1_bridge dynamic_bridge" Enter # change topic if necessary

# # Run nav flow in pane 5
# tmux select-pane -t 5
# tmux send-keys "docker exec -it ros1toros2 zsh" Enter
# sleep 1
# tmux send-keys "mamba activate foundation_policy" Enter
# tmux send-keys "cd /home/nuvo/SR/foundation-policy/Policy_part" Enter
# # 接收上位机LLM发来的目标点，并导航到目标点
# tmux send-keys "python run_navflow_realsense_vis.py --onnx onnx_ckpts/vint_goal_less.onnx --targets ./topomaps/LLM_Policy/LLM_UTM_odom_poses.txt --no_record_video" Enter


# # run pd controller in pane 6
# tmux select-pane -t 6
# tmux send-keys "docker exec -it ros1toros2 zsh" Enter
# sleep 1
# tmux send-keys "mamba activate foundation_policy" Enter
# tmux send-keys "cd /home/nuvo/SR/foundation-policy/Policy_part" Enter
# tmux send-keys "python pd_controller_ros2_new2.py" Enter 

# record bag in pane 7
tmux select-pane -t 3
tmux send-keys "docker exec -it ros1toros2 zsh" Enter
sleep 1
tmux send-keys "cd /home/nuvo/SR/foundation-policy/Policy_part/bags/" Enter
# 录制需要手动确认
tmux send-keys "ros2 bag record /camera/camera/color/image_raw/compressed  /gps/fix  /mfla/dr_odom  /tf -o $1" 



# Attach to the tmux session
tmux -2 attach-session -t $session_name