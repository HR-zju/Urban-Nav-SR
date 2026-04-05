#!/bin/zsh
# notice running this script out of docker container, and running ./record_bag.sh <bag_name> 
# Create a new tmux session
session_name="record_bag_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into 6 panes
tmux selectp -t 0
tmux splitw -v -p 50 

tmux selectp -t 0
tmux splitw -h -p 66
tmux selectp -t 2
tmux splitw -h -p 50

tmux selectp -t 1
tmux splitw -h -p 66 
tmux selectp -t 4
tmux splitw -h -p 50 
tmux select-layout tiled


# [ 0 ] [ 2 ] [ 3 ]
# [ 1 ] [ 4 ] [ 5 ]


# Run the realsense2 launch command in the first pane
tmux select-pane -t 0
tmux send-keys "tmux set -g mouse on" Enter
tmux send-keys "sudo nvidia-smi -pm 1" Enter
tmux send-keys "sudo nvidia-smi -pl 100" Enter
tmux send-keys "docker start ros1toros2" Enter
tmux send-keys "docker exec -it ros1toros2 zsh" Enter
sleep 1
tmux send-keys "ros2 launch realsense2_camera rs_launch.py \
    rgb_camera.color_profile:=640,360,30 \
    depth_module.depth_profile:=640,360,30 \
    align_depth.enable:=true" Enter


# Run the ros_bride script in the second pane
tmux select-pane -t 1
tmux send-keys "docker exec -it ros1toros2 zsh" Enter
sleep 1
tmux send-keys "source /ros-humble-ros1-bridge/install/setup.zsh
ros2 run ros1_bridge dynamic_bridge" Enter # change topic if necessary


# control the robot by joy teleopic in the third pane
tmux select-pane -t 2
tmux send-keys "docker exec -it ros1toros2 zsh" Enter
sleep 1
tmux send-keys "mamba activate foundation_policy" Enter
tmux send-keys "cd /home/nuvo/SR/foundation-policy" Enter
tmux send-keys "python test_nav_flow_il_realsense.py" Enter


# run teleop in the fourth pane
tmux select-pane -t 3
tmux send-keys "docker exec -it unitree bash" Enter
sleep 1
tmux send-keys "cd driver_ws/" Enter
tmux send-keys "source devel/setup.bash" Enter
tmux send-keys "roslaunch joystick_control seekur_joy.launch" Enter 

# run pd controller in the fifth pane
tmux select-pane -t 4
tmux send-keys "docker exec -it ros1toros2 zsh" Enter
tmux send-keys "mamba activate foundation_policy" Enter
tmux send-keys "cd /home/nuvo/SR/foundation-policy" Enter
tmux send-keys "python pd_controller_ros2_new2.py" Enter 

# Run the rosbag record command in the forth pane
tmux select-pane -t 5
tmux send-keys "docker exec -it ros1toros2 zsh" Enter
tmux send-keys "cd /home/nuvo/SR/foundation-policy/bags/" Enter
tmux send-keys "ros2 bag record /camera/camera/color/image_raw/compressed  /mfla/dr_odom  /tf_static -o $1"  

# Attach to the tmux session
tmux -2 attach-session -t $session_name