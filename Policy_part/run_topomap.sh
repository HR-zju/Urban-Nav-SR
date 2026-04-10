#!/bin/zsh
# notice running this script out of docker container, and running ./record_bag.sh <bag_name> 
# Create a new tmux session
session_name="run_topomap"
tmux kill-session -t $session_name 2>/dev/null
tmux new-session -d -s $session_name

# Split the window into 6 panes
tmux selectp -t 0
tmux splitw -v -p 50 

tmux selectp -t 0
tmux splitw -h -p 50

tmux selectp -t 1
tmux splitw -h -p 50 
tmux select-layout tiled


# [ 0 ] [ 2 ] 
# [ 1 ] [ 3 ] 


# Run the realsense2 launch command in the first pane
tmux select-pane -t 0
tmux send-keys "tmux set -g mouse on" Enter
# tmux send-keys "sudo nvidia-smi -pm 1" Enter
# tmux send-keys "sudo nvidia-smi -pl 100" Enter
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



# run teleop in the fourth pane
tmux select-pane -t 2
tmux send-keys "docker exec -it unitree bash" Enter
sleep 1
tmux send-keys "cd driver_ws/" Enter
tmux send-keys "source devel/setup.bash" Enter
tmux send-keys "roslaunch joystick_control seekur_joy.launch" Enter 

tmux select-pane -t 3
tmux send-keys "docker exec -it ros1toros2 zsh" Enter
sleep 1
tmux send-keys "cd /home/nuvo/SR/foundation-policy/Policy_part" Enter
tmux send-keys "mamba activate foundation_policy" Enter
tmux send-keys "python tools/create_topomap_ros2.py --distance-threshold 2.0" Enter


# Attach to the tmux session
tmux -2 attach-session -t $session_name