#!/bin/zsh
# Orin-native LLM policy runner.
# Runs on orin26 with host ROS2 Humble and the foundation_nav mamba env.

session_name="LLM_foundation_policy_orin"
repo_root="/home/orin26/code/Urban-Nav-SR"
policy_dir="${repo_root}/Policy_part"
conda_setup="/home/orin26/miniforge3/etc/profile.d/conda.sh"

tmux kill-session -t $session_name 2>/dev/null
tmux new-session -d -s $session_name

for i in {1..7}; do
    tmux splitw
    tmux select-layout tiled
done

# [ 0 ] [ 1 ] [ 2 ] [ 3 ]
# [ 4 ] [ 5 ] [ 6 ] [ 7 ]

tmux select-pane -t 0
tmux send-keys "tmux set -g mouse on" Enter
tmux send-keys "source /opt/ros/humble/setup.zsh" Enter
tmux send-keys "source ${conda_setup} && conda activate foundation_nav" Enter
tmux send-keys "cd ${policy_dir}" Enter
tmux send-keys "python3 tools/usb_camera_node.py" Enter

# GPS / NTRIP ROS2 node
tmux select-pane -t 1
tmux send-keys "source /opt/ros/humble/setup.zsh" Enter
tmux send-keys "source ${conda_setup} && conda activate foundation_nav" Enter
tmux send-keys "cd ${policy_dir}" Enter
tmux send-keys "python3 tools/ros2_gps_ntrip_node.py" Enter

# VINS-fusion replacement ROS2 Python node
tmux select-pane -t 2
tmux send-keys "source /opt/ros/humble/setup.zsh" Enter
tmux send-keys "source ${conda_setup} && conda activate foundation_nav" Enter
tmux send-keys "cd ${policy_dir}" Enter
tmux send-keys "python3 tools/ros2_vins_fusion_node.py" Enter


# tmux select-pane -t 2
# tmux send-keys "source /opt/ros/humble/setup.zsh" Enter
# tmux send-keys "source ${conda_setup} && conda activate foundation_nav" Enter
# tmux send-keys "cd ${policy_dir}" Enter
# tmux send-keys "zsh tools/agilex_api/run_keyboard_mux_switch.sh" Enter

# sub agilex odom_filter on Orin.
tmux select-pane -t 3
tmux send-keys "source /opt/ros/humble/setup.zsh" Enter
tmux send-keys "mamba activate foundation_nav" Enter
tmux send-keys "python tools/sub_agilex_odom.py" Enter

# Bridge Orin ROS2 /cmd_vel to AgileX WebSocket /navflow_cmd_vel.
# It publishes immediately; use the remote controller for manual/auto switching.
tmux select-pane -t 4
tmux send-keys "source /opt/ros/humble/setup.zsh" Enter
tmux send-keys "cd ${policy_dir}" Enter
tmux send-keys "mamba activate foundation_nav" Enter
tmux send-keys "zsh tools/agilex_api/run_navflow_cmd_vel_ws_bridge.sh --debug --publish --max-linear 0.8 --max-angular 0.4" Enter


# NavFlow policy node
tmux select-pane -t 5
tmux send-keys "source /opt/ros/humble/setup.zsh" Enter
tmux send-keys "source ${conda_setup} && conda activate foundation_nav" Enter
tmux send-keys "cd ${policy_dir}" Enter
tmux send-keys "export NAVFLOW_DISABLE_TRT=0" Enter
tmux send-keys "export NAVFLOW_TRT_FP16=1" Enter
tmux send-keys "export NAVFLOW_TRT_BUILDER_OPT_LEVEL=0" Enter
tmux send-keys "export NAVFLOW_TRT_ENGINE_CACHE=/home/orin26/.cache/navflow_ort_trt_static224_fp16" Enter
tmux send-keys "python run_navflow_realsense_vis_orin.py --onnx onnx_ckpts/vint_goal_free_orin_static224.onnx  --image_topic /image_raw --image_transport raw --overlay_transport compressed --no_show_window" Enter
# tmux send-keys "python run_navflow_realsense_vis_goal_orin.py --onnx onnx_ckpts/vint_goal_conditioned_orin_static224.onnx --targets ./topomaps/LLM_Policy/LLM_UTM_odom_poses.txt --image_topic /image_raw --image_transport raw --overlay_transport compressed --no_show_window" Enter
# PD controller
tmux select-pane -t 6
tmux send-keys "source /opt/ros/humble/setup.zsh" Enter
tmux send-keys "source ${conda_setup} && conda activate foundation_nav" Enter
tmux send-keys "cd ${policy_dir}" Enter
tmux send-keys "python pd_controller_ros2_new2_agilex.py" Enter

# Record ROS2 bag
tmux select-pane -t 7
tmux send-keys "source /opt/ros/humble/setup.zsh" Enter
tmux send-keys "cd ${policy_dir}" Enter
tmux send-keys "ros2 bag record -s mcap -o bags/staging/nav_\$(date +%Y%m%d_%H%M%S) /image_raw /gps/fix /mfla/dr_odom /globalEstimator/global_odometry /navflow/overlay/image_raw/compressed"

tmux -2 attach-session -t $session_name
