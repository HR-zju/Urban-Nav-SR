#!/bin/zsh
# Orin-native LLM policy runner.
# Runs on orin26 with host ROS2 Humble and the foundation_nav mamba env.

session_name="LLM_foundation_policy_orin"
repo_root="/home/orin26/code/Urban-Nav-SR"
policy_dir="${repo_root}/Policy_part"
conda_setup="/home/orin26/miniforge3/etc/profile.d/conda.sh"
usb_camera_device="/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0"
usb_camera_id_path="platform-3610000.usb-usb-0:2.4:1.0"

if [[ ! -e "${usb_camera_device}" ]]; then
    echo "[ERROR] USB camera device not found: ${usb_camera_device}"
    echo "[INFO] Available video devices:"
    ls -l /dev/video* /dev/v4l/by-id /dev/v4l/by-path 2>/dev/null
    exit 1
fi

tmux kill-session -t $session_name 2>/dev/null
tmux new-session -d -s $session_name

for i in {1..8}; do
    tmux splitw
    tmux select-layout tiled
done

# [ 0 ] [ 1 ] [ 2 ]
# [ 3 ] [ 4 ] [ 5 ]
# [ 6 ] [ 7 ] [ 8 ]

tmux select-pane -t 0
tmux send-keys "tmux set -g mouse on" Enter
tmux send-keys "source ${conda_setup} && conda activate foundation_nav" Enter
tmux send-keys "cd ${policy_dir}" Enter
# Insight 9 publishes through its CDC/NCM network device and ROS node; this
# policy camera node must only open the Realtek UVC USB camera.
tmux send-keys "export USB_CAMERA_DEVICE=${usb_camera_device}" Enter
tmux send-keys "export USB_CAMERA_ID_MODEL=USB_Camera" Enter
tmux send-keys "export USB_CAMERA_ID_VENDOR_ID=0bda" Enter
tmux send-keys "export USB_CAMERA_ID_MODEL_ID=3035" Enter
tmux send-keys "export USB_CAMERA_ID_SERIAL=Generic_USB_Camera_200901010001" Enter
tmux send-keys "export USB_CAMERA_ID_PATH=${usb_camera_id_path}" Enter
tmux send-keys "python3 tools/usb_camera_node.py" Enter

# GPS / NTRIP ROS2 node
tmux select-pane -t 1
tmux send-keys "source ${conda_setup} && conda activate foundation_nav" Enter 
tmux send-keys "cd ${policy_dir}" Enter
tmux send-keys "python3 tools/ros2_gps_ntrip_node.py --ros-args -p use_agilex_time:=false" Enter

# VINS-fusion replacement ROS2 Python node
tmux select-pane -t 2 
tmux send-keys "source ${conda_setup} && conda activate foundation_nav" Enter
tmux send-keys "cd ${policy_dir}" Enter
# tmux send-keys "python3 tools/ros2_vins_fusion_node.py --ros-args -p use_sim_time:=true" Enter
tmux send-keys "python3 tools/ros2_cpp_logic_fusion_node.py --ros-args \
    -p vio_topic:=/base_odom \
    -p gps_topic:=/gps/fix \
    -p sync_tolerance_sec:=0.01 \
    -p optimization_period_sec:=2.0 \
    -p publish_before_alignment:=true" Enter


tmux select-pane -t 3 
tmux send-keys "source ${conda_setup} && conda activate foundation_nav" Enter
tmux send-keys "cd ${policy_dir}" Enter
tmux send-keys "zsh tools/agilex_api/run_keyboard_mux_switch.sh" Enter

# sub agilex odom_filter on Orin.
tmux select-pane -t 4 
tmux send-keys "/home/orin26/code/start_livox_odom.sh" Enter
tmux send-keys "ros2 topic hz /base_odom" Enter

# Bridge Orin ROS2 /cmd_vel to AgileX WebSocket /navflow_cmd_vel.
# It publishes immediately; use the remote controller for manual/auto switching.
tmux select-pane -t 5 
tmux send-keys "cd ${policy_dir}" Enter
tmux send-keys "mamba activate foundation_nav" Enter
tmux send-keys "zsh tools/agilex_api/run_navflow_cmd_vel_ws_bridge.sh --debug --publish --max-linear 1.5 --max-angular 0.45" Enter


# NavFlow policy node
tmux select-pane -t 6
tmux send-keys "source ${conda_setup} && conda activate foundation_nav" Enter
tmux send-keys "cd ${policy_dir}" Enter
tmux send-keys "export NAVFLOW_DISABLE_TRT=0" Enter
tmux send-keys "export NAVFLOW_TRT_FP16=1" Enter
tmux send-keys "export NAVFLOW_TRT_BUILDER_OPT_LEVEL=0" Enter
tmux send-keys "export NAVFLOW_TRT_ENGINE_CACHE=/home/orin26/.cache/navflow_ort_trt_static224_fp16" Enter
tmux send-keys "python run_navflow_realsense_vis_orin.py --onnx onnx_ckpts/test_mask90_goal_orin_static224_goal_free.onnx  --image_topic /image_raw --image_transport raw --overlay_transport compressed --no_show_window " Enter # --no_publish_overlay
# tmux send-keys "python run_navflow_realsense_vis_goal_orin.py --onnx onnx_ckpts/vint_goal_conditioned_orin_static224.onnx --targets ./topomaps/LLM_Policy/LLM_UTM_odom_poses.txt --image_topic /image_raw --image_transport raw --overlay_transport raw --no_show_window " Enter
# tmux send-keys "python run_navflow_realsense_vis_goal_orin.py --onnx onnx_ckpts/vint_goal_conditioned_orin_static224.onnx --targets ./topomaps/odom_poses.txt --image_topic /image_raw --image_transport raw --overlay_transport raw --no_show_window  --save_vis" Enter


# PD controller
tmux select-pane -t 7 
tmux send-keys "source ${conda_setup} && conda activate foundation_nav" Enter
tmux send-keys "cd ${policy_dir}" Enter
tmux send-keys "python pd_controller_ros2_new2_agilex.py" Enter

# Record ROS2 bag
tmux select-pane -t 8
tmux send-keys "cd ${policy_dir}/bags/staging/" Enter
tmux send-keys "ros2 bag record -s mcap -o nav_\$(date +%Y%m%d_%H%M%S) /image_raw /gps/fix /base_odom /globalEstimator/global_odometry /navflow/overlay/image_raw/compressed  " # 

tmux -2 attach-session -t $session_name
