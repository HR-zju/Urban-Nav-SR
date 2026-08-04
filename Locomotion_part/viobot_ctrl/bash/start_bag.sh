#!/bin/bash
sleep 1
ros_domain_id_line=$(grep "export ROS_DOMAIN_ID" /home/PRR/.bashrc)
export ROS_HOME=/root
source /opt/ros/humble/setup.bash
source /root/Baton/install/local_setup.bash  --extend
source /root/viobot_ctrl/install/local_setup.bash --extend
eval $ros_domain_id_line
export LD_LIBRARY_PATH=/root/Baton/install/baton/lib/baton/:/usr/lib/aarch64-linux-gnu/libv4l/:$LD_LIBRARY_PATH

GNSS_LAUNCH_PID=""
BAG_RECORD_PID=""
MONITOR_LOG_DIR=""
MONITOR_PIDS=()

CAMERA_TOPIC="/camera_x001/image_raw/compressed"
TOPIC_SAMPLE_SECONDS=12
CAMERA_MONITOR_INTERVAL_SECONDS=30
SYSTEM_MONITOR_INTERVAL_SECONDS=60
ENABLE_TOPIC_HZ_SAMPLE="${ENABLE_TOPIC_HZ_SAMPLE:-0}"
ENABLE_TOPIC_BW_SAMPLE="${ENABLE_TOPIC_BW_SAMPLE:-0}"
BAG_MAX_CACHE_SIZE="${BAG_MAX_CACHE_SIZE:-536870912}"

stop_monitoring()
{
    if [ "${#MONITOR_PIDS[@]}" -eq 0 ]; then
        return
    fi

    echo "[CLEANUP] 停止录包监控进程..."
    for pid in "${MONITOR_PIDS[@]}"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            if command -v pkill > /dev/null 2>&1; then
                pkill -TERM -P "$pid" 2>/dev/null || true
            fi
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done

    for pid in "${MONITOR_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    MONITOR_PIDS=()
}

start_monitoring()
{
    local bag_basename
    bag_basename=$(basename "$BAG_NAME")
    MONITOR_LOG_DIR="${RECORD_DIR}/log/monitor_${bag_basename}"
    mkdir -p "$MONITOR_LOG_DIR"

    {
        echo "start_time=$(date '+%F %T')"
        echo "bag_name=$BAG_NAME"
        echo "camera_topic=$CAMERA_TOPIC"
        echo "topic_sample_seconds=$TOPIC_SAMPLE_SECONDS"
        echo "camera_monitor_interval_seconds=$CAMERA_MONITOR_INTERVAL_SECONDS"
        echo "system_monitor_interval_seconds=$SYSTEM_MONITOR_INTERVAL_SECONDS"
        echo "enable_topic_hz_sample=$ENABLE_TOPIC_HZ_SAMPLE"
        echo "enable_topic_bw_sample=$ENABLE_TOPIC_BW_SAMPLE"
        echo "bag_max_cache_size=$BAG_MAX_CACHE_SIZE"
    } > "$MONITOR_LOG_DIR/monitor_info.log" 2>&1

    echo "[MONITOR] 录包监控日志目录: $MONITOR_LOG_DIR"

    (
        while true; do
            echo "========== $(date '+%F %T') =========="
            echo "[topic info]"
            ros2 topic info "$CAMERA_TOPIC" -v 2>&1 || true

            echo "[topic hz sample]"
            if [ "$ENABLE_TOPIC_HZ_SAMPLE" = "1" ]; then
                hz_output=$(timeout "$TOPIC_SAMPLE_SECONDS" ros2 topic hz "$CAMERA_TOPIC" 2>&1)
                echo "$hz_output"
                if echo "$hz_output" | grep -q "average rate"; then
                    echo "[OK] camera topic is publishing"
                else
                    echo "[WARN] no camera hz sample within ${TOPIC_SAMPLE_SECONDS}s"
                fi
            else
                echo "[INFO] disabled by ENABLE_TOPIC_HZ_SAMPLE=$ENABLE_TOPIC_HZ_SAMPLE"
            fi

            echo "[topic bw sample]"
            if [ "$ENABLE_TOPIC_BW_SAMPLE" = "1" ]; then
                timeout "$TOPIC_SAMPLE_SECONDS" ros2 topic bw "$CAMERA_TOPIC" 2>&1 || true
            else
                echo "[INFO] disabled by ENABLE_TOPIC_BW_SAMPLE=$ENABLE_TOPIC_BW_SAMPLE"
            fi

            echo "[camera process]"
            pgrep -af "cam_pub|mjpeg_camera_node" 2>&1 || echo "[WARN] cam_pub process not found"

            echo "[video devices]"
            ls -lah /dev/video* 2>&1 || true
            if [ -d /dev/v4l/by-id ]; then
                ls -lah /dev/v4l/by-id/ 2>&1 || true
            else
                echo "[INFO] /dev/v4l/by-id not found"
            fi

            sleep "$CAMERA_MONITOR_INTERVAL_SECONDS"
        done
    ) >> "$MONITOR_LOG_DIR/camera_topic_watch.log" 2>&1 &
    MONITOR_PIDS+=("$!")

    (
        while true; do
            echo "========== $(date '+%F %T') =========="
            echo "[disk]"
            df -h "$RECORD_DIR" 2>&1 || true

            echo "[memory]"
            free -h 2>&1 || true

            echo "[load]"
            uptime 2>&1 || true

            echo "[key processes]"
            pgrep -af "ros2 bag record|cam_pub|mjpeg_camera_node|gnss_spp|gnss_comm" 2>&1 || true

            echo "[top cpu processes]"
            ps -eo pid,ppid,stat,pcpu,pmem,comm,args --sort=-pcpu 2>&1 | head -30 || true

            echo "[bag directory]"
            if [ -d "$BAG_NAME" ]; then
                ls -lah "$BAG_NAME" 2>&1 || true
            else
                echo "[INFO] $BAG_NAME not created yet"
            fi

            sleep "$SYSTEM_MONITOR_INTERVAL_SECONDS"
        done
    ) >> "$MONITOR_LOG_DIR/system_watch.log" 2>&1 &
    MONITOR_PIDS+=("$!")

    if command -v dmesg > /dev/null 2>&1; then
        dmesg -T 2>&1 | tail -200 > "$MONITOR_LOG_DIR/dmesg_initial.log" || true
        dmesg -wT >> "$MONITOR_LOG_DIR/dmesg_watch.log" 2>&1 &
        MONITOR_PIDS+=("$!")
    else
        echo "[WARN] dmesg command not found" > "$MONITOR_LOG_DIR/dmesg_watch.log"
    fi
}

cleanup()
{
    local exit_code=$?
    trap - EXIT INT TERM

    if [ -n "$BAG_RECORD_PID" ] && kill -0 "$BAG_RECORD_PID" 2>/dev/null; then
        echo "[CLEANUP] 停止 ros2 bag record (PID: $BAG_RECORD_PID)..."
        kill -INT "$BAG_RECORD_PID" 2>/dev/null || true
        wait "$BAG_RECORD_PID" 2>/dev/null || true
    fi

    if [ -n "$GNSS_LAUNCH_PID" ] && kill -0 "$GNSS_LAUNCH_PID" 2>/dev/null; then
        echo "[CLEANUP] 停止 GNSS launch (PID: $GNSS_LAUNCH_PID)..."
        kill -INT "$GNSS_LAUNCH_PID" 2>/dev/null || true
        sleep 2
        if kill -0 "$GNSS_LAUNCH_PID" 2>/dev/null; then
            kill -TERM "$GNSS_LAUNCH_PID" 2>/dev/null || true
        fi
        wait "$GNSS_LAUNCH_PID" 2>/dev/null || true
    fi

    stop_monitoring

    exit "$exit_code"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# 等待核心驱动节点上线
echo "[1/3] 等待核心驱动节点上线..."
sleep 25

# 启动定位算法
echo "[2/3] 启动定位算法..."
ros2 topic pub --once /baton/stereo3_ctrl system_ctrl/AlgoCtrl "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, algo_enable: true, algo_reboot: false, algo_reset: false}" > /dev/null 2>&1
sleep 1
ros2 launch gnss_comm_impl gnss_spp_node.launch.py &
GNSS_LAUNCH_PID=$!

echo "[3/3] 等待算法状态反馈 (algo_status)..."
# 设置最大超时时间为 15 秒
# 等到 stereo3_running 状态出现，匹配到第一条后让 ros2 topic echo 自己退出
timeout 15 ros2 topic echo /baton/algo_status \
    --filter "m.algo_status == 'stereo3_running'" \
    --once > /dev/null 2>&1

# 检查上一条命令（即 timeout/grep）的退出状态码
if [ $? -eq 0 ]; then
    echo "算法成功运行 (stereo3_running)！开始录包..."
else
    echo "[错误] 等待算法反馈超时 (15s) 或未获取到正确状态！"
    exit 1 # 状态不对，建议直接退出脚本，不执行后续的录包
fi

# RECORD_DIR="/root/record_data"
RECORD_DIR="/data/upload"

# 确保父目录存在
mkdir -p "$RECORD_DIR" "$RECORD_DIR/log"

# 获取当前时间作为基础名称
BASE_BAG_NAME="${RECORD_DIR}/bag_$(date +%Y%m%d_%H%M%S)"
BAG_NAME="$BASE_BAG_NAME"
SUFFIX=1

# 如果文件夹已存在，则自动添加 _1, _2 等后缀
while [ -d "$BAG_NAME" ]; do
    echo "[WARN] 目录 $BAG_NAME 已存在，自动追加后缀..."
    BAG_NAME="${BASE_BAG_NAME}_${SUFFIX}"
    let SUFFIX+=1
done

start_monitoring

ros2 bag record \
    /baton/gnss/ephem /baton/gnss/meas /baton/stereo3/odometry /baton/algo_status /camera_x001/image_raw/compressed /baton/gnss/status /baton/gnss/spp_fix_raw /baton/gnss/spp_fix /baton/gnss/heading /insight/vio_20hz \
    --storage mcap \
    --storage-config-file /root/viobot_ctrl/bash/mcap_config.yaml\
    --qos-profile-overrides-path /root/viobot_ctrl/bash/qos_overrides.yaml \
    --max-cache-size "$BAG_MAX_CACHE_SIZE" \
    --max-bag-duration 60 \
    -o "$BAG_NAME" &
BAG_RECORD_PID=$!
wait "$BAG_RECORD_PID"
BAG_RECORD_EXIT_CODE=$?
BAG_RECORD_PID=""
exit "$BAG_RECORD_EXIT_CODE"
