#!/usr/bin/env python3
"""ROS 2 helper: log waypoints and cmd_vel while running
test_nav_flow_il and PD controller.

This node subscribes to:
  - WAYPOINT_TOPIC (Float32MultiArray), same as test_nav_flow_il / PDController
  - VEL_TOPIC from config/robot.yaml (Twist), same as PDController output

It periodically prints the latest waypoint and cmd_vel pair and appends them
into a timestamped txt file under ./logs/.

Usage (run in a third terminal alongside test_nav_flow_il
and pd_controller_ros2_new2):

    ros2 run (if packaged) ...
    # or directly:
    python log_waypoint_cmd_vel.py

    ############## data format:#############
    v w wp_0 wp_1 ... wp_{2N-1}

You can optionally set the logging frequency with --hz.
"""
import argparse
import os
import yaml
import numpy as np
from datetime import datetime
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

# 假设 topic_names.py 在同一目录下
try:
    from topic_names import WAYPOINT_TOPIC
except ImportError:
    WAYPOINT_TOPIC = "/waypoint_cmd"  # 备用默认值

# 加载机器人配置
CONFIG_PATH = "config/robot.yaml"
try:
    with open(CONFIG_PATH, "r") as f:
        _robot_config = yaml.safe_load(f)
    VEL_TOPIC = _robot_config["vel_navi_topic"]
except Exception as e:
    print(f"Warning: Could not load config from {CONFIG_PATH}, using default. Error: {e}")
    VEL_TOPIC = "/cmd_vel"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hz", type=float, default=5.0, help="Logging frequency (Hz)")
    parser.add_argument("--out_dir", type=str, default="logs", help="Log directory")
    return parser.parse_args()

class WaypointCmdLogger(Node):
    def __init__(self, hz: float, out_dir: str):
        super().__init__("waypoint_cmd_logger")

        self.last_waypoint: Optional[np.ndarray] = None
        self.last_cmd: Optional[Twist] = None

        # --- 重要：设置兼容性强的 QoS 策略 ---
        # 很多传感器和底层控制节点使用 Best Effort 模式
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 订阅话题
        self.create_subscription(Float32MultiArray, WAYPOINT_TOPIC, self._waypoint_callback, qos_profile)
        self.create_subscription(Twist, VEL_TOPIC, self._cmd_callback, qos_profile)

        # 初始化日志文件
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = os.path.join(out_dir, f"waypoint_cmd_{stamp}.txt")
        self._fh = open(self.file_path, "w", encoding="utf-8")
        
        self.get_logger().info(f"Logger initialized.")
        self.get_logger().info(f"Subscribing to: {WAYPOINT_TOPIC} and {VEL_TOPIC}")
        self.get_logger().info(f"Saving to: {self.file_path}")

        # 定时器
        period = 1.0 / max(hz, 1e-3)
        self.timer = self.create_timer(period, self._timer_callback)

    def _waypoint_callback(self, msg: Float32MultiArray):
        data = np.asarray(msg.data, dtype=float).reshape(-1, 2)
        self.last_waypoint = data

    def _cmd_callback(self, msg: Twist):
        self.last_cmd = msg

    def _timer_callback(self):
        # 调试逻辑：如果长时间没数据，打印出到底是哪个缺席
        if self.last_waypoint is None or self.last_cmd is None:
            wp_stat = "OK" if self.last_waypoint is not None else "Missing"
            cmd_stat = "OK" if self.last_cmd is not None else "Missing"
            # 降低频率打印警告，避免刷屏
            if hasattr(self, "_count"): self._count += 1
            else: self._count = 0
            if self._count % 20 == 0: 
                self.get_logger().warn(f"Waiting for data... Waypoint: {wp_stat}, Cmd_vel: {cmd_stat}")
            return

        v = float(self.last_cmd.linear.x)
        w = float(self.last_cmd.angular.z)
        wp_flat = self.last_waypoint.reshape(-1)
        wp_str = " ".join(f"{x:.6f}" for x in wp_flat)

        line = f"{v:.6f} {w:.6f} {wp_str}"
        self.get_logger().info(line)
        
        self._fh.write(line + "\n")
        self._fh.flush() # 确保数据实时写入硬盘

    def close_file(self):
        if hasattr(self, "_fh") and self._fh:
            self._fh.flush()
            self._fh.close()
            self.get_logger().info("Log file flushed and closed.")

def main(args=None):
    cli_args = parse_args()
    rclpy.init(args=args)
    
    node = WaypointCmdLogger(hz=cli_args.hz, out_dir=cli_args.out_dir)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user (Ctrl+C).')
    finally:
        # 1. 先安全关闭文件
        node.close_file()
        # 2. 销毁节点
        node.destroy_node()
        # 3. 最后关闭 ROS 2 
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()