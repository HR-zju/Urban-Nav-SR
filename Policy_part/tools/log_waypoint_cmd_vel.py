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
from datetime import datetime
from typing import Optional

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

from topic_names import WAYPOINT_TOPIC


# Load velocity topic from the same config as PD controller
CONFIG_PATH = "config/robot.yaml"
with open(CONFIG_PATH, "r") as f:
    _robot_config = yaml.safe_load(f)
VEL_TOPIC = _robot_config["vel_navi_topic"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hz",
        type=float,
        default=5.0,
        help="Logging frequency in Hz (default: 5)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="logs",
        help="Directory to store txt logs (default: ./logs)",
    )
    return parser.parse_args()


class WaypointCmdLogger(Node):
    def __init__(self, hz: float, out_dir: str):
        super().__init__("waypoint_cmd_logger")

        # latest messages
        self.last_waypoint: Optional[np.ndarray] = None
        self.last_cmd: Optional[Twist] = None

        # subscribers
        self.create_subscription(
            Float32MultiArray,
            WAYPOINT_TOPIC,
            self._waypoint_callback,
            10,
        )
        self.create_subscription(
            Twist,
            VEL_TOPIC,
            self._cmd_callback,
            10,
        )

        # log file
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = os.path.join(out_dir, f"waypoint_cmd_{stamp}.txt")
        self._fh = open(self.file_path, "w", encoding="utf-8")
        self.get_logger().info(f"Logging to {self.file_path}")

        # timer for periodic logging
        period = 1.0 / max(hz, 1e-3)
        self.timer = self.create_timer(period, self._timer_callback)

    def _waypoint_callback(self, msg: Float32MultiArray) -> None:
        # store as (N, 2) array for easier printing & processing
        data = np.asarray(msg.data, dtype=float).reshape(-1, 2)
        self.last_waypoint = data

    def _cmd_callback(self, msg: Twist) -> None:
        self.last_cmd = msg

    def _timer_callback(self) -> None:
        """Periodically log v, w and flattened waypoint vector.

        Each line format:

            v w wp_0 wp_1 ... wp_{2N-1}

        where waypoints are flattened as [x0, y0, x1, y1, ...].
        """

        # Only log when both waypoint and cmd_vel are available
        if self.last_waypoint is None or self.last_cmd is None:
            return

        v = float(self.last_cmd.linear.x)
        w = float(self.last_cmd.angular.z)

        wp_flat = self.last_waypoint.reshape(-1)
        wp_str = " ".join(f"{x:.6f}" for x in wp_flat)

        line = f"{v:.6f} {w:.6f} {wp_str}"

        # print to console
        self.get_logger().info(line)

        # append to file
        self._fh.write(line + "\n")
        self._fh.flush()

    def destroy_node(self):  # type: ignore[override]
        try:
            if hasattr(self, "_fh") and self._fh:
                self._fh.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    cli_args = parse_args()

    rclpy.init(args=args)
    node = WaypointCmdLogger(hz=cli_args.hz, out_dir=cli_args.out_dir)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
