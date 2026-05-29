#!/usr/bin/env python3
"""Forward ROS2 Twist commands to the AgileX rosbridge WebSocket.

This is intended for deployments where NavFlow publishes a ROS2
``/navflow_cmd_vel`` topic on Orin, while the AgileX chassis mux listens on the
robot-side rosbridge/ROS graph.
"""

import argparse
import asyncio
import inspect
import json
import threading
import time
from typing import Any

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

try:
    import websockets
except ModuleNotFoundError:
    websockets = None


DEFAULT_URI = "ws://192.168.1.102:9090"
DEFAULT_ROS2_TOPIC = "/navflow_cmd_vel"
DEFAULT_WS_TOPIC = "/navflow_cmd_vel"
DEFAULT_WS_TYPE = "geometry_msgs/Twist"


def clamp(value: float, limit: float) -> float:
    if limit <= 0.0:
        return float(value)
    return max(-limit, min(limit, float(value)))


def twist_to_rosbridge_msg(msg: Twist, max_linear: float, max_angular: float) -> dict[str, Any]:
    return {
        "linear": {
            "x": clamp(msg.linear.x, max_linear),
            "y": clamp(msg.linear.y, max_linear),
            "z": clamp(msg.linear.z, max_linear),
        },
        "angular": {
            "x": clamp(msg.angular.x, max_angular),
            "y": clamp(msg.angular.y, max_angular),
            "z": clamp(msg.angular.z, max_angular),
        },
    }


def zero_twist_msg() -> dict[str, Any]:
    return {
        "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
        "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
    }


class CmdVelBuffer(Node):
    def __init__(self, topic: str, max_linear: float, max_angular: float):
        super().__init__("navflow_cmd_vel_ws_bridge")
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.lock = threading.Lock()
        self.latest_msg = zero_twist_msg()
        self.latest_time = 0.0
        self.count = 0
        self.subscription = self.create_subscription(Twist, topic, self.callback, 10)
        self.get_logger().info(f"Subscribing ROS2 Twist topic: {topic}")

    def callback(self, msg: Twist) -> None:
        converted = twist_to_rosbridge_msg(msg, self.max_linear, self.max_angular)
        with self.lock:
            self.latest_msg = converted
            self.latest_time = time.monotonic()
            self.count += 1

    def snapshot(self) -> tuple[dict[str, Any], float, int]:
        with self.lock:
            return dict(self.latest_msg), self.latest_time, self.count


async def ros_spin_loop(node: Node) -> None:
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        await asyncio.sleep(0.0)


async def websocket_publish_loop(
    node: CmdVelBuffer,
    uri: str,
    ws_topic: str,
    ws_type: str,
    rate_hz: float,
    timeout_sec: float,
    debug: bool,
) -> None:
    if websockets is None:
        raise RuntimeError(
            "Missing Python package 'websockets'. Install it in the active ROS2 "
            "environment: python3 -m pip install websockets"
        )

    connect_kwargs: dict[str, Any] = {}
    if "proxy" in inspect.signature(websockets.connect).parameters:
        connect_kwargs["proxy"] = None

    period = 1.0 / max(rate_hz, 1e-6)

    async with websockets.connect(uri, ping_interval=None, **connect_kwargs) as websocket:
        advertise_msg = {
            "op": "advertise",
            "topic": ws_topic,
            "type": ws_type,
        }
        await websocket.send(json.dumps(advertise_msg))
        node.get_logger().info(f"Connected AgileX WebSocket: {uri}")
        node.get_logger().info(f"Publishing rosbridge Twist topic: {ws_topic} ({ws_type})")

        last_report_count = -1
        while rclpy.ok():
            msg, latest_time, count = node.snapshot()
            if latest_time <= 0.0 or time.monotonic() - latest_time > timeout_sec:
                msg = zero_twist_msg()

            publish_msg = {
                "op": "publish",
                "topic": ws_topic,
                "msg": msg,
            }
            await websocket.send(json.dumps(publish_msg))

            if debug or count != last_report_count:
                last_report_count = count
                node.get_logger().info(
                    "Forwarded cmd_vel "
                    f"#{count}: linear.x={msg['linear']['x']:.3f}, "
                    f"angular.z={msg['angular']['z']:.3f}"
                )

            await asyncio.sleep(period)


async def run_bridge(args: argparse.Namespace) -> None:
    rclpy.init()
    node = CmdVelBuffer(args.ros2_topic, args.max_linear, args.max_angular)
    spin_task = asyncio.create_task(ros_spin_loop(node))
    try:
        await websocket_publish_loop(
            node=node,
            uri=args.uri,
            ws_topic=args.ws_topic,
            ws_type=args.ws_type,
            rate_hz=args.rate,
            timeout_sec=args.timeout,
            debug=args.debug,
        )
    finally:
        spin_task.cancel()
        try:
            await spin_task
        except asyncio.CancelledError:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge ROS2 /navflow_cmd_vel to AgileX rosbridge.")
    parser.add_argument("--uri", default=DEFAULT_URI, help=f"AgileX WebSocket URI, default: {DEFAULT_URI}")
    parser.add_argument("--ros2-topic", default=DEFAULT_ROS2_TOPIC, help=f"ROS2 Twist input topic, default: {DEFAULT_ROS2_TOPIC}")
    parser.add_argument("--ws-topic", default=DEFAULT_WS_TOPIC, help=f"rosbridge Twist output topic, default: {DEFAULT_WS_TOPIC}")
    parser.add_argument("--ws-type", default=DEFAULT_WS_TYPE, help=f"rosbridge message type, default: {DEFAULT_WS_TYPE}")
    parser.add_argument("--rate", type=float, default=10.0, help="WebSocket publish rate in Hz, default: 10")
    parser.add_argument("--timeout", type=float, default=0.5, help="Publish zero if no ROS2 cmd_vel arrives within this many seconds")
    parser.add_argument("--max-linear", type=float, default=0.3, help="Absolute linear component clamp. Set <=0 to disable")
    parser.add_argument("--max-angular", type=float, default=0.4, help="Absolute angular component clamp. Set <=0 to disable")
    parser.add_argument("--debug", action="store_true", help="Log every forwarded WebSocket publish")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run_bridge(args))
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
