#!/usr/bin/env python3
import argparse
import asyncio
import inspect
import json
import math
import time
import uuid
from typing import Any

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

try:
    import websockets
except ModuleNotFoundError:
    websockets = None


DEFAULT_URI = "ws://192.168.1.102:9090"
DEFAULT_ROS_TOPIC = "/agilex/odom"
DEFAULT_HEARTBEAT_INTERVAL = 2.0
DEFAULT_CLIENT_ID = f"agilex{uuid.uuid4().hex[:8]}"
DEFAULT_SUB_TOPIC = "/web_global_topic"
DEFAULT_SUB_TYPE = "board_ros/WebGlobalTopic"
ODOM_SUB_TYPE = "nav_msgs/Odometry"


def pretty_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def debug_print(enabled: bool, *args: Any) -> None:
    if enabled:
        print(*args)


def get_nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
        if value is None:
            return default
    return value


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def fill_vector3(ros_vector: Any, data: dict[str, Any] | None) -> None:
    data = data or {}
    ros_vector.x = to_float(data.get("x"))
    ros_vector.y = to_float(data.get("y"))
    ros_vector.z = to_float(data.get("z"))


def fill_quaternion(ros_quaternion: Any, data: dict[str, Any] | None) -> None:
    data = data or {}
    ros_quaternion.x = to_float(data.get("x"))
    ros_quaternion.y = to_float(data.get("y"))
    ros_quaternion.z = to_float(data.get("z"))
    ros_quaternion.w = to_float(data.get("w"), 1.0)


def fill_covariance(target: list[float], source: Any) -> None:
    if not isinstance(source, list):
        return
    for i, value in enumerate(source[:36]):
        target[i] = to_float(value)


def yaw_to_quaternion(yaw: float) -> dict[str, float]:
    half_yaw = yaw * 0.5
    return {
        "x": 0.0,
        "y": 0.0,
        "z": math.sin(half_yaw),
        "w": math.cos(half_yaw),
    }


def stamp_from_json(stamp_data: dict[str, Any] | None, node: Node) -> Any:
    stamp = node.get_clock().now().to_msg()
    if not isinstance(stamp_data, dict):
        return stamp

    sec = stamp_data.get("sec", stamp_data.get("secs"))
    nanosec = stamp_data.get("nanosec", stamp_data.get("nsecs"))
    if sec is not None:
        stamp.sec = to_int(sec)
    if nanosec is not None:
        stamp.nanosec = to_int(nanosec)
    return stamp


def odom_json_to_msg(data: dict[str, Any], node: Node) -> Odometry:
    source_msg = data.get("msg", data)
    if not isinstance(source_msg, dict):
        source_msg = {}

    odom = Odometry()
    header = source_msg.get("header") if isinstance(source_msg.get("header"), dict) else {}
    odom.header.stamp = stamp_from_json(header.get("stamp"), node)
    odom.header.frame_id = str(header.get("frame_id") or "odom")
    odom.child_frame_id = str(source_msg.get("child_frame_id") or "base_link")

    pose = get_nested(source_msg, "pose", "pose", default={})
    twist = get_nested(source_msg, "twist", "twist", default={})

    fill_vector3(odom.pose.pose.position, get_nested(pose, "position", default={}))
    fill_quaternion(odom.pose.pose.orientation, get_nested(pose, "orientation", default={}))
    fill_vector3(odom.twist.twist.linear, get_nested(twist, "linear", default={}))
    fill_vector3(odom.twist.twist.angular, get_nested(twist, "angular", default={}))

    fill_covariance(odom.pose.covariance, get_nested(source_msg, "pose", "covariance", default=[]))
    fill_covariance(odom.twist.covariance, get_nested(source_msg, "twist", "covariance", default=[]))
    return odom


def web_global_json_to_msg(data: dict[str, Any], node: Node) -> Odometry:
    source_msg = data.get("msg", data)
    if not isinstance(source_msg, dict):
        source_msg = {}

    odom = Odometry()
    odom.header.stamp = node.get_clock().now().to_msg()
    odom.header.frame_id = "map"
    odom.child_frame_id = "base_link"

    odom.pose.pose.position.x = to_float(source_msg.get("x"))
    odom.pose.pose.position.y = to_float(source_msg.get("y"))
    odom.pose.pose.position.z = 0.0
    fill_quaternion(odom.pose.pose.orientation, yaw_to_quaternion(to_float(source_msg.get("rotation"))))

    odom.twist.twist.linear.x = to_float(source_msg.get("linear_velocity"))
    odom.twist.twist.linear.y = 0.0
    odom.twist.twist.linear.z = 0.0
    odom.twist.twist.angular.x = 0.0
    odom.twist.twist.angular.y = 0.0
    odom.twist.twist.angular.z = to_float(source_msg.get("angular_velocity"))

    return odom


def is_target_message(data: dict[str, Any], topic: str) -> bool:
    return data.get("op") == "publish" and data.get("topic") == topic and isinstance(data.get("msg"), dict)


def summarize_json_message(data: dict[str, Any]) -> str:
    parts = []
    for key in ("op", "topic", "id", "level", "type"):
        if key in data:
            parts.append(f"{key}={data.get(key)!r}")

    msg = data.get("msg")
    if isinstance(msg, str):
        parts.append(f"msg={msg!r}")
    elif isinstance(msg, dict):
        msg_keys = ",".join(msg.keys())
        parts.append(f"msg_keys=[{msg_keys}]")
    elif msg is not None:
        parts.append(f"msg_type={type(msg).__name__}")

    return ", ".join(parts) if parts else "no common fields"


class AgileXOdomBridge(Node):
    def __init__(self, ros_topic: str):
        super().__init__("agilex_odom_bridge")
        self.publisher = self.create_publisher(Odometry, ros_topic, 10)
        self.ros_topic = ros_topic
        self.publish_count = 0
        self.get_logger().info(f"Publishing AgileX odometry to ROS2 topic: {ros_topic}")

    def publish_json_odom(self, data: dict[str, Any], sub_type: str, debug: bool = False) -> None:
        if sub_type == DEFAULT_SUB_TYPE:
            odom_msg = web_global_json_to_msg(data, self)
        else:
            odom_msg = odom_json_to_msg(data, self)
        pos = odom_msg.pose.pose.position
        self.publisher.publish(odom_msg)
        self.publish_count += 1
        if debug or self.publish_count == 1 or self.publish_count % 50 == 0:
            self.get_logger().info(
                f"Published odom #{self.publish_count}: "
                f"frame_id={odom_msg.header.frame_id}, child_frame_id={odom_msg.child_frame_id}, "
                f"x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}, "
                f"linear={odom_msg.twist.twist.linear.x:.3f}, angular={odom_msg.twist.twist.angular.z:.3f}"
            )


async def heartbeat_loop(websocket: Any, interval: float, client_id: str, debug: bool) -> None:
    if interval <= 0.0:
        return

    while True:
        await asyncio.sleep(interval)
        heartbeat_msg = {
            "op": "ping",
            "id": client_id,
            "timeStamp": int(time.time() * 1000),
        }
        heartbeat_text = json.dumps(heartbeat_msg)
        await websocket.send(heartbeat_text)
        debug_print(debug, "\nSent heartbeat message:")
        debug_print(debug, pretty_json(heartbeat_msg))


async def subscribe_odom(
    uri: str,
    node: AgileXOdomBridge,
    sub_topic: str,
    sub_type: str,
    heartbeat_interval: float,
    client_id: str,
    debug: bool,
) -> None:
    if websockets is None:
        raise RuntimeError(
            "Missing Python package 'websockets'. Install it in the ROS2 Python "
            "environment, for example: python3 -m pip install websockets"
        )

    subscribe_msg = {
        "op": "subscribe",
        "topic": sub_topic,
        "type": sub_type,
    }

    connect_kwargs: dict[str, Any] = {}
    if "proxy" in inspect.signature(websockets.connect).parameters:
        # websockets >= 15 reads proxy environment variables by default.
        # Robot LAN addresses should be contacted directly.
        connect_kwargs["proxy"] = None

    async with websockets.connect(uri, ping_interval=None, **connect_kwargs) as websocket:
        print(f"Connected to AgileX WebSocket: {uri}")
        subscribe_text = json.dumps(subscribe_msg)
        await websocket.send(subscribe_text)
        print(f"Subscribed WebSocket topic: {sub_topic} ({sub_type})")
        debug_print(debug, "Sent subscribe message:")
        debug_print(debug, pretty_json(subscribe_msg))

        first_heartbeat_msg = {
            "op": "ping",
            "id": client_id,
            "timeStamp": int(time.time() * 1000),
        }
        await websocket.send(json.dumps(first_heartbeat_msg))
        debug_print(debug, "Sent initial heartbeat message:")
        debug_print(debug, pretty_json(first_heartbeat_msg))

        heartbeat_task = asyncio.create_task(
            heartbeat_loop(websocket, heartbeat_interval, client_id, debug)
        )
        try:
            async for response in websocket:
                debug_print(debug, "\nReceived raw message:")
                debug_print(debug, response)

                try:
                    data = json.loads(response)
                except json.JSONDecodeError:
                    debug_print(debug, "Received message is not JSON; skip ROS2 publish.")
                    continue

                debug_print(debug, "\nReceived JSON message:")
                debug_print(debug, pretty_json(data))

                if data.get("op") == "pong":
                    debug_print(debug, "Received heartbeat pong.")
                    continue

                if not is_target_message(data, sub_topic):
                    debug_print(
                        debug,
                        "Received JSON is not target publish; skip ROS2 publish. "
                        f"Summary: {summarize_json_message(data)}",
                    )
                    continue

                if sub_type in (DEFAULT_SUB_TYPE, ODOM_SUB_TYPE):
                    node.publish_json_odom(data, sub_type, debug)
                    rclpy.spin_once(node, timeout_sec=0.0)
                else:
                    debug_print(
                        debug,
                        f"Received target topic {sub_topic}; not publishing to ROS2 because type is {sub_type}.",
                    )
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subscribe AgileX /odom over WebSocket.")
    parser.add_argument(
        "--uri",
        default=DEFAULT_URI,
        help=f"AgileX WebSocket URI, default: {DEFAULT_URI}",
    )
    parser.add_argument(
        "--ros-topic",
        default=DEFAULT_ROS_TOPIC,
        help=f"ROS2 Odometry topic to publish, default: {DEFAULT_ROS_TOPIC}",
    )
    parser.add_argument(
        "--sub-topic",
        default=DEFAULT_SUB_TOPIC,
        help=f"WebSocket topic to subscribe, default: {DEFAULT_SUB_TOPIC}",
    )
    parser.add_argument(
        "--sub-type",
        default=DEFAULT_SUB_TYPE,
        help=f"WebSocket topic type, default: {DEFAULT_SUB_TYPE}",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=DEFAULT_HEARTBEAT_INTERVAL,
        help=(
            "JSON heartbeat interval in seconds. The User API requires a ping "
            "message within 5 seconds. Set <= 0 to disable. "
            f"Default: {DEFAULT_HEARTBEAT_INTERVAL}"
        ),
    )
    parser.add_argument(
        "--client-id",
        default=DEFAULT_CLIENT_ID,
        help=f"Client id used in heartbeat ping messages, default: {DEFAULT_CLIENT_ID}",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw WebSocket messages, formatted JSON, heartbeats, and every ROS2 publish.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = AgileXOdomBridge(args.ros_topic)
    try:
        asyncio.run(
            subscribe_odom(
                args.uri,
                node,
                args.sub_topic,
                args.sub_type,
                args.heartbeat_interval,
                args.client_id,
                args.debug,
            )
        )
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except OSError as exc:
        print(f"WebSocket connection failed: {exc}")
    except RuntimeError as exc:
        print(exc)
    except Exception as exc:  # noqa: BLE001
        if websockets is not None and isinstance(exc, websockets.exceptions.InvalidMessage):
            print(
                "WebSocket handshake failed: did not receive a valid HTTP response.\n"
                f"Current URI: {args.uri}\n"
                "This usually means the host/port/path is not a WebSocket endpoint. "
                "Check the AgileX User API WebSocket address and rerun with "
                "--uri ws://<robot-ip>:<port>/<path>."
            )
        elif websockets is not None and isinstance(exc, websockets.exceptions.ConnectionClosed):
            print(f"WebSocket connection closed: code={exc.code}, reason={exc.reason}")
        else:
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
