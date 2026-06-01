#!/usr/bin/env python3
"""Forward ROS2 Twist commands to the AgileX rosbridge WebSocket.

This is intended for deployments where NavFlow publishes a ROS2
``/cmd_vel`` topic on Orin, while the AgileX chassis mux listens on the
robot-side rosbridge/ROS graph.
"""

import argparse
import asyncio
import base64
import hashlib
import inspect
import json
import os
import socket
import ssl
import struct
import threading
import time
from typing import Any
from urllib.parse import urlparse

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

try:
    import websockets
except ModuleNotFoundError:
    websockets = None


DEFAULT_URI = "ws://192.168.1.102:9090"
DEFAULT_ROS2_TOPIC = "/cmd_vel"
DEFAULT_WS_TOPIC = "/navflow_cmd_vel"
DEFAULT_WS_TYPE = "geometry_msgs/Twist"
DEFAULT_ENABLE_FILE = "/tmp/navflow_cmd_vel_bridge_enabled"
MUX_ADD_SERVICE = "/mux_vel/add"


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


def bridge_enabled(enable_file: str) -> bool:
    try:
        with open(enable_file, "r", encoding="utf-8") as file:
            return file.read().strip().lower() in ("1", "true", "yes", "on", "auto")
    except FileNotFoundError:
        return False
    except OSError:
        return False


class StdlibWebSocket:
    """Small ws:// text-frame sender used when python-websockets is unavailable."""

    def __init__(self, uri: str):
        parsed = urlparse(uri)
        if parsed.scheme not in ("ws", "wss"):
            raise ValueError(f"Unsupported WebSocket URI scheme: {parsed.scheme}")
        self.host = parsed.hostname
        if self.host is None:
            raise ValueError(f"Invalid WebSocket URI: {uri}")
        self.port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        self.path = parsed.path or "/"
        if parsed.query:
            self.path += f"?{parsed.query}"

        raw_sock = socket.create_connection((self.host, self.port), timeout=5.0)
        if parsed.scheme == "wss":
            raw_sock = ssl.create_default_context().wrap_socket(raw_sock, server_hostname=self.host)
        self.sock = raw_sock
        self._handshake()

    def _handshake(self) -> None:
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {self.path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        self.sock.sendall(request.encode("ascii"))

        response = b""
        while b"\r\n\r\n" not in response:
            chunk = self.sock.recv(4096)
            if not chunk:
                break
            response += chunk

        header = response.decode("latin1", errors="replace")
        if " 101 " not in header.split("\r\n", 1)[0]:
            raise RuntimeError(f"WebSocket handshake failed: {header.splitlines()[0] if header else 'empty response'}")

        accept_src = (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")
        expected_accept = base64.b64encode(hashlib.sha1(accept_src).digest()).decode("ascii")
        if f"Sec-WebSocket-Accept: {expected_accept}" not in header:
            raise RuntimeError("WebSocket handshake failed: invalid Sec-WebSocket-Accept")

    def send_text(self, text: str) -> None:
        payload = text.encode("utf-8")
        mask = os.urandom(4)
        first_byte = 0x81
        length = len(payload)
        if length < 126:
            header = struct.pack("!BB", first_byte, 0x80 | length)
        elif length < 65536:
            header = struct.pack("!BBH", first_byte, 0x80 | 126, length)
        else:
            header = struct.pack("!BBQ", first_byte, 0x80 | 127, length)
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        self.sock.sendall(header + mask + masked)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


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
        try:
            rclpy.spin_once(node, timeout_sec=0.05)
        except ExternalShutdownException:
            break
        await asyncio.sleep(0.0)


async def websocket_publish_loop(
    node: CmdVelBuffer,
    uri: str,
    ws_topic: str,
    ws_type: str,
    rate_hz: float,
    timeout_sec: float,
    debug: bool,
    enable_file: str | None,
) -> None:
    period = 1.0 / max(rate_hz, 1e-6)
    advertise_msg = {
        "op": "advertise",
        "topic": ws_topic,
        "type": ws_type,
    }

    if websockets is None:
        websocket = StdlibWebSocket(uri)
        send_text = websocket.send_text
        close_websocket = websocket.close
        node.get_logger().info("Using stdlib WebSocket sender; python-websockets is not installed.")
    else:
        connect_kwargs: dict[str, Any] = {}
        if "proxy" in inspect.signature(websockets.connect).parameters:
            connect_kwargs["proxy"] = None
        websocket = await websockets.connect(uri, ping_interval=None, **connect_kwargs)
        send_text = websocket.send
        close_websocket = websocket.close

    try:
        maybe_awaitable = send_text(json.dumps(advertise_msg))
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable
        add_mux_msg = {
            "op": "call_service",
            "service": MUX_ADD_SERVICE,
            "id": f"add:{ws_topic}:{time.time():.3f}",
            "args": {"topic": ws_topic},
        }
        maybe_awaitable = send_text(json.dumps(add_mux_msg))
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable
        node.get_logger().info(f"Connected AgileX WebSocket: {uri}")
        node.get_logger().info(f"Publishing rosbridge Twist topic: {ws_topic} ({ws_type})")
        node.get_logger().info(f"Requested mux add for {ws_topic} via {MUX_ADD_SERVICE}")

        last_report_count = -1
        while rclpy.ok():
            if enable_file is not None and not bridge_enabled(enable_file):
                node.get_logger().warn(
                    f"Bridge disabled by {enable_file}. Closing WebSocket publisher."
                )
                break

            msg, latest_time, count = node.snapshot()
            if latest_time <= 0.0 or time.monotonic() - latest_time > timeout_sec:
                msg = zero_twist_msg()

            publish_msg = {
                "op": "publish",
                "topic": ws_topic,
                "msg": msg,
            }
            maybe_awaitable = send_text(json.dumps(publish_msg))
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable

            if debug or count != last_report_count:
                last_report_count = count
                node.get_logger().info(
                    "Forwarded cmd_vel "
                    f"#{count}: linear.x={msg['linear']['x']:.3f}, "
                    f"angular.z={msg['angular']['z']:.3f}"
                )

            await asyncio.sleep(period)
    finally:
        maybe_awaitable = close_websocket()
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable


async def dry_run_loop(
    node: CmdVelBuffer,
    rate_hz: float,
    timeout_sec: float,
    debug: bool,
    enable_file: str,
) -> None:
    period = 1.0 / max(rate_hz, 1e-6)
    node.get_logger().warn(
        "WebSocket publishing is DISABLED. Observing ROS2 cmd_vel only. "
        f"Write '1' to {enable_file} or restart with --publish to advertise/publish "
        "navflow_cmd_vel to AgileX."
    )

    last_report_count = -1
    while rclpy.ok():
        msg, latest_time, count = node.snapshot()
        stale = latest_time <= 0.0 or time.monotonic() - latest_time > timeout_sec
        if stale:
            msg = zero_twist_msg()

        if debug or count != last_report_count:
            last_report_count = count
            node.get_logger().info(
                "Observed cmd_vel "
                f"#{count}: linear.x={msg['linear']['x']:.3f}, "
                f"angular.z={msg['angular']['z']:.3f}, "
                f"would_publish={str(not stale).lower()}"
            )

        if bridge_enabled(enable_file):
            node.get_logger().warn(f"Bridge enabled by {enable_file}. Starting WebSocket publisher.")
            return

        await asyncio.sleep(period)


async def run_bridge(args: argparse.Namespace) -> None:
    rclpy.init()
    node = CmdVelBuffer(args.ros2_topic, args.max_linear, args.max_angular)
    spin_task = asyncio.create_task(ros_spin_loop(node))
    try:
        while rclpy.ok():
            try:
                if args.publish:
                    await websocket_publish_loop(
                        node=node,
                        uri=args.uri,
                        ws_topic=args.ws_topic,
                        ws_type=args.ws_type,
                        rate_hz=args.rate,
                        timeout_sec=args.timeout,
                        debug=args.debug,
                        enable_file=None,
                    )
                else:
                    if bridge_enabled(args.enable_file):
                        await websocket_publish_loop(
                            node=node,
                            uri=args.uri,
                            ws_topic=args.ws_topic,
                            ws_type=args.ws_type,
                            rate_hz=args.rate,
                            timeout_sec=args.timeout,
                            debug=args.debug,
                            enable_file=args.enable_file,
                        )
                    else:
                        await dry_run_loop(
                            node=node,
                            rate_hz=args.rate,
                            timeout_sec=args.timeout,
                            debug=args.debug,
                            enable_file=args.enable_file,
                        )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                node.get_logger().error(f"WebSocket bridge disconnected: {exc}. Reconnecting in 1s.")
                await asyncio.sleep(1.0)
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
    parser = argparse.ArgumentParser(description="Bridge ROS2 /cmd_vel to AgileX rosbridge.")
    parser.add_argument("--uri", default=DEFAULT_URI, help=f"AgileX WebSocket URI, default: {DEFAULT_URI}")
    parser.add_argument("--ros2-topic", default=DEFAULT_ROS2_TOPIC, help=f"ROS2 Twist input topic, default: {DEFAULT_ROS2_TOPIC}")
    parser.add_argument("--ws-topic", default=DEFAULT_WS_TOPIC, help=f"rosbridge Twist output topic, default: {DEFAULT_WS_TOPIC}")
    parser.add_argument("--ws-type", default=DEFAULT_WS_TYPE, help=f"rosbridge message type, default: {DEFAULT_WS_TYPE}")
    parser.add_argument("--rate", type=float, default=10.0, help="WebSocket publish rate in Hz, default: 10")
    parser.add_argument("--timeout", type=float, default=0.5, help="Publish zero if no ROS2 cmd_vel arrives within this many seconds")
    parser.add_argument("--max-linear", type=float, default=0.8, help="Absolute linear component clamp. Set <=0 to disable")
    parser.add_argument("--max-angular", type=float, default=0.4, help="Absolute angular component clamp. Set <=0 to disable")
    parser.add_argument("--debug", action="store_true", help="Log every forwarded WebSocket publish")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Actually advertise/publish the WebSocket topic. Default is dry-run observation only.",
    )
    parser.add_argument(
        "--enable-file",
        default=DEFAULT_ENABLE_FILE,
        help=(
            "Runtime enable file used when --publish is not set. "
            f"Default: {DEFAULT_ENABLE_FILE}. Write '1' to enable, remove or write '0' to disable."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run_bridge(args))
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
