#!/usr/bin/env python3
import argparse
import asyncio
import inspect
import json
import time
from typing import Any

import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock

try:
    import websockets
except ModuleNotFoundError:
    websockets = None

DEFAULT_URI = 'ws://192.168.1.102:9090'
DEFAULT_TOPIC = '/system_timestamp'
DEFAULT_ROS_TOPIC = '/clock'
DEFAULT_RECONNECT_DELAY = 2.0


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class AgileXSystemClockBridge(Node):
    def __init__(self, ros_topic: str):
        super().__init__('agilex_system_clock_bridge')
        self.publisher = self.create_publisher(Clock, ros_topic, 10)
        self.ros_topic = ros_topic
        self.receive_count = 0
        self.get_logger().info(f'Publishing AgileX system timestamp to ROS clock topic: {ros_topic}')

    def publish_clock(self, data: dict[str, Any]) -> None:
        msg = data.get('msg') if isinstance(data.get('msg'), dict) else {}
        stamp = msg.get('stamp') if isinstance(msg.get('stamp'), dict) else {}
        clock = Clock()
        clock.clock.sec = to_int(stamp.get('secs', stamp.get('sec')))
        clock.clock.nanosec = to_int(stamp.get('nsecs', stamp.get('nanosec')))
        self.publisher.publish(clock)
        self.receive_count += 1
        if self.receive_count == 1 or self.receive_count % 100 == 0:
            self.get_logger().info(
                f'Published clock #{self.receive_count}: {clock.clock.sec}.{clock.clock.nanosec:09d}'
            )


async def subscribe_clock(uri: str, node: AgileXSystemClockBridge, topic: str, debug: bool) -> None:
    if websockets is None:
        raise RuntimeError('Missing Python package websockets. Run in foundation_nav or install websockets.')

    request = {
        'op': 'subscribe',
        'topic': topic,
    }
    connect_kwargs: dict[str, Any] = {}
    if 'proxy' in inspect.signature(websockets.connect).parameters:
        connect_kwargs['proxy'] = None

    async with websockets.connect(uri, ping_interval=None, **connect_kwargs) as websocket:
        print(f'Connected to AgileX WebSocket: {uri}')
        await websocket.send(json.dumps(request))
        print('Sent system timestamp subscribe request:')
        print(json.dumps(request, ensure_ascii=False, indent=2))

        async for response in websocket:
            if debug:
                print(response)
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                continue
            if data.get('op') == 'publish' and data.get('topic') == topic:
                node.publish_clock(data)
                rclpy.spin_once(node, timeout_sec=0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Bridge AgileX system timestamp WebSocket data to ROS2 clock.')
    parser.add_argument('--uri', default=DEFAULT_URI, help=f'AgileX WebSocket URI, default: {DEFAULT_URI}')
    parser.add_argument('--topic', default=DEFAULT_TOPIC, help=f'AgileX timestamp topic, default: {DEFAULT_TOPIC}')
    parser.add_argument('--ros-topic', default=DEFAULT_ROS_TOPIC, help=f'ROS2 clock topic, default: {DEFAULT_ROS_TOPIC}')
    parser.add_argument('--debug', action='store_true', help='Print raw WebSocket messages.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = AgileXSystemClockBridge(args.ros_topic)
    try:
        while rclpy.ok():
            try:
                asyncio.run(subscribe_clock(args.uri, node, args.topic, args.debug))
            except KeyboardInterrupt:
                print('\nStopped by user.')
                break
            except RuntimeError as exc:
                print(exc)
                break
            except OSError as exc:
                print(f'WebSocket connection failed: {exc}')
            except Exception as exc:
                if websockets is not None and isinstance(exc, websockets.exceptions.ConnectionClosed):
                    print(f'WebSocket connection closed: code={exc.code}, reason={exc.reason}')
                else:
                    raise
            print(f'Reconnecting in {DEFAULT_RECONNECT_DELAY:.1f}s...')
            time.sleep(DEFAULT_RECONNECT_DELAY)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
