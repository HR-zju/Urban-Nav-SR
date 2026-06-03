#!/usr/bin/env python3
import argparse
import asyncio
import copy
import inspect
import json
import time
from typing import Any

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node

try:
    import websockets
except ModuleNotFoundError:
    websockets = None

DEFAULT_URI = 'ws://192.168.1.102:9090'
DEFAULT_FILTERED_WS_TOPIC = '/odom_filtered'
DEFAULT_ODOM_WS_TOPIC = '/odom'
DEFAULT_FILTERED_ROS_TOPIC = '/agilex/odom_filtered'
DEFAULT_ODOM_ROS_TOPIC = '/agilex/odom'
DEFAULT_FILTERED_PATH_ROS_TOPIC = '/agilex/odom_filtered/path'
DEFAULT_ODOM_PATH_ROS_TOPIC = '/agilex/odom/path'
DEFAULT_SUB_TYPE = 'nav_msgs/Odometry'
DEFAULT_RECONNECT_DELAY = 2.0
DEFAULT_MAX_PATH_LENGTH = 2000


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


def get_nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
        if value is None:
            return default
    return value


def fill_vector3(ros_vector: Any, data: dict[str, Any] | None) -> None:
    data = data or {}
    ros_vector.x = to_float(data.get('x'))
    ros_vector.y = to_float(data.get('y'))
    ros_vector.z = to_float(data.get('z'))


def fill_quaternion(ros_quaternion: Any, data: dict[str, Any] | None) -> None:
    data = data or {}
    ros_quaternion.x = to_float(data.get('x'))
    ros_quaternion.y = to_float(data.get('y'))
    ros_quaternion.z = to_float(data.get('z'))
    ros_quaternion.w = to_float(data.get('w'), 1.0)


def fill_covariance(target: list[float], source: Any) -> None:
    if not isinstance(source, list):
        return
    for i, value in enumerate(source[:36]):
        target[i] = to_float(value)


def stamp_from_json(stamp_data: dict[str, Any] | None, node: Node) -> Any:
    stamp = node.get_clock().now().to_msg()
    if not isinstance(stamp_data, dict):
        return stamp
    sec = stamp_data.get('sec', stamp_data.get('secs'))
    nanosec = stamp_data.get('nanosec', stamp_data.get('nsecs'))
    if sec is not None:
        stamp.sec = to_int(sec)
    if nanosec is not None:
        stamp.nanosec = to_int(nanosec)
    return stamp


def odom_json_to_msg(data: dict[str, Any], node: Node) -> Odometry:
    source_msg = data.get('msg', data)
    if not isinstance(source_msg, dict):
        source_msg = {}

    odom = Odometry()
    header = source_msg.get('header') if isinstance(source_msg.get('header'), dict) else {}
    odom.header.stamp = stamp_from_json(header.get('stamp'), node)
    odom.header.frame_id = str(header.get('frame_id') or 'odom')
    odom.child_frame_id = str(source_msg.get('child_frame_id') or 'base_link')

    pose = get_nested(source_msg, 'pose', 'pose', default={})
    twist = get_nested(source_msg, 'twist', 'twist', default={})

    fill_vector3(odom.pose.pose.position, get_nested(pose, 'position', default={}))
    fill_quaternion(odom.pose.pose.orientation, get_nested(pose, 'orientation', default={}))
    fill_vector3(odom.twist.twist.linear, get_nested(twist, 'linear', default={}))
    fill_vector3(odom.twist.twist.angular, get_nested(twist, 'angular', default={}))
    fill_covariance(odom.pose.covariance, get_nested(source_msg, 'pose', 'covariance', default=[]))
    fill_covariance(odom.twist.covariance, get_nested(source_msg, 'twist', 'covariance', default=[]))
    return odom


class AgileXDualOdomBridge(Node):
    def __init__(
        self,
        topic_map: dict[str, str],
        path_topic_map: dict[str, str],
        restamp_published: bool,
        max_path_length: int,
    ):
        super().__init__('agilex_dual_odom_bridge')
        self.topic_map = topic_map
        self.path_topic_map = path_topic_map
        self.restamp_published = restamp_published
        self.max_path_length = int(max_path_length)
        self.odom_publishers = {
            ws_topic: self.create_publisher(Odometry, ros_topic, 10)
            for ws_topic, ros_topic in topic_map.items()
        }
        self.path_publishers = {
            ws_topic: self.create_publisher(Path, path_topic, 10)
            for ws_topic, path_topic in path_topic_map.items()
        }
        self.paths = {ws_topic: Path() for ws_topic in topic_map}
        self.counts = {ws_topic: 0 for ws_topic in topic_map}
        for ws_topic, ros_topic in topic_map.items():
            self.get_logger().info(f'Bridge WebSocket {ws_topic} -> ROS2 {ros_topic}')
            self.get_logger().info(f'Bridge WebSocket {ws_topic} -> ROS2 path {path_topic_map[ws_topic]}')

    def publish_ws_odom(self, ws_topic: str, data: dict[str, Any], debug: bool = False) -> None:
        publisher = self.odom_publishers.get(ws_topic)
        if publisher is None:
            return
        odom = odom_json_to_msg(data, self)
        if self.restamp_published:
            odom = copy.deepcopy(odom)
            odom.header.stamp = self.get_clock().now().to_msg()
        publisher.publish(odom)
        self.publish_odom_path(ws_topic, odom)
        self.counts[ws_topic] += 1
        count = self.counts[ws_topic]
        if debug or count == 1 or count % 100 == 0:
            pos = odom.pose.pose.position
            path_len = len(self.paths[ws_topic].poses)
            self.get_logger().info(
                f'Published {ws_topic} #{count} to {self.topic_map[ws_topic]}: '
                f'stamp={odom.header.stamp.sec}.{odom.header.stamp.nanosec:09d}, '
                f'frame_id={odom.header.frame_id}, x={pos.x:.3f}, y={pos.y:.3f}, '
                f'path={self.path_topic_map[ws_topic]} len={path_len}'
            )

    def publish_odom_path(self, ws_topic: str, odom: Odometry) -> None:
        path_publisher = self.path_publishers.get(ws_topic)
        path = self.paths.get(ws_topic)
        if path_publisher is None or path is None:
            return
        pose = PoseStamped()
        pose.header = odom.header
        pose.pose = odom.pose.pose
        path.header.stamp = odom.header.stamp
        path.header.frame_id = odom.header.frame_id
        path.poses.append(pose)
        if self.max_path_length > 0 and len(path.poses) > self.max_path_length:
            path.poses = path.poses[-self.max_path_length:]
        path_publisher.publish(path)


async def subscribe_dual_odom(
    uri: str,
    node: AgileXDualOdomBridge,
    ws_topics: list[str],
    sub_type: str,
    debug: bool,
) -> None:
    if websockets is None:
        raise RuntimeError('Missing Python package websockets. Run in foundation_nav or install websockets.')

    connect_kwargs: dict[str, Any] = {}
    if 'proxy' in inspect.signature(websockets.connect).parameters:
        connect_kwargs['proxy'] = None

    async with websockets.connect(uri, ping_interval=None, **connect_kwargs) as websocket:
        print(f'Connected to AgileX WebSocket: {uri}')
        for topic in ws_topics:
            request = {
                'op': 'subscribe',
                'topic': topic,
                'type': sub_type,
            }
            await websocket.send(json.dumps(request))
            print('Sent subscribe request:')
            print(json.dumps(request, ensure_ascii=False, indent=2))

        async for response in websocket:
            if debug:
                print(response)
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                continue

            topic = data.get('topic')
            if data.get('op') != 'publish' or topic not in ws_topics:
                continue
            node.publish_ws_odom(topic, data, debug)
            rclpy.spin_once(node, timeout_sec=0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Subscribe AgileX /odom_filtered and /odom over WebSocket and publish both as ROS2 Odometry.'
    )
    parser.add_argument('--uri', default=DEFAULT_URI, help=f'AgileX WebSocket URI, default: {DEFAULT_URI}')
    parser.add_argument('--filtered-ws-topic', default=DEFAULT_FILTERED_WS_TOPIC)
    parser.add_argument('--odom-ws-topic', default=DEFAULT_ODOM_WS_TOPIC)
    parser.add_argument('--filtered-ros-topic', default=DEFAULT_FILTERED_ROS_TOPIC)
    parser.add_argument('--odom-ros-topic', default=DEFAULT_ODOM_ROS_TOPIC)
    parser.add_argument('--filtered-path-ros-topic', default=DEFAULT_FILTERED_PATH_ROS_TOPIC)
    parser.add_argument('--odom-path-ros-topic', default=DEFAULT_ODOM_PATH_ROS_TOPIC)
    parser.add_argument('--max-path-length', type=int, default=DEFAULT_MAX_PATH_LENGTH)
    parser.add_argument('--sub-type', default=DEFAULT_SUB_TYPE)
    parser.add_argument('--restamp-published', action='store_true')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topic_map = {
        args.filtered_ws_topic: args.filtered_ros_topic,
        args.odom_ws_topic: args.odom_ros_topic,
    }
    path_topic_map = {
        args.filtered_ws_topic: args.filtered_path_ros_topic,
        args.odom_ws_topic: args.odom_path_ros_topic,
    }
    ws_topics = list(topic_map.keys())

    rclpy.init()
    node = AgileXDualOdomBridge(topic_map, path_topic_map, args.restamp_published, args.max_path_length)
    try:
        while rclpy.ok():
            try:
                asyncio.run(subscribe_dual_odom(args.uri, node, ws_topics, args.sub_type, args.debug))
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
