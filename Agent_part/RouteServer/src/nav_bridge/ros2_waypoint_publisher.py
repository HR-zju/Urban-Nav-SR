#!/usr/bin/env python3
"""
ros2_waypoint_publisher.py — 持续发布 GPS 航点到 ROS2 topic

Usage (直接运行):
    ros2 run nav_bridge ros2_waypoint_publisher.py --ros-args -p waypoints:='[{"location":"121.474588,31.230221"},{"location":"121.473667,31.230525"}]'

Usage (作为节点运行):
    from nav_bridge.ros2_waypoint_publisher import WaypointPublisher
    # 在 launch 中使用
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
import json
import sys
import argparse


class WaypointPublisher(Node):
    """持续发布 GPS 航点到 ROS2 topic的节点"""

    def __init__(self, waypoints_json: str = None):
        super().__init__('ros2_waypoint_publisher')

        self.declare_parameter('waypoints', '[]')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('publish_interval', 1.0)  # 秒

        self.publisher = self.create_publisher(
            NavSatFix,
            '/navigation/waypoints',
            10
        )

        self.waypoints = []
        self.current_idx = 0

        # 从参数加载 waypoints
        if waypoints_json:
            try:
                self.waypoints = json.loads(waypoints_json)
                self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints from args')
            except json.JSONDecodeError as e:
                self.get_logger().error(f'Failed to parse waypoints: {e}')
        else:
            # 从 ros 参数加载
            waypoints_param = self.get_parameter('waypoints').value
            if waypoints_param and waypoints_param != '[]':
                try:
                    self.waypoints = json.loads(waypoints_param) if isinstance(waypoints_param, str) else waypoints_param
                    self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints from parameter')
                except Exception as e:
                    self.get_logger().error(f'Failed to load waypoints: {e}')

        # 启动定时发布
        interval = self.get_parameter('publish_interval').value
        self.timer = self.create_timer(interval, self.publish_next)

        if self.waypoints:
            self.get_logger().info(f'Waypoint publisher ready: {len(self.waypoints)} waypoints')
            self.get_logger().info(f'Topic: /navigation/waypoints')
        else:
            self.get_logger().warn('No waypoints loaded, waiting...')

    def set_waypoints(self, waypoints):
        """动态设置航点列表"""
        self.waypoints = waypoints
        self.current_idx = 0
        self.get_logger().info(f'Set {len(waypoints)} waypoints')

    def publish_next(self):
        """发布下一个航点"""
        if not self.waypoints:
            return

        if self.current_idx >= len(self.waypoints):
            self.get_logger().info('All waypoints published')
            return

        wp = self.waypoints[self.current_idx]
        location = wp.get('location', '')

        if not location:
            self.current_idx += 1
            return

        try:
            lon, lat = map(float, location.split(','))
        except (ValueError, AttributeError) as e:
            self.get_logger().error(f'Invalid location format: {location}')
            self.current_idx += 1
            return

        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.get_parameter('frame_id').value

        # 注意：MCP 输出是 lon,lat， ROS2 NavSatFix 用 lat,lon
        msg.latitude = lat
        msg.longitude = lon
        msg.altitude = 0.0

        # 精度字段设为 NaN 表示未知
        msg.position_covariance_type = 0

        self.publisher.publish(msg)
        self.get_logger().debug(f'Published [{self.current_idx}/{len(self.waypoints)}]: {lat},{lon}')

        self.current_idx += 1

    def publish_all_once(self):
        """一次性发布所有航点（不等待）"""
        for i, wp in enumerate(self.waypoints):
            location = wp.get('location', '')
            if not location:
                continue

            try:
                lon, lat = map(float, location.split(','))
            except (ValueError, AttributeError):
                continue

            msg = NavSatFix()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.get_parameter('frame_id').value
            msg.latitude = lat
            msg.longitude = lon
            msg.altitude = 0.0
            msg.position_covariance_type = 0

            self.publisher.publish(msg)

        self.get_logger().info(f'Published all {len(self.waypoints)} waypoints at once')


def main(args=None):
    parser = argparse.ArgumentParser(description='ROS2 Waypoint Publisher')
    parser.add_argument('--waypoints', '-w', type=str, default='[]',
                      help='JSON array of waypoints, e.g. \'[{"location":"121.474,31.23"},{"location":"121.475,31.231"}]\'')
    parsed_args, ros_args = parser.parse_known_args(args)

    rclpy.init(args=ros_args)
    node = WaypointPublisher(waypoints_json=parsed_args.waypoints)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()