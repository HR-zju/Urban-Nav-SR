#!/usr/bin/env python3
"""
ros2_ws_bridge.py — WebSocket → ROS2 Topic 桥接

运行在车上(Raspberry Pi)，接收 Mac 发来的 GPS 航点，转发到 ROS2 topic

Usage:
    ros2 run nav_bridge ros2_ws_bridge --ros-args -p ws_host:=0.0.0.0 -p ws_port:=8765
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
import json
import asyncio
import threading
import argparse

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False


class WaypointWSBridge(Node):
    """WebSocket → ROS2 Topic 桥接节点"""

    def __init__(self, ws_host: str = "0.0.0.0", ws_port: int = 8765):
        super().__init__('ros2_ws_bridge')

        self.declare_parameter('ws_host', ws_host)
        self.declare_parameter('ws_port', ws_port)
        self.declare_parameter('frame_id', 'map')

        host = self.get_parameter('ws_host').value
        port = self.get_parameter('ws_port').value
        self.frame_id = self.get_parameter('frame_id').value

        self.publisher = self.create_publisher(
            'nav_bridge/NavWaypoint',
            '/navigation/waypoints',
            10
        )

        self.get_logger().info(f'WebSocket bridge ready')
        self.get_logger().info(f'WS server: ws://{host}:{port}')
        self.get_logger().info(f'ROS2 topic: /navigation/waypoints')

        if not WEBSOCKETS_AVAILABLE:
            self.get_logger().error('websockets library not found. Install: pip install websockets')
            return

        # 在独立线程中运行 asyncio WebSocket server
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_ws_server, args=(host, port), daemon=True)
        self._thread.start()

    def _run_ws_server(self, host: str, port: int):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._ws_server(host, port))

    async def _ws_server(self, host: str, port: int):
        """WebSocket server，接收航点数据并发布到 ROS2"""
        async def handler(websocket, path):
            self.get_logger().info(f'Client connected from {websocket.remote_address}')
            try:
                async for raw in websocket:
                    try:
                        data = json.loads(raw)
                        self._handle_waypoint_msg(data)
                    except json.JSONDecodeError:
                        self.get_logger().warn(f'Invalid JSON: {raw[:100]}')
                    except Exception as e:
                        self.get_logger().error(f'Error handling message: {e}')
            except websockets.exceptions.ConnectionClosed:
                self.get_logger().info('Client disconnected')

        server = await websockets.serve(handler, host, port)
        self.get_logger().info(f'WebSocket server started on ws://{host}:{port}')
        await asyncio.Future()  # 永久运行

    def _handle_waypoint_msg(self, data):
        """
        处理收到的航点数据，发布到 ROS2 topic

        支持两种格式:
        1. action_send_ws 格式 (新):
           {"route_id": "...", "waypoints": [{"waypoint_index": 0, "longitude": 121.4737, "latitude": 31.2304, ...}]}
        2. 旧格式 (兼容):
           {"waypoints": [{"location": "lon,lat", "waypoint_index": 0, ...}]}
        """
        route_id = data.get('route_id', '')

        waypoints = data.get('waypoints', [])
        if not waypoints:
            # 单个航点（旧格式）
            if 'location' in data:
                waypoints = [data]
            else:
                self.get_logger().warn(f'No waypoints in message: {list(data.keys())}')
                return

        for wp in waypoints:
            self._publish_waypoint(wp, route_id)

    def _publish_waypoint(self, wp: dict, route_id: str = ''):
        from nav_bridge.msg import NavWaypoint
        msg = NavWaypoint()

        # 支持两种格式:
        # 1. location = "lon,lat" (旧)
        # 2. longitude + latitude 分离 (新, action_send_ws)
        location = wp.get('location', '')
        if location and ',' in location:
            try:
                lon_str, lat_str = location.split(',')
                msg.latitude = float(lat_str)
                msg.longitude = float(lon_str)
            except (ValueError, AttributeError):
                self.get_logger().warn(f'Invalid location: {location}')
                return
        else:
            # 新格式: longitude/latitude 分离
            try:
                msg.longitude = float(wp.get('longitude', 0))
                msg.latitude = float(wp.get('latitude', 0))
                if msg.longitude == 0 and msg.latitude == 0:
                    self.get_logger().warn(f'No valid location in waypoint: {wp}')
                    return
            except (ValueError, TypeError):
                self.get_logger().warn(f'Invalid longitude/latitude: {wp}')
                return

        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.waypoint_index = int(wp.get('waypoint_index', wp.get('global_waypoint_index', 0)))
        msg.is_terminal = bool(wp.get('is_terminal', wp.get('is_leg_terminal', False)))
        msg.is_via_stop = bool(wp.get('is_via_stop', False))
        msg.action = str(wp.get('action', wp.get('action_raw', '')))
        msg.instruction = str(wp.get('instruction', wp.get('instruction_raw', '')))
        msg.distance_m = float(wp.get('distance_m', 0.0))
        msg.duration_s = float(wp.get('duration_s', 0.0))
        msg.leg_index = int(wp.get('leg_index', 0))
        msg.leg_origin = str(wp.get('leg_origin', ''))
        msg.leg_destination = str(wp.get('leg_destination', ''))

        self.publisher.publish(msg)
        self.get_logger().info(
            f'Published wp[{msg.waypoint_index}]: {msg.latitude},{msg.longitude} '
            f'terminal={msg.is_terminal} via={msg.is_via_stop}'
        )


def main(args=None):
    parser = argparse.ArgumentParser(description='ROS2 WebSocket Bridge')
    parser.add_argument('--ws-host', default='0.0.0.0', help='WebSocket host')
    parser.add_argument('--ws-port', type=int, default=8765, help='WebSocket port')
    parsed_args, ros_args = parser.parse_known_args(args)

    rclpy.init(args=ros_args)
    node = WaypointWSBridge(ws_host=parsed_args.ws_host, ws_port=parsed_args.ws_port)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
