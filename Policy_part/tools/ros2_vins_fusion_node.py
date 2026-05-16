#!/usr/bin/env python3
import csv
import json
import math
import os
from collections import deque
from dataclasses import dataclass
from threading import Lock

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import NavSatFix
from visualization_msgs.msg import Marker, MarkerArray


WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


@dataclass
class JsonWaypoint:
    latitude: float
    longitude: float
    altitude: float = 33.0
    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0


def stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def sec_to_msg_time(seconds, clock):
    msg = clock.now().to_msg()
    msg.sec = int(math.floor(seconds))
    msg.nanosec = int(round((seconds - msg.sec) * 1e9))
    if msg.nanosec >= 1000000000:
        msg.sec += 1
        msg.nanosec -= 1000000000
    return msg


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    q = q / norm
    if q[0] < 0.0:
        q = -q
    return q


def quat_to_xyzw(q):
    q = quat_normalize(q)
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


def quat_from_xyzw(q):
    return quat_normalize([q[3], q[0], q[1], q[2]])


def quat_to_matrix(q):
    return Rotation.from_quat(quat_to_xyzw(q)).as_matrix()


def matrix_to_quat(matrix):
    return quat_from_xyzw(Rotation.from_matrix(matrix).as_quat())


def rotvec_to_quat(rotvec):
    return quat_from_xyzw(Rotation.from_rotvec(rotvec).as_quat())


def quat_to_rotvec(q):
    return Rotation.from_quat(quat_to_xyzw(q)).as_rotvec()


def quat_inverse(q):
    q = quat_normalize(q)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = quat_normalize(q1)
    w2, x2, y2, z2 = quat_normalize(q2)
    return quat_normalize([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def relative_pose(p_i, q_i, p_j, q_j):
    r_i = quat_to_matrix(q_i)
    rel_p = r_i.T @ (p_j - p_i)
    rel_q = quat_multiply(quat_inverse(q_i), q_j)
    return rel_p, rel_q


def llh_to_ecef(latitude, longitude, altitude):
    lat = math.radians(latitude)
    lon = math.radians(longitude)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + altitude) * cos_lat * cos_lon
    y = (n + altitude) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + altitude) * sin_lat
    return np.array([x, y, z], dtype=float)


def ecef_to_enu(ecef, origin_ecef, origin_latitude, origin_longitude):
    lat = math.radians(origin_latitude)
    lon = math.radians(origin_longitude)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    dx, dy, dz = ecef - origin_ecef
    return np.array([
        -sin_lon * dx + cos_lon * dy,
        -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz,
        cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz,
    ], dtype=float)


def find_key_recursive(value, target_key):
    if isinstance(value, dict):
        if target_key in value:
            return value[target_key]
        for child in value.values():
            found = find_key_recursive(child, target_key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_key_recursive(child, target_key)
            if found is not None:
                return found
    return None


def first_number(mapping, keys, default=None):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return float(mapping[key])
    return default


def parse_waypoints_from_json(data, default_altitude):
    route_polyline = find_key_recursive(data, 'route_polyline')
    if not isinstance(route_polyline, list):
        return []

    waypoints = []
    for item in route_polyline:
        if isinstance(item, dict):
            lat = first_number(item, ('latitude', 'lat', 'y'))
            lon = first_number(item, ('longitude', 'lon', 'lng', 'x'))
            if lat is None or lon is None:
                continue
            waypoints.append(JsonWaypoint(
                latitude=lat,
                longitude=lon,
                altitude=first_number(item, ('altitude', 'alt', 'z'), default_altitude),
                qw=first_number(item, ('qw', 'w'), 1.0),
                qx=first_number(item, ('qx',), 0.0),
                qy=first_number(item, ('qy',), 0.0),
                qz=first_number(item, ('qz',), 0.0),
            ))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            # Route arrays are encoded as [longitude, latitude, altitude?].
            altitude = float(item[2]) if len(item) >= 3 and item[2] is not None else default_altitude
            waypoints.append(JsonWaypoint(
                latitude=float(item[1]),
                longitude=float(item[0]),
                altitude=altitude,
            ))
    return waypoints


class GlobalOptimization:
    def __init__(self, clock):
        self.clock = clock
        self.local_pose_map = {}
        self.global_pose_map = {}
        self.gps_position_map = {}
        self.origin_ecef = None
        self.origin_latitude = None
        self.origin_longitude = None
        self.wgps_r_wvio = np.eye(3)
        self.wgps_t_wvio = np.zeros(3)
        self.last_p = np.zeros(3)
        self.last_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.new_gps = False
        self.path_msg = Path()
        self.lock = Lock()

    def gps_to_xyz(self, latitude, longitude, altitude):
        ecef = llh_to_ecef(latitude, longitude, altitude)
        if self.origin_ecef is None:
            self.origin_ecef = ecef
            self.origin_latitude = latitude
            self.origin_longitude = longitude
        return ecef_to_enu(ecef, self.origin_ecef, self.origin_latitude, self.origin_longitude)

    def input_odom(self, t, odom_p, odom_q):
        with self.lock:
            odom_q = quat_normalize(odom_q)
            self.local_pose_map[t] = np.concatenate((odom_p, odom_q))

            global_p = self.wgps_r_wvio @ odom_p + self.wgps_t_wvio
            global_q = matrix_to_quat(self.wgps_r_wvio @ quat_to_matrix(odom_q))
            self.global_pose_map[t] = np.concatenate((global_p, global_q))
            self.last_p = global_p
            self.last_q = global_q
            self._update_path_locked()

    def input_gps(self, t, latitude, longitude, altitude, pos_accuracy):
        xyz = self.gps_to_xyz(latitude, longitude, altitude)
        self.gps_position_map[t] = np.array([xyz[0], xyz[1], xyz[2], max(pos_accuracy, 1e-6)], dtype=float)
        self.new_gps = True

    def get_global_odom(self):
        with self.lock:
            return self.last_p.copy(), self.last_q.copy()

    def optimize(self, logger):
        if not self.new_gps:
            return

        with self.lock:
            if len(self.local_pose_map) < 1:
                self.new_gps = False
                return
            self.new_gps = False

            times = sorted(self.local_pose_map.keys())
            index_by_time = {t: i for i, t in enumerate(times)}
            initial = np.zeros(len(times) * 6, dtype=float)
            for i, t in enumerate(times):
                pose = self.global_pose_map[t]
                initial[i * 6:i * 6 + 3] = pose[:3]
                initial[i * 6 + 3:i * 6 + 6] = quat_to_rotvec(pose[3:7])

            local_relative = []
            for t_i, t_j in zip(times[:-1], times[1:]):
                pose_i = self.local_pose_map[t_i]
                pose_j = self.local_pose_map[t_j]
                rel_p, rel_q = relative_pose(pose_i[:3], pose_i[3:7], pose_j[:3], pose_j[3:7])
                local_relative.append((index_by_time[t_i], index_by_time[t_j], rel_p, rel_q))

            gps_constraints = []
            for t, gps_xyz_acc in self.gps_position_map.items():
                if t in index_by_time:
                    gps_constraints.append((index_by_time[t], gps_xyz_acc[:3], gps_xyz_acc[3]))

        if not gps_constraints and not local_relative:
            return

        logger.info('global optimization')

        def unpack(params, i):
            base = i * 6
            return params[base:base + 3], rotvec_to_quat(params[base + 3:base + 6])

        def residuals(params):
            result = []
            for i, j, measured_p, measured_q in local_relative:
                p_i, q_i = unpack(params, i)
                p_j, q_j = unpack(params, j)
                candidate_p, candidate_q = relative_pose(p_i, q_i, p_j, q_j)
                result.extend((candidate_p - measured_p) / 0.1)
                q_err = quat_multiply(quat_inverse(measured_q), candidate_q)
                result.extend((2.0 * q_err[1:4]) / 0.01)
            for i, gps_xyz, pos_accuracy in gps_constraints:
                p_i, _ = unpack(params, i)
                result.extend((p_i - gps_xyz) / pos_accuracy)
            return np.asarray(result, dtype=float)

        try:
            solution = least_squares(
                residuals,
                initial,
                loss='huber',
                f_scale=1.0,
                max_nfev=20,
            )
        except Exception as exc:
            logger.error(f'Global optimization failed: {exc}')
            return

        with self.lock:
            for i, t in enumerate(times):
                p, q = unpack(solution.x, i)
                self.global_pose_map[t] = np.concatenate((p, q))
                if i == len(times) - 1:
                    local_pose = self.local_pose_map[t]
                    local_r = quat_to_matrix(local_pose[3:7])
                    global_r = quat_to_matrix(q)
                    self.wgps_r_wvio = global_r @ local_r.T
                    self.wgps_t_wvio = p - self.wgps_r_wvio @ local_pose[:3]
                    self.last_p = p
                    self.last_q = q
            self._update_path_locked()

    def _update_path_locked(self):
        path = Path()
        path.header.frame_id = 'world'
        if self.global_pose_map:
            path.header.stamp = sec_to_msg_time(max(self.global_pose_map.keys()), self.clock)
        for t in sorted(self.global_pose_map.keys()):
            pose = self.global_pose_map[t]
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = sec_to_msg_time(t, self.clock)
            pose_stamped.header.frame_id = 'world'
            pose_stamped.pose.position.x = float(pose[0])
            pose_stamped.pose.position.y = float(pose[1])
            pose_stamped.pose.position.z = float(pose[2])
            pose_stamped.pose.orientation.w = float(pose[3])
            pose_stamped.pose.orientation.x = float(pose[4])
            pose_stamped.pose.orientation.y = float(pose[5])
            pose_stamped.pose.orientation.z = float(pose[6])
            path.poses.append(pose_stamped)
        self.path_msg = path


class GlobalFusionNode(Node):
    def __init__(self):
        super().__init__('globalEstimator')

        self.declare_parameter('route_json_path', '/home/nuvo/SR/foundation-policy/Policy_part/topomaps/LLM_Policy/route_waypoints.json')
        self.declare_parameter('route_output_path', '/home/nuvo/SR/foundation-policy/Policy_part/topomaps/LLM_Policy/LLM_UTM_odom_poses.txt')
        self.declare_parameter('vio_global_csv_path', '/home/nuvo/SR/foundation-policy/Policy_part/topomaps/LLM_Policy/gkj/output/vio_global.csv')
        self.declare_parameter('gps_topic', '/gps/fix')
        self.declare_parameter('vio_topic', '/mfla/dr_odom')
        self.declare_parameter('sync_tolerance_sec', 0.01)
        self.declare_parameter('route_default_altitude', 33.0)
        self.declare_parameter('publish_car_model', False)
        self.declare_parameter('route_poll_sec', 1.0)
        self.declare_parameter('optimization_period_sec', 2.0)

        self.estimator = GlobalOptimization(self.get_clock())
        self.gps_queue = deque()
        self.gps_lock = Lock()
        self.last_route_id = ''

        self.sub_gps = self.create_subscription(
            NavSatFix,
            self.get_parameter('gps_topic').value,
            self.gps_callback,
            100,
        )
        self.sub_vio = self.create_subscription(
            Odometry,
            self.get_parameter('vio_topic').value,
            self.vio_callback,
            100,
        )
        self.pub_global_path = self.create_publisher(Path, '~/global_path', 100)
        self.pub_global_odometry = self.create_publisher(Odometry, '~/global_odometry', 100)
        self.pub_car = self.create_publisher(MarkerArray, '~/car_model', 1000)

        self.route_timer = self.create_timer(
            float(self.get_parameter('route_poll_sec').value),
            self.route_file_timer_callback,
        )
        self.optimization_timer = self.create_timer(
            float(self.get_parameter('optimization_period_sec').value),
            lambda: self.estimator.optimize(self.get_logger()),
        )

        self.get_logger().info(f"Watching route JSON file: {self.get_parameter('route_json_path').value}")
        self.get_logger().info(f"Route UTM output file: {self.get_parameter('route_output_path').value}")

    def gps_callback(self, msg):
        with self.gps_lock:
            self.gps_queue.append(msg)

    def vio_callback(self, pose_msg):
        t = stamp_to_sec(pose_msg.header.stamp)
        vio_t = np.array([
            pose_msg.pose.pose.position.x,
            pose_msg.pose.pose.position.y,
            pose_msg.pose.pose.position.z,
        ], dtype=float)
        vio_q = np.array([
            pose_msg.pose.pose.orientation.w,
            pose_msg.pose.pose.orientation.x,
            pose_msg.pose.pose.orientation.y,
            pose_msg.pose.pose.orientation.z,
        ], dtype=float)
        self.estimator.input_odom(t, vio_t, vio_q)
        self._consume_matching_gps(t)

        global_t, global_q = self.estimator.get_global_odom()
        odometry = Odometry()
        odometry.header = pose_msg.header
        odometry.header.frame_id = 'world'
        odometry.child_frame_id = 'world'
        odometry.pose.pose.position.x = float(global_t[0])
        odometry.pose.pose.position.y = float(global_t[1])
        odometry.pose.pose.position.z = float(global_t[2])
        odometry.pose.pose.orientation.w = float(global_q[0])
        odometry.pose.pose.orientation.x = float(global_q[1])
        odometry.pose.pose.orientation.y = float(global_q[2])
        odometry.pose.pose.orientation.z = float(global_q[3])

        self.pub_global_odometry.publish(odometry)
        self.pub_global_path.publish(self.estimator.path_msg)
        if self.get_parameter('publish_car_model').value:
            self.publish_car_model(t, global_t, global_q)
        self._append_global_csv(pose_msg.header.stamp, global_t, global_q)

    def _consume_matching_gps(self, vio_t):
        tolerance = float(self.get_parameter('sync_tolerance_sec').value)
        with self.gps_lock:
            while self.gps_queue:
                gps_msg = self.gps_queue[0]
                gps_t = stamp_to_sec(gps_msg.header.stamp)
                if gps_t < vio_t - tolerance:
                    self.gps_queue.popleft()
                    continue
                if gps_t > vio_t + tolerance:
                    break

                pos_accuracy = gps_msg.position_covariance[0]
                if pos_accuracy <= 0.0 or math.isnan(pos_accuracy):
                    pos_accuracy = 1.0
                self.estimator.input_gps(
                    vio_t,
                    gps_msg.latitude,
                    gps_msg.longitude,
                    gps_msg.altitude,
                    pos_accuracy,
                )
                self.gps_queue.popleft()
                break

    def route_file_timer_callback(self):
        route_json_path = self.get_parameter('route_json_path').value
        route_output_path = self.get_parameter('route_output_path').value
        try:
            with open(route_json_path, 'r', encoding='utf-8') as fin:
                data = json.load(fin)
        except FileNotFoundError:
            self.get_logger().warn(f'Route JSON file not readable: {route_json_path}')
            return
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f'Route JSON decode failed: {exc}')
            return

        route_id = data.get('route_id')
        if not route_id:
            self.get_logger().warn(f'Route JSON missing top-level route_id in {route_json_path}')
            return
        if self.last_route_id and route_id == self.last_route_id:
            return

        default_altitude = float(self.get_parameter('route_default_altitude').value)
        waypoints = parse_waypoints_from_json(data, default_altitude)
        if not waypoints:
            self.get_logger().warn(f'Route JSON changed but no valid waypoints found in {route_json_path}')
            return

        self.convert_json_waypoints_to_utm(waypoints, route_output_path)
        self.last_route_id = route_id

    def convert_json_waypoints_to_utm(self, waypoints, route_output_path):
        self.get_logger().info(f'Received route with {len(waypoints)} waypoints')
        output_dir = os.path.dirname(route_output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(route_output_path, 'w', newline='', encoding='utf-8') as fout:
            writer = csv.writer(fout)
            writer.writerow(['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'])
            for i, waypoint in enumerate(waypoints, start=1):
                xyz = self.estimator.gps_to_xyz(waypoint.latitude, waypoint.longitude, waypoint.altitude)
                writer.writerow([
                    f'{xyz[0]:.15f}',
                    f'{xyz[1]:.15f}',
                    f'{xyz[2]:.15f}',
                    f'{waypoint.qw:.15f}',
                    f'{waypoint.qx:.15f}',
                    f'{waypoint.qy:.15f}',
                    f'{waypoint.qz:.15f}',
                ])
                self.get_logger().info(
                    'Waypoint %d: Lat=%.6f, Lon=%.6f, Alt=%.2f -> ENU(%.2f, %.2f, %.2f)'
                    % (i, waypoint.latitude, waypoint.longitude, waypoint.altitude, xyz[0], xyz[1], xyz[2])
                )
        self.get_logger().info(f'Route conversion completed. Results saved to {route_output_path}')

    def _append_global_csv(self, stamp, global_t, global_q):
        csv_path = self.get_parameter('vio_global_csv_path').value
        output_dir = os.path.dirname(csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        timestamp_ns = stamp.sec * 1000000000 + stamp.nanosec
        with open(csv_path, 'a', newline='', encoding='utf-8') as fout:
            writer = csv.writer(fout)
            writer.writerow([
                str(timestamp_ns),
                f'{global_t[0]:.5f}',
                f'{global_t[1]:.5f}',
                f'{global_t[2]:.5f}',
                f'{global_q[0]:.5f}',
                f'{global_q[1]:.5f}',
                f'{global_q[2]:.5f}',
                f'{global_q[3]:.5f}',
            ])

    def publish_car_model(self, t, position, orientation):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.stamp = sec_to_msg_time(t, self.get_clock())
        marker.header.frame_id = 'world'
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.id = 0
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        marker.pose.orientation.w = float(orientation[0])
        marker.pose.orientation.x = float(orientation[1])
        marker.pose.orientation.y = float(orientation[2])
        marker.pose.orientation.z = float(orientation[3])
        marker.scale.x = 2.0
        marker.scale.y = 1.0
        marker.scale.z = 0.5
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker_array.markers.append(marker)
        self.pub_car.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = GlobalFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
