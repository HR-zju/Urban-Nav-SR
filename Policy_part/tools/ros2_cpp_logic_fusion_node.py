#!/usr/bin/env python3
import csv
import json
import math
import os
from collections import deque
from dataclasses import dataclass
from threading import Event, Lock, Thread

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import NavSatFix


WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3
POLICY_PART_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def sec_to_msg_time(seconds):
    sec = int(math.floor(seconds))
    nanosec = int(round((seconds - sec) * 1e9))
    if nanosec >= 1000000000:
        sec += 1
        nanosec -= 1000000000
    msg = rclpy.time.Time(seconds=sec, nanoseconds=nanosec).to_msg()
    msg.sec = sec
    msg.nanosec = nanosec
    return msg


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    q = q / norm
    return -q if q[0] < 0.0 else q


def quat_to_rotation(q):
    q = quat_normalize(q)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def rotation_to_quat(rotation):
    x, y, z, w = rotation.as_quat()
    return quat_normalize([w, x, y, z])


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = quat_normalize(q1)
    w2, x2, y2, z2 = quat_normalize(q2)
    return quat_normalize([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_inverse(q):
    w, x, y, z = quat_normalize(q)
    return np.array([w, -x, -y, -z], dtype=float)


def pose_to_matrix(p, q):
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = quat_to_rotation(q).as_matrix()
    transform[:3, 3] = np.asarray(p, dtype=float)
    return transform


def matrix_to_pose(transform):
    p = np.asarray(transform[:3, 3], dtype=float)
    q = rotation_to_quat(Rotation.from_matrix(transform[:3, :3]))
    return p, q


def llh_to_ecef(latitude, longitude, altitude):
    lat = math.radians(latitude)
    lon = math.radians(longitude)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    return np.array([
        (n + altitude) * cos_lat * cos_lon,
        (n + altitude) * cos_lat * sin_lon,
        (n * (1.0 - WGS84_E2) + altitude) * sin_lat,
    ], dtype=float)


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


class Ros1GlobalFusionEstimator:
    def __init__(
        self,
        max_optimization_poses=0,
        optimization_mode='alignment',
        alignment_min_gps_constraints=10,
        alignment_min_baseline_m=5.0,
        freeze_alignment_after_init=True,
    ):
        self.lock = Lock()
        self.max_optimization_poses = int(max_optimization_poses)
        self.optimization_mode = str(optimization_mode).lower()
        self.alignment_min_gps_constraints = int(alignment_min_gps_constraints)
        self.alignment_min_baseline_m = float(alignment_min_baseline_m)
        self.freeze_alignment_after_init = bool(freeze_alignment_after_init)
        self.local_pose_map = {}
        self.global_pose_map = {}
        self.gps_position_map = {}
        self.origin_ecef = None
        self.origin_latitude = None
        self.origin_longitude = None
        self.wgps_t_wvio = np.eye(4, dtype=float)
        self.last_p = np.zeros(3, dtype=float)
        self.last_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.new_gps = False
        self.has_alignment = False
        self.alignment_status = 'waiting for matched GPS/VIO constraints'
        self.path_msg = Path()

    def gps_to_xyz(self, latitude, longitude, altitude):
        ecef = llh_to_ecef(latitude, longitude, altitude)
        if self.origin_ecef is None:
            self.origin_ecef = ecef
            self.origin_latitude = latitude
            self.origin_longitude = longitude
        return ecef_to_enu(ecef, self.origin_ecef, self.origin_latitude, self.origin_longitude)

    def has_gps_origin(self):
        with self.lock:
            return self.origin_ecef is not None

    def input_odom(self, t, odom_p, odom_q):
        with self.lock:
            odom_q = quat_normalize(odom_q)
            self.local_pose_map[t] = np.concatenate((odom_p, odom_q))

            local_t = pose_to_matrix(odom_p, odom_q)
            global_t = self.wgps_t_wvio @ local_t
            global_p, global_q = matrix_to_pose(global_t)
            self.global_pose_map[t] = np.concatenate((global_p, global_q))
            self.last_p = global_p
            self.last_q = global_q
            if self.has_alignment:
                self._append_path_pose_locked(t, global_p, global_q)

    def input_gps(self, t, latitude, longitude, altitude, pos_accuracy):
        xyz = self.gps_to_xyz(latitude, longitude, altitude)
        with self.lock:
            self.gps_position_map[t] = np.array(
                [xyz[0], xyz[1], xyz[2], max(float(pos_accuracy), 1e-6)],
                dtype=float,
            )
            self.new_gps = True

    def optimize(self):
        if self.optimization_mode == 'pose_graph':
            return self._optimize_pose_graph()
        return self._optimize_alignment()

    @staticmethod
    def _estimate_planar_alignment(local_points, gps_points, weights):
        local_points = np.asarray(local_points, dtype=float)
        gps_points = np.asarray(gps_points, dtype=float)
        weights = np.asarray(weights, dtype=float)
        weights = np.maximum(weights, 1e-6)
        weights = weights / np.sum(weights)

        local_centroid = np.sum(local_points * weights[:, None], axis=0)
        gps_centroid = np.sum(gps_points * weights[:, None], axis=0)
        local_xy = local_points[:, :2] - local_centroid[:2]
        gps_xy = gps_points[:, :2] - gps_centroid[:2]

        cross = np.sum(weights * (local_xy[:, 0] * gps_xy[:, 1] - local_xy[:, 1] * gps_xy[:, 0]))
        dot = np.sum(weights * (local_xy[:, 0] * gps_xy[:, 0] + local_xy[:, 1] * gps_xy[:, 1]))
        yaw = math.atan2(cross, dot)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        rotation = np.array([
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)
        translation = gps_centroid - rotation @ local_centroid

        transform = np.eye(4, dtype=float)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        return transform

    def _optimize_alignment(self):
        with self.lock:
            if not self.new_gps:
                return False
            self.new_gps = False
            if self.has_alignment and self.freeze_alignment_after_init:
                return False

            times = sorted(t for t in self.gps_position_map.keys() if t in self.local_pose_map)
            if len(times) < self.alignment_min_gps_constraints:
                self.alignment_status = (
                    'waiting for alignment: matched_gps=%d/%d'
                    % (len(times), self.alignment_min_gps_constraints)
                )
                return False

            local_points = np.array([self.local_pose_map[t][:3] for t in times], dtype=float)
            gps_points = np.array([self.gps_position_map[t][:3] for t in times], dtype=float)
            accuracies = np.array([self.gps_position_map[t][3] for t in times], dtype=float)
            gps_xy_span = np.linalg.norm(
                np.max(gps_points[:, :2], axis=0) - np.min(gps_points[:, :2], axis=0)
            )
            if gps_xy_span < self.alignment_min_baseline_m:
                self.alignment_status = (
                    'waiting for alignment baseline: matched_gps=%d baseline=%.2fm/%.2fm'
                    % (len(times), gps_xy_span, self.alignment_min_baseline_m)
                )
                return False

            weights = 1.0 / np.maximum(accuracies, 1e-6)
            self.wgps_t_wvio = self._estimate_planar_alignment(local_points, gps_points, weights)

            for t, local_pose in self.local_pose_map.items():
                local_t = pose_to_matrix(local_pose[:3], local_pose[3:7])
                global_t = self.wgps_t_wvio @ local_t
                global_p, global_q = matrix_to_pose(global_t)
                self.global_pose_map[t] = np.concatenate((global_p, global_q))

            latest_t = max(self.local_pose_map.keys())
            latest = self.global_pose_map[latest_t]
            self.last_p = latest[:3].copy()
            self.last_q = latest[3:7].copy()
            self.has_alignment = True
            self.alignment_status = (
                'aligned: matched_gps=%d baseline=%.2fm'
                % (len(times), gps_xy_span)
            )
            self._update_path_locked()
        return True

    def _optimize_pose_graph(self):
        with self.lock:
            if not self.new_gps:
                return False
            self.new_gps = False
            times = sorted(self.local_pose_map.keys())
            if self.max_optimization_poses > 0 and len(times) > self.max_optimization_poses:
                times = times[-self.max_optimization_poses:]
            if not times:
                return False

            local_snapshot = {t: self.local_pose_map[t].copy() for t in times}
            global_snapshot = {
                t: self.global_pose_map.get(t, local_snapshot[t]).copy()
                for t in times
            }
            gps_snapshot = {
                t: gps.copy()
                for t, gps in self.gps_position_map.items()
                if t in local_snapshot
            }

        index_by_time = {t: idx for idx, t in enumerate(times)}
        x0 = np.zeros(len(times) * 6, dtype=float)
        for idx, t in enumerate(times):
            pose = global_snapshot[t]
            x0[6*idx:6*idx+3] = pose[:3]
            x0[6*idx+3:6*idx+6] = quat_to_rotation(pose[3:7]).as_rotvec()

        local_relative = []
        for t_i, t_j in zip(times[:-1], times[1:]):
            pose_i = local_snapshot[t_i]
            pose_j = local_snapshot[t_j]
            w_t_i = pose_to_matrix(pose_i[:3], pose_i[3:7])
            w_t_j = pose_to_matrix(pose_j[:3], pose_j[3:7])
            i_t_j = np.linalg.inv(w_t_i) @ w_t_j
            rel_p, rel_q = matrix_to_pose(i_t_j)
            local_relative.append((index_by_time[t_i], index_by_time[t_j], rel_p, rel_q))

        gps_constraints = [
            (index_by_time[t], gps[:3], float(gps[3]))
            for t, gps in gps_snapshot.items()
        ]
        if not gps_constraints and not local_relative:
            return False

        def unpack(params, idx):
            p = params[6*idx:6*idx+3]
            q = rotation_to_quat(Rotation.from_rotvec(params[6*idx+3:6*idx+6]))
            return p, q

        def residuals(params):
            out = []
            for idx_i, idx_j, rel_p, rel_q in local_relative:
                p_i, q_i = unpack(params, idx_i)
                p_j, q_j = unpack(params, idx_j)
                t_w_ij = p_j - p_i
                t_i_ij = quat_to_rotation(quat_inverse(q_i)).apply(t_w_ij)
                out.extend((t_i_ij - rel_p) / 0.1)

                q_i_j = quat_multiply(quat_inverse(q_i), q_j)
                error_q = quat_multiply(quat_inverse(rel_q), q_i_j)
                out.extend((2.0 * error_q[1:4]) / 0.01)

            for idx, gps_xyz, accuracy in gps_constraints:
                p, _q = unpack(params, idx)
                out.extend((p - gps_xyz) / accuracy)
            return np.asarray(out, dtype=float)

        try:
            result = least_squares(
                residuals,
                x0,
                loss='huber',
                f_scale=1.0,
                max_nfev=5,
                verbose=0,
            )
            x_opt = result.x
        except Exception:
            return False

        with self.lock:
            for idx, t in enumerate(times):
                p, q = unpack(x_opt, idx)
                self.global_pose_map[t] = np.concatenate((p, q))

                if idx == len(times) - 1:
                    local = self.local_pose_map[t]
                    wvio_t_body = pose_to_matrix(local[:3], local[3:7])
                    wgps_t_body = pose_to_matrix(p, q)
                    self.wgps_t_wvio = wgps_t_body @ np.linalg.inv(wvio_t_body)

            latest = self.global_pose_map[times[-1]]
            self.last_p = latest[:3].copy()
            self.last_q = latest[3:7].copy()
            self.has_alignment = True
            self._update_path_locked()
        return True

    def get_global_odom(self):
        with self.lock:
            return self.last_p.copy(), self.last_q.copy(), self.has_alignment

    def get_alignment_status(self):
        with self.lock:
            return self.alignment_status

    def _make_path_pose(self, t, p, q):
        pose = PoseStamped()
        pose.header.stamp = sec_to_msg_time(t)
        pose.header.frame_id = 'world'
        pose.pose.position.x = float(p[0])
        pose.pose.position.y = float(p[1])
        pose.pose.position.z = float(p[2])
        pose.pose.orientation.w = float(q[0])
        pose.pose.orientation.x = float(q[1])
        pose.pose.orientation.y = float(q[2])
        pose.pose.orientation.z = float(q[3])
        return pose

    def _append_path_pose_locked(self, t, p, q):
        self.path_msg.header.frame_id = 'world'
        self.path_msg.header.stamp = sec_to_msg_time(t)
        self.path_msg.poses.append(self._make_path_pose(t, p, q))

    def _update_path_locked(self):
        path = Path()
        path.header.frame_id = 'world'
        for t in sorted(self.global_pose_map.keys()):
            pose = self.global_pose_map[t]
            path.poses.append(self._make_path_pose(t, pose[:3], pose[3:7]))
            path.header.stamp = sec_to_msg_time(t)
        self.path_msg = path


class CppLogicFusionNode(Node):
    def __init__(self):
        super().__init__('globalEstimator')
        self.declare_parameter('gps_topic', '/gps/fix')
        self.declare_parameter('vio_topic', '/base_odom')
        self.declare_parameter('sync_tolerance_sec', 0.2)
        self.declare_parameter('optimization_period_sec', 2.0)
        self.declare_parameter('publish_before_alignment', False)
        self.declare_parameter('max_optimization_poses', 50)
        self.declare_parameter('optimization_mode', 'alignment')
        self.declare_parameter('alignment_min_gps_constraints', 10)
        self.declare_parameter('alignment_min_baseline_m', 5.0)
        self.declare_parameter('freeze_alignment_after_init', True)
        self.declare_parameter('record_vio_global_csv', True)
        self.declare_parameter(
            'vio_global_csv_path',
            os.path.join(
                POLICY_PART_DIR,
                'topomaps',
                'LLM_Policy',
                'gkj',
                'output',
                'vio_global.csv',
            ),
        )
        self.declare_parameter('enable_route_json_conversion', True)
        self.declare_parameter(
            'route_json_path',
            '/home/orin26/code/Urban-Nav-SR/Policy_part/topomaps/LLM_Policy/route_waypoints.json',
        )
        self.declare_parameter(
            'route_output_path',
            '/home/orin26/code/Urban-Nav-SR/Policy_part/topomaps/LLM_Policy/LLM_UTM_odom_poses.txt',
        )
        self.declare_parameter('route_default_altitude', 33.0)
        self.declare_parameter('route_poll_sec', 1.0)
        self.estimator = Ros1GlobalFusionEstimator(
            max_optimization_poses=int(self.get_parameter('max_optimization_poses').value),
            optimization_mode=str(self.get_parameter('optimization_mode').value),
            alignment_min_gps_constraints=int(self.get_parameter('alignment_min_gps_constraints').value),
            alignment_min_baseline_m=float(self.get_parameter('alignment_min_baseline_m').value),
            freeze_alignment_after_init=bool(self.get_parameter('freeze_alignment_after_init').value),
        )
        self.gps_queue = deque()
        self.gps_lock = Lock()
        self.stop_event = Event()
        self.last_route_signature = ''
        self.route_waiting_origin_warned = False
        self.last_alignment_wait_log = ''
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
        self.pub_odom = self.create_publisher(Odometry, '~/global_odometry', 100)
        self.pub_path = self.create_publisher(Path, '~/global_path', 100)
        self.thread = Thread(target=self.optimization_loop, daemon=True)
        self.thread.start()
        self.route_timer = None
        if bool(self.get_parameter('enable_route_json_conversion').value):
            route_poll_sec = float(self.get_parameter('route_poll_sec').value)
            if route_poll_sec > 0.0:
                self.route_timer = self.create_timer(route_poll_sec, self.route_file_timer_callback)
            self.get_logger().info(
                'route json conversion enabled: %s -> %s'
                % (
                    self.get_parameter('route_json_path').value,
                    self.get_parameter('route_output_path').value,
                )
            )
        self.get_logger().info(
            'ros1-global-fusion python port: vio_topic=%s gps_topic=%s sync=%.3f mode=%s max_optimization_poses=%d'
            % (
                self.get_parameter('vio_topic').value,
                self.get_parameter('gps_topic').value,
                float(self.get_parameter('sync_tolerance_sec').value),
                self.get_parameter('optimization_mode').value,
                int(self.get_parameter('max_optimization_poses').value),
            )
        )
        if str(self.get_parameter('optimization_mode').value).lower() == 'alignment':
            self.get_logger().info(
                'alignment gate: min_gps_constraints=%d min_baseline=%.2f freeze_after_init=%s'
                % (
                    int(self.get_parameter('alignment_min_gps_constraints').value),
                    float(self.get_parameter('alignment_min_baseline_m').value),
                    bool(self.get_parameter('freeze_alignment_after_init').value),
                )
            )
        if bool(self.get_parameter('record_vio_global_csv').value):
            self.get_logger().info(
                'recording vio global csv: %s'
                % self.get_parameter('vio_global_csv_path').value
            )

    def destroy_node(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        super().destroy_node()

    def optimization_loop(self):
        period = float(self.get_parameter('optimization_period_sec').value)
        while rclpy.ok() and not self.stop_event.wait(max(0.0, period)):
            if self.estimator.optimize():
                self.get_logger().info(
                    'global optimization (ros1 python logic): %s'
                    % self.estimator.get_alignment_status()
                )
                self.last_alignment_wait_log = ''
            elif (
                str(self.get_parameter('optimization_mode').value).lower() == 'alignment'
                and not self.estimator.has_alignment
            ):
                status = self.estimator.get_alignment_status()
                if status != self.last_alignment_wait_log:
                    self.last_alignment_wait_log = status
                    self.get_logger().info(
                        f'{status}; global_odometry is not published yet',
                        throttle_duration_sec=5.0,
                    )

    def gps_callback(self, msg):
        with self.gps_lock:
            self.gps_queue.append((stamp_to_sec(msg.header.stamp), msg))

    def vio_callback(self, msg):
        t = stamp_to_sec(msg.header.stamp)
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        odom_p = np.array([p.x, p.y, p.z], dtype=float)
        odom_q = np.array([q.w, q.x, q.y, q.z], dtype=float)
        self.estimator.input_odom(t, odom_p, odom_q)
        self.consume_matching_gps(t)

        global_p, global_q, has_alignment = self.estimator.get_global_odom()
        if not has_alignment and not bool(self.get_parameter('publish_before_alignment').value):
            return

        odom = Odometry()
        odom.header = msg.header
        odom.header.frame_id = 'world'
        odom.child_frame_id = 'world'
        odom.pose.pose.position.x = float(global_p[0])
        odom.pose.pose.position.y = float(global_p[1])
        odom.pose.pose.position.z = float(global_p[2])
        odom.pose.pose.orientation.w = float(global_q[0])
        odom.pose.pose.orientation.x = float(global_q[1])
        odom.pose.pose.orientation.y = float(global_q[2])
        odom.pose.pose.orientation.z = float(global_q[3])
        self.pub_odom.publish(odom)
        self.pub_path.publish(self.estimator.path_msg)
        if bool(self.get_parameter('record_vio_global_csv').value):
            self.append_vio_global_csv(odom.header.stamp, global_p, global_q)

    def consume_matching_gps(self, vio_t):
        tol = float(self.get_parameter('sync_tolerance_sec').value)
        with self.gps_lock:
            while self.gps_queue:
                gps_t, gps_msg = self.gps_queue[0]
                if gps_t < vio_t - tol:
                    self.gps_queue.popleft()
                    continue
                if gps_t > vio_t + tol:
                    break
                acc = float(gps_msg.position_covariance[0])
                if acc <= 0.0 or not math.isfinite(acc):
                    acc = 1.0
                self.estimator.input_gps(
                    vio_t,
                    gps_msg.latitude,
                    gps_msg.longitude,
                    gps_msg.altitude,
                    acc,
                )
                self.gps_queue.popleft()
                break

    def route_file_timer_callback(self):
        route_json_path = str(self.get_parameter('route_json_path').value)
        route_output_path = str(self.get_parameter('route_output_path').value)
        try:
            stat = os.stat(route_json_path)
            with open(route_json_path, 'r', encoding='utf-8') as fin:
                data = json.load(fin)
        except FileNotFoundError:
            self.get_logger().warn(f'Route JSON file not readable: {route_json_path}')
            return
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f'Route JSON decode failed: {exc}')
            return
        except OSError as exc:
            self.get_logger().warn(f'Route JSON read failed: {exc}')
            return

        route_id = data.get('route_id') if isinstance(data, dict) else None
        signature = str(route_id) if route_id else f'{stat.st_mtime_ns}:{stat.st_size}'
        if signature == self.last_route_signature:
            return

        if not self.estimator.has_gps_origin():
            if not self.route_waiting_origin_warned:
                self.get_logger().warn(
                    'Route JSON conversion is waiting for the first GPS fix, so the output uses the fusion GPS origin.'
                )
                self.route_waiting_origin_warned = True
            return
        self.route_waiting_origin_warned = False

        default_altitude = float(self.get_parameter('route_default_altitude').value)
        waypoints = parse_waypoints_from_json(data, default_altitude)
        if not waypoints:
            self.get_logger().warn(f'Route JSON changed but no valid waypoints found in {route_json_path}')
            return

        self.convert_json_waypoints_to_enu_txt(waypoints, route_output_path)
        self.last_route_signature = signature

    def convert_json_waypoints_to_enu_txt(self, waypoints, route_output_path):
        self.get_logger().info(f'Converting route with {len(waypoints)} waypoints')
        output_dir = os.path.dirname(route_output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(route_output_path, 'w', newline='', encoding='utf-8') as fout:
            writer = csv.writer(fout)
            writer.writerow(['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'])
            for index, waypoint in enumerate(waypoints, start=1):
                xyz = self.estimator.gps_to_xyz(
                    waypoint.latitude,
                    waypoint.longitude,
                    waypoint.altitude,
                )
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
                    % (
                        index,
                        waypoint.latitude,
                        waypoint.longitude,
                        waypoint.altitude,
                        xyz[0],
                        xyz[1],
                        xyz[2],
                    )
                )
        self.get_logger().info(f'Route conversion completed. Results saved to {route_output_path}')

    def append_vio_global_csv(self, stamp, global_p, global_q):
        csv_path = str(self.get_parameter('vio_global_csv_path').value)
        if not csv_path:
            return
        timestamp_ns = int(stamp.sec) * 1000000000 + int(stamp.nanosec)
        try:
            output_dir = os.path.dirname(csv_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(csv_path, 'a', newline='', encoding='utf-8') as fout:
                writer = csv.writer(fout)
                writer.writerow([
                    str(timestamp_ns),
                    f'{global_p[0]:.5f}',
                    f'{global_p[1]:.5f}',
                    f'{global_p[2]:.5f}',
                    f'{global_q[0]:.5f}',
                    f'{global_q[1]:.5f}',
                    f'{global_q[2]:.5f}',
                    f'{global_q[3]:.5f}',
                ])
        except OSError as exc:
            self.get_logger().warn(
                f'Failed to append vio global csv {csv_path}: {exc}',
                throttle_duration_sec=5.0,
            )


def main(args=None):
    rclpy.init(args=args)
    node = CppLogicFusionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
