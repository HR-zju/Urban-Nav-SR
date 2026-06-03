#!/usr/bin/env python3
import csv
import json
import math
import multiprocessing as mp
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from threading import Event, Lock, Thread

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
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
    w, x, y, z = quat_normalize(q)
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=float)


def matrix_to_quat(matrix):
    m = np.asarray(matrix, dtype=float)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        diag = np.diag(m)
        idx = int(np.argmax(diag))
        if idx == 0:
            s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12)) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12)) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    return quat_normalize([w, x, y, z])


def rotvec_to_quat(rotvec):
    rotvec = np.asarray(rotvec, dtype=float)
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = rotvec / theta
    half = 0.5 * theta
    return quat_normalize([
        math.cos(half),
        axis[0] * math.sin(half),
        axis[1] * math.sin(half),
        axis[2] * math.sin(half),
    ])


def quat_to_rotvec(q):
    q = quat_normalize(q)
    if q[0] < 0.0:
        q = -q
    vec = np.asarray(q[1:4], dtype=float)
    sin_half = float(np.linalg.norm(vec))
    if sin_half < 1e-12:
        return 2.0 * vec
    angle = 2.0 * math.atan2(sin_half, q[0])
    return vec / sin_half * angle


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


def huber_weight(norm_value, delta=1.0):
    if norm_value <= delta or norm_value < 1e-12:
        return 1.0
    return math.sqrt(delta / norm_value)


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


def solve_global_optimization_job(job):
    try:
        os.nice(5)
    except OSError:
        pass

    times = job['times']
    poses = job['poses']
    local_relative = job['local_relative']
    gps_constraints = job['gps_constraints']
    last_local_pose = job['last_local_pose']

    if not times:
        return None

    def copy_poses(src_poses):
        return [[pose[0].copy(), pose[1].copy()] for pose in src_poses]

    def apply_delta(src_poses, delta):
        trial = copy_poses(src_poses)
        for idx in range(len(trial)):
            base = idx * 6
            trial[idx][0] += delta[base:base + 3]
            trial[idx][1] = quat_multiply(rotvec_to_quat(delta[base + 3:base + 6]), trial[idx][1])
        return trial

    def relative_translation_residual(src_poses, i, j, measured_p):
        p_i, q_i = src_poses[i]
        p_j, _ = src_poses[j]
        return (quat_to_matrix(q_i).T @ (p_j - p_i) - measured_p) / 0.1

    def relative_rotation_residual(src_poses, i, j, measured_q):
        q_i = src_poses[i][1]
        q_j = src_poses[j][1]
        rel_q = quat_multiply(quat_inverse(q_i), q_j)
        q_err = quat_multiply(quat_inverse(measured_q), rel_q)
        return (2.0 * q_err[1:4]) / 0.01

    def gps_residual(src_poses, i, gps_xyz, pos_accuracy):
        return (src_poses[i][0] - gps_xyz) / pos_accuracy

    def numeric_rot_jacobian(residual_fn, src_poses, pose_idx, *args, eps=1e-6):
        base_q = src_poses[pose_idx][1].copy()
        base_res = residual_fn(src_poses, *args)
        jac = np.zeros((base_res.size, 3), dtype=float)
        for axis in range(3):
            delta = np.zeros(3, dtype=float)
            delta[axis] = eps
            src_poses[pose_idx][1] = quat_multiply(rotvec_to_quat(delta), base_q)
            perturbed_res = residual_fn(src_poses, *args)
            jac[:, axis] = (perturbed_res - base_res) / eps
        src_poses[pose_idx][1] = base_q
        return jac

    def huber_scale_block(residual):
        return huber_weight(float(np.linalg.norm(residual)))

    def add_measurement(hessian, gradient, residual, blocks):
        for pose_i, jac_i in blocks:
            slice_i = slice(pose_i * 6, pose_i * 6 + 6)
            gradient[slice_i] += jac_i.T @ residual
            for pose_j, jac_j in blocks:
                slice_j = slice(pose_j * 6, pose_j * 6 + 6)
                hessian[slice_i, slice_j] += jac_i.T @ jac_j

    def compute_cost(src_poses):
        total = 0.0
        for i, j, measured_p, measured_q in local_relative:
            residual_t = relative_translation_residual(src_poses, i, j, measured_p)
            residual_r = relative_rotation_residual(src_poses, i, j, measured_q)
            wt = huber_scale_block(residual_t)
            wr = huber_scale_block(residual_r)
            total += 0.5 * float((residual_t * wt) @ (residual_t * wt))
            total += 0.5 * float((residual_r * wr) @ (residual_r * wr))
        for i, gps_xyz, pos_accuracy in gps_constraints:
            residual_gps = gps_residual(src_poses, i, gps_xyz, pos_accuracy)
            wg = huber_scale_block(residual_gps)
            total += 0.5 * float((residual_gps * wg) @ (residual_gps * wg))
        return total

    current_poses = copy_poses(poses)
    current_cost = compute_cost(current_poses)
    lambda_damping = 1e-3

    for _ in range(6):
        dim = len(current_poses) * 6
        hessian = np.zeros((dim, dim), dtype=float)
        gradient = np.zeros(dim, dtype=float)

        for i, j, measured_p, measured_q in local_relative:
            residual_t = relative_translation_residual(current_poses, i, j, measured_p)
            residual_r = relative_rotation_residual(current_poses, i, j, measured_q)
            wt = huber_scale_block(residual_t)
            wr = huber_scale_block(residual_r)

            _, q_i = current_poses[i]
            rot_i_t = quat_to_matrix(q_i).T / 0.1
            jac_t_i = np.hstack([
                -rot_i_t,
                numeric_rot_jacobian(relative_translation_residual, current_poses, i, i, j, measured_p),
            ]) * wt
            jac_t_j = np.hstack([
                rot_i_t,
                np.zeros((3, 3), dtype=float),
            ]) * wt
            add_measurement(
                hessian,
                gradient,
                residual_t * wt,
                [(i, jac_t_i), (j, jac_t_j)],
            )

            jac_r_i = np.hstack([
                np.zeros((3, 3), dtype=float),
                numeric_rot_jacobian(relative_rotation_residual, current_poses, i, i, j, measured_q),
            ]) * wr
            jac_r_j = np.hstack([
                np.zeros((3, 3), dtype=float),
                numeric_rot_jacobian(relative_rotation_residual, current_poses, j, i, j, measured_q),
            ]) * wr
            add_measurement(
                hessian,
                gradient,
                residual_r * wr,
                [(i, jac_r_i), (j, jac_r_j)],
            )

        for i, gps_xyz, pos_accuracy in gps_constraints:
            residual_gps = gps_residual(current_poses, i, gps_xyz, pos_accuracy)
            wg = huber_scale_block(residual_gps)
            jac_gps = np.hstack([
                np.eye(3, dtype=float) / pos_accuracy,
                np.zeros((3, 3), dtype=float),
            ]) * wg
            add_measurement(
                hessian,
                gradient,
                residual_gps * wg,
                [(i, jac_gps)],
            )

        hessian += lambda_damping * np.eye(dim, dtype=float)

        try:
            step = np.linalg.solve(hessian, -gradient)
        except np.linalg.LinAlgError:
            lambda_damping *= 10.0
            continue

        if float(np.linalg.norm(step)) < 1e-6:
            break

        trial_poses = apply_delta(current_poses, step)
        trial_cost = compute_cost(trial_poses)
        if trial_cost < current_cost:
            current_poses = trial_poses
            current_cost = trial_cost
            lambda_damping = max(lambda_damping * 0.5, 1e-6)
        else:
            lambda_damping *= 10.0

    optimized_pose_items = [
        (t, np.concatenate((current_poses[i][0], current_poses[i][1])))
        for i, t in enumerate(times)
    ]
    last_global_p, last_global_q = current_poses[-1]
    next_wgps_r_wvio = quat_to_matrix(last_global_q) @ quat_to_matrix(last_local_pose[3:7]).T
    next_wgps_t_wvio = last_global_p - next_wgps_r_wvio @ last_local_pose[:3]

    return {
        'optimized_pose_items': optimized_pose_items,
        'wgps_r_wvio': next_wgps_r_wvio,
        'wgps_t_wvio': next_wgps_t_wvio,
    }


class GlobalOptimization:
    def __init__(self, clock, max_pose_count=200):
        self.clock = clock
        self.max_pose_count = int(max_pose_count)
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
        self.optimization_pool = ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp.get_context('spawn'),
        )
        self.optimization_future = None

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
            pruned = self._prune_maps_locked()
            if pruned:
                self._update_path_locked()
            else:
                self._append_path_pose_locked(t, global_p, global_q)

    def input_gps(self, t, latitude, longitude, altitude, pos_accuracy):
        xyz = self.gps_to_xyz(latitude, longitude, altitude)
        with self.lock:
            self.gps_position_map[t] = np.array([xyz[0], xyz[1], xyz[2], max(pos_accuracy, 1e-6)], dtype=float)
            self._prune_maps_locked()
            self.new_gps = True

    def _prune_maps_locked(self):
        if self.max_pose_count <= 0 or len(self.local_pose_map) <= self.max_pose_count:
            return False
        keep_times = set(sorted(self.local_pose_map.keys())[-self.max_pose_count:])
        for mapping in (self.local_pose_map, self.global_pose_map):
            for t in list(mapping.keys()):
                if t not in keep_times:
                    mapping.pop(t, None)
        for t in list(self.gps_position_map.keys()):
            if t not in keep_times:
                self.gps_position_map.pop(t, None)
        return True

    def get_global_odom(self):
        with self.lock:
            return self.last_p.copy(), self.last_q.copy()

    def shutdown(self):
        if self.optimization_future is not None:
            self.optimization_future.cancel()
            self.optimization_future = None
        self.optimization_pool.shutdown(wait=False, cancel_futures=True)

    def optimize(self, logger):
        self._finish_optimization(logger)
        if self.optimization_future is not None:
            return

        job = self._make_optimization_job()
        if job is None:
            return

        logger.info('global optimization')
        self.optimization_future = self.optimization_pool.submit(
            solve_global_optimization_job,
            job,
        )

    def _make_optimization_job(self):
        with self.lock:
            if not self.new_gps:
                return None
            if len(self.local_pose_map) < 1:
                self.new_gps = False
                return None
            self.new_gps = False

            times = sorted(self.local_pose_map.keys())
            local_pose_snapshot = {
                t: self.local_pose_map[t].copy()
                for t in times
            }
            index_by_time = {t: i for i, t in enumerate(times)}
            poses = []
            for t in times:
                pose = self.global_pose_map.get(t)
                if pose is None:
                    pose = local_pose_snapshot[t]
                poses.append([
                    np.asarray(pose[:3], dtype=float).copy(),
                    quat_normalize(np.asarray(pose[3:7], dtype=float)),
                ])

            local_relative = []
            for t_i, t_j in zip(times[:-1], times[1:]):
                pose_i = local_pose_snapshot[t_i]
                pose_j = local_pose_snapshot[t_j]
                rel_p, rel_q = relative_pose(pose_i[:3], pose_i[3:7], pose_j[:3], pose_j[3:7])
                local_relative.append((index_by_time[t_i], index_by_time[t_j], rel_p, rel_q))

            gps_constraints = []
            for t, gps_xyz_acc in self.gps_position_map.items():
                if t in index_by_time:
                    gps_constraints.append((index_by_time[t], gps_xyz_acc[:3], gps_xyz_acc[3]))

        if not gps_constraints and not local_relative:
            return None

        return {
            'times': times,
            'poses': poses,
            'local_relative': local_relative,
            'gps_constraints': gps_constraints,
            'last_local_pose': local_pose_snapshot[times[-1]],
        }

    def _finish_optimization(self, logger):
        if self.optimization_future is None or not self.optimization_future.done():
            return
        future = self.optimization_future
        self.optimization_future = None
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001
            logger.error(f'Global optimization failed: {exc}')
            return
        if result is not None:
            self._apply_optimization_result(result)

    def _apply_optimization_result(self, result):
        optimized_pose_map = dict(result['optimized_pose_items'])
        with self.lock:
            self.wgps_r_wvio = result['wgps_r_wvio']
            self.wgps_t_wvio = result['wgps_t_wvio']
            for t, pose in optimized_pose_map.items():
                if t in self.local_pose_map:
                    self.global_pose_map[t] = pose

            for t, local_pose in self.local_pose_map.items():
                if t in optimized_pose_map:
                    continue
                local_p = local_pose[:3]
                local_q = local_pose[3:7]
                global_p = self.wgps_r_wvio @ local_p + self.wgps_t_wvio
                global_q = matrix_to_quat(self.wgps_r_wvio @ quat_to_matrix(local_q))
                self.global_pose_map[t] = np.concatenate((global_p, global_q))

            if self.local_pose_map:
                latest_t = max(self.local_pose_map.keys())
                latest_pose = self.global_pose_map.get(latest_t)
                if latest_pose is not None:
                    self.last_p = latest_pose[:3].copy()
                    self.last_q = latest_pose[3:7].copy()
            self._update_path_locked()

    def _make_path_pose(self, t, p, q):
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = sec_to_msg_time(t, self.clock)
        pose_stamped.header.frame_id = 'world'
        pose_stamped.pose.position.x = float(p[0])
        pose_stamped.pose.position.y = float(p[1])
        pose_stamped.pose.position.z = float(p[2])
        pose_stamped.pose.orientation.w = float(q[0])
        pose_stamped.pose.orientation.x = float(q[1])
        pose_stamped.pose.orientation.y = float(q[2])
        pose_stamped.pose.orientation.z = float(q[3])
        return pose_stamped

    def _append_path_pose_locked(self, t, p, q):
        self.path_msg.header.frame_id = 'world'
        self.path_msg.header.stamp = sec_to_msg_time(t, self.clock)
        self.path_msg.poses.append(self._make_path_pose(t, p, q))

    def _update_path_locked(self):
        path = Path()
        path.header.frame_id = 'world'
        if self.global_pose_map:
            path.header.stamp = sec_to_msg_time(max(self.global_pose_map.keys()), self.clock)
        for t in sorted(self.global_pose_map.keys()):
            pose = self.global_pose_map[t]
            path.poses.append(self._make_path_pose(t, pose[:3], pose[3:7]))
        self.path_msg = path


class GlobalFusionNode(Node):
    def __init__(self):
        super().__init__('globalEstimator')

        self.declare_parameter('route_json_path', '/home/orin26/code/Urban-Nav-SR/Policy_part/topomaps/LLM_Policy/route_waypoints.json')
        self.declare_parameter('route_output_path', '/home/orin26/code/Urban-Nav-SR/Policy_part/topomaps/LLM_Policy/LLM_UTM_odom_poses.txt')
        self.declare_parameter('vio_global_csv_path', '/home/orin26/code/Urban-Nav-SR/Policy_part/topomaps/LLM_Policy/gkj/output/vio_global.csv')
        self.declare_parameter('gps_topic', '/gps/fix')
        self.declare_parameter('vio_topic', '/agilex/odom')
        self.declare_parameter('sync_tolerance_sec', 0.2)
        self.declare_parameter('sync_time_source', 'header')
        self.declare_parameter('output_stamp_source', 'input_header')
        self.declare_parameter('max_optimization_poses', 200)
        self.declare_parameter('route_default_altitude', 33.0)
        self.declare_parameter('publish_car_model', False)
        self.declare_parameter('route_poll_sec', 1.0)
        self.declare_parameter('optimization_period_sec', 2.0)
        self.declare_parameter('path_publish_period_sec', 0.2)

        self.estimator = GlobalOptimization(
            self.get_clock(),
            max_pose_count=int(self.get_parameter('max_optimization_poses').value),
        )
        self.gps_queue = deque()
        self.gps_lock = Lock()
        self.last_route_id = ''
        self.last_path_publish_mono = 0.0

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
        self.optimization_period_sec = float(self.get_parameter('optimization_period_sec').value)
        self.optimization_stop = Event()
        self.optimization_thread = Thread(
            target=self.optimization_loop,
            daemon=True,
        )
        self.optimization_thread.start()

        self.get_logger().info(f"Watching route JSON file: {self.get_parameter('route_json_path').value}")
        self.get_logger().info(f"Route UTM output file: {self.get_parameter('route_output_path').value}")
        self.get_logger().info(
            'Fusion sync: gps_topic=%s, vio_topic=%s, sync_time_source=%s, output_stamp_source=%s, sync_tolerance_sec=%.3f'
            % (
                self.get_parameter('gps_topic').value,
                self.get_parameter('vio_topic').value,
                self.get_parameter('sync_time_source').value,
                self.get_parameter('output_stamp_source').value,
                float(self.get_parameter('sync_tolerance_sec').value),
            )
        )
        self.get_logger().info(
            'Global optimization runs in a background thread every %.3f seconds'
            % self.optimization_period_sec
        )
        self.get_logger().info(
            'Global path publish period: %.3f seconds'
            % float(self.get_parameter('path_publish_period_sec').value)
        )

    def _now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _sync_time(self, msg):
        source = str(self.get_parameter('sync_time_source').value).lower()
        if source == 'header':
            return stamp_to_sec(msg.header.stamp)
        if source != 'receive':
            self.get_logger().warn(
                f"Unsupported sync_time_source '{source}', falling back to receive time",
                throttle_duration_sec=5.0,
            )
        return self._now_sec()

    def _output_stamp(self, input_stamp, sync_time):
        source = str(self.get_parameter('output_stamp_source').value).lower()
        if source == 'input_header':
            return input_stamp
        if source != 'sync':
            self.get_logger().warn(
                f"Unsupported output_stamp_source '{source}', falling back to sync time",
                throttle_duration_sec=5.0,
            )
        return sec_to_msg_time(sync_time, self.get_clock())

    def destroy_node(self):
        if hasattr(self, 'optimization_stop'):
            self.optimization_stop.set()
        if hasattr(self, 'optimization_thread') and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=2.0)
        if hasattr(self, 'estimator'):
            self.estimator.shutdown()
        super().destroy_node()

    def optimization_loop(self):
        period = self.optimization_period_sec
        while rclpy.ok() and not self.optimization_stop.wait(max(0.0, period)):
            try:
                self.estimator.optimize(self.get_logger())
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f'Global optimization failed: {exc}')

    def gps_callback(self, msg):
        gps_t = self._sync_time(msg)
        with self.gps_lock:
            self.gps_queue.append((gps_t, msg))

    def vio_callback(self, pose_msg):
        t = self._sync_time(pose_msg)
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
        odometry.header.stamp = self._output_stamp(pose_msg.header.stamp, t)
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
        self._publish_global_path_if_due()
        if self.get_parameter('publish_car_model').value:
            self.publish_car_model(t, global_t, global_q)
        self._append_global_csv(odometry.header.stamp, global_t, global_q)

    def _publish_global_path_if_due(self):
        period = float(self.get_parameter('path_publish_period_sec').value)
        now = time.monotonic()
        if period > 0.0 and now - self.last_path_publish_mono < period:
            return
        self.last_path_publish_mono = now
        self.pub_global_path.publish(self.estimator.path_msg)

    def _consume_matching_gps(self, vio_t):
        tolerance = float(self.get_parameter('sync_tolerance_sec').value)
        with self.gps_lock:
            while self.gps_queue:
                gps_t, gps_msg = self.gps_queue[0]
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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
