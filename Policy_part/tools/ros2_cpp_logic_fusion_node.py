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
GCJ02_A = 6378245.0
GCJ02_EE = 0.00669342162296594323
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


def is_in_china_for_gcj02(longitude, latitude):
    return 73.66 < longitude < 135.05 and 3.86 < latitude < 53.55


def transform_gcj02_lat(x, y):
    ret = -100.0 + 2.0*x + 3.0*y + 0.2*y*y + 0.1*x*y + 0.2*math.sqrt(abs(x))
    ret += (20.0*math.sin(6.0*x*math.pi) + 20.0*math.sin(2.0*x*math.pi)) * 2.0 / 3.0
    ret += (20.0*math.sin(y*math.pi) + 40.0*math.sin(y/3.0*math.pi)) * 2.0 / 3.0
    ret += (160.0*math.sin(y/12.0*math.pi) + 320.0*math.sin(y*math.pi/30.0)) * 2.0 / 3.0
    return ret


def transform_gcj02_lon(x, y):
    ret = 300.0 + x + 2.0*y + 0.1*x*x + 0.1*x*y + 0.1*math.sqrt(abs(x))
    ret += (20.0*math.sin(6.0*x*math.pi) + 20.0*math.sin(2.0*x*math.pi)) * 2.0 / 3.0
    ret += (20.0*math.sin(x*math.pi) + 40.0*math.sin(x/3.0*math.pi)) * 2.0 / 3.0
    ret += (150.0*math.sin(x/12.0*math.pi) + 300.0*math.sin(x/30.0*math.pi)) * 2.0 / 3.0
    return ret


def wgs84_to_gcj02(longitude, latitude):
    if not is_in_china_for_gcj02(longitude, latitude):
        return longitude, latitude

    dlat = transform_gcj02_lat(longitude - 105.0, latitude - 35.0)
    dlon = transform_gcj02_lon(longitude - 105.0, latitude - 35.0)
    radlat = math.radians(latitude)
    magic = math.sin(radlat)
    magic = 1.0 - GCJ02_EE * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((GCJ02_A * (1.0 - GCJ02_EE)) / (magic * sqrtmagic) * math.pi)
    dlon = (dlon * 180.0) / (GCJ02_A / sqrtmagic * math.cos(radlat) * math.pi)
    return longitude + dlon, latitude + dlat

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
        alignment_min_baseline_m=50.0,
        freeze_alignment_after_init=True,
        alignment_min_odom_baseline_m=50.0,
        alignment_min_odom_sample_separation_m=0.5,
        alignment_min_baseline_ratio=0.5,
        alignment_max_baseline_ratio=2.0,
        alignment_inlier_threshold_m=2.5,
        alignment_min_inlier_ratio=0.75,
        alignment_max_xy_rmse_m=1.5,
        alignment_required_stable_yaw_estimates=4,
        alignment_stable_yaw_min_path_increment_m=2.0,
        alignment_max_stable_yaw_delta_deg=2.0,
        alignment_observation_distance_m=10.0,
        alignment_observation_min_gps_constraints=5,
        alignment_observation_min_inlier_ratio=0.75,
        alignment_observation_max_xy_rmse_m=2.0,
        alignment_max_initial_to_observed_yaw_delta_deg=2.0,
        alignment_candidate_start_timeout_m=15.0,
        alignment_restart_on_observation_failure=True,
    ):
        self.lock = Lock()
        self.max_optimization_poses = int(max_optimization_poses)
        self.optimization_mode = str(optimization_mode).lower()
        self.alignment_min_gps_constraints = int(alignment_min_gps_constraints)
        self.alignment_min_baseline_m = float(alignment_min_baseline_m)
        self.alignment_min_odom_baseline_m = float(alignment_min_odom_baseline_m)
        self.alignment_min_odom_sample_separation_m = max(
            0.0, float(alignment_min_odom_sample_separation_m)
        )
        self.alignment_min_baseline_ratio = float(alignment_min_baseline_ratio)
        self.alignment_max_baseline_ratio = float(alignment_max_baseline_ratio)
        self.alignment_inlier_threshold_m = max(1e-6, float(alignment_inlier_threshold_m))
        self.alignment_min_inlier_ratio = min(
            1.0, max(0.0, float(alignment_min_inlier_ratio))
        )
        self.alignment_max_xy_rmse_m = max(1e-6, float(alignment_max_xy_rmse_m))
        self.alignment_required_stable_yaw_estimates = max(
            1, int(alignment_required_stable_yaw_estimates)
        )
        self.alignment_stable_yaw_min_path_increment_m = max(
            0.0, float(alignment_stable_yaw_min_path_increment_m)
        )
        self.alignment_max_stable_yaw_delta_deg = max(
            0.0, float(alignment_max_stable_yaw_delta_deg)
        )
        self.alignment_observation_distance_m = max(
            0.0, float(alignment_observation_distance_m)
        )
        self.alignment_observation_min_gps_constraints = max(
            2, int(alignment_observation_min_gps_constraints)
        )
        self.alignment_observation_min_inlier_ratio = min(
            1.0, max(0.0, float(alignment_observation_min_inlier_ratio))
        )
        self.alignment_observation_max_xy_rmse_m = max(
            1e-6, float(alignment_observation_max_xy_rmse_m)
        )
        self.alignment_max_initial_to_observed_yaw_delta_deg = max(
            0.0, float(alignment_max_initial_to_observed_yaw_delta_deg)
        )
        self.alignment_candidate_start_timeout_m = max(
            0.0, float(alignment_candidate_start_timeout_m)
        )
        self.alignment_restart_on_observation_failure = bool(
            alignment_restart_on_observation_failure
        )
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
        self.alignment_yaw_candidates = deque(
            maxlen=self.alignment_required_stable_yaw_estimates
        )
        self.last_alignment_candidate_path_m = None
        self.alignment_gate_reached_path_m = None
        self.alignment_observation_transform = None
        self.alignment_observation_yaw = None
        self.alignment_observation_start_path_m = None
        self.alignment_observation_start_time = None
        self.alignment_reset_count = 0
        self.pending_alignment_reset_messages = deque()
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

    def _select_motion_diverse_times(self, times):
        """Keep one recent GPS constraint per meaningful odom displacement."""
        if not times:
            return []
        min_separation = self.alignment_min_odom_sample_separation_m
        if min_separation <= 0.0:
            return list(times)

        selected_reversed = [times[-1]]
        last_xy = self.local_pose_map[times[-1]][:2]
        for t in reversed(times[:-1]):
            xy = self.local_pose_map[t][:2]
            if np.linalg.norm(xy - last_xy) >= min_separation:
                selected_reversed.append(t)
                last_xy = xy
        selected_reversed.reverse()
        return selected_reversed

    def _estimate_robust_planar_alignment(self, local_points, gps_points, weights):
        """Fit yaw + translation with Huber-style iterative reweighting."""
        base_weights = np.maximum(np.asarray(weights, dtype=float), 1e-12)
        robust_weights = np.ones_like(base_weights)
        threshold = self.alignment_inlier_threshold_m
        transform = np.eye(4, dtype=float)
        residuals = np.full(len(local_points), np.inf, dtype=float)

        for _ in range(5):
            transform = self._estimate_planar_alignment(
                local_points,
                gps_points,
                base_weights * robust_weights,
            )
            predicted = (
                (transform[:3, :3] @ local_points.T).T + transform[:3, 3]
            )
            residuals = np.linalg.norm(predicted[:, :2] - gps_points[:, :2], axis=1)
            robust_weights = np.ones_like(residuals)
            large = residuals > threshold
            robust_weights[large] = threshold / np.maximum(residuals[large], 1e-12)

        inliers = residuals <= threshold
        if np.count_nonzero(inliers) >= 2:
            transform = self._estimate_planar_alignment(
                local_points[inliers],
                gps_points[inliers],
                base_weights[inliers],
            )
            predicted = (
                (transform[:3, :3] @ local_points.T).T + transform[:3, 3]
            )
            residuals = np.linalg.norm(predicted[:, :2] - gps_points[:, :2], axis=1)
            inliers = residuals <= threshold
        return transform, residuals, inliers

    @staticmethod
    def _angle_difference(a, b):
        return math.atan2(math.sin(a - b), math.cos(a - b))

    def _yaw_is_stable(self, candidate_yaw, odom_path_m):
        min_increment = self.alignment_stable_yaw_min_path_increment_m
        if (
            self.last_alignment_candidate_path_m is None
            or odom_path_m - self.last_alignment_candidate_path_m >= min_increment
        ):
            self.alignment_yaw_candidates.append((odom_path_m, candidate_yaw))
            self.last_alignment_candidate_path_m = odom_path_m

        required = self.alignment_required_stable_yaw_estimates
        if len(self.alignment_yaw_candidates) < required:
            return False, 0.0

        stored_yaws = [item[1] for item in self.alignment_yaw_candidates]
        mean_yaw = math.atan2(
            sum(math.sin(yaw) for yaw in stored_yaws),
            sum(math.cos(yaw) for yaw in stored_yaws),
        )
        yaws_to_check = stored_yaws + [candidate_yaw]
        max_delta_deg = max(
            abs(math.degrees(self._angle_difference(yaw, mean_yaw)))
            for yaw in yaws_to_check
        )
        return max_delta_deg <= self.alignment_max_stable_yaw_delta_deg, max_delta_deg

    def _clear_alignment_trial_state_locked(self):
        self.alignment_yaw_candidates.clear()
        self.last_alignment_candidate_path_m = None
        self.alignment_gate_reached_path_m = None
        self.alignment_observation_transform = None
        self.alignment_observation_yaw = None
        self.alignment_observation_start_path_m = None
        self.alignment_observation_start_time = None

    def _reset_alignment_cycle_locked(self, reason):
        """Discard this failed trial and start a fresh 30 m-style alignment cycle."""
        self.alignment_reset_count += 1
        message = (
            'fusion已重置: 第%d次; 原因=%s; 重新开始累计%.1fm对齐'
            % (self.alignment_reset_count, reason, self.alignment_min_baseline_m)
        )
        self.local_pose_map.clear()
        self.global_pose_map.clear()
        self.gps_position_map.clear()
        self.wgps_t_wvio = np.eye(4, dtype=float)
        self.last_p = np.zeros(3, dtype=float)
        self.last_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.new_gps = False
        self.has_alignment = False
        self.path_msg = Path()
        self._clear_alignment_trial_state_locked()
        self.alignment_status = message
        self.pending_alignment_reset_messages.append(message)

    def _alignment_failure_or_wait_locked(self, reason, odom_path_m):
        """Wait inside the trial, or reset once its candidate/observation budget expires."""
        if self.alignment_observation_transform is not None:
            phase = 'observation'
            start_path_m = self.alignment_observation_start_path_m
            limit_m = self.alignment_observation_distance_m
        else:
            phase = 'candidate'
            start_path_m = self.alignment_gate_reached_path_m
            limit_m = self.alignment_candidate_start_timeout_m

        if start_path_m is None:
            self.alignment_status = reason
            return False

        progress_m = max(0.0, odom_path_m - start_path_m)
        if (
            self.alignment_restart_on_observation_failure
            and progress_m >= limit_m
        ):
            self._reset_alignment_cycle_locked(
                '%s; %s_progress=%.2fm/%.2fm'
                % (reason, phase, progress_m, limit_m)
            )
            return False

        self.alignment_status = (
            '%s; %s_progress=%.2fm/%.2fm'
            % (reason, phase, progress_m, limit_m)
        )
        return False

    def _evaluate_observation_locked(self, times, local_points, gps_points):
        holdout_indices = [
            index
            for index, timestamp in enumerate(times)
            if timestamp > self.alignment_observation_start_time
        ]
        if len(holdout_indices) < self.alignment_observation_min_gps_constraints:
            return {
                'passed': False,
                'reason': (
                    'insufficient observation constraints: holdout=%d/%d'
                    % (
                        len(holdout_indices),
                        self.alignment_observation_min_gps_constraints,
                    )
                ),
                'count': len(holdout_indices),
                'rmse': float('inf'),
                'inlier_ratio': 0.0,
            }

        indices = np.asarray(holdout_indices, dtype=int)
        holdout_local = local_points[indices]
        holdout_gps = gps_points[indices]
        transform = self.alignment_observation_transform
        predicted = (
            (transform[:3, :3] @ holdout_local.T).T + transform[:3, 3]
        )
        residuals = np.linalg.norm(predicted[:, :2] - holdout_gps[:, :2], axis=1)
        inliers = residuals <= self.alignment_inlier_threshold_m
        inlier_count = int(np.count_nonzero(inliers))
        inlier_ratio = inlier_count / float(len(residuals))
        rmse = (
            float(np.sqrt(np.mean(np.square(residuals[inliers]))))
            if inlier_count > 0
            else float('inf')
        )
        passed = (
            inlier_ratio >= self.alignment_observation_min_inlier_ratio
            and rmse <= self.alignment_observation_max_xy_rmse_m
        )
        reason = (
            'observation quality: holdout=%d rmse=%.2fm/%.2fm '
            'inliers=%.0f%%/%.0f%%'
            % (
                len(residuals),
                rmse,
                self.alignment_observation_max_xy_rmse_m,
                100.0 * inlier_ratio,
                100.0 * self.alignment_observation_min_inlier_ratio,
            )
        )
        return {
            'passed': passed,
            'reason': reason,
            'count': len(residuals),
            'rmse': rmse,
            'inlier_ratio': inlier_ratio,
        }

    def _optimize_alignment(self):
        with self.lock:
            if not self.new_gps:
                return False
            self.new_gps = False
            if self.has_alignment and self.freeze_alignment_after_init:
                return False

            matched_times = sorted(
                t for t in self.gps_position_map.keys() if t in self.local_pose_map
            )
            if len(matched_times) < self.alignment_min_gps_constraints:
                self.alignment_status = (
                    'waiting for alignment: matched_gps=%d/%d'
                    % (len(matched_times), self.alignment_min_gps_constraints)
                )
                return False

            # Do not let many stationary GPS fixes dominate the alignment. Starting
            # from the latest fix also retains the GPS bias closest to motion onset.
            times = self._select_motion_diverse_times(matched_times)
            if len(times) < self.alignment_min_gps_constraints:
                self.alignment_status = (
                    'waiting for motion-diverse constraints: matched_gps=%d motion_samples=%d/%d'
                    % (
                        len(matched_times),
                        len(times),
                        self.alignment_min_gps_constraints,
                    )
                )
                return False

            local_points = np.array([self.local_pose_map[t][:3] for t in times], dtype=float)
            gps_points = np.array([self.gps_position_map[t][:3] for t in times], dtype=float)
            accuracies = np.array([self.gps_position_map[t][3] for t in times], dtype=float)
            odom_steps = np.linalg.norm(np.diff(local_points[:, :2], axis=0), axis=1)
            odom_path_m = float(np.sum(odom_steps))
            odom_xy_span = np.linalg.norm(
                np.max(local_points[:, :2], axis=0) - np.min(local_points[:, :2], axis=0)
            )
            if odom_xy_span < self.alignment_min_odom_baseline_m:
                self.alignment_status = (
                    'waiting for odom movement: motion_samples=%d odom_baseline=%.2fm/%.2fm'
                    % (len(times), odom_xy_span, self.alignment_min_odom_baseline_m)
                )
                return False

            if self.alignment_gate_reached_path_m is None:
                self.alignment_gate_reached_path_m = odom_path_m

            gps_xy_span = np.linalg.norm(
                np.max(gps_points[:, :2], axis=0) - np.min(gps_points[:, :2], axis=0)
            )
            if gps_xy_span < self.alignment_min_baseline_m:
                return self._alignment_failure_or_wait_locked(
                    'waiting for alignment baseline: matched_gps=%d baseline=%.2fm/%.2fm'
                    % (len(times), gps_xy_span, self.alignment_min_baseline_m),
                    odom_path_m,
                )

            baseline_ratio = gps_xy_span / max(odom_xy_span, 1e-6)
            if not (
                self.alignment_min_baseline_ratio
                <= baseline_ratio
                <= self.alignment_max_baseline_ratio
            ):
                return self._alignment_failure_or_wait_locked(
                    'rejecting alignment motion mismatch: gps_baseline=%.2fm '
                    'odom_baseline=%.2fm ratio=%.2f allowed=[%.2f,%.2f]'
                    % (
                        gps_xy_span,
                        odom_xy_span,
                        baseline_ratio,
                        self.alignment_min_baseline_ratio,
                        self.alignment_max_baseline_ratio,
                    ),
                    odom_path_m,
                )

            weights = 1.0 / np.maximum(accuracies, 1e-6)
            candidate_transform, xy_residuals, inliers = (
                self._estimate_robust_planar_alignment(local_points, gps_points, weights)
            )
            inlier_count = int(np.count_nonzero(inliers))
            inlier_ratio = inlier_count / float(len(times))
            required_inliers = max(
                2, int(math.ceil(self.alignment_min_inlier_ratio * len(times)))
            )
            if inlier_count < required_inliers:
                return self._alignment_failure_or_wait_locked(
                    'rejecting alignment outliers: inliers=%d/%d(%.0f%%) required=%.0f%%'
                    % (
                        inlier_count,
                        len(times),
                        100.0 * inlier_ratio,
                        100.0 * self.alignment_min_inlier_ratio,
                    ),
                    odom_path_m,
                )

            xy_rmse = float(np.sqrt(np.mean(np.square(xy_residuals[inliers]))))
            if xy_rmse > self.alignment_max_xy_rmse_m:
                return self._alignment_failure_or_wait_locked(
                    'rejecting alignment residual: xy_rmse=%.2fm/%.2fm inliers=%d/%d'
                    % (xy_rmse, self.alignment_max_xy_rmse_m, inlier_count, len(times)),
                    odom_path_m,
                )

            candidate_yaw = math.atan2(
                candidate_transform[1, 0], candidate_transform[0, 0]
            )
            if self.alignment_observation_transform is None:
                self.alignment_observation_transform = candidate_transform.copy()
                self.alignment_observation_yaw = candidate_yaw
                self.alignment_observation_start_path_m = odom_path_m
                self.alignment_observation_start_time = times[-1]
                self.alignment_yaw_candidates.clear()
                self.last_alignment_candidate_path_m = None

            yaw_stable, yaw_max_delta_deg = self._yaw_is_stable(
                candidate_yaw, odom_path_m
            )
            observation_progress_m = max(
                0.0, odom_path_m - self.alignment_observation_start_path_m
            )
            if observation_progress_m < self.alignment_observation_distance_m:
                candidate_count = len(self.alignment_yaw_candidates)
                self.alignment_status = (
                    'observing alignment: progress=%.2fm/%.2fm candidates=%d/%d '
                    'yaw=%.2fdeg yaw_delta=%.2fdeg/%.2fdeg '
                    'xy_rmse=%.2fm inliers=%.0f%%'
                    % (
                        observation_progress_m,
                        self.alignment_observation_distance_m,
                        candidate_count,
                        self.alignment_required_stable_yaw_estimates,
                        math.degrees(candidate_yaw),
                        yaw_max_delta_deg,
                        self.alignment_max_stable_yaw_delta_deg,
                        xy_rmse,
                        100.0 * inlier_ratio,
                    )
                )
                return False

            if not yaw_stable:
                return self._alignment_failure_or_wait_locked(
                    'rejecting unstable yaw after observation: candidates=%d/%d '
                    'max_delta=%.2fdeg/%.2fdeg yaw=%.2fdeg'
                    % (
                        len(self.alignment_yaw_candidates),
                        self.alignment_required_stable_yaw_estimates,
                        yaw_max_delta_deg,
                        self.alignment_max_stable_yaw_delta_deg,
                        math.degrees(candidate_yaw),
                    ),
                    odom_path_m,
                )

            initial_to_observed_yaw_delta_deg = abs(math.degrees(
                self._angle_difference(candidate_yaw, self.alignment_observation_yaw)
            ))
            if (
                initial_to_observed_yaw_delta_deg
                > self.alignment_max_initial_to_observed_yaw_delta_deg
            ):
                return self._alignment_failure_or_wait_locked(
                    'rejecting inaccurate heading after observation: yaw_change=%.2fdeg/%.2fdeg'
                    % (
                        initial_to_observed_yaw_delta_deg,
                        self.alignment_max_initial_to_observed_yaw_delta_deg,
                    ),
                    odom_path_m,
                )

            observation = self._evaluate_observation_locked(
                times, local_points, gps_points
            )
            if not observation['passed']:
                return self._alignment_failure_or_wait_locked(
                    'rejecting failed holdout validation: %s' % observation['reason'],
                    odom_path_m,
                )

            self.wgps_t_wvio = candidate_transform

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
                'aligned: yaw=%.2fdeg samples=%d gps_baseline=%.2fm '
                'odom_baseline=%.2fm xy_rmse=%.2fm inliers=%.0f%% yaw_delta=%.2fdeg '
                'observation=%.2fm holdout_rmse=%.2fm holdout_inliers=%.0f%% '
                'observed_yaw_change=%.2fdeg resets=%d'
                % (
                    math.degrees(candidate_yaw),
                    len(times),
                    gps_xy_span,
                    odom_xy_span,
                    xy_rmse,
                    100.0 * inlier_ratio,
                    yaw_max_delta_deg,
                    observation_progress_m,
                    observation['rmse'],
                    100.0 * observation['inlier_ratio'],
                    initial_to_observed_yaw_delta_deg,
                    self.alignment_reset_count,
                )
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

    def consume_alignment_reset_messages(self):
        with self.lock:
            messages = list(self.pending_alignment_reset_messages)
            self.pending_alignment_reset_messages.clear()
            return messages

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
        self.declare_parameter('alignment_min_baseline_m', 50.0)
        self.declare_parameter('alignment_min_odom_baseline_m', 50.0)
        self.declare_parameter('alignment_min_odom_sample_separation_m', 0.5)
        self.declare_parameter('alignment_min_baseline_ratio', 0.5)
        self.declare_parameter('alignment_max_baseline_ratio', 2.0)
        self.declare_parameter('alignment_inlier_threshold_m', 2.5)
        self.declare_parameter('alignment_min_inlier_ratio', 0.75)
        self.declare_parameter('alignment_max_xy_rmse_m', 1.5)
        self.declare_parameter('alignment_required_stable_yaw_estimates', 4)
        self.declare_parameter('alignment_stable_yaw_min_path_increment_m', 2.0)
        self.declare_parameter('alignment_max_stable_yaw_delta_deg', 2.0)
        self.declare_parameter('alignment_observation_distance_m', 10.0)
        self.declare_parameter('alignment_observation_min_gps_constraints', 5)
        self.declare_parameter('alignment_observation_min_inlier_ratio', 0.75)
        self.declare_parameter('alignment_observation_max_xy_rmse_m', 2.0)
        self.declare_parameter(
            'alignment_max_initial_to_observed_yaw_delta_deg', 2.0
        )
        self.declare_parameter('alignment_candidate_start_timeout_m', 15.0)
        self.declare_parameter('alignment_restart_on_observation_failure', True)
        self.declare_parameter('freeze_alignment_after_init', True)
        self.declare_parameter('record_vio_global_csv', True)
        self.declare_parameter('convert_gps_topic_wgs84_to_gcj02', True)

        #融合节点（fusion_node）自己生成vio_global.csv，记录了融合后的全局位姿。每一行代表一个时间点机器人在哪里、朝向哪里。
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

        #接收高层导航指令。
        #route_waypoints.json是上层输出的文件。它包含了机器人需要前往的目标点（Waypoints），通常包含经纬度（Lat/Lon）或地理围栏信息。
        #fusion_node 会定时（route_poll_sec）读取这个文件。
        #解析 JSON 数据，将其转换为机器人能理解的坐标（ENU，东北天坐标系）。
        self.declare_parameter(
            'route_json_path',
            os.path.join(
                POLICY_PART_DIR,
                'topomaps',
                'LLM_Policy',
                'route_waypoints.json',
            ),
        )

        #读取 route_json_path 中的人类可读坐标（经纬度）。
        #结合当前的 GPS 位置，计算出相对于机器人的坐标（ENU）。
        #将计算好的坐标写入 route_output_path.txt，供推理节点（model_inference.py Node）。
        self.declare_parameter(
            'route_output_path',
            os.path.join(
                POLICY_PART_DIR,
                'topomaps',
                'LLM_Policy',
                'LLM_UTM_odom_poses.txt',
            ),
        )


        self.declare_parameter('route_default_altitude', 33.0)
        self.declare_parameter('route_poll_sec', 1.0)
        self.estimator = Ros1GlobalFusionEstimator(
            max_optimization_poses=int(self.get_parameter('max_optimization_poses').value),
            optimization_mode=str(self.get_parameter('optimization_mode').value),
            alignment_min_gps_constraints=int(self.get_parameter('alignment_min_gps_constraints').value),
            alignment_min_baseline_m=float(self.get_parameter('alignment_min_baseline_m').value),
            alignment_min_odom_baseline_m=float(
                self.get_parameter('alignment_min_odom_baseline_m').value
            ),
            alignment_min_odom_sample_separation_m=float(
                self.get_parameter('alignment_min_odom_sample_separation_m').value
            ),
            alignment_min_baseline_ratio=float(
                self.get_parameter('alignment_min_baseline_ratio').value
            ),
            alignment_max_baseline_ratio=float(
                self.get_parameter('alignment_max_baseline_ratio').value
            ),
            alignment_inlier_threshold_m=float(
                self.get_parameter('alignment_inlier_threshold_m').value
            ),
            alignment_min_inlier_ratio=float(
                self.get_parameter('alignment_min_inlier_ratio').value
            ),
            alignment_max_xy_rmse_m=float(
                self.get_parameter('alignment_max_xy_rmse_m').value
            ),
            alignment_required_stable_yaw_estimates=int(
                self.get_parameter('alignment_required_stable_yaw_estimates').value
            ),
            alignment_stable_yaw_min_path_increment_m=float(
                self.get_parameter('alignment_stable_yaw_min_path_increment_m').value
            ),
            alignment_max_stable_yaw_delta_deg=float(
                self.get_parameter('alignment_max_stable_yaw_delta_deg').value
            ),
            alignment_observation_distance_m=float(
                self.get_parameter('alignment_observation_distance_m').value
            ),
            alignment_observation_min_gps_constraints=int(
                self.get_parameter('alignment_observation_min_gps_constraints').value
            ),
            alignment_observation_min_inlier_ratio=float(
                self.get_parameter('alignment_observation_min_inlier_ratio').value
            ),
            alignment_observation_max_xy_rmse_m=float(
                self.get_parameter('alignment_observation_max_xy_rmse_m').value
            ),
            alignment_max_initial_to_observed_yaw_delta_deg=float(
                self.get_parameter(
                    'alignment_max_initial_to_observed_yaw_delta_deg'
                ).value
            ),
            alignment_candidate_start_timeout_m=float(
                self.get_parameter('alignment_candidate_start_timeout_m').value
            ),
            alignment_restart_on_observation_failure=bool(
                self.get_parameter(
                    'alignment_restart_on_observation_failure'
                ).value
            ),
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
                'alignment gate: min_gps_constraints=%d min_gps_baseline=%.2f '
                'min_odom_baseline=%.2f sample_separation=%.2f baseline_ratio=[%.2f,%.2f] '
                'inlier_threshold=%.2f min_inlier_ratio=%.2f max_xy_rmse=%.2f '
                'stable_yaws=%d path_increment=%.2f max_yaw_delta=%.2fdeg '
                'observation=%.2fm observation_constraints=%d observation_inliers=%.2f '
                'observation_rmse=%.2f observed_yaw_delta=%.2fdeg '
                'candidate_timeout=%.2fm auto_restart=%s freeze_after_init=%s'
                % (
                    int(self.get_parameter('alignment_min_gps_constraints').value),
                    float(self.get_parameter('alignment_min_baseline_m').value),
                    float(self.get_parameter('alignment_min_odom_baseline_m').value),
                    float(self.get_parameter('alignment_min_odom_sample_separation_m').value),
                    float(self.get_parameter('alignment_min_baseline_ratio').value),
                    float(self.get_parameter('alignment_max_baseline_ratio').value),
                    float(self.get_parameter('alignment_inlier_threshold_m').value),
                    float(self.get_parameter('alignment_min_inlier_ratio').value),
                    float(self.get_parameter('alignment_max_xy_rmse_m').value),
                    int(self.get_parameter('alignment_required_stable_yaw_estimates').value),
                    float(
                        self.get_parameter('alignment_stable_yaw_min_path_increment_m').value
                    ),
                    float(self.get_parameter('alignment_max_stable_yaw_delta_deg').value),
                    float(self.get_parameter('alignment_observation_distance_m').value),
                    int(
                        self.get_parameter(
                            'alignment_observation_min_gps_constraints'
                        ).value
                    ),
                    float(
                        self.get_parameter(
                            'alignment_observation_min_inlier_ratio'
                        ).value
                    ),
                    float(
                        self.get_parameter(
                            'alignment_observation_max_xy_rmse_m'
                        ).value
                    ),
                    float(
                        self.get_parameter(
                            'alignment_max_initial_to_observed_yaw_delta_deg'
                        ).value
                    ),
                    float(
                        self.get_parameter('alignment_candidate_start_timeout_m').value
                    ),
                    bool(
                        self.get_parameter(
                            'alignment_restart_on_observation_failure'
                        ).value
                    ),
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
            aligned = self.estimator.optimize()
            reset_messages = self.estimator.consume_alignment_reset_messages()
            for message in reset_messages:
                self.get_logger().warning(message)
                self.last_alignment_wait_log = message

            if aligned:
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
                gps_latitude = float(gps_msg.latitude)
                gps_longitude = float(gps_msg.longitude)
                if bool(self.get_parameter('convert_gps_topic_wgs84_to_gcj02').value):
                    gps_longitude, gps_latitude = wgs84_to_gcj02(gps_longitude, gps_latitude)
                self.estimator.input_gps(
                    vio_t,
                    gps_latitude,
                    gps_longitude,
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
                    'Route JSON conversion is waiting for the first GPS fix; hardware WGS-84 GPS will be converted to GCJ-02 before building the ENU origin.'
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
        self.get_logger().info(
            'Converting route with %d GCJ-02 waypoints using the GCJ-02 hardware GPS origin'
            % len(waypoints)
        )
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
                    'Waypoint %d: GCJ02 Lat=%.6f, Lon=%.6f, Alt=%.2f -> ENU(%.2f, %.2f, %.2f)'
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
