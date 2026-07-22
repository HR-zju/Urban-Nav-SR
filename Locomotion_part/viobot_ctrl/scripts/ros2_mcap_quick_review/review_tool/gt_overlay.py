from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


COLORS_BGR = [
    (100, 230, 100),
    (100, 100, 255),
    (255, 100, 100),
    (100, 255, 255),
]


def _normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("VIO contains a zero-length quaternion")
    return quaternions / norms


def _continuous_quaternions(quaternions: np.ndarray) -> np.ndarray:
    result = _normalize_quaternions(quaternions.astype(np.float64, copy=True))
    for index in range(1, len(result)):
        if float(np.dot(result[index - 1], result[index])) < 0.0:
            result[index] *= -1.0
    return result


def _slerp_one(
    sample_time: float,
    times: np.ndarray,
    quaternions: np.ndarray,
) -> np.ndarray:
    right = int(np.searchsorted(times, sample_time, side="right"))
    right = max(1, min(len(times) - 1, right))
    left = right - 1
    alpha = (sample_time - times[left]) / (times[right] - times[left])
    q0, q1 = quaternions[left], quaternions[right]
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    dot = max(0.0, min(1.0, dot))
    if dot > 0.9995:
        result = q0 + alpha * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = math.acos(dot)
    return (
        math.sin((1.0 - alpha) * theta) * q0
        + math.sin(alpha * theta) * q1
    ) / math.sin(theta)


def _yaw_from_quaternion(quaternion: np.ndarray) -> float:
    x, y, z, w = quaternion
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _rotate_xy(points: np.ndarray, yaw: float) -> np.ndarray:
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    rotation = np.array(
        [[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64
    )
    return points @ rotation.T


def _offset_curves(waypoints: np.ndarray, half_width: float):
    if len(waypoints) < 2:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty
    tangents = np.empty_like(waypoints, dtype=np.float64)
    tangents[0] = waypoints[1] - waypoints[0]
    tangents[-1] = waypoints[-1] - waypoints[-2]
    if len(waypoints) > 2:
        tangents[1:-1] = waypoints[2:] - waypoints[:-2]
    norms = np.linalg.norm(tangents, axis=1)
    valid = norms >= 1e-6
    if not np.any(valid):
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty
    valid_indices = np.flatnonzero(valid)
    for index in np.flatnonzero(~valid):
        nearest = valid_indices[np.argmin(np.abs(valid_indices - index))]
        tangents[index], norms[index] = tangents[nearest], norms[nearest]
    unit = tangents / norms[:, None]
    normals = np.column_stack((-unit[:, 1], unit[:, 0]))
    return waypoints + half_width * normals, waypoints - half_width * normals


def _project_ground_points(
    points: np.ndarray,
    intrinsics,
    distortion,
    camera_height: float,
    camera_x_offset: float,
    pitch_deg: float,
) -> np.ndarray:
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    if len(points) == 0:
        return pixels
    ego_points = np.zeros((len(points), 3), dtype=np.float64)
    ego_points[:, :2] = points
    translation = np.array(
        [camera_x_offset, 0.0, camera_height], dtype=np.float64
    )
    base_rotation = np.array(
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    pitch = math.radians(-pitch_deg)
    pitch_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ],
        dtype=np.float64,
    )
    camera_points = (
        pitch_rotation @ base_rotation @ (ego_points - translation).T
    ).T
    visible = camera_points[:, 2] > 1e-6
    if not np.any(visible):
        return pixels
    x = camera_points[visible, 0] / camera_points[visible, 2]
    y = camera_points[visible, 1] / camera_points[visible, 2]
    k1, k2, k3 = distortion
    radius2 = x * x + y * y
    radial = 1.0 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
    fx, fy, cx, cy = intrinsics
    pixels[visible, 0] = fx * x * radial + cx
    pixels[visible, 1] = fy * y * radial + cy
    return pixels


def _bounded(points: np.ndarray, width: int, height: int) -> np.ndarray:
    if len(points) == 0:
        return points
    finite = points[np.all(np.isfinite(points), axis=1)]
    if len(finite) == 0:
        return finite
    margin = 8.0
    return finite[
        (finite[:, 0] >= -width * margin)
        & (finite[:, 0] <= width * (1.0 + margin))
        & (finite[:, 1] >= -height * margin)
        & (finite[:, 1] <= height * (1.0 + margin))
    ]


def _draw_projected_band(
    image: np.ndarray,
    waypoints: np.ndarray,
    center: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    color,
) -> None:
    if len(waypoints) < 2:
        return
    path_length = float(np.linalg.norm(np.diff(waypoints, axis=0), axis=1).sum())
    if path_length < 0.05:
        return
    height, width = image.shape[:2]
    center = _bounded(center, width, height)
    left = _bounded(left, width, height)
    right = _bounded(right, width, height)
    if len(left) > 1 and len(right) > 1:
        polygon = np.vstack([left, right[::-1]]).astype(np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        ys, xs = np.where(mask > 0)
        if len(ys):
            alpha = (0.15 + 0.45 * (ys / max(1, height - 1)) ** 1.5)[:, None]
            source = image[ys, xs].astype(np.float32)
            target = np.asarray(color, dtype=np.float32)[None, :]
            image[ys, xs] = source * (1.0 - alpha) + target * alpha
        edge_color = tuple(int(component * 0.7) for component in color)
        for boundary in (left, right):
            visible = boundary[
                (boundary[:, 0] >= 0)
                & (boundary[:, 0] < width)
                & (boundary[:, 1] >= 0)
                & (boundary[:, 1] < height)
            ].astype(np.int32)
            if len(visible) > 1:
                cv2.polylines(image, [visible], False, edge_color, 3, cv2.LINE_AA)
    visible_center = center[
        (center[:, 0] >= 0)
        & (center[:, 0] < width)
        & (center[:, 1] >= 0)
        & (center[:, 1] < height)
    ]
    if len(visible_center):
        visible_center = np.vstack(
            [np.array([[width // 2, height - 1]]), visible_center]
        ).astype(np.int32)
        cv2.polylines(image, [visible_center], False, color, 2, cv2.LINE_AA)


class GroundTruthOverlay:
    def __init__(self, poses, config: dict[str, Any], yaw_offset_deg: float):
        self.config = config
        self.enabled = bool(config.get("enabled", False))
        self.available = False
        self.applied_frames = 0
        self.skipped_frames = 0
        self.warnings: list[str] = []
        self.horizons = [float(value) for value in config.get("horizons_sec", [])]
        self.horizons = [value for value in self.horizons if value > 0.0]
        self.sample_fps = float(config.get("trajectory_sample_fps", 20.0))
        self.time_offset_sec = float(config.get("camera_time_offset_ms", 0.0)) / 1000.0
        self.yaw_offset_rad = math.radians(float(yaw_offset_deg))
        self.intrinsics = tuple(float(x) for x in config.get("intrinsics", []))
        self.distortion = tuple(float(x) for x in config.get("distortion", []))
        self.camera_height = float(config.get("camera_height", 0.68))
        self.camera_x_offset = float(config.get("camera_x_offset", -0.1))
        self.pitch_deg = float(config.get("pitch_deg", 8.0))
        self.path_half_width = float(config.get("path_half_width", 0.22))
        self.lateral_convention = str(
            config.get("odom_lateral_convention", "left_positive")
        )

        if not self.enabled:
            return
        if not self.horizons or self.sample_fps <= 0.0:
            raise ValueError("gt_overlay horizons_sec/sample_fps are invalid")
        if len(self.intrinsics) != 4 or len(self.distortion) != 3:
            raise ValueError("gt_overlay requires 4 intrinsics and 3 distortion values")
        if self.camera_height <= 0.0 or self.path_half_width <= 0.0:
            raise ValueError("gt_overlay camera height/path width must be positive")
        if self.lateral_convention not in {"left_positive", "right_positive"}:
            raise ValueError("gt_overlay odom_lateral_convention is invalid")

        rows = []
        invalid_pose_count = 0
        for pose in poses:
            row = [
                pose.t,
                pose.x,
                pose.y,
                pose.qx,
                pose.qy,
                pose.qz,
                pose.qw,
            ]
            quaternion_norm = math.sqrt(sum(float(value) ** 2 for value in row[3:7]))
            if (
                all(math.isfinite(float(value)) for value in row)
                and quaternion_norm >= 1e-8
            ):
                rows.append(row)
            else:
                invalid_pose_count += 1
        if invalid_pose_count:
            self.warnings.append(
                "GT ignored {} invalid VIO pose(s)".format(invalid_pose_count)
            )
        if len(rows) < 2:
            self.warnings.append("GT disabled: fewer than two valid VIO poses")
            return
        values = np.asarray(rows, dtype=np.float64)
        values = values[np.argsort(values[:, 0], kind="stable")]
        keep = np.r_[values[1:, 0] != values[:-1, 0], True]
        values = values[keep]
        if len(values) < 2:
            self.warnings.append("GT disabled: VIO timestamps are not distinct")
            return
        self.times = values[:, 0]
        self.positions = values[:, 1:3]
        self.quaternions = _continuous_quaternions(values[:, 3:7])
        self.available = True

    def draw(self, image: np.ndarray, camera_time: float) -> bool:
        if not self.enabled or not self.available:
            return False
        pose_time = float(camera_time) - self.time_offset_sec
        if pose_time < self.times[0] or pose_time > self.times[-1]:
            self.skipped_frames += 1
            return False
        max_end = min(pose_time + max(self.horizons), self.times[-1])
        if max_end <= pose_time:
            self.skipped_frames += 1
            return False
        step = 1.0 / self.sample_fps
        sample_times = np.arange(pose_time, max_end + step * 0.5, step)
        sample_times = sample_times[sample_times <= max_end]
        horizon_ends = [min(pose_time + value, max_end) for value in self.horizons]
        sample_times = np.unique(np.r_[sample_times, horizon_ends, max_end])
        positions = np.column_stack(
            [
                np.interp(sample_times, self.times, self.positions[:, axis])
                for axis in range(2)
            ]
        )
        current_position = np.array(
            [
                np.interp(pose_time, self.times, self.positions[:, axis])
                for axis in range(2)
            ]
        )
        orientation = _slerp_one(pose_time, self.times, self.quaternions)
        yaw = _yaw_from_quaternion(orientation) + self.yaw_offset_rad
        waypoints = _rotate_xy(positions - current_position, -yaw)
        if self.lateral_convention == "right_positive":
            waypoints[:, 1] *= -1.0

        left, right = _offset_curves(waypoints, self.path_half_width)
        if len(left) == 0:
            self.skipped_frames += 1
            return False
        projection_args = (
            self.intrinsics,
            self.distortion,
            self.camera_height,
            self.camera_x_offset,
            self.pitch_deg,
        )
        center_pixels = _project_ground_points(waypoints, *projection_args)
        left_pixels = _project_ground_points(left, *projection_args)
        right_pixels = _project_ground_points(right, *projection_args)
        for index, horizon in enumerate(self.horizons):
            end_time = min(pose_time + horizon, max_end)
            point_count = int(np.searchsorted(sample_times, end_time, side="right"))
            point_count = max(2, min(point_count, len(sample_times)))
            _draw_projected_band(
                image,
                waypoints[:point_count],
                center_pixels[:point_count],
                left_pixels[:point_count],
                right_pixels[:point_count],
                COLORS_BGR[index % len(COLORS_BGR)],
            )
        self.applied_frames += 1
        return True

    def draw_legend(self, image: np.ndarray) -> None:
        if not self.enabled or not self.horizons:
            return
        height, width = image.shape[:2]
        box_width = 118
        box_height = 12 + 23 * len(self.horizons)
        x0, y0 = max(5, width - box_width - 10), 10
        overlay = image.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_width, y0 + box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)
        for index, horizon in enumerate(self.horizons):
            y = y0 + 20 + index * 23
            color = COLORS_BGR[index % len(COLORS_BGR)]
            cv2.rectangle(image, (x0 + 8, y - 11), (x0 + 22, y + 3), color, -1)
            cv2.putText(
                image,
                "GT {:.1f}s".format(horizon),
                (x0 + 29, y + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "available": self.available,
            "applied_source_frames": self.applied_frames,
            "skipped_source_frames": self.skipped_frames,
            "horizons_sec": self.horizons,
            "warnings": self.warnings,
        }
