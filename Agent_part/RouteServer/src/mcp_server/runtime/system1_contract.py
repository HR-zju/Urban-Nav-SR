"""Canonical System1 trajectory contract and coordinate conversions.

RouteServer may ingest historical sidewalk labels in image-oriented
``x_right/y_front`` axes, but every v2 System1 boundary exports one metric FLU
trajectory.  Keeping the conversion here prevents trainer and runtime code
from inferring semantics from a bare ``[T, 3]`` shape.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


SYSTEM1_TARGET_FIELD_TRAJECTORY_EGO_T = "trajectory_ego_t"
SYSTEM1_TARGET_FORMAT_EGO_POSE_XYT_FLU_V1 = "ego_pose_xyt_flu_v1"
SYSTEM1_REPRESENTATION_CUMULATIVE_RELATIVE_POSE = "cumulative_relative_pose"
SYSTEM1_COORDINATE_FRAME_CURRENT_EGO_FLU = "current_ego_flu"
SYSTEM1_AXIS_NAMES_EGO_POSE_XYT_FLU = (
    "x_forward_m",
    "y_left_m",
    "yaw_rel_ccw_rad",
)
SYSTEM1_UNITS_EGO_POSE_XYT_FLU = ("meter", "meter", "radian")
SYSTEM1_YAW_SOURCES = frozenset({"odometry", "path_tangent"})

# 只用于识别并迁移历史 RouteServer row，不能作为新 v2 输出格式。
LEGACY_SYSTEM1_TARGET_FORMAT_RIGHT_FRONT_XYT = "ego_xy_heading_delta"
LEGACY_SYSTEM1_TARGET_FORMAT_RIGHT_FRONT_XYTS = "ego_xy_t_speed_right_front_v1"
LEGACY_RIGHT_FRONT_FORMATS = frozenset(
    {
        LEGACY_SYSTEM1_TARGET_FORMAT_RIGHT_FRONT_XYT,
        LEGACY_SYSTEM1_TARGET_FORMAT_RIGHT_FRONT_XYTS,
    }
)


def wrap_angle_rad(angle: float) -> float:
    """Wrap one angle to ``[-pi, pi)``.

    Args:
        angle: Angle in radians.

    Returns:
        Wrapped angle in radians.
    """

    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def canonicalize_trajectory_ego_t(
    value: Any,
    *,
    source_format: str | None = None,
) -> list[list[float]]:
    """Convert one trajectory to metric cumulative FLU ``[x, y, yaw]``.

    Args:
        value: Sequence shaped ``[T, 3]`` or historical ``[T, 4]``.
        source_format: Declared input format.  Canonical FLU input is preserved;
            absent/legacy input is interpreted as ``x_right/y_front`` and its
            yaw is deterministically recomputed from the path tangent.

    Returns:
        Rows ``[x_forward_m, y_left_m, yaw_rel_ccw_rad]`` in the fixed ego
        frame at the current observation time.

    Raises:
        ValueError: If the input is empty, non-finite, has an unsupported width,
            or declares an unknown coordinate format.

    Notes:
        Historical coordinates are converted by
        ``x_forward = y_front`` and ``y_left = -x_right``.  Their tangent yaw is
        ``atan2(delta_y_left, delta_x_forward)`` and is therefore CCW-positive.
    """

    rows = list(value or [])
    if not rows:
        raise ValueError("trajectory_ego_t must be a non-empty sequence for System1 export.")
    normalized_format = str(source_format or "").strip()
    if normalized_format and normalized_format not in {
        SYSTEM1_TARGET_FORMAT_EGO_POSE_XYT_FLU_V1,
        *LEGACY_RIGHT_FRONT_FORMATS,
    }:
        raise ValueError(f"Unsupported trajectory_ego_t source format: {normalized_format}")

    # 已声明 canonical 的 row 只做数值和角度校验，禁止再次交换坐标轴。
    if normalized_format == SYSTEM1_TARGET_FORMAT_EGO_POSE_XYT_FLU_V1:
        canonical = [_validated_xyt_row(row, index=index) for index, row in enumerate(rows)]
        return [[x, y, wrap_angle_rad(yaw)] for x, y, yaw in canonical]

    # 未声明格式的现有 RouteServer row 均来自历史 x_right/y_front exporter。
    canonical: list[list[float]] = []
    prev_forward = 0.0
    prev_left = 0.0
    prev_yaw = 0.0
    for index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)):
            raise ValueError(f"trajectory_ego_t row {index} is not a sequence.")
        if len(row) not in {3, 4}:
            raise ValueError(f"trajectory_ego_t row {index} has width {len(row)}; expected 3 or 4.")
        # legacy 的 dt/speed 不属于 System1 三维目标；这里只读取累计平面位置。
        x_right = _finite_float(row[0], name=f"trajectory_ego_t[{index}][0]")
        y_front = _finite_float(row[1], name=f"trajectory_ego_t[{index}][1]")
        x_forward = y_front
        y_left = -x_right
        delta_forward = x_forward - prev_forward
        delta_left = y_left - prev_left
        if abs(delta_forward) + abs(delta_left) > 1e-8:
            yaw = math.atan2(delta_left, delta_forward)
        else:
            yaw = prev_yaw
        canonical.append([x_forward, y_left, wrap_angle_rad(yaw)])
        prev_forward = x_forward
        prev_left = y_left
        prev_yaw = yaw
    return canonical


def canonical_trajectory_policy(
    policy: Mapping[str, Any] | None,
    *,
    horizon_steps: int,
    yaw_source: str,
) -> dict[str, Any]:
    """Return trajectory metadata that fully declares the FLU contract.

    Args:
        policy: Existing source metadata to preserve where it does not conflict.
        horizon_steps: Number of exported future waypoints.
        yaw_source: ``odometry`` or ``path_tangent``.

    Returns:
        Canonical policy metadata with metric units and no normalization.

    Raises:
        ValueError: If ``yaw_source`` is unsupported.
    """

    if yaw_source not in SYSTEM1_YAW_SOURCES:
        raise ValueError(f"Unsupported System1 yaw_source: {yaw_source}")
    payload = dict(policy or {})
    # canonical 关键字段必须覆盖 source metadata，避免旧坐标语义继续透传。
    payload.update(
        {
            "system1_target_field": SYSTEM1_TARGET_FIELD_TRAJECTORY_EGO_T,
            "system1_target_format": SYSTEM1_TARGET_FORMAT_EGO_POSE_XYT_FLU_V1,
            "representation": SYSTEM1_REPRESENTATION_CUMULATIVE_RELATIVE_POSE,
            "coordinate_frame": SYSTEM1_COORDINATE_FRAME_CURRENT_EGO_FLU,
            "axis_names": list(SYSTEM1_AXIS_NAMES_EGO_POSE_XYT_FLU),
            "system1_target_columns": list(SYSTEM1_AXIS_NAMES_EGO_POSE_XYT_FLU),
            "units": list(SYSTEM1_UNITS_EGO_POSE_XYT_FLU),
            "yaw_zero_direction": "forward",
            "yaw_positive_direction": "counter_clockwise",
            "yaw_source": yaw_source,
            "sampling_type": "temporal",
            "horizon_steps": int(horizon_steps),
            "normalized": False,
        }
    )
    # 采样间隔属于 policy metadata，不能重新占用 trajectory_ego_t 的列。
    if payload.get("dt_sec") is None and payload.get("frame_dt_sec") is not None:
        payload["dt_sec"] = float(payload["frame_dt_sec"])
    return payload


def _validated_xyt_row(row: Sequence[Any], *, index: int) -> list[float]:
    if not isinstance(row, (list, tuple)) or len(row) != 3:
        width = len(row) if isinstance(row, (list, tuple)) else None
        raise ValueError(f"trajectory_ego_t row {index} has width {width}; expected canonical width 3.")
    return [
        _finite_float(row[0], name=f"trajectory_ego_t[{index}][0]"),
        _finite_float(row[1], name=f"trajectory_ego_t[{index}][1]"),
        _finite_float(row[2], name=f"trajectory_ego_t[{index}][2]"),
    ]


def _finite_float(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number
