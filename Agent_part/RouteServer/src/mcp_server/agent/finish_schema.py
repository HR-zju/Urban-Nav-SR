"""Canonical ``<finish>`` schema helpers for SideWalk Agentic Stage2.

RouteServer keeps trajectory supervision in top-level row fields such as
``pixel_trajectory``.  The assistant text should use the ms-swift prompt
contract: ``trajectory_grid1000`` plus ``trajectory_coord_frame=norm1000``.
This module is the shared compatibility layer for building canonical finish
payloads and validating legacy ``trajectory`` payloads. Trajectory-bearing
finishes explicitly carry ``decision_type=navigate`` so the XML-like tag is
never interpreted as task completion.
"""

from __future__ import annotations

import math
from typing import Any


NORM1000_COORD_FRAME = "norm1000"
GRID_SIZE = 1000
TRAJECTORY_MATCH_ATOL = 1.5 / GRID_SIZE


def norm01_to_grid1000(points: Any, *, grid_size: int = GRID_SIZE) -> list[list[int]]:
    """Convert normalized FPV points to integer norm1000 grid points."""

    trajectory = coerce_trajectory_points(points)
    if not trajectory:
        return []
    return [
        [
            int(round(_clamp(float(point[0]), 0.0, 1.0) * grid_size)),
            int(round(_clamp(float(point[1]), 0.0, 1.0) * grid_size)),
        ]
        for point in trajectory
    ]


def grid1000_to_norm01(points: Any, *, grid_size: int = GRID_SIZE) -> list[list[float]]:
    """Convert integer norm1000 grid points to normalized FPV points."""

    trajectory = coerce_trajectory_points(points)
    if not trajectory:
        return []
    return [
        [
            _clamp(float(point[0]) / grid_size, 0.0, 1.0),
            _clamp(float(point[1]) / grid_size, 0.0, 1.0),
        ]
        for point in trajectory
    ]


def coerce_trajectory_points(points: Any) -> list[list[float]]:
    """Return ``[[x, y], ...]`` from a JSON-like trajectory payload."""

    if not isinstance(points, list):
        return []
    trajectory: list[list[float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return []
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            return []
        if not math.isfinite(x) or not math.isfinite(y):
            return []
        trajectory.append([x, y])
    return trajectory


def build_finish_payload(base: dict[str, Any], *, trajectory_norm: Any) -> dict[str, Any]:
    """Build a canonical assistant ``<finish>`` payload.

    Args:
        base: Non-trajectory finish fields such as route action, intent, tool
            use, and teacher source.
        trajectory_norm: Normalized FPV trajectory points in ``[0, 1]``.

    Returns:
        A JSON-ready finish object that uses the ms-swift canonical trajectory
        fields and omits legacy ``trajectory``.
    """

    finish = {
        key: value
        for key, value in base.items()
        if value is not None and key not in {"trajectory", "trajectory_norm", "trajectory_grid1000"}
    }
    # 轨迹型 canonical finish 始终是继续导航，不允许由 <finish> 标签暗示任务完成。
    finish.setdefault("decision_type", "navigate")
    finish["trajectory_grid1000"] = norm01_to_grid1000(trajectory_norm)
    finish["trajectory_coord_frame"] = NORM1000_COORD_FRAME
    return finish


def parse_finish_trajectory_norm(finish: dict[str, Any] | None) -> list[list[float]] | None:
    """Extract normalized trajectory points from new or legacy finish payloads."""

    if not isinstance(finish, dict):
        return None
    if finish.get("trajectory_grid1000") is not None:
        trajectory = grid1000_to_norm01(finish.get("trajectory_grid1000"))
        return trajectory or None
    if finish.get("trajectory_norm") is not None:
        trajectory = coerce_trajectory_points(finish.get("trajectory_norm"))
        return trajectory or None

    trajectory = finish.get("trajectory")
    if trajectory is None:
        return None
    if is_norm1000_frame(
        finish.get("trajectory_coord_frame")
        or finish.get("trajectory_coordinate")
        or finish.get("coordinate_frame")
    ):
        parsed = grid1000_to_norm01(trajectory)
    else:
        parsed = coerce_trajectory_points(trajectory)
    return parsed or None


def finish_trajectory_matches(
    finish: dict[str, Any] | None,
    expected_norm: Any,
    *,
    atol: float = TRAJECTORY_MATCH_ATOL,
) -> bool:
    """Return whether finish text trajectory matches top-level supervision."""

    actual = parse_finish_trajectory_norm(finish)
    expected = coerce_trajectory_points(expected_norm)
    if actual is None or not expected or len(actual) != len(expected):
        return False
    for actual_point, expected_point in zip(actual, expected):
        if abs(float(actual_point[0]) - float(expected_point[0])) > atol:
            return False
        if abs(float(actual_point[1]) - float(expected_point[1])) > atol:
            return False
    return True


def is_norm1000_frame(value: Any) -> bool:
    """Return True for accepted norm1000 coordinate-frame aliases."""

    text = str(value or "").strip().lower()
    return text in {"norm1000", "grid1000", "fpv_norm1000", "image_norm1000"}


def canonicalize_finish_trajectory(payload: dict[str, Any], *, trajectory_norm: Any) -> dict[str, Any]:
    """Return ``payload`` with canonical trajectory fields replaced."""

    canonical = dict(payload)
    for key in ("trajectory", "trajectory_norm", "trajectory_grid1000", "trajectory_coordinate", "coordinate_frame"):
        canonical.pop(key, None)
    canonical["trajectory_grid1000"] = norm01_to_grid1000(trajectory_norm)
    canonical["trajectory_coord_frame"] = NORM1000_COORD_FRAME
    canonical.setdefault("decision_type", "navigate")
    return canonical


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
