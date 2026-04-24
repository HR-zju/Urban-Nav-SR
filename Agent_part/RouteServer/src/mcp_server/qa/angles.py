from __future__ import annotations

import math


THETA_MIN_DEG = -90.0
THETA_MAX_DEG = 90.0
DEFAULT_THETA_MAX_DEG = 50.0
DEFAULT_BIN_COUNT = 12


def clamp_theta_deg(theta_deg: float) -> float:
    return max(THETA_MIN_DEG, min(THETA_MAX_DEG, float(theta_deg)))


def wrap_angle_deg(angle_deg: float) -> float:
    wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
    if wrapped == -180.0:
        return 180.0
    return wrapped


def bev_theta_deg(x_m: float, y_m: float) -> float:
    """Ego BEV theta: 0 is forward, positive is right, negative is left."""
    return clamp_theta_deg(math.degrees(math.atan2(float(x_m), float(y_m))))


def fpv_theta_from_u(u: float, width: int | float, theta_max_deg: float = DEFAULT_THETA_MAX_DEG) -> float:
    image_width = max(float(width), 1.0)
    normalized = (2.0 * float(u) / image_width) - 1.0
    return clamp_theta_deg(float(theta_max_deg) * normalized)


def u_from_fpv_theta(theta: float, width: int | float, theta_max_deg: float = DEFAULT_THETA_MAX_DEG) -> float:
    image_width = max(float(width), 1.0)
    max_theta = max(float(theta_max_deg), 1e-6)
    clamped = max(-max_theta, min(max_theta, float(theta)))
    return image_width * ((clamped / max_theta) + 1.0) * 0.5


def coarse_bin_from_theta(theta: float) -> str:
    clamped = clamp_theta_deg(theta)
    if clamped <= -60.0:
        return "left"
    if clamped <= -15.0:
        return "left_front"
    if clamped < 15.0:
        return "front"
    if clamped < 60.0:
        return "right_front"
    return "right"


def theta_bin_residual(theta: float, bin_count: int = DEFAULT_BIN_COUNT) -> tuple[int, float]:
    bins = max(int(bin_count), 1)
    clamped = clamp_theta_deg(theta)
    bin_width = (THETA_MAX_DEG - THETA_MIN_DEG) / bins
    raw_index = int((clamped - THETA_MIN_DEG) / bin_width)
    bin_index = min(max(raw_index, 0), bins - 1)
    center = THETA_MIN_DEG + (bin_index + 0.5) * bin_width
    return bin_index, wrap_angle_deg(clamped - center)
