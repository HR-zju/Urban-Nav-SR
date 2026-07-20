from __future__ import annotations
import math
from typing import Any
import cv2
import numpy as np

def nested_get(obj: Any, dotted_path: str) -> Any:
    if not dotted_path:
        raise AttributeError("Empty field path")
    current = obj
    for part in dotted_path.split("."):
        current = getattr(current, part)
    return current

def header_stamp_ns(msg: Any) -> int | None:
    try:
        stamp = msg.header.stamp
        value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        return value if value > 0 else None
    except Exception:
        return None

def _manual_image_to_bgr(msg: Any) -> np.ndarray:
    height, width, step = int(msg.height), int(msg.width), int(msg.step)
    encoding = str(msg.encoding).lower()
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if height <= 0 or width <= 0 or step <= 0 or raw.size < height * step:
        raise ValueError("Invalid image dimensions/step/data")
    rows = raw[: height * step].reshape(height, step)
    if encoding in {"bgr8", "8uc3"}:
        return rows[:, : width * 3].reshape(height, width, 3).copy()
    if encoding == "rgb8":
        return cv2.cvtColor(rows[:, : width * 3].reshape(height, width, 3), cv2.COLOR_RGB2BGR)
    if encoding in {"bgra8", "8uc4"}:
        return cv2.cvtColor(rows[:, : width * 4].reshape(height, width, 4), cv2.COLOR_BGRA2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(rows[:, : width * 4].reshape(height, width, 4), cv2.COLOR_RGBA2BGR)
    if encoding in {"mono8", "8uc1"}:
        return cv2.cvtColor(rows[:, :width].reshape(height, width), cv2.COLOR_GRAY2BGR)
    if encoding in {"mono16", "16uc1"}:
        raw16 = np.frombuffer(bytes(msg.data), dtype=np.uint16)
        pixels_per_row = step // 2
        img16 = raw16[: height * pixels_per_row].reshape(height, pixels_per_row)[:, :width]
        lo, hi = np.percentile(img16, [1, 99])
        gray = (
            np.zeros((height, width), dtype=np.uint8)
            if hi <= lo
            else np.clip((img16 - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        )
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unsupported image encoding without cv_bridge: {encoding}")

def image_to_bgr(msg: Any, type_name: str) -> np.ndarray:
    if type_name.endswith("/CompressedImage"):
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("cv2.imdecode failed")
        return image
    try:
        from cv_bridge import CvBridge
        return CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")
    except Exception:
        return _manual_image_to_bgr(msg)

def choose_topic(topic_types: dict[str, str], role_config: dict) -> str | None:
    names = [str(x) for x in role_config.get("names", [])]
    supported = set(str(x) for x in role_config.get("supported_types", []))
    hints = [str(x).lower() for x in role_config.get("name_hints", [])]
    for name in names:
        if name in topic_types:
            return name
    def fallback_rank(topic: str):
        # Repaired bags can contain the canonical topic together with diagnostic
        # aliases such as ``*_orig`` and ``*_repaired``.  Prefer the canonical
        # stream instead of depending on channel insertion order in the MCAP.
        leaf = topic.rsplit("/", 1)[-1].lower()
        alias_penalty = 1 if leaf.endswith("_repaired") else 0
        if leaf.endswith(("_orig", "_raw")):
            alias_penalty = 2
        return alias_penalty, len(topic), topic

    typed = sorted(
        (topic for topic, type_name in topic_types.items() if type_name in supported),
        key=fallback_rank,
    )
    for topic in typed:
        if any(hint in topic.lower() for hint in hints):
            return topic
    # Custom message types are accepted only when the topic name strongly
    # matches the role. Otherwise an Odometry topic could be incorrectly used
    # as both GPS and VIO. Ambiguous topics should be configured explicitly.
    for topic in sorted(topic_types, key=fallback_rank):
        if any(hint in topic.lower() for hint in hints):
            return topic
    if bool(role_config.get("allow_single_type_fallback", False)) and len(typed) == 1:
        return typed[0]
    return None

def _first_available(obj: Any, paths: list[str]) -> Any:
    last_exc = None
    for path in paths:
        if not path:
            continue
        try:
            return nested_get(obj, path)
        except Exception as exc:
            last_exc = exc
    if last_exc:
        raise last_exc
    raise AttributeError("No candidate field path")

def extract_gps(msg: Any, role_config: dict) -> dict[str, float | str]:
    fields = role_config.get("fields", {})
    mode = str(role_config.get("coordinate_mode", "auto")).lower()
    lat_path, lon_path = str(fields.get("latitude", "") or ""), str(fields.get("longitude", "") or "")
    x_path, y_path, z_path = (
        str(fields.get("x", "") or ""), str(fields.get("y", "") or ""), str(fields.get("z", "") or "")
    )
    if mode in {"auto", "latlon"}:
        try:
            lat = float(_first_available(msg, [lat_path, "latitude"]))
            lon = float(_first_available(msg, [lon_path, "longitude"]))
            if math.isfinite(lat) and math.isfinite(lon):
                return {"mode": "latlon", "x": lon, "y": lat, "z": 0.0}
        except Exception:
            if mode == "latlon":
                raise
    x = float(_first_available(msg, [x_path, "pose.pose.position.x", "pose.position.x", "position.x", "x"]))
    y = float(_first_available(msg, [y_path, "pose.pose.position.y", "pose.position.y", "position.y", "y"]))
    try:
        z = float(_first_available(msg, [z_path, "pose.pose.position.z", "pose.position.z", "position.z", "z"]))
    except Exception:
        z = 0.0
    if not all(math.isfinite(v) for v in (x, y, z)):
        raise ValueError("GPS position contains NaN/Inf")
    return {"mode": "xy", "x": x, "y": y, "z": z}

def extract_pose(msg: Any, role_config: dict) -> dict[str, float]:
    fields = role_config.get("fields", {})
    def value(name: str, candidates: list[str], default: float | None = None):
        custom = str(fields.get(name, "") or "")
        try:
            return float(_first_available(msg, [custom, *candidates]))
        except Exception:
            if default is not None:
                return default
            raise
    result = {
        "x": value("x", ["pose.pose.position.x", "pose.position.x", "position.x", "x"]),
        "y": value("y", ["pose.pose.position.y", "pose.position.y", "position.y", "y"]),
        "z": value("z", ["pose.pose.position.z", "pose.position.z", "position.z", "z"], 0.0),
        "qx": value("qx", ["pose.pose.orientation.x", "pose.orientation.x", "orientation.x"], 0.0),
        "qy": value("qy", ["pose.pose.orientation.y", "pose.orientation.y", "orientation.y"], 0.0),
        "qz": value("qz", ["pose.pose.orientation.z", "pose.orientation.z", "orientation.z"], 0.0),
        "qw": value("qw", ["pose.pose.orientation.w", "pose.orientation.w", "orientation.w"], 1.0),
    }
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("VIO pose contains NaN/Inf")
    return result

def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

def latlon_to_local_xy(longitudes: np.ndarray, latitudes: np.ndarray):
    if len(longitudes) == 0:
        return longitudes.copy(), latitudes.copy()
    earth_radius = 6_378_137.0
    lon0, lat0 = math.radians(float(longitudes[0])), math.radians(float(latitudes[0]))
    lons, lats = np.radians(longitudes.astype(float)), np.radians(latitudes.astype(float))
    return earth_radius * (lons - lon0) * math.cos(lat0), earth_radius * (lats - lat0)
