"""Geometry responsibilities for destination recommendation."""
from __future__ import annotations

from typing import Any
import math

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6371000
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return 2 * radius_m * math.asin(math.sqrt(a))


def extract_key_waypoints(route_data: dict[str, Any], max_points: int = 12) -> list[dict[str, Any]]:
    coarse_waypoints = route_data.get("coarse_waypoints", [])
    if coarse_waypoints:
        extracted = []
        for item in coarse_waypoints:
            coordinate = item.get("coordinate", {}) or {}
            longitude = coordinate.get("longitude")
            latitude = coordinate.get("latitude")
            if longitude is None or latitude is None:
                location_text = item.get("location", "")
                try:
                    lon_str, lat_str = str(location_text).split(",", 1)
                    longitude = float(lon_str)
                    latitude = float(lat_str)
                except Exception:
                    continue
            extracted.append(
                {
                    "longitude": longitude,
                    "latitude": latitude,
                    "instruction": item.get("instruction", ""),
                    "road": item.get("road", ""),
                    "action": item.get("action", ""),
                }
            )
        if len(extracted) <= max_points:
            return extracted
        stride = max(1, len(extracted) // max_points)
        return extracted[::stride][:max_points]

    waypoints: list[dict[str, Any]] = []
    routes = route_data.get("routes", [])
    if not routes:
        return waypoints

    route = routes[0]
    for leg in route.get("legs", []):
        for step in leg.get("steps", []):
            maneuver = step.get("maneuver", {})
            location = maneuver.get("location")
            if location and isinstance(location, list) and len(location) == 2:
                waypoints.append(
                    {
                        "longitude": location[0],
                        "latitude": location[1],
                        "instruction": maneuver.get("instruction", ""),
                        "road": step.get("name", ""),
                    }
                )

    if len(waypoints) <= max_points:
        return waypoints

    stride = max(1, len(waypoints) // max_points)
    return waypoints[::stride][:max_points]


def extract_waypoint_sequence(route_data: dict[str, Any], max_points: int = 12) -> list[dict[str, float]]:
    semantic_points = extract_key_waypoints(route_data, max_points=max_points)
    return [
        {
            "longitude": float(item["longitude"]),
            "latitude": float(item["latitude"]),
        }
        for item in semantic_points
        if item.get("longitude") is not None and item.get("latitude") is not None
    ]
