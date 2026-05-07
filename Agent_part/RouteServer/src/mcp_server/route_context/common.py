"""Shared route-geometry primitives for GPS-to-route grounding.

This module is intentionally small and deterministic: it converts provider
route polylines into typed points, projects noisy GPS poses onto the route, and
computes bearing/distance features used by the visual-prompt pipeline.

Core formulas:

    Haversine distance:
        a = sin(d_lat / 2)^2 + cos(lat_1) * cos(lat_2) * sin(d_lon / 2)^2
        d = 2R * asin(sqrt(a))

    Bearing:
        theta = atan2(sin(d_lon) * cos(lat_2),
                      cos(lat_1) * sin(lat_2)
                      - sin(lat_1) * cos(lat_2) * cos(d_lon))

    Segment projection in local meters:
        t = clamp(dot(P - A, B - A) / ||B - A||^2, 0, 1)
        P_proj = A + t * (B - A)

All public helpers keep coordinates explicit: route points are latitude /
longitude, while projection internals use a local tangent-plane approximation in
meters around the query point.
"""

from __future__ import annotations

import math
from typing import Any

from .schema import RoutePoint


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in meters between two WGS84 points."""
    radius_m = 6371000.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return 2 * radius_m * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return clockwise bearing in degrees, where 0 is north and 90 is east."""
    y = math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2))
    x = (
        math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
        - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(lon2 - lon1))
    )
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def smallest_heading_delta_deg(source_heading: float | None, target_heading: float | None) -> float | None:
    if source_heading is None or target_heading is None:
        return None
    delta = (target_heading - source_heading + 180.0) % 360.0 - 180.0
    return abs(delta)


def signed_heading_delta_deg(source_heading: float | None, target_heading: float | None) -> float | None:
    if source_heading is None or target_heading is None:
        return None
    return (target_heading - source_heading + 180.0) % 360.0 - 180.0


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def normalize_route_polyline(route_polyline: list[Any]) -> list[RoutePoint]:
    normalized: list[RoutePoint] = []
    for item in route_polyline or []:
        lat = None
        lng = None
        if isinstance(item, dict):
            lat = item.get("latitude", item.get("lat"))
            lng = item.get("longitude", item.get("lng", item.get("lon")))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            lng, lat = item[0], item[1]
        if lat is None or lng is None:
            continue
        point = RoutePoint(lat=float(lat), lng=float(lng))
        if normalized and abs(normalized[-1].lat - point.lat) < 1e-9 and abs(normalized[-1].lng - point.lng) < 1e-9:
            continue
        normalized.append(point)
    return normalized


def normalize_coarse_waypoints(coarse_waypoints: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(coarse_waypoints or []):
        if not isinstance(item, dict):
            continue
        coordinate = item.get("coordinate", {}) or {}
        lat = item.get("latitude", coordinate.get("latitude"))
        lng = item.get("longitude", coordinate.get("longitude"))
        if lat is None or lng is None:
            location = item.get("location")
            if isinstance(location, str) and "," in location:
                try:
                    lng_text, lat_text = location.split(",", 1)
                    lng = float(lng_text)
                    lat = float(lat_text)
                except Exception:
                    lat = None
                    lng = None
        normalized.append(
            {
                "index": index,
                "lat": float(lat) if lat is not None else None,
                "lng": float(lng) if lng is not None else None,
                "action": str(item.get("action") or "").strip() or "go_straight",
                "instruction": str(item.get("instruction") or "").strip(),
                "road": str(item.get("road") or "").strip(),
                "distance_m": float(item.get("distance_m") or 0.0),
                "duration_s": float(item.get("duration_s") or 0.0),
            }
        )
    return normalized


def _to_local_xy_m(point: RoutePoint, reference: RoutePoint) -> tuple[float, float]:
    mean_lat = math.radians((point.lat + reference.lat) / 2.0)
    x = math.radians(point.lng - reference.lng) * 6371000.0 * math.cos(mean_lat)
    y = math.radians(point.lat - reference.lat) * 6371000.0
    return x, y


def project_point_to_segment(point: RoutePoint, start: RoutePoint, end: RoutePoint) -> tuple[RoutePoint, float, float]:
    """Project a GPS point onto one route segment.

    The computation uses a local tangent plane centered at ``point`` so the
    projection is stable for short urban-navigation segments:

        t = clamp(-dot(A_local, V) / ||V||^2, 0, 1)

    where A_local is the segment start relative to the query point and V is the
    segment vector. The returned error is the cross-track distance in meters.
    """
    sx, sy = _to_local_xy_m(start, point)
    ex, ey = _to_local_xy_m(end, point)
    vx = ex - sx
    vy = ey - sy
    seg_len_sq = vx * vx + vy * vy
    if seg_len_sq <= 1e-9:
        projected = RoutePoint(lat=start.lat, lng=start.lng)
        return projected, 0.0, math.hypot(sx, sy)

    t = clamp((-(sx * vx + sy * vy)) / seg_len_sq, 0.0, 1.0)
    proj_x = sx + t * vx
    proj_y = sy + t * vy
    proj_lat = point.lat + math.degrees(proj_y / 6371000.0)
    mean_lat = math.radians((proj_lat + point.lat) / 2.0)
    proj_lng = point.lng + math.degrees(proj_x / (6371000.0 * max(1e-6, math.cos(mean_lat))))
    return RoutePoint(lat=proj_lat, lng=proj_lng), t, math.hypot(proj_x, proj_y)


def route_cumulative_distances(route_polyline: list[RoutePoint]) -> list[float]:
    if not route_polyline:
        return []
    cumulative = [0.0]
    for index in range(1, len(route_polyline)):
        cumulative.append(
            cumulative[-1]
            + haversine_m(
                route_polyline[index - 1].lat,
                route_polyline[index - 1].lng,
                route_polyline[index].lat,
                route_polyline[index].lng,
            )
        )
    return cumulative


def project_point_to_polyline(
    point: RoutePoint,
    route_polyline: list[RoutePoint],
) -> dict[str, Any]:
    """Find the nearest route-polyline segment projection for a GPS point."""
    if not route_polyline:
        raise ValueError("route_polyline is empty")
    if len(route_polyline) == 1:
        return {
            "matched_point": route_polyline[0],
            "matched_segment_index": 0,
            "matched_segment_ratio": 0.0,
            "cross_track_error_m": haversine_m(point.lat, point.lng, route_polyline[0].lat, route_polyline[0].lng),
            "route_distance_m": 0.0,
            "segment_bearing_deg": None,
        }

    cumulative = route_cumulative_distances(route_polyline)
    best: dict[str, Any] | None = None
    for index in range(len(route_polyline) - 1):
        start = route_polyline[index]
        end = route_polyline[index + 1]
        matched_point, ratio, error_m = project_point_to_segment(point, start, end)
        segment_length = haversine_m(start.lat, start.lng, end.lat, end.lng)
        route_distance_m = cumulative[index] + ratio * segment_length
        candidate = {
            "matched_point": matched_point,
            "matched_segment_index": index,
            "matched_segment_ratio": ratio,
            "cross_track_error_m": error_m,
            "route_distance_m": route_distance_m,
            "segment_bearing_deg": bearing_deg(start.lat, start.lng, end.lat, end.lng),
        }
        if best is None or candidate["cross_track_error_m"] < best["cross_track_error_m"]:
            best = candidate

    assert best is not None
    return best
