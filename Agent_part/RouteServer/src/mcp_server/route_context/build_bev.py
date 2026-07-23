"""Build route-aligned bird's-eye-view context for visual prompting.

The BEV context is an intent prior, not a visual observation. It rotates route
geometry into the rider/camera heading frame so downstream reasoning and LVLM
prompt contracts can reason about "front", "left-front", and "right-front"
without mixing geographic and image coordinates.

Coordinate transform:

    east_m  ~= (lng - lng_0) * 111320 * cos(mean_lat)
    north_m ~= (lat - lat_0) * 110540

    x_bev = east_m * cos(theta) - north_m * sin(theta)
    y_bev = east_m * sin(theta) + north_m * cos(theta)

where theta is the chosen BEV rotation heading. The current pose is always
anchored at (0, 0); positive y points toward the route/camera front.
"""

from __future__ import annotations

import math
from typing import Any

from .common import bearing_deg, normalize_coarse_waypoints, normalize_route_polyline
from .schema import BEVBBox, BEVPoint, BEVPose, LocalBEVContext, RouteContext, RoutePoint


def _routepoint_to_bev(
    point: RoutePoint,
    *,
    origin: RoutePoint,
    rotation_heading_deg: float,
) -> BEVPoint:
    """Project one WGS84 point into the local route-aligned BEV frame."""
    mean_lat = (point.lat + origin.lat) / 2.0
    east_m = (point.lng - origin.lng) * 111320.0 * max(0.1, math.cos(math.radians(mean_lat)))
    north_m = (point.lat - origin.lat) * 110540.0

    theta = math.radians(rotation_heading_deg)
    x_m = east_m * math.cos(theta) - north_m * math.sin(theta)
    y_m = east_m * math.sin(theta) + north_m * math.cos(theta)
    return BEVPoint(x_m=round(x_m, 3), y_m=round(y_m, 3))


def _within_bbox(point: BEVPoint, bbox: BEVBBox, margin_m: float = 3.0) -> bool:
    return (
        bbox.x_min_m - margin_m <= point.x_m <= bbox.x_max_m + margin_m
        and bbox.y_min_m - margin_m <= point.y_m <= bbox.y_max_m + margin_m
    )


def _dedupe_bev_points(points: list[BEVPoint]) -> list[BEVPoint]:
    deduped: list[BEVPoint] = []
    for point in points:
        if deduped and abs(deduped[-1].x_m - point.x_m) < 1e-6 and abs(deduped[-1].y_m - point.y_m) < 1e-6:
            continue
        deduped.append(point)
    return deduped


def _corridor_sector(corridor_type: str) -> dict[str, float]:
    mapping = {
        "front": {"center_deg": 0.0, "span_deg": 40.0},
        "slight_left_front": {"center_deg": 30.0, "span_deg": 35.0},
        "left_front": {"center_deg": 65.0, "span_deg": 45.0},
        "slight_right_front": {"center_deg": -30.0, "span_deg": 35.0},
        "right_front": {"center_deg": -65.0, "span_deg": 45.0},
        "uncertain": {"center_deg": 0.0, "span_deg": 140.0},
    }
    return mapping.get(corridor_type, {"center_deg": 0.0, "span_deg": 60.0})


def _turn_sector(turn_type: str) -> dict[str, float]:
    mapping = {
        "turn_left": {"center_deg": 70.0, "span_deg": 50.0},
        "keep_left": {"center_deg": 35.0, "span_deg": 40.0},
        "bear_left": {"center_deg": 35.0, "span_deg": 40.0},
        "turn_right": {"center_deg": -70.0, "span_deg": 50.0},
        "keep_right": {"center_deg": -35.0, "span_deg": 40.0},
        "bear_right": {"center_deg": -35.0, "span_deg": 40.0},
        "u_turn": {"center_deg": 180.0, "span_deg": 50.0},
        "arrive": {"center_deg": 0.0, "span_deg": 30.0},
        "go_straight": {"center_deg": 0.0, "span_deg": 35.0},
    }
    return mapping.get(turn_type, {"center_deg": 0.0, "span_deg": 35.0})


def _rotation_heading_deg(route_context: RouteContext, heading_deg: float | None) -> float:
    if heading_deg is not None:
        return float(heading_deg)
    local_points = route_context.geometry.local_segment_polyline
    if len(local_points) >= 2:
        start = local_points[0]
        for point in local_points[1:]:
            if abs(start.lat - point.lat) > 1e-9 or abs(start.lng - point.lng) > 1e-9:
                return float(bearing_deg(start.lat, start.lng, point.lat, point.lng))
    return 0.0


def _crop_route_polyline_local(
    route_points: list[RoutePoint],
    *,
    origin: RoutePoint,
    bbox: BEVBBox,
    rotation_heading_deg: float,
) -> list[BEVPoint]:
    local_points = [
        _routepoint_to_bev(point, origin=origin, rotation_heading_deg=rotation_heading_deg)
        for point in route_points
    ]
    cropped = [point for point in local_points if _within_bbox(point, bbox)]
    if not cropped:
        return []
    return _dedupe_bev_points(cropped)


def build_local_bev_context(
    *,
    gps: dict[str, float],
    heading_deg: float | None,
    route_context: RouteContext,
    route_polyline: list[dict[str, Any]] | list[list[float]],
    coarse_waypoints: list[dict[str, Any]],
    front_extent_m: float = 30.0,
    back_extent_m: float = 10.0,
    side_extent_m: float = 15.0,
) -> LocalBEVContext:
    """Create the local BEV window consumed by cross-view reasoning and prompt priors.

    The output deliberately keeps the geometric layers separate:
    ``route_polyline_local`` is the cropped provider route, ``local_segment_bev``
    is the immediate lookahead segment, and ``connector_bev`` is only populated
    when start-grace recovery needs to guide the user back to the route start.
    """
    route_points = normalize_route_polyline(route_polyline)
    if not route_points:
        raise ValueError("route_polyline is empty")

    origin = RoutePoint(lat=float(gps["lat"]), lng=float(gps["lng"]))
    rotation_heading_deg = _rotation_heading_deg(route_context, heading_deg)
    bbox = BEVBBox(
        x_min_m=round(-float(side_extent_m), 3),
        x_max_m=round(float(side_extent_m), 3),
        y_min_m=round(-float(back_extent_m), 3),
        y_max_m=round(float(front_extent_m), 3),
    )

    matched_point_bev = _routepoint_to_bev(
        route_context.pose_match.matched_point,
        origin=origin,
        rotation_heading_deg=rotation_heading_deg,
    )
    local_segment_bev = _dedupe_bev_points(
        [
            _routepoint_to_bev(point, origin=origin, rotation_heading_deg=rotation_heading_deg)
            for point in route_context.geometry.local_segment_polyline
        ]
    )
    connector_bev = _dedupe_bev_points(
        [
            _routepoint_to_bev(point, origin=origin, rotation_heading_deg=rotation_heading_deg)
            for point in route_context.geometry.connector_polyline
        ]
    )
    route_polyline_local = _crop_route_polyline_local(
        route_points,
        origin=origin,
        bbox=bbox,
        rotation_heading_deg=rotation_heading_deg,
    )
    if not route_polyline_local:
        route_polyline_local = list(local_segment_bev)

    normalized_waypoints = [
        item for item in normalize_coarse_waypoints(coarse_waypoints)
        if item.get("lat") is not None and item.get("lng") is not None
    ]
    next_waypoint = next(
        (item for item in normalized_waypoints if item["index"] == route_context.progress.next_waypoint_index),
        None,
    )
    next_waypoint_bev = None
    if next_waypoint is not None:
        next_waypoint_bev = _routepoint_to_bev(
            RoutePoint(lat=float(next_waypoint["lat"]), lng=float(next_waypoint["lng"])),
            origin=origin,
            rotation_heading_deg=rotation_heading_deg,
        )

    next_turn_point_bev = next_waypoint_bev

    return LocalBEVContext(
        current_pose_bev=BEVPose(x_m=0.0, y_m=0.0, yaw_deg=0.0),
        matched_point_bev=matched_point_bev,
        route_polyline_local=route_polyline_local,
        local_segment_bev=local_segment_bev,
        connector_bev=connector_bev,
        next_turn_point_bev=next_turn_point_bev,
        next_waypoint_bev=next_waypoint_bev,
        bev_bbox=bbox,
        bev_rotation_deg=round(rotation_heading_deg, 2),
        corridor_type=route_context.local_semantics.corridor_type,
        turn_type=route_context.local_semantics.turn_type,
        turn_semantics=route_context.local_semantics.turn_semantics,
        corridor_sector_deg=_corridor_sector(route_context.local_semantics.corridor_type),
        turn_sector_deg=_turn_sector(route_context.local_semantics.turn_type),
        lookahead_m=route_context.geometry.lookahead_m,
        semantic_bev_layers={
            "lane_or_sidewalk_mask": None,
            "crossing_mask": None,
            "entrance_candidates_bev": [],
        },
    )
