from __future__ import annotations

from .common import (
    bearing_deg,
    clamp,
    normalize_route_polyline,
    project_point_to_polyline,
    smallest_heading_delta_deg,
    haversine_m,
)
from .schema import PoseMatch, RoutePoint


def match_pose_to_route(
    *,
    gps: dict[str, float],
    heading_deg: float | None,
    route_polyline: list[dict] | list[list[float]],
    coarse_waypoints: list[dict] | None = None,
    off_route_threshold_m: float = 20.0,
    start_grace_distance_m: float | None = None,
    start_progress_window_m: float | None = None,
) -> PoseMatch:
    del coarse_waypoints  # Reserved for future matching refinements.

    route_points = normalize_route_polyline(route_polyline)
    if not route_points:
        raise ValueError("route_polyline is empty")

    current_point = RoutePoint(lat=float(gps["lat"]), lng=float(gps["lng"]))
    projected = project_point_to_polyline(current_point, route_points)
    matched_route_distance_m = float(projected["route_distance_m"])
    segment_bearing_deg = projected.get("segment_bearing_deg")
    heading_delta = smallest_heading_delta_deg(heading_deg, segment_bearing_deg)

    route_start = route_points[0]
    distance_to_route_start_m = haversine_m(current_point.lat, current_point.lng, route_start.lat, route_start.lng)

    if start_grace_distance_m is None:
        start_grace_distance_m = max(float(off_route_threshold_m) * 4.0, 80.0)
    if start_progress_window_m is None:
        start_progress_window_m = max(float(off_route_threshold_m) * 2.0, 35.0)

    start_phase = (
        distance_to_route_start_m <= float(start_grace_distance_m)
        and matched_route_distance_m <= float(start_progress_window_m)
    )
    start_grace_applied = False
    off_route = float(projected["cross_track_error_m"]) > float(off_route_threshold_m)

    guidance_bearing_deg = segment_bearing_deg
    if off_route and start_phase:
        start_grace_applied = True
        off_route = False
        if distance_to_route_start_m > 2.0:
            guidance_bearing_deg = bearing_deg(
                current_point.lat,
                current_point.lng,
                route_start.lat,
                route_start.lng,
            )
        heading_delta = smallest_heading_delta_deg(heading_deg, guidance_bearing_deg)

    if start_grace_applied:
        error_score = 1.0 - clamp(
            distance_to_route_start_m / max(float(start_grace_distance_m), 1.0),
            0.0,
            1.0,
        )
    else:
        error_score = 1.0 - clamp(
            float(projected["cross_track_error_m"]) / max(float(off_route_threshold_m), 1.0),
            0.0,
            1.0,
        )
    if heading_delta is None:
        route_confidence = error_score
    else:
        heading_score = 1.0 - clamp(float(heading_delta) / 90.0, 0.0, 1.0)
        route_confidence = clamp(0.7 * error_score + 0.3 * heading_score, 0.0, 1.0)

    return PoseMatch(
        matched_point=projected["matched_point"],
        matched_segment_index=int(projected["matched_segment_index"]),
        matched_segment_ratio=round(float(projected["matched_segment_ratio"]), 4),
        matched_route_distance_m=round(matched_route_distance_m, 2),
        cross_track_error_m=round(float(projected["cross_track_error_m"]), 2),
        heading_delta_deg=round(float(heading_delta), 2) if heading_delta is not None else None,
        distance_to_route_start_m=round(distance_to_route_start_m, 2),
        start_phase=start_phase,
        start_grace_applied=start_grace_applied,
        off_route=off_route,
        route_confidence=round(float(route_confidence), 4),
    )
