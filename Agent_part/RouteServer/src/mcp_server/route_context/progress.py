from __future__ import annotations

from .common import (
    normalize_coarse_waypoints,
    normalize_route_polyline,
    project_point_to_polyline,
    route_cumulative_distances,
)
from .schema import PoseMatch, RoutePoint, RouteProgress


def get_route_progress(
    *,
    pose_match: PoseMatch,
    route_polyline: list[dict] | list[list[float]],
    coarse_waypoints: list[dict],
) -> RouteProgress:
    route_points = normalize_route_polyline(route_polyline)
    if not route_points:
        raise ValueError("route_polyline is empty")

    route_total_m = route_cumulative_distances(route_points)[-1] if len(route_points) >= 2 else 0.0
    matched_distance_m = float(pose_match.matched_route_distance_m)

    normalized_waypoints = [
        item for item in normalize_coarse_waypoints(coarse_waypoints)
        if item.get("lat") is not None and item.get("lng") is not None
    ]
    waypoint_distances: list[tuple[int, float, dict]] = []
    for item in normalized_waypoints:
        waypoint_point = RoutePoint(lat=float(item["lat"]), lng=float(item["lng"]))
        projected = project_point_to_polyline(waypoint_point, route_points)
        waypoint_distances.append((int(item["index"]), float(projected["route_distance_m"]), item))

    current_waypoint_index = 0
    next_waypoint_index = None
    next_waypoint_distance = None
    next_turn_type = "go_straight"
    next_waypoint = None

    for index, route_distance_m, item in sorted(waypoint_distances, key=lambda x: x[1]):
        if route_distance_m <= matched_distance_m:
            current_waypoint_index = index
            continue
        next_waypoint_index = index
        next_waypoint_distance = max(0.0, route_distance_m - matched_distance_m)
        next_turn_type = str(item.get("action") or "go_straight")
        next_waypoint = item
        break

    remaining_distance_m = max(0.0, route_total_m - matched_distance_m)
    progress_ratio = 0.0 if route_total_m <= 1e-6 else matched_distance_m / route_total_m

    return RouteProgress(
        current_waypoint_index=current_waypoint_index,
        next_waypoint_index=next_waypoint_index,
        remaining_distance_m=round(remaining_distance_m, 2),
        distance_to_next_waypoint_m=round(next_waypoint_distance, 2) if next_waypoint_distance is not None else None,
        distance_to_next_turn_m=round(next_waypoint_distance, 2) if next_waypoint_distance is not None else None,
        next_turn_type=next_turn_type,
        progress_ratio=round(progress_ratio, 4),
        off_route=pose_match.off_route,
        route_confidence=pose_match.route_confidence,
    )
