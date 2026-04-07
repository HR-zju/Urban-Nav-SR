from __future__ import annotations

from .describe_segment import describe_local_segment
from .match_pose import match_pose_to_route
from .progress import get_route_progress
from .schema import RouteContext, route_context_to_dict


def build_route_context(
    *,
    gps: dict[str, float],
    heading_deg: float | None,
    route_polyline: list[dict] | list[list[float]],
    coarse_waypoints: list[dict],
    lookahead_m: float = 20.0,
    off_route_threshold_m: float = 20.0,
) -> RouteContext:
    pose_match = match_pose_to_route(
        gps=gps,
        heading_deg=heading_deg,
        route_polyline=route_polyline,
        coarse_waypoints=coarse_waypoints,
        off_route_threshold_m=off_route_threshold_m,
    )
    progress = get_route_progress(
        pose_match=pose_match,
        route_polyline=route_polyline,
        coarse_waypoints=coarse_waypoints,
    )
    local_semantics, geometry = describe_local_segment(
        gps=gps,
        pose_match=pose_match,
        progress=progress,
        route_polyline=route_polyline,
        coarse_waypoints=coarse_waypoints,
        heading_deg=heading_deg,
        lookahead_m=lookahead_m,
    )
    return RouteContext(
        pose_match=pose_match,
        progress=progress,
        local_semantics=local_semantics,
        geometry=geometry,
    )


__all__ = [
    "RouteContext",
    "build_route_context",
    "describe_local_segment",
    "get_route_progress",
    "match_pose_to_route",
    "route_context_to_dict",
]
