from __future__ import annotations

from .build_bev import build_local_bev_context
from .describe_segment import describe_local_segment
from .match_pose import match_pose_to_route
from .progress import get_route_progress
from .schema import LocalBEVContext, RouteContext, RouteState, local_bev_context_to_dict, route_context_to_dict, route_state_to_dict


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


def build_route_state(
    *,
    gps: dict[str, float],
    heading_deg: float | None,
    route_polyline: list[dict] | list[list[float]],
    coarse_waypoints: list[dict],
    lookahead_m: float = 20.0,
    off_route_threshold_m: float = 20.0,
    front_extent_m: float = 30.0,
    back_extent_m: float = 10.0,
    side_extent_m: float = 15.0,
) -> RouteState:
    route_context = build_route_context(
        gps=gps,
        heading_deg=heading_deg,
        route_polyline=route_polyline,
        coarse_waypoints=coarse_waypoints,
        lookahead_m=lookahead_m,
        off_route_threshold_m=off_route_threshold_m,
    )
    local_bev_context = build_local_bev_context(
        gps=gps,
        heading_deg=heading_deg,
        route_context=route_context,
        route_polyline=route_polyline,
        coarse_waypoints=coarse_waypoints,
        front_extent_m=front_extent_m,
        back_extent_m=back_extent_m,
        side_extent_m=side_extent_m,
    )
    return RouteState(route_context=route_context, local_bev_context=local_bev_context)


__all__ = [
    "LocalBEVContext",
    "RouteContext",
    "RouteState",
    "build_local_bev_context",
    "build_route_context",
    "build_route_state",
    "describe_local_segment",
    "get_route_progress",
    "local_bev_context_to_dict",
    "match_pose_to_route",
    "route_context_to_dict",
    "route_state_to_dict",
]
