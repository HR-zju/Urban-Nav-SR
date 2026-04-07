from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RoutePoint:
    lat: float
    lng: float


@dataclass
class PoseMatch:
    matched_point: RoutePoint
    matched_segment_index: int
    matched_segment_ratio: float
    matched_route_distance_m: float
    cross_track_error_m: float
    heading_delta_deg: float | None
    distance_to_route_start_m: float
    start_phase: bool
    start_grace_applied: bool
    off_route: bool
    route_confidence: float


@dataclass
class RouteProgress:
    current_waypoint_index: int
    next_waypoint_index: int | None
    remaining_distance_m: float
    distance_to_next_waypoint_m: float | None
    distance_to_next_turn_m: float | None
    next_turn_type: str
    progress_ratio: float
    off_route: bool
    route_confidence: float


@dataclass
class LocalSemantics:
    turn_type: str
    turn_semantics: str
    corridor_type: str
    short_action_summary: str
    instruction_hint: str
    road_name: str


@dataclass
class LocalGeometry:
    local_segment_polyline: list[RoutePoint]
    connector_polyline: list[RoutePoint]
    connector_active: bool
    lookahead_m: float


@dataclass
class RouteContext:
    pose_match: PoseMatch
    progress: RouteProgress
    local_semantics: LocalSemantics
    geometry: LocalGeometry


def route_context_to_dict(context: RouteContext) -> dict[str, Any]:
    return asdict(context)
