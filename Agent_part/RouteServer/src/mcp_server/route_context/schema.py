from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RoutePoint:
    lat: float
    lng: float


@dataclass
class BEVPoint:
    x_m: float
    y_m: float


@dataclass
class BEVPose:
    x_m: float
    y_m: float
    yaw_deg: float


@dataclass
class BEVBBox:
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float


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
class LocalBEVContext:
    current_pose_bev: BEVPose
    matched_point_bev: BEVPoint
    route_polyline_local: list[BEVPoint]
    local_segment_bev: list[BEVPoint]
    connector_bev: list[BEVPoint]
    next_turn_point_bev: BEVPoint | None
    next_waypoint_bev: BEVPoint | None
    bev_bbox: BEVBBox
    bev_rotation_deg: float
    corridor_type: str
    turn_type: str
    turn_semantics: str
    corridor_sector_deg: dict[str, float]
    turn_sector_deg: dict[str, float]
    lookahead_m: float
    semantic_bev_layers: dict[str, Any]


@dataclass
class RouteContext:
    pose_match: PoseMatch
    progress: RouteProgress
    local_semantics: LocalSemantics
    geometry: LocalGeometry


@dataclass
class RouteState:
    route_context: RouteContext
    local_bev_context: LocalBEVContext


@dataclass
class VisualPromptDecision:
    selected_candidate_id: str | None
    selected_prompt_type: str | None
    instruction: str
    confidence: float
    reason: str
    reject_all: bool
    fallback_used: bool
    arbiter_source: str
    rendered_prompt: dict[str, Any] | None
    debug: dict[str, Any]


def route_context_to_dict(context: RouteContext) -> dict[str, Any]:
    return asdict(context)


def local_bev_context_to_dict(context: LocalBEVContext) -> dict[str, Any]:
    return asdict(context)


def route_state_to_dict(state: RouteState) -> dict[str, Any]:
    return asdict(state)


def visual_prompt_decision_to_dict(decision: VisualPromptDecision) -> dict[str, Any]:
    return asdict(decision)
