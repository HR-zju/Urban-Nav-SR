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
class PromptProposal:
    proposal_id: str
    proposal_type: str
    frame: str
    sector_center_deg: float
    sector_span_deg: float
    range_m: list[float]
    priority: float
    geometry_source: str
    semantic_tag: str
    rationale: str
    prompt_text: str
    debug: dict[str, Any]


@dataclass
class PromptProposalBundle:
    proposals: list[PromptProposal]


@dataclass
class VisualRegion:
    region_id: str
    region_type: str
    frame: str
    bbox_uv: list[float]
    polygon_uv: list[list[float]]
    score: float
    mask: Any | None
    source: str
    debug: dict[str, Any]


@dataclass
class VisualRegionBundle:
    image_width: int
    image_height: int
    navigability_heatmap: list[list[float]]
    candidate_regions: list[VisualRegion]


@dataclass
class PoVPromptCandidate:
    proposal_id: str
    source_bev_proposal_id: str
    prompt_type: str
    frame: str
    center_uv: list[float] | None
    direction_vec: list[float] | None
    span_px: float | None
    mask: Any | None
    priority: float
    projection_confidence: float
    visual_confidence: float | None
    semantic_tag: str | None
    rationale: str | None
    debug: dict[str, Any]


@dataclass
class PoVPromptCandidateBundle:
    candidates: list[PoVPromptCandidate]


def route_context_to_dict(context: RouteContext) -> dict[str, Any]:
    return asdict(context)


def local_bev_context_to_dict(context: LocalBEVContext) -> dict[str, Any]:
    return asdict(context)


def route_state_to_dict(state: RouteState) -> dict[str, Any]:
    return asdict(state)


def prompt_proposal_bundle_to_dict(bundle: PromptProposalBundle) -> dict[str, Any]:
    return asdict(bundle)


def visual_region_bundle_to_dict(bundle: VisualRegionBundle) -> dict[str, Any]:
    return asdict(bundle)


def pov_prompt_candidate_bundle_to_dict(bundle: PoVPromptCandidateBundle) -> dict[str, Any]:
    return asdict(bundle)
