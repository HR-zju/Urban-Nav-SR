"""Typed contracts for language and planner-guided UrbanNav benchmarks.

The public benchmark condition is deliberately split into language, route
guidance, and hidden future trajectory.  This prevents future odometry from
being silently reused as a waypoint prompt.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


LANGUAGE_GOAL_ANNOTATION_SCHEMA_VERSION = "urbannav_language_goal_annotation_v1"
LANGUAGE_GOAL_CANDIDATE_SCHEMA_VERSION = "urbannav_language_goal_candidate_v1"
WAYPOINT_GUIDANCE_SCHEMA_VERSION = "urbannav_waypoint_guidance_v1"
LANGUAGE_WAYPOINT_BUILD_CONFIG_VERSION = "urbannav_language_waypoint_build_config_v1"
NATIVE_PLANNER_ANCHOR_SCHEMA_VERSION = "urbannav_native_planner_anchor_v1"
NATIVE_PLANNER_ROUTE_LOG_SCHEMA_VERSION = "urbannav_native_planner_route_log_v1"

LanguageInstructionTypeV1 = Literal[
    "landmark_object_goal",
    "visible_corridor_goal",
    "crossing_entrance_region",
    "accessibility_constraint",
    "social_rule_constraint",
    "arrival_stop",
]
LanguageAnnotationStatusV1 = Literal["accepted", "rejected", "pending_human"]
BenchmarkArmV1 = Literal[
    "fpv_only",
    "fpv_language",
    "fpv_waypoint",
    "fpv_language_waypoint",
]


class MetricPoint2DV1(BaseModel):
    """One finite 2D point in a named coordinate frame.

    Args:
        x_m: Forward or east-like metric coordinate in the declared frame.
        y_m: Left or north-like metric coordinate in the declared frame.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    x_m: float
    y_m: float

    @model_validator(mode="after")
    def validate_finite(self) -> "MetricPoint2DV1":
        if not math.isfinite(self.x_m) or not math.isfinite(self.y_m):
            raise ValueError("metric point coordinates must be finite")
        return self


class Wgs84CoordinateV1(BaseModel):
    """One WGS84 latitude/longitude coordinate.

    Args:
        latitude_deg: Geodetic latitude in degrees.
        longitude_deg: Geodetic longitude in degrees.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    latitude_deg: float = Field(ge=-90.0, le=90.0)
    longitude_deg: float = Field(ge=-180.0, le=180.0)


class LanguageGoalCandidateV1(BaseModel):
    """Teacher-only goal candidate derived from a trusted future sidecar.

    This object is input to annotation, never a student-visible benchmark
    prompt.  ``candidate_xy_flu_m`` is intentionally not copied into an
    instruction or waypoint arm.

    Args:
        candidate_id: Stable teacher-only candidate identity.
        parent_sample_id: Physical parent shared by all benchmark arms.
        candidate_time_s: Chosen future-sidecar timestamp.
        candidate_xy_flu_m: Hidden candidate location in current ego FLU.
        source_trajectory_sidecar_digest: Exact trusted-sidecar identity.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[LANGUAGE_GOAL_CANDIDATE_SCHEMA_VERSION] = (
        LANGUAGE_GOAL_CANDIDATE_SCHEMA_VERSION
    )
    candidate_id: str = Field(min_length=1, max_length=256)
    parent_sample_id: str = Field(min_length=1, max_length=480)
    source_group_id: str = Field(min_length=1, max_length=480)
    split_group_id: str = Field(min_length=1, max_length=480)
    dataset_source: str = Field(min_length=1, max_length=128)
    candidate_time_s: float = Field(gt=0.0)
    candidate_xy_flu_m: MetricPoint2DV1
    candidate_yaw_rel_rad: float | None = None
    target_frame_id: str | None = Field(default=None, max_length=256)
    source_trajectory_sidecar_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_yaw(self) -> "LanguageGoalCandidateV1":
        if self.candidate_yaw_rel_rad is not None and not math.isfinite(
            self.candidate_yaw_rel_rad
        ):
            raise ValueError("candidate_yaw_rel_rad must be finite")
        return self


class LanguageGoalAnnotationV1(BaseModel):
    """An externally accepted, model-visible language instruction.

    The annotation must bind language to an anchor-visible referent.  The
    builder accepts only ``accepted`` records, while rejected and pending
    records remain useful for annotation reports.

    Args:
        annotation_id: Stable external annotation identity.
        parent_sample_id: Physical parent the language grounds.
        instruction_text: Student-visible natural-language condition.
        target_ref: Anchor-visible evidence reference for the instruction target.
        validator_status: Acceptance state produced by annotation validation.
        provenance: Human or LVLM annotation provenance.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[LANGUAGE_GOAL_ANNOTATION_SCHEMA_VERSION] = (
        LANGUAGE_GOAL_ANNOTATION_SCHEMA_VERSION
    )
    annotation_id: str = Field(min_length=1, max_length=256)
    parent_sample_id: str = Field(min_length=1, max_length=480)
    instruction_text: str | None = Field(default=None, max_length=2048)
    instruction_type: LanguageInstructionTypeV1 | None = None
    target_ref: str | None = Field(default=None, max_length=160)
    target_bbox_anchor_xyxy: tuple[float, float, float, float] | None = None
    target_bbox_target_xyxy: tuple[float, float, float, float] | None = None
    spatial_relation: str | None = Field(default=None, max_length=160)
    target_route_distance_m: float | None = Field(default=None, ge=0.0)
    target_time_offset_s: float | None = Field(default=None, gt=0.0)
    visibility_at_anchor: bool = False
    evidence_refs: tuple[str, ...] = Field(default_factory=tuple)
    teacher_model: str | None = Field(default=None, max_length=160)
    prompt_version: str | None = Field(default=None, max_length=160)
    confidence: Literal["high", "medium", "low"] | None = None
    validator_status: LanguageAnnotationStatusV1
    human_review: Literal["not_required", "accepted", "rejected", "pending"] = (
        "not_required"
    )
    provenance: Literal["lvlm_annotation", "human_annotation"]

    @model_validator(mode="after")
    def validate_accepted_content(self) -> "LanguageGoalAnnotationV1":
        if self.validator_status != "accepted":
            return self
        if not self.instruction_text or not self.instruction_text.strip():
            raise ValueError("accepted language annotation requires instruction_text")
        if self.instruction_type is None:
            raise ValueError("accepted language annotation requires instruction_type")
        if not self.target_ref or not self.visibility_at_anchor:
            raise ValueError("accepted language annotation requires visible target_ref")
        if not self.evidence_refs:
            raise ValueError("accepted language annotation requires evidence_refs")
        for bbox in (self.target_bbox_anchor_xyxy, self.target_bbox_target_xyxy):
            if bbox is not None and not all(math.isfinite(item) for item in bbox):
                raise ValueError("language target bounding boxes must be finite")
        return self


class WaypointGuidanceProvenanceV1(BaseModel):
    """Record why a planner route is legal student-visible context.

    Args:
        planner_id: Stable planner or route-provider identity.
        route_version: Planner route version or request version.
        plan_timestamp_s: Planner creation time in seconds.
        datum: Datum of supplied WGS84 coordinates.
        independent_of_future_target: Required anti-leakage declaration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    planner_id: str = Field(min_length=1, max_length=160)
    route_version: str = Field(min_length=1, max_length=160)
    plan_timestamp_s: float = Field(ge=0.0)
    coordinate_reference_system: Literal["WGS84"] = "WGS84"
    datum: str = Field(min_length=1, max_length=64)
    independent_of_future_target: Literal[True]
    source_kind: Literal["map_planner", "gps_route", "human_route"]


class WaypointGuidanceV1(BaseModel):
    """Independent planner route in metric FLU, before model normalization.

    Args:
        guidance_id: Stable route guidance identity.
        parent_sample_id: Physical parent receiving the route condition.
        route_points_ego_flu_m: Spatially sampled planner points in FLU.
        route_terminal_within_horizon: Whether the real route endpoint is public.
        provenance: Independent planner-route provenance.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[WAYPOINT_GUIDANCE_SCHEMA_VERSION] = (
        WAYPOINT_GUIDANCE_SCHEMA_VERSION
    )
    guidance_id: str = Field(min_length=1, max_length=256)
    parent_sample_id: str = Field(min_length=1, max_length=480)
    current_wgs84: Wgs84CoordinateV1
    current_heading_rad: float
    goal_wgs84: Wgs84CoordinateV1
    planner_polyline_wgs84: list[Wgs84CoordinateV1] = Field(min_length=2)
    route_points_ego_flu_m: list[MetricPoint2DV1] = Field(min_length=1, max_length=64)
    route_heading_rel_rad: list[float] = Field(min_length=1, max_length=64)
    route_arc_length_m: list[float] = Field(min_length=1, max_length=64)
    route_terminal_within_horizon: bool
    route_step_m: float = Field(gt=0.0)
    route_horizon_m: float = Field(gt=0.0)
    route_max_points: int = Field(ge=1, le=32)
    local_goal_distance_m: float = Field(gt=0.0)
    local_goal_ego_flu_m: MetricPoint2DV1
    along_track_m: float | None = None
    cross_track_m: float | None = None
    valid: bool = True
    stale: bool = False
    reroute: bool = False
    provenance: WaypointGuidanceProvenanceV1

    @model_validator(mode="after")
    def validate_guidance(self) -> "WaypointGuidanceV1":
        count = len(self.route_points_ego_flu_m)
        if len(self.route_heading_rel_rad) != count or len(self.route_arc_length_m) != count:
            raise ValueError("waypoint route feature lengths must match")
        if not math.isfinite(self.current_heading_rad):
            raise ValueError("current_heading_rad must be finite")
        prior = -math.inf
        for heading, arc in zip(self.route_heading_rel_rad, self.route_arc_length_m):
            if not math.isfinite(heading) or not math.isfinite(arc) or arc <= prior:
                raise ValueError("route headings must be finite and arc lengths increasing")
            prior = arc
        for value in (self.along_track_m, self.cross_track_m):
            if value is not None and not math.isfinite(value):
                raise ValueError("route track metrics must be finite")
        if not self.valid or self.stale:
            raise ValueError("benchmark waypoint guidance must be valid and non-stale")
        return self


class PlannerRouteRecordWgs84V1(BaseModel):
    """Canonical adapter input for one independent WGS84 planner route.

    Dataset-specific importers only need to map their native field names into
    this record.  RouteServer then shares one FLU conversion and normalization
    path across SideWalk, Foxglove, EgoWalk, or future datasets.

    Args:
        record_id: Stable native-adapter record identity.
        parent_sample_id: Physical parent joined to benchmark source data.
        current_wgs84: Robot position when the plan was produced.
        current_heading_rad: Compass heading, clockwise from north.
        goal_wgs84: Planner destination.
        planner_polyline_wgs84: Independent ordered planner route.
        provenance: Typed anti-leakage planner provenance.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str = Field(min_length=1, max_length=256)
    parent_sample_id: str = Field(min_length=1, max_length=480)
    current_wgs84: Wgs84CoordinateV1
    current_heading_rad: float
    goal_wgs84: Wgs84CoordinateV1
    planner_polyline_wgs84: list[Wgs84CoordinateV1] = Field(min_length=2)
    provenance: WaypointGuidanceProvenanceV1
    along_track_m: float | None = None
    cross_track_m: float | None = None
    reroute: bool = False


class NativePlannerAnchorV1(BaseModel):
    """One timestamped native localization anchor for a planner-route join.

    A localization anchor is not route guidance.  It only records the current
    WGS84 pose that a separately logged planner route must match.  ``available``
    becomes true only when both current position and compass heading are known.

    Args:
        anchor_id: Stable native-adapter output identity.
        parent_sample_id: Canonical physical parent receiving a potential route.
        dataset_source: Dataset identity used to prevent cross-dataset joins.
        source_record_id: Native record identity used by the planner log.
        observation_timestamp_s: Anchor observation time in Unix seconds.
        current_wgs84: Native current GPS position when available.
        current_heading_rad: Compass heading, clockwise from north, when known.
        available: Whether this anchor is eligible for a planner-route join.
        reason: Deterministic availability or rejection reason.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[NATIVE_PLANNER_ANCHOR_SCHEMA_VERSION] = (
        NATIVE_PLANNER_ANCHOR_SCHEMA_VERSION
    )
    anchor_id: str = Field(min_length=1, max_length=256)
    parent_sample_id: str = Field(min_length=1, max_length=480)
    dataset_source: str = Field(min_length=1, max_length=128)
    source_record_id: str = Field(min_length=1, max_length=480)
    observation_timestamp_s: float = Field(ge=0.0)
    current_wgs84: Wgs84CoordinateV1 | None = None
    current_heading_rad: float | None = None
    localization_source: Literal[
        "native_gps_past_bearing",
        "native_gnss_heading",
        "operator_pose",
        "unavailable",
    ] = "unavailable"
    raw_clip_id: str | None = Field(default=None, max_length=480)
    raw_frame_index: int | None = Field(default=None, ge=0)
    available: bool
    reason: str = Field(min_length=1, max_length=160)

    @model_validator(mode="after")
    def validate_availability(self) -> "NativePlannerAnchorV1":
        """Require a finite complete pose whenever planner joining is enabled."""

        if self.current_heading_rad is not None and not math.isfinite(self.current_heading_rad):
            raise ValueError("current_heading_rad must be finite when supplied")
        if self.available:
            if self.current_wgs84 is None or self.current_heading_rad is None:
                raise ValueError("available native planner anchor requires WGS84 and heading")
            if self.localization_source == "unavailable":
                raise ValueError("available native planner anchor requires a localization source")
        return self


class NativePlannerRouteLogV1(BaseModel):
    """A dataset-native planner response before it is joined to an anchor.

    This contract intentionally does not contain trajectory target fields.  A
    producer must write the original planner request identity, version, route
    time, destination, and polyline so an offline builder can reject stale or
    mismatched guidance deterministically.

    Args:
        route_log_id: Stable planner response identity.
        dataset_source: Dataset identity of the observation that requested it.
        source_record_id: Native observation identifier used for the join.
        plan_timestamp_s: Unix time when the planner produced the route.
        current_wgs84: Position supplied to the planner request.
        current_heading_rad: Compass heading supplied to the planner request.
        goal_wgs84: Explicit planner destination.
        planner_polyline_wgs84: Ordered independent route geometry.
        planner_id: Provider identity, never a generic inferred label.
        route_version: Provider route or request version.
        datum: Geodetic datum of the coordinates.
        source_kind: Route provenance class.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[NATIVE_PLANNER_ROUTE_LOG_SCHEMA_VERSION] = (
        NATIVE_PLANNER_ROUTE_LOG_SCHEMA_VERSION
    )
    route_log_id: str = Field(min_length=1, max_length=256)
    dataset_source: str = Field(min_length=1, max_length=128)
    source_record_id: str = Field(min_length=1, max_length=480)
    plan_timestamp_s: float = Field(ge=0.0)
    current_wgs84: Wgs84CoordinateV1
    current_heading_rad: float
    goal_wgs84: Wgs84CoordinateV1
    planner_polyline_wgs84: list[Wgs84CoordinateV1] = Field(min_length=2)
    planner_id: str = Field(min_length=1, max_length=160)
    route_version: str = Field(min_length=1, max_length=160)
    datum: str = Field(min_length=1, max_length=64)
    source_kind: Literal["map_planner", "gps_route", "human_route"]
    independent_of_future_target: Literal[True]
    reroute: bool = False

    @model_validator(mode="after")
    def validate_heading(self) -> "NativePlannerRouteLogV1":
        """Reject non-finite planner request headings before geometry conversion."""

        if not math.isfinite(self.current_heading_rad):
            raise ValueError("planner route current_heading_rad must be finite")
        return self


class NativePlannerAdapterConfigV1(BaseModel):
    """Fail-closed consistency thresholds for native planner-route joins.

    Args:
        max_plan_age_s: Largest allowed age from route creation to observation.
        max_origin_distance_m: Largest GPS mismatch between anchor and planner request.
        max_heading_delta_rad: Largest compass-heading mismatch at the join.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_plan_age_s: float = Field(default=10.0, gt=0.0)
    max_origin_distance_m: float = Field(default=15.0, gt=0.0)
    max_heading_delta_rad: float = Field(default=math.pi / 4.0, gt=0.0, le=math.pi)


class LanguageWaypointBenchmarkBuildConfigV1(BaseModel):
    """Parameterized immutable package policy for all native datasets.

    Args:
        arm: Public condition arm rendered into benchmark inputs.
        include_bev: Require and include one current BEV image for every case.
        required_fpv_frame_count: Exact ordered FPV context length.
        public_route_horizon_m: Fixed route normalization distance.
        require_language_annotation_for_controls: Preserve paired parent sets.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[LANGUAGE_WAYPOINT_BUILD_CONFIG_VERSION] = (
        LANGUAGE_WAYPOINT_BUILD_CONFIG_VERSION
    )
    arm: BenchmarkArmV1
    include_bev: bool = False
    required_fpv_frame_count: int = Field(default=5, ge=1, le=16)
    public_route_horizon_m: float = Field(default=30.0, gt=0.0)
    public_route_step_m: float = Field(default=2.0, gt=0.0)
    public_route_max_points: int = Field(default=8, ge=1, le=32)
    public_local_goal_distance_m: float = Field(default=10.0, gt=0.0)
    teacher_candidate_time_s: float = Field(default=4.0, gt=0.0)
    require_language_annotation_for_controls: bool = True
    allow_missing_images: bool = False
    system_prompt: str = (
        "You are an UrbanNav navigation policy. Use only the visible images, "
        "instruction, and route guidance. Predict a safe local motion plan."
    )


def wgs84_to_current_ego_flu_v1(
    *,
    origin: Wgs84CoordinateV1,
    heading_rad: float,
    point: Wgs84CoordinateV1,
) -> MetricPoint2DV1:
    """Approximate a nearby WGS84 point in the current ego FLU plane.

    North/east offsets are formed on a local tangent plane, then rotated into
    ``x_forward/y_left`` with the current heading measured clockwise from north.
    This is valid for local planner routes, not global geodesic measurement.

    Args:
        origin: Current WGS84 robot coordinate.
        heading_rad: Current compass heading, clockwise from north.
        point: Route coordinate to project.

    Returns:
        Local ``x_forward/y_left`` metric point.

    Raises:
        ValueError: If the heading is non-finite.
    """

    if not math.isfinite(heading_rad):
        raise ValueError("heading_rad must be finite")
    earth_radius_m = 6_378_137.0
    d_lat = math.radians(point.latitude_deg - origin.latitude_deg)
    d_lon = math.radians(point.longitude_deg - origin.longitude_deg)
    north_m = earth_radius_m * d_lat
    east_m = earth_radius_m * math.cos(math.radians(origin.latitude_deg)) * d_lon
    forward_m = north_m * math.cos(heading_rad) + east_m * math.sin(heading_rad)
    left_m = north_m * math.sin(heading_rad) - east_m * math.cos(heading_rad)
    return MetricPoint2DV1(x_m=forward_m, y_m=left_m)


def normalize_waypoint_guidance_for_model_v1(
    guidance: WaypointGuidanceV1,
    *,
    distance_scale_m: float,
) -> list[list[float]]:
    """Return normalized public route features without exposing target odometry.

    Every point contains ``forward/D, left/D, sin(heading), cos(heading),
    arc/D, terminal``.  ``D`` is a fixed public planner policy parameter and
    must never be fitted from the future trajectory target.

    Args:
        guidance: Valid independent planner guidance.
        distance_scale_m: Public fixed spatial normalization scale.

    Returns:
        Ordered six-feature normalized planner-route rows.

    Raises:
        ValueError: If the public scale is non-positive or non-finite.
    """

    if not math.isfinite(distance_scale_m) or distance_scale_m <= 0.0:
        raise ValueError("distance_scale_m must be positive and finite")
    terminal_index = len(guidance.route_points_ego_flu_m) - 1
    features: list[list[float]] = []
    for index, (point, heading, arc) in enumerate(
        zip(
            guidance.route_points_ego_flu_m,
            guidance.route_heading_rel_rad,
            guidance.route_arc_length_m,
        )
    ):
        features.append(
            [
                point.x_m / distance_scale_m,
                point.y_m / distance_scale_m,
                math.sin(heading),
                math.cos(heading),
                arc / distance_scale_m,
                1.0 if guidance.route_terminal_within_horizon and index == terminal_index else 0.0,
            ]
        )
    return features


def build_waypoint_guidance_from_planner_v1(
    *,
    guidance_id: str,
    parent_sample_id: str,
    current_wgs84: Wgs84CoordinateV1,
    current_heading_rad: float,
    goal_wgs84: Wgs84CoordinateV1,
    planner_polyline_wgs84: list[Wgs84CoordinateV1],
    provenance: WaypointGuidanceProvenanceV1,
    route_step_m: float = 2.0,
    route_horizon_m: float = 30.0,
    route_max_points: int = 8,
    local_goal_distance_m: float = 10.0,
    along_track_m: float | None = None,
    cross_track_m: float | None = None,
    reroute: bool = False,
) -> WaypointGuidanceV1:
    """Convert an independent WGS84 planner polyline to public FLU guidance.

    The geometry only comes from the supplied planner polyline.  The future
    trajectory target is intentionally absent from this function, making the
    provenance boundary mechanically visible in the API.

    Args:
        guidance_id: Stable output guidance identity.
        parent_sample_id: Physical parent joined to the source manifest.
        current_wgs84: Current planner origin.
        current_heading_rad: Current compass heading in radians.
        goal_wgs84: Planner destination.
        planner_polyline_wgs84: Ordered independent route geometry.
        provenance: Typed planner provenance with anti-leakage declaration.
        route_step_m: Spatial sampling interval.
        route_horizon_m: Maximum public route distance.
        route_max_points: Maximum published point count.
        local_goal_distance_m: Public local-goal arc distance.

    Returns:
        Valid metric-FLU guidance contract.

    Raises:
        ValueError: If spatial policy values are invalid or polyline degenerates.
    """

    if route_step_m <= 0.0 or route_horizon_m <= 0.0 or local_goal_distance_m <= 0.0:
        raise ValueError("planner route distances must be positive")
    if route_max_points < 1:
        raise ValueError("route_max_points must be positive")
    points = [
        wgs84_to_current_ego_flu_v1(
            origin=current_wgs84,
            heading_rad=current_heading_rad,
            point=coordinate,
        )
        for coordinate in planner_polyline_wgs84
    ]
    resampled, arc_lengths, terminal_within_horizon = _resample_polyline_flu_v1(
        points,
        step_m=route_step_m,
        horizon_m=route_horizon_m,
        max_points=route_max_points,
    )
    headings = _polyline_headings_v1(resampled)
    local_goal = _point_at_arc_length_v1(
        points,
        target_arc_m=local_goal_distance_m,
    )
    return WaypointGuidanceV1(
        guidance_id=guidance_id,
        parent_sample_id=parent_sample_id,
        current_wgs84=current_wgs84,
        current_heading_rad=current_heading_rad,
        goal_wgs84=goal_wgs84,
        planner_polyline_wgs84=planner_polyline_wgs84,
        route_points_ego_flu_m=resampled,
        route_heading_rel_rad=headings,
        route_arc_length_m=arc_lengths,
        route_terminal_within_horizon=terminal_within_horizon,
        route_step_m=route_step_m,
        route_horizon_m=route_horizon_m,
        route_max_points=route_max_points,
        local_goal_distance_m=local_goal_distance_m,
        local_goal_ego_flu_m=local_goal,
        along_track_m=along_track_m,
        cross_track_m=cross_track_m,
        reroute=reroute,
        provenance=provenance,
    )


def build_waypoint_guidance_from_record_v1(
    record: PlannerRouteRecordWgs84V1,
    *,
    route_step_m: float = 2.0,
    route_horizon_m: float = 30.0,
    route_max_points: int = 8,
    local_goal_distance_m: float = 10.0,
) -> WaypointGuidanceV1:
    """Apply the shared planner conversion to one canonical adapter record.

    Args:
        record: Typed WGS84 planner route from a dataset-specific adapter.
        route_step_m: Spatial sampling interval.
        route_horizon_m: Maximum public route distance.
        route_max_points: Maximum published point count.
        local_goal_distance_m: Public local-goal arc distance.

    Returns:
        Valid metric-FLU waypoint guidance.
    """

    return build_waypoint_guidance_from_planner_v1(
        guidance_id=record.record_id,
        parent_sample_id=record.parent_sample_id,
        current_wgs84=record.current_wgs84,
        current_heading_rad=record.current_heading_rad,
        goal_wgs84=record.goal_wgs84,
        planner_polyline_wgs84=record.planner_polyline_wgs84,
        provenance=record.provenance,
        route_step_m=route_step_m,
        route_horizon_m=route_horizon_m,
        route_max_points=route_max_points,
        local_goal_distance_m=local_goal_distance_m,
        along_track_m=record.along_track_m,
        cross_track_m=record.cross_track_m,
        reroute=record.reroute,
    )


def _resample_polyline_flu_v1(
    points: list[MetricPoint2DV1],
    *,
    step_m: float,
    horizon_m: float,
    max_points: int,
) -> tuple[list[MetricPoint2DV1], list[float], bool]:
    """Sample one finite local polyline by cumulative spatial arc length."""

    if len(points) < 2:
        raise ValueError("planner polyline requires at least two coordinates")
    segments: list[tuple[MetricPoint2DV1, MetricPoint2DV1, float]] = []
    total_m = 0.0
    for start, end in zip(points, points[1:]):
        length = math.hypot(end.x_m - start.x_m, end.y_m - start.y_m)
        if length > 1e-6:
            segments.append((start, end, length))
            total_m += length
    if not segments:
        raise ValueError("planner polyline has no non-zero segment")
    last_arc = min(total_m, horizon_m)
    requested = [min(last_arc, step_m * (index + 1)) for index in range(max_points)]
    requested = list(dict.fromkeys(value for value in requested if value > 0.0))
    if not requested:
        requested = [last_arc]
    return (
        [_point_at_arc_length_v1(points, target_arc_m=value) for value in requested],
        requested,
        total_m <= horizon_m,
    )


def _point_at_arc_length_v1(
    points: list[MetricPoint2DV1],
    *,
    target_arc_m: float,
) -> MetricPoint2DV1:
    """Linearly interpolate a local polyline at a non-negative arc length."""

    remaining = max(0.0, target_arc_m)
    last = points[0]
    for start, end in zip(points, points[1:]):
        length = math.hypot(end.x_m - start.x_m, end.y_m - start.y_m)
        if length <= 1e-6:
            continue
        if remaining <= length:
            ratio = remaining / length
            return MetricPoint2DV1(
                x_m=start.x_m + ratio * (end.x_m - start.x_m),
                y_m=start.y_m + ratio * (end.y_m - start.y_m),
            )
        remaining -= length
        last = end
    return last


def _polyline_headings_v1(points: list[MetricPoint2DV1]) -> list[float]:
    """Derive tangent headings in current FLU from public route geometry."""

    headings: list[float] = []
    for index, point in enumerate(points):
        other = points[min(index + 1, len(points) - 1)]
        if index == len(points) - 1 and index:
            other = points[index - 1]
            heading = math.atan2(point.y_m - other.y_m, point.x_m - other.x_m)
        else:
            heading = math.atan2(other.y_m - point.y_m, other.x_m - point.x_m)
        headings.append(heading)
    return headings


__all__ = [
    "BenchmarkArmV1",
    "build_waypoint_guidance_from_planner_v1",
    "LANGUAGE_GOAL_ANNOTATION_SCHEMA_VERSION",
    "LANGUAGE_GOAL_CANDIDATE_SCHEMA_VERSION",
    "LANGUAGE_WAYPOINT_BUILD_CONFIG_VERSION",
    "LanguageGoalAnnotationV1",
    "LanguageGoalCandidateV1",
    "LanguageWaypointBenchmarkBuildConfigV1",
    "MetricPoint2DV1",
    "NATIVE_PLANNER_ANCHOR_SCHEMA_VERSION",
    "NATIVE_PLANNER_ROUTE_LOG_SCHEMA_VERSION",
    "NativePlannerAdapterConfigV1",
    "NativePlannerAnchorV1",
    "NativePlannerRouteLogV1",
    "PlannerRouteRecordWgs84V1",
    "build_waypoint_guidance_from_record_v1",
    "WaypointGuidanceProvenanceV1",
    "WaypointGuidanceV1",
    "Wgs84CoordinateV1",
    "WAYPOINT_GUIDANCE_SCHEMA_VERSION",
    "normalize_waypoint_guidance_for_model_v1",
    "wgs84_to_current_ego_flu_v1",
]
