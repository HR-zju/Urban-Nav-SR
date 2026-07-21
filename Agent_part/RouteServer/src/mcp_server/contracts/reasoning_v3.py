"""Canonical contracts for the UrbanNav Reasoning V3.3 Full-CoT pipeline.

V3.3 keeps six non-overlapping reasoning sections. Input evidence bookkeeping
stays in the frozen source while generated sections contain only task semantics,
grounded scene facts, local affordances, optional navigation context, grounded
interaction/rule relations, and the final symbolic decision.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mcp_server.contracts.reasoning import (
    ConfidenceLevel,
    EvidenceRef,
    FallbackAction,
    MotionPrimitive,
    ReasonCode,
    SocialAction,
    SpeedPolicy,
    UserRequestV2,
)


REASONING_TRACE_V3_SCHEMA_VERSION = "urbannav_reasoning_trace_v3_3"
FINISH_V3_SCHEMA_VERSION = "urbannav_finish_v3_3"
REASONING_V3_SECTION_NAMES = (
    "TaskGrounding",
    "SceneGrounding",
    "LocalTraversability",
    "NavigationContextGrounding",
    "InteractionAndRuleGrounding",
    "Decision",
)

LocalObjectiveV3 = Literal[
    "follow_local_path",
    "follow_instruction",
    "continue_route",
    "track_waypoint",
    "approach_goal",
    "enter_crosswalk",
    "exit_crosswalk",
    "hold_position",
    "stop_at_goal",
    "unknown",
]
InstructionTypeV3 = Literal[
    "generic_policy",
    "directional",
    "goal_description",
    "route_following",
    "accessibility",
    "compound",
    "unknown",
]
CaptionRoleV3 = Literal["instruction", "scene_description", "mixed", "not_caption", "unknown"]
RequirementCodeV3 = Literal[
    "avoid_stairs",
    "use_curb_cut",
    "accessible_route_required",
    "use_designated_crossing",
    "avoid_vehicle_lane",
    "maintain_clearance",
    "slow_near_people",
    "other",
]
RequirementPolarityV3 = Literal["required", "preferred", "avoid", "forbidden"]
ReferentResolutionV3 = Literal["resolved", "unresolved", "ambiguous"]
AmbiguityTypeV3 = Literal[
    "instruction_scope",
    "referent",
    "route_choice",
    "accessibility_requirement",
    "caption_role",
    "other",
]
DecisionRationaleStatusV3 = Literal["supported", "partial", "unsupported", "not_applicable"]
DecisionReasonV3 = Literal[
    "explicit_user_stop",
    "authoritative_goal_reached",
    "direct_corridor_available",
    "navigation_context_aligned",
    "navigation_context_blocked",
    "local_path_blocked",
    "local_risk_requires_slow",
    "interaction_requires_slow",
    "interaction_requires_yield",
    "interaction_requires_wait",
    "traffic_control_requires_action",
    "accessibility_requirement_applies",
    "insufficient_visual_support",
]
DesiredDirectionV3 = Literal["left", "straight", "right", "stop", "unknown"]
ContextSourceTypeV3 = Literal["instruction", "bev", "goal", "waypoint", "route", "tool"]
ContextRoleV3 = Literal[
    "direction_command",
    "route_topology",
    "goal_context",
    "waypoint_sequence",
    "tool_route",
    "context_only",
    "unknown",
]
ContextUsageV3 = Literal["used", "ignored"]
ContextStatusV3 = Literal[
    "aligned",
    "locally_blocked",
    "conflict",
    "not_visible",
    "insufficient_evidence",
    "context_only",
    "ignored",
]
FusedContextStatusV3 = Literal[
    "not_applicable",
    "clear",
    "locally_blocked",
    "conflict",
    "not_visible",
    "insufficient_evidence",
    "context_only",
]
ContextConsistencyV3 = Literal[
    "not_applicable",
    "single_source",
    "consistent",
    "partial",
    "conflict",
    "unknown",
]
CrossModalRelationTypeV3 = Literal[
    "route_to_corridor",
    "goal_to_corridor",
    "waypoint_to_corridor",
    "tool_route_to_corridor",
    "same_place_context",
    "context_only",
    "unknown",
]
CrossModalStatusV3 = Literal[
    "aligned",
    "partial",
    "conflict",
    "not_visible",
    "insufficient_evidence",
    "context_only",
]
LabelStrengthV3 = Literal["strong", "weak", "negative", "disabled"]
ReliabilityV3 = Literal["high", "medium", "low", "unknown"]
VisibilityV3 = Literal["clear", "partial_occlusion", "low_light", "crowded", "uncertain", "unknown"]
RiskLevelV3 = Literal["low", "medium", "high", "unknown"]
TraversabilityStateV3 = Literal["direct", "conditional", "blocked", "forbidden"]
BlockSeverityV3 = Literal["partial", "full", "unknown"]
SurfaceTypeV3 = Literal[
    "sidewalk",
    "pedestrian_path",
    "crosswalk",
    "curb_cut",
    "plaza",
    "stairs",
    "ramp",
    "narrow_gap",
    "temporary_wait_zone",
    "vehicle_lane",
    "unknown",
]
SceneEntityTypeV3 = Literal[
    "pedestrian",
    "cyclist",
    "mobility_device_user",
    "vehicle",
    "animal",
    "static_obstacle",
    "construction",
    "unknown",
]
LateralPositionV3 = Literal["left", "center", "right", "spanning", "unknown"]
DepthPositionV3 = Literal["near", "mid", "far", "unknown"]
MotionStateV3 = Literal[
    "stationary",
    "apparent_moving",
    "crossing_path",
    "same_direction",
    "approaching",
    "receding",
    "unknown",
]
TrafficControlTypeV3 = Literal[
    "pedestrian_signal",
    "traffic_light",
    "crosswalk",
    "stop_line",
    "sign",
    "barrier",
    "unknown",
]
ControlStateV3 = Literal["go", "stop", "warning", "uncontrolled", "occluded", "unknown"]
InteractionRelationV3 = Literal[
    "occupies_path",
    "crossing_path",
    "approaching_path",
    "ahead_same_direction",
    "close_pass_risk",
    "waiting_near_path",
    "no_path_conflict",
]
TemporalBasisV3 = Literal["single_frame", "multi_frame", "tracked_metadata", "unknown"]
RequiredBehaviorV3 = Literal[
    "slow",
    "yield",
    "wait",
    "follow",
    "pass_left",
    "pass_right",
    "detour",
    "stop",
]
TrafficRuleV3 = Literal[
    "pedestrian_priority",
    "obey_pedestrian_signal",
    "use_designated_crossing",
    "avoid_vehicle_lane",
    "maintain_clearance",
    "accessible_route_required",
    "unknown",
]


class GroundedFactV3(BaseModel):
    """Represent one controlled fact supported by model-visible evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fact_code: ReasonCode
    evidence_refs: list[EvidenceRef] = Field(min_length=1, max_length=8)


class TaskRequirementV3(BaseModel):
    """Represent one instruction-supported user or accessibility requirement.

    Args:
        code: Controlled requirement category.
        polarity: Whether the phrase requires, prefers, avoids, or forbids it.
        text_span: Exact supporting span copied from the active instruction.
        target_ref: Optional resolved scene or navigation target.
        evidence_refs: Input-visible references supporting the extraction.
        confidence: Confidence in the semantic extraction.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: RequirementCodeV3
    polarity: RequirementPolarityV3
    text_span: str = Field(min_length=1, max_length=320)
    target_ref: EvidenceRef | None = None
    evidence_refs: list[EvidenceRef] = Field(min_length=1, max_length=8)
    confidence: ConfidenceLevel = "unknown"


class TaskReferentV3(BaseModel):
    """Ground one instruction phrase to a visible or navigation-context referent.

    Args:
        phrase: Exact or minimally normalized referring phrase.
        resolution: Whether the phrase resolves uniquely.
        resolved_ref: Resolved evidence or generated local reference.
        evidence_refs: Instruction and optional visual/context evidence.
        confidence: Confidence in the grounding relation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    phrase: str = Field(min_length=1, max_length=320)
    resolution: ReferentResolutionV3
    resolved_ref: EvidenceRef | None = None
    evidence_refs: list[EvidenceRef] = Field(min_length=1, max_length=12)
    confidence: ConfidenceLevel = "unknown"

    @model_validator(mode="after")
    def validate_resolution(self) -> "TaskReferentV3":
        """Validate resolved and unresolved referent payloads.

        Returns:
            Validated referent grounding.

        Raises:
            ValueError: If resolution and ``resolved_ref`` disagree.
        """

        if self.resolution == "resolved" and self.resolved_ref is None:
            raise ValueError("resolved task referent requires resolved_ref")
        if self.resolution != "resolved" and self.resolved_ref is not None:
            raise ValueError("unresolved task referent cannot carry resolved_ref")
        return self


class TaskAmbiguityV3(BaseModel):
    """Record one material task ambiguity without guessing its resolution.

    Args:
        type: Controlled ambiguity category.
        text_span: Optional ambiguous instruction span.
        evidence_refs: Input references demonstrating the ambiguity.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: AmbiguityTypeV3
    text_span: str | None = Field(default=None, min_length=1, max_length=320)
    evidence_refs: list[EvidenceRef] = Field(min_length=1, max_length=8)


class TaskGroundingV3(BaseModel):
    """Represent the requested task without describing the current scene.

    Args:
        applicable: Whether a valid local navigation task is active.
        instruction_ref: Reference to the active instruction.
        instruction_type: Semantic type of the instruction.
        caption_role: Whether an EgoWalk-style caption is an instruction, scene
            description, mixed text, or not applicable.
        goal_ref: Optional active goal reference.
        active_navigation_refs: Optional route, BEV, waypoint, or tool sources.
        local_objective: Current short-horizon task phase.
        objective_evidence_refs: Evidence supporting the semantic objective.
        requirements: Grounded user or accessibility requirements.
        referents: Instruction phrases grounded to scene or navigation refs.
        ambiguities: Material unresolved task ambiguities.
        confidence: Confidence in the LVLM-authored task semantics.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    applicable: bool
    instruction_ref: EvidenceRef | None = None
    instruction_type: InstructionTypeV3 = "unknown"
    caption_role: CaptionRoleV3 = "unknown"
    goal_ref: EvidenceRef | None = None
    active_navigation_refs: list[EvidenceRef] = Field(default_factory=list, max_length=12)
    local_objective: LocalObjectiveV3 = "unknown"
    objective_evidence_refs: list[EvidenceRef] = Field(default_factory=list, max_length=12)
    requirements: list[TaskRequirementV3] = Field(default_factory=list, max_length=8)
    referents: list[TaskReferentV3] = Field(default_factory=list, max_length=12)
    ambiguities: list[TaskAmbiguityV3] = Field(default_factory=list, max_length=8)
    confidence: ConfidenceLevel = "unknown"

    @model_validator(mode="after")
    def validate_applicability(self) -> "TaskGroundingV3":
        """Validate task applicability and task-only fields.

        Returns:
            Validated task grounding.

        Raises:
            ValueError: If a disabled task carries active semantics.
        """

        if not self.applicable:
            if (
                self.goal_ref
                or self.active_navigation_refs
                or self.objective_evidence_refs
                or self.requirements
                or self.referents
                or self.ambiguities
            ):
                raise ValueError("non-applicable task grounding cannot contain active task claims")
            if (
                self.local_objective != "unknown"
                or self.instruction_type != "unknown"
            ):
                raise ValueError("non-applicable task grounding must keep task semantics unknown")
            if self.caption_role not in {"scene_description", "unknown"}:
                raise ValueError("non-applicable task grounding permits only scene-description caption role")
            if self.caption_role == "scene_description" and self.instruction_ref is None:
                raise ValueError("scene-description caption role requires instruction_ref")
        elif self.instruction_ref is None:
            raise ValueError("applicable task grounding requires instruction_ref")
        elif self.local_objective == "unknown" and not self.ambiguities:
            raise ValueError("unknown task objective requires an explicit ambiguity")
        return self


class SceneEntityV3(BaseModel):
    """Represent one FPV-grounded actor or obstacle with local geometry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_ref: str = Field(pattern=r"^entity\.[a-z0-9_]+$")
    type: SceneEntityTypeV3
    lateral_position: LateralPositionV3 = "unknown"
    depth_position: DepthPositionV3 = "unknown"
    motion_state: MotionStateV3 = "unknown"
    evidence_refs: list[EvidenceRef] = Field(min_length=1, max_length=12)
    confidence: ConfidenceLevel = "unknown"


class TrafficControlV3(BaseModel):
    """Represent one visible traffic-control or crossing element."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    control_ref: str = Field(pattern=r"^control\.[a-z0-9_]+$")
    type: TrafficControlTypeV3
    state: ControlStateV3 = "unknown"
    lateral_position: LateralPositionV3 = "unknown"
    evidence_refs: list[EvidenceRef] = Field(min_length=1, max_length=12)
    confidence: ConfidenceLevel = "unknown"


class ContextObservationV3(BaseModel):
    """Account for one optional non-FPV context input without forcing a route claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_ref: EvidenceRef
    role: ContextRoleV3
    usage: ContextUsageV3
    facts: list[ReasonCode] = Field(default_factory=list, max_length=8)
    ignored_reason: ReasonCode | None = None
    evidence_refs: list[EvidenceRef] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def validate_usage(self) -> "ContextObservationV3":
        """Validate explicit used-versus-ignored context accounting.

        Returns:
            Validated context observation.

        Raises:
            ValueError: If usage and ignored reason conflict.
        """

        if self.source_ref not in self.evidence_refs:
            raise ValueError("context observation must cite its source_ref")
        if self.usage == "ignored" and self.ignored_reason is None:
            raise ValueError("ignored context requires ignored_reason")
        if self.usage == "used" and self.ignored_reason is not None:
            raise ValueError("used context cannot carry ignored_reason")
        return self


class CrossModalRelationV3(BaseModel):
    """Represent one optional-context to FPV semantic relation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_ref: EvidenceRef
    relation: CrossModalRelationTypeV3
    target_ref: EvidenceRef | None = None
    status: CrossModalStatusV3
    evidence_refs: list[EvidenceRef] = Field(min_length=1, max_length=16)
    confidence: ConfidenceLevel = "unknown"

    @model_validator(mode="after")
    def validate_relation(self) -> "CrossModalRelationV3":
        """Validate positive cross-modal relations.

        Returns:
            Validated relation.

        Raises:
            ValueError: If a positive relation omits source or target evidence.
        """

        refs = set(self.evidence_refs)
        if self.source_ref not in refs:
            raise ValueError("cross-modal relation must cite its source_ref")
        if self.status in {"aligned", "partial", "conflict"}:
            if self.target_ref is None or self.target_ref not in refs:
                raise ValueError("resolved cross-modal relation requires a cited target_ref")
        elif self.target_ref is not None:
            raise ValueError("unresolved cross-modal relation cannot select a target_ref")
        return self


class SceneGroundingV3(BaseModel):
    """Collect model-visible scene facts without selecting an action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    applicable: bool
    scene_facts: list[GroundedFactV3] = Field(default_factory=list, max_length=20)
    entities: list[SceneEntityV3] = Field(default_factory=list, max_length=16)
    traffic_controls: list[TrafficControlV3] = Field(default_factory=list, max_length=12)
    context_observations: list[ContextObservationV3] = Field(default_factory=list, max_length=12)
    cross_modal_relations: list[CrossModalRelationV3] = Field(default_factory=list, max_length=12)
    visibility: VisibilityV3 = "unknown"
    confidence: ConfidenceLevel = "unknown"

    @model_validator(mode="after")
    def validate_scene(self) -> "SceneGroundingV3":
        """Validate scene applicability and local reference uniqueness.

        Returns:
            Validated scene grounding.

        Raises:
            ValueError: If disabled scenes carry claims or local ids collide.
        """

        claims = (
            self.scene_facts,
            self.entities,
            self.traffic_controls,
            self.context_observations,
            self.cross_modal_relations,
        )
        if not self.applicable:
            if any(claims) or self.visibility != "unknown" or self.confidence != "unknown":
                raise ValueError("non-applicable scene grounding cannot contain claims")
            return self
        entity_refs = [item.entity_ref for item in self.entities]
        control_refs = [item.control_ref for item in self.traffic_controls]
        context_refs = [item.source_ref for item in self.context_observations]
        if len(entity_refs) != len(set(entity_refs)):
            raise ValueError("scene entity refs must be unique")
        if len(control_refs) != len(set(control_refs)):
            raise ValueError("traffic control refs must be unique")
        if len(context_refs) != len(set(context_refs)):
            raise ValueError("context observations must account for each source once")
        return self


class TraversabilityRegionV3(BaseModel):
    """Classify one FPV region into exactly one local affordance state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    region_ref: EvidenceRef
    surface: SurfaceTypeV3
    state: TraversabilityStateV3
    required_condition: ReasonCode | None = None
    blocker_refs: list[str] = Field(default_factory=list, max_length=8)
    reason: ReasonCode | None = None
    severity: BlockSeverityV3 = "unknown"
    evidence_refs: list[EvidenceRef] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def validate_state(self) -> "TraversabilityRegionV3":
        """Validate state-specific condition, blocker, and reason fields.

        Returns:
            Validated region classification.

        Raises:
            ValueError: If state-specific fields are missing or contradictory.
        """

        if self.region_ref not in self.evidence_refs:
            raise ValueError("traversability region must cite its region_ref")
        if self.state == "direct":
            if self.required_condition is not None or self.blocker_refs or self.reason is not None:
                raise ValueError("direct region cannot carry condition, blockers, or rejection reason")
        elif self.state == "conditional":
            if self.required_condition is None:
                raise ValueError("conditional region requires required_condition")
        elif self.state == "blocked":
            if not self.blocker_refs and self.reason is None:
                raise ValueError("blocked region requires blocker_refs or reason")
        elif self.state == "forbidden" and self.reason is None:
            raise ValueError("forbidden region requires reason")
        return self


class LocalTraversabilityV3(BaseModel):
    """Represent FPV-first semantic geometry independent of route decisions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    applicable: bool
    regions: list[TraversabilityRegionV3] = Field(default_factory=list, max_length=16)
    preferred_clear_corridor_ref: EvidenceRef | None = None
    visibility: VisibilityV3 = "unknown"
    local_risk: RiskLevelV3 = "unknown"
    confidence: ConfidenceLevel = "unknown"

    @model_validator(mode="after")
    def validate_regions(self) -> "LocalTraversabilityV3":
        """Validate unique region classes and executable selection.

        Returns:
            Validated local traversability.

        Raises:
            ValueError: If regions are absent, duplicated, or selection is unsafe.
        """

        if not self.applicable:
            if self.regions or self.preferred_clear_corridor_ref is not None:
                raise ValueError("non-applicable traversability cannot contain regions")
            if self.visibility != "unknown" or self.local_risk != "unknown" or self.confidence != "unknown":
                raise ValueError("non-applicable traversability must keep status unknown")
            return self
        if not self.regions:
            raise ValueError("applicable traversability requires at least one region")
        refs = [item.region_ref for item in self.regions]
        if len(refs) != len(set(refs)):
            raise ValueError("each traversability region must have exactly one state")
        direct = {item.region_ref for item in self.regions if item.state == "direct"}
        if self.preferred_clear_corridor_ref is not None and self.preferred_clear_corridor_ref not in direct:
            raise ValueError("preferred clear corridor must be a direct region")
        return self


class NavigationContextSourceSpecV3(BaseModel):
    """Describe one model-visible navigation source before LVLM grounding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_type: ContextSourceTypeV3
    source_ref: EvidenceRef
    role: ContextRoleV3
    deterministic_direction: DesiredDirectionV3 = "unknown"
    reliability: ReliabilityV3 = "unknown"


class NavigationSourceGroundingV3(BaseModel):
    """Ground one navigation source independently into the current FPV."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_type: ContextSourceTypeV3
    source_ref: EvidenceRef
    directive: DesiredDirectionV3 = "unknown"
    target_corridor_ref: EvidenceRef | None = None
    status: ContextStatusV3
    reliability: ReliabilityV3 = "unknown"
    ignored_reason: ReasonCode | None = None
    evidence_refs: list[EvidenceRef] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def validate_grounding(self) -> "NavigationSourceGroundingV3":
        """Validate one source-to-FPV relation.

        Returns:
            Validated source grounding.

        Raises:
            ValueError: If status, direction, target, or ignored reason conflict.
        """

        if self.source_ref not in self.evidence_refs:
            raise ValueError("navigation source grounding must cite source_ref")
        if self.status in {"aligned", "locally_blocked"}:
            if self.directive == "unknown":
                raise ValueError("resolved navigation source requires known directive")
            if self.directive != "stop" and self.target_corridor_ref is None:
                raise ValueError("non-stop navigation source requires target corridor")
            if self.directive == "stop" and self.target_corridor_ref is not None:
                raise ValueError("stop navigation source cannot select a corridor")
        elif self.target_corridor_ref is not None:
            raise ValueError("unresolved navigation source cannot select a corridor")
        if self.target_corridor_ref is not None and self.target_corridor_ref not in self.evidence_refs:
            raise ValueError("navigation source grounding must cite target corridor")
        if self.status == "ignored" and self.ignored_reason is None:
            raise ValueError("ignored navigation source requires ignored_reason")
        if self.status != "ignored" and self.ignored_reason is not None:
            raise ValueError("non-ignored navigation source cannot carry ignored_reason")
        return self


class NavigationContextGroundingV3(BaseModel):
    """Fuse independently grounded optional navigation sources."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    applicable: bool
    source_groundings: list[NavigationSourceGroundingV3] = Field(default_factory=list, max_length=12)
    fused_direction: DesiredDirectionV3 = "unknown"
    fused_target_corridor_ref: EvidenceRef | None = None
    source_consistency: ContextConsistencyV3 = "not_applicable"
    fused_execution_status: FusedContextStatusV3 = "not_applicable"

    @model_validator(mode="after")
    def validate_context(self) -> "NavigationContextGroundingV3":
        """Validate optional-source coverage and fused semantics.

        Returns:
            Validated navigation context.

        Raises:
            ValueError: If applicability or fused positive state is inconsistent.
        """

        if not self.applicable:
            if self.source_groundings or self.fused_target_corridor_ref is not None:
                raise ValueError("non-applicable navigation context cannot contain source groundings")
            if self.fused_direction != "unknown":
                raise ValueError("non-applicable navigation context must use unknown direction")
            if self.source_consistency != "not_applicable" or self.fused_execution_status != "not_applicable":
                raise ValueError("non-applicable navigation context must use not_applicable statuses")
            return self
        if not self.source_groundings:
            raise ValueError("applicable navigation context requires source_groundings")
        source_refs = [item.source_ref for item in self.source_groundings]
        if len(source_refs) != len(set(source_refs)):
            raise ValueError("navigation context must ground each source exactly once")
        if self.fused_execution_status in {"clear", "locally_blocked"}:
            if self.fused_direction == "unknown":
                raise ValueError("resolved fused context requires known direction")
            if self.fused_direction != "stop" and self.fused_target_corridor_ref is None:
                raise ValueError("resolved non-stop context requires fused target corridor")
            if self.fused_direction == "stop" and self.fused_target_corridor_ref is not None:
                raise ValueError("resolved stop context cannot select a corridor")
        elif self.fused_target_corridor_ref is not None:
            raise ValueError("unresolved fused context cannot select a corridor")
        return self


class InteractionGroundingV3(BaseModel):
    """Represent one actor-to-path interaction and its required behavior."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    actor_ref: str = Field(pattern=r"^entity\.[a-z0-9_]+$")
    relation: InteractionRelationV3
    affected_corridor_ref: EvidenceRef
    temporal_basis: TemporalBasisV3
    required_behavior: RequiredBehaviorV3
    rule: TrafficRuleV3 = "maintain_clearance"
    evidence_refs: list[EvidenceRef] = Field(min_length=2, max_length=16)
    confidence: ConfidenceLevel = "unknown"


class TrafficRuleGroundingV3(BaseModel):
    """Represent one visible traffic-control rule affecting a local corridor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    control_ref: str = Field(pattern=r"^control\.[a-z0-9_]+$")
    rule: TrafficRuleV3
    affected_corridor_ref: EvidenceRef
    required_behavior: RequiredBehaviorV3
    evidence_refs: list[EvidenceRef] = Field(min_length=2, max_length=16)
    confidence: ConfidenceLevel = "unknown"


class InteractionAndRuleGroundingV3(BaseModel):
    """Represent only path-relevant social interactions and traffic rules."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    applicable: bool
    interactions: list[InteractionGroundingV3] = Field(default_factory=list, max_length=8)
    traffic_rules: list[TrafficRuleGroundingV3] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_applicability(self) -> "InteractionAndRuleGroundingV3":
        """Validate that applicability reflects real grounded relations.

        Returns:
            Validated interaction/rule grounding.

        Raises:
            ValueError: If applicability and relation lists disagree.
        """

        has_relations = bool(self.interactions or self.traffic_rules)
        if self.applicable != has_relations:
            raise ValueError("interaction/rule applicability must match grounded relations")
        return self


class DecisionSupportV3(BaseModel):
    """Bind one decision reason to already grounded evidence.

    Args:
        reason: Controlled reason category.
        evidence_refs: References already present in task, scene, affordance,
            navigation-context, or interaction observations.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: DecisionReasonV3
    evidence_refs: list[EvidenceRef] = Field(min_length=1, max_length=12)


class RejectedDecisionAlternativeV3(BaseModel):
    """Explain why one plausible local motion was not selected.

    Args:
        motion: Rejected symbolic motion.
        reason: Evidence-backed reason for rejection.
        evidence_refs: Grounded references supporting rejection.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    motion: MotionPrimitive
    reason: DecisionReasonV3
    evidence_refs: list[EvidenceRef] = Field(min_length=1, max_length=12)


class DecisionV3(BaseModel):
    """Represent the final policy without restating scene observations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    motion: MotionPrimitive
    speed: SpeedPolicy
    social_action: SocialAction
    selected_corridor_ref: EvidenceRef | None = None
    fallback: FallbackAction
    confidence: ConfidenceLevel
    rationale_status: DecisionRationaleStatusV3
    supporting_reasons: list[DecisionSupportV3] = Field(default_factory=list, max_length=8)
    rejected_alternatives: list[RejectedDecisionAlternativeV3] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def validate_rationale(self) -> "DecisionV3":
        """Validate support status and rejected motion alternatives.

        Returns:
            Validated decision.

        Raises:
            ValueError: If rationale status and evidence-bearing relations disagree.
        """

        if self.rationale_status in {"supported", "partial"} and not self.supporting_reasons:
            raise ValueError("supported or partial decision requires supporting_reasons")
        if self.rationale_status in {"unsupported", "not_applicable"} and self.supporting_reasons:
            raise ValueError("unsupported decision cannot carry supporting_reasons")
        if any(item.motion == self.motion for item in self.rejected_alternatives):
            raise ValueError("rejected alternative cannot repeat selected motion")
        return self


class ReasoningTraceV3(BaseModel):
    """Represent the canonical six-section Reasoning V3.3 trace."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[REASONING_TRACE_V3_SCHEMA_VERSION] = REASONING_TRACE_V3_SCHEMA_VERSION
    task_grounding: TaskGroundingV3
    scene_grounding: SceneGroundingV3
    local_traversability: LocalTraversabilityV3
    navigation_context_grounding: NavigationContextGroundingV3
    interaction_and_rule_grounding: InteractionAndRuleGroundingV3
    decision: DecisionV3


class FinishV3(BaseModel):
    """Represent the deterministic compact projection of ``DecisionV3``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[FINISH_V3_SCHEMA_VERSION] = FINISH_V3_SCHEMA_VERSION
    decision_type: Literal["navigate", "ask_user", "complete_task", "abort_task"]
    motion: MotionPrimitive | None = None
    speed: SpeedPolicy | None = None
    social_action: SocialAction | None = None
    confidence_level: ConfidenceLevel | None = None
    trajectory_type: str | None = Field(default=None, min_length=1, max_length=64)
    route_action: str | None = Field(default=None, min_length=1, max_length=96)
    tool_used: bool | None = None
    reason_code: ReasonCode | None = None
    reason: str | None = Field(default=None, max_length=512)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list, max_length=12)
    user_request: UserRequestV2 | None = None
    retryable: bool | None = None

    @model_validator(mode="after")
    def validate_decision_payload(self) -> "FinishV3":
        """Validate decision-specific finish payload fields.

        Returns:
            Validated finish projection.

        Raises:
            ValueError: If navigation and non-navigation payloads mix.
        """

        navigation = (self.motion, self.speed, self.social_action, self.confidence_level)
        if self.decision_type == "navigate":
            if any(value is None for value in navigation):
                raise ValueError("navigate finish requires motion, speed, social_action, and confidence_level")
            if self.user_request is not None or self.reason_code is not None:
                raise ValueError("navigate finish cannot contain interaction or terminal fields")
        elif any(value is not None for value in navigation) or self.trajectory_type is not None:
            raise ValueError("non-navigation finish cannot contain navigation fields")
        if (self.decision_type == "ask_user") != (self.user_request is not None):
            raise ValueError("ask_user finish requires user_request and other decisions forbid it")
        if self.decision_type in {"complete_task", "abort_task"} and not self.reason_code:
            raise ValueError("terminal finish requires reason_code")
        if self.decision_type not in {"complete_task", "abort_task"} and self.reason_code is not None:
            raise ValueError("non-terminal finish cannot contain reason_code")
        return self


class ReasoningTargetV3(BaseModel):
    """Pair a V3.3 reasoning trace with its deterministic finish projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reasoning: ReasoningTraceV3
    finish: FinishV3

    @model_validator(mode="after")
    def validate_navigation_consistency(self) -> "ReasoningTargetV3":
        """Validate Decision, local affordance, context, interaction, and finish.

        Returns:
            Validated Full-CoT target.

        Raises:
            ValueError: If execution semantics conflict across sections.
        """

        if self.finish.decision_type != "navigate":
            return self
        trace = self.reasoning
        decision = trace.decision
        if (
            decision.motion != self.finish.motion
            or decision.speed != self.finish.speed
            or decision.social_action != self.finish.social_action
            or decision.confidence != self.finish.confidence_level
        ):
            raise ValueError("navigate finish must be derived from Decision")
        rationale_supported = decision.rationale_status in {"supported", "partial"}
        direct = {item.region_ref for item in trace.local_traversability.regions if item.state == "direct"}
        unavailable = {
            item.region_ref
            for item in trace.local_traversability.regions
            if item.state in {"conditional", "blocked"}
        }
        if decision.selected_corridor_ref is not None and decision.selected_corridor_ref not in direct:
            raise ValueError("Decision corridor must be directly traversable")
        context = trace.navigation_context_grounding
        if context.fused_execution_status == "clear" and context.fused_direction != "stop":
            if context.fused_target_corridor_ref not in direct:
                raise ValueError("clear navigation context target must be directly traversable")
            if rationale_supported and decision.selected_corridor_ref != context.fused_target_corridor_ref:
                raise ValueError("clear navigation context target must match Decision corridor")
        if context.fused_execution_status == "locally_blocked" and context.fused_direction != "stop":
            if context.fused_target_corridor_ref not in unavailable:
                raise ValueError("blocked navigation context target must be conditional or blocked")
        required = {
            item.required_behavior for item in trace.interaction_and_rule_grounding.interactions
        } | {
            item.required_behavior for item in trace.interaction_and_rule_grounding.traffic_rules
        }
        if (
            rationale_supported
            and {"stop", "wait"} & required
            and decision.motion != "stop"
        ):
            raise ValueError("stop/wait interaction policy requires stop Decision")
        if (
            rationale_supported
            and "yield" in required
            and decision.social_action not in {"yield", "wait"}
        ):
            raise ValueError("yield interaction policy requires yield or wait Decision")
        return self


class LabelProvenanceEntryV3(BaseModel):
    """Record source, strength, generator version, and confidence for a label."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: ReasonCode
    strength: LabelStrengthV3
    generator_version: str = Field(min_length=1, max_length=128)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    details: dict[str, Any] = Field(default_factory=dict)


class SectionLossMaskV3(BaseModel):
    """Represent loss eligibility for each canonical V3.3 section."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_grounding: bool = False
    scene_grounding: bool = False
    local_traversability: bool = False
    navigation_context_grounding: bool = False
    interaction_and_rule_grounding: bool = False
    decision: bool = False


def reasoning_evidence_refs_v3(trace: ReasoningTraceV3) -> set[str]:
    """Collect input-visible evidence references cited by a V3.3 trace.

    Args:
        trace: Canonical six-section reasoning trace.

    Returns:
        Unique input-visible references. Generated entity/control ids are not
        included unless they appear incorrectly in an evidence list.
    """

    refs: set[str] = set()
    task = trace.task_grounding
    if task.instruction_ref:
        refs.add(task.instruction_ref)
    if task.goal_ref:
        refs.add(task.goal_ref)
    refs.update(task.active_navigation_refs)
    refs.update(task.objective_evidence_refs)
    for requirement in task.requirements:
        if requirement.target_ref:
            refs.add(requirement.target_ref)
        refs.update(requirement.evidence_refs)
    for referent in task.referents:
        if referent.resolved_ref:
            refs.add(referent.resolved_ref)
        refs.update(referent.evidence_refs)
    for ambiguity in task.ambiguities:
        refs.update(ambiguity.evidence_refs)
    scene = trace.scene_grounding
    for fact in scene.scene_facts:
        refs.update(fact.evidence_refs)
    for item in (*scene.entities, *scene.traffic_controls):
        refs.update(item.evidence_refs)
    for item in scene.context_observations:
        refs.add(item.source_ref)
        refs.update(item.evidence_refs)
    for item in scene.cross_modal_relations:
        refs.add(item.source_ref)
        if item.target_ref:
            refs.add(item.target_ref)
        refs.update(item.evidence_refs)
    local = trace.local_traversability
    if local.preferred_clear_corridor_ref:
        refs.add(local.preferred_clear_corridor_ref)
    for item in local.regions:
        refs.add(item.region_ref)
        refs.update(item.evidence_refs)
    context = trace.navigation_context_grounding
    if context.fused_target_corridor_ref:
        refs.add(context.fused_target_corridor_ref)
    for item in context.source_groundings:
        refs.add(item.source_ref)
        if item.target_corridor_ref:
            refs.add(item.target_corridor_ref)
        refs.update(item.evidence_refs)
    interactions = trace.interaction_and_rule_grounding
    for item in (*interactions.interactions, *interactions.traffic_rules):
        refs.add(item.affected_corridor_ref)
        refs.update(item.evidence_refs)
    if trace.decision.selected_corridor_ref:
        refs.add(trace.decision.selected_corridor_ref)
    for support in trace.decision.supporting_reasons:
        refs.update(support.evidence_refs)
    for alternative in trace.decision.rejected_alternatives:
        refs.update(alternative.evidence_refs)
    return refs


__all__ = [
    "FINISH_V3_SCHEMA_VERSION",
    "REASONING_TRACE_V3_SCHEMA_VERSION",
    "REASONING_V3_SECTION_NAMES",
    "ContextObservationV3",
    "CrossModalRelationV3",
    "DecisionSupportV3",
    "DecisionV3",
    "FinishV3",
    "GroundedFactV3",
    "InteractionAndRuleGroundingV3",
    "InteractionGroundingV3",
    "LabelProvenanceEntryV3",
    "LocalTraversabilityV3",
    "NavigationContextGroundingV3",
    "NavigationContextSourceSpecV3",
    "NavigationSourceGroundingV3",
    "ReasoningTargetV3",
    "ReasoningTraceV3",
    "RejectedDecisionAlternativeV3",
    "SceneEntityV3",
    "SceneGroundingV3",
    "SectionLossMaskV3",
    "TaskGroundingV3",
    "TaskAmbiguityV3",
    "TaskReferentV3",
    "TaskRequirementV3",
    "TrafficControlV3",
    "TrafficRuleGroundingV3",
    "TraversabilityRegionV3",
    "reasoning_evidence_refs_v3",
]
