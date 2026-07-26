"""Canonical UrbanNav Reasoning V4 semantic and wire contracts.

Reasoning V4 is a single-turn, shared-input representation for short-horizon
pedestrian navigation.  The five grounding sections independently interpret
the same prompt-visible observation.  Cross-section references are semantic
links, while ``evidence_refs`` always point to raw prompt-visible evidence.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mcp_server.contracts.reasoning import ReasonCode


REASONING_TRACE_V4_SCHEMA_VERSION = "urbannav_reasoning_trace_v4_0"
FINISH_V4_SCHEMA_VERSION = "urbannav_finish_v4_0"
REASONING_TRAINING_VIEW_V4_SCHEMA_VERSION = (
    "urbannav_reasoning_training_view_v4_0"
)
REASONING_V4_WIRE_FORMAT_VERSION = "urbannav_reasoning_v4_wire_v1"
REASONING_V4_SECTION_DEFINITION_VERSION = (
    "urbannav_reasoning_v4_section_definitions_v2"
)
REASONING_V4_EVIDENCE_POLICY_VERSION = (
    "urbannav_reasoning_v4_evidence_policy_v2"
)
REASONING_V4_RENDERER_ID = "render_reasoning_target_v4_v1"
REASONING_V4_PARSER_ID = "parse_reasoning_target_v4_v1"
REASONING_V4_VALIDATOR_ID = "validate_reasoning_target_v4_v2"
REASONING_V4_SECTION_NAMES = (
    "TaskGrounding",
    "SceneGrounding",
    "LocalTraversability",
    "NavigationContextGrounding",
    "InteractionConstraintGrounding",
    "Decision",
)

ExternalEvidenceRefV4 = Annotated[
    str,
    Field(
        min_length=5,
        max_length=160,
        pattern=r"^(instruction|fpv|bev|goal|waypoint)\.[a-z0-9_]+$",
    ),
]
TaskGoalRefV4 = Literal["task.goal"]
RequirementRefV4 = Annotated[
    str,
    Field(min_length=13, max_length=128, pattern=r"^requirement\.[a-z0-9_]+$"),
]
AmbiguityRefV4 = Annotated[
    str,
    Field(min_length=11, max_length=128, pattern=r"^ambiguity\.[a-z0-9_]+$"),
]
EntityRefV4 = Annotated[
    str,
    Field(min_length=8, max_length=128, pattern=r"^entity\.[a-z0-9_]+$"),
]
ControlRefV4 = Annotated[
    str,
    Field(min_length=9, max_length=128, pattern=r"^control\.[a-z0-9_]+$"),
]
RegionRefV4 = Annotated[
    str,
    Field(min_length=8, max_length=128, pattern=r"^region\.[a-z0-9_]+$"),
]
AlignmentRefV4 = Annotated[
    str,
    Field(min_length=11, max_length=128, pattern=r"^alignment\.[a-z0-9_]+$"),
]
ConstraintRefV4 = Annotated[
    str,
    Field(min_length=12, max_length=128, pattern=r"^constraint\.[a-z0-9_]+$"),
]
SemanticRefV4 = Annotated[
    str,
    Field(
        min_length=8,
        max_length=160,
        pattern=(
            r"^(task\.goal|nav\.task_direction|"
            r"requirement\.[a-z0-9_]+|ambiguity\.[a-z0-9_]+|"
            r"entity\.[a-z0-9_]+|control\.[a-z0-9_]+|"
            r"region\.[a-z0-9_]+|alignment\.[a-z0-9_]+|"
            r"constraint\.[a-z0-9_]+)$"
        ),
    ),
]
ConstraintSourceRefV4 = Annotated[
    str,
    Field(
        min_length=5,
        max_length=160,
        pattern=(
            r"^(instruction|fpv|bev|goal|waypoint)\.[a-z0-9_]+$|"
            r"^(task\.goal|requirement\.[a-z0-9_]+|"
            r"entity\.[a-z0-9_]+|control\.[a-z0-9_]+)$"
        ),
    ),
]

ObjectiveV4 = Literal[
    "approach_goal",
    "follow_instruction",
    "follow_route",
    "follow_waypoint",
    "local_navigation",
    "unknown",
]
GoalTypeV4 = Literal[
    "visual_entity",
    "language_goal",
    "pixel_goal",
    "waypoint",
    "route_goal",
    "multimodal_goal",
    "none",
]
GoalSpecificationV4 = Literal["specified", "ambiguous", "underspecified", "absent"]
RequirementPolarityV4 = Literal["required", "preferred", "avoid", "forbidden"]
LateralPositionV4 = Literal["left", "center", "right", "spanning", "unknown"]
DepthPositionV4 = Literal["near", "mid", "far", "unknown"]
TraversabilityStateV4 = Literal["direct", "conditional", "blocked", "forbidden", "unknown"]
ClearanceV4 = Literal["sufficient", "narrow", "insufficient", "unknown"]
GoalAlignmentStatusV4 = Literal[
    "resolved",
    "ambiguous",
    "not_visible",
    "insufficient_evidence",
]
SourceAlignmentStatusV4 = Literal["supports", "conflicts", "not_visible", "insufficient"]
LocalEffectV4 = Literal[
    "approach_goal",
    "continue_route",
    "follow_waypoint",
    "left",
    "straight",
    "right",
    "stop",
    "unknown",
]
DesiredMotionV4 = Literal["left", "straight", "right", "stop", "unknown"]
TaskDirectionStatusV4 = Literal[
    "aligned",
    "conflict",
    "not_visible",
    "insufficient_evidence",
]
ConstraintKindV4 = Literal["social", "traffic", "accessibility", "local_safety"]
ConstraintRelationV4 = Literal[
    "occupies_candidate_region",
    "crosses_candidate_path",
    "approaches_candidate_path",
    "near_candidate_path",
    "narrows_candidate_region",
    "blocks_candidate_region",
    "controls_candidate_region",
    "violates_requirement",
    "supports_requirement",
    "unknown",
]
TemporalBasisV4 = Literal["single_frame", "multi_frame", "static_control", "instruction"]
ConstraintStatusV4 = Literal["applicable", "uncertain"]
ConstraintEffectV4 = Literal[
    "proceed",
    "slow",
    "yield",
    "wait",
    "stop",
    "detour",
    "request_clarification",
]
DecisionTypeV4 = Literal["navigate", "stop", "finish", "request_user", "request_replan"]
STAGE_A_DECISION_TYPES_V4 = ("navigate", "stop")
MotionV4 = Literal["go_straight", "turn_left", "turn_right", "stop"]
SpeedV4 = Literal["normal", "slow", "stop"]
SocialActionV4 = Literal[
    "none",
    "yield",
    "wait",
    "follow",
    "pass_left",
    "pass_right",
    "detour",
    "unknown",
]
FallbackV4 = Literal[
    "continue",
    "slow",
    "stop",
    "request_clarification",
    "request_replan",
    "unknown",
]


class TaskGoalV4(BaseModel):
    """Normalize the goal defined by prompt-visible task sources."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    goal_ref: TaskGoalRefV4 = "task.goal"
    goal_type: GoalTypeV4
    description: str | None = Field(default=None, min_length=1, max_length=320)
    specification: GoalSpecificationV4
    source_refs: list[ExternalEvidenceRefV4] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_goal_specification(self) -> "TaskGoalV4":
        """Keep absence, goal type, and source evidence mutually consistent."""

        if self.specification == "absent" and self.goal_type != "none":
            raise ValueError("absent goal specification requires goal_type=none")
        if self.goal_type == "none" and self.specification != "absent":
            raise ValueError("goal_type=none requires specification=absent")
        if self.specification == "specified" and not self.source_refs:
            raise ValueError("specified goal requires at least one raw source ref")
        return self


class TaskRequirementV4(BaseModel):
    """Represent one explicit task-specific behavioral requirement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_ref: RequirementRefV4
    code: ReasonCode
    polarity: RequirementPolarityV4
    target_ref: SemanticRefV4 | ExternalEvidenceRefV4 | None = None
    evidence_refs: list[ExternalEvidenceRefV4] = Field(min_length=1, max_length=12)


class TaskAmbiguityV4(BaseModel):
    """Represent one task ambiguity that may change the next action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ambiguity_ref: AmbiguityRefV4
    code: ReasonCode
    source_refs: list[ExternalEvidenceRefV4] = Field(min_length=1, max_length=12)


class TaskGroundingV4(BaseModel):
    """Normalize task sources without predicting visual scene facts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    objective: ObjectiveV4
    goal: TaskGoalV4
    requirements: list[TaskRequirementV4] = Field(default_factory=list, max_length=12)
    ambiguities: list[TaskAmbiguityV4] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_local_refs(self) -> "TaskGroundingV4":
        """Require unique task-local requirement and ambiguity references."""

        _require_unique(
            [item.requirement_ref for item in self.requirements],
            "requirement_ref",
        )
        _require_unique(
            [item.ambiguity_ref for item in self.ambiguities],
            "ambiguity_ref",
        )
        return self


class SceneEntityV4(BaseModel):
    """Describe one directly visible navigation-relevant entity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_ref: EntityRefV4
    category: ReasonCode
    lateral_position: LateralPositionV4 = "unknown"
    depth_position: DepthPositionV4 = "unknown"
    evidence_refs: list[ExternalEvidenceRefV4] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def validate_visual_evidence(self) -> "SceneEntityV4":
        """Require raw FPV support for every visual entity."""

        _require_fpv_evidence(self.evidence_refs, "scene entity")
        return self


class TrafficControlV4(BaseModel):
    """Describe one directly visible traffic-control facility and state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    control_ref: ControlRefV4
    category: ReasonCode
    state: ReasonCode = "unknown"
    lateral_position: LateralPositionV4 = "unknown"
    evidence_refs: list[ExternalEvidenceRefV4] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def validate_visual_evidence(self) -> "TrafficControlV4":
        """Require raw FPV support for every visual traffic control."""

        _require_fpv_evidence(self.evidence_refs, "traffic control")
        return self


class SceneGroundingV4(BaseModel):
    """Collect visible scene facts without path or interaction judgments."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entities: list[SceneEntityV4] = Field(default_factory=list, max_length=32)
    traffic_controls: list[TrafficControlV4] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def validate_local_refs(self) -> "SceneGroundingV4":
        """Require unique scene-local entity and control references."""

        _require_unique([item.entity_ref for item in self.entities], "entity_ref")
        _require_unique(
            [item.control_ref for item in self.traffic_controls],
            "control_ref",
        )
        return self


class TraversabilityRegionV4(BaseModel):
    """Describe one visible local region that may be selected for navigation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    region_ref: RegionRefV4
    surface: ReasonCode
    lateral_position: LateralPositionV4 = "unknown"
    state: TraversabilityStateV4
    clearance: ClearanceV4 = "unknown"
    related_entity_refs: list[EntityRefV4] = Field(default_factory=list, max_length=12)
    limiting_factor_codes: list[ReasonCode] = Field(default_factory=list, max_length=12)
    evidence_refs: list[ExternalEvidenceRefV4] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def validate_region(self) -> "TraversabilityRegionV4":
        """Enforce raw FPV support and non-contradictory direct clearance."""

        _require_fpv_evidence(self.evidence_refs, "traversability region")
        if self.state == "direct" and self.clearance == "insufficient":
            raise ValueError("direct region cannot have insufficient clearance")
        if self.state in {"conditional", "blocked", "forbidden"} and not self.limiting_factor_codes:
            raise ValueError(f"{self.state} region requires a limiting factor code")
        return self


class LocalTraversabilityV4(BaseModel):
    """Organize the visible environment into candidate local regions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    regions: list[TraversabilityRegionV4] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def validate_local_refs(self) -> "LocalTraversabilityV4":
        """Require one unique definition for every local region."""

        _require_unique([item.region_ref for item in self.regions], "region_ref")
        return self


class GoalAlignmentV4(BaseModel):
    """Align the normalized task goal with visible targets and regions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    goal_ref: TaskGoalRefV4 = "task.goal"
    status: GoalAlignmentStatusV4
    target_ref: SemanticRefV4 | ExternalEvidenceRefV4 | None = None
    candidate_target_refs: list[SemanticRefV4 | ExternalEvidenceRefV4] = Field(
        default_factory=list,
        max_length=12,
    )
    aligned_region_refs: list[RegionRefV4] = Field(default_factory=list, max_length=12)
    evidence_refs: list[ExternalEvidenceRefV4] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def validate_alignment(self) -> "GoalAlignmentV4":
        """Keep target candidates consistent with the alignment state."""

        if self.status == "resolved" and self.target_ref is None:
            raise ValueError("resolved goal alignment requires target_ref")
        if self.status != "resolved" and self.target_ref is not None:
            raise ValueError("non-resolved goal alignment cannot carry target_ref")
        if self.status == "resolved" and self.candidate_target_refs:
            raise ValueError(
                "resolved goal alignment cannot carry candidate_target_refs"
            )
        if self.status == "ambiguous" and len(
            self.candidate_target_refs
        ) < 2:
            raise ValueError(
                "ambiguous goal alignment requires at least two candidates"
            )
        if self.status in {"not_visible", "insufficient_evidence"} and (
            self.candidate_target_refs
        ):
            raise ValueError(
                f"{self.status} goal alignment cannot carry candidates"
            )
        return self


class NavigationSourceAlignmentV4(BaseModel):
    """Align one prompt-visible navigation source to local regions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    alignment_ref: AlignmentRefV4
    source_ref: ExternalEvidenceRefV4
    local_effect: LocalEffectV4
    aligned_region_refs: list[RegionRefV4] = Field(default_factory=list, max_length=12)
    status: SourceAlignmentStatusV4
    evidence_refs: list[ExternalEvidenceRefV4] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def validate_source_evidence(self) -> "NavigationSourceAlignmentV4":
        """Require each alignment to cite the source it interprets."""

        if self.source_ref not in self.evidence_refs:
            raise ValueError("source alignment must cite source_ref as raw evidence")
        return self


class TaskDirectionV4(BaseModel):
    """Represent the coarse direction requested by task and navigation context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    direction_ref: Literal["nav.task_direction"] = "nav.task_direction"
    desired_motion: DesiredMotionV4
    candidate_region_refs: list[RegionRefV4] = Field(default_factory=list, max_length=12)
    status: TaskDirectionStatusV4
    evidence_refs: list[ExternalEvidenceRefV4] = Field(min_length=1, max_length=16)


class NavigationContextGroundingV4(BaseModel):
    """Ground task and optional navigation sources to the current local scene."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    goal_alignment: GoalAlignmentV4 | None = None
    source_alignments: list[NavigationSourceAlignmentV4] = Field(
        default_factory=list,
        max_length=16,
    )
    task_direction: TaskDirectionV4

    @model_validator(mode="after")
    def validate_local_refs(self) -> "NavigationContextGroundingV4":
        """Require unique source-alignment references."""

        _require_unique(
            [item.alignment_ref for item in self.source_alignments],
            "alignment_ref",
        )
        return self


class InteractionConstraintV4(BaseModel):
    """Convert a visible task, scene, or control relation into a behavior constraint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    constraint_ref: ConstraintRefV4
    constraint_code: ReasonCode
    kind: ConstraintKindV4
    source_refs: list[ConstraintSourceRefV4] = Field(min_length=1, max_length=16)
    applies_to_region_refs: list[RegionRefV4] = Field(min_length=1, max_length=12)
    relation: ConstraintRelationV4
    temporal_basis: TemporalBasisV4
    status: ConstraintStatusV4
    effect: ConstraintEffectV4
    evidence_refs: list[ExternalEvidenceRefV4] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def validate_constraint(self) -> "InteractionConstraintV4":
        """Enforce evidence-family and temporal-basis invariants."""

        dynamic = self.relation in {
            "crosses_candidate_path",
            "approaches_candidate_path",
        }
        fpv_refs = {ref for ref in self.evidence_refs if ref.startswith("fpv.")}
        if dynamic and (self.temporal_basis != "multi_frame" or len(fpv_refs) < 2):
            raise ValueError(
                "dynamic interaction requires temporal_basis=multi_frame "
                "and at least two FPV refs"
            )
        if self.kind == "traffic" and not any(
            ref.startswith("control.") for ref in self.source_refs
        ):
            raise ValueError("traffic constraint requires a control source ref")
        return self


class InteractionConstraintGroundingV4(BaseModel):
    """Collect only constraints that affect current candidate regions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    constraints: list[InteractionConstraintV4] = Field(default_factory=list, max_length=24)

    @model_validator(mode="after")
    def validate_local_refs(self) -> "InteractionConstraintGroundingV4":
        """Require unique constraint references."""

        _require_unique(
            [item.constraint_ref for item in self.constraints],
            "constraint_ref",
        )
        return self


class DecisionV4(BaseModel):
    """Represent one short-horizon semantic navigation decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_type: DecisionTypeV4
    motion: MotionV4
    speed: SpeedV4
    social_action: SocialActionV4
    selected_region_ref: RegionRefV4 | None = None
    fallback: FallbackV4
    support_refs: list[SemanticRefV4] = Field(default_factory=list, max_length=16)
    evidence_refs: list[ExternalEvidenceRefV4] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def validate_decision_shape(self) -> "DecisionV4":
        """Enforce the local shape of navigation and stop decisions."""

        if self.decision_type == "navigate" and self.selected_region_ref is None:
            raise ValueError("navigate decision requires selected_region_ref")
        if self.decision_type == "stop" and (
            self.motion != "stop" or self.speed != "stop"
        ):
            raise ValueError("stop decision requires motion=stop and speed=stop")
        return self


class ReasoningTraceV4(BaseModel):
    """Canonical six-section shared-input Reasoning V4 trace."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[REASONING_TRACE_V4_SCHEMA_VERSION] = REASONING_TRACE_V4_SCHEMA_VERSION
    task_grounding: TaskGroundingV4
    scene_grounding: SceneGroundingV4
    local_traversability: LocalTraversabilityV4
    navigation_context_grounding: NavigationContextGroundingV4
    interaction_constraint_grounding: InteractionConstraintGroundingV4
    decision: DecisionV4


class FinishDirectionV4(BaseModel):
    """Expose the trainable language projection of a strong direction label."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    choice: str = Field(min_length=1, max_length=32)
    label: ReasonCode
    angle_deg: float | None = Field(default=None, ge=-180.0, le=180.0)


class FinishV4(BaseModel):
    """Runtime projection of Decision and optional strong direction supervision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[FINISH_V4_SCHEMA_VERSION] = FINISH_V4_SCHEMA_VERSION
    decision_type: DecisionTypeV4
    motion: MotionV4
    speed: SpeedV4
    social_action: SocialActionV4
    direction: FinishDirectionV4 | None = None


class ReasoningTargetV4(BaseModel):
    """Pair a canonical Reasoning V4 trace with its deterministic finish."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reasoning: ReasoningTraceV4
    finish: FinishV4

    @model_validator(mode="after")
    def validate_finish_projection(self) -> "ReasoningTargetV4":
        """Require Finish to preserve every Decision field it exposes."""

        decision = self.reasoning.decision
        if (
            self.finish.decision_type,
            self.finish.motion,
            self.finish.speed,
            self.finish.social_action,
        ) != (
            decision.decision_type,
            decision.motion,
            decision.speed,
            decision.social_action,
        ):
            raise ValueError("finish must be an exact semantic projection of Decision")
        if self.finish.direction is not None:
            expected_motion = coarse_motion_for_direction_label_v4(
                self.finish.direction.label
            )
            if expected_motion != decision.motion:
                raise ValueError(
                    "finish direction is incompatible with Decision.motion: "
                    f"label={self.finish.direction.label}, motion={decision.motion}"
                )
        return self


def coarse_motion_for_direction_label_v4(label: str) -> MotionV4:
    """Map an approved direction label to the coarse Decision motion.

    Args:
        label: Canonical direction label such as ``left_20`` or ``straight``.

    Returns:
        The corresponding coarse V4 motion.

    Raises:
        ValueError: If the direction label cannot be mapped safely.
    """

    normalized = str(label).strip().lower()
    if normalized == "stop":
        return "stop"
    if normalized == "straight":
        return "go_straight"
    if normalized.startswith("left_"):
        return "turn_left"
    if normalized.startswith("right_"):
        return "turn_right"
    raise ValueError(f"unsupported direction label for coarse motion: {label!r}")


def reasoning_v4_contract_schema_digest() -> str:
    """Return a semantic SHA-256 digest for the canonical V4 target schema."""

    schema = _semantic_contract_schema(ReasoningTargetV4.model_json_schema())
    payload = json.dumps(
        schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def raw_evidence_refs_v4(trace: ReasoningTraceV4) -> set[str]:
    """Collect all raw prompt-visible evidence references in a V4 trace."""

    refs: set[str] = set(trace.task_grounding.goal.source_refs)
    for requirement in trace.task_grounding.requirements:
        refs.update(requirement.evidence_refs)
    for ambiguity in trace.task_grounding.ambiguities:
        refs.update(ambiguity.source_refs)
    for entity in trace.scene_grounding.entities:
        refs.update(entity.evidence_refs)
    for control in trace.scene_grounding.traffic_controls:
        refs.update(control.evidence_refs)
    for region in trace.local_traversability.regions:
        refs.update(region.evidence_refs)
    navigation = trace.navigation_context_grounding
    if navigation.goal_alignment is not None:
        refs.update(navigation.goal_alignment.evidence_refs)
    for alignment in navigation.source_alignments:
        refs.add(alignment.source_ref)
        refs.update(alignment.evidence_refs)
    refs.update(navigation.task_direction.evidence_refs)
    for constraint in trace.interaction_constraint_grounding.constraints:
        refs.update(
            ref
            for ref in constraint.source_refs
            if _is_external_evidence_ref(ref)
        )
        refs.update(constraint.evidence_refs)
    refs.update(trace.decision.evidence_refs)
    return refs


def semantic_refs_v4(trace: ReasoningTraceV4) -> set[str]:
    """Collect cross-section semantic links without treating them as evidence."""

    refs: set[str] = set()
    for region in trace.local_traversability.regions:
        refs.update(region.related_entity_refs)
    navigation = trace.navigation_context_grounding
    if navigation.goal_alignment is not None:
        if (
            navigation.goal_alignment.target_ref is not None
            and not _is_external_evidence_ref(navigation.goal_alignment.target_ref)
        ):
            refs.add(navigation.goal_alignment.target_ref)
        refs.update(
            ref
            for ref in navigation.goal_alignment.candidate_target_refs
            if not _is_external_evidence_ref(ref)
        )
        refs.update(navigation.goal_alignment.aligned_region_refs)
    for alignment in navigation.source_alignments:
        refs.update(alignment.aligned_region_refs)
    refs.update(navigation.task_direction.candidate_region_refs)
    for constraint in trace.interaction_constraint_grounding.constraints:
        refs.update(
            ref
            for ref in constraint.source_refs
            if not _is_external_evidence_ref(ref)
        )
        refs.update(constraint.applies_to_region_refs)
    refs.update(trace.decision.support_refs)
    if trace.decision.selected_region_ref is not None:
        refs.add(trace.decision.selected_region_ref)
    return refs


def _require_unique(values: list[str], field_name: str) -> None:
    """Raise when one sample-local reference is defined more than once."""

    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} values must be unique")


def _require_fpv_evidence(values: list[str], object_name: str) -> None:
    """Require at least one raw FPV frame for a visual claim."""

    if not any(value.startswith("fpv.") for value in values):
        raise ValueError(f"{object_name} requires at least one FPV evidence ref")


def _is_external_evidence_ref(value: str) -> bool:
    """Return whether a value belongs to the Stage-A raw evidence namespace."""

    return str(value).startswith(
        ("instruction.", "fpv.", "bev.", "goal.", "waypoint.")
    )


def _semantic_contract_schema(value: Any) -> Any:
    """Remove documentation-only JSON-Schema keys recursively."""

    if isinstance(value, dict):
        return {
            key: _semantic_contract_schema(item)
            for key, item in sorted(value.items())
            if key not in {"title", "description"}
        }
    if isinstance(value, list):
        return [_semantic_contract_schema(item) for item in value]
    return value


REASONING_V4_CONTRACT_SCHEMA_DIGEST = reasoning_v4_contract_schema_digest()
