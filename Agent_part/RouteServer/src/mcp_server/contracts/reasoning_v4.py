"""Canonical UrbanNav Reasoning V4.2 semantic and wire contracts.

V4.2 separates deterministic demonstrated direction, weak navigation intent,
and the immediately selected action. Prompt-visible direction labels define the
shared vocabulary; the deterministic composer derives ``Decision.motion`` only
from ``Decision.selected_action_label``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mcp_server.contracts.reasoning import ReasonCode


REASONING_TRACE_V4_SCHEMA_VERSION = "urbannav_reasoning_trace_v4_2"
FINISH_V4_SCHEMA_VERSION = "urbannav_finish_v4_2"
REASONING_TRAINING_VIEW_V4_SCHEMA_VERSION = (
    "urbannav_reasoning_training_view_v4_2"
)
REASONING_V4_WIRE_FORMAT_VERSION = "urbannav_reasoning_v4_wire_v3"
REASONING_V4_SECTION_DEFINITION_VERSION = (
    "urbannav_reasoning_v4_section_definitions_v4"
)
REASONING_V4_EVIDENCE_POLICY_VERSION = (
    "urbannav_reasoning_v4_evidence_policy_v4"
)
REASONING_V4_RENDERER_ID = "render_reasoning_target_v4_v3"
REASONING_V4_PARSER_ID = "parse_reasoning_target_v4_v3"
REASONING_V4_VALIDATOR_ID = "validate_reasoning_target_v4_v4"
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
    Field(
        min_length=13,
        max_length=128,
        pattern=r"^requirement\.[a-z0-9_]+$",
    ),
]
AmbiguityRefV4 = Annotated[
    str,
    Field(
        min_length=11,
        max_length=128,
        pattern=r"^ambiguity\.[a-z0-9_]+$",
    ),
]
EntityRefV4 = Annotated[
    str,
    Field(min_length=10, max_length=32, pattern=r"^entity\.e[0-9]{2,}$"),
]
ControlRefV4 = Annotated[
    str,
    Field(min_length=11, max_length=32, pattern=r"^control\.c[0-9]{2,}$"),
]
AlignmentRefV4 = Annotated[
    str,
    Field(
        min_length=13,
        max_length=32,
        pattern=r"^alignment\.a[0-9]{2,}$",
    ),
]
ConstraintRefV4 = Annotated[
    str,
    Field(
        min_length=14,
        max_length=32,
        pattern=r"^constraint\.k[0-9]{2,}$",
    ),
]
UncertaintyRefV4 = Annotated[
    str,
    Field(
        min_length=15,
        max_length=32,
        pattern=r"^uncertainty\.u[0-9]{2,}$",
    ),
]
SemanticRefV4 = Annotated[
    str,
    Field(
        min_length=8,
        max_length=160,
        pattern=(
            r"^(task\.goal|nav\.task_direction|"
            r"requirement\.[a-z0-9_]+|ambiguity\.[a-z0-9_]+|"
            r"entity\.e[0-9]{2,}|control\.c[0-9]{2,}|"
            r"alignment\.a[0-9]{2,}|constraint\.k[0-9]{2,}|"
            r"uncertainty\.u[0-9]{2,})$"
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
GoalSpecificationV4 = Literal[
    "specified",
    "ambiguous",
    "underspecified",
    "absent",
]
RequirementPolarityV4 = Literal[
    "required",
    "preferred",
    "avoid",
    "forbidden",
]
LateralPositionV4 = Literal[
    "left",
    "center",
    "right",
    "spanning",
    "unknown",
]
DepthPositionV4 = Literal["near", "mid", "far", "unknown"]
EvidenceStatusV4 = Literal["supported", "ambiguous", "insufficient"]
ObservationQualityV4 = Literal["clear", "partial", "poor", "unknown"]
DirectionAffordanceStateV4 = Literal[
    "direct",
    "conditional",
    "blocked",
    "forbidden",
    "unknown",
]
ClearanceV4 = Literal[
    "sufficient",
    "narrow",
    "insufficient",
    "unknown",
]
GoalAlignmentStatusV4 = Literal[
    "resolved",
    "ambiguous",
    "not_visible",
    "insufficient_evidence",
]
SourceAlignmentStatusV4 = Literal[
    "supports",
    "conflicts",
    "not_visible",
    "insufficient",
]
AlignmentRoleV4 = Literal[
    "task_intent",
    "route_intent",
    "local_safety_bias",
]
TaskDirectionStatusV4 = Literal[
    "aligned",
    "conflict",
    "not_visible",
    "insufficient_evidence",
]
ConstraintKindV4 = Literal[
    "task",
    "social",
    "traffic",
    "accessibility",
    "local_safety",
]
ConstraintRelationV4 = Literal[
    "controls_current_path",
    "occupies_candidate_path",
    "crosses_candidate_path",
    "approaches_candidate_path",
    "near_candidate_path",
    "reduces_clearance",
    "blocks_candidate_path",
    "conflicts_with_requirement",
    "unknown",
]
ConstraintStatusV4 = Literal["applicable", "uncertain"]
ConstraintEffectV4 = Literal[
    "slow",
    "yield",
    "stop",
    "avoid",
]
UncertaintyTypeV4 = Literal[
    "task",
    "perception",
    "temporal",
    "cross_modal",
    "rule_applicability",
]
ResolutionPolicyV4 = Literal[
    "continue_observing",
    "slow",
    "yield",
    "wait",
    "stop",
]
DecisionTypeV4 = Literal[
    "navigate",
    "stop",
    "finish",
    "request_user",
    "request_replan",
]
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
    """Normalize a goal from prompt-visible task sources."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    goal_ref: TaskGoalRefV4 = "task.goal"
    goal_type: GoalTypeV4
    description: str | None = Field(
        default=None,
        min_length=1,
        max_length=320,
    )
    specification: GoalSpecificationV4
    source_refs: list[ExternalEvidenceRefV4] = Field(
        default_factory=list,
        max_length=12,
    )

    @model_validator(mode="after")
    def validate_goal_specification(self) -> "TaskGoalV4":
        """Keep goal absence and evidence mutually consistent."""

        if self.specification == "absent" and self.goal_type != "none":
            raise ValueError(
                "absent goal specification requires goal_type=none"
            )
        if self.goal_type == "none" and self.specification != "absent":
            raise ValueError(
                "goal_type=none requires specification=absent"
            )
        if self.specification == "specified" and not self.source_refs:
            raise ValueError(
                "specified goal requires at least one raw source ref"
            )
        return self


class TaskRequirementV4(BaseModel):
    """Represent one explicit task-specific behavioral requirement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_ref: RequirementRefV4
    code: ReasonCode
    polarity: RequirementPolarityV4
    target_description: str | None = Field(
        default=None,
        min_length=1,
        max_length=320,
    )
    evidence_refs: list[ExternalEvidenceRefV4] = Field(
        min_length=1,
        max_length=12,
    )


class TaskAmbiguityV4(BaseModel):
    """Represent one task ambiguity that can change the next action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ambiguity_ref: AmbiguityRefV4
    code: ReasonCode
    source_refs: list[ExternalEvidenceRefV4] = Field(
        min_length=1,
        max_length=12,
    )


class TaskGroundingV4(BaseModel):
    """Normalize task sources without predicting scene facts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    objective: ObjectiveV4
    goal: TaskGoalV4
    requirements: list[TaskRequirementV4] = Field(
        default_factory=list,
        max_length=12,
    )
    ambiguities: list[TaskAmbiguityV4] = Field(
        default_factory=list,
        max_length=12,
    )

    @model_validator(mode="after")
    def validate_local_refs(self) -> "TaskGroundingV4":
        """Require unique task-local references."""

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
    category: ReasonCode | None = None
    category_candidates: list[ReasonCode] = Field(
        default_factory=list,
        max_length=8,
    )
    evidence_status: EvidenceStatusV4
    observation_quality: ObservationQualityV4 = "unknown"
    lateral_position: LateralPositionV4 = "unknown"
    depth_position: DepthPositionV4 = "unknown"
    evidence_refs: list[ExternalEvidenceRefV4] = Field(
        min_length=1,
        max_length=12,
    )

    @model_validator(mode="after")
    def validate_entity(self) -> "SceneEntityV4":
        """Require FPV support and a coherent open-category claim."""

        _require_fpv_evidence(self.evidence_refs, "scene entity")
        _validate_open_category(
            category=self.category,
            candidates=self.category_candidates,
            evidence_status=self.evidence_status,
            object_name="scene entity",
        )
        return self


class TrafficControlV4(BaseModel):
    """Describe one directly visible traffic-control facility."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    control_ref: ControlRefV4
    category: ReasonCode | None = None
    category_candidates: list[ReasonCode] = Field(
        default_factory=list,
        max_length=8,
    )
    state: ReasonCode | None = None
    evidence_status: EvidenceStatusV4
    observation_quality: ObservationQualityV4 = "unknown"
    lateral_position: LateralPositionV4 = "unknown"
    depth_position: DepthPositionV4 = "unknown"
    evidence_refs: list[ExternalEvidenceRefV4] = Field(
        min_length=1,
        max_length=12,
    )

    @model_validator(mode="after")
    def validate_control(self) -> "TrafficControlV4":
        """Require FPV support and a coherent open-category claim."""

        _require_fpv_evidence(self.evidence_refs, "traffic control")
        _validate_open_category(
            category=self.category,
            candidates=self.category_candidates,
            evidence_status=self.evidence_status,
            object_name="traffic control",
        )
        return self


class SceneGroundingV4(BaseModel):
    """Collect visible scene facts without path-policy judgments."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entities: list[SceneEntityV4] = Field(
        default_factory=list,
        max_length=32,
    )
    traffic_controls: list[TrafficControlV4] = Field(
        default_factory=list,
        max_length=16,
    )

    @model_validator(mode="after")
    def validate_local_refs(self) -> "SceneGroundingV4":
        """Require unique neutral scene references."""

        _require_unique(
            [item.entity_ref for item in self.entities],
            "entity_ref",
        )
        _require_unique(
            [item.control_ref for item in self.traffic_controls],
            "control_ref",
        )
        return self


class DirectionAffordanceV4(BaseModel):
    """Describe FPV support for one public direction label.

    Args:
        direction_label: Label selected from the public direction vocabulary.
        support_type: Open semantic code for the visible walking support.
        affordance: Categorical local executability judgment.
        clearance: Coarse visual clearance category.
        limiting_factor_codes: Open codes explaining a non-direct judgment.
        evidence_status: Categorical support state, not a probability.
        evidence_refs: Raw FPV evidence addresses supporting the judgment.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    direction_label: ReasonCode
    support_type: ReasonCode
    affordance: DirectionAffordanceStateV4
    clearance: ClearanceV4 = "unknown"
    limiting_factor_codes: list[ReasonCode] = Field(
        default_factory=list,
        max_length=12,
    )
    evidence_status: EvidenceStatusV4
    evidence_refs: list[ExternalEvidenceRefV4] = Field(
        min_length=1,
        max_length=12,
    )

    @model_validator(mode="after")
    def validate_affordance(self) -> "DirectionAffordanceV4":
        """Require visible support and non-contradictory status fields."""

        _require_fpv_evidence(
            self.evidence_refs,
            "direction affordance",
        )
        if self.affordance == "direct" and self.clearance == "insufficient":
            raise ValueError(
                "direct direction cannot have insufficient clearance"
            )
        if (
            self.affordance in {"conditional", "blocked", "forbidden"}
            and not self.limiting_factor_codes
        ):
            raise ValueError(
                f"{self.affordance} direction requires limiting factors"
            )
        return self


class LocalTraversabilityV4(BaseModel):
    """Judge the public direction vocabulary from current FPV evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    direction_affordances: list[DirectionAffordanceV4] = Field(
        default_factory=list,
        max_length=24,
    )

    @model_validator(mode="after")
    def validate_local_labels(self) -> "LocalTraversabilityV4":
        """Require at most one record per public direction label."""

        _require_unique(
            [
                str(item.direction_label)
                for item in self.direction_affordances
            ],
            "direction_label",
        )
        return self


class GoalAlignmentV4(BaseModel):
    """Align a task goal to open target semantics and public directions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    goal_ref: TaskGoalRefV4 = "task.goal"
    status: GoalAlignmentStatusV4
    target_description: str | None = Field(
        default=None,
        min_length=1,
        max_length=320,
    )
    candidate_target_descriptions: list[str] = Field(
        default_factory=list,
        max_length=12,
    )
    supported_direction_labels: list[ReasonCode] = Field(
        default_factory=list,
        max_length=24,
    )
    evidence_refs: list[ExternalEvidenceRefV4] = Field(
        min_length=1,
        max_length=16,
    )

    @model_validator(mode="after")
    def validate_alignment(self) -> "GoalAlignmentV4":
        """Keep open target descriptions consistent with alignment status."""

        if self.status == "resolved" and self.target_description is None:
            raise ValueError(
                "resolved goal alignment requires target_description"
            )
        if self.status != "resolved" and self.target_description is not None:
            raise ValueError(
                "non-resolved goal alignment cannot carry target_description"
            )
        if self.status == "resolved" and self.candidate_target_descriptions:
            raise ValueError(
                "resolved goal alignment cannot carry target candidates"
            )
        if (
            self.status == "ambiguous"
            and len(self.candidate_target_descriptions) < 2
        ):
            raise ValueError(
                "ambiguous goal alignment requires two target descriptions"
            )
        if self.status in {"not_visible", "insufficient_evidence"} and (
            self.candidate_target_descriptions
            or self.supported_direction_labels
        ):
            raise ValueError(
                f"{self.status} goal alignment cannot assert directions"
            )
        if (
            self.supported_direction_labels
            and _contains_disabled_direction_source(self.evidence_refs)
        ):
            raise ValueError(
                "pixel-goal and waypoint evidence cannot support goal "
                "direction labels in V4.2"
            )
        return self


class NavigationSourceAlignmentV4(BaseModel):
    """Record evidence-supported directions for one visible task source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    alignment_ref: AlignmentRefV4
    source_ref: ExternalEvidenceRefV4
    alignment_role: AlignmentRoleV4
    supported_direction_labels: list[ReasonCode] = Field(
        default_factory=list,
        max_length=24,
    )
    status: SourceAlignmentStatusV4
    evidence_refs: list[ExternalEvidenceRefV4] = Field(
        min_length=1,
        max_length=16,
    )

    @model_validator(mode="after")
    def validate_source_alignment(self) -> "NavigationSourceAlignmentV4":
        """Require direct source evidence and block unsupported geometry."""

        if self.source_ref not in self.evidence_refs:
            raise ValueError(
                "source alignment must cite source_ref as raw evidence"
            )
        unavailable_geometry = _contains_disabled_direction_source(
            [self.source_ref]
        )
        if unavailable_geometry and (
            self.status != "insufficient"
            or self.supported_direction_labels
        ):
            raise ValueError(
                "pixel-goal and waypoint source alignment must be "
                "insufficient without direction labels in V4.2"
            )
        if self.status == "supports" and not self.supported_direction_labels:
            raise ValueError(
                "supporting source alignment requires direction labels"
            )
        if self.status != "supports" and self.supported_direction_labels:
            raise ValueError(
                f"{self.status} source alignment cannot assert supported "
                "directions"
            )
        return self


class TaskDirectionV4(BaseModel):
    """Collect public direction candidates implied by visible task evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    direction_ref: Literal["nav.task_direction"] = "nav.task_direction"
    candidate_direction_labels: list[ReasonCode] = Field(
        default_factory=list,
        max_length=24,
    )
    status: TaskDirectionStatusV4
    evidence_refs: list[ExternalEvidenceRefV4] = Field(
        min_length=1,
        max_length=16,
    )

    @model_validator(mode="after")
    def validate_candidates(self) -> "TaskDirectionV4":
        """Keep candidate availability consistent with resolution status."""

        if "stop" in self.candidate_direction_labels:
            raise ValueError(
                "task direction cannot use stop as navigation intent"
            )
        if self.status == "aligned" and not self.candidate_direction_labels:
            raise ValueError(
                "aligned task direction requires candidate labels"
            )
        if self.status in {"not_visible", "insufficient_evidence"} and (
            self.candidate_direction_labels
        ):
            raise ValueError(
                f"{self.status} task direction cannot assert candidates"
            )
        return self


class NavigationContextGroundingV4(BaseModel):
    """Ground visible task sources to the public direction vocabulary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    goal_alignment: GoalAlignmentV4 | None = None
    source_alignments: list[NavigationSourceAlignmentV4] = Field(
        default_factory=list,
        max_length=16,
    )
    task_direction: TaskDirectionV4

    @model_validator(mode="after")
    def validate_local_refs(self) -> "NavigationContextGroundingV4":
        """Require unique refs and close task candidates to local alignments."""

        _require_unique(
            [item.alignment_ref for item in self.source_alignments],
            "alignment_ref",
        )
        grounded_labels: set[str] = set()
        if self.goal_alignment is not None:
            grounded_labels.update(
                str(value)
                for value in (
                    self.goal_alignment.supported_direction_labels
                )
            )
        for alignment in self.source_alignments:
            if alignment.alignment_role == "local_safety_bias":
                continue
            grounded_labels.update(
                str(value)
                for value in alignment.supported_direction_labels
            )
        ungrounded = sorted(
            {
                str(value)
                for value in (
                    self.task_direction.candidate_direction_labels
                )
            }
            - grounded_labels
        )
        if ungrounded:
            raise ValueError(
                "task direction candidates require same-section source "
                f"support: {ungrounded}"
            )
        return self


class InteractionConstraintV4(BaseModel):
    """Convert visible evidence into a global or subset behavior constraint.

    An empty ``affected_direction_labels`` list means that the constraint applies
    to the current action globally. A non-empty list identifies only the affected
    subset; it must not enumerate the complement or expand an ``all_moving``
    pseudo-scope.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    constraint_ref: ConstraintRefV4
    kind: ConstraintKindV4
    source_type: ReasonCode
    source_state: ReasonCode
    affected_direction_labels: list[ReasonCode] = Field(
        default_factory=list,
        max_length=24,
    )
    relation: ConstraintRelationV4
    status: ConstraintStatusV4
    effect: ConstraintEffectV4
    evidence_refs: list[ExternalEvidenceRefV4] = Field(
        min_length=1,
        max_length=16,
    )

    @model_validator(mode="after")
    def validate_constraint(self) -> "InteractionConstraintV4":
        """Validate global-stop, subset-avoid, and dynamic evidence semantics."""

        _require_unique(
            [str(value) for value in self.affected_direction_labels],
            "affected_direction_labels",
        )
        if self.effect == "stop" and self.affected_direction_labels:
            raise ValueError(
                "stop constraint must not enumerate affected directions"
            )
        if self.effect == "avoid" and not self.affected_direction_labels:
            raise ValueError(
                "avoid constraint requires affected_direction_labels"
            )
        dynamic = self.relation in {
            "crosses_candidate_path",
            "approaches_candidate_path",
        }
        fpv_refs = {
            ref
            for ref in self.evidence_refs
            if ref.startswith("fpv.")
        }
        if dynamic and len(fpv_refs) < 2:
            raise ValueError(
                "dynamic interaction requires two distinct FPV refs"
            )
        return self


class InteractionConstraintGroundingV4(BaseModel):
    """Collect direction-scoped interaction and rule constraints."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    constraints: list[InteractionConstraintV4] = Field(
        default_factory=list,
        max_length=24,
    )

    @model_validator(mode="after")
    def validate_local_refs(self) -> "InteractionConstraintGroundingV4":
        """Require unique neutral constraint references."""

        _require_unique(
            [item.constraint_ref for item in self.constraints],
            "constraint_ref",
        )
        return self


class CriticalUncertaintyV4(BaseModel):
    """Represent a material unresolved issue without pseudo-confidence.

    Args:
        uncertainty_ref: Neutral sample-local uncertainty identifier.
        uncertainty_code: Open semantic code for the unresolved issue.
        uncertainty_type: Closed uncertainty family.
        affected_direction_labels: Public directions affected by the issue.
        resolution_policy: Closed conservative response policy.
        evidence_refs: Raw visible evidence addresses.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    uncertainty_ref: UncertaintyRefV4
    uncertainty_code: ReasonCode
    uncertainty_type: UncertaintyTypeV4
    affected_direction_labels: list[ReasonCode] = Field(
        default_factory=list,
        max_length=24,
    )
    resolution_policy: ResolutionPolicyV4
    evidence_refs: list[ExternalEvidenceRefV4] = Field(
        min_length=1,
        max_length=16,
    )


class DecisionProposalV4(BaseModel):
    """Teacher-produced decision fields before deterministic motion mapping.

    Args:
        decision_type: Stage-specific semantic decision type.
        navigation_intent_direction_label: Optional non-stop task direction
            inferred from prompt-visible evidence.
        selected_action_label: Immediate action selected from the public
            DirectionProfile, including ``stop`` when required.
        speed: Coarse non-metric speed policy.
        social_action: Closed social-interaction action.
        fallback: Conservative policy if the selected action becomes invalid.
        critical_uncertainties: Material unresolved issues for this action.
        support_refs: Sample-local semantic objects supporting the decision.
        evidence_refs: Raw prompt-visible evidence supporting the decision.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_type: DecisionTypeV4
    navigation_intent_direction_label: ReasonCode | None = None
    selected_action_label: ReasonCode
    speed: SpeedV4
    social_action: SocialActionV4
    fallback: FallbackV4
    critical_uncertainties: list[CriticalUncertaintyV4] = Field(
        default_factory=list,
        max_length=12,
    )
    support_refs: list[SemanticRefV4] = Field(
        default_factory=list,
        max_length=16,
    )
    evidence_refs: list[ExternalEvidenceRefV4] = Field(
        min_length=1,
        max_length=16,
    )

    @model_validator(mode="after")
    def validate_proposal_shape(self) -> "DecisionProposalV4":
        """Enforce the local shape of a stop proposal."""

        _require_unique(
            [
                item.uncertainty_ref
                for item in self.critical_uncertainties
            ],
            "uncertainty_ref",
        )
        if self.navigation_intent_direction_label == "stop":
            raise ValueError(
                "navigation intent direction cannot be stop"
            )
        selected_is_stop = self.selected_action_label == "stop"
        if (self.decision_type == "stop") != selected_is_stop:
            raise ValueError(
                "decision_type must match selected_action_label stop state"
            )
        if self.decision_type == "stop" and (
            self.speed != "stop"
            or self.social_action != "wait"
            or self.fallback != "stop"
        ):
            raise ValueError(
                "stop decision proposal requires speed=stop, "
                "social_action=wait, and fallback=stop"
            )
        if self.decision_type == "navigate" and self.speed == "stop":
            raise ValueError(
                "navigate decision proposal cannot use speed=stop"
            )
        return self


class DecisionV4(DecisionProposalV4):
    """Canonical decision after profile-owned motion composition.

    Args:
        motion: Coarse motion deterministically projected by DirectionProfile.
        **inherited: All weak proposal fields from :class:`DecisionProposalV4`.
    """

    motion: MotionV4

    @model_validator(mode="after")
    def validate_decision_shape(self) -> "DecisionV4":
        """Require a stop decision to carry stop motion and speed."""

        if self.decision_type == "stop" and (
            self.motion != "stop" or self.speed != "stop"
        ):
            raise ValueError(
                "stop decision requires motion=stop and speed=stop"
            )
        return self


class ReasoningTraceV4(BaseModel):
    """Canonical six-section shared-input Reasoning V4.2 trace."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[REASONING_TRACE_V4_SCHEMA_VERSION] = (
        REASONING_TRACE_V4_SCHEMA_VERSION
    )
    task_grounding: TaskGroundingV4
    scene_grounding: SceneGroundingV4
    local_traversability: LocalTraversabilityV4
    navigation_context_grounding: NavigationContextGroundingV4
    interaction_constraint_grounding: InteractionConstraintGroundingV4
    decision: DecisionV4


class FinishDirectionV4(BaseModel):
    """Expose exactly one trainable demonstrated direction label."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: ReasonCode


class FinishV4(BaseModel):
    """Project the current Decision and deterministic demonstrated direction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[FINISH_V4_SCHEMA_VERSION] = (
        FINISH_V4_SCHEMA_VERSION
    )
    decision_type: DecisionTypeV4
    motion: MotionV4
    speed: SpeedV4
    social_action: SocialActionV4
    selected_action_label: ReasonCode
    demonstrated_direction: FinishDirectionV4


class ReasoningTargetV4(BaseModel):
    """Pair a canonical Reasoning V4.2 trace with its finish projection."""

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
            self.finish.selected_action_label,
        ) != (
            decision.decision_type,
            decision.motion,
            decision.speed,
            decision.social_action,
            decision.selected_action_label,
        ):
            raise ValueError(
                "finish must be an exact semantic projection of Decision"
            )
        return self


def reasoning_v4_contract_schema_digest() -> str:
    """Return a semantic SHA-256 digest for the canonical V4.2 target."""

    schema = _semantic_contract_schema(
        ReasoningTargetV4.model_json_schema()
    )
    payload = json.dumps(
        schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def raw_evidence_refs_v4(trace: ReasoningTraceV4) -> set[str]:
    """Collect all prompt-visible evidence references in a V4.2 trace."""

    refs: set[str] = set(trace.task_grounding.goal.source_refs)
    for requirement in trace.task_grounding.requirements:
        refs.update(requirement.evidence_refs)
    for ambiguity in trace.task_grounding.ambiguities:
        refs.update(ambiguity.source_refs)
    for entity in trace.scene_grounding.entities:
        refs.update(entity.evidence_refs)
    for control in trace.scene_grounding.traffic_controls:
        refs.update(control.evidence_refs)
    for affordance in trace.local_traversability.direction_affordances:
        refs.update(affordance.evidence_refs)
    navigation = trace.navigation_context_grounding
    if navigation.goal_alignment is not None:
        refs.update(navigation.goal_alignment.evidence_refs)
    for alignment in navigation.source_alignments:
        refs.add(alignment.source_ref)
        refs.update(alignment.evidence_refs)
    refs.update(navigation.task_direction.evidence_refs)
    for constraint in (
        trace.interaction_constraint_grounding.constraints
    ):
        refs.update(constraint.evidence_refs)
    for uncertainty in trace.decision.critical_uncertainties:
        refs.update(uncertainty.evidence_refs)
    refs.update(trace.decision.evidence_refs)
    return refs


def semantic_refs_v4(trace: ReasoningTraceV4) -> set[str]:
    """Collect sample-local links used by the final Decision."""

    return set(trace.decision.support_refs)


def _validate_open_category(
    *,
    category: str | None,
    candidates: list[str],
    evidence_status: str,
    object_name: str,
) -> None:
    """Keep open category values and ambiguity status coherent."""

    if evidence_status == "supported":
        if category is None:
            raise ValueError(
                f"supported {object_name} requires category"
            )
        if candidates:
            raise ValueError(
                f"supported {object_name} cannot carry candidates"
            )
    elif evidence_status == "ambiguous":
        if category is not None or len(candidates) < 2:
            raise ValueError(
                f"ambiguous {object_name} requires two candidates"
            )
    elif category is not None or candidates:
        raise ValueError(
            f"insufficient {object_name} cannot assert category"
        )


def _require_unique(values: list[str], field_name: str) -> None:
    """Raise when one sample-local value is emitted more than once."""

    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} values must be unique")


def _require_fpv_evidence(
    values: list[str],
    object_name: str,
) -> None:
    """Require at least one raw FPV frame for a visual claim."""

    if not any(value.startswith("fpv.") for value in values):
        raise ValueError(
            f"{object_name} requires at least one FPV evidence ref"
        )


def _contains_disabled_direction_source(values: list[str]) -> bool:
    """Return whether evidence depends on disabled direction geometry."""

    return any(
        value.startswith(("waypoint.", "goal.pixel_"))
        for value in values
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


REASONING_V4_CONTRACT_SCHEMA_DIGEST = (
    reasoning_v4_contract_schema_digest()
)
