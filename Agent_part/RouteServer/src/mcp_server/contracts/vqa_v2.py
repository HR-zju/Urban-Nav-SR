"""Region-free typed contracts for UrbanNav Navigation VQA V2."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mcp_server.contracts.reasoning import ReasonCode


VQA_V2_SCHEMA_VERSION = "urbannav_navigation_vqa_v2_0"
VQA_V2_WIRE_FORMAT_VERSION = "urbannav_navigation_vqa_wire_v2"
VQA_V2_TASK_REGISTRY_VERSION = (
    "urbannav_navigation_vqa_task_registry_v2_0"
)
VQA_V2_PARSER_ID = "urbannav_navigation_vqa_parser_v2_0"
VQA_V2_RENDERER_ID = "urbannav_navigation_vqa_renderer_v2_0"
VQA_V2_ANSWER_TAG = "VQAAnswer"

VQATaskIdV2 = Literal[
    "a1_goal_understanding",
    "a2_explicit_requirement",
    "a3_task_ambiguity",
    "b1_direction_affordance",
    "b2_direction_clearance",
    "b3_direction_surface_effect",
    "b4_conditional_affordance",
    "b5_direction_comparison",
    "b6_unavailable_directions",
    "c1_goal_to_direction",
    "c2_pixel_goal_to_direction",
    "c2_waypoint_to_direction",
    "c3_bev_route_alignment",
    "c4_bev_object_anchoring",
    "c5_outside_view_direction",
    "c6_bev_fpv_executability",
    "d1_agent_relevance",
    "d2_static_interaction_relation",
    "d3_temporal_interaction_relation",
    "d4_applicable_constraint",
    "d5_appropriate_response",
    "d6_traffic_control_applicability",
    "e0_demonstrated_direction_selection",
    "e1_candidate_trajectory_preference",
    "e2_candidate_direction_classification",
    "e3_trajectory_compatibility",
    "e4_alternative_direction",
]

VQA_V2_TASK_IDS: tuple[str, ...] = (
    "a1_goal_understanding",
    "a2_explicit_requirement",
    "a3_task_ambiguity",
    "b1_direction_affordance",
    "b2_direction_clearance",
    "b3_direction_surface_effect",
    "b4_conditional_affordance",
    "b5_direction_comparison",
    "b6_unavailable_directions",
    "c1_goal_to_direction",
    "c2_pixel_goal_to_direction",
    "c2_waypoint_to_direction",
    "c3_bev_route_alignment",
    "c4_bev_object_anchoring",
    "c5_outside_view_direction",
    "c6_bev_fpv_executability",
    "d1_agent_relevance",
    "d2_static_interaction_relation",
    "d3_temporal_interaction_relation",
    "d4_applicable_constraint",
    "d5_appropriate_response",
    "d6_traffic_control_applicability",
    "e0_demonstrated_direction_selection",
    "e1_candidate_trajectory_preference",
    "e2_candidate_direction_classification",
    "e3_trajectory_compatibility",
    "e4_alternative_direction",
)

VQAAnswerabilityV2 = Literal[
    "answerable",
    "insufficient_evidence",
    "not_applicable",
]
DirectionAlignmentStatusV2 = Literal[
    "aligned",
    "conflict",
    "not_visible",
    "insufficient_evidence",
]
NavigationEffectV2 = Literal[
    "proceed",
    "slow",
    "yield",
    "wait",
    "stop",
    "detour",
    "avoid",
    "unknown",
]

_FORBIDDEN_REGION_PATTERN = re.compile(
    r"(?:^|[^a-z])region(?:_|[^a-z]|$)",
    flags=re.IGNORECASE,
)


class GoalUnderstandingPayloadV2(BaseModel):
    """Answer one explicit task-goal understanding question."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    goal: str | None = Field(default=None, max_length=320)
    status: Literal[
        "resolved",
        "absent",
        "ambiguous",
        "insufficient_evidence",
    ]

    @model_validator(mode="after")
    def validate_goal_status(self) -> "GoalUnderstandingPayloadV2":
        """Keep resolved goals and null states consistent."""

        if self.status == "resolved":
            _validate_open_text(self.goal, field_name="goal")
        elif self.goal is not None:
            raise ValueError("unresolved goal payload must keep goal null")
        return self


class ExplicitRequirementPayloadV2(BaseModel):
    """Answer which explicit task requirements are present."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirements: list[ReasonCode] = Field(
        default_factory=list,
        max_length=16,
    )
    status: Literal[
        "resolved",
        "none_stated",
        "ambiguous",
        "insufficient_evidence",
    ]

    @model_validator(mode="after")
    def validate_requirements(self) -> "ExplicitRequirementPayloadV2":
        """Require values only when explicit requirements are resolved."""

        if len(set(self.requirements)) != len(self.requirements):
            raise ValueError("requirements must be unique")
        if self.status == "resolved" and not self.requirements:
            raise ValueError(
                "resolved requirement payload requires a requirement"
            )
        if self.status != "resolved" and self.requirements:
            raise ValueError(
                "non-resolved requirement payload must keep requirements empty"
            )
        return self


class TaskAmbiguityPayloadV2(BaseModel):
    """Answer whether prompt-visible task language is materially ambiguous."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ambiguities: list[ReasonCode] = Field(
        default_factory=list,
        max_length=16,
    )
    status: Literal[
        "clear",
        "ambiguous",
        "insufficient_evidence",
    ]

    @model_validator(mode="after")
    def validate_ambiguities(self) -> "TaskAmbiguityPayloadV2":
        """Keep ambiguity codes aligned with the declared status."""

        if len(set(self.ambiguities)) != len(self.ambiguities):
            raise ValueError("ambiguities must be unique")
        if self.status == "ambiguous" and not self.ambiguities:
            raise ValueError(
                "ambiguous task payload requires an ambiguity code"
            )
        if self.status != "ambiguous" and self.ambiguities:
            raise ValueError(
                "non-ambiguous task payload must keep ambiguities empty"
            )
        return self


class DirectionAffordancePayloadV2(BaseModel):
    """Classify local affordance along a question-specified direction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    affordance: Literal[
        "direct",
        "conditional",
        "blocked",
        "forbidden",
        "unknown",
    ]


class DirectionClearancePayloadV2(BaseModel):
    """Classify coarse visible clearance along one direction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clearance: Literal[
        "sufficient",
        "narrow",
        "insufficient",
        "unknown",
    ]


class DirectionSurfaceEffectPayloadV2(BaseModel):
    """Describe visible support and its navigation effect."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    surface: ReasonCode
    effect: Literal["proceed", "slow", "avoid", "unknown"]


class ConditionalAffordancePayloadV2(BaseModel):
    """Describe immediate usability and any required behavior."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    usability: Literal[
        "immediate",
        "not_immediately",
        "unavailable",
        "unknown",
    ]
    condition_code: ReasonCode | None = None
    effect: NavigationEffectV2


class DirectionComparisonPayloadV2(BaseModel):
    """Select the more suitable direction from supplied candidates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    preferred_direction_label: ReasonCode
    reason_codes: list[ReasonCode] = Field(
        default_factory=list,
        max_length=8,
    )


class UnavailableDirectionsPayloadV2(BaseModel):
    """Identify unusable labels only within a supplied candidate set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    direction_labels: list[ReasonCode] = Field(
        default_factory=list,
        max_length=20,
    )
    reason_codes: list[ReasonCode] = Field(
        default_factory=list,
        max_length=16,
    )


class GoalToDirectionPayloadV2(BaseModel):
    """Ground a visible goal to one or more profile directions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    direction_label: ReasonCode | None = None
    candidate_direction_labels: list[ReasonCode] = Field(
        default_factory=list,
        max_length=8,
    )
    status: Literal[
        "resolved",
        "ambiguous",
        "not_visible",
        "insufficient_evidence",
    ]

    @model_validator(mode="after")
    def validate_resolution(self) -> "GoalToDirectionPayloadV2":
        """Require one label for resolved and candidates for ambiguous."""

        if self.status == "resolved":
            if self.direction_label is None:
                raise ValueError("resolved goal direction requires one label")
            if self.candidate_direction_labels:
                raise ValueError(
                    "resolved goal direction cannot carry candidates"
                )
        elif self.status == "ambiguous":
            if self.direction_label is not None:
                raise ValueError(
                    "ambiguous goal direction cannot select one label"
                )
            if len(self.candidate_direction_labels) < 2:
                raise ValueError(
                    "ambiguous goal direction requires two candidates"
                )
        elif (
            self.direction_label is not None
            or self.candidate_direction_labels
        ):
            raise ValueError(
                "unresolved goal direction must keep labels empty"
            )
        return self


class SuppliedTargetDirectionPayloadV2(BaseModel):
    """Align a supplied pixel goal or waypoint to a direction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    direction_label: ReasonCode | None = None
    status: DirectionAlignmentStatusV2

    @model_validator(mode="after")
    def validate_alignment(self) -> "SuppliedTargetDirectionPayloadV2":
        """Keep aligned targets and null failure states consistent."""

        if self.status == "aligned" and self.direction_label is None:
            raise ValueError("aligned target requires direction_label")
        if self.status != "aligned" and self.direction_label is not None:
            raise ValueError(
                "non-aligned target must keep direction_label null"
            )
        return self


class BEVRouteAlignmentPayloadV2(BaseModel):
    """Align an independently validated BEV route to FPV direction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    direction_label: ReasonCode | None = None
    alignment_status: Literal[
        "aligned",
        "locally_blocked",
        "not_visible",
        "conflict",
        "insufficient_evidence",
    ]

    @model_validator(mode="after")
    def validate_alignment(self) -> "BEVRouteAlignmentPayloadV2":
        """Require a requested direction when route intent is resolved."""

        resolved = self.alignment_status in {
            "aligned",
            "locally_blocked",
        }
        if resolved != (self.direction_label is not None):
            raise ValueError(
                "resolved BEV alignment and direction_label must coexist"
            )
        return self


class BEVObjectAnchoringPayloadV2(BaseModel):
    """Bind validated prompt-visible BEV and FPV entity references."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bev_ref: str | None = Field(default=None, max_length=160)
    fpv_entity_ref: str | None = Field(default=None, max_length=160)
    direction_label: ReasonCode | None = None
    status: Literal[
        "matched",
        "ambiguous",
        "not_visible",
        "insufficient_evidence",
    ]

    @model_validator(mode="after")
    def validate_match(self) -> "BEVObjectAnchoringPayloadV2":
        """Require all match identities only for matched outputs."""

        values = (
            self.bev_ref,
            self.fpv_entity_ref,
            self.direction_label,
        )
        if self.status == "matched":
            if any(value is None for value in values):
                raise ValueError(
                    "matched BEV object requires refs and direction"
                )
        elif any(value is not None for value in values):
            raise ValueError(
                "unmatched BEV object must keep identities null"
            )
        return self


class OutsideViewDirectionPayloadV2(BaseModel):
    """Separate present motion from validated future BEV information."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outside_view_information: list[ReasonCode] = Field(
        default_factory=list,
        max_length=16,
    )
    current_direction_label: ReasonCode | None = None
    future_direction_label: ReasonCode | None = None


class BEVFPVExecutabilityPayloadV2(BaseModel):
    """Judge current FPV executability of a validated BEV direction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    direction_label: ReasonCode | None = None
    status: Literal[
        "executable",
        "conditional",
        "locally_blocked",
        "conflict",
        "insufficient_evidence",
    ]

    @model_validator(mode="after")
    def validate_status(self) -> "BEVFPVExecutabilityPayloadV2":
        """Require direction except when route evidence is insufficient."""

        if (
            self.status == "insufficient_evidence"
        ) != (self.direction_label is None):
            raise ValueError(
                "executability direction is absent only for "
                "insufficient_evidence"
            )
        return self


class AgentRelevancePayloadV2(BaseModel):
    """Classify visible-agent relevance to a candidate direction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relevance: Literal[
        "not_relevant",
        "near_candidate_path",
        "occupies_candidate_path",
        "blocks_candidate_path",
        "uncertain",
    ]


class StaticInteractionRelationPayloadV2(BaseModel):
    """Classify a static visible relation to a candidate direction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relation: Literal[
        "outside_candidate_path",
        "near_candidate_path",
        "occupies_candidate_path",
        "narrows_candidate_path",
        "blocks_candidate_path",
        "unknown",
    ]


class TemporalInteractionRelationPayloadV2(BaseModel):
    """Classify a temporal relation supported by distinct FPV frames."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relation: Literal[
        "crosses_candidate_path",
        "approaches_candidate_path",
        "moves_away_from_path",
        "same_direction",
        "opposite_direction",
        "persistent_occupancy",
        "unknown",
    ]


class ApplicableConstraintPayloadV2(BaseModel):
    """Describe one constraint and its affected supplied directions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    constraint_code: ReasonCode
    kind: Literal[
        "task",
        "traffic",
        "social",
        "accessibility",
        "local_safety",
    ]
    effect: NavigationEffectV2
    affected_direction_labels: list[ReasonCode] = Field(
        default_factory=list,
        max_length=20,
    )


class AppropriateResponsePayloadV2(BaseModel):
    """Select the appropriate behavior before taking a direction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    effect: NavigationEffectV2


class TrafficControlApplicabilityPayloadV2(BaseModel):
    """Judge whether a visible control governs supplied directions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    applicability: Literal[
        "applies",
        "does_not_apply",
        "uncertain",
    ]
    effect: NavigationEffectV2
    controlled_direction_labels: list[ReasonCode] = Field(
        default_factory=list,
        max_length=20,
    )


class DemonstratedDirectionPayloadV2(BaseModel):
    """Predict the demonstrated short-horizon policy direction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    direction_label: ReasonCode


class CandidateTrajectoryPreferencePayloadV2(BaseModel):
    """Select one prompt-visible candidate trajectory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    preferred_candidate: str = Field(min_length=1, max_length=128)
    reason_codes: list[ReasonCode] = Field(
        default_factory=list,
        max_length=8,
    )


class CandidateDirectionClassificationPayloadV2(BaseModel):
    """Classify one prompt-visible candidate under a direction profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    direction_label: ReasonCode


class TrajectoryCompatibilityPayloadV2(BaseModel):
    """Classify task and visible-scene compatibility of one candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal[
        "compatible",
        "conditional",
        "incompatible",
        "insufficient_evidence",
    ]
    reason_codes: list[ReasonCode] = Field(
        default_factory=list,
        max_length=16,
    )


class AlternativeDirectionPayloadV2(BaseModel):
    """Select an alternative from question-supplied directions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    alternative_direction_label: ReasonCode | None = None
    reason_codes: list[ReasonCode] = Field(
        default_factory=list,
        max_length=8,
    )
    status: Literal[
        "resolved",
        "no_safe_alternative",
        "insufficient_evidence",
    ]

    @model_validator(mode="after")
    def validate_alternative(self) -> "AlternativeDirectionPayloadV2":
        """Keep resolved alternatives and null states consistent."""

        if (
            self.status == "resolved"
        ) != (self.alternative_direction_label is not None):
            raise ValueError(
                "resolved alternative requires exactly one direction"
            )
        return self


class _AnswerEnvelopeBaseV2(BaseModel):
    """Base for task-discriminated canonical annotation answers."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class A1AnswerV2(_AnswerEnvelopeBaseV2):
    """Envelope for A1 goal-understanding answers."""

    task_id: Literal["a1_goal_understanding"]
    payload: GoalUnderstandingPayloadV2


class A2AnswerV2(_AnswerEnvelopeBaseV2):
    """Envelope for A2 explicit-requirement answers."""

    task_id: Literal["a2_explicit_requirement"]
    payload: ExplicitRequirementPayloadV2


class A3AnswerV2(_AnswerEnvelopeBaseV2):
    """Envelope for A3 task-ambiguity answers."""

    task_id: Literal["a3_task_ambiguity"]
    payload: TaskAmbiguityPayloadV2


class B1AnswerV2(_AnswerEnvelopeBaseV2):
    """Envelope for B1 direction-affordance answers."""

    task_id: Literal["b1_direction_affordance"]
    payload: DirectionAffordancePayloadV2


class B2AnswerV2(_AnswerEnvelopeBaseV2):
    """Envelope for B2 direction-clearance answers."""

    task_id: Literal["b2_direction_clearance"]
    payload: DirectionClearancePayloadV2


class B3AnswerV2(_AnswerEnvelopeBaseV2):
    """Envelope for B3 direction-surface-effect answers."""

    task_id: Literal["b3_direction_surface_effect"]
    payload: DirectionSurfaceEffectPayloadV2


class B4AnswerV2(_AnswerEnvelopeBaseV2):
    """Envelope for B4 conditional-affordance answers."""

    task_id: Literal["b4_conditional_affordance"]
    payload: ConditionalAffordancePayloadV2


class B5AnswerV2(_AnswerEnvelopeBaseV2):
    """Envelope for B5 direction-comparison answers."""

    task_id: Literal["b5_direction_comparison"]
    payload: DirectionComparisonPayloadV2


class B6AnswerV2(_AnswerEnvelopeBaseV2):
    """Envelope for B6 unavailable-direction answers."""

    task_id: Literal["b6_unavailable_directions"]
    payload: UnavailableDirectionsPayloadV2


class C1AnswerV2(_AnswerEnvelopeBaseV2):
    """Envelope for C1 goal-to-direction answers."""

    task_id: Literal["c1_goal_to_direction"]
    payload: GoalToDirectionPayloadV2


class C2PixelAnswerV2(_AnswerEnvelopeBaseV2):
    """Envelope for dormant C2 pixel-goal answers."""

    task_id: Literal["c2_pixel_goal_to_direction"]
    payload: SuppliedTargetDirectionPayloadV2


class C2WaypointAnswerV2(_AnswerEnvelopeBaseV2):
    """Envelope for dormant C2 waypoint answers."""

    task_id: Literal["c2_waypoint_to_direction"]
    payload: SuppliedTargetDirectionPayloadV2


class C3AnswerV2(_AnswerEnvelopeBaseV2):
    """Envelope for dormant C3 BEV-route-alignment answers."""

    task_id: Literal["c3_bev_route_alignment"]
    payload: BEVRouteAlignmentPayloadV2


class C4AnswerV2(_AnswerEnvelopeBaseV2):
    """Envelope for dormant C4 cross-view anchoring answers."""

    task_id: Literal["c4_bev_object_anchoring"]
    payload: BEVObjectAnchoringPayloadV2


class C5AnswerV2(_AnswerEnvelopeBaseV2):
    """Envelope for dormant C5 outside-view-direction answers."""

    task_id: Literal["c5_outside_view_direction"]
    payload: OutsideViewDirectionPayloadV2


class C6AnswerV2(_AnswerEnvelopeBaseV2):
    """Envelope for dormant C6 BEV-FPV-executability answers."""

    task_id: Literal["c6_bev_fpv_executability"]
    payload: BEVFPVExecutabilityPayloadV2


class D1AnswerV2(_AnswerEnvelopeBaseV2):
    """Envelope for D1 visible-agent-relevance answers."""

    task_id: Literal["d1_agent_relevance"]
    payload: AgentRelevancePayloadV2


class D2AnswerV2(_AnswerEnvelopeBaseV2):
    """Envelope for D2 static-interaction answers."""

    task_id: Literal["d2_static_interaction_relation"]
    payload: StaticInteractionRelationPayloadV2


class D3AnswerV2(_AnswerEnvelopeBaseV2):
    """Envelope for D3 temporal-interaction answers."""

    task_id: Literal["d3_temporal_interaction_relation"]
    payload: TemporalInteractionRelationPayloadV2


class D4AnswerV2(_AnswerEnvelopeBaseV2):
    """Envelope for D4 applicable-constraint answers."""

    task_id: Literal["d4_applicable_constraint"]
    payload: ApplicableConstraintPayloadV2


class D5AnswerV2(_AnswerEnvelopeBaseV2):
    """Envelope for D5 appropriate-response answers."""

    task_id: Literal["d5_appropriate_response"]
    payload: AppropriateResponsePayloadV2


class D6AnswerV2(_AnswerEnvelopeBaseV2):
    """Envelope for D6 traffic-control-applicability answers."""

    task_id: Literal["d6_traffic_control_applicability"]
    payload: TrafficControlApplicabilityPayloadV2


class E0AnswerV2(_AnswerEnvelopeBaseV2):
    """Envelope for E0 demonstrated-direction answers."""

    task_id: Literal["e0_demonstrated_direction_selection"]
    payload: DemonstratedDirectionPayloadV2


class E1AnswerV2(_AnswerEnvelopeBaseV2):
    """Envelope for dormant E1 candidate-preference answers."""

    task_id: Literal["e1_candidate_trajectory_preference"]
    payload: CandidateTrajectoryPreferencePayloadV2


class E2AnswerV2(_AnswerEnvelopeBaseV2):
    """Envelope for dormant E2 candidate-direction answers."""

    task_id: Literal["e2_candidate_direction_classification"]
    payload: CandidateDirectionClassificationPayloadV2


class E3AnswerV2(_AnswerEnvelopeBaseV2):
    """Envelope for dormant E3 trajectory-compatibility answers."""

    task_id: Literal["e3_trajectory_compatibility"]
    payload: TrajectoryCompatibilityPayloadV2


class E4AnswerV2(_AnswerEnvelopeBaseV2):
    """Envelope for E4 alternative-direction answers."""

    task_id: Literal["e4_alternative_direction"]
    payload: AlternativeDirectionPayloadV2


VQAAnswerV2 = Annotated[
    Union[
        A1AnswerV2,
        A2AnswerV2,
        A3AnswerV2,
        B1AnswerV2,
        B2AnswerV2,
        B3AnswerV2,
        B4AnswerV2,
        B5AnswerV2,
        B6AnswerV2,
        C1AnswerV2,
        C2PixelAnswerV2,
        C2WaypointAnswerV2,
        C3AnswerV2,
        C4AnswerV2,
        C5AnswerV2,
        C6AnswerV2,
        D1AnswerV2,
        D2AnswerV2,
        D3AnswerV2,
        D4AnswerV2,
        D5AnswerV2,
        D6AnswerV2,
        E0AnswerV2,
        E1AnswerV2,
        E2AnswerV2,
        E3AnswerV2,
        E4AnswerV2,
    ],
    Field(discriminator="task_id"),
]

VQA_ANSWER_ENVELOPE_BY_TASK: dict[str, type[BaseModel]] = {
    "a1_goal_understanding": A1AnswerV2,
    "a2_explicit_requirement": A2AnswerV2,
    "a3_task_ambiguity": A3AnswerV2,
    "b1_direction_affordance": B1AnswerV2,
    "b2_direction_clearance": B2AnswerV2,
    "b3_direction_surface_effect": B3AnswerV2,
    "b4_conditional_affordance": B4AnswerV2,
    "b5_direction_comparison": B5AnswerV2,
    "b6_unavailable_directions": B6AnswerV2,
    "c1_goal_to_direction": C1AnswerV2,
    "c2_pixel_goal_to_direction": C2PixelAnswerV2,
    "c2_waypoint_to_direction": C2WaypointAnswerV2,
    "c3_bev_route_alignment": C3AnswerV2,
    "c4_bev_object_anchoring": C4AnswerV2,
    "c5_outside_view_direction": C5AnswerV2,
    "c6_bev_fpv_executability": C6AnswerV2,
    "d1_agent_relevance": D1AnswerV2,
    "d2_static_interaction_relation": D2AnswerV2,
    "d3_temporal_interaction_relation": D3AnswerV2,
    "d4_applicable_constraint": D4AnswerV2,
    "d5_appropriate_response": D5AnswerV2,
    "d6_traffic_control_applicability": D6AnswerV2,
    "e0_demonstrated_direction_selection": E0AnswerV2,
    "e1_candidate_trajectory_preference": E1AnswerV2,
    "e2_candidate_direction_classification": E2AnswerV2,
    "e3_trajectory_compatibility": E3AnswerV2,
    "e4_alternative_direction": E4AnswerV2,
}


@dataclass(frozen=True)
class VQATaskContractSpecV2:
    """Describe stable task semantics without activation state.

    Args:
        task_id: Atomic wire-protocol task identity.
        answer_model_name: Typed answer envelope class name.
        required_modalities: Source modality flags required when producing.
        required_context: Dynamic eligibility requirement codes.
        allowed_label_sources: Provenance classes legal for the task.
        source_interface_field: Private external-GT interface when required.
        assistant_target_projector_id: Minimal target renderer identity.
        evaluation_adapter_id: Typed evaluation semantics identity.
    """

    task_id: str
    answer_model_name: str
    required_modalities: tuple[str, ...]
    required_context: tuple[str, ...]
    allowed_label_sources: tuple[str, ...]
    source_interface_field: str | None
    assistant_target_projector_id: str = VQA_V2_RENDERER_ID
    evaluation_adapter_id: str = (
        "urbannav_navigation_vqa_typed_evaluator_v2"
    )


def _task_spec(
    task_id: str,
    *,
    required_modalities: tuple[str, ...] = ("fpv",),
    required_context: tuple[str, ...] = (),
    allowed_label_sources: tuple[str, ...],
    source_interface_field: str | None = None,
) -> VQATaskContractSpecV2:
    """Build one immutable task contract specification."""

    return VQATaskContractSpecV2(
        task_id=task_id,
        answer_model_name=VQA_ANSWER_ENVELOPE_BY_TASK[
            task_id
        ].__name__,
        required_modalities=required_modalities,
        required_context=required_context,
        allowed_label_sources=allowed_label_sources,
        source_interface_field=source_interface_field,
    )


_WEAK_DIRECTION_CONTEXT = (
    "canonical_direction",
    "non_stop_direction_path",
)

VQA_V2_TASK_SPECS: dict[str, VQATaskContractSpecV2] = {
    **{
        task_id: _task_spec(
            task_id,
            required_context=("visible_task_context",),
            allowed_label_sources=("lvlm_weak",),
        )
        for task_id in (
            "a1_goal_understanding",
            "a2_explicit_requirement",
            "a3_task_ambiguity",
        )
    },
    **{
        task_id: _task_spec(
            task_id,
            required_context=_WEAK_DIRECTION_CONTEXT,
            allowed_label_sources=("lvlm_weak",),
        )
        for task_id in (
            "b1_direction_affordance",
            "b2_direction_clearance",
            "b3_direction_surface_effect",
            "b4_conditional_affordance",
            "b5_direction_comparison",
            "b6_unavailable_directions",
            "d1_agent_relevance",
            "d2_static_interaction_relation",
            "d4_applicable_constraint",
            "d5_appropriate_response",
            "d6_traffic_control_applicability",
        )
    },
    "c1_goal_to_direction": _task_spec(
        "c1_goal_to_direction",
        required_context=("canonical_direction", "visible_goal"),
        allowed_label_sources=("lvlm_weak",),
    ),
    "c2_pixel_goal_to_direction": _task_spec(
        "c2_pixel_goal_to_direction",
        required_modalities=("fpv", "pixel_goal"),
        allowed_label_sources=("external_gt",),
        source_interface_field="pixel_goal_direction",
    ),
    "c2_waypoint_to_direction": _task_spec(
        "c2_waypoint_to_direction",
        required_modalities=("fpv", "waypoint"),
        allowed_label_sources=("external_gt",),
        source_interface_field="waypoint_direction",
    ),
    "c3_bev_route_alignment": _task_spec(
        "c3_bev_route_alignment",
        required_modalities=("fpv", "bev"),
        allowed_label_sources=("external_gt",),
        source_interface_field="bev_route_alignment",
    ),
    "c4_bev_object_anchoring": _task_spec(
        "c4_bev_object_anchoring",
        required_modalities=("fpv", "bev"),
        allowed_label_sources=("external_gt",),
        source_interface_field="cross_view_entity_binding",
    ),
    "c5_outside_view_direction": _task_spec(
        "c5_outside_view_direction",
        required_modalities=("fpv", "bev"),
        allowed_label_sources=("external_gt",),
        source_interface_field="outside_view_direction",
    ),
    "c6_bev_fpv_executability": _task_spec(
        "c6_bev_fpv_executability",
        required_modalities=("fpv", "bev"),
        allowed_label_sources=("external_gt",),
        source_interface_field="bev_fpv_executability",
    ),
    "d3_temporal_interaction_relation": _task_spec(
        "d3_temporal_interaction_relation",
        required_modalities=("fpv", "history_fpv"),
        required_context=(
            *_WEAK_DIRECTION_CONTEXT,
            "verified_history_fpv",
        ),
        allowed_label_sources=("lvlm_weak",),
    ),
    "e0_demonstrated_direction_selection": _task_spec(
        "e0_demonstrated_direction_selection",
        required_context=("canonical_direction",),
        allowed_label_sources=("derived_gt",),
    ),
    "e1_candidate_trajectory_preference": _task_spec(
        "e1_candidate_trajectory_preference",
        allowed_label_sources=("external_gt",),
        source_interface_field="candidate_preference",
    ),
    "e2_candidate_direction_classification": _task_spec(
        "e2_candidate_direction_classification",
        allowed_label_sources=("external_gt",),
        source_interface_field="candidate_direction",
    ),
    "e3_trajectory_compatibility": _task_spec(
        "e3_trajectory_compatibility",
        allowed_label_sources=("external_gt",),
        source_interface_field="trajectory_compatibility",
    ),
    "e4_alternative_direction": _task_spec(
        "e4_alternative_direction",
        required_context=("canonical_direction",),
        allowed_label_sources=("lvlm_weak",),
    ),
}

if set(VQA_V2_TASK_SPECS) != set(VQA_V2_TASK_IDS):
    raise RuntimeError("VQA V2 task specs do not cover the task registry")


def get_vqa_task_spec_v2(task_id: str) -> VQATaskContractSpecV2:
    """Resolve stable task semantics independently of capability state.

    Args:
        task_id: Exact VQA V2 task id.

    Returns:
        Immutable task contract specification.

    Raises:
        ValueError: If the task id is not in the V2 registry.
    """

    try:
        return VQA_V2_TASK_SPECS[str(task_id)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported VQA V2 task: {task_id}"
        ) from exc


class VQAVisibleCandidateV2(BaseModel):
    """Bind a candidate ID to its prompt-visible representation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1, max_length=128)
    prompt_visible_representation_ref: str = Field(
        min_length=1,
        max_length=240,
    )


class VQAQuestionV2(BaseModel):
    """Represent one answer-independent atomic navigation question."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: VQATaskIdV2
    template_id: str = Field(min_length=1, max_length=128)
    question_text: str = Field(min_length=1, max_length=1000)
    direction_profile_id: str | None = Field(
        default=None,
        max_length=128,
    )
    conditioned_direction_label: ReasonCode | None = None
    candidate_direction_labels: list[ReasonCode] = Field(
        default_factory=list,
        max_length=20,
    )
    target_refs: list[str] = Field(default_factory=list, max_length=16)
    evidence_refs: list[str] = Field(
        default_factory=list,
        max_length=64,
    )
    candidate_set_id: str | None = Field(
        default=None,
        max_length=160,
    )
    visible_candidates: list[VQAVisibleCandidateV2] = Field(
        default_factory=list,
        max_length=16,
    )
    target_candidate_id: str | None = Field(
        default=None,
        max_length=128,
    )

    @model_validator(mode="after")
    def validate_question(self) -> "VQAQuestionV2":
        """Reject region vocabulary, duplicates, and incomplete direction use."""

        _validate_open_text(
            self.question_text,
            field_name="question_text",
        )
        if _FORBIDDEN_REGION_PATTERN.search(self.question_text):
            raise ValueError(
                "VQA V2 question must not use region terminology"
            )
        for field_name, values in (
            ("candidate_direction_labels", self.candidate_direction_labels),
            ("target_refs", self.target_refs),
            ("evidence_refs", self.evidence_refs),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must be unique")
        candidate_ids = [
            item.candidate_id for item in self.visible_candidates
        ]
        visible_refs = [
            item.prompt_visible_representation_ref
            for item in self.visible_candidates
        ]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("visible candidate ids must be unique")
        if len(visible_refs) != len(set(visible_refs)):
            raise ValueError(
                "visible candidate representations must be unique"
            )
        if bool(self.visible_candidates) != bool(self.candidate_set_id):
            raise ValueError(
                "visible candidates and candidate_set_id must coexist"
            )
        if (
            self.target_candidate_id is not None
            and self.target_candidate_id not in set(candidate_ids)
        ):
            raise ValueError(
                "target candidate must belong to visible candidates"
            )
        has_direction_context = bool(
            self.conditioned_direction_label
            or self.candidate_direction_labels
        )
        if has_direction_context != bool(self.direction_profile_id):
            raise ValueError(
                "direction context and direction_profile_id must coexist"
            )
        if (
            self.conditioned_direction_label is not None
            and self.conditioned_direction_label
            in self.candidate_direction_labels
        ):
            raise ValueError(
                "conditioned direction must not be duplicated as a candidate"
            )
        return self


def build_vqa_answer_v2(
    task_id: str,
    payload: dict[str, Any] | BaseModel,
) -> VQAAnswerV2:
    """Validate one payload under the expected task-specific answer model.

    Args:
        task_id: Exact VQA V2 atomic task identifier.
        payload: JSON-compatible payload or an already validated model.

    Returns:
        Discriminated typed answer envelope.

    Raises:
        ValueError: If the task id or payload is invalid.
    """

    try:
        model_type = VQA_ANSWER_ENVELOPE_BY_TASK[str(task_id)]
    except KeyError as exc:
        raise ValueError(f"unsupported VQA V2 task: {task_id}") from exc
    raw_payload = (
        payload.model_dump(mode="json")
        if isinstance(payload, BaseModel)
        else payload
    )
    return model_type.model_validate(
        {"task_id": task_id, "payload": raw_payload}
    )


def validate_answer_for_question_v2(
    question: VQAQuestionV2,
    answer: VQAAnswerV2,
) -> None:
    """Validate profile-controlled answer values against one question.

    Args:
        question: Exact answer-independent question.
        answer: Typed task answer.

    Returns:
        ``None``.

    Raises:
        ValueError: If task identity or supplied candidates disagree.
    """

    if answer.task_id != question.task_id:
        raise ValueError("VQA question and answer task_id mismatch")
    payload = answer.payload.model_dump(mode="python")
    candidate_set = set(question.candidate_direction_labels)
    visible_candidate_ids = {
        item.candidate_id for item in question.visible_candidates
    }
    preferred_candidate = payload.get("preferred_candidate")
    if preferred_candidate is not None and (
        not visible_candidate_ids
        or preferred_candidate not in visible_candidate_ids
    ):
        raise ValueError(
            "preferred_candidate must belong to visible candidates"
        )
    selected_fields = (
        "direction_label",
        "current_direction_label",
        "future_direction_label",
        "preferred_direction_label",
        "alternative_direction_label",
    )
    for field_name in selected_fields:
        selected = payload.get(field_name)
        if selected is not None and (
            not candidate_set or selected not in candidate_set
        ):
            raise ValueError(
                f"{field_name} must belong to question candidates"
            )
    for field_name in (
        "direction_labels",
        "affected_direction_labels",
        "controlled_direction_labels",
    ):
        values = set(payload.get(field_name) or [])
        if values and (
            not candidate_set or not values.issubset(candidate_set)
        ):
            raise ValueError(
                f"{field_name} must be a subset of question candidates"
            )
    if question.conditioned_direction_label is not None:
        copied = {
            payload.get("direction_label"),
            payload.get("current_direction_label"),
        } - {None}
        if question.conditioned_direction_label in copied:
            raise ValueError(
                "conditioned direction must not be copied into the answer"
            )


def vqa_v2_contract_schema_digest() -> str:
    """Hash the complete VQA answer/question JSON-schema contract.

    Returns:
        Lowercase SHA-256 digest.
    """

    payload = {
        "schema_version": VQA_V2_SCHEMA_VERSION,
        "wire_format_version": VQA_V2_WIRE_FORMAT_VERSION,
        "task_registry_version": VQA_V2_TASK_REGISTRY_VERSION,
        "question_schema": VQAQuestionV2.model_json_schema(),
        "answer_schemas": {
            task_id: model_type.model_json_schema()
            for task_id, model_type in sorted(
                VQA_ANSWER_ENVELOPE_BY_TASK.items()
            )
        },
        "task_specs": {
            task_id: asdict(spec)
            for task_id, spec in sorted(VQA_V2_TASK_SPECS.items())
        },
    }
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


VQA_V2_CONTRACT_SCHEMA_DIGEST = vqa_v2_contract_schema_digest()


def _validate_open_text(value: str | None, *, field_name: str) -> None:
    """Reject empty, multiline, tag-like, or region-based open text."""

    if value is None:
        raise ValueError(f"{field_name} must not be null")
    text = str(value)
    if not text.strip() or text != text.strip():
        raise ValueError(
            f"{field_name} must be stripped and non-empty"
        )
    if any(ord(char) < 32 or ord(char) == 127 for char in text):
        raise ValueError(
            f"{field_name} must not contain control characters"
        )
    if re.search(r"<\s*/?\s*[A-Za-z][^>]*>", text):
        raise ValueError(f"{field_name} must not contain tags")


__all__ = [
    "A1AnswerV2",
    "A2AnswerV2",
    "A3AnswerV2",
    "AlternativeDirectionPayloadV2",
    "ApplicableConstraintPayloadV2",
    "BEVFPVExecutabilityPayloadV2",
    "BEVObjectAnchoringPayloadV2",
    "BEVRouteAlignmentPayloadV2",
    "CandidateDirectionClassificationPayloadV2",
    "CandidateTrajectoryPreferencePayloadV2",
    "ConditionalAffordancePayloadV2",
    "DemonstratedDirectionPayloadV2",
    "DirectionAffordancePayloadV2",
    "DirectionClearancePayloadV2",
    "DirectionComparisonPayloadV2",
    "DirectionSurfaceEffectPayloadV2",
    "GoalToDirectionPayloadV2",
    "GoalUnderstandingPayloadV2",
    "OutsideViewDirectionPayloadV2",
    "SuppliedTargetDirectionPayloadV2",
    "TrafficControlApplicabilityPayloadV2",
    "TrajectoryCompatibilityPayloadV2",
    "UnavailableDirectionsPayloadV2",
    "VQAAnswerV2",
    "VQAQuestionV2",
    "VQATaskContractSpecV2",
    "VQAVisibleCandidateV2",
    "VQATaskIdV2",
    "VQA_ANSWER_ENVELOPE_BY_TASK",
    "VQA_V2_ANSWER_TAG",
    "VQA_V2_CONTRACT_SCHEMA_DIGEST",
    "VQA_V2_PARSER_ID",
    "VQA_V2_RENDERER_ID",
    "VQA_V2_SCHEMA_VERSION",
    "VQA_V2_TASK_IDS",
    "VQA_V2_TASK_SPECS",
    "VQA_V2_TASK_REGISTRY_VERSION",
    "VQA_V2_WIRE_FORMAT_VERSION",
    "build_vqa_answer_v2",
    "get_vqa_task_spec_v2",
    "validate_answer_for_question_v2",
    "vqa_v2_contract_schema_digest",
]
