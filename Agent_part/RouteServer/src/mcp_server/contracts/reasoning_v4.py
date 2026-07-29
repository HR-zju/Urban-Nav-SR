"""Canonical UrbanNav Reasoning V4.5 semantic contracts.

V4.5 is a breaking simplification of the six-section FullCoT wire.  It models
only stable scene summaries, physical direction affordances, visible route
direction, high-level behavior constraints, and one internally consistent
short-horizon action tuple.  Object-level causal closure, temporal labels,
route-progress semantics, raw evidence references, and ``Finish`` are outside
this contract.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mcp_server.contracts.reasoning import ReasonCode


REASONING_TRACE_V4_SCHEMA_VERSION = "urbannav_reasoning_trace_v4_5"
REASONING_TRAINING_VIEW_V4_SCHEMA_VERSION = (
    "urbannav_reasoning_training_view_v4_5_compact_v1"
)
REASONING_V4_WIRE_FORMAT_VERSION = "urbannav_reasoning_v4_wire_v5"
REASONING_V4_SECTION_DEFINITION_VERSION = (
    "urbannav_reasoning_v4_section_definitions_v7"
)
REASONING_V4_EVIDENCE_POLICY_VERSION = (
    "urbannav_reasoning_v4_evidence_policy_v7"
)
REASONING_V4_RENDERER_ID = "render_reasoning_target_v4_v5"
REASONING_V4_PARSER_ID = "parse_reasoning_target_v4_v5"
REASONING_V4_VALIDATOR_ID = "validate_reasoning_target_v4_v7"
REASONING_V4_COMPACT_FULL_COT_RENDERER_ID = (
    "reasoning_v4_compact_full_cot_renderer_v2"
)
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

TaskDirectionStatusV4 = Literal[
    "aligned",
    "conflict",
    "not_visible",
    "insufficient_evidence",
]
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
PhysicalLimitingFactorV4 = Literal[
    "static_obstacle",
    "dynamic_occupancy",
    "narrow_clearance",
    "non_walkable_surface",
    "curb",
    "wall",
    "vegetation",
    "occlusion",
    "unknown",
]
ConstraintStatusV4 = Literal["applicable", "uncertain"]
ConstraintEffectV4 = Literal["slow", "yield", "stop", "avoid"]
SpeedV4 = Literal["normal", "slow", "stop"]
BehaviorResponseV4 = Literal[
    "proceed",
    "slow",
    "yield",
    "detour",
    "stop",
]


class TaskGroundingV4(BaseModel):
    """Store a canonical task taxonomy derived from the instruction.

    Args:
        objective: Versioned normalized task objective.
        requirements: Explicit task-specific requirements only.
        ambiguities: Material task ambiguities, empty for the common template.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    objective: ReasonCode
    requirements: list[ReasonCode] = Field(default_factory=list, max_length=12)
    ambiguities: list[ReasonCode] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_taxonomy(self) -> "TaskGroundingV4":
        """Require stable, duplicate-free canonical codes."""

        _require_unique(
            [str(value) for value in self.requirements],
            "requirements",
        )
        _require_unique(
            [str(value) for value in self.ambiguities],
            "ambiguities",
        )
        return self


class SceneGroundingV4(BaseModel):
    """Summarize the visible navigation environment without object binding.

    Args:
        scene_type: Open coarse scene code such as ``urban_sidewalk``.
        walkable_support: Open visible support code such as ``crosswalk``.
        key_entities: Open categories that materially affect local navigation.
        traffic_context: Open coarse traffic-control context code.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    scene_type: ReasonCode
    walkable_support: ReasonCode
    key_entities: list[ReasonCode] = Field(default_factory=list, max_length=16)
    traffic_context: ReasonCode

    @model_validator(mode="after")
    def validate_summary(self) -> "SceneGroundingV4":
        """Require duplicate-free coarse entity categories."""

        _require_unique(
            [str(value) for value in self.key_entities],
            "key_entities",
        )
        return self


class DirectionAffordanceV4(BaseModel):
    """Describe physical FPV executability for one public probe direction.

    Args:
        direction_label: Direction from the active affordance-probe profile.
        affordance: Coarse physical executability state.
        clearance: Coarse visible physical clearance.
        limiting_factors: Physical factors only; traffic rules are forbidden.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    direction_label: ReasonCode
    affordance: DirectionAffordanceStateV4
    clearance: ClearanceV4 = "unknown"
    limiting_factors: list[PhysicalLimitingFactorV4] = Field(
        default_factory=list,
        max_length=12,
    )

    @model_validator(mode="after")
    def validate_affordance(self) -> "DirectionAffordanceV4":
        """Require non-contradictory physical affordance fields."""

        _require_unique(
            [str(value) for value in self.limiting_factors],
            "limiting_factors",
        )
        if self.affordance == "direct" and self.clearance == "insufficient":
            raise ValueError(
                "direct direction cannot have insufficient clearance"
            )
        if (
            self.affordance in {"conditional", "blocked", "forbidden"}
            and not self.limiting_factors
        ):
            raise ValueError(
                f"{self.affordance} direction requires limiting_factors"
            )
        return self


class LocalTraversabilityV4(BaseModel):
    """Collect physical affordances for the public probe vocabulary.

    Args:
        direction_affordances: At most one record per probe direction.
    """

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


class TaskDirectionV4(BaseModel):
    """Store the visible short-horizon route direction.

    Args:
        status: Whether a prompt-visible direction is usable.
        direction: Public moving direction when status is ``aligned``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: TaskDirectionStatusV4
    direction: ReasonCode | None = None

    @model_validator(mode="after")
    def validate_direction(self) -> "TaskDirectionV4":
        """Keep direction availability consistent with resolution status."""

        if self.direction == "stop":
            raise ValueError("task direction cannot use stop")
        if self.status == "aligned" and self.direction is None:
            raise ValueError("aligned task direction requires direction")
        if self.status != "aligned" and self.direction is not None:
            raise ValueError(
                f"{self.status} task direction cannot assert direction"
            )
        return self


class NavigationContextGroundingV4(BaseModel):
    """Store only the visible Stage-A route direction context.

    Args:
        task_direction: Nullable prompt-visible movement direction.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_direction: TaskDirectionV4


class InteractionConstraintV4(BaseModel):
    """Represent one high-level current behavior restriction.

    Args:
        kind: Open high-level behavior-constraint category.
        status: Whether the restriction is currently applicable or uncertain.
        effect: Required behavior response family.
        affected_direction_labels: Empty for a global constraint, otherwise
            the affected public action subset.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ReasonCode
    status: ConstraintStatusV4
    effect: ConstraintEffectV4
    affected_direction_labels: list[ReasonCode] = Field(
        default_factory=list,
        max_length=24,
    )

    @model_validator(mode="after")
    def validate_constraint(self) -> "InteractionConstraintV4":
        """Validate global stop and direction-subset avoid semantics."""

        _require_unique(
            [str(value) for value in self.affected_direction_labels],
            "affected_direction_labels",
        )
        if self.effect == "stop" and self.affected_direction_labels:
            raise ValueError(
                "stop constraint must be global in Stage-A"
            )
        if self.effect == "avoid" and not self.affected_direction_labels:
            raise ValueError(
                "avoid constraint requires affected_direction_labels"
            )
        return self


class InteractionConstraintGroundingV4(BaseModel):
    """Collect high-level behavior constraints without source binding.

    Args:
        constraints: Current restrictions that can alter the short action.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    constraints: list[InteractionConstraintV4] = Field(
        default_factory=list,
        max_length=24,
    )


class DecisionProposalV4(BaseModel):
    """Store the blind LVLM short-horizon action proposal.

    Args:
        selected_action_label: Immediate action from the public profile.
        speed: Coarse short-horizon speed.
        response: Coarse behavior response.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_action_label: ReasonCode
    speed: SpeedV4
    response: BehaviorResponseV4

    @model_validator(mode="after")
    def validate_action_tuple(self) -> "DecisionProposalV4":
        """Require a locally coherent action, speed, and response tuple."""

        _validate_action_tuple(
            selected_action_label=str(self.selected_action_label),
            speed=self.speed,
            response=self.response,
        )
        return self


class DecisionV4(DecisionProposalV4):
    """Canonical deterministic short-horizon action tuple."""


class ReasoningTraceV4(BaseModel):
    """Canonical six-section shared-input Reasoning V4.5 trace.

    Args:
        schema_version: Exact V4.5 semantic contract stamp.
        task_grounding: Deterministically normalized task.
        scene_grounding: Coarse visible scene summary.
        local_traversability: Physical direction affordances.
        navigation_context_grounding: Visible task direction.
        interaction_constraint_grounding: High-level behavior restrictions.
        decision: One consistent short-horizon action tuple.
    """

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


class ReasoningTargetV4(BaseModel):
    """Wrap the canonical trace used as a training audit sidecar.

    Args:
        reasoning: Exact six-section V4.5 trace.  V4.5 has no Finish object.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    reasoning: ReasoningTraceV4


def reasoning_v4_contract_schema_digest() -> str:
    """Return the semantic digest of the canonical V4.5 target schema.

    Returns:
        SHA256 over a normalized JSON Schema representation.
    """

    payload = _semantic_contract_schema(
        ReasoningTargetV4.model_json_schema()
    )
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _validate_action_tuple(
    *,
    selected_action_label: str,
    speed: SpeedV4,
    response: BehaviorResponseV4,
) -> None:
    """Validate the deterministic local action-tuple invariant."""

    is_stop = selected_action_label == "stop"
    if is_stop and (speed != "stop" or response != "stop"):
        raise ValueError(
            "stop action requires speed=stop and response=stop"
        )
    if not is_stop and speed == "stop":
        raise ValueError("moving action cannot use speed=stop")
    if not is_stop and response == "stop":
        raise ValueError("moving action cannot use response=stop")
    if response == "slow" and speed != "slow":
        raise ValueError("response=slow requires speed=slow")
    if response in {"yield", "detour"} and speed != "slow":
        raise ValueError(
            f"response={response} requires speed=slow"
        )


def _require_unique(values: list[str], field_name: str) -> None:
    """Reject duplicate values in one bounded list."""

    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} values must be unique")


def _semantic_contract_schema(value: Any) -> Any:
    """Remove unstable schema decoration before contract hashing."""

    if isinstance(value, dict):
        return {
            key: _semantic_contract_schema(item)
            for key, item in sorted(value.items())
            if key not in {"description", "title"}
        }
    if isinstance(value, list):
        return [_semantic_contract_schema(item) for item in value]
    return value


REASONING_V4_CONTRACT_SCHEMA_DIGEST = (
    reasoning_v4_contract_schema_digest()
)


__all__ = [
    "BehaviorResponseV4",
    "ClearanceV4",
    "ConstraintEffectV4",
    "ConstraintStatusV4",
    "DecisionProposalV4",
    "DecisionV4",
    "DirectionAffordanceStateV4",
    "DirectionAffordanceV4",
    "ExternalEvidenceRefV4",
    "InteractionConstraintGroundingV4",
    "InteractionConstraintV4",
    "LocalTraversabilityV4",
    "NavigationContextGroundingV4",
    "PhysicalLimitingFactorV4",
    "REASONING_TRACE_V4_SCHEMA_VERSION",
    "REASONING_TRAINING_VIEW_V4_SCHEMA_VERSION",
    "REASONING_V4_COMPACT_FULL_COT_RENDERER_ID",
    "REASONING_V4_CONTRACT_SCHEMA_DIGEST",
    "REASONING_V4_EVIDENCE_POLICY_VERSION",
    "REASONING_V4_PARSER_ID",
    "REASONING_V4_RENDERER_ID",
    "REASONING_V4_SECTION_DEFINITION_VERSION",
    "REASONING_V4_SECTION_NAMES",
    "REASONING_V4_VALIDATOR_ID",
    "REASONING_V4_WIRE_FORMAT_VERSION",
    "ReasoningTargetV4",
    "ReasoningTraceV4",
    "SceneGroundingV4",
    "SpeedV4",
    "TaskDirectionStatusV4",
    "TaskDirectionV4",
    "TaskGroundingV4",
    "reasoning_v4_contract_schema_digest",
]
