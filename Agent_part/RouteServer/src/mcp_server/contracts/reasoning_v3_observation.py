"""Two-pass LVLM observation contracts for Reasoning V3.4.

Pass 1 contains only task, scene, traversability, and navigation-source
observations.  Pass 2 receives a frozen deterministic direction label and may
add interaction/rule grounding plus evidence-backed support for that label.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mcp_server.contracts.reasoning import ReasonCode
from mcp_server.contracts.reasoning_v3 import (
    DecisionRationaleStatusV3,
    DecisionSupportV3,
    InteractionAndRuleGroundingV3,
    NavigationSourceGroundingV3,
    SceneGroundingV3,
    TaskAmbiguityV3,
    TaskReferentV3,
    TaskRequirementV3,
    TraversabilityRegionV3,
)


REASONING_OBSERVATION_V3_SCHEMA_VERSION = "urbannav_reasoning_observation_v3_4"
DECISION_RATIONALE_OBSERVATION_V3_SCHEMA_VERSION = (
    "urbannav_decision_rationale_observation_v3_4"
)


class TaskSemanticObservationV3(BaseModel):
    """Student-visible task semantics authored by the first LVLM pass."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    applicable: bool
    requirements: list[TaskRequirementV3] = Field(default_factory=list, max_length=8)
    referents: list[TaskReferentV3] = Field(default_factory=list, max_length=12)
    ambiguities: list[TaskAmbiguityV3] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_semantics(self) -> "TaskSemanticObservationV3":
        if not self.applicable and (self.requirements or self.referents or self.ambiguities):
            raise ValueError("non-applicable task semantics cannot contain task claims")
        return self


class LocalTraversabilityObservationV3(BaseModel):
    """FPV region labels authored by the first LVLM pass."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    applicable: bool
    regions: list[TraversabilityRegionV3] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def validate_regions(self) -> "LocalTraversabilityObservationV3":
        if self.applicable != bool(self.regions):
            raise ValueError("traversability applicability must match region presence")
        refs = [item.region_ref for item in self.regions]
        if len(refs) != len(set(refs)):
            raise ValueError("each traversability region must occur exactly once")
        return self


class NavigationContextObservationV3(BaseModel):
    """Independent source relations authored by the first LVLM pass."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    applicable: bool
    source_groundings: list[NavigationSourceGroundingV3] = Field(
        default_factory=list,
        max_length=12,
    )

    @model_validator(mode="after")
    def validate_sources(self) -> "NavigationContextObservationV3":
        if self.applicable != bool(self.source_groundings):
            raise ValueError("navigation context applicability must match source groundings")
        refs = [item.source_ref for item in self.source_groundings]
        if len(refs) != len(set(refs)):
            raise ValueError("navigation context must ground each source exactly once")
        return self


class ReasoningObservationV3(BaseModel):
    """Pass-1 observation; interaction and decision fields are intentionally absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[REASONING_OBSERVATION_V3_SCHEMA_VERSION] = (
        REASONING_OBSERVATION_V3_SCHEMA_VERSION
    )
    task_semantic: TaskSemanticObservationV3
    scene_grounding: SceneGroundingV3
    local_traversability: LocalTraversabilityObservationV3
    navigation_context_grounding: NavigationContextObservationV3


class DecisionRationaleObservationV3(BaseModel):
    """Pass-2 interaction/rule labels and support for a frozen direction target."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[DECISION_RATIONALE_OBSERVATION_V3_SCHEMA_VERSION] = (
        DECISION_RATIONALE_OBSERVATION_V3_SCHEMA_VERSION
    )
    interaction_and_rule_grounding: InteractionAndRuleGroundingV3
    direction_profile_id: str = Field(min_length=1, max_length=128)
    direction_profile_version: str = Field(min_length=1, max_length=64)
    direction_label_index: int = Field(ge=0)
    direction_label_name: ReasonCode
    direction_choice: str | None = Field(default=None, min_length=1, max_length=32)
    direction_angle_deg: float | None = Field(default=None, ge=-180.0, le=180.0)
    direction_source: ReasonCode
    rationale_status: DecisionRationaleStatusV3
    supporting_reasons: list[DecisionSupportV3] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_rationale(self) -> "DecisionRationaleObservationV3":
        if self.rationale_status in {"supported", "partial"} and not self.supporting_reasons:
            raise ValueError("supported or partial rationale requires supporting_reasons")
        if self.rationale_status == "unsupported" and self.supporting_reasons:
            raise ValueError("unsupported rationale cannot contain supporting_reasons")
        return self


__all__ = [
    "DECISION_RATIONALE_OBSERVATION_V3_SCHEMA_VERSION",
    "DecisionRationaleObservationV3",
    "LocalTraversabilityObservationV3",
    "NavigationContextObservationV3",
    "REASONING_OBSERVATION_V3_SCHEMA_VERSION",
    "ReasoningObservationV3",
    "TaskSemanticObservationV3",
]
