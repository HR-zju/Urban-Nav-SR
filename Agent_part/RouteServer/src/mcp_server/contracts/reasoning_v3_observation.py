"""Typed LVLM semantic contracts used before V3.3 Full-CoT composition."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mcp_server.contracts.reasoning import ConfidenceLevel, EvidenceRef
from mcp_server.contracts.reasoning_v3 import (
    CaptionRoleV3,
    DecisionRationaleStatusV3,
    DecisionSupportV3,
    InstructionTypeV3,
    InteractionAndRuleGroundingV3,
    LocalObjectiveV3,
    NavigationSourceGroundingV3,
    RejectedDecisionAlternativeV3,
    RiskLevelV3,
    SceneGroundingV3,
    TaskAmbiguityV3,
    TaskReferentV3,
    TaskRequirementV3,
    TraversabilityRegionV3,
    VisibilityV3,
)


REASONING_OBSERVATION_V3_SCHEMA_VERSION = "urbannav_reasoning_observation_v3_3"
DECISION_RATIONALE_OBSERVATION_V3_SCHEMA_VERSION = "urbannav_decision_rationale_observation_v3_3"


class TaskSemanticObservationV3(BaseModel):
    """Interpret task language without reading hidden trajectory or action labels.

    Args:
        applicable: Whether the supplied text expresses an active navigation task.
        instruction_type: Semantic task category inferred from the complete text.
        caption_role: Whether caption-like text acts as an instruction or scene description.
        inferred_objective: Short-horizon semantic objective inferred from visible inputs.
        objective_evidence_refs: Input references supporting the objective.
        requirements: Explicit requirements with source spans and polarity.
        referents: Instruction phrases grounded to visible or navigation references.
        ambiguities: Material unresolved language or grounding ambiguities.
        confidence: Confidence in the semantic interpretation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    applicable: bool
    instruction_type: InstructionTypeV3 = "unknown"
    caption_role: CaptionRoleV3 = "unknown"
    inferred_objective: LocalObjectiveV3 = "unknown"
    objective_evidence_refs: list[EvidenceRef] = Field(default_factory=list, max_length=12)
    requirements: list[TaskRequirementV3] = Field(default_factory=list, max_length=8)
    referents: list[TaskReferentV3] = Field(default_factory=list, max_length=12)
    ambiguities: list[TaskAmbiguityV3] = Field(default_factory=list, max_length=8)
    confidence: ConfidenceLevel = "unknown"

    @model_validator(mode="after")
    def validate_semantics(self) -> "TaskSemanticObservationV3":
        """Validate active, inactive, and ambiguous task interpretations.

        Returns:
            Validated task-semantic observation.

        Raises:
            ValueError: If applicability conflicts with semantic claims.
        """

        claims = (
            self.objective_evidence_refs,
            self.requirements,
            self.referents,
            self.ambiguities,
        )
        if not self.applicable:
            if any(claims):
                raise ValueError("non-applicable task semantics cannot contain claims")
            if (
                self.instruction_type != "unknown"
                or self.inferred_objective != "unknown"
            ):
                raise ValueError("non-applicable task semantics must remain unknown")
            if self.caption_role not in {"scene_description", "unknown"}:
                raise ValueError("non-applicable semantics permit only scene-description caption role")
            return self
        if self.inferred_objective == "unknown" and not self.ambiguities:
            raise ValueError("unknown inferred objective requires explicit ambiguity")
        if self.inferred_objective != "unknown" and not self.objective_evidence_refs:
            raise ValueError("known inferred objective requires evidence refs")
        return self


class LocalTraversabilityObservationV3(BaseModel):
    """Record FPV affordances without selecting the final execution corridor.

    Args:
        applicable: Whether current FPV evidence exists.
        regions: Unique semantic-geometric affordance labels.
        preferred_clear_corridor_ref: FPV-only preferred direct corridor.
        visibility: Current scene visibility.
        local_risk: Current local visual risk.
        confidence: Confidence in the region partition.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    applicable: bool
    regions: list[TraversabilityRegionV3] = Field(default_factory=list, max_length=16)
    preferred_clear_corridor_ref: EvidenceRef | None = None
    visibility: VisibilityV3 = "unknown"
    local_risk: RiskLevelV3 = "unknown"
    confidence: ConfidenceLevel = "unknown"

    @model_validator(mode="after")
    def validate_regions(self) -> "LocalTraversabilityObservationV3":
        """Validate region uniqueness and the FPV-only preference.

        Returns:
            Validated local observation.

        Raises:
            ValueError: If applicability, region states, or preference conflict.
        """

        if not self.applicable:
            if self.regions or self.preferred_clear_corridor_ref is not None:
                raise ValueError("non-applicable local observation cannot contain regions")
            if self.visibility != "unknown" or self.local_risk != "unknown" or self.confidence != "unknown":
                raise ValueError("non-applicable local observation must keep status unknown")
            return self
        if not self.regions:
            raise ValueError("applicable local observation requires regions")
        refs = [item.region_ref for item in self.regions]
        if len(refs) != len(set(refs)):
            raise ValueError("each local region must have exactly one affordance state")
        direct = {item.region_ref for item in self.regions if item.state == "direct"}
        if self.preferred_clear_corridor_ref is not None and self.preferred_clear_corridor_ref not in direct:
            raise ValueError("preferred clear corridor must be a direct region")
        return self


class NavigationContextObservationV3(BaseModel):
    """Record per-source navigation context before deterministic fusion.

    Args:
        applicable: Whether at least one route-specific context source exists.
        source_groundings: Exactly one relation for each supplied source.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    applicable: bool
    source_groundings: list[NavigationSourceGroundingV3] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_sources(self) -> "NavigationContextObservationV3":
        """Validate applicability and per-source uniqueness.

        Returns:
            Validated context observation.

        Raises:
            ValueError: If applicability or source coverage is malformed.
        """

        if self.applicable != bool(self.source_groundings):
            raise ValueError("navigation context applicability must match source groundings")
        refs = [item.source_ref for item in self.source_groundings]
        if len(refs) != len(set(refs)):
            raise ValueError("navigation context must ground each source exactly once")
        return self


class ReasoningObservationV3(BaseModel):
    """Collect LVLM-authored task and visual observations before composition.

    Args:
        schema_version: Fixed observation schema identifier.
        task_semantic: Instruction/caption semantics without hidden outcome labels.
        scene_grounding: Model-visible scene entities, controls, and context.
        local_traversability: FPV-first affordance partition.
        navigation_context_grounding: Per-source optional navigation relations.
        interaction_and_rule_grounding: Grounded actor/path and control/path rules.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[REASONING_OBSERVATION_V3_SCHEMA_VERSION] = REASONING_OBSERVATION_V3_SCHEMA_VERSION
    task_semantic: TaskSemanticObservationV3
    scene_grounding: SceneGroundingV3
    local_traversability: LocalTraversabilityObservationV3
    navigation_context_grounding: NavigationContextObservationV3
    interaction_and_rule_grounding: InteractionAndRuleGroundingV3


class DecisionRationaleObservationV3(BaseModel):
    """Explain a frozen decision outcome using only validated prior observations.

    Args:
        schema_version: Fixed V3.3 decision-rationale schema identifier.
        outcome_motion: Motion supplied by the deterministic outcome composer.
        status: Whether the frozen outcome has complete observable support.
        supporting_reasons: Evidence-backed relations supporting the outcome.
        rejected_alternatives: Evidence-backed local alternatives not selected.
        confidence: Confidence in the hindsight explanation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[DECISION_RATIONALE_OBSERVATION_V3_SCHEMA_VERSION] = (
        DECISION_RATIONALE_OBSERVATION_V3_SCHEMA_VERSION
    )
    outcome_motion: Literal["go_straight", "turn_left", "turn_right", "stop"]
    status: DecisionRationaleStatusV3
    supporting_reasons: list[DecisionSupportV3] = Field(default_factory=list, max_length=8)
    rejected_alternatives: list[RejectedDecisionAlternativeV3] = Field(default_factory=list, max_length=4)
    confidence: ConfidenceLevel = "unknown"

    @model_validator(mode="after")
    def validate_rationale(self) -> "DecisionRationaleObservationV3":
        """Validate support status without assessing cross-section evidence.

        Returns:
            Validated decision-rationale observation.

        Raises:
            ValueError: If status and support lists conflict.
        """

        if self.status in {"supported", "partial"} and not self.supporting_reasons:
            raise ValueError("supported or partial rationale requires supporting reasons")
        if self.status in {"unsupported", "not_applicable"} and self.supporting_reasons:
            raise ValueError("unsupported rationale cannot contain supporting reasons")
        if any(item.motion == self.outcome_motion for item in self.rejected_alternatives):
            raise ValueError("rejected alternative cannot repeat outcome motion")
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
