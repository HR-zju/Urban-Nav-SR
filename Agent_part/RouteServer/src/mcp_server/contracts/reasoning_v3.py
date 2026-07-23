"""Minimal, provenance-aware contracts for Reasoning V3.4.

V3.4 keeps the six-section Full-CoT wire shape, but removes researcher-authored
semantic choices that are neither dataset ground truth nor deterministic
projections of an existing label.  Open-world visual facts use ``ReasonCode``;
only protocol states and deterministic direction-label identity remain closed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mcp_server.contracts.reasoning import EvidenceRef, ReasonCode


REASONING_TRACE_V3_SCHEMA_VERSION = "urbannav_reasoning_trace_v3_4"
FINISH_V3_SCHEMA_VERSION = "urbannav_finish_v3_4"
REASONING_V3_SECTION_NAMES = (
    "TaskGrounding",
    "SceneGrounding",
    "LocalTraversability",
    "NavigationContextGrounding",
    "InteractionAndRuleGrounding",
    "Decision",
)

RequirementPolarityV3 = Literal["required", "preferred", "avoid", "forbidden"]
ReferentResolutionV3 = Literal["resolved", "unresolved", "ambiguous"]
LateralPositionV3 = Literal["left", "center", "right", "spanning", "unknown"]
DepthPositionV3 = Literal["near", "mid", "far", "unknown"]
TraversabilityStateV3 = Literal["direct", "conditional", "blocked", "forbidden", "unknown"]
ContextSourceTypeV3 = Literal["instruction", "bev", "goal", "waypoint", "route", "tool"]
ContextUsageV3 = Literal["used", "ignored"]
TemporalBasisV3 = Literal["single_frame", "multi_frame"]
DecisionRationaleStatusV3 = Literal["supported", "partial", "unsupported"]
LabelSourceKindV3 = Literal[
    "input_reference",
    "dataset_gt",
    "deterministic_adapter",
    "lvlm_weak",
]
LabelSupervisionV3 = Literal["strong", "weak", "audit_only", "disabled"]
LabelLossChannelV3 = Literal["none", "weak_lm", "direction_choice", "system1_action"]

LocalEntityRefV3 = Annotated[
    str,
    Field(min_length=8, max_length=96, pattern=r"^entity\.[a-z0-9_]+$"),
]
LocalControlRefV3 = Annotated[
    str,
    Field(min_length=9, max_length=96, pattern=r"^control\.[a-z0-9_]+$"),
]


class TaskRequirementV3(BaseModel):
    """One requirement copied from and grounded to the visible instruction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: ReasonCode
    polarity: RequirementPolarityV3
    text_span: str = Field(min_length=1, max_length=320)
    target_ref: EvidenceRef | None = None
    evidence_refs: list[EvidenceRef] = Field(min_length=1, max_length=8)


class TaskReferentV3(BaseModel):
    """One instruction phrase and its explicit grounding state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phrase: str = Field(min_length=1, max_length=320)
    resolution: ReferentResolutionV3
    resolved_ref: EvidenceRef | None = None
    evidence_refs: list[EvidenceRef] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def validate_resolution(self) -> "TaskReferentV3":
        if self.resolution == "resolved" and self.resolved_ref is None:
            raise ValueError("resolved task referent requires resolved_ref")
        if self.resolution != "resolved" and self.resolved_ref is not None:
            raise ValueError("non-resolved task referent cannot carry resolved_ref")
        return self


class TaskAmbiguityV3(BaseModel):
    """One material ambiguity, represented without forcing a taxonomy choice."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: ReasonCode
    text_span: str | None = Field(default=None, min_length=1, max_length=320)
    evidence_refs: list[EvidenceRef] = Field(min_length=1, max_length=8)


class TaskGroundingV3(BaseModel):
    """Ground only instruction facts that are visible to the student."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    applicable: bool
    instruction_ref: EvidenceRef | None = None
    goal_ref: EvidenceRef | None = None
    active_navigation_refs: list[EvidenceRef] = Field(default_factory=list, max_length=12)
    requirements: list[TaskRequirementV3] = Field(default_factory=list, max_length=8)
    referents: list[TaskReferentV3] = Field(default_factory=list, max_length=12)
    ambiguities: list[TaskAmbiguityV3] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_applicability(self) -> "TaskGroundingV3":
        if self.applicable and self.instruction_ref is None:
            raise ValueError("applicable task grounding requires instruction_ref")
        if not self.applicable and (
            self.goal_ref
            or self.active_navigation_refs
            or self.requirements
            or self.referents
            or self.ambiguities
        ):
            raise ValueError("non-applicable task grounding cannot contain active task claims")
        if len(self.active_navigation_refs) != len(set(self.active_navigation_refs)):
            raise ValueError("active_navigation_refs must be unique")
        return self


class SceneEntityV3(BaseModel):
    """One sample-local entity grounded in visible evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_ref: LocalEntityRefV3
    category: ReasonCode
    lateral_position: LateralPositionV3 = "unknown"
    depth_position: DepthPositionV3 = "unknown"
    evidence_refs: list[EvidenceRef] = Field(min_length=1, max_length=12)


class TrafficControlV3(BaseModel):
    """One sample-local traffic control grounded in visible evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    control_ref: LocalControlRefV3
    category: ReasonCode
    state: ReasonCode | None = None
    lateral_position: LateralPositionV3 = "unknown"
    evidence_refs: list[EvidenceRef] = Field(min_length=1, max_length=12)


class SceneGroundingV3(BaseModel):
    """Record only visible entities and traffic controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    applicable: bool
    entities: list[SceneEntityV3] = Field(default_factory=list, max_length=24)
    traffic_controls: list[TrafficControlV3] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def validate_scene(self) -> "SceneGroundingV3":
        if not self.applicable and (self.entities or self.traffic_controls):
            raise ValueError("non-applicable scene grounding cannot contain scene claims")
        entity_refs = [item.entity_ref for item in self.entities]
        control_refs = [item.control_ref for item in self.traffic_controls]
        if len(entity_refs) != len(set(entity_refs)):
            raise ValueError("scene entity refs must be unique")
        if len(control_refs) != len(set(control_refs)):
            raise ValueError("traffic control refs must be unique")
        return self


class TraversabilityRegionV3(BaseModel):
    """One visible region and its weak, evidence-backed traversability label."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    region_ref: EvidenceRef
    surface: ReasonCode
    state: TraversabilityStateV3
    blocker_refs: list[EvidenceRef] = Field(default_factory=list, max_length=8)
    reason: ReasonCode | None = None
    evidence_refs: list[EvidenceRef] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def validate_state(self) -> "TraversabilityRegionV3":
        if self.state == "direct" and self.blocker_refs:
            raise ValueError("direct region cannot cite blockers")
        if self.state in {"conditional", "blocked", "forbidden"} and not (
            self.blocker_refs or self.reason
        ):
            raise ValueError(f"{self.state} region requires blocker_refs or reason")
        return self


class LocalTraversabilityV3(BaseModel):
    """Collect visible region labels without route matching or GPS geometry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    applicable: bool
    regions: list[TraversabilityRegionV3] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def validate_regions(self) -> "LocalTraversabilityV3":
        if self.applicable != bool(self.regions):
            raise ValueError("traversability applicability must match region presence")
        refs = [item.region_ref for item in self.regions]
        if len(refs) != len(set(refs)):
            raise ValueError("each traversability region must occur exactly once")
        return self


class NavigationContextSourceSpecV3(BaseModel):
    """Deterministic identity of one model-visible navigation source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_type: ContextSourceTypeV3
    source_ref: EvidenceRef


class NavigationSourceGroundingV3(BaseModel):
    """LVLM grounding for one supplied navigation source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_type: ContextSourceTypeV3
    source_ref: EvidenceRef
    usage: ContextUsageV3
    relation: ReasonCode | None = None
    ignored_reason: ReasonCode | None = None
    evidence_refs: list[EvidenceRef] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def validate_grounding(self) -> "NavigationSourceGroundingV3":
        if self.source_ref not in self.evidence_refs:
            raise ValueError("navigation source grounding must cite source_ref")
        if self.usage == "ignored" and self.ignored_reason is None:
            raise ValueError("ignored navigation source requires ignored_reason")
        if self.usage == "used" and self.ignored_reason is not None:
            raise ValueError("used navigation source cannot carry ignored_reason")
        return self


class NavigationContextGroundingV3(BaseModel):
    """Collect independent source groundings without route/FPV fusion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    applicable: bool
    source_groundings: list[NavigationSourceGroundingV3] = Field(
        default_factory=list,
        max_length=12,
    )

    @model_validator(mode="after")
    def validate_sources(self) -> "NavigationContextGroundingV3":
        if self.applicable != bool(self.source_groundings):
            raise ValueError("navigation context applicability must match source groundings")
        refs = [item.source_ref for item in self.source_groundings]
        if len(refs) != len(set(refs)):
            raise ValueError("navigation context must ground each source exactly once")
        return self


class InteractionGroundingV3(BaseModel):
    """One LVLM-labelled actor/path relation, without tracker- or TTC-derived GT."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    actor_ref: LocalEntityRefV3
    relation: ReasonCode
    temporal_basis: TemporalBasisV3
    suggested_behavior: ReasonCode | None = None
    evidence_refs: list[EvidenceRef] = Field(min_length=1, max_length=16)


class TrafficRuleGroundingV3(BaseModel):
    """One LVLM-labelled control rule grounded in visible evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    control_ref: LocalControlRefV3
    rule: ReasonCode
    suggested_behavior: ReasonCode | None = None
    evidence_refs: list[EvidenceRef] = Field(min_length=1, max_length=16)


class InteractionAndRuleGroundingV3(BaseModel):
    """Collect only explicitly grounded interactions and traffic rules."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    applicable: bool
    interactions: list[InteractionGroundingV3] = Field(default_factory=list, max_length=12)
    traffic_rules: list[TrafficRuleGroundingV3] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_applicability(self) -> "InteractionAndRuleGroundingV3":
        if self.applicable != bool(self.interactions or self.traffic_rules):
            raise ValueError("interaction/rule applicability must match grounded relations")
        return self


class DecisionSupportV3(BaseModel):
    """One open reason code and the evidence supporting it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: ReasonCode
    evidence_refs: list[EvidenceRef] = Field(min_length=1, max_length=12)


class DecisionV3(BaseModel):
    """Canonical identity of the deterministic direction-choice target."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str = Field(min_length=1, max_length=128)
    profile_version: str = Field(min_length=1, max_length=64)
    label_index: int = Field(ge=0)
    label_name: ReasonCode
    choice: str | None = Field(default=None, min_length=1, max_length=32)
    angle_deg: float | None = Field(default=None, ge=-180.0, le=180.0)
    source: ReasonCode
    rationale_status: DecisionRationaleStatusV3
    supporting_reasons: list[DecisionSupportV3] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_rationale(self) -> "DecisionV3":
        if self.rationale_status in {"supported", "partial"} and not self.supporting_reasons:
            raise ValueError("supported or partial decision requires supporting_reasons")
        if self.rationale_status == "unsupported" and self.supporting_reasons:
            raise ValueError("unsupported decision cannot contain supporting_reasons")
        return self


class ReasoningTraceV3(BaseModel):
    """Canonical six-section Reasoning V3.4 trace."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[REASONING_TRACE_V3_SCHEMA_VERSION] = REASONING_TRACE_V3_SCHEMA_VERSION
    task_grounding: TaskGroundingV3
    scene_grounding: SceneGroundingV3
    local_traversability: LocalTraversabilityV3
    navigation_context_grounding: NavigationContextGroundingV3
    interaction_and_rule_grounding: InteractionAndRuleGroundingV3
    decision: DecisionV3

    @model_validator(mode="after")
    def validate_local_references(self) -> "ReasoningTraceV3":
        entity_refs = {item.entity_ref for item in self.scene_grounding.entities}
        control_refs = {item.control_ref for item in self.scene_grounding.traffic_controls}
        missing_entities = {
            item.actor_ref
            for item in self.interaction_and_rule_grounding.interactions
            if item.actor_ref not in entity_refs
        }
        missing_controls = {
            item.control_ref
            for item in self.interaction_and_rule_grounding.traffic_rules
            if item.control_ref not in control_refs
        }
        if missing_entities:
            raise ValueError("interaction actor_ref must resolve to a scene entity")
        if missing_controls:
            raise ValueError("traffic-rule control_ref must resolve to a scene traffic control")
        return self


class FinishV3(BaseModel):
    """Deterministic navigate-only projection of ``DecisionV3``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[FINISH_V3_SCHEMA_VERSION] = FINISH_V3_SCHEMA_VERSION
    decision_type: Literal["navigate"] = "navigate"
    profile_id: str = Field(min_length=1, max_length=128)
    profile_version: str = Field(min_length=1, max_length=64)
    label_index: int = Field(ge=0)
    label_name: ReasonCode
    choice: str | None = Field(default=None, min_length=1, max_length=32)
    angle_deg: float | None = Field(default=None, ge=-180.0, le=180.0)
    source: ReasonCode


class ReasoningTargetV3(BaseModel):
    """Pair a V3.4 trace with its lossless deterministic finish projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reasoning: ReasoningTraceV3
    finish: FinishV3

    @model_validator(mode="after")
    def validate_navigation_consistency(self) -> "ReasoningTargetV3":
        decision = self.reasoning.decision
        decision_identity = (
            decision.profile_id,
            decision.profile_version,
            decision.label_index,
            decision.label_name,
            decision.choice,
            decision.angle_deg,
            decision.source,
        )
        finish_identity = (
            self.finish.profile_id,
            self.finish.profile_version,
            self.finish.label_index,
            self.finish.label_name,
            self.finish.choice,
            self.finish.angle_deg,
            self.finish.source,
        )
        if decision_identity != finish_identity:
            raise ValueError("navigate finish must be an exact projection of Decision identity")
        return self


class LabelProvenanceEntryV3(BaseModel):
    """Declare source, supervision semantics, inferability, and loss routing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: ReasonCode
    source_kind: LabelSourceKindV3
    supervision: LabelSupervisionV3
    student_inferable: bool
    loss_channel: LabelLossChannelV3
    generator_version: str = Field(min_length=1, max_length=128)
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_loss_routing(self) -> "LabelProvenanceEntryV3":
        if self.supervision in {"audit_only", "disabled"} and self.loss_channel != "none":
            raise ValueError("audit-only or disabled labels must use loss_channel=none")
        if self.loss_channel == "weak_lm" and self.supervision != "weak":
            raise ValueError("weak_lm labels require weak supervision")
        if self.loss_channel in {"direction_choice", "system1_action"} and self.supervision != "strong":
            raise ValueError("deterministic action losses require strong supervision")
        return self


class SectionLossMaskV3(BaseModel):
    """Loss eligibility for each canonical six-section output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_grounding: bool = False
    scene_grounding: bool = False
    local_traversability: bool = False
    navigation_context_grounding: bool = False
    interaction_and_rule_grounding: bool = False
    decision: bool = False


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


def reasoning_v3_contract_schema_digest() -> str:
    """Return the semantic JSON-Schema digest for the V3.4 target."""

    schema = _semantic_contract_schema(ReasoningTargetV3.model_json_schema())
    payload = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


REASONING_V3_CONTRACT_SCHEMA_DIGEST = reasoning_v3_contract_schema_digest()


def reasoning_evidence_refs_v3(trace: ReasoningTraceV3) -> set[str]:
    """Collect every source, local object, and evidence reference cited by a trace."""

    refs: set[str] = set()
    task = trace.task_grounding
    if task.instruction_ref:
        refs.add(task.instruction_ref)
    if task.goal_ref:
        refs.add(task.goal_ref)
    refs.update(task.active_navigation_refs)
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

    for entity in trace.scene_grounding.entities:
        refs.add(entity.entity_ref)
        refs.update(entity.evidence_refs)
    for control in trace.scene_grounding.traffic_controls:
        refs.add(control.control_ref)
        refs.update(control.evidence_refs)
    for region in trace.local_traversability.regions:
        refs.add(region.region_ref)
        refs.update(region.blocker_refs)
        refs.update(region.evidence_refs)
    for source in trace.navigation_context_grounding.source_groundings:
        refs.add(source.source_ref)
        refs.update(source.evidence_refs)
    for interaction in trace.interaction_and_rule_grounding.interactions:
        refs.add(interaction.actor_ref)
        refs.update(interaction.evidence_refs)
    for rule in trace.interaction_and_rule_grounding.traffic_rules:
        refs.add(rule.control_ref)
        refs.update(rule.evidence_refs)
    for support in trace.decision.supporting_reasons:
        refs.update(support.evidence_refs)
    return refs


__all__ = [
    "ContextSourceTypeV3",
    "ContextUsageV3",
    "DecisionRationaleStatusV3",
    "DecisionSupportV3",
    "DecisionV3",
    "DepthPositionV3",
    "FINISH_V3_SCHEMA_VERSION",
    "FinishV3",
    "InteractionAndRuleGroundingV3",
    "InteractionGroundingV3",
    "LabelLossChannelV3",
    "LabelProvenanceEntryV3",
    "LabelSourceKindV3",
    "LabelSupervisionV3",
    "LateralPositionV3",
    "LocalTraversabilityV3",
    "NavigationContextGroundingV3",
    "NavigationContextSourceSpecV3",
    "NavigationSourceGroundingV3",
    "REASONING_TRACE_V3_SCHEMA_VERSION",
    "REASONING_V3_CONTRACT_SCHEMA_DIGEST",
    "REASONING_V3_SECTION_NAMES",
    "ReasoningTargetV3",
    "ReasoningTraceV3",
    "ReferentResolutionV3",
    "RequirementPolarityV3",
    "SceneEntityV3",
    "SceneGroundingV3",
    "SectionLossMaskV3",
    "TaskAmbiguityV3",
    "TaskGroundingV3",
    "TaskReferentV3",
    "TaskRequirementV3",
    "TemporalBasisV3",
    "TrafficControlV3",
    "TrafficRuleGroundingV3",
    "TraversabilityRegionV3",
    "TraversabilityStateV3",
    "reasoning_evidence_refs_v3",
    "reasoning_v3_contract_schema_digest",
]
