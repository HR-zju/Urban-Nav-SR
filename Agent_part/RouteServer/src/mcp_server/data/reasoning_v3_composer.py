"""Compose Reasoning V3.4 without inventing action labels.

The composer copies one approved, profile-stamped direction record from the
teacher-private source sidecar.  LVLM weak observations and hindsight
relations may explain that record, but they never select, alter, re-bin, or
replace it.  Samples without such a record remain observation-only artifacts
and cannot be composed into a Full-CoT target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mcp_server.contracts.reasoning_dataset_v3 import ReasoningSourceV3
from mcp_server.contracts.reasoning_v3 import (
    DecisionV3,
    FinishV3,
    LabelProvenanceEntryV3,
    LocalTraversabilityV3,
    NavigationContextGroundingV3,
    ReasoningTargetV3,
    ReasoningTraceV3,
    SceneGroundingV3,
    SectionLossMaskV3,
    TaskGroundingV3,
)
from mcp_server.contracts.reasoning_v3_observation import (
    DecisionRationaleObservationV3,
    ReasoningObservationV3,
    TaskSemanticObservationV3,
)
from mcp_server.data.reasoning_v3_conditions import build_navigation_context_sources_v3
from mcp_server.data.reasoning_v3_labels import REASONING_V3_FIXED_SUPERVISION_VERSION


REASONING_V3_COMPOSER_VERSION = "reasoning_v3_4_minimal_composer_v1"


@dataclass(frozen=True)
class DecisionOutcomeV3:
    """Immutable copy of one approved canonical direction record."""

    profile_id: str
    profile_version: str
    label_index: int
    label_name: str
    source: str
    choice: str | None = None
    angle_deg: float | None = None

    def model_dump(self) -> dict[str, Any]:
        """Return a JSON-compatible provider prompt payload."""

        payload: dict[str, Any] = {
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "label_index": self.label_index,
            "label_name": self.label_name,
            "source": self.source,
        }
        if self.choice is not None:
            payload["choice"] = self.choice
        if self.angle_deg is not None:
            payload["angle_deg"] = self.angle_deg
        return payload


@dataclass(frozen=True)
class ReasoningCompositionResultV3:
    """Composed target plus field provenance and candidate section mask."""

    target: ReasoningTargetV3
    label_provenance: dict[str, LabelProvenanceEntryV3]
    section_loss_mask: SectionLossMaskV3
    warnings: list[str] = field(default_factory=list)


def derive_decision_outcome_v3(
    source: ReasoningSourceV3,
    observation: ReasoningObservationV3 | None = None,
) -> DecisionOutcomeV3:
    """Return the source's canonical direction without using observations."""

    del observation
    fixed = _fixed_labels(source)
    raw = fixed.get("canonical_direction")
    if not isinstance(raw, dict):
        raise ValueError("canonical_direction_required_for_full_cot")
    return DecisionOutcomeV3(
        profile_id=str(raw["profile_id"]),
        profile_version=str(raw["profile_version"]),
        label_index=int(raw["label_index"]),
        label_name=str(raw["name"]),
        choice=str(raw["choice"]) if raw.get("choice") is not None else None,
        angle_deg=float(raw["angle_deg"]) if raw.get("angle_deg") is not None else None,
        source=str(raw["source"]),
    )


def compose_reasoning_target_v3(
    source: ReasoningSourceV3,
    observation: ReasoningObservationV3,
    decision_rationale: DecisionRationaleObservationV3,
) -> ReasoningCompositionResultV3:
    """Compose a Full-CoT target only when canonical direction GT exists."""

    outcome = derive_decision_outcome_v3(source)
    _validate_rationale_identity(decision_rationale, outcome)
    task = _compose_task(source, observation.task_semantic)
    decision = DecisionV3(
        profile_id=outcome.profile_id,
        profile_version=outcome.profile_version,
        label_index=outcome.label_index,
        label_name=outcome.label_name,
        choice=outcome.choice,
        angle_deg=outcome.angle_deg,
        source=outcome.source,
        rationale_status=decision_rationale.rationale_status,
        supporting_reasons=decision_rationale.supporting_reasons,
    )
    trace = ReasoningTraceV3(
        task_grounding=task,
        scene_grounding=observation.scene_grounding,
        local_traversability=LocalTraversabilityV3.model_validate(
            observation.local_traversability.model_dump(mode="python")
        ),
        navigation_context_grounding=NavigationContextGroundingV3.model_validate(
            observation.navigation_context_grounding.model_dump(mode="python")
        ),
        interaction_and_rule_grounding=decision_rationale.interaction_and_rule_grounding,
        decision=decision,
    )
    finish = FinishV3(
        profile_id=outcome.profile_id,
        profile_version=outcome.profile_version,
        label_index=outcome.label_index,
        label_name=outcome.label_name,
        choice=outcome.choice,
        angle_deg=outcome.angle_deg,
        source=outcome.source,
    )
    target = ReasoningTargetV3(reasoning=trace, finish=finish)
    provenance = _composition_provenance(
        source,
        target=target,
    )
    return ReasoningCompositionResultV3(
        target=target,
        label_provenance=provenance,
        section_loss_mask=_candidate_section_mask(target),
    )


def _compose_task(
    source: ReasoningSourceV3,
    semantic: TaskSemanticObservationV3,
) -> TaskGroundingV3:
    """Attach deterministic source identities to LVLM task weak labels."""

    catalog_refs = {entry.ref for entry in source.model_input.evidence_catalog}
    instruction_ref = "instruction.active" if "instruction.active" in catalog_refs else None
    applicable = bool(semantic.applicable and instruction_ref)
    if not applicable:
        return TaskGroundingV3(applicable=False, instruction_ref=instruction_ref)
    context_refs = [
        item.source_ref
        for item in build_navigation_context_sources_v3(source)
        if item.source_ref != instruction_ref
    ]
    return TaskGroundingV3(
        applicable=True,
        instruction_ref=instruction_ref,
        goal_ref="goal.active" if "goal.active" in catalog_refs else None,
        active_navigation_refs=_unique(context_refs),
        requirements=semantic.requirements,
        referents=semantic.referents,
        ambiguities=semantic.ambiguities,
    )


def _fixed_labels(source: ReasoningSourceV3) -> dict[str, Any]:
    version = source.teacher_supervision_context.get("fixed_supervision_version")
    if version != REASONING_V3_FIXED_SUPERVISION_VERSION:
        raise ValueError(f"source_requires_{REASONING_V3_FIXED_SUPERVISION_VERSION}")
    raw = source.teacher_supervision_context.get("fixed_labels")
    return dict(raw) if isinstance(raw, dict) else {}


def _validate_rationale_identity(
    rationale: DecisionRationaleObservationV3,
    outcome: DecisionOutcomeV3,
) -> None:
    actual = (
        rationale.direction_profile_id,
        rationale.direction_profile_version,
        rationale.direction_label_index,
        rationale.direction_label_name,
        rationale.direction_choice,
        rationale.direction_angle_deg,
        rationale.direction_source,
    )
    expected = (
        outcome.profile_id,
        outcome.profile_version,
        outcome.label_index,
        outcome.label_name,
        outcome.choice,
        outcome.angle_deg,
        outcome.source,
    )
    if actual != expected:
        raise ValueError(
            "decision_rationale_direction_identity_mismatch:"
            f"expected={expected!r}:actual={actual!r}"
        )


def _composition_provenance(
    source: ReasoningSourceV3,
    *,
    target: ReasoningTargetV3,
) -> dict[str, LabelProvenanceEntryV3]:
    """Declare field groups without confidence-like synthetic scores."""

    result = dict(source.label_provenance)
    weak_fields = {
        "task_grounding": target.reasoning.task_grounding.applicable,
        "scene_grounding": target.reasoning.scene_grounding.applicable,
        "local_traversability": target.reasoning.local_traversability.applicable,
        "navigation_context_grounding": target.reasoning.navigation_context_grounding.applicable,
        "interaction_and_rule_grounding":
            target.reasoning.interaction_and_rule_grounding.applicable,
        "decision.rationale": True,
    }
    for field_path, applicable in weak_fields.items():
        if not applicable:
            continue
        result[field_path] = LabelProvenanceEntryV3(
            source="validated_lvlm_weak_label",
            source_kind="lvlm_weak",
            supervision="weak",
            student_inferable=True,
            # V3.4 annotation is audit-only by default.  An explicit future
            # weak-distillation profile may remap this to weak_lm after review.
            loss_channel="none",
            generator_version=REASONING_V3_COMPOSER_VERSION,
            details={"field_path": field_path},
        )
    return result


def _candidate_section_mask(target: ReasoningTargetV3) -> SectionLossMaskV3:
    """Report weak-section availability without enabling Decision LM loss."""

    trace = target.reasoning
    return SectionLossMaskV3(
        task_grounding=trace.task_grounding.applicable,
        scene_grounding=trace.scene_grounding.applicable,
        local_traversability=trace.local_traversability.applicable,
        navigation_context_grounding=trace.navigation_context_grounding.applicable,
        interaction_and_rule_grounding=trace.interaction_and_rule_grounding.applicable,
        decision=False,
    )


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


__all__ = [
    "DecisionOutcomeV3",
    "REASONING_V3_COMPOSER_VERSION",
    "ReasoningCompositionResultV3",
    "compose_reasoning_target_v3",
    "derive_decision_outcome_v3",
]
