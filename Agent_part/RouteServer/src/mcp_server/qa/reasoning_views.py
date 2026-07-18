"""Project canonical reasoning traces into sparse, label-gated QA views.

The full CoT target and all section QA targets are views over the same
``ReasoningTraceV2`` object.  This prevents QA labels from drifting away from
the navigation decision while keeping weak social heuristics out of training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from mcp_server.agent.reasoning_text import render_reasoning_section_v2
from mcp_server.contracts.reasoning import (
    REASONING_TRACE_V2_SCHEMA_VERSION,
    ReasoningTraceV2,
    reasoning_evidence_refs,
)


REASONING_QA_SCHEMA_VERSION = "urbannav_reasoning_qa_v2"


@dataclass(frozen=True)
class ReasoningQATaskSpec:
    """Declare one QA projection and its evidence prerequisites.

    Args:
        task_type: Stable training task identifier.
        task_family: ``map_navigation`` or ``social_navigation``.
        section_name: V2 section rendered as the answer target.
        question: Deterministic user question.
        required_modalities_all: Modalities that must all be present.
        required_modalities_any: At least one of these modalities must exist.
        accepted_label_sources: Strong sources allowed to supervise the task.
    """

    task_type: str
    task_family: str
    section_name: str
    question: str
    required_modalities_all: frozenset[str] = frozenset()
    required_modalities_any: frozenset[str] = frozenset()
    accepted_label_sources: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ReasoningQALabelValidity:
    """Record whether one task has a trustworthy teacher label.

    Args:
        valid: Whether this QA target may enter LM supervision.
        confidence: Teacher confidence in ``[0, 1]``.
        label_source: Provenance such as route geometry or tracked agents.
        evidence_refs: Input-visible references used by this QA label.
        reason: Optional exclusion or review reason.
    """

    valid: bool
    confidence: float
    label_source: str
    evidence_refs: tuple[str, ...] = ()
    reason: str | None = None

    @classmethod
    def from_value(cls, value: Any) -> "ReasoningQALabelValidity":
        """Normalize a dictionary or existing validity record.

        Args:
            value: Mapping or ``ReasoningQALabelValidity`` instance.

        Returns:
            Normalized immutable validity record.

        Raises:
            TypeError: If ``value`` is not a supported representation.
            ValueError: If confidence is outside ``[0, 1]``.
        """

        if isinstance(value, cls):
            result = value
        elif isinstance(value, Mapping):
            result = cls(
                valid=bool(value.get("valid", False)),
                confidence=float(value.get("confidence", 0.0)),
                label_source=str(value.get("label_source") or value.get("source") or "unknown"),
                evidence_refs=tuple(str(item) for item in value.get("evidence_refs") or ()),
                reason=str(value.get("reason")) if value.get("reason") is not None else None,
            )
        else:
            raise TypeError("QA label validity must be a mapping or ReasoningQALabelValidity")
        if not 0.0 <= result.confidence <= 1.0:
            raise ValueError("QA label confidence must be in [0, 1]")
        return result


@dataclass(frozen=True)
class ReasoningQAView:
    """One ms-swift-ready LM-only QA view of a reasoning section.

    Args:
        sample_id: Unique QA row identifier.
        source_group_id: Id shared with full CoT and navigation views.
        task_type: QA task identifier.
        task_family: Map or social QA family.
        section_name: Projected V2 section.
        question: User prompt.
        teacher_answer: Canonical one-section assistant target.
        label_source: Strong-label provenance.
        confidence: Teacher confidence.
        evidence_refs: Input-visible references used for the label.
        metadata: Additional non-target audit fields.
    """

    sample_id: str
    source_group_id: str
    task_type: str
    task_family: str
    section_name: str
    question: str
    teacher_answer: str
    label_source: str
    confidence: float
    evidence_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible QA artifact.

        Returns:
            Dictionary with explicit LM/System1 loss masks.
        """

        payload = asdict(self)
        payload["evidence_refs"] = list(self.evidence_refs)
        payload.update(
            {
                "schema_version": REASONING_QA_SCHEMA_VERSION,
                "reasoning_schema_version": REASONING_TRACE_V2_SCHEMA_VERSION,
                # QA 只训练 section token，不触发连续动作监督。
                "loss_mask": {
                    "lm": True,
                    "system1_action": False,
                    "trajectory": False,
                    "segmentation": False,
                },
            }
        )
        return payload


REASONING_QA_TASK_REGISTRY: dict[str, ReasoningQATaskSpec] = {
    "instruction_to_map_selection": ReasoningQATaskSpec(
        task_type="instruction_to_map_selection",
        task_family="map_navigation",
        section_name="TaskGrounding",
        question="Ground the active instruction, goal, and route into the current local objective.",
        required_modalities_all=frozenset({"instruction"}),
        required_modalities_any=frozenset({"route", "bev", "tool_route"}),
        accepted_label_sources=frozenset({"route_geometry", "route_session", "human_annotation"}),
    ),
    "map_route_state_reading": ReasoningQATaskSpec(
        task_type="map_route_state_reading",
        task_family="map_navigation",
        section_name="EvidenceGrounding",
        question="Read the route, waypoint, map, and tool evidence relevant to this navigation tick.",
        required_modalities_any=frozenset({"route", "bev", "tool_route"}),
        accepted_label_sources=frozenset({"route_geometry", "route_session", "tool_response", "human_annotation"}),
    ),
    "route_to_fpv_grounding": ReasoningQATaskSpec(
        task_type="route_to_fpv_grounding",
        task_family="map_navigation",
        section_name="CrossModalAlignment",
        question="Align the active route direction with the visible FPV corridor.",
        required_modalities_all=frozenset({"fpv"}),
        required_modalities_any=frozenset({"route", "bev", "tool_route"}),
        accepted_label_sources=frozenset(
            {"calibrated_cross_view", "human_annotation", "validated_vlm_teacher"}
        ),
    ),
    "local_route_executability": ReasoningQATaskSpec(
        task_type="local_route_executability",
        task_family="map_navigation",
        section_name="CrossModalAlignment",
        question="Determine whether the route-aligned local corridor is currently executable.",
        required_modalities_all=frozenset({"fpv"}),
        required_modalities_any=frozenset({"route", "bev", "tool_route"}),
        accepted_label_sources=frozenset(
            {
                "calibrated_cross_view",
                "human_annotation",
                "validated_segmentation",
                "validated_vlm_teacher",
            }
        ),
    ),
    "control_path_applicability": ReasoningQATaskSpec(
        task_type="control_path_applicability",
        task_family="map_navigation",
        section_name="ConstraintApplicability",
        question="Determine which visible traffic controls apply to the selected pedestrian path.",
        required_modalities_all=frozenset({"fpv", "control", "selected_path"}),
        accepted_label_sources=frozenset({"tracked_control_region", "human_annotation"}),
    ),
    "social_traversability": ReasoningQATaskSpec(
        task_type="social_traversability",
        task_family="social_navigation",
        section_name="EvidenceGrounding",
        question="Identify the visible evidence that makes the local corridor socially traversable or blocked.",
        required_modalities_all=frozenset({"fpv"}),
        accepted_label_sources=frozenset(
            {"validated_segmentation", "human_annotation", "tracked_agents", "validated_vlm_teacher"}
        ),
    ),
    "social_agent_path_relation": ReasoningQATaskSpec(
        task_type="social_agent_path_relation",
        task_family="social_navigation",
        section_name="EvidenceGrounding",
        question="Ground tracked agents relative to the selected local path.",
        required_modalities_all=frozenset({"fpv", "track", "selected_path"}),
        accepted_label_sources=frozenset({"tracked_agents", "human_annotation"}),
    ),
    "social_constraint_applicability": ReasoningQATaskSpec(
        task_type="social_constraint_applicability",
        task_family="social_navigation",
        section_name="ConstraintApplicability",
        question="Determine which social constraints actually apply to the selected local path.",
        required_modalities_all=frozenset({"fpv", "track", "selected_path"}),
        accepted_label_sources=frozenset({"tracked_agents", "human_annotation"}),
    ),
    "social_trajectory_selection": ReasoningQATaskSpec(
        task_type="social_trajectory_selection",
        task_family="social_navigation",
        section_name="Decision",
        question="Choose the socially compliant local motion, speed, and fallback policy.",
        required_modalities_all=frozenset({"fpv", "selected_path"}),
        accepted_label_sources=frozenset({"demonstration", "human_annotation", "validated_controller"}),
    ),
}


def eligible_reasoning_qa_tasks(
    *,
    available_modalities: Iterable[str],
    label_validity: Mapping[str, Any],
    minimum_confidence: float = 0.7,
) -> list[tuple[ReasoningQATaskSpec, ReasoningQALabelValidity]]:
    """Return QA tasks whose modalities and labels satisfy export gates.

    Args:
        available_modalities: Input-visible modality names.
        label_validity: Per-task validity and provenance records.
        minimum_confidence: Minimum accepted teacher confidence.

    Returns:
        Registry-order list of eligible task specs and validity records.
    """

    modalities = {str(item) for item in available_modalities}
    eligible: list[tuple[ReasoningQATaskSpec, ReasoningQALabelValidity]] = []
    for task_type, spec in REASONING_QA_TASK_REGISTRY.items():
        if task_type not in label_validity:
            continue
        validity = ReasoningQALabelValidity.from_value(label_validity[task_type])
        if not validity.valid or validity.confidence < minimum_confidence:
            continue
        if not spec.required_modalities_all.issubset(modalities):
            continue
        if spec.required_modalities_any and not (spec.required_modalities_any & modalities):
            continue
        if spec.accepted_label_sources and validity.label_source not in spec.accepted_label_sources:
            continue
        eligible.append((spec, validity))
    return eligible


def build_reasoning_qa_views(
    trace: ReasoningTraceV2,
    *,
    source_group_id: str,
    available_modalities: Iterable[str],
    label_validity: Mapping[str, Any],
    minimum_confidence: float = 0.7,
) -> list[ReasoningQAView]:
    """Build sparse QA rows by projecting one canonical V2 trace.

    Args:
        trace: Canonical full reasoning target.
        source_group_id: Id shared with the source navigation row.
        available_modalities: Modalities visible in the corresponding prompt.
        label_validity: Explicit task-specific label gates.
        minimum_confidence: Minimum accepted teacher confidence.

    Returns:
        QA views with LM-only loss masks. No System1 labels are copied.
    """

    views: list[ReasoningQAView] = []
    resolvable_refs = reasoning_evidence_refs(trace)
    for spec, validity in eligible_reasoning_qa_tasks(
        available_modalities=available_modalities,
        label_validity=label_validity,
        minimum_confidence=minimum_confidence,
    ):
        if not validity.evidence_refs or not set(validity.evidence_refs).issubset(resolvable_refs):
            continue
        views.append(
            ReasoningQAView(
                sample_id=f"{source_group_id}:{spec.task_type}",
                source_group_id=source_group_id,
                task_type=spec.task_type,
                task_family=spec.task_family,
                section_name=spec.section_name,
                question=spec.question,
                teacher_answer=render_reasoning_section_v2(trace, spec.section_name),
                label_source=validity.label_source,
                confidence=validity.confidence,
                evidence_refs=validity.evidence_refs,
                metadata={"projected_from_full_cot": True},
            )
        )
    return views


__all__ = [
    "REASONING_QA_SCHEMA_VERSION",
    "REASONING_QA_TASK_REGISTRY",
    "ReasoningQALabelValidity",
    "ReasoningQATaskSpec",
    "ReasoningQAView",
    "build_reasoning_qa_views",
    "eligible_reasoning_qa_tasks",
]
