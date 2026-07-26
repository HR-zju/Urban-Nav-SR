"""Semantic validation for shared-input UrbanNav Reasoning V4."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from pydantic import BaseModel

from mcp_server.agent.reasoning_v4_text import parse_reasoning_target_v4
from mcp_server.contracts.reasoning_dataset_v4 import NavObservationV4
from mcp_server.contracts.reasoning_v4 import (
    DecisionV4,
    InteractionConstraintGroundingV4,
    LocalTraversabilityV4,
    NavigationContextGroundingV4,
    ReasoningTargetV4,
    STAGE_A_DECISION_TYPES_V4,
    SceneGroundingV4,
    TaskGroundingV4,
    raw_evidence_refs_v4,
)


@dataclass(frozen=True)
class ReasoningValidationResultV4:
    """Return hard failures separately from soft cross-section findings.

    Args:
        errors: Hard schema, evidence, safety, or executability violations.
        warnings: Non-fatal missing semantic links or coverage findings.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Return whether no hard validation error occurred."""

        return not self.errors


def validate_reasoning_section_v4(
    section_name: str,
    section: BaseModel,
    *,
    observation: NavObservationV4,
) -> ReasoningValidationResultV4:
    """Validate one independently generated grounding or Decision section.

    Args:
        section_name: Canonical V4 section tag.
        section: Typed section object.
        observation: Complete prompt-visible input shared by all sections.

    Returns:
        Section-local errors and warnings.

    Raises:
        ValueError: If ``section_name`` and the typed object disagree.
    """

    expected_types: dict[str, type[BaseModel]] = {
        "TaskGrounding": TaskGroundingV4,
        "SceneGrounding": SceneGroundingV4,
        "LocalTraversability": LocalTraversabilityV4,
        "NavigationContextGrounding": NavigationContextGroundingV4,
        "InteractionConstraintGrounding": InteractionConstraintGroundingV4,
        "Decision": DecisionV4,
    }
    expected = expected_types.get(section_name)
    if expected is None or not isinstance(section, expected):
        raise ValueError(
            f"section/type mismatch: {section_name}/{type(section).__name__}"
        )
    errors: list[str] = []
    warnings: list[str] = []
    raw_refs = _raw_refs_from_value(section.model_dump(mode="json"))
    catalog_refs = {entry.ref for entry in observation.evidence_catalog}
    unknown = sorted(raw_refs - catalog_refs)
    if unknown:
        errors.append(f"unknown_raw_evidence_refs:{unknown}")
    if (
        section_name == "Decision"
        and isinstance(section, DecisionV4)
        and section.decision_type not in STAGE_A_DECISION_TYPES_V4
    ):
        errors.append(
            "stage_a_decision_type_not_allowed:"
            f"{section.decision_type}"
        )
    _validate_visual_evidence_roles(
        section_name,
        section,
        observation=observation,
        errors=errors,
        warnings=warnings,
    )
    return ReasoningValidationResultV4(
        errors=_unique(errors),
        warnings=_unique(warnings),
    )


def validate_reasoning_target_v4(
    target: ReasoningTargetV4,
    *,
    observation: NavObservationV4,
    assistant_text: str | None = None,
) -> ReasoningValidationResultV4:
    """Validate raw evidence closure, weak links, and Decision executability.

    Args:
        target: Fully assembled V4 target.
        observation: Prompt-visible input used to produce every section.
        assistant_text: Optional rendered text for strict round-trip validation.

    Returns:
        Hard failures and soft cross-section warnings.
    """

    errors: list[str] = []
    warnings: list[str] = []
    trace = target.reasoning
    catalog = {entry.ref: entry for entry in observation.evidence_catalog}
    unknown_raw = sorted(raw_evidence_refs_v4(trace) - set(catalog))
    if unknown_raw:
        errors.append(f"unknown_raw_evidence_refs:{unknown_raw}")

    for section_name, section in (
        ("TaskGrounding", trace.task_grounding),
        ("SceneGrounding", trace.scene_grounding),
        ("LocalTraversability", trace.local_traversability),
        (
            "NavigationContextGrounding",
            trace.navigation_context_grounding,
        ),
        (
            "InteractionConstraintGrounding",
            trace.interaction_constraint_grounding,
        ),
        ("Decision", trace.decision),
    ):
        result = validate_reasoning_section_v4(
            section_name,
            section,
            observation=observation,
        )
        errors.extend(f"{section_name}:{item}" for item in result.errors)
        warnings.extend(
            f"{section_name}:{item}" for item in result.warnings
        )

    entity_refs = {
        item.entity_ref for item in trace.scene_grounding.entities
    }
    control_refs = {
        item.control_ref
        for item in trace.scene_grounding.traffic_controls
    }
    region_by_ref = {
        item.region_ref: item
        for item in trace.local_traversability.regions
    }
    requirement_refs = {
        item.requirement_ref
        for item in trace.task_grounding.requirements
    }
    ambiguity_refs = {
        item.ambiguity_ref for item in trace.task_grounding.ambiguities
    }
    navigation = trace.navigation_context_grounding
    goal_is_absent = trace.task_grounding.goal.goal_type == "none"
    if goal_is_absent and navigation.goal_alignment is not None:
        errors.append("absent_task_goal_cannot_have_goal_alignment")
    if not goal_is_absent and navigation.goal_alignment is None:
        errors.append("explicit_task_goal_requires_goal_alignment")

    for region in trace.local_traversability.regions:
        missing = sorted(set(region.related_entity_refs) - entity_refs)
        if missing:
            warnings.append(
                f"cross_section_missing_region_entities:{region.region_ref}:{missing}"
            )

    if navigation.goal_alignment is not None:
        alignment = navigation.goal_alignment
        semantic_targets = {
            ref
            for ref in [
                alignment.target_ref,
                *alignment.candidate_target_refs,
            ]
            if isinstance(ref, str)
            and ref.startswith(("entity.", "control."))
        }
        known_targets = entity_refs | control_refs
        missing = sorted(semantic_targets - known_targets)
        if missing:
            warnings.append(
                f"cross_section_missing_goal_targets:{missing}"
            )
        unresolved_regions = sorted(
            set(alignment.aligned_region_refs) - set(region_by_ref)
        )
        if unresolved_regions:
            warnings.append(
                f"cross_section_missing_goal_regions:{unresolved_regions}"
            )
    for alignment in navigation.source_alignments:
        unresolved = sorted(
            set(alignment.aligned_region_refs) - set(region_by_ref)
        )
        if unresolved:
            warnings.append(
                f"cross_section_missing_alignment_regions:"
                f"{alignment.alignment_ref}:{unresolved}"
            )

    candidate_refs = set(navigation.task_direction.candidate_region_refs)
    alignment_refs = {
        item.alignment_ref for item in navigation.source_alignments
    }
    unresolved_candidates = sorted(candidate_refs - set(region_by_ref))
    if unresolved_candidates:
        warnings.append(
            f"cross_section_missing_candidate_regions:{unresolved_candidates}"
        )

    constraint_by_ref = {
        item.constraint_ref: item
        for item in trace.interaction_constraint_grounding.constraints
    }
    known_constraint_sources = (
        entity_refs | control_refs | requirement_refs | {"task.goal"}
    )
    for constraint in constraint_by_ref.values():
        semantic_sources = {
            ref
            for ref in constraint.source_refs
            if not ref.startswith(
                ("instruction.", "fpv.", "bev.", "goal.", "waypoint.")
            )
        }
        missing_sources = sorted(
            semantic_sources - known_constraint_sources
        )
        if missing_sources:
            warnings.append(
                f"cross_section_missing_constraint_sources:"
                f"{constraint.constraint_ref}:{missing_sources}"
            )
        non_candidate = sorted(
            set(constraint.applies_to_region_refs) - candidate_refs
        )
        if non_candidate:
            warnings.append(
                f"constraint_regions_not_in_task_candidates:"
                f"{constraint.constraint_ref}:{non_candidate}"
            )

    decision = trace.decision
    if decision.decision_type == "navigate":
        selected = decision.selected_region_ref
        if selected not in region_by_ref:
            errors.append(f"selected_region_does_not_resolve:{selected}")
        else:
            region = region_by_ref[selected]
            if region.state in {"blocked", "forbidden", "unknown"}:
                errors.append(
                    f"selected_region_is_not_executable:{selected}:{region.state}"
                )
        if selected not in candidate_refs:
            errors.append(
                f"selected_region_not_in_task_candidates:{selected}"
            )
    known_support_refs = {
        "task.goal",
        "nav.task_direction",
        *entity_refs,
        *control_refs,
        *region_by_ref,
        *alignment_refs,
        *requirement_refs,
        *ambiguity_refs,
        *constraint_by_ref,
    }
    missing_support = sorted(
        set(decision.support_refs) - known_support_refs
    )
    if missing_support:
        errors.append(
            f"decision_support_refs_do_not_resolve:{missing_support}"
        )
    _validate_decision_constraints(
        decision,
        constraint_by_ref=constraint_by_ref,
        errors=errors,
    )

    if assistant_text is not None:
        leaked = _forbidden_text_tokens(assistant_text)
        if leaked:
            errors.append(f"forbidden_fields_in_assistant:{leaked}")
        parsed = parse_reasoning_target_v4(assistant_text)
        if not parsed.valid or parsed.target is None:
            errors.extend(
                f"assistant_text:{item}" for item in parsed.errors
            )
        elif parsed.target != target:
            errors.append("assistant_text_target_roundtrip_mismatch")
    return ReasoningValidationResultV4(
        errors=_unique(errors),
        warnings=_unique(warnings),
    )


def _validate_visual_evidence_roles(
    section_name: str,
    section: BaseModel,
    *,
    observation: NavObservationV4,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Apply input-dependent temporal and evidence-authority rules."""

    del warnings
    if section_name != "InteractionConstraintGrounding":
        return
    constraints = getattr(section, "constraints", [])
    image_by_ref = {
        role: path
        for path, role in zip(
            observation.images,
            observation.image_roles,
        )
        if role.startswith("fpv.")
    }
    for constraint in constraints:
        if constraint.relation not in {
            "crosses_candidate_path",
            "approaches_candidate_path",
        }:
            continue
        fpv_refs = {
            ref
            for ref in constraint.evidence_refs
            if ref.startswith("fpv.")
        }
        distinct_images = {
            image_by_ref[ref]
            for ref in fpv_refs
            if ref in image_by_ref
        }
        if (
            not observation.temporal_evidence_eligible
            or len(distinct_images) < 2
        ):
            errors.append(
                f"dynamic_constraint_requires_distinct_frames:"
                f"{constraint.constraint_ref}"
            )


def _validate_decision_constraints(
    decision: DecisionV4,
    *,
    constraint_by_ref: dict[str, Any],
    errors: list[str],
) -> None:
    """Reject decisions that contradict applicable supported constraints."""

    selected = decision.selected_region_ref
    for ref in decision.support_refs:
        constraint = constraint_by_ref.get(ref)
        if constraint is None or constraint.status != "applicable":
            continue
        if (
            selected is not None
            and selected not in constraint.applies_to_region_refs
        ):
            errors.append(
                "decision_support_constraint_not_applicable_to_selected_region:"
                f"{constraint.constraint_ref}:{selected}"
            )
            continue
        effect = constraint.effect
        if effect in {"stop", "wait"} and decision.speed != "stop":
            errors.append(
                f"decision_violates_{effect}_constraint:{constraint.constraint_ref}"
            )
        if effect == "slow" and decision.speed == "normal":
            errors.append(
                f"decision_violates_slow_constraint:{constraint.constraint_ref}"
            )
        if effect == "yield" and decision.social_action not in {
            "yield",
            "wait",
        }:
            errors.append(
                f"decision_violates_yield_constraint:{constraint.constraint_ref}"
            )
        if effect == "detour" and decision.social_action != "detour":
            errors.append(
                f"decision_violates_detour_constraint:{constraint.constraint_ref}"
            )


def _raw_refs_from_value(value: Any) -> set[str]:
    """Collect raw evidence-shaped strings from an arbitrary JSON value."""

    refs: set[str] = set()
    if isinstance(value, dict):
        for item in value.values():
            refs.update(_raw_refs_from_value(item))
    elif isinstance(value, list):
        for item in value:
            refs.update(_raw_refs_from_value(item))
    elif isinstance(value, str) and value.startswith(
        ("instruction.", "fpv.", "bev.", "goal.", "waypoint.")
    ):
        refs.add(value)
    return refs


def _forbidden_text_tokens(text: str) -> list[str]:
    """Return private or unsupported field names leaked into assistant text."""

    lowered = text.lower()
    forbidden = (
        "active_route_state",
        "arrival_gt",
        "bbox",
        "calibration",
        "confidence",
        "gnss",
        "goal_distance",
        "gps",
        "homography",
        "mask",
        "pose",
        "projected_gt",
        "route_theta",
        "runtime_task_phase",
        "track_id",
        "trajectory_ego_t",
        "trajectory_fpv",
        "ttc",
    )
    return [token for token in forbidden if token in lowered]


def _unique(values: Iterable[str]) -> list[str]:
    """Return stable unique diagnostics."""

    return list(dict.fromkeys(str(value) for value in values))
