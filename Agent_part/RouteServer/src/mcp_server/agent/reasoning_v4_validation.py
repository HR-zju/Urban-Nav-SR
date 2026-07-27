"""Semantic validation for shared-input UrbanNav Reasoning V4.2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from pydantic import BaseModel

from mcp_server.agent.reasoning_v4_text import (
    parse_reasoning_target_v4,
)
from mcp_server.contracts.direction_profile import (
    coarse_motion_for_direction_label,
    direction_labels_are_intent_compatible,
)
from mcp_server.contracts.reasoning_dataset_v4 import NavObservationV4
from mcp_server.contracts.reasoning_v4 import (
    DecisionProposalV4,
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
    """Return hard failures separately from non-fatal findings."""

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
    """Validate one independent grounding or Decision proposal."""

    expected_types: dict[str, tuple[type[BaseModel], ...]] = {
        "TaskGrounding": (TaskGroundingV4,),
        "SceneGrounding": (SceneGroundingV4,),
        "LocalTraversability": (LocalTraversabilityV4,),
        "NavigationContextGrounding": (
            NavigationContextGroundingV4,
        ),
        "InteractionConstraintGrounding": (
            InteractionConstraintGroundingV4,
        ),
        "Decision": (DecisionProposalV4, DecisionV4),
    }
    expected = expected_types.get(section_name)
    if expected is None or not isinstance(section, expected):
        raise ValueError(
            f"section/type mismatch: "
            f"{section_name}/{type(section).__name__}"
        )
    errors: list[str] = []
    raw_refs = _raw_refs_from_value(
        section.model_dump(mode="json")
    )
    catalog_refs = {
        entry.ref for entry in observation.evidence_catalog
    }
    unknown = sorted(raw_refs - catalog_refs)
    if unknown:
        errors.append(f"unknown_raw_evidence_refs:{unknown}")
    if (
        section_name == "Decision"
        and getattr(section, "decision_type", None)
        not in STAGE_A_DECISION_TYPES_V4
    ):
        errors.append(
            "stage_a_decision_type_not_allowed:"
            f"{getattr(section, 'decision_type', None)}"
        )
    _validate_direction_membership(
        section_name,
        section,
        observation=observation,
        errors=errors,
    )
    _validate_temporal_evidence(
        section_name,
        section,
        observation=observation,
        errors=errors,
    )
    return ReasoningValidationResultV4(
        errors=_unique(errors),
        warnings=[],
    )


def validate_reasoning_target_v4(
    target: ReasoningTargetV4,
    *,
    observation: NavObservationV4,
    assistant_text: str | None = None,
) -> ReasoningValidationResultV4:
    """Validate evidence closure, public labels, and final action semantics."""

    errors: list[str] = []
    warnings: list[str] = []
    trace = target.reasoning
    catalog_refs = {
        entry.ref for entry in observation.evidence_catalog
    }
    unknown_raw = sorted(
        raw_evidence_refs_v4(trace) - catalog_refs
    )
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
        checked = validate_reasoning_section_v4(
            section_name,
            section,
            observation=observation,
        )
        errors.extend(
            f"{section_name}:{item}"
            for item in checked.errors
        )
        warnings.extend(
            f"{section_name}:{item}"
            for item in checked.warnings
        )

    goal_absent = trace.task_grounding.goal.goal_type == "none"
    goal_alignment = (
        trace.navigation_context_grounding.goal_alignment
    )
    if goal_absent and goal_alignment is not None:
        errors.append(
            "absent_task_goal_cannot_have_goal_alignment"
        )
    if not goal_absent and goal_alignment is None:
        errors.append(
            "explicit_task_goal_requires_goal_alignment"
        )

    known_support_refs = _known_support_refs(target)
    missing_support = sorted(
        set(trace.decision.support_refs) - known_support_refs
    )
    if missing_support:
        errors.append(
            "decision_support_refs_do_not_resolve:"
            f"{missing_support}"
        )

    allowed = {
        str(value)
        for value in observation.direction_output_space.allowed_labels
    }
    selected_action = str(trace.decision.selected_action_label)
    if selected_action not in allowed:
        errors.append(
            f"selected_action_outside_output_space:{selected_action}"
        )
    else:
        expected_motion = coarse_motion_for_direction_label(
            observation.direction_output_space.profile_id,
            selected_action,
        )
        if trace.decision.motion != expected_motion:
            errors.append(
                "profile_motion_mismatch:"
                f"label={selected_action}, "
                f"motion={trace.decision.motion}, "
                f"expected={expected_motion}"
            )
        expected_type = (
            "stop" if expected_motion == "stop" else "navigate"
        )
        if trace.decision.decision_type != expected_type:
            errors.append(
                "selected_action_decision_type_mismatch:"
                f"label={selected_action}, "
                f"decision={trace.decision.decision_type}"
            )
        _validate_selected_direction_grounding(
            target,
            direction_label=selected_action,
            errors=errors,
        )
        _validate_constraint_effects(
            target,
            selected_action_label=selected_action,
            errors=errors,
        )
        if selected_action == "stop" and not _stop_action_has_support(
            target
        ):
            errors.append("stop_action_lacks_visible_support")

    intent = trace.decision.navigation_intent_direction_label
    task_direction = trace.navigation_context_grounding.task_direction
    if task_direction.status == "aligned":
        if intent is None:
            errors.append(
                "aligned_task_direction_requires_navigation_intent"
            )
        elif str(intent) not in {
            str(value)
            for value in task_direction.candidate_direction_labels
        }:
            errors.append(
                "navigation_intent_not_in_task_direction_candidates:"
                f"{intent}"
            )
    demonstrated = str(target.finish.demonstrated_direction.label)
    if demonstrated not in allowed:
        errors.append(
            "demonstrated_direction_outside_output_space:"
            f"{demonstrated}"
        )
    elif intent is not None and not direction_labels_are_intent_compatible(
        observation.direction_output_space.profile_id,
        demonstrated,
        str(intent),
    ):
        warnings.append(
            "demonstrated_navigation_intent_mismatch:"
            f"demonstrated={demonstrated},intent={intent}"
        )

    if assistant_text is not None:
        leaked = _forbidden_text_tokens(assistant_text)
        if leaked:
            errors.append(
                f"forbidden_fields_in_assistant:{leaked}"
            )
        parsed = parse_reasoning_target_v4(assistant_text)
        if not parsed.valid or parsed.target is None:
            errors.extend(
                f"assistant_text:{item}"
                for item in parsed.errors
            )
        elif parsed.target != target:
            errors.append(
                "assistant_text_target_roundtrip_mismatch"
            )
    return ReasoningValidationResultV4(
        errors=_unique(errors),
        warnings=_unique(warnings),
    )


def _validate_direction_membership(
    section_name: str,
    section: BaseModel,
    *,
    observation: NavObservationV4,
    errors: list[str],
) -> None:
    """Require every semantic direction to use the public profile."""

    allowed = {
        str(value)
        for value in observation.direction_output_space.allowed_labels
    }
    labels: list[str] = []
    if section_name == "LocalTraversability":
        labels.extend(
            str(item.direction_label)
            for item in section.direction_affordances
        )
    elif section_name == "NavigationContextGrounding":
        if section.goal_alignment is not None:
            labels.extend(
                str(value)
                for value in (
                    section.goal_alignment.supported_direction_labels
                )
            )
        for alignment in section.source_alignments:
            labels.extend(
                str(value)
                for value in alignment.supported_direction_labels
            )
        labels.extend(
            str(value)
            for value in (
                section.task_direction.candidate_direction_labels
            )
        )
    elif section_name == "InteractionConstraintGrounding":
        for constraint in section.constraints:
            affected = {
                str(value)
                for value in constraint.affected_direction_labels
            }
            labels.extend(
                affected
            )
            if affected and affected == allowed - {"stop"}:
                errors.append(
                    "global_constraint_must_not_enumerate_all_moving:"
                    f"{constraint.constraint_ref}"
                )
    elif section_name == "Decision":
        if section.navigation_intent_direction_label is not None:
            labels.append(
                str(section.navigation_intent_direction_label)
            )
        labels.append(str(section.selected_action_label))
        for uncertainty in section.critical_uncertainties:
            labels.extend(
                str(value)
                for value in (
                    uncertainty.affected_direction_labels
                )
            )
    outside = sorted(set(labels) - allowed)
    if outside:
        errors.append(
            "direction_labels_outside_public_output_space:"
            f"{outside}"
        )


def _validate_temporal_evidence(
    section_name: str,
    section: BaseModel,
    *,
    observation: NavObservationV4,
    errors: list[str],
) -> None:
    """Require distinct FPV images for temporal relation claims."""

    if section_name != "InteractionConstraintGrounding":
        return
    image_by_ref = {
        role: path
        for path, role in zip(
            observation.images,
            observation.image_roles,
        )
        if role.startswith("fpv.")
    }
    for constraint in section.constraints:
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
                "dynamic_constraint_requires_distinct_fpv:"
                f"{constraint.constraint_ref}"
            )


def _known_support_refs(
    target: ReasoningTargetV4,
) -> set[str]:
    """Return all semantic objects that Decision may cite."""

    trace = target.reasoning
    refs = {"task.goal", "nav.task_direction"}
    refs.update(
        item.requirement_ref
        for item in trace.task_grounding.requirements
    )
    refs.update(
        item.ambiguity_ref
        for item in trace.task_grounding.ambiguities
    )
    refs.update(
        item.entity_ref
        for item in trace.scene_grounding.entities
    )
    refs.update(
        item.control_ref
        for item in trace.scene_grounding.traffic_controls
    )
    refs.update(
        item.alignment_ref
        for item in (
            trace.navigation_context_grounding.source_alignments
        )
    )
    refs.update(
        item.constraint_ref
        for item in (
            trace.interaction_constraint_grounding.constraints
        )
    )
    refs.update(
        item.uncertainty_ref
        for item in trace.decision.critical_uncertainties
    )
    return refs


def _validate_constraint_effects(
    target: ReasoningTargetV4,
    *,
    selected_action_label: str,
    errors: list[str],
) -> None:
    """Reject a selected action that violates an applicable constraint."""

    decision = target.reasoning.decision
    for constraint in (
        target.reasoning.interaction_constraint_grounding.constraints
    ):
        affected = {
            str(value)
            for value in constraint.affected_direction_labels
        }
        if constraint.status != "applicable":
            continue
        if affected and selected_action_label not in affected:
            continue
        compatible = {
            "slow": decision.speed in {"slow", "stop"},
            "yield": (
                decision.social_action == "yield"
                or decision.speed == "stop"
            ),
            "stop": (
                decision.decision_type == "stop"
                and decision.selected_action_label == "stop"
            ),
            "avoid": selected_action_label not in affected,
        }[constraint.effect]
        if not compatible:
            errors.append(
                "constraint_effect_conflicts_with_final_decision:"
                f"{constraint.constraint_ref}:{constraint.effect}"
            )


def _validate_selected_direction_grounding(
    target: ReasoningTargetV4,
    *,
    direction_label: str,
    errors: list[str],
) -> None:
    """Reject full supervision when weak FPV affordance vetoes the GT label."""

    for affordance in (
        target.reasoning.local_traversability.direction_affordances
    ):
        if (
            str(affordance.direction_label) == direction_label
            and affordance.affordance in {"blocked", "forbidden"}
        ):
            errors.append(
                "weak_strong_grounding_conflict:"
                f"direction={direction_label}, "
                f"affordance={affordance.affordance}"
            )


def _stop_action_has_support(target: ReasoningTargetV4) -> bool:
    """Return whether a stop action cites a visible stop-capable reason."""

    trace = target.reasoning
    support_refs = {str(value) for value in trace.decision.support_refs}
    for constraint in (
        trace.interaction_constraint_grounding.constraints
    ):
        if (
            constraint.status == "applicable"
            and not constraint.affected_direction_labels
            and constraint.effect in {"stop", "yield"}
            and str(constraint.constraint_ref) in support_refs
        ):
            return True
    intent_labels = {
        str(value)
        for value in (
            trace.navigation_context_grounding.task_direction
            .candidate_direction_labels
        )
    }
    blocked_labels = {
        str(item.direction_label)
        for item in trace.local_traversability.direction_affordances
        if item.affordance in {"blocked", "forbidden"}
    }
    if (
        intent_labels
        and intent_labels.issubset(blocked_labels)
        and "nav.task_direction" in support_refs
    ):
        return True
    return any(
        item.resolution_policy == "stop"
        and str(item.uncertainty_ref) in support_refs
        for item in trace.decision.critical_uncertainties
    )


def _raw_refs_from_value(value: Any) -> set[str]:
    """Recursively collect raw evidence-address strings."""

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
    """Return unsupported private or numeric-uncertainty fields."""

    lowered = text.lower()
    forbidden = (
        '"confidence"',
        '"confidence_score"',
        '"cost"',
        '"cost_score"',
        "active_route_state",
        "arrival_gt",
        "bbox",
        "calibration",
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
