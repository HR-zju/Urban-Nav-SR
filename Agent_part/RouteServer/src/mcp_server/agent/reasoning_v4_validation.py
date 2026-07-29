"""Semantic validation for compact shared-input Reasoning V4.6."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from pydantic import BaseModel

from mcp_server.agent.reasoning_v4_text import (
    parse_reasoning_target_v4,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    DirectionSupervisionV4,
    NavObservationV4,
)
from mcp_server.contracts.reasoning_v4 import (
    DecisionProposalV4,
    DecisionV4,
    InteractionConstraintGroundingV4,
    LocalTraversabilityV4,
    NavigationContextGroundingV4,
    ReasoningTargetV4,
    SceneGroundingV4,
    TaskGroundingV4,
)


@dataclass(frozen=True)
class ReasoningValidationResultV4:
    """Return hard failures separately from non-fatal findings.

    Args:
        errors: Hard semantic or wire failures.
        warnings: Non-fatal audit findings.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Return whether no hard validation error occurred.

        Returns:
            True when ``errors`` is empty.
        """

        return not self.errors


def validate_reasoning_section_v4(
    section_name: str,
    section: BaseModel,
    *,
    observation: NavObservationV4,
) -> ReasoningValidationResultV4:
    """Validate one independent section against public vocabularies.

    Args:
        section_name: Canonical six-section name.
        section: Parsed typed section.
        observation: Complete prompt-visible input envelope.

    Returns:
        Bounded errors and warnings.

    Raises:
        ValueError: If section name and model type disagree.
    """

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
            "section/type mismatch: "
            f"{section_name}/{type(section).__name__}"
        )

    errors: list[str] = []
    warnings: list[str] = []
    action_labels = set(observation.direction_context.action_labels)
    probe_labels = set(
        observation.direction_context.affordance_probe_labels
    )

    if isinstance(section, LocalTraversabilityV4):
        for item in section.direction_affordances:
            label = str(item.direction_label)
            if label not in probe_labels:
                errors.append(
                    f"unknown_affordance_probe_label:{label}"
                )
            if label == "stop":
                errors.append(
                    "local_traversability_cannot_probe_stop"
                )

    if isinstance(section, NavigationContextGroundingV4):
        task_direction = section.task_direction
        if (
            task_direction.direction is not None
            and str(task_direction.direction) not in action_labels
        ):
            errors.append(
                "task_direction_outside_action_profile:"
                f"{task_direction.direction}"
            )
        if (
            task_direction.status == "aligned"
            and task_direction.source == "bev_route"
            and not _has_visible_bev_direction_authority(observation)
        ):
            errors.append(
                "bev_route_task_direction_without_visible_authority"
            )
        if (
            task_direction.status == "aligned"
            and task_direction.source == "instruction"
            and not _instruction_has_explicit_direction(
                observation.instruction
            )
        ):
            errors.append(
                "instruction_task_direction_without_explicit_direction"
            )
        if (
            task_direction.status == "aligned"
            and task_direction.source not in {
                "bev_route",
                "instruction",
            }
        ):
            errors.append(
                "aligned_task_direction_without_visible_source"
            )

    if isinstance(section, InteractionConstraintGroundingV4):
        for constraint in section.constraints:
            unknown = sorted(
                set(
                    str(label)
                    for label in constraint.affected_direction_labels
                )
                - action_labels
            )
            if unknown:
                errors.append(
                    "constraint_direction_outside_action_profile:"
                    f"{unknown}"
                )
            if constraint.status == "uncertain":
                warnings.append(
                    "uncertain_constraint_requires_review_if_action_relevant"
                )

    if isinstance(section, (DecisionProposalV4, DecisionV4)):
        label = str(section.selected_action_label)
        if label not in action_labels:
            errors.append(
                f"decision_action_outside_profile:{label}"
            )

    return ReasoningValidationResultV4(
        errors=_unique(errors),
        warnings=_unique(warnings),
    )


def sanitize_reasoning_section_evidence_v4(
    section_name: str,
    section: BaseModel,
    *,
    observation: NavObservationV4,
) -> tuple[BaseModel, list[str]]:
    """Return the already typed V4.6 section unchanged.

    V4.6 section models contain no evidence refs or temporal claims, so there
    is no lossy repair step.  The function remains as an explicit batch-runner
    boundary and compatibility point.

    Args:
        section_name: Canonical section name.
        section: Typed section.
        observation: Prompt-visible observation.

    Returns:
        Original section and an empty warning list.

    Raises:
        ValueError: If section name and model type disagree.
    """

    validate_reasoning_section_v4(
        section_name,
        section,
        observation=observation,
    )
    return section, []


def validate_reasoning_target_v4(
    target: ReasoningTargetV4,
    *,
    observation: NavObservationV4,
    direction_supervision: DirectionSupervisionV4 | None = None,
) -> ReasoningValidationResultV4:
    """Validate all six sections and optional strong-direction ownership.

    Args:
        target: Typed V4.6 target.
        observation: Complete prompt-visible input.
        direction_supervision: Optional upstream deterministic direction.

    Returns:
        Aggregated bounded findings.
    """

    trace = target.reasoning
    sections: tuple[tuple[str, BaseModel], ...] = (
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
    )
    errors: list[str] = []
    warnings: list[str] = []
    for section_name, section in sections:
        result = validate_reasoning_section_v4(
            section_name,
            section,
            observation=observation,
        )
        errors.extend(result.errors)
        warnings.extend(result.warnings)
    if (
        direction_supervision is not None
        and str(trace.decision.selected_action_label)
        != str(direction_supervision.label_name)
    ):
        errors.append(
            "decision_selected_action_must_equal_strong_direction"
        )
    errors.extend(
        decision_constraint_errors_v4(
            trace.decision,
            trace.interaction_constraint_grounding,
        )
    )
    return ReasoningValidationResultV4(
        errors=_unique(errors),
        warnings=_unique(warnings),
    )


def validate_reasoning_training_text_v4(
    assistant_text: str,
    *,
    expected_target: ReasoningTargetV4,
) -> ReasoningValidationResultV4:
    """Validate canonical text round-trip equality.

    Args:
        assistant_text: Rendered assistant target.
        expected_target: Exact typed target expected after parsing.

    Returns:
        Wire/schema mismatch findings.
    """

    parsed = parse_reasoning_target_v4(assistant_text)
    if not parsed.valid or parsed.target is None:
        return ReasoningValidationResultV4(errors=parsed.errors)
    if parsed.target != expected_target:
        return ReasoningValidationResultV4(
            errors=["assistant_text_target_mismatch"],
        )
    return ReasoningValidationResultV4()


def decision_constraint_errors_v4(
    decision: DecisionProposalV4 | DecisionV4,
    constraints: InteractionConstraintGroundingV4,
) -> list[str]:
    """Return deterministic action/constraint consistency failures.

    Only ``applicable`` constraints affect the tuple.  A strong moving action
    under a global stop, or inside an applicable avoid subset, is a review
    conflict and must not be silently rewritten.

    Args:
        decision: Action tuple to audit.
        constraints: Independently annotated behavior constraints.

    Returns:
        Stable error codes.
    """

    label = str(decision.selected_action_label)
    relevant = [
        constraint
        for constraint in constraints.constraints
        if (
            constraint.status == "applicable"
            and (
                not constraint.affected_direction_labels
                or label
                in {
                    str(value)
                    for value
                    in constraint.affected_direction_labels
                }
            )
        )
    ]
    effects = {
        constraint.effect for constraint in relevant
    }
    if label == "stop":
        return (
            []
            if decision.speed == "stop"
            and decision.response == "stop"
            else ["stop_action_tuple_conflict"]
        )
    if "stop" in effects:
        return [
            "strong_moving_action_conflicts_with_stop_constraint"
        ]
    expected = (
        ("slow", "detour")
        if "avoid" in effects
        else ("slow", "yield")
        if "yield" in effects
        else ("slow", "slow")
        if "slow" in effects
        else ("normal", "proceed")
    )
    if (decision.speed, decision.response) != expected:
        return [
            "constraint_effect_tuple_conflict:"
            f"expected={expected[0]}/{expected[1]}"
        ]
    return []


def applicable_constraint_effects_v4(
    selected_action_label: str,
    constraints: InteractionConstraintGroundingV4,
) -> tuple[str, ...]:
    """Return applicable effects for one action in deterministic order.

    Args:
        selected_action_label: Public action label.
        constraints: Independently annotated constraints.

    Returns:
        Unique effects ordered by ``stop, avoid, yield, slow`` priority.
    """

    effects = {
        constraint.effect
        for constraint in constraints.constraints
        if (
            constraint.status == "applicable"
            and (
                not constraint.affected_direction_labels
                or selected_action_label
                in {
                    str(value)
                    for value
                    in constraint.affected_direction_labels
                }
            )
        )
    }
    return tuple(
        effect
        for effect in ("stop", "avoid", "yield", "slow")
        if effect in effects
    )


def _has_visible_bev_direction_authority(
    observation: NavObservationV4,
) -> bool:
    """Return whether one visible BEV may support route direction."""

    return any(
        entry.modality == "bev"
        and entry.bev_route_evidence is not None
        and entry.bev_route_evidence.route_overlay_visible
        and entry.bev_route_evidence.route_direction_authority
        == "visual_conditional"
        and entry.bev_route_evidence.frame != "unknown"
        and entry.bev_route_evidence.ego_forward != "unknown"
        for entry in observation.evidence_catalog
    )


def _instruction_has_explicit_direction(
    instruction: str | None,
) -> bool:
    """Return a conservative lexical explicit-direction signal."""

    if instruction is None:
        return False
    normalized = instruction.casefold()
    markers = (
        "left",
        "right",
        "straight",
        "forward",
        "turn",
        "左",
        "右",
        "直行",
        "前方",
        "转",
    )
    return any(marker in normalized for marker in markers)


def _unique(values: Iterable[str]) -> list[str]:
    """Return stable duplicate-free strings."""

    return list(dict.fromkeys(str(value) for value in values))


__all__ = [
    "ReasoningValidationResultV4",
    "applicable_constraint_effects_v4",
    "decision_constraint_errors_v4",
    "sanitize_reasoning_section_evidence_v4",
    "validate_reasoning_section_v4",
    "validate_reasoning_target_v4",
    "validate_reasoning_training_text_v4",
]
