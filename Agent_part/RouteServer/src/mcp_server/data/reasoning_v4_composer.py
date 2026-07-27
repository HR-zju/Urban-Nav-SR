"""Deterministically assemble independently generated Reasoning V4 sections."""

from __future__ import annotations

from dataclasses import dataclass, field

from mcp_server.agent.reasoning_v4_validation import (
    validate_reasoning_target_v4,
)
from mcp_server.contracts.direction_profile import (
    coarse_motion_for_direction_label,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    GroundingOutputsV4,
    ReasoningSourceV4,
)
from mcp_server.contracts.reasoning_v4 import (
    DecisionV4,
    DecisionProposalV4,
    FinishDirectionV4,
    FinishV4,
    ReasoningTargetV4,
    ReasoningTraceV4,
)


REASONING_V4_COMPOSER_VERSION = "reasoning_v4_composer_v3"


@dataclass(frozen=True)
class ReasoningCompositionResultV4:
    """Return an assembled target and non-fatal consistency findings.

    Args:
        target: Canonical V4 target.
        warnings: Soft cross-section findings from the global validator.
    """

    target: ReasoningTargetV4
    warnings: list[str] = field(default_factory=list)


def compose_reasoning_target_v4(
    source: ReasoningSourceV4,
    groundings: GroundingOutputsV4,
    decision_proposal: DecisionProposalV4,
) -> ReasoningCompositionResultV4:
    """Assemble a complete V4 target without changing teacher semantics.

    Args:
        source: Frozen observation and optional strong sidecars.
        groundings: Five independently generated grounding sections.
        decision_proposal: Weak Decision fields generated from raw input and
            groundings. Motion is intentionally absent.

    Returns:
        Validated target and soft cross-section warnings.

    Raises:
        ValueError: If a grounding is missing, demonstrated direction is absent,
            the selected action is invalid, or a hard invariant fails.
    """

    if not groundings.complete:
        raise ValueError("cannot compose V4 target from incomplete groundings")
    direction = source.direction_supervision
    if direction is None:
        raise ValueError(
            "full V4.2 composition requires private direction supervision"
        )
    expected_motion = coarse_motion_for_direction_label(
        source.observation.direction_output_space.profile_id,
        decision_proposal.selected_action_label,
    )
    expected_decision_type = (
        "stop" if expected_motion == "stop" else "navigate"
    )
    if decision_proposal.decision_type != expected_decision_type:
        raise ValueError(
            "selected_action_decision_type_conflict:"
            f"proposal={decision_proposal.decision_type}, "
            f"selected_action={decision_proposal.selected_action_label}"
        )
    decision = DecisionV4(
        **decision_proposal.model_dump(mode="python"),
        motion=expected_motion,
    )
    finish_direction = FinishDirectionV4(
        label=direction.label_name,
    )

    trace = ReasoningTraceV4(
        task_grounding=groundings.task_grounding,
        scene_grounding=groundings.scene_grounding,
        local_traversability=groundings.local_traversability,
        navigation_context_grounding=groundings.navigation_context_grounding,
        interaction_constraint_grounding=groundings.interaction_constraint_grounding,
        decision=decision,
    )
    finish = FinishV4(
        decision_type=decision.decision_type,
        motion=decision.motion,
        speed=decision.speed,
        social_action=decision.social_action,
        selected_action_label=decision.selected_action_label,
        demonstrated_direction=finish_direction,
    )
    target = ReasoningTargetV4(reasoning=trace, finish=finish)
    validation = validate_reasoning_target_v4(
        target,
        observation=source.observation,
    )
    if not validation.valid:
        raise ValueError(
            "assembled V4 target failed validation: "
            f"{validation.errors}"
        )
    return ReasoningCompositionResultV4(
        target=target,
        warnings=validation.warnings,
    )
