"""Deterministically compose strong-action Reasoning V4.5 targets."""

from __future__ import annotations

from dataclasses import dataclass, field

from mcp_server.agent.reasoning_v4_validation import (
    applicable_constraint_effects_v4,
    validate_reasoning_target_v4,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    GroundingOutputsV4,
    ReasoningSourceV4,
)
from mcp_server.contracts.reasoning_v4 import (
    DecisionProposalV4,
    DecisionV4,
    ReasoningTargetV4,
    ReasoningTraceV4,
)


REASONING_V4_COMPOSER_VERSION = "reasoning_v4_5_composer_v1"


@dataclass(frozen=True)
class ReasoningCompositionResultV4:
    """Return a deterministic candidate and non-fatal findings.

    Args:
        candidate_target: Strong-direction-owned six-section target.
        warnings: Non-fatal consistency findings.
    """

    candidate_target: ReasoningTargetV4
    warnings: list[str] = field(default_factory=list)


def compose_reasoning_target_v4(
    source: ReasoningSourceV4,
    groundings: GroundingOutputsV4,
    decision_proposal: DecisionProposalV4,
) -> ReasoningCompositionResultV4:
    """Compose the final tuple without allowing teacher action override.

    The strong direction owns ``selected_action_label``.  Speed and response
    are deterministically projected from applicable constraint effects:
    none -> normal/proceed, slow -> slow/slow, yield -> slow/yield,
    avoid -> slow/detour, and stop -> stop/stop.  A real global stop
    constraint conflicting with a strong moving label is invalid and requires
    review; the composer never rewrites that label to stop.

    Args:
        source: Frozen observation and private deterministic sidecars.
        groundings: Five independently accepted grounding sections.
        decision_proposal: Blind teacher proposal retained for audit only.

    Returns:
        Validated strong-action candidate.

    Raises:
        ValueError: If strong direction is absent, groundings are incomplete,
            or deterministic composition is inconsistent.
    """

    del decision_proposal
    if not groundings.complete:
        raise ValueError(
            "cannot compose V4.5 target from incomplete groundings"
        )
    direction = source.direction_supervision
    if direction is None:
        raise ValueError(
            "V4.5 composition requires strong direction supervision"
        )
    interaction = groundings.interaction_constraint_grounding
    assert interaction is not None
    selected_action = str(direction.label_name)
    effects = applicable_constraint_effects_v4(
        selected_action,
        interaction,
    )
    speed, response = _compose_behavior_tuple_v4(
        selected_action,
        effects,
    )
    decision = DecisionV4(
        selected_action_label=selected_action,
        speed=speed,
        response=response,
    )
    target = ReasoningTargetV4(
        reasoning=ReasoningTraceV4(
            task_grounding=groundings.task_grounding,
            scene_grounding=groundings.scene_grounding,
            local_traversability=groundings.local_traversability,
            navigation_context_grounding=(
                groundings.navigation_context_grounding
            ),
            interaction_constraint_grounding=interaction,
            decision=decision,
        )
    )
    validation = validate_reasoning_target_v4(
        target,
        observation=source.observation,
        direction_supervision=direction,
    )
    if not validation.valid:
        raise ValueError(
            "assembled V4.5 target failed validation: "
            f"{validation.errors}"
        )
    return ReasoningCompositionResultV4(
        candidate_target=target,
        warnings=validation.warnings,
    )


def _compose_behavior_tuple_v4(
    selected_action: str,
    effects: tuple[str, ...],
) -> tuple[str, str]:
    """Project strong action and ordered effects to speed/response."""

    if selected_action == "stop":
        return "stop", "stop"
    if "stop" in effects:
        raise ValueError(
            "strong_moving_action_conflicts_with_stop_constraint"
        )
    if "avoid" in effects:
        return "slow", "detour"
    if "yield" in effects:
        return "slow", "yield"
    if "slow" in effects:
        return "slow", "slow"
    return "normal", "proceed"


__all__ = [
    "REASONING_V4_COMPOSER_VERSION",
    "ReasoningCompositionResultV4",
    "compose_reasoning_target_v4",
]
