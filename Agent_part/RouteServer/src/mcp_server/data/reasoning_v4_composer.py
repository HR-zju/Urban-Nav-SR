"""Deterministically assemble independently generated Reasoning V4 sections."""

from __future__ import annotations

from dataclasses import dataclass, field

from mcp_server.agent.reasoning_v4_validation import (
    validate_reasoning_target_v4,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    GroundingOutputsV4,
    ReasoningSourceV4,
)
from mcp_server.contracts.reasoning_v4 import (
    DecisionV4,
    FinishDirectionV4,
    FinishV4,
    ReasoningTargetV4,
    ReasoningTraceV4,
    coarse_motion_for_direction_label_v4,
)


REASONING_V4_COMPOSER_VERSION = "reasoning_v4_composer_v1"


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
    decision: DecisionV4,
) -> ReasoningCompositionResultV4:
    """Assemble a complete V4 target without changing teacher semantics.

    Args:
        source: Frozen observation and optional strong sidecars.
        groundings: Five independently generated grounding sections.
        decision: Joint Decision generated from raw input and groundings.

    Returns:
        Validated target and soft cross-section warnings.

    Raises:
        ValueError: If a grounding is missing, strong direction conflicts with
            Decision, or the assembled target violates a hard V4 invariant.
    """

    if not groundings.complete:
        raise ValueError("cannot compose V4 target from incomplete groundings")
    direction = source.direction_supervision
    finish_direction = None
    if direction is not None:
        expected_motion = coarse_motion_for_direction_label_v4(
            direction.label_name
        )
        if expected_motion != decision.motion:
            raise ValueError(
                "decision_direction_coarse_mismatch:"
                f"decision={decision.motion}, direction={direction.label_name}"
            )
        finish_direction = FinishDirectionV4(
            choice=direction.choice,
            label=direction.label_name,
            angle_deg=direction.angle_deg,
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
        direction=finish_direction,
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
