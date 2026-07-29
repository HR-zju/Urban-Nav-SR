"""Deterministic compact FullCoT projection for Reasoning V4.5.

The canonical V4.5 trace is already minimal, so the compact projection is an
exact six-section serialization.  Action supervision appears once, in
``Decision``.  There is no Finish, readout, ref graph, or history label.
"""

from __future__ import annotations

from dataclasses import dataclass

from mcp_server.agent.reasoning_v4_text import (
    parse_reasoning_target_v4,
    render_reasoning_target_v4,
)
from mcp_server.contracts.direction_profile import (
    direction_label_names,
)
from mcp_server.contracts.reasoning_v4 import (
    REASONING_V4_COMPACT_FULL_COT_RENDERER_ID,
    ReasoningTargetV4,
)


@dataclass(frozen=True)
class CompactFullCotRenderResultV4:
    """Describe one compact FullCoT language projection.

    Args:
        assistant_text: Exact six-section target text.
        renderer_id: Versioned projector identity.
        action_location: Unique LM action location, always ``decision``.
    """

    assistant_text: str
    renderer_id: str
    action_location: str


@dataclass(frozen=True)
class CompactFullCotParseResultV4:
    """Return the unique action from compact FullCoT text.

    Args:
        selected_action_label: Parsed action when valid.
        action_location: ``decision`` when valid.
        errors: Bounded structural or vocabulary failures.
    """

    selected_action_label: str | None
    action_location: str | None
    errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        """Return whether one in-profile action was parsed.

        Returns:
            True when action and location exist without errors.
        """

        return (
            self.selected_action_label is not None
            and self.action_location == "decision"
            and not self.errors
        )


def render_compact_full_cot_v4(
    target: ReasoningTargetV4,
    *,
    action_profile_id: str,
) -> CompactFullCotRenderResultV4:
    """Render an exact compact six-section target.

    Args:
        target: Typed canonical target.
        action_profile_id: Active public action profile.

    Returns:
        Canonical compact render result.

    Raises:
        ValueError: If the target action is outside the profile.
    """

    action = str(
        target.reasoning.decision.selected_action_label
    )
    if action not in direction_label_names(action_profile_id):
        raise ValueError(
            f"selected action outside profile: {action}"
        )
    return CompactFullCotRenderResultV4(
        assistant_text=render_reasoning_target_v4(target),
        renderer_id=(
            REASONING_V4_COMPACT_FULL_COT_RENDERER_ID
        ),
        action_location="decision",
    )


def validate_compact_full_cot_projection_v4(
    assistant_text: str,
    target: ReasoningTargetV4,
    *,
    action_profile_id: str,
) -> None:
    """Require exact deterministic compact projection equality.

    Args:
        assistant_text: Stored assistant text.
        target: Parent typed target.
        action_profile_id: Active action profile.

    Raises:
        ValueError: If rendering or parsing differs.
    """

    expected = render_compact_full_cot_v4(
        target,
        action_profile_id=action_profile_id,
    )
    if assistant_text != expected.assistant_text:
        raise ValueError(
            "compact FullCoT text differs from deterministic render"
        )
    parsed = parse_compact_full_cot_action_v4(
        assistant_text,
        action_profile_id=action_profile_id,
    )
    if not parsed.valid:
        raise ValueError(
            f"invalid compact FullCoT: {parsed.errors}"
        )
    if (
        parsed.selected_action_label
        != str(target.reasoning.decision.selected_action_label)
    ):
        raise ValueError(
            "compact FullCoT action differs from target"
        )


def parse_compact_full_cot_action_v4(
    text: str,
    *,
    action_profile_id: str,
) -> CompactFullCotParseResultV4:
    """Parse the unique Decision action from compact FullCoT.

    Args:
        text: Generated compact assistant text.
        action_profile_id: Active public action profile.

    Returns:
        Parsed action or bounded errors.
    """

    parsed = parse_reasoning_target_v4(text)
    if not parsed.valid or parsed.target is None:
        return CompactFullCotParseResultV4(
            selected_action_label=None,
            action_location=None,
            errors=tuple(parsed.errors),
        )
    action = str(
        parsed.target.reasoning.decision.selected_action_label
    )
    if action not in direction_label_names(action_profile_id):
        return CompactFullCotParseResultV4(
            selected_action_label=None,
            action_location=None,
            errors=(f"action_outside_profile:{action}",),
        )
    return CompactFullCotParseResultV4(
        selected_action_label=action,
        action_location="decision",
    )


__all__ = [
    "CompactFullCotParseResultV4",
    "CompactFullCotRenderResultV4",
    "parse_compact_full_cot_action_v4",
    "render_compact_full_cot_v4",
    "validate_compact_full_cot_projection_v4",
]
