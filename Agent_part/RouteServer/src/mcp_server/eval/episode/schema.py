"""Pydantic schemas for frozen closed-loop episode cases."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mcp_server.contracts.task import ExternalUserEvent


FROZEN_EPISODE_CASE_SCHEMA_VERSION = "urbannav_frozen_episode_case_v1"


class FrozenEpisodeTransition(BaseModel):
    """One immutable simulator/replay transition.

    Args:
        replay_key: Stable transition cache key.
        observation: Observation returned after executing the action chunk.
        reward: Frozen deterministic reward.
        done: Whether this transition terminates the episode.
        info: Arrival, collision and recovery labels.
        expected_action_shape: Optional expected System1 ``[T,D]`` shape.
        expected_action_type: Optional canonical System2 action type. Legacy
            ``tool_call``/``finish`` syntax-level expectations remain readable.
        expected_tool_op: Optional tool operation required by this transition.
        external_events: User events deterministically emitted after execution.
    """

    model_config = ConfigDict(extra="forbid")

    replay_key: str = Field(min_length=1)
    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)
    expected_action_shape: tuple[int, int] | None = None
    expected_action_type: str | None = None
    expected_tool_op: str | None = None
    external_events: list[ExternalUserEvent] = Field(default_factory=list)


class FrozenEpisodeCase(BaseModel):
    """One deterministic closed-loop navigation benchmark episode.

    Args:
        episode_id: Stable benchmark id.
        instruction: User navigation instruction.
        initial_observation: First environment observation.
        initial_replay_key: Frozen reset record key.
        transitions: Ordered immutable environment transitions.
        optimal_steps: Reference number of environment steps.
        requires_recovery: Whether recovery success is a main case outcome.
        initial_external_events: Events available with the reset observation.
        metadata: Dataset and scenario metadata.
        schema_version: Case contract version.
    """

    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=1)
    instruction: str
    initial_observation: dict[str, Any]
    initial_replay_key: str = Field(min_length=1)
    transitions: list[FrozenEpisodeTransition]
    optimal_steps: int = Field(ge=1)
    requires_recovery: bool = False
    initial_external_events: list[ExternalUserEvent] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = FROZEN_EPISODE_CASE_SCHEMA_VERSION

    @model_validator(mode="after")
    def _validate_terminal_transition(self) -> "FrozenEpisodeCase":
        """Require a non-empty sequence ending in a terminal transition.

        Returns:
            Validated case.

        Raises:
            ValueError: If transitions are empty or non-terminal.
        """

        if not self.transitions:
            raise ValueError("frozen episode must contain transitions")
        if not self.transitions[-1].done:
            raise ValueError("last frozen episode transition must be terminal")
        if any(transition.done for transition in self.transitions[:-1]):
            raise ValueError("only the final frozen transition may be terminal")
        events = list(self.initial_external_events)
        for transition in self.transitions:
            events.extend(transition.external_events)
        if any(event.episode_id != self.episode_id for event in events):
            raise ValueError("frozen external event belongs to a different episode")
        return self
