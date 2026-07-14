"""Typed service boundary for external ms-swift episode actions."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class EpisodeResetResponse(BaseModel):
    """HTTP reset response consumed by ``UrbanNavEnvClient``.

    Args:
        episode_id: Stable active episode id.
        observation: Initial model-visible observation.
        replay_key: Frozen reset key required for training rollouts.
        info: Environment metadata.
        observation_text: Optional pre-rendered user message.
    """

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    observation: dict[str, Any]
    replay_key: str
    info: dict[str, Any] = Field(default_factory=dict)
    observation_text: str | None = None


class EpisodeStepResponse(BaseModel):
    """HTTP step response consumed by ``UrbanNavEnvClient``.

    Args:
        episode_id: Stable active episode id.
        observation: Next model-visible observation.
        reward: Frozen deterministic step reward.
        done: Whether the episode is terminal.
        replay_key: Frozen transition key.
        reward_components: Auditable reward decomposition.
        info: Environment outcome metadata.
        observation_text: Optional pre-rendered user message.
        episode_trace: Optional canonical trace snapshot.
        stop_reason: Optional terminal reason.
    """

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    observation: dict[str, Any]
    reward: float
    done: bool
    replay_key: str
    reward_components: dict[str, float] = Field(default_factory=dict)
    info: dict[str, Any] = Field(default_factory=dict)
    observation_text: str | None = None
    episode_trace: dict[str, Any] | None = None
    stop_reason: str | None = None


class ExternalEpisodeService(Protocol):
    """Service implemented by frozen simulator or deployment episode backends."""

    async def reset(self, payload: dict[str, Any]) -> EpisodeResetResponse:
        """Create one externally-driven episode."""

    async def step(self, episode_id: str, payload: dict[str, Any]) -> EpisodeStepResponse:
        """Apply one external model action to an active episode."""
