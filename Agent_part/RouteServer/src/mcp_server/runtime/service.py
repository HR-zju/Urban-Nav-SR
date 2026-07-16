"""Typed service boundary for external ms-swift episode actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from mcp_server.contracts.task import ExternalUserEvent


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
        terminal_status: Optional canonical episode outcome.
        terminal_validation: Optional completion/abort validation result.
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
    terminal_status: str | None = None
    terminal_validation: dict[str, Any] | None = None


class EpisodeEventSubmitResponse(BaseModel):
    """Response for an out-of-band user event.

    Args:
        episode_id: Target episode id.
        event: Validated submitted event.
        accepted: Whether the event id was new and accepted by the service.
        duplicate: Whether this was an idempotent retry.
        info: Service-specific deterministic diagnostics.
    """

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    event: ExternalUserEvent
    accepted: bool
    duplicate: bool = False
    info: dict[str, Any] = Field(default_factory=dict)


class ExternalEpisodeService(Protocol):
    """Service implemented by frozen simulator or deployment episode backends."""

    async def reset(self, payload: dict[str, Any]) -> EpisodeResetResponse:
        """Create one externally-driven episode."""

    async def step(self, episode_id: str, payload: dict[str, Any]) -> EpisodeStepResponse:
        """Apply one external model action to an active episode."""

    async def submit_event(self, episode_id: str, payload: dict[str, Any]) -> EpisodeEventSubmitResponse:
        """Submit an event independently of model-action stepping."""


@dataclass(frozen=True)
class _RuntimeRegistration:
    """One registered graph runtime.

    Args:
        runtime: ``EpisodeRuntime``-compatible instance.
        thread_id: LangGraph checkpoint thread id.
    """

    runtime: Any
    thread_id: str


class EpisodeRuntimeEventService:
    """Expose registered ``EpisodeRuntime`` instances through event ingress."""

    def __init__(self) -> None:
        self._registrations: dict[str, _RuntimeRegistration] = {}

    def register(self, *, episode_id: str, runtime: Any, thread_id: str | None = None) -> None:
        """Register an active runtime.

        Args:
            episode_id: Active episode id.
            runtime: Runtime exposing ``submit_event`` and ``resume``.
            thread_id: Optional checkpoint thread id; defaults to episode id.

        Returns:
            None.
        """

        self._registrations[episode_id] = _RuntimeRegistration(
            runtime=runtime,
            thread_id=thread_id or episode_id,
        )

    def unregister(self, episode_id: str) -> None:
        """Remove a completed runtime registration.

        Args:
            episode_id: Episode id to remove.

        Returns:
            None.
        """

        self._registrations.pop(episode_id, None)

    async def submit_event(self, episode_id: str, payload: dict[str, Any]) -> EpisodeEventSubmitResponse:
        """Submit and optionally resume a registered graph.

        Args:
            episode_id: Active episode id.
            payload: ``event`` plus optional ``resume_if_waiting``.

        Returns:
            Event acceptance and resume diagnostics.

        Raises:
            KeyError: If no runtime is registered.
        """

        registration = self._registrations.get(episode_id)
        if registration is None:
            raise KeyError(f"unregistered runtime episode: {episode_id}")
        value = dict(payload.get("event") or payload)
        value.pop("resume_if_waiting", None)
        value.setdefault("episode_id", episode_id)
        event = ExternalUserEvent.model_validate(value)
        if event.episode_id != episode_id:
            raise ValueError(
                f"event episode_id mismatch: path={episode_id}, payload={event.episode_id}"
            )
        inserted = await registration.runtime.submit_event(event)
        resumed = False
        status = None
        if inserted and bool(payload.get("resume_if_waiting")):
            # interrupt 节点会再次 append 同一 event_id；EventStore 幂等去重保证只 reduce 一次。
            state = await registration.runtime.resume(
                thread_id=registration.thread_id,
                user_response=event.model_dump(mode="json"),
            )
            resumed = True
            status = state.get("terminal_status") or ("interrupted" if "__interrupt__" in state else "running")
        return EpisodeEventSubmitResponse(
            episode_id=episode_id,
            event=event,
            accepted=inserted,
            duplicate=not inserted,
            info={"resumed": resumed, "runtime_status": status},
        )
