"""Small deterministic metric set for frozen closed-loop episodes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mcp_server.contracts.episode import EpisodeTraceV2
from mcp_server.eval.episode.schema import FrozenEpisodeCase


class EpisodeMetricResult(BaseModel):
    """Main outcome metrics and diagnostic counters for one episode.

    Args:
        episode_id: Stable benchmark id.
        arrival_success: Whether the terminal environment label reports arrival.
        collision_free_success: Whether no transition reports a collision.
        recovery_success: Recovery outcome for recovery-designated cases.
        normalized_efficiency: Success-weighted step efficiency.
        total_reward: Sum of frozen per-step rewards.
        diagnostics: Tool/safety/step counters not promoted to main score.
    """

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    arrival_success: float = Field(ge=0.0, le=1.0)
    collision_free_success: float = Field(ge=0.0, le=1.0)
    recovery_success: float | None = Field(default=None, ge=0.0, le=1.0)
    normalized_efficiency: float = Field(ge=0.0, le=1.0)
    total_reward: float
    diagnostics: dict[str, int | float | bool | None] = Field(default_factory=dict)


def compute_episode_metrics(trace: EpisodeTraceV2 | dict[str, Any], case: FrozenEpisodeCase) -> EpisodeMetricResult:
    r"""Compute deterministic outcome metrics from frozen environment events.

    Args:
        trace: Episode trace produced by the closed-loop runtime.
        case: Matching frozen benchmark case.

    Returns:
        Main outcome metrics plus diagnostics.

    The success-weighted efficiency is

    \[
    E = S_{arrival} \cdot \frac{N^*}{\max(N^*, \hat N)},
    \]

    where ``N*`` is the frozen optimal step count and ``N_hat`` is the executed
    environment-step count.
    """

    episode = trace if isinstance(trace, EpisodeTraceV2) else EpisodeTraceV2.model_validate(trace)
    environment_events = [event for event in episode.events if event.event_type.value == "environment_step"]
    safety_events = [event for event in episode.events if event.event_type.value == "safety_validation"]
    tool_events = [event for event in episode.events if event.event_type.value == "tool_call"]
    results = [(event.payload.get("result") or {}) for event in environment_events]
    terminal_info = results[-1].get("info") or {} if results else {}
    arrival = float(bool(terminal_info.get("arrival_success")))
    collision = any(bool((result.get("info") or {}).get("collision")) for result in results)
    executed_steps = len(environment_events)
    efficiency = arrival * case.optimal_steps / max(case.optimal_steps, executed_steps or case.optimal_steps)
    rejected = sum(
        not bool(((event.payload.get("result") or {}).get("accepted")))
        for event in safety_events
    )
    recovery = None
    if case.requires_recovery:
        recovery = float(bool(arrival and rejected > 0))
    return EpisodeMetricResult(
        episode_id=case.episode_id,
        arrival_success=arrival,
        collision_free_success=float(bool(arrival) and not collision),
        recovery_success=recovery,
        normalized_efficiency=float(efficiency),
        total_reward=float(sum(float(result.get("reward", 0.0)) for result in results)),
        diagnostics={
            "environment_steps": executed_steps,
            "tool_calls": len(tool_events),
            "safety_rejections": rejected,
            "terminal": bool(episode.events and episode.events[-1].event_type.value == "episode_finished"),
        },
    )
