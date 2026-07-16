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
    external_events = [event for event in episode.events if event.event_type.value == "external_event_received"]
    preemptions = [event for event in episode.events if event.event_type.value == "action_preempted"]
    conflicts = [event for event in episode.events if event.event_type.value == "revision_conflict"]
    interaction_requests = [event for event in episode.events if event.event_type.value == "interaction_requested"]
    interaction_resolutions = [event for event in episode.events if event.event_type.value == "interaction_resolved"]
    terminal_validations = [event for event in episode.events if event.event_type.value == "terminal_validation"]
    control_halts = [event for event in episode.events if event.event_type.value == "control_halted"]
    stale_dispatches, total_dispatches = _count_stale_action_dispatches(episode)
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
    rejected_completions = sum(
        (event.payload.get("request") or {}).get("action_type") == "complete_task"
        and not bool((event.payload.get("result") or {}).get("accepted"))
        for event in terminal_validations
    )
    abort_requests = sum(
        (event.payload.get("request") or {}).get("action_type") == "abort_task"
        for event in terminal_validations
    )
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
            "external_events": len(external_events),
            "action_preemptions": len(preemptions),
            "revision_conflicts": len(conflicts),
            "interaction_resolution_rate": (
                len(interaction_resolutions) / len(interaction_requests) if interaction_requests else None
            ),
            "stale_action_dispatches": stale_dispatches,
            "stale_action_execution_rate": stale_dispatches / total_dispatches if total_dispatches else 0.0,
            "safety_rejections": rejected,
            "terminal_validation_requests": len(terminal_validations),
            "rejected_completion_requests": rejected_completions,
            "abort_requests": abort_requests,
            "control_halts": len(control_halts),
            "terminal": bool(episode.events and episode.events[-1].event_type.value == "episode_finished"),
        },
    )


def _count_stale_action_dispatches(trace: EpisodeTraceV2) -> tuple[int, int]:
    """Audit dispatch leases against the latest event-sourced runtime revisions."""

    task_revision = 0
    route_revision = 0
    control_epoch = 0
    observation_id = ""
    stale = 0
    total = 0
    for event in trace.events:
        payload = event.payload or {}
        if event.event_type.value == "episode_started":
            task_state = payload.get("task_state") or {}
            control_state = payload.get("control_state") or {}
            task_revision = int(task_state.get("task_revision") or 0)
            route_revision = int(task_state.get("route_revision") or 0)
            control_epoch = int(control_state.get("control_epoch") or 0)
        elif event.event_type.value == "observation_received":
            observation_id = str(payload.get("observation_id") or "")
        elif event.event_type.value == "task_state_updated":
            task_state = payload.get("task_state") or {}
            control_state = payload.get("control_state") or {}
            if task_state.get("task_revision") is not None:
                task_revision = int(task_state["task_revision"])
            if task_state.get("route_revision") is not None:
                route_revision = int(task_state["route_revision"])
            if control_state.get("control_epoch") is not None:
                control_epoch = int(control_state["control_epoch"])
        elif event.event_type.value == "route_state_delta":
            state_revision = payload.get("state_revision")
            if state_revision is not None:
                route_revision = int(state_revision)
        elif event.event_type.value == "action_dispatched":
            total += 1
            lease = payload.get("lease") or {}
            fresh = (
                lease.get("task_revision") == task_revision
                and lease.get("route_revision") == route_revision
                and lease.get("control_epoch") == control_epoch
                and str(lease.get("observation_id") or "") == observation_id
            )
            stale += int(not fresh)
    return stale, total
