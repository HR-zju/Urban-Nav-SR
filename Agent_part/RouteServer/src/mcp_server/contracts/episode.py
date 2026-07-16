"""Event-sourced Pydantic contract for closed-loop navigation episodes."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


EPISODE_TRACE_SCHEMA_VERSION = "urbannav_episode_trace_v2"


class EpisodeEventType(str, Enum):
    """Stable event names emitted by every RouteServer runner."""

    EPISODE_STARTED = "episode_started"
    SYSTEM_MESSAGE = "system_message"
    OBSERVATION_RECEIVED = "observation_received"
    MODEL_OUTPUT = "model_output"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    READOUT_REQUESTED = "readout_requested"
    ROUTE_STATE_DELTA = "route_state_delta"
    SYSTEM1_OUTPUT = "system1_output"
    SAFETY_VALIDATION = "safety_validation"
    ENVIRONMENT_STEP = "environment_step"
    USER_INTERRUPT = "user_interrupt"
    USER_RESUME = "user_resume"
    EXTERNAL_EVENT_RECEIVED = "external_event_received"
    EXTERNAL_EVENT_REJECTED = "external_event_rejected"
    TASK_STATE_UPDATED = "task_state_updated"
    INTERACTION_REQUESTED = "interaction_requested"
    INTERACTION_RESOLVED = "interaction_resolved"
    ACTION_DISPATCHED = "action_dispatched"
    ACTION_PREEMPTED = "action_preempted"
    CONTROLLER_FEEDBACK = "controller_feedback"
    CONTROL_HALTED = "control_halted"
    SAFETY_INTERVENTION = "safety_intervention"
    TERMINAL_VALIDATION = "terminal_validation"
    REVISION_CONFLICT = "revision_conflict"
    EPISODE_FINISHED = "episode_finished"


class EpisodeStatus(str, Enum):
    """Lifecycle state of an episode trace."""

    RUNNING = "running"
    WAITING_FOR_USER = "waiting_for_user"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TRUNCATED = "truncated"
    CANCELLED = "cancelled"


class EpisodeEvent(BaseModel):
    """One append-only event in a navigation episode.

    Args:
        episode_id: Parent episode id.
        tick_id: Zero-based control tick.
        event_type: Typed event kind.
        payload: Event-specific JSON payload.
        event_id: Globally unique event id.
        parent_event_id: Previous causal event id.
        timestamp: UTC creation time.
        source: Node or adapter that emitted the event.
        metadata: Diagnostic metadata not consumed as model supervision.
        schema_version: Episode trace contract version.
    """

    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=1)
    tick_id: int = Field(ge=0)
    event_type: EpisodeEventType
    payload: dict[str, Any] = Field(default_factory=dict)
    event_id: str = Field(default_factory=lambda: f"event:{uuid4().hex}")
    parent_event_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = EPISODE_TRACE_SCHEMA_VERSION


class EpisodeTraceV2(BaseModel):
    """Canonical event-sourced trace shared by runtime, training and evaluation.

    Args:
        episode_id: Stable episode id.
        session_id: Optional long-lived route session id.
        events: Causally ordered append-only events.
        status: Current lifecycle status.
        stop_reason: Terminal or interrupt reason.
        metadata: Episode-level model, tools and replay metadata.
        schema_version: Episode trace contract version.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    episode_id: str = Field(min_length=1)
    session_id: str | None = None
    events: list[EpisodeEvent] = Field(default_factory=list)
    status: EpisodeStatus = EpisodeStatus.RUNNING
    stop_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = EPISODE_TRACE_SCHEMA_VERSION

    @model_validator(mode="after")
    def _validate_event_chain(self) -> "EpisodeTraceV2":
        """Validate episode ids, unique event ids and causal parent order.

        Returns:
            Validated trace.

        Raises:
            ValueError: If the event chain is inconsistent.
        """

        seen: set[str] = set()
        for event in self.events:
            if event.episode_id != self.episode_id:
                raise ValueError("episode event belongs to a different episode")
            if event.event_id in seen:
                raise ValueError(f"duplicate episode event id: {event.event_id}")
            if event.parent_event_id is not None and event.parent_event_id not in seen:
                raise ValueError(f"event parent is not earlier in trace: {event.parent_event_id}")
            seen.add(event.event_id)
        return self

    def append(
        self,
        event_type: EpisodeEventType,
        *,
        tick_id: int,
        source: str,
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EpisodeEvent:
        """Append one causally linked event.

        Args:
            event_type: Event kind.
            tick_id: Zero-based control tick.
            source: Emitting node or adapter.
            payload: Event-specific payload.
            metadata: Diagnostic metadata.

        Returns:
            Newly appended event.
        """

        parent_id = self.events[-1].event_id if self.events else None
        event = EpisodeEvent(
            episode_id=self.episode_id,
            tick_id=tick_id,
            event_type=event_type,
            payload=dict(payload or {}),
            parent_event_id=parent_id,
            source=source,
            metadata=dict(metadata or {}),
        )
        self.events.append(event)
        return event

    def finish(self, *, status: EpisodeStatus, stop_reason: str, tick_id: int, source: str) -> EpisodeEvent:
        """Mark the trace terminal and append ``episode_finished``.

        Args:
            status: Terminal lifecycle status.
            stop_reason: Auditable terminal reason.
            tick_id: Last control tick.
            source: Finishing node or adapter.

        Returns:
            Terminal event.

        Raises:
            ValueError: If ``status`` is not terminal.
        """

        if status not in {
            EpisodeStatus.SUCCEEDED,
            EpisodeStatus.FAILED,
            EpisodeStatus.TRUNCATED,
            EpisodeStatus.CANCELLED,
        }:
            raise ValueError("finish status must be terminal")
        self.status = status
        self.stop_reason = stop_reason
        return self.append(
            EpisodeEventType.EPISODE_FINISHED,
            tick_id=tick_id,
            source=source,
            payload={"status": status.value, "stop_reason": stop_reason},
        )
