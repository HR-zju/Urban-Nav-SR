"""Typed task, dialogue, control and external-event contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


TASK_CONTRACT_SCHEMA_VERSION = "urbannav_task_contract_v1"


class TaskStatus(str, Enum):
    """Lifecycle state of the user navigation task."""

    ACTIVE = "active"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


class ControlStatus(str, Enum):
    """Lifecycle state of the short-horizon action executor."""

    IDLE = "idle"
    EXECUTING = "executing"
    PAUSED = "paused"
    STOPPED = "stopped"


class ExternalUserEventType(str, Enum):
    """User-originated events accepted while an episode is running."""

    NEW_INSTRUCTION = "new_instruction"
    MODIFY_TASK = "modify_task"
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    USER_RESPONSE = "user_response"
    EMERGENCY_STOP = "emergency_stop"


class PendingInteraction(BaseModel):
    """One auditable question waiting for a correlated user response.

    Args:
        interaction_id: Stable correlation id.
        request: Model/tool-generated question and options.
        motion_policy: Required robot behavior while waiting.
    """

    model_config = ConfigDict(extra="forbid")

    interaction_id: str
    request: dict[str, Any]
    motion_policy: str = "stop"


class DialogueTurn(BaseModel):
    """One bounded dialogue turn stored outside the model target.

    Args:
        role: Turn source.
        content: JSON-compatible turn content.
        event_id: External event grounding this turn when available.
        interaction_id: Question/answer correlation id when available.
    """

    model_config = ConfigDict(extra="forbid")

    role: str
    content: Any
    event_id: str | None = None
    interaction_id: str | None = None


class DialogueState(BaseModel):
    """Checkpoint-safe bounded conversation state.

    Args:
        recent_turns: Recent structured turns retained verbatim.
        summary: Optional older-turn summary.
        pending_interaction: Current blocking question.
    """

    model_config = ConfigDict(extra="forbid")

    recent_turns: list[DialogueTurn] = Field(default_factory=list)
    summary: str | None = None
    pending_interaction: PendingInteraction | None = None


class TaskState(BaseModel):
    """Revisioned user task state owned by RouteServer.

    Args:
        task_id: Stable task id within one episode.
        active_instruction: Current user instruction.
        goal_spec: Optional multimodal/coordinate goal description.
        constraints: Current local navigation constraints.
        route_session_id: Bound route session when one exists.
        task_revision: Revision of accepted user-task semantics.
        route_revision: Revision of the committed route session.
        status: Current task lifecycle state.
        last_event_id: Last accepted external event.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str
    active_instruction: str = ""
    goal_spec: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    route_session_id: str | None = None
    task_revision: int = Field(default=0, ge=0)
    route_revision: int = Field(default=0, ge=0)
    status: TaskStatus = TaskStatus.ACTIVE
    last_event_id: str | None = None


class ControlState(BaseModel):
    """Revisioned short-horizon execution state.

    Args:
        control_epoch: Monotonic generation invalidating stale action chunks.
        active_action_id: Currently dispatched action lease.
        status: Controller lifecycle state.
        last_preempt_reason: Last deterministic preemption reason.
    """

    model_config = ConfigDict(extra="forbid")

    control_epoch: int = Field(default=0, ge=0)
    active_action_id: str | None = None
    status: ControlStatus = ControlStatus.IDLE
    last_preempt_reason: str | None = None


class ExternalUserEvent(BaseModel):
    """One idempotent user event delivered independently of graph execution.

    Args:
        episode_id: Target runtime episode.
        event_type: Deterministic or semantic event family.
        content: User text or structured task update.
        expected_task_revision: Optional optimistic concurrency guard.
        interaction_id: Correlation id for ``user_response``.
        event_id: Stable idempotency key.
        timestamp: Source creation time.
        metadata: Transport diagnostics excluded from model targets.
    """

    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=1)
    event_type: ExternalUserEventType
    content: Any = None
    expected_task_revision: int | None = Field(default=None, ge=0)
    interaction_id: str | None = None
    event_id: str = Field(default_factory=lambda: f"user-event:{uuid4().hex}")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = TASK_CONTRACT_SCHEMA_VERSION


def initial_task_state(
    *,
    episode_id: str,
    initial_context: dict[str, Any],
    route_session_id: str | None = None,
) -> TaskState:
    """Build the initial revision-zero task state.

    Args:
        episode_id: Parent episode id.
        initial_context: Runtime reset context.
        route_session_id: Optional existing route session.

    Returns:
        Validated initial task state.
    """

    instruction = str(initial_context.get("instruction") or initial_context.get("request") or "")
    return TaskState(
        task_id=str(initial_context.get("task_id") or f"task:{episode_id}"),
        active_instruction=instruction,
        goal_spec=dict(initial_context.get("goal_spec") or {}),
        constraints=dict(initial_context.get("constraints") or {}),
        route_session_id=route_session_id,
        route_revision=int(initial_context.get("route_revision") or 0),
    )
