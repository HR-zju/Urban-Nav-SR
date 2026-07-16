"""Typed policy, safety and environment boundaries for episode execution."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mcp_server.contracts.tool import ToolCallV2, ToolResponseV2
from mcp_server.runtime.system1_contract import (
    SYSTEM1_AXIS_NAMES_EGO_POSE_XYT_FLU,
    SYSTEM1_COORDINATE_FRAME_CURRENT_EGO_FLU,
    SYSTEM1_REPRESENTATION_CUMULATIVE_RELATIVE_POSE,
    SYSTEM1_TARGET_FIELD_TRAJECTORY_EGO_T,
    SYSTEM1_TARGET_FORMAT_EGO_POSE_XYT_FLU_V1,
)


class System2ActionType(str, Enum):
    """Actions available to the slow System2 policy."""

    TOOL_CALL = "tool_call"
    ASK_USER = "ask_user"
    NAVIGATE = "navigate"
    COMPLETE_TASK = "complete_task"
    ABORT_TASK = "abort_task"


class TerminalOutcome(str, Enum):
    """Outcome category requested by a terminal System2 action."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class TerminalDecision(BaseModel):
    """Structured task-terminal request emitted by System2.

    Args:
        outcome: Requested episode outcome. Runtime validation remains
            authoritative, especially for successful completion.
        reason_code: Stable machine-readable completion or abort reason.
        reason: Optional human-readable explanation.
        evidence_refs: Model-declared evidence references for auditing. These
            references never replace deterministic runtime verification.
        retryable: Whether a new task revision may retry the failed task.
    """

    model_config = ConfigDict(extra="forbid")

    outcome: TerminalOutcome
    reason_code: str = Field(min_length=1)
    reason: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    retryable: bool = False


class TerminalValidationResult(BaseModel):
    """Deterministic validation result for a terminal System2 request.

    Args:
        action_type: Canonical terminal action under validation.
        accepted: Whether runtime may commit the requested terminal outcome.
        reason_code: Stable validation reason.
        evidence_refs: Authoritative environment or route evidence references.
        metadata: Deterministic validator diagnostics.
    """

    model_config = ConfigDict(extra="forbid")

    action_type: System2ActionType
    accepted: bool
    reason_code: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class System2Action(BaseModel):
    """One typed System2 decision.

    Args:
        action_type: Selected action family.
        tool_call: Tool request for ``tool_call`` actions.
        user_request: HITL request for ``ask_user`` actions.
        finish: Symbolic assistant ``<finish>`` payload. The tag is a turn
            envelope and does not by itself mean task completion.
        terminal: Strongly typed request for ``complete_task`` or
            ``abort_task``.
        metadata: Diagnostic policy metadata.
    """

    model_config = ConfigDict(extra="forbid")

    action_type: System2ActionType
    tool_call: ToolCallV2 | None = None
    user_request: dict[str, Any] | None = None
    finish: dict[str, Any] | None = None
    terminal: TerminalDecision | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_terminal_actions(cls, value: Any) -> Any:
        """Normalize legacy ``finish``/``stop`` values at the input boundary.

        Args:
            value: Raw action mapping or an already validated action.

        Returns:
            Mapping using canonical terminal action values.
        """

        if not isinstance(value, dict):
            return value
        payload = dict(value)
        raw_action = payload.get("action_type")
        raw_text = raw_action.value if isinstance(raw_action, Enum) else str(raw_action or "")
        legacy_aliases = {
            "finish": System2ActionType.COMPLETE_TASK.value,
            "stop": System2ActionType.ABORT_TASK.value,
        }
        canonical = legacy_aliases.get(raw_text, raw_text)
        payload["action_type"] = canonical
        is_legacy = raw_text in legacy_aliases
        if is_legacy:
            metadata = dict(payload.get("metadata") or {})
            metadata.setdefault("legacy_action_type", raw_text)
            metadata.setdefault("legacy_normalized", True)
            payload["metadata"] = metadata

        if canonical not in {
            System2ActionType.COMPLETE_TASK.value,
            System2ActionType.ABORT_TASK.value,
        } or payload.get("terminal") is not None:
            return payload

        # 兼容旧 runtime 的 untyped finish 字段；canonical 新输出仍必须给出 reason_code。
        finish = payload.get("finish") if isinstance(payload.get("finish"), dict) else {}
        reason_code = finish.get("reason_code") or finish.get("stop_reason")
        if reason_code is None and is_legacy:
            reason_code = "legacy_system2_finish" if raw_text == "finish" else "legacy_system2_stop"
        if reason_code is not None:
            payload["terminal"] = {
                "outcome": (
                    TerminalOutcome.SUCCEEDED.value
                    if canonical == System2ActionType.COMPLETE_TASK.value
                    else TerminalOutcome.FAILED.value
                ),
                "reason_code": str(reason_code),
                "reason": finish.get("reason"),
                "evidence_refs": finish.get("evidence_refs") or [],
                "retryable": bool(finish.get("retryable", False)),
            }
        return payload

    @model_validator(mode="after")
    def _validate_payload(self) -> "System2Action":
        """Validate that the selected action has its required payload.

        Returns:
            Validated action.

        Raises:
            ValueError: If an action-specific payload is absent.
        """

        if self.action_type == System2ActionType.TOOL_CALL and self.tool_call is None:
            raise ValueError("tool_call action requires tool_call payload")
        if self.action_type == System2ActionType.ASK_USER and self.user_request is None:
            raise ValueError("ask_user action requires user_request payload")
        if self.action_type in {System2ActionType.COMPLETE_TASK, System2ActionType.ABORT_TASK}:
            if self.terminal is None:
                raise ValueError(f"{self.action_type.value} action requires terminal payload with reason_code")
            expected_outcome = (
                TerminalOutcome.SUCCEEDED
                if self.action_type == System2ActionType.COMPLETE_TASK
                else TerminalOutcome.FAILED
            )
            if self.terminal.outcome != expected_outcome:
                raise ValueError(
                    f"{self.action_type.value} requires terminal.outcome={expected_outcome.value}"
                )
        elif self.terminal is not None:
            raise ValueError("terminal payload is only valid for complete_task or abort_task")
        return self


class System1Output(BaseModel):
    """Metric cumulative FLU trajectory produced by System1.

    Args:
        trajectory_ego_t: ``[T,3]`` points
            ``[x_forward_m, y_left_m, yaw_rel_ccw_rad]`` in the current fixed
            ego reference frame.
        action_mask: Optional valid-step mask.
        runtime_commands: Optional controller-ready commands.
        task_revision: Task revision used to generate this chunk.
        route_revision: Route revision used to generate this chunk.
        control_epoch: Controller generation used to generate this chunk.
        observation_id: Observation used to generate this chunk.
        action_id: Stable execution lease id.
        metadata: Head and sampling metadata.
    """

    model_config = ConfigDict(extra="forbid")

    trajectory_ego_t: list[list[float]]
    action_mask: list[bool] | None = None
    runtime_commands: list[dict[str, Any]] = Field(default_factory=list)
    task_revision: int | None = Field(default=None, ge=0)
    route_revision: int | None = Field(default=None, ge=0)
    control_epoch: int | None = Field(default=None, ge=0)
    observation_id: str | None = None
    action_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("trajectory_ego_t")
    @classmethod
    def _validate_xyt(cls, value: list[list[float]]) -> list[list[float]]:
        """Require a finite non-empty canonical ``[T,3]`` trajectory.

        Args:
            value: Candidate trajectory.

        Returns:
            Validated trajectory.

        Raises:
            ValueError: If any waypoint is not three-dimensional.
        """

        if not value or any(len(point) != 3 for point in value):
            raise ValueError("trajectory_ego_t must have shape [T,3] with T > 0")
        if any(not math.isfinite(float(component)) for point in value for component in point):
            raise ValueError("trajectory_ego_t values must be finite")
        return value

    @model_validator(mode="after")
    def _bind_canonical_contract(self) -> "System1Output":
        """Attach immutable coordinate semantics to runtime output metadata.

        Returns:
            Output with canonical contract fields present in ``metadata``.

        Raises:
            ValueError: If caller metadata contradicts the production format.
        """

        # Runtime 对外只暴露 canonical 累计轨迹；调用方 metadata 不能改写其物理语义。
        expected = {
            "system1_target_field": SYSTEM1_TARGET_FIELD_TRAJECTORY_EGO_T,
            "system1_target_format": SYSTEM1_TARGET_FORMAT_EGO_POSE_XYT_FLU_V1,
            "representation": SYSTEM1_REPRESENTATION_CUMULATIVE_RELATIVE_POSE,
            "coordinate_frame": SYSTEM1_COORDINATE_FRAME_CURRENT_EGO_FLU,
            "axis_names": list(SYSTEM1_AXIS_NAMES_EGO_POSE_XYT_FLU),
            "normalized": False,
        }
        metadata = dict(self.metadata)
        for key, expected_value in expected.items():
            if key in metadata and metadata[key] != expected_value:
                raise ValueError(f"System1Output metadata {key}={metadata[key]!r} contradicts {expected_value!r}")
            metadata[key] = expected_value
        self.metadata = metadata
        return self


class SafetyResult(BaseModel):
    """Deterministic validator decision for one System1 chunk.

    Args:
        accepted: Whether the chunk may be executed.
        reason: Stable rejection or acceptance reason.
        metrics: Collision, social-risk and kinematic diagnostics.
        intervention: Optional safety-controller replacement command.
    """

    model_config = ConfigDict(extra="forbid")

    accepted: bool
    reason: str
    metrics: dict[str, float | bool | None] = Field(default_factory=dict)
    intervention: dict[str, Any] | None = None


class ObservationResult(BaseModel):
    """Environment reset observation.

    Args:
        observation: Model/runtime-visible multimodal observation references.
        info: Deterministic environment metadata.
        replay_key: Frozen record key when running offline.
        external_events: Frozen/simulator events available with this observation.
    """

    model_config = ConfigDict(extra="forbid")

    observation: dict[str, Any]
    info: dict[str, Any] = Field(default_factory=dict)
    replay_key: str | None = None
    external_events: list[dict[str, Any]] = Field(default_factory=list)


class EnvironmentStepResult(BaseModel):
    """Result of executing one accepted System1 chunk.

    Args:
        observation: Next multimodal observation references.
        reward: Deterministic per-step reward used by DAgger/RL export.
        done: Whether the episode has terminated.
        info: Arrival, safety and simulator diagnostics.
        replay_key: Frozen transition record key.
        external_events: Events deterministically emitted after this transition.
    """

    model_config = ConfigDict(extra="forbid")

    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)
    replay_key: str | None = None
    external_events: list[dict[str, Any]] = Field(default_factory=list)


class System2Policy(Protocol):
    """Protocol implemented by checkpoint or teacher System2 policies."""

    async def decide(self, state: dict[str, Any]) -> System2Action:
        """Return one typed System2 action for the current state."""


class TerminalValidator(Protocol):
    """Protocol for deterministic completion and abort validation."""

    async def validate(self, action: System2Action, state: dict[str, Any]) -> TerminalValidationResult:
        """Validate a terminal request against authoritative runtime state.

        Args:
            action: Canonical terminal System2 action.
            state: Current graph state containing trusted environment evidence.

        Returns:
            Deterministic terminal validation result.
        """


class System1Policy(Protocol):
    """Protocol implemented by CFM or direct-query System1 heads."""

    async def predict(self, state: dict[str, Any]) -> System1Output:
        """Return one continuous ego-frame action chunk."""


class SafetyValidator(Protocol):
    """Protocol for deterministic trajectory safety validators."""

    async def validate(self, output: System1Output, state: dict[str, Any]) -> SafetyResult:
        """Validate a proposed System1 chunk."""


class EpisodeEnvironment(Protocol):
    """Protocol for live, simulator and frozen replay environments."""

    async def reset(self, initial_context: dict[str, Any]) -> ObservationResult:
        """Reset an episode and return its first observation."""

    async def step(self, output: System1Output, state: dict[str, Any]) -> EnvironmentStepResult:
        """Execute one accepted System1 chunk."""

    async def preempt(self, *, reason: str) -> None:
        """Cancel the currently executing short-horizon chunk, if any.

        Args:
            reason: Stable preemption reason.

        Returns:
            None after controller acknowledgement.
        """

    async def halt(self, *, reason: str) -> None:
        """Issue an idempotent deterministic physical stop command.

        Args:
            reason: Stable terminal stop reason.

        Returns:
            None after physical-stop acknowledgement.
        """


class EpisodeToolExecutor(Protocol):
    """Protocol for registry-conformant live or replay tool executors."""

    async def execute(self, call: ToolCallV2, state: dict[str, Any]) -> ToolResponseV2:
        """Execute one validated tool call."""
