"""Pydantic contracts for versioned tool calls and responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


TOOL_CONTRACT_SCHEMA_VERSION = "urbannav_tool_contract_v2"


class ToolCallV2(BaseModel):
    """Canonical model-requested tool call.

    Args:
        tool: Stable student-visible meta-tool name.
        arguments: Operation arguments validated by ``ToolSpecRegistry``.
        call_id: Stable id used for tracing and idempotency.
        schema_version: Tool contract version.
    """

    model_config = ConfigDict(extra="forbid")

    tool: str = Field(min_length=1)
    arguments: dict[str, Any]
    call_id: str | None = None
    schema_version: str = TOOL_CONTRACT_SCHEMA_VERSION

    @property
    def op(self) -> str | None:
        """Return the requested operation name.

        Returns:
            Stripped operation name, or ``None`` when absent.
        """

        value = self.arguments.get("op")
        return str(value).strip() if value is not None else None


class ToolResponseV2(BaseModel):
    """Canonical deterministic tool execution response.

    Args:
        tool: Stable student-visible meta-tool name.
        op: Executed operation name.
        success: Whether execution completed successfully.
        result: Model-visible result payload.
        error: Structured failure payload.
        artifacts: External artifact references.
        call_id: Id of the originating tool call.
        executor: Executor profile, such as ``live`` or ``urbannav_replay``.
        latency_ms: Measured executor latency in milliseconds.
        state_revision: Route/environment revision after execution.
        state_delta: Explicit state mutation caused by the operation.
        idempotency_key: Key used to deduplicate side-effecting calls.
        replay_key: Frozen replay record key when available.
        schema_version: Tool contract version.
    """

    model_config = ConfigDict(extra="forbid")

    tool: str = Field(min_length=1)
    op: str | None = None
    success: bool
    result: Any = Field(default_factory=dict)
    error: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    call_id: str | None = None
    executor: str | None = None
    latency_ms: float | None = Field(default=None, ge=0.0)
    state_revision: str | int | None = None
    state_delta: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = None
    replay_key: str | None = None
    schema_version: str = TOOL_CONTRACT_SCHEMA_VERSION

    @classmethod
    def from_legacy(
        cls,
        response: Any,
        *,
        op: str | None = None,
        executor: str | None = None,
        latency_ms: float | None = None,
    ) -> "ToolResponseV2":
        """Build a v2 response from a legacy dataclass or mapping.

        Args:
            response: Legacy ``ToolResponse`` or JSON-compatible mapping.
            op: Executed operation when it is not present in the response.
            executor: Executor profile name.
            latency_ms: Optional measured latency.

        Returns:
            Validated ``ToolResponseV2``.

        Raises:
            TypeError: If ``response`` cannot be converted to a mapping.
        """

        if hasattr(response, "to_dict"):
            payload = response.to_dict()
        elif isinstance(response, dict):
            payload = dict(response)
        else:
            raise TypeError("response must be a mapping or expose to_dict()")
        result = payload.get("result")
        result_mapping = result if isinstance(result, dict) else {}
        return cls(
            tool=str(payload.get("tool") or ""),
            op=op or result_mapping.get("op"),
            success=bool(payload.get("success")),
            result=result if result is not None else {},
            error=payload.get("error"),
            artifacts=list(payload.get("artifacts") or []),
            call_id=payload.get("call_id"),
            executor=executor,
            latency_ms=latency_ms,
            state_revision=result_mapping.get("state_revision") or result_mapping.get("route_revision"),
            state_delta=dict(result_mapping.get("state_delta") or {}),
            replay_key=result_mapping.get("replay_key") or result_mapping.get("cache_key"),
        )
