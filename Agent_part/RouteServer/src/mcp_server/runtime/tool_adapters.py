"""Adapters connecting existing RouteServer executors to Episode Runtime V2."""

from __future__ import annotations

from typing import Any

from mcp_server.agent.schemas import AgentStage2Case, ToolCall
from mcp_server.contracts.task import TaskState
from mcp_server.contracts.tool import ToolCallV2, ToolResponseV2
from mcp_server.tools.registry import DEFAULT_TOOL_REGISTRY, ToolSpecRegistry


_SESSION_BOUND_MAP_OPS = {
    "get_route",
    "get_route_context",
    "get_route_session",
    "add_via",
    "add_via_and_replan",
    "remove_via_and_replan",
    "replace_destination",
    "replan_from_current_pose",
}
_MUTATING_ROUTE_OPS = {
    "create_route_session",
    "add_via",
    "add_via_and_replan",
    "remove_via_and_replan",
    "replace_destination",
    "replan_from_current_pose",
}


def bind_tool_call_context(call: ToolCallV2, state: dict[str, Any]) -> ToolCallV2:
    """Bind route identity and mutation guards from authoritative runtime state."""

    if call.tool != "map_tool":
        return call
    task = TaskState.model_validate(state.get("task_state") or {"task_id": "unbound"})
    arguments = dict(call.arguments)
    # active session 由 runtime 所有，不允许模型通过幻觉 session_id 访问其他路线。
    if call.op in _SESSION_BOUND_MAP_OPS and task.route_session_id:
        arguments["session_id"] = task.route_session_id
    if call.op in _MUTATING_ROUTE_OPS:
        arguments.setdefault("expected_state_revision", task.route_revision)
        arguments.setdefault("idempotency_key", call.call_id or f"{state.get('decision_id')}:{call.op}")
        latest_event = state.get("latest_external_event") or {}
        if latest_event.get("event_id"):
            arguments.setdefault("source_event_id", latest_event["event_id"])
    return call.model_copy(update={"arguments": arguments})


def validate_tool_call_context(call: ToolCallV2, state: dict[str, Any]) -> list[str]:
    """Reject operations that would bypass an active route-session transaction."""

    if call.tool != "map_tool":
        return []
    task = TaskState.model_validate(state.get("task_state") or {"task_id": "unbound"})
    if (
        task.route_session_id
        and call.op in {"plan_route", "create_route_session"}
        and not bool(task.constraints.get("allow_new_route_session"))
    ):
        return ["active_route_session_requires_session_operation"]
    return []


class LiveToolExecutorAdapter:
    """Adapt the existing asynchronous live executor to the V2 protocol.

    Args:
        executor: Existing ``mcp_server.agent.tool_executor.ToolExecutor``.
        case: Optional Stage2 case used by deterministic image tools.
        registry: Tool registry used for validation and canonicalization.
    """

    def __init__(
        self,
        executor: Any,
        *,
        case: AgentStage2Case | None = None,
        registry: ToolSpecRegistry = DEFAULT_TOOL_REGISTRY,
    ) -> None:
        self.executor = executor
        self.case = case
        self.registry = registry

    async def execute(self, call: ToolCallV2, state: dict[str, Any]) -> ToolResponseV2:
        """Validate and execute one live tool call.

        Args:
            call: V2 tool call emitted by System2.
            state: Current episode state; retained for protocol compatibility.

        Returns:
            Canonical V2 tool response.
        """

        bound_call = bind_tool_call_context(call, state)
        valid, errors, arguments = self.registry.validate_call(bound_call.model_dump(mode="python"))
        errors = [*errors, *validate_tool_call_context(bound_call, state)]
        valid = valid and not errors
        if not valid:
            return ToolResponseV2(
                tool=bound_call.tool,
                op=bound_call.op,
                success=False,
                error={"code": "invalid_tool_call", "messages": errors},
                call_id=bound_call.call_id,
                executor="live",
            )
        legacy_call = ToolCall(
            tool=bound_call.tool,
            arguments=dict(arguments or {}),
            call_id=bound_call.call_id,
        )
        response = await self.executor.execute(legacy_call, case=self.case)
        return ToolResponseV2.from_legacy(response, op=legacy_call.op, executor="live")


class UrbanNavReplayToolExecutorAdapter:
    """Adapt one case-local frozen UrbanNav executor to the V2 protocol.

    Args:
        executor: Existing ``ReplayUrbanNavToolExecutor``.
        case: Frozen UrbanNav benchmark case.
        tool_state: Case-local state returned by ``executor.new_state(case)``.
        registry: Tool registry used for validation and canonicalization.
    """

    def __init__(
        self,
        executor: Any,
        *,
        case: Any,
        tool_state: Any,
        registry: ToolSpecRegistry = DEFAULT_TOOL_REGISTRY,
    ) -> None:
        if tool_state is None:
            raise ValueError("UrbanNav replay tool_state must not be None")
        self.executor = executor
        self.case = case
        self.tool_state = tool_state
        self.registry = registry

    async def execute(self, call: ToolCallV2, state: dict[str, Any]) -> ToolResponseV2:
        """Validate and execute one frozen replay tool call.

        Args:
            call: V2 tool call emitted by System2.
            state: Current episode state; replay state is held case-locally.

        Returns:
            Canonical V2 tool response with a replay key when available.
        """

        bound_call = bind_tool_call_context(call, state)
        valid, errors, arguments = self.registry.validate_call(bound_call.model_dump(mode="python"))
        errors = [*errors, *validate_tool_call_context(bound_call, state)]
        valid = valid and not errors
        if not valid:
            return ToolResponseV2(
                tool=bound_call.tool,
                op=bound_call.op,
                success=False,
                error={"code": "invalid_tool_call", "messages": errors},
                call_id=bound_call.call_id,
                executor="urbannav_replay",
            )
        legacy_call = ToolCall(
            tool=bound_call.tool,
            arguments=dict(arguments or {}),
            call_id=bound_call.call_id,
        )
        response = self.executor.execute(case=self.case, state=self.tool_state, call=legacy_call)
        return ToolResponseV2.from_legacy(response, op=legacy_call.op, executor="urbannav_replay")
