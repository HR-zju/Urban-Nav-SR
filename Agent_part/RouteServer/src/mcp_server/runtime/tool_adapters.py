"""Adapters connecting existing RouteServer executors to Episode Runtime V2."""

from __future__ import annotations

from typing import Any

from mcp_server.agent.schemas import AgentStage2Case, ToolCall
from mcp_server.contracts.tool import ToolCallV2, ToolResponseV2
from mcp_server.tools.registry import DEFAULT_TOOL_REGISTRY, ToolSpecRegistry


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

        del state
        valid, errors, arguments = self.registry.validate_call(call.model_dump(mode="python"))
        if not valid:
            return ToolResponseV2(
                tool=call.tool,
                op=call.op,
                success=False,
                error={"code": "invalid_tool_call", "messages": errors},
                call_id=call.call_id,
                executor="live",
            )
        legacy_call = ToolCall(tool=call.tool, arguments=dict(arguments or {}), call_id=call.call_id)
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

        del state
        valid, errors, arguments = self.registry.validate_call(call.model_dump(mode="python"))
        if not valid:
            return ToolResponseV2(
                tool=call.tool,
                op=call.op,
                success=False,
                error={"code": "invalid_tool_call", "messages": errors},
                call_id=call.call_id,
                executor="urbannav_replay",
            )
        legacy_call = ToolCall(tool=call.tool, arguments=dict(arguments or {}), call_id=call.call_id)
        response = self.executor.execute(case=self.case, state=self.tool_state, call=legacy_call)
        return ToolResponseV2.from_legacy(response, op=legacy_call.op, executor="urbannav_replay")
