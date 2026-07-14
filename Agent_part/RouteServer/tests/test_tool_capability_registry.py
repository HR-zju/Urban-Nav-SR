from __future__ import annotations

import asyncio
import unittest

from mcp_server.agent.schemas import ToolCall, ToolResponse
from mcp_server.agent.tool_executor import ToolExecutor
from mcp_server.agent.tool_schema import get_tool_definitions, validate_tool_call
from mcp_server.contracts.tool import ToolCallV2
from mcp_server.eval.urbannav_replay_tool_executor import ReplayUrbanNavToolExecutor
from mcp_server.runtime import LiveToolExecutorAdapter
from mcp_server.tools.registry import DEFAULT_TOOL_REGISTRY


class ToolCapabilityRegistryTest(unittest.TestCase):
    """Verify schema, live executor and replay executor remain conformant."""

    def test_executor_capability_profiles_match_registry(self) -> None:
        """Both executor implementations should match their declared profiles."""

        DEFAULT_TOOL_REGISTRY.assert_capabilities("live", ToolExecutor.supported_capabilities())
        DEFAULT_TOOL_REGISTRY.assert_capabilities(
            "urbannav_replay",
            ReplayUrbanNavToolExecutor.supported_capabilities(),
        )

    def test_generated_tool_schemas_are_strict_and_keep_names(self) -> None:
        """Generated OpenAI schemas should preserve names and reject unknown keys."""

        definitions = get_tool_definitions()
        names = {item["function"]["name"] for item in definitions}

        self.assertEqual(names, {"map_tool", "image_tool", "user_interaction_tool"})
        self.assertTrue(all(item["function"]["parameters"]["additionalProperties"] is False for item in definitions))
        valid, errors = validate_tool_call(
            ToolCall(tool="map_tool", arguments={"op": "search_destination", "query": "coffee", "invented": 1})
        )
        self.assertFalse(valid)
        self.assertTrue(any(error.startswith("invalid_argument:invented") for error in errors), errors)

    def test_legacy_aliases_normalize_before_strict_validation(self) -> None:
        """Known aliases should remain compatible without permitting arbitrary fields."""

        valid, errors, canonical = DEFAULT_TOOL_REGISTRY.validate_call(
            ToolCall(
                tool="map_tool",
                arguments={
                    "op": "replan_from_current_pose",
                    "current_lat": 30.0,
                    "current_lng": 120.0,
                    "route_session_id": "session-1",
                },
            )
        )

        self.assertTrue(valid, errors)
        self.assertEqual(canonical["current_pose"], {"lat": 30.0, "lng": 120.0})
        self.assertEqual(canonical["session_id"], "session-1")

    def test_live_adapter_returns_canonical_v2_response(self) -> None:
        """Episode runtime adapters should bridge legacy responses without reflection."""

        legacy = _LegacyExecutor()
        adapter = LiveToolExecutorAdapter(legacy)
        response = asyncio.run(adapter.execute(
            ToolCallV2(
                tool="map_tool",
                arguments={"op": "get_route_context", "session_id": "session-1"},
                call_id="call-1",
            ),
            {},
        ))

        self.assertTrue(response.success)
        self.assertEqual(response.op, "get_route_context")
        self.assertEqual(response.executor, "live")
        self.assertEqual(legacy.calls[0].arguments["session_id"], "session-1")


class _LegacyExecutor:
    """Minimal legacy executor used to verify the V2 runtime adapter."""

    def __init__(self) -> None:
        self.calls: list[ToolCall] = []

    async def execute(self, call: ToolCall, *, case=None) -> ToolResponse:
        """Return a deterministic legacy response for one canonical call."""

        del case
        self.calls.append(call)
        return ToolResponse(
            tool=call.tool,
            success=True,
            result={"op": call.op, "session_id": call.arguments.get("session_id")},
            call_id=call.call_id,
        )


if __name__ == "__main__":
    unittest.main()
