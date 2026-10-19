from __future__ import annotations

import json
import unittest

from mcp_server.agent.schemas import AGENT_STAGE2_CASE_SCHEMA_VERSION, AgentStage2Case, ToolCall
from mcp_server.agent.tool_executor import ToolExecutor
from mcp_server.agent.tool_schema import IMAGE_TOOL_NAME, IMAGE_TOOL_OPS, get_tool_definitions, validate_tool_call
from mcp_server.eval.agent_tools.exporters import trace_to_ms_swift_stage2_row
from mcp_server.teacher.agent_tools.agentic_stage2_trace_builder import AgenticStage2TraceBuilder


class ImageToolPolicyTest(unittest.IsolatedAsyncioTestCase):
    """Lock the Stage2 image-tool decision: no student-visible inspect op."""

    def test_image_tool_schema_exposes_only_crop_and_zoom_in(self) -> None:
        tools = get_tool_definitions()
        image_tool = _tool_definition(tools, IMAGE_TOOL_NAME)
        enum_values = image_tool["function"]["parameters"]["properties"]["op"]["enum"]
        serialized = json.dumps(image_tool, ensure_ascii=False)

        self.assertEqual(set(enum_values), IMAGE_TOOL_OPS)
        self.assertEqual(IMAGE_TOOL_OPS, {"crop", "zoom_in"})
        self.assertNotIn("inspect", enum_values)
        self.assertNotIn("image_tool.inspect", serialized)

    def test_exported_ms_swift_tools_do_not_expose_inspect(self) -> None:
        case = _sample_case()
        trace = AgenticStage2TraceBuilder().build_trace(case, call=None)
        row = trace_to_ms_swift_stage2_row(trace, case)
        tools = json.loads(row["tools"])
        image_tool = _tool_definition(tools, IMAGE_TOOL_NAME)
        enum_values = image_tool["function"]["parameters"]["properties"]["op"]["enum"]

        self.assertEqual(set(enum_values), {"crop", "zoom_in"})
        self.assertNotIn("inspect", json.dumps(tools, ensure_ascii=False))

    def test_validate_tool_call_rejects_image_tool_inspect(self) -> None:
        call = ToolCall(
            tool=IMAGE_TOOL_NAME,
            arguments={"op": "inspect", "image": "fpv", "bbox": [0.0, 0.0, 1.0, 1.0]},
        )

        is_valid, errors = validate_tool_call(call)

        self.assertFalse(is_valid)
        self.assertIn("unsupported_op:image_tool.inspect", errors)

    async def test_executor_returns_invalid_tool_call_for_inspect(self) -> None:
        call = ToolCall(
            tool=IMAGE_TOOL_NAME,
            arguments={"op": "inspect", "image": "fpv", "bbox": [0.0, 0.0, 1.0, 1.0]},
        )

        response = await ToolExecutor().execute(call, case=_sample_case())

        self.assertFalse(response.success)
        self.assertEqual(response.error["code"], "invalid_tool_call")
        self.assertIn("unsupported_op:image_tool.inspect", response.error["messages"])


def _tool_definition(tools: list[dict], name: str) -> dict:
    """Return one tool definition by function name.

    Args:
        tools: Decoded tool definition list.
        name: Tool function name.

    Returns:
        Matching tool definition.

    Raises:
        AssertionError: If the tool is absent.
    """

    for tool in tools:
        if tool.get("function", {}).get("name") == name:
            return tool
    raise AssertionError(f"Missing tool definition: {name}")


def _sample_case() -> AgentStage2Case:
    """Return a minimal Stage2 case for image-tool policy tests."""

    return AgentStage2Case(
        case_id="image-policy",
        schema_version=AGENT_STAGE2_CASE_SCHEMA_VERSION,
        user_request="继续沿当前可见人行道安全前进",
        observation={
            "images": {"fpv": "/data/fpv.jpg", "bev": "/data/bev.jpg"},
            "current_location": "120.0,30.0",
            "force_no_tool": True,
        },
        ground_truth={
            "tool_policy": "no_tool",
            "pixel_trajectory": [[0.5, 0.9], [0.5, 0.82]],
        },
    )


if __name__ == "__main__":
    unittest.main()
