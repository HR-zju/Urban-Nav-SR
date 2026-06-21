from __future__ import annotations

import tempfile
from pathlib import Path
import unittest


from mcp_server.agent.schemas import AgentEvalCase, ToolCall
from mcp_server.agent.tool_parser import parse_agent_action, parse_finish_text, parse_tool_call_text
from mcp_server.agent.tool_schema import validate_tool_call
from mcp_server.eval.cases import load_eval_cases, write_eval_cases
from mcp_server.eval.exporters import trace_to_ms_swift_row
from mcp_server.eval.metrics import compute_trace_metrics
from mcp_server.eval.runners import RouteServerTeacherRunner, RuleRunner


class AgentProtocolTest(unittest.TestCase):
    def test_case_roundtrip_jsonl(self) -> None:
        case = _sample_case()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "cases.jsonl"
            write_eval_cases([case], path)
            loaded = load_eval_cases(path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].case_id, "case-1")
        self.assertEqual(loaded[0].images, ["/data/fpv.jpg", "/data/bev.jpg"])

    def test_parser_rejects_tool_call_inside_think(self) -> None:
        valid_text = '<tool_call>{"tool":"map_tool","op":"search_poi","query":"coffee","location":"120,30"}</tool_call>'
        call, errors = parse_tool_call_text(valid_text)
        self.assertEqual(call.tool, "map_tool")
        self.assertEqual(errors, [])

        bad_text = '<think><tool_call>{"tool":"map_tool","op":"search_poi","query":"coffee"}</tool_call></think>'
        parsed = parse_agent_action(bad_text)
        self.assertEqual(parsed.action_type, "invalid")
        self.assertIn("tool_call_inside_reasoning", parsed.errors)

    def test_finish_parser(self) -> None:
        finish, errors = parse_finish_text('<finish>{"route_action":"continue"}</finish>')
        self.assertEqual(errors, [])
        self.assertEqual(finish["route_action"], "continue")

    def test_tool_schema_validation(self) -> None:
        ok = ToolCall(tool="map_tool", arguments={"op": "search_poi", "query": "restroom", "location": "120,30"})
        self.assertTrue(validate_tool_call(ok)[0])
        bad = ToolCall(tool="map_tool", arguments={"op": "search_poi"})
        self.assertFalse(validate_tool_call(bad)[0])


class AgentEvalFlowTest(unittest.IsolatedAsyncioTestCase):
    async def test_rule_runner_metrics_and_export(self) -> None:
        case = _sample_case()
        trace = await RuleRunner().run_case(case)
        metric_result = compute_trace_metrics(trace, case)
        row = trace_to_ms_swift_row(trace, case)

        self.assertEqual(metric_result.metrics["parse_success"], 1.0)
        self.assertEqual(metric_result.metrics["tool_op_accuracy"], 1.0)
        self.assertIn("messages", row)
        self.assertIn("tools", row)
        self.assertEqual(row["images"], ["/data/fpv.jpg", "/data/bev.jpg"])
        self.assertEqual(row["pixel_trajectory"], [[0.5, 0.9], [0.55, 0.75]])

    async def test_routeserver_teacher_runner_offline_builds_trace(self) -> None:
        case = AgentEvalCase(
            case_id="teacher-1",
            user_request="把目的地改成市图书馆",
            observation={
                "route_session_id": "route_demo",
                "current_location": "120.0,30.0",
                "destination": "120.012,30.011",
            },
            ground_truth={"expected_tool_ops": ["replace_destination"]},
        )
        trace = await RouteServerTeacherRunner(use_llm=False).run_case(case)
        row = trace_to_ms_swift_row(trace, case)

        self.assertEqual(trace.metrics["parse_success"], 1.0)
        self.assertEqual(trace.metrics["tool_op_accuracy"], 1.0)
        self.assertEqual(trace.metadata["teacher_internal"]["selected_student_tool_call"]["arguments"]["op"], "replace_destination")
        self.assertEqual(row["sidewalk_meta"]["model_run"]["runner"], "routeserver_teacher")


def _sample_case() -> AgentEvalCase:
    return AgentEvalCase(
        case_id="case-1",
        user_request="帮我找一个可以休息的地方",
        observation={
            "images": {"fpv": "/data/fpv.jpg", "bev": "/data/bev.jpg"},
            "current_location": "120.0,30.0",
        },
        ground_truth={
            "expected_tool_ops": ["search_poi"],
            "pixel_trajectory": [[0.5, 0.9], [0.55, 0.75]],
        },
    )


if __name__ == "__main__":
    unittest.main()
