from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import json


from mcp_server.agent.schemas import AGENT_STAGE2_CASE_SCHEMA_VERSION, AgentStage2Case, ToolCall
from mcp_server.agent.tool_parser import parse_agent_action, parse_finish_text, parse_tool_call_text
from mcp_server.agent.tool_schema import validate_tool_call
from mcp_server.eval.cases import load_stage2_cases, write_stage2_cases
from mcp_server.eval.exporters import trace_to_ms_swift_stage2_row
from mcp_server.eval.metrics import compute_trace_metrics
from mcp_server.eval.phase1_archive_adapter import phase1_archive_row_to_stage2_case
from mcp_server.eval.runners import RouteServerTeacherRunner, RuleRunner
from mcp_server.eval.stage2_validator import validate_ms_swift_stage2_row, validate_stage2_trace
from mcp_server.teacher.agentic_stage2_trace_builder import AgenticStage2TraceBuilder


class AgentProtocolTest(unittest.TestCase):
    def test_case_roundtrip_jsonl(self) -> None:
        case = _sample_case()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "cases.jsonl"
            write_stage2_cases([case], path)
            loaded = load_stage2_cases(path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].case_id, "case-1")
        self.assertEqual(loaded[0].images, ["/data/fpv.jpg", "/data/bev.jpg"])
        self.assertIsInstance(loaded[0], AgentStage2Case)

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
        row = trace_to_ms_swift_stage2_row(trace, case)

        self.assertEqual(metric_result.metrics["parse_success"], 1.0)
        self.assertEqual(metric_result.metrics["tool_op_accuracy"], 1.0)
        self.assertEqual(metric_result.metrics["stage2_role_order_valid"], 1.0)
        self.assertEqual(metric_result.metrics["trajectory_readout_order_valid"], 1.0)
        self.assertEqual(metric_result.metrics["final_cot_validity"], 1.0)
        self.assertIn("messages", row)
        self.assertIn("tools", row)
        self.assertIsInstance(row["tools"], str)
        self.assertEqual(json.loads(row["tools"])[0]["function"]["name"], "map_tool")
        self.assertEqual(row["images"], ["/data/fpv.jpg", "/data/bev.jpg"])
        self.assertEqual(row["pixel_trajectory"], [[0.5, 0.9], [0.55, 0.75]])
        roles = [message["role"] for message in row["messages"]]
        self.assertIn("tool_call", roles)
        self.assertIn("tool_response", roles)
        self.assertNotIn("tool", roles)
        tool_call = next(message for message in row["messages"] if message["role"] == "tool_call")
        self.assertEqual(json.loads(tool_call["content"])["name"], "map_tool")
        readout_index = next(index for index, message in enumerate(row["messages"]) if "[TRAJ_0]" in message["content"])
        tool_response_index = next(index for index, message in enumerate(row["messages"]) if message["role"] == "tool_response")
        self.assertGreater(readout_index, tool_response_index)
        self.assertTrue(validate_stage2_trace(trace, case).valid)
        self.assertTrue(validate_ms_swift_stage2_row(row).valid)

    async def test_agentic_stage2_trace_builder_normalizes_roles(self) -> None:
        case = _sample_case()
        call = ToolCall(tool="map_tool", arguments={"op": "search_poi", "query": "rest stop", "location": "120,30"})
        trace = AgenticStage2TraceBuilder().build_trace(case, call=call)
        self.assertEqual([turn.role for turn in trace.turns], ["user", "assistant", "tool_call", "tool_response", "user", "assistant"])
        self.assertEqual(trace.turns[2].content["name"], "map_tool")
        self.assertIn("[TRAJ_0]", trace.turns[4].content)
        self.assertTrue(validate_stage2_trace(trace, case).valid)

    async def test_routeserver_teacher_runner_offline_builds_trace(self) -> None:
        case = AgentStage2Case(
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
        row = trace_to_ms_swift_stage2_row(trace, case)

        self.assertEqual(trace.metrics["parse_success"], 1.0)
        self.assertEqual(trace.metrics["tool_op_accuracy"], 1.0)
        self.assertEqual(trace.metrics["stage2_role_order_valid"], 1.0)
        self.assertEqual(trace.metadata["teacher_internal"]["selected_student_tool_call"]["arguments"]["op"], "replace_destination")
        self.assertEqual([turn.role for turn in trace.turns], ["user", "assistant", "tool_call", "tool_response", "user", "assistant"])
        self.assertEqual(trace.turns[2].content["name"], "map_tool")
        self.assertIn("[TRAJ_15]", trace.turns[4].content)
        self.assertIn("<ToolResultUse>", trace.turns[5].content)
        self.assertIn("<finish>", trace.turns[5].content)
        tool_response = json.loads(next(message["content"] for message in row["messages"] if message["role"] == "tool_response"))
        self.assertTrue(tool_response["success"])
        self.assertEqual(row["sidewalk_meta"]["model_run"]["runner"], "routeserver_teacher")

    async def test_phase1_archive_case_runs_with_real_tool_executor_path(self) -> None:
        phase1_row = _sample_phase1_archive_row()
        case = phase1_archive_row_to_stage2_case(
            phase1_row,
            index=7,
            source_path="/archive/manifests/train.jsonl",
        )
        app_context = SimpleNamespace(provider="fake_test_map", osm_client=_FakeMapClient())
        trace = await RouteServerTeacherRunner(
            app_context=app_context,
            use_llm=False,
            execute_tools=True,
            enable_recommendation=False,
        ).run_case(case)
        row = trace_to_ms_swift_stage2_row(trace, case)
        tool_response = json.loads(next(message["content"] for message in row["messages"] if message["role"] == "tool_response"))

        self.assertEqual(case.case_id, "phase1-stage2-clip-a_000001")
        self.assertEqual(case.user_request, "Navigate safely on the sidewalk.")
        self.assertTrue(tool_response["success"])
        self.assertEqual(tool_response["tool"], "map_tool")
        self.assertIn("<Scene>", trace.turns[5].content)
        self.assertIn("<ToolResultUse>", trace.turns[5].content)
        self.assertEqual(row["pixel_trajectory"], phase1_row["pixel_trajectory"])
        self.assertEqual(row["trajectory_fpv_t"], phase1_row["trajectory_fpv_t"])
        self.assertEqual(row["semantic_segmentation_path"], "/archive/manifests/sidecars/seg/000001.png")
        self.assertTrue(validate_stage2_trace(trace, case).valid)
        self.assertTrue(validate_ms_swift_stage2_row(row).valid)


def _sample_case() -> AgentStage2Case:
    return AgentStage2Case(
        case_id="case-1",
        schema_version=AGENT_STAGE2_CASE_SCHEMA_VERSION,
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


def _sample_phase1_archive_row() -> dict:
    trajectory = [[round(0.5 + index * 0.002, 4), round(0.9 - index * 0.025, 4)] for index in range(16)]
    response = (
        "<think>\n"
        '<Scene>{"summary":"urban sidewalk scene","visibility":"clear","occlusion_sources":[],"uncertainty":"low"}</Scene>\n'
        '<FPVScene>{"navigable_support":["sidewalk"],"conditional_support":[],"risk_objects":[],"visibility":"clear","local_risk":"low"}</FPVScene>\n'
        '<BEVContext>{"scene_summary":"local BEV route context","route_summary":"route continues forward","route_direction_bin":"front","intent":"go_straight","target_direction":"front","bev_anchor":"front route segment"}</BEVContext>\n'
        '<CrossView>{"bev_anchor":"front route","fpv_anchor":"sidewalk center","near_reference":"curb","blind_area_context":{"left":"clear","right":"clear","front":"clear"},"fpv_route_region":"center","route_to_fpv_direction":"forward_center","safe_corridor":"center_corridor","alignment":"aligned","confidence":"high"}</CrossView>\n'
        '<Plan>{"motion_primitive":"go_straight","lateral_preference":"center","avoidance_side":"none","speed_policy":"normal","primary_route":"follow sidewalk","safety_policy":"stay on visible support","contingency":"slow if blocked"}</Plan>\n'
        "</think>\n"
        f'<finish>{json.dumps({"trajectory": trajectory, "intent": "go_straight", "confidence_level": "normal", "trajectory_type": "forward", "trajectory_validity": "valid"}, ensure_ascii=False, separators=(",", ":"))}</finish>'
    )
    return {
        "messages": [
            {"role": "system", "content": "Phase1 system prompt"},
            {
                "role": "user",
                "content": (
                    "<image><image>\n"
                    "Task: Sidewalk navigation structured reasoning and trajectory prediction\n"
                    "Instruction: Navigate safely on the sidewalk.\n"
                ),
            },
            {"role": "assistant", "content": response},
        ],
        "images": ["media/fpv/000001.jpg", "media/bev/000001.jpg"],
        "pixel_trajectory": trajectory,
        "trajectory_fpv_t": [[point[0], point[1], round(index * 0.25, 2)] for index, point in enumerate(trajectory)],
        "semantic_segmentation_path": "sidecars/seg/000001.png",
        "sidewalk_meta": {
            "archive_schema_version": "sidewalk_phase1_archive_v1",
            "source_id": "clip-a_000001",
            "clip_id": "clip-a",
            "frame_name": "000001",
            "lat": 30.0,
            "lng": 120.0,
            "intent_gt": "go_straight",
            "trajectory_source": "calibrated_pose_projection",
        },
    }


class _FakeMapClient:
    async def search_semantic_pois(
        self,
        *,
        location: str,
        keywords: str,
        radius: int = 1000,
        sortrule: str = "distance",
        limit: int = 10,
    ) -> list[dict]:
        return [
            {
                "name": keywords or "sidewalk candidate",
                "location": location or "120.0,30.0",
                "distance_m": 32,
                "source": "fake_test_map",
            }
        ]


if __name__ == "__main__":
    unittest.main()
