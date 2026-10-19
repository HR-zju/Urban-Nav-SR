from __future__ import annotations

import unittest

from mcp_server.agent.contracts import (
    RUNTIME_STATE_WAITING_FOR_USER,
    AgentStepResult,
    NavObservation,
    ToolAction,
)
from mcp_server.agent.schemas import AGENT_STAGE2_CASE_SCHEMA_VERSION, AgentStage2Case, AgentTrace, ToolCall
from mcp_server.agent.tool_executor import ToolExecutor
from mcp_server.agent.tool_schema import validate_tool_call
from mcp_server.eval.agent_tools.exporters import trace_to_ms_swift_stage2_row
from mcp_server.eval.agent_tools.runners import RuleRunner


class AgentContractTest(unittest.IsolatedAsyncioTestCase):
    async def test_stage2_case_roundtrips_nav_observation_contract(self) -> None:
        case = _sample_case()
        observation = case.to_nav_observation()
        rebuilt = AgentStage2Case.from_nav_observation(
            observation.to_dict(),
            ground_truth=case.ground_truth,
            tools=case.tools,
        )

        self.assertIsInstance(observation, NavObservation)
        self.assertEqual(observation.request["active_instruction"], case.user_request)
        self.assertEqual(observation.sensors["fpv_images"], ["/data/fpv.jpg"])
        self.assertEqual(rebuilt.case_id, case.case_id)
        self.assertEqual(rebuilt.images, case.images)
        self.assertEqual(rebuilt.metadata["nav_observation"]["observation_id"], case.case_id)

    async def test_tool_action_contract_wraps_user_interaction_tool(self) -> None:
        call = ToolCall(
            tool="user_interaction_tool",
            arguments={
                "op": "present_options",
                "message": "请选择目的地。",
                "options": [{"option_id": "opt_1", "label": "A"}],
                "recommended_option_id": "opt_1",
                "requires_user_response": True,
            },
        )

        action = ToolAction.from_tool_call(call)

        self.assertEqual(action.action_type, "user_interaction")
        self.assertEqual(action.to_tool_call().tool, "user_interaction_tool")
        self.assertTrue(validate_tool_call(call)[0])

    async def test_executor_returns_pending_user_interaction_contract(self) -> None:
        call = ToolCall(
            tool="user_interaction_tool",
            arguments={
                "op": "present_options",
                "message": "附近有两个入口，请选择。",
                "options": [{"option_id": "opt_1", "label": "入口 A"}],
                "recommended_option_id": "opt_1",
                "requires_user_response": True,
            },
            call_id="ui-1",
        )

        response = await ToolExecutor().execute(call, case=_sample_case())

        self.assertTrue(response.success)
        self.assertTrue(response.result["await_user"])
        self.assertEqual(response.result["runtime_state"], RUNTIME_STATE_WAITING_FOR_USER)
        pending = response.result["dialogue_state"]["pending_user_interaction"]
        self.assertEqual(pending["interaction_id"], "ui-1")
        self.assertEqual(pending["motion_policy_while_waiting"], "stop")

    async def test_executor_returns_route_context_assets_without_live_provider(self) -> None:
        call = ToolCall(
            tool="map_tool",
            arguments={
                "op": "get_route_context",
                "route_id": "route-1",
                "asset_types": ["route_instruction_text", "next_waypoint", "route_bev_view"],
                "horizon_m": 20,
                "maneuver": "turn_right",
            },
            call_id="tr-1",
        )

        response = await ToolExecutor().execute(call, case=_sample_case())

        self.assertTrue(response.success)
        assets = response.result["context_assets"]
        self.assertEqual([asset["asset_type"] for asset in assets], ["route_instruction_text", "next_waypoint", "route_bev_view"])
        self.assertEqual(assets[1]["payload"]["maneuver"], "turn_right")
        self.assertEqual(assets[2]["payload"]["render_status"], "not_implemented")

    async def test_exporter_writes_contract_views_next_to_flat_fields(self) -> None:
        case = _sample_case()
        trace = await RuleRunner().run_case(case)
        row = trace_to_ms_swift_stage2_row(trace, case)

        self.assertIn("observation_meta", row)
        self.assertIn("system2_supervision", row)
        self.assertIn("system1_supervision", row)
        self.assertIn("contract_meta", row)
        self.assertEqual(row["observation_meta"]["observation_id"], case.case_id)
        self.assertEqual(row["system1_supervision"]["pixel_trajectory"], case.ground_truth["pixel_trajectory"])
        self.assertEqual(row["sidewalk_meta"]["agent_step_result"]["schema_version"], "sidewalk_agent_step_result_v1")

    async def test_agent_step_result_from_trace(self) -> None:
        case = _sample_case()
        trace = await RuleRunner().run_case(case)

        step = AgentStepResult.from_trace(trace, case=case)

        self.assertEqual(step.step_id, case.case_id)
        self.assertIsNotNone(step.observation)
        self.assertIsNotNone(step.tool_action)
        self.assertIsNotNone(step.navigation_decision)
        self.assertEqual(step.to_dict()["system2"]["tool_action"]["tool"], "map_tool")


def _sample_case() -> AgentStage2Case:
    return AgentStage2Case(
        case_id="contract-case-1",
        schema_version=AGENT_STAGE2_CASE_SCHEMA_VERSION,
        user_request="Please find a nearby rest stop.",
        observation={
            "images": {"fpv": "/data/fpv.jpg", "bev": "/data/bev.jpg"},
            "current_location": "120.0,30.0",
            "mode": "foot",
        },
        ground_truth={
            "expected_tool_ops": ["search_poi"],
            "pixel_trajectory": [[0.5, 0.9], [0.55, 0.75]],
        },
    )


if __name__ == "__main__":
    unittest.main()
