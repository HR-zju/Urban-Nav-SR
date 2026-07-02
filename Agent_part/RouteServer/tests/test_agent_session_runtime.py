from __future__ import annotations

import unittest

from mcp_server.agent import RouteSessionRuntime, SessionRuntimeConfig
from mcp_server.agent.schemas import AGENT_STAGE2_CASE_SCHEMA_VERSION, AgentStage2Case
from mcp_server.eval.runners import RuleRunner


class RouteSessionRuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_session_runtime_stops_on_user_interaction_tick(self) -> None:
        runtime = RouteSessionRuntime(
            runner=RuleRunner(),
            config=SessionRuntimeConfig(session_id="session-user-stop", max_ticks=4),
        )

        result = await runtime.run_until_stop_condition(
            [
                _search_case("tick-search"),
                _clarification_case("tick-clarify"),
                _search_case("tick-after-stop"),
            ]
        )

        self.assertEqual(result.stopped_reason, "user_interaction")
        self.assertEqual([tick.case_id for tick in result.ticks], ["tick-search", "tick-clarify"])
        self.assertFalse(result.ticks[0].needs_user_interaction)
        self.assertTrue(result.ticks[1].needs_user_interaction)
        self.assertEqual(result.ticks[1].route_action, "request_user_input")
        self.assertTrue(all(tick.trace_validation.valid for tick in result.ticks))
        self.assertEqual(result.traces[0].metadata["session_runtime"]["session_id"], "session-user-stop")
        self.assertEqual(result.to_dict()["schema_version"], "sidewalk_agent_session_trace_v1")

    async def test_session_runtime_stops_at_max_ticks_without_running_extra_case(self) -> None:
        runtime = RouteSessionRuntime(
            runner=RuleRunner(),
            config=SessionRuntimeConfig(session_id="session-max", max_ticks=1, stop_on_user_interaction=False),
        )

        result = await runtime.run_cases([_search_case("tick-0"), _search_case("tick-1")])

        self.assertEqual(result.stopped_reason, "max_ticks")
        self.assertEqual([tick.case_id for tick in result.ticks], ["tick-0"])
        self.assertTrue(result.ticks[0].trace_validation.valid, result.ticks[0].trace_validation.errors)

    def test_session_runtime_rejects_non_positive_tick_budget(self) -> None:
        with self.assertRaises(ValueError):
            RouteSessionRuntime(runner=RuleRunner(), config=SessionRuntimeConfig(max_ticks=0))


def _search_case(case_id: str) -> AgentStage2Case:
    return AgentStage2Case(
        case_id=case_id,
        schema_version=AGENT_STAGE2_CASE_SCHEMA_VERSION,
        user_request="Please find a nearby coffee or rest place.",
        observation={
            "images": {"fpv": "/data/fpv.jpg", "bev": "/data/bev.jpg"},
            "current_location": "120.0,30.0",
        },
        ground_truth={
            "expected_tool_ops": ["search_poi"],
            "pixel_trajectory": [[0.5, 0.9], [0.55, 0.75]],
        },
    )


def _clarification_case(case_id: str) -> AgentStage2Case:
    return AgentStage2Case(
        case_id=case_id,
        schema_version=AGENT_STAGE2_CASE_SCHEMA_VERSION,
        user_request="Which destination should I use?",
        observation={
            "images": {"fpv": "/data/fpv.jpg", "bev": "/data/bev.jpg"},
            "current_location": "120.0,30.0",
        },
        ground_truth={
            "expected_tool_ops": ["request_clarification"],
            "pixel_trajectory": [[0.5, 0.9], [0.5, 0.82]],
        },
    )


if __name__ == "__main__":
    unittest.main()
