from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from mcp_server.core.replay_map_backend import ReplayMapBackend
from mcp_server.agent.tool_parser import parse_agent_action
from mcp_server.agent.tool_schema import validate_tool_call
from mcp_server.eval.urbannav_replay_tool_executor import ReplayUrbanNavToolExecutor
from mcp_server.eval.urbannav_runners import ReplayUrbanNavBackendAdapter
from mcp_server.eval.urbannav_schema import UrbanNavCase, UrbanNavDestination, UrbanNavPoint
from mcp_server.eval.urbannav_tool_agent_runner import UrbanNavToolAgentConfig, UrbanNavToolAgentRunner
from mcp_server.llm import LLMSettings


class UrbanNavToolAgentRunnerTest(unittest.TestCase):
    """Tests for explicit UrbanNav tool-use runner."""

    def test_replay_tool_executor_searches_and_plans_from_cache(self) -> None:
        """Executor should expose frozen POI and route ids without selecting for the model."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            case, backend = _case_and_backend(tmp)
            executor = ReplayUrbanNavToolExecutor(backend=backend)
            state = executor.new_state(case)
            assert state is not None

            search = executor.execute(
                case=case,
                state=state,
                call=_tool_call("search_destination", query="convenience store"),
            )
            route = executor.execute(
                case=case,
                state=state,
                call=_tool_call("plan_route", destination_id="poi_1"),
            )

            self.assertTrue(search.success)
            self.assertEqual(search.result["candidates"][0]["alias"], "poi_1")
            self.assertTrue(route.success)
            self.assertEqual(route.result["route_id"], "route_1")
            self.assertEqual(route.result["distance_m"], 100.0)

    def test_tool_agent_runner_converts_finish_to_prediction(self) -> None:
        """Runner should convert model tool calls and finish ids into a prediction."""

        async def run_test() -> None:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp = Path(tmp_dir)
                case, backend = _case_and_backend(tmp)
                outputs = iter(
                    [
                        '<tool_call>{"name":"map_tool","arguments":{"op":"search_destination","query":"convenience store"}}</tool_call>',
                        '<tool_call>{"name":"map_tool","arguments":{"op":"plan_route","destination_id":"poi_1"}}</tool_call>',
                        '<finish>{"chosen_destination_id":"poi_1","route_id":"route_1"}</finish>',
                    ]
                )

                async def fake_complete(messages):
                    return next(outputs)

                runner = UrbanNavToolAgentRunner(
                    backend=backend,
                    settings=_fake_settings(),
                    config=UrbanNavToolAgentConfig(max_turns=4),
                    llm_complete=fake_complete,
                )
                prediction = await runner.run_case_async(case)

                self.assertEqual(prediction.case_id, case.case_id)
                self.assertEqual(prediction.chosen_destination.provider_poi_id, "poi-a")
                self.assertEqual(prediction.route_distance_m, 100.0)
                self.assertEqual(prediction.trace["runner"], "urbannav_tool_agent")
                self.assertEqual(prediction.trace["tool_metrics"]["search_count"], 1)
                self.assertEqual(prediction.trace["tool_metrics"]["route_count"], 1)

        asyncio.run(run_test())

    def test_parser_accepts_typed_map_tool_names(self) -> None:
        """Parser should normalize SpotAgent-style typed tool names."""

        parsed = parse_agent_action(
            '<tool_call>{"name":"search_destination","arguments":{"query":"pharmacy"}}</tool_call>'
        )

        self.assertEqual(parsed.action_type, "tool_call")
        self.assertEqual(parsed.tool_call.tool, "map_tool")
        self.assertEqual(parsed.tool_call.arguments["op"], "search_destination")

    def test_parser_normalizes_common_map_aliases(self) -> None:
        """Parser and validator should accept equivalent tool argument aliases."""

        parsed = parse_agent_action(
            '<tool_call>{"name":"replace_destination","arguments":{"new_destination_id":"poi_1"}}</tool_call>'
        )

        self.assertEqual(parsed.action_type, "tool_call")
        self.assertEqual(parsed.tool_call.arguments["op"], "replace_destination")
        self.assertEqual(parsed.tool_call.arguments["destination_id"], "poi_1")

        replan = parse_agent_action(
            '<tool_call>{"name":"replan_from_current_pose","arguments":{"current_lat":30.0,"current_lng":120.0,"destination_id":"poi_1"}}</tool_call>'
        )
        self.assertEqual(replan.action_type, "tool_call")
        self.assertEqual(replan.tool_call.arguments["current_pose"], {"lat": 30.0, "lng": 120.0})
        valid, errors = validate_tool_call(replan.tool_call)
        self.assertTrue(valid, errors)

    def test_parser_accepts_only_trailing_brace_json_repair(self) -> None:
        """Parser should tolerate extra closing braces but not arbitrary text."""

        repaired = parse_agent_action(
            '<tool_call>{"name":"map_tool","arguments":{"op":"plan_route","destination_id":"poi_1"}}}</tool_call>'
        )
        invalid = parse_agent_action(
            '<tool_call>{"name":"map_tool","arguments":{"op":"plan_route","destination_id":"poi_1"}} trailing</tool_call>'
        )

        self.assertEqual(repaired.action_type, "tool_call")
        self.assertEqual(repaired.tool_call.arguments["destination_id"], "poi_1")
        self.assertEqual(invalid.action_type, "invalid")

    def test_replay_executor_accepts_goal_and_via_aliases(self) -> None:
        """Executor should route canonicalized goal_index and via id aliases."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            case, backend = _case_and_backend(tmp)
            executor = ReplayUrbanNavToolExecutor(backend=backend)
            state = executor.new_state(case)
            assert state is not None

            multi_case = _replace_case(case, category="multi_stop_route", metadata={"goal_count": 2})
            search = executor.execute(
                case=multi_case,
                state=state,
                call=_tool_call("search_destination", query="store", goal_index=2),
            )
            normal_search = executor.execute(
                case=case,
                state=state,
                call=_tool_call("search_destination", query="store"),
            )
            via = executor.execute(
                case=case,
                state=state,
                call=_tool_call("add_via", via_destination_id="poi_1"),
            )

            self.assertTrue(search.success)
            self.assertEqual(search.result["stop_index"], 2)
            self.assertTrue(normal_search.success)
            self.assertEqual(normal_search.result["candidate_count"], 1)
            self.assertTrue(via.success, via.error)
            self.assertEqual(via.result["route_id"], "route_1")

    def test_tool_agent_uses_category_adaptive_max_turns(self) -> None:
        """Auto max_turns should give more budget to route-update and multi-stop cases."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            case, backend = _case_and_backend(tmp)
            runner = UrbanNavToolAgentRunner(
                backend=backend,
                settings=_fake_settings(),
                config=UrbanNavToolAgentConfig(max_turns=0),
                llm_complete=None,
            )

            self.assertEqual(runner._max_turns_for_case(case), 6)
            multi = _replace_case(case, category="multi_stop_route", metadata={"goal_count": 3})
            update = _replace_case(
                case,
                category="route_update_add_via",
                metadata={"route_update": {"update_type": "replace_destination"}},
            )
            self.assertEqual(runner._max_turns_for_case(multi), 10)
            self.assertEqual(runner._max_turns_for_case(update), 8)

    def test_tool_agent_runner_marks_invalid_output_as_exception(self) -> None:
        """Invalid model output should not trigger runner fallback selection."""

        async def run_test() -> None:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp = Path(tmp_dir)
                case, backend = _case_and_backend(tmp)

                async def fake_complete(messages):
                    return "I will pick the first candidate."

                runner = UrbanNavToolAgentRunner(
                    backend=backend,
                    settings=_fake_settings(),
                    config=UrbanNavToolAgentConfig(max_turns=2),
                    llm_complete=fake_complete,
                )
                prediction = await runner.run_case_async(case)

                self.assertIsNone(prediction.chosen_destination)
                self.assertEqual(prediction.trace["error"], "prediction_exception")
                self.assertIn("invalid_agent_action", prediction.trace["detail"])

        asyncio.run(run_test())


def _tool_call(op: str, **kwargs):
    from mcp_server.agent.schemas import ToolCall

    return ToolCall(tool="map_tool", arguments={"op": op, **kwargs})


def _case_and_backend(tmp: Path) -> tuple[UrbanNavCase, ReplayUrbanNavBackendAdapter]:
    poi_cache = tmp / "poi_cache.jsonl"
    route_cache = tmp / "route_cache.jsonl"
    poi_key = "poi:case-1"
    route_key = "route:case-1:001"
    candidate = {
        "provider_poi_id": "poi-a",
        "name": "Store A",
        "type": "convenience_store",
        "lat": 30.0,
        "lng": 120.001,
    }
    poi_cache.write_text(
        json.dumps(
            {
                "cache_key": poi_key,
                "provider": "replay",
                "request": {"query": "convenience store"},
                "normalized_candidates": [candidate],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    route_cache.write_text(
        json.dumps(
            {
                "cache_key": route_key,
                "provider": "replay",
                "request": {
                    "origin": {"lat": 30.0, "lng": 120.0},
                    "destination": candidate,
                    "mode": "foot",
                },
                "normalized_route": {
                    "polyline": [{"lat": 30.0, "lng": 120.0}, {"lat": 30.0, "lng": 120.001}],
                    "distance_m": 100.0,
                    "duration_s": 80.0,
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    case = UrbanNavCase(
        case_id="case-1",
        instruction="Please find a nearby convenience store.",
        origin=UrbanNavPoint(lat=30.0, lng=120.0),
        gt_destination=UrbanNavDestination.from_dict(candidate),
        acceptable_destinations=[
            UrbanNavDestination(
                provider_poi_id="poi-a",
                name="Store A",
                type="convenience_store",
                lat=30.0,
                lng=120.001,
                reference_route_distance_m=100.0,
            )
        ],
        ranked_destinations=[UrbanNavDestination.from_dict({**candidate, "rank": 1, "route_distance_m": 100.0})],
        gt_route=[UrbanNavPoint(lat=30.0, lng=120.0), UrbanNavPoint(lat=30.0, lng=120.001)],
        gt_distance_m=100.0,
        category="poi_navigation",
        subcategory="convenience_store",
        city="unit_city",
        cache_keys={"poi_search": [poi_key], "routes": [route_key]},
    )
    return case, ReplayUrbanNavBackendAdapter(ReplayMapBackend(poi_cache_path=poi_cache, route_cache_path=route_cache))


def _replace_case(case: UrbanNavCase, **updates) -> UrbanNavCase:
    from dataclasses import replace

    return replace(case, **updates)


def _fake_settings() -> LLMSettings:
    return LLMSettings(
        provider="test",
        api_style="chat_completions",
        api_key="test",
        model="fake-tool-agent",
        base_url="http://127.0.0.1:1/v1",
    )


if __name__ == "__main__":
    unittest.main()
