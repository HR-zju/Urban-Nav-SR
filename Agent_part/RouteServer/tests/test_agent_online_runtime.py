from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mcp_server.agent.contracts import RUNTIME_STATE_WAITING_FOR_USER
from mcp_server.agent.schemas import AGENT_STAGE2_CASE_SCHEMA_VERSION, AgentStage2Case, ToolCall, ToolResponse
from mcp_server.agent.stage2 import build_agent_state, build_agent_think_text, build_final_assistant_text, build_tool_decision
from mcp_server.agent.tool_executor import ToolExecutor
from mcp_server.agent.tool_runtime import AgentToolRuntime
from mcp_server.eval.agent_tools.online_agent_runner import MsSwiftInferenceAdapter, MsSwiftInferenceConfig, OpenAICompatibleInferenceConfig, OpenAICompatibleStage2Adapter, OnlineAgentRunner, build_initial_model_messages
from mcp_server.eval.agent_tools.stage2_validator import validate_ms_swift_stage2_row, validate_stage2_trace
from mcp_server.llm import LLMSettings


class OnlineAgentRunnerTest(unittest.IsolatedAsyncioTestCase):
    async def test_online_runner_tool_path_uses_model_then_executor_then_model(self) -> None:
        case = _sample_case()
        call = ToolCall(tool="map_tool", arguments={"op": "search_poi", "query": "coffee", "location": "120,30"})
        adapter = _FakeOnlineModelAdapter(case=case, call=call)
        runtime = AgentToolRuntime(
            executor=ToolExecutor(app_context=SimpleNamespace(provider="fake_test_map", osm_client=_FakeMapClient()))
        )
        runner = OnlineAgentRunner(model_adapter=adapter, runtime=runtime)

        result = await runner.run_case_with_report(case)

        self.assertEqual(adapter.calls, ["pre_tool", "final"])
        self.assertEqual([turn.role for turn in result.trace.turns], ["user", "assistant", "tool_call", "tool_response", "user", "assistant"])
        self.assertTrue(result.trace.turns[3].content["success"])
        self.assertEqual(result.trace.turns[3].content["result"]["provider"], "fake_test_map")
        self.assertIn("<ToolResultUse>", result.trace.turns[5].content)
        self.assertTrue(result.trace_validation.valid, result.trace_validation.errors)
        self.assertTrue(result.row_validation.valid, result.row_validation.errors)
        self.assertTrue(validate_stage2_trace(result.trace, case).valid)
        self.assertTrue(validate_ms_swift_stage2_row(result.ms_swift_row).valid)
        self.assertEqual(result.ms_swift_row["sidewalk_meta"]["model_run"]["runner"], "online_agent_runner")

    async def test_online_runner_no_tool_path(self) -> None:
        case = _sample_no_tool_case()
        adapter = _FakeOnlineModelAdapter(case=case, call=None)
        runner = OnlineAgentRunner(model_adapter=adapter, runtime=AgentToolRuntime())

        result = await runner.run_case_with_report(case)

        self.assertEqual(adapter.calls, ["pre_tool", "final"])
        self.assertEqual([turn.role for turn in result.trace.turns], ["user", "assistant", "user", "assistant"])
        self.assertIn('"should_call_tool":false', result.trace.turns[1].content)
        self.assertNotIn("<ToolResultUse>", result.trace.turns[3].content)
        self.assertTrue(result.trace_validation.valid, result.trace_validation.errors)
        self.assertTrue(result.row_validation.valid, result.row_validation.errors)

    async def test_online_runner_failed_tool_path_still_generates_final(self) -> None:
        case = _sample_case()
        call = ToolCall(tool="map_tool", arguments={"op": "search_poi", "query": "coffee", "location": "120,30"})
        adapter = _FakeOnlineModelAdapter(case=case, call=call)
        runner = OnlineAgentRunner(model_adapter=adapter, runtime=AgentToolRuntime(executor=ToolExecutor()))

        result = await runner.run_case_with_report(case)

        self.assertEqual(adapter.calls, ["pre_tool", "final"])
        self.assertFalse(result.trace.turns[3].content["success"])
        self.assertEqual(result.trace.turns[3].content["error"]["code"], "missing_app_context")
        self.assertIn('"failure_recovery"', result.trace.turns[5].content)
        self.assertTrue(result.trace_validation.valid, result.trace_validation.errors)
        self.assertTrue(result.row_validation.valid, result.row_validation.errors)

    async def test_online_runner_user_interaction_report_exposes_pending_state(self) -> None:
        case = _sample_case()
        call = ToolCall(
            tool="user_interaction_tool",
            arguments={
                "op": "present_options",
                "message": "附近有两个入口，请选择。",
                "options": [
                    {"option_id": "entry_a", "label": "入口 A", "candidate_id": "poi_a"},
                    {"option_id": "entry_b", "label": "入口 B", "candidate_id": "poi_b"},
                ],
                "recommended_option_id": "entry_a",
                "requires_user_response": True,
                "motion_policy_while_waiting": "stop",
                "evidence_refs": ["poi_a", "poi_b"],
            },
            call_id="ui-choose-entry",
        )
        adapter = _FakeOnlineModelAdapter(case=case, call=call)
        runner = OnlineAgentRunner(model_adapter=adapter, runtime=AgentToolRuntime(executor=ToolExecutor()))

        result = await runner.run_case_with_report(case)
        report = result.to_dict()

        self.assertTrue(result.trace_validation.valid, result.trace_validation.errors)
        self.assertEqual(result.trace.metadata["runtime_state"], RUNTIME_STATE_WAITING_FOR_USER)
        pending = result.trace.metadata["dialogue_state"]["pending_user_interaction"]
        self.assertEqual(pending["interaction_id"], "ui-choose-entry")
        self.assertEqual(report["runtime_state"], RUNTIME_STATE_WAITING_FOR_USER)
        self.assertEqual(report["dialogue_state"]["pending_user_interaction"]["recommended_option_id"], "entry_a")
        self.assertEqual(report["agent_step_result"]["runtime_state"], RUNTIME_STATE_WAITING_FOR_USER)


class MsSwiftInferenceAdapterTest(unittest.TestCase):
    def test_ms_swift_adapter_builds_agent_template_inference_row(self) -> None:
        case = _sample_case()
        adapter = MsSwiftInferenceAdapter(
            MsSwiftInferenceConfig(
                ms_swift_root="/repo/ms-swift",
                model="/models/base",
                adapter_dir="/models/lora",
                checkpoint_dir="/models/checkpoint",
                phase="phase1",
                agent_template="hermes",
                python_executable="/usr/bin/python",
                max_tokens=128,
                device_map="cpu",
            )
        )
        tools = [{"type": "function", "function": {"name": "map_tool", "parameters": {"type": "object"}}}]
        row = adapter.build_inference_row(
            build_initial_model_messages(case),
            tools,
            case.images,
            target="tool_decision",
        )
        command = adapter._command(input_path=Path("/tmp/input.jsonl"), output_path=Path("/tmp/output.jsonl"))

        self.assertEqual(row["sidewalk_meta"]["prompt_template"], "agentic_stage2_cot_trajectory")
        self.assertEqual(row["sidewalk_meta"]["stage2_eval_target"], "tool_decision")
        self.assertEqual(json.loads(row["tools"])[0]["function"]["name"], "map_tool")
        self.assertEqual(row["images"], ["/data/fpv.jpg", "/data/bev.jpg"])
        self.assertIn("swift.sidewalk_nav.cli.infer", command)
        self.assertIn("--agent-template", command)
        self.assertIn("hermes", command)
        self.assertIn("--device-map", command)
        self.assertIn("cpu", command)


class OpenAICompatibleStage2AdapterTest(unittest.IsolatedAsyncioTestCase):
    async def test_openai_compatible_adapter_builds_protocol_chat_request(self) -> None:
        fake_client = _FakeOpenAIClient(
            response_text=(
                "<think>\n"
                '<AgentState>{"case_id":"online-case-1"}</AgentState>\n'
                '<ToolDecision>{"should_call_tool":false,"reason":"test"}</ToolDecision>\n'
                "</think>\n"
            )
        )
        adapter = OpenAICompatibleStage2Adapter(
            OpenAICompatibleInferenceConfig(
                settings=LLMSettings(
                    provider="ark",
                    api_style="chat_completions",
                    api_key="test-key",
                    model="doubao-test",
                    base_url="https://example.invalid/api/v3",
                    reasoning_effort="minimal",
                ),
                case=_sample_case(),
                client_factory=lambda settings: fake_client,
                pre_tool_max_tokens=321,
                final_max_tokens=654,
                temperature=0.0,
            )
        )

        text = await adapter.generate_pre_tool(
            build_initial_model_messages(_sample_case()),
            [{"type": "function", "function": {"name": "map_tool", "parameters": {"type": "object"}}}],
            ["/data/fpv.jpg", "/data/bev.jpg"],
        )

        self.assertIn("<ToolDecision>", text)
        payload = fake_client.payloads[0]
        self.assertEqual(payload["model"], "doubao-test")
        self.assertEqual(payload["max_tokens"], 321)
        self.assertEqual(payload["reasoning_effort"], "minimal")
        user_content = payload["messages"][1]["content"]
        self.assertIn('"target": "tool_decision"', user_content)
        self.assertIn('"current_location": "120.0,30.0"', user_content)
        self.assertIn("map_tool", user_content)
        self.assertIn("/data/fpv.jpg", user_content)
        self.assertNotIn("pixel_trajectory", user_content)

    async def test_openai_compatible_adapter_final_prompt_requires_canonical_finish(self) -> None:
        fake_client = _FakeOpenAIClient(
            response_text=(
                "<think>\n"
                '<Scene>{"summary":"test"}</Scene>\n'
                '<FPVScene>{"navigable_area":"visible"}</FPVScene>\n'
                '<BEVContext>{"mode":"foot"}</BEVContext>\n'
                '<CrossView>{"alignment":"ok"}</CrossView>\n'
                '<Plan>{"route_action":"continue_navigation"}</Plan>\n'
                '<ToolResultUse>{"used_for":"final_navigation_decision"}</ToolResultUse>\n'
                "</think>\n"
                '<finish>{"route_action":"continue_navigation","trajectory_grid1000":[[500,900]],'
                '"trajectory_coord_frame":"norm1000"}</finish>'
            )
        )
        adapter = OpenAICompatibleStage2Adapter(
            OpenAICompatibleInferenceConfig(
                settings=LLMSettings(
                    provider="ark",
                    api_style="chat_completions",
                    api_key="test-key",
                    model="doubao-test",
                    base_url="https://example.invalid/api/v3",
                    reasoning_effort="minimal",
                ),
                client_factory=lambda settings: fake_client,
                final_max_tokens=654,
            )
        )
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "<image>request"},
            {"role": "assistant", "content": "<think></think>"},
            {"role": "tool_call", "content": '{"name":"map_tool","arguments":{"op":"search_poi"}}'},
            {"role": "tool_response", "content": '{"tool":"map_tool","success":true,"result":{}}'},
            {"role": "user", "content": "Produce final [TRAJ_0]"},
        ]

        text = await adapter.generate_final(messages, [], [])

        self.assertIn("<finish>", text)
        payload = fake_client.payloads[0]
        self.assertEqual(payload["max_tokens"], 654)
        user_content = payload["messages"][1]["content"]
        self.assertIn('"target": "final_navigation"', user_content)
        self.assertIn("trajectory_grid1000", user_content)
        self.assertIn("ToolResultUse", user_content)
        self.assertIn('"tool_used": true', user_content)


class _FakeOnlineModelAdapter:
    def __init__(self, *, case: AgentStage2Case, call: ToolCall | None) -> None:
        self.case = case
        self.call = call
        self.calls: list[str] = []

    async def generate_pre_tool(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        images: list[str],
    ) -> str:
        self.calls.append("pre_tool")
        self.pre_tool_messages = list(messages)
        self.pre_tool_tools = list(tools)
        self.pre_tool_images = list(images)
        return _runtime_first_output(self.case, self.call)

    async def generate_final(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        images: list[str],
    ) -> str:
        self.calls.append("final")
        self.final_messages = list(messages)
        self.final_tools = list(tools)
        self.final_images = list(images)
        call = _first_tool_call(messages)
        response = _first_tool_response(messages)
        return build_final_assistant_text(
            self.case,
            call=call,
            response=response,
            source="fake_online_model",
        )


def _runtime_first_output(case: AgentStage2Case, call: ToolCall | None) -> str:
    decision_reason = (
        "external tool context is required before final navigation"
        if call is not None
        else "visible FPV, BEV, and route state are enough for local navigation"
    )
    output = build_agent_think_text(
        state=build_agent_state(case),
        decision=build_tool_decision(call, reason=decision_reason),
    )
    if call is None:
        return output
    payload = json.dumps(call.to_dict(), ensure_ascii=False, separators=(",", ":"))
    return f"{output}<tool_call>{payload}</tool_call>"


def _first_tool_call(messages: list[dict[str, Any]]) -> ToolCall | None:
    for message in messages:
        if message.get("role") != "tool_call":
            continue
        payload = json.loads(str(message.get("content") or "{}"))
        return ToolCall.from_dict(payload)
    return None


def _first_tool_response(messages: list[dict[str, Any]]) -> ToolResponse | None:
    for message in messages:
        if message.get("role") != "tool_response":
            continue
        payload = json.loads(str(message.get("content") or "{}"))
        return ToolResponse(
            tool=str(payload.get("tool") or ""),
            success=bool(payload.get("success")),
            result=payload.get("result") or {},
            error=payload.get("error"),
            artifacts=list(payload.get("artifacts") or []),
            call_id=payload.get("call_id"),
        )
    return None


def _sample_case() -> AgentStage2Case:
    return AgentStage2Case(
        case_id="online-case-1",
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


def _sample_no_tool_case() -> AgentStage2Case:
    return AgentStage2Case(
        case_id="online-no-tool",
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
            "intent": "go_straight",
        },
    )


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


class _FakeOpenAIClient:
    def __init__(self, *, response_text: str) -> None:
        self.response_text = response_text
        self.payloads: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **payload: Any) -> Any:
        self.payloads.append(payload)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.response_text),
                )
            ]
        )


if __name__ == "__main__":
    unittest.main()
