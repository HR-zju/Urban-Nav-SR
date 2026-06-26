from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from typing import Any

from mcp_server.agent.final_inference import (
    RouteServerAgentInferenceConfig,
    RouteServerAgentInferenceRequest,
    run_routeserver_final_inference,
    write_routeserver_final_inference_artifacts,
)
from mcp_server.agent.schemas import AgentStage2Case, ToolCall, ToolResponse
from mcp_server.agent.stage2 import (
    build_agent_state,
    build_agent_think_text,
    build_final_assistant_text,
    build_tool_decision,
)
from mcp_server.eval.online_agent_runner import CallableModelAdapter


ROUTESERVER_ROOT = Path(__file__).resolve().parents[1]


class FinalInferenceInterfaceTest(unittest.IsolatedAsyncioTestCase):
    async def test_final_inference_returns_finish_trace_validation_and_replay_row(self) -> None:
        request = _sample_request(tool_policy="tool")
        case = request.to_case()
        adapter = CallableModelAdapter(
            pre_tool=lambda messages, tools, images: _first_output(case, call=_sample_call(case)),
            final=lambda messages, tools, images: build_final_assistant_text(
                case,
                call=_first_tool_call(messages),
                response=_first_tool_response(messages),
                source="unit_final_inference",
            ),
        )

        result = await run_routeserver_final_inference(
            request,
            model_adapter=adapter,
            config=RouteServerAgentInferenceConfig(source="unit_final_inference"),
        )

        self.assertIsNotNone(result.finish)
        self.assertEqual(result.finish["trajectory"], request.supervision["pixel_trajectory"])
        self.assertEqual(result.trajectory, request.supervision["pixel_trajectory"])
        self.assertTrue(result.validation_report["trace"]["valid"], result.validation_report["trace"]["errors"])
        self.assertTrue(result.validation_report["ms_swift_row"]["valid"], result.validation_report["ms_swift_row"]["errors"])
        roles = [message["role"] for message in result.ms_swift_row["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool_call", "tool_response", "user", "assistant"])
        self.assertEqual(result.ms_swift_row["sidewalk_meta"]["case_id"], "final-iface-tool")

        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = write_routeserver_final_inference_artifacts(result, tmp_dir)
            self.assertTrue(Path(paths["result"]).exists())
            self.assertTrue(Path(paths["ms_swift_replay_row"]).exists())
            artifact_index = json.loads(Path(paths["artifact_index"]).read_text(encoding="utf-8"))
            self.assertEqual(artifact_index["case_id"], "final-iface-tool")


class FinalInferenceCliTest(unittest.TestCase):
    def test_cli_fake_backend_writes_replay_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            request_path = tmp / "request.json"
            output_dir = tmp / "out"
            request_path.write_text(
                json.dumps(_sample_request(tool_policy="no_tool").to_dict(), ensure_ascii=False),
                encoding="utf-8",
            )
            cmd = [
                sys.executable,
                str(ROUTESERVER_ROOT / "tools" / "evaluation" / "run_sidewalk_agentic_stage2_inference.py"),
                "--request-json",
                str(request_path),
                "--output-dir",
                str(output_dir),
                "--model-backend",
                "fake",
                "--fake-policy",
                "no_tool",
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROUTESERVER_ROOT / "src")
            completed = subprocess.run(cmd, cwd=ROUTESERVER_ROOT, env=env, check=True, capture_output=True, text=True)
            payload = json.loads(completed.stdout)
            self.assertTrue(payload["finish_valid"])
            self.assertTrue(payload["trace_valid"])
            self.assertTrue(payload["row_valid"])
            result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
            roles = [message["role"] for message in result["ms_swift_row"]["messages"]]
            self.assertEqual(roles, ["system", "user", "assistant", "user", "assistant"])
            self.assertEqual(result["finish"]["tool_used"], False)


def _sample_request(*, tool_policy: str) -> RouteServerAgentInferenceRequest:
    return RouteServerAgentInferenceRequest(
        case_id=f"final-iface-{tool_policy.replace('_', '-')}",
        user_request="Navigate toward the visible sidewalk corridor.",
        images={"fpv": "/data/fpv.jpg", "bev": "/data/bev.jpg"},
        current_location="120.0,30.0",
        route_state={"route_session_id": "route-unit", "mode": "foot"},
        supervision={
            "tool_policy": tool_policy,
            "pixel_trajectory": [[round(0.5 + index * 0.001, 4), round(0.9 - index * 0.02, 4)] for index in range(16)],
            "intent": "go_straight",
        },
    )


def _sample_call(case: AgentStage2Case) -> ToolCall:
    return ToolCall(
        tool="map_tool",
        arguments={
            "op": "search_poi",
            "query": "sidewalk destination",
            "location": case.observation.get("current_location") or "120.0,30.0",
            "limit": 3,
        },
    )


def _first_output(case: AgentStage2Case, *, call: ToolCall | None) -> str:
    text = build_agent_think_text(
        state=build_agent_state(case),
        decision=build_tool_decision(call, reason="unit final inference decision"),
    )
    if call is None:
        return text
    return f"{text}<tool_call>{json.dumps(call.to_dict(), ensure_ascii=False, separators=(',', ':'))}</tool_call>"


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


if __name__ == "__main__":
    unittest.main()
