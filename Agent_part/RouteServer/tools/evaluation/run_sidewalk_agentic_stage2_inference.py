#!/usr/bin/env python3
"""Run one RouteServer Agentic Stage2 final inference request."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
from dataclasses import replace
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcp_server.agent.final_inference import (  # noqa: E402
    RouteServerAgentInferenceConfig,
    load_routeserver_final_inference_request,
    run_routeserver_final_inference,
    write_routeserver_final_inference_artifacts,
)
from mcp_server.agent.schemas import AgentStage2Case, ToolCall, ToolResponse  # noqa: E402
from mcp_server.agent.stage2 import (  # noqa: E402
    build_agent_state,
    build_agent_think_text,
    build_final_assistant_text,
    build_tool_decision,
)
from mcp_server.agent.tool_executor import ToolExecutor  # noqa: E402
from mcp_server.eval.online_agent_runner import (  # noqa: E402
    CallableModelAdapter,
    MsSwiftInferenceAdapter,
    MsSwiftInferenceConfig,
    OpenAICompatibleInferenceConfig,
    OpenAICompatibleStage2Adapter,
)
from mcp_server.llm import LLMSettings  # noqa: E402
from mcp_server.map_providers import create_map_client  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argparse namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-backend", choices=["fake", "ms-swift", "llm"], default="fake")
    parser.add_argument("--fake-policy", choices=["auto", "tool", "no_tool"], default="auto")
    parser.add_argument("--execute-tools", action="store_true", help="Use ToolExecutor instead of structured missing-executor responses.")
    parser.add_argument(
        "--tool-context",
        choices=["none", "provider"],
        default="none",
        help="ToolExecutor context. Use provider to build a live map client from MAP_PROVIDER or --map-provider.",
    )
    parser.add_argument(
        "--map-provider",
        choices=["osm", "amap", "baidu", "google"],
        default=None,
        help="Optional live map provider override for --tool-context provider. Defaults to MAP_PROVIDER or osm.",
    )
    parser.add_argument("--runtime-artifact-dir", default=None)
    parser.add_argument("--source", default="routeserver_final_inference")
    parser.add_argument("--max-tool-calls", type=int, default=1)
    parser.add_argument(
        "--drop-supervision",
        action="store_true",
        help="Ignore request supervision labels for live open-loop validation.",
    )

    parser.add_argument("--llm-provider", default=None, help="Optional LLM provider override, otherwise use LLM_PROVIDER.")
    parser.add_argument("--llm-model", default=None, help="Optional LLM model override, otherwise use LLM_MODEL/ARK_MODEL.")
    parser.add_argument("--llm-api-base-url", default=None, help="Optional OpenAI-compatible base URL override.")
    parser.add_argument("--llm-max-tokens-pre-tool", type=int, default=512)
    parser.add_argument("--llm-max-tokens-final", type=int, default=1024)
    parser.add_argument("--llm-temperature", type=float, default=None)

    parser.add_argument("--ms-swift-root", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--phase", choices=["phase0", "phase1", "phase2"], default="phase0")
    parser.add_argument("--agent-template", default="hermes")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--device-map", default=None)
    return parser.parse_args()


async def async_main() -> None:
    """Run final inference and write artifacts.

    Returns:
        ``None``.  The output artifact mapping is printed to stdout.

    Raises:
        ValueError: If required ms-swift backend arguments are missing.
    """

    args = parse_args()
    request = load_routeserver_final_inference_request(args.request_json)
    if args.drop_supervision:
        request = replace(request, supervision={})
    adapter = _build_model_adapter(args, request.to_case())
    executor, map_client = await _build_tool_executor(args)
    try:
        result = await run_routeserver_final_inference(
            request,
            model_adapter=adapter,
            executor=executor,
            config=RouteServerAgentInferenceConfig(
                source=args.source,
                max_tool_calls=args.max_tool_calls,
                review_artifact_dir=args.runtime_artifact_dir,
            ),
        )
    finally:
        if map_client is not None:
            await map_client.disconnect()
    artifact_paths = write_routeserver_final_inference_artifacts(result, args.output_dir)
    print(
        json.dumps(
            {
                "case_id": request.case_id,
                "finish_valid": result.validation_report["finish"]["valid"],
                "trace_valid": result.validation_report["trace"]["valid"],
                "row_valid": result.validation_report["ms_swift_row"]["valid"],
                "artifacts": artifact_paths,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def _build_model_adapter(args: argparse.Namespace, case: AgentStage2Case):
    if args.model_backend == "fake":
        return _fake_model_adapter(case, policy=args.fake_policy)
    if args.model_backend == "llm":
        return OpenAICompatibleStage2Adapter(
            OpenAICompatibleInferenceConfig(
                settings=_llm_settings_from_args(args),
                case=case,
                pre_tool_max_tokens=args.llm_max_tokens_pre_tool,
                final_max_tokens=args.llm_max_tokens_final,
                temperature=args.llm_temperature,
            )
        )
    if not args.ms_swift_root or not args.model:
        raise ValueError("--ms-swift-root and --model are required for --model-backend ms-swift")
    return MsSwiftInferenceAdapter(
        MsSwiftInferenceConfig(
            ms_swift_root=args.ms_swift_root,
            model=args.model,
            adapter_dir=args.adapter_dir,
            checkpoint_dir=args.checkpoint_dir,
            phase=args.phase,
            agent_template=args.agent_template,
            python_executable=args.python_bin,
            work_dir=args.work_dir,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
        )
    )


def _llm_settings_from_args(args: argparse.Namespace) -> LLMSettings:
    settings = LLMSettings.from_env()
    updates: dict[str, Any] = {}
    if args.llm_provider:
        updates["provider"] = str(args.llm_provider).strip().lower()
    if args.llm_model:
        updates["model"] = str(args.llm_model).strip()
    if args.llm_api_base_url:
        updates["base_url"] = str(args.llm_api_base_url).strip().rstrip("/")
    if args.llm_temperature is not None:
        updates["temperature"] = float(args.llm_temperature)
    return replace(settings, **updates) if updates else settings


async def _build_tool_executor(args: argparse.Namespace):
    if not args.execute_tools:
        if args.tool_context != "none":
            raise ValueError("--tool-context requires --execute-tools")
        if args.map_provider:
            raise ValueError("--map-provider requires --execute-tools --tool-context provider")
        return None, None
    if args.tool_context == "provider":
        if args.map_provider:
            os.environ["MAP_PROVIDER"] = args.map_provider
        map_client, provider = create_map_client()
        await map_client.connect()
        return ToolExecutor(app_context=SimpleNamespace(osm_client=map_client, provider=provider)), map_client
    if args.map_provider:
        raise ValueError("--map-provider requires --tool-context provider")
    return ToolExecutor(), None


def _fake_model_adapter(case: AgentStage2Case, *, policy: str) -> CallableModelAdapter:
    def pre_tool(messages: list[dict[str, Any]], tools: list[dict[str, Any]], images: list[str]) -> str:
        call = _fake_tool_call(case, policy=policy)
        reason = "fake runtime requests one map lookup" if call is not None else "fake runtime uses visible route context only"
        text = build_agent_think_text(
            state=build_agent_state(case),
            decision=build_tool_decision(call, reason=reason),
        )
        if call is None:
            return text
        return f"{text}<tool_call>{json.dumps(call.to_dict(), ensure_ascii=False, separators=(',', ':'))}</tool_call>"

    def final(messages: list[dict[str, Any]], tools: list[dict[str, Any]], images: list[str]) -> str:
        return build_final_assistant_text(
            case,
            call=_first_tool_call(messages),
            response=_first_tool_response(messages),
            source="fake_routeserver_final_inference",
        )

    return CallableModelAdapter(pre_tool=pre_tool, final=final)


def _fake_tool_call(case: AgentStage2Case, *, policy: str) -> ToolCall | None:
    if policy == "no_tool" or (policy == "auto" and case.ground_truth.get("tool_policy") == "no_tool"):
        return None
    location = case.observation.get("current_location") or "0,0"
    return ToolCall(
        tool="map_tool",
        arguments={"op": "search_poi", "query": "sidewalk destination", "location": location, "limit": 3},
    )


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


def main() -> None:
    """Entrypoint for synchronous execution."""

    asyncio.run(async_main())


if __name__ == "__main__":
    main()
