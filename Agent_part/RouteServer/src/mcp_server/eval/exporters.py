"""Export Agentic Stage2 traces to ms-swift agent-template JSONL rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from mcp_server.agent.schemas import (
    MS_SWIFT_AGENTIC_STAGE2_SCHEMA_VERSION,
    AgentEvalCase,
    AgentTrace,
    ToolCall,
)
from mcp_server.agent.tool_schema import get_tool_definitions


DEFAULT_AGENT_SYSTEM_PROMPT = (
    "You are a sidewalk navigation agent. First reason about whether external map, image, "
    "or user-interaction context is needed. When needed, emit a tool call through the "
    "tool_call role. After tool responses, produce the final navigation CoT and finish block."
)


def _text_content(content) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def _json_content(content) -> str:
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return content
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def _tool_call_content(content) -> str:
    if isinstance(content, str):
        payload = json.loads(content)
    else:
        payload = content
    call = ToolCall.from_dict(payload if isinstance(payload, dict) else {})
    return json.dumps(call.to_ms_swift_dict(), ensure_ascii=False, sort_keys=True)


def _case_from_trace(trace: AgentTrace) -> AgentEvalCase | None:
    request = trace.request
    if isinstance(request, dict) and request.get("case_id"):
        return AgentEvalCase.from_dict(request)
    return None


def trace_to_ms_swift_row(trace: AgentTrace, case: AgentEvalCase | None = None) -> dict:
    """Convert one ``AgentTrace`` to a trainable ms-swift agent-template JSONL row.

    输出字段:
        ``messages``: system/user/assistant/tool_call/tool_response 对话。
        ``tools``: 三个元工具 schema 的 JSON 字符串，避免 pyarrow 类型漂移。
        ``images``: FPV/BEV 图片路径列表。
        ``pixel_trajectory``: 如果 case 或 trace 里存在轨迹监督，则透传。
        ``sidewalk_meta``: trace schema、case_id、metrics 等审计信息。
    """

    case = case or _case_from_trace(trace)
    messages: list[dict] = []
    has_system = any(turn.role == "system" for turn in trace.turns)
    if not has_system:
        messages.append({"role": "system", "content": DEFAULT_AGENT_SYSTEM_PROMPT})

    if case is not None and not any(turn.role == "user" for turn in trace.turns):
        image_prefix = "".join("<image>" for _ in case.images)
        messages.append({"role": "user", "content": f"{image_prefix}{case.user_request}"})

    for turn in trace.turns:
        if turn.role == "tool_call":
            messages.append({"role": "tool_call", "content": _tool_call_content(turn.content)})
        elif turn.role == "tool_response":
            messages.append({"role": "tool_response", "content": _json_content(turn.content)})
        elif turn.role in {"system", "user", "assistant"}:
            messages.append({"role": turn.role, "content": _text_content(turn.content)})
        else:
            messages.append({"role": "assistant", "content": _text_content(turn.content)})

    supervision = dict(trace.supervision)
    if case is not None:
        supervision.update(case.ground_truth)
    tools = trace.tools or (case.tools if case is not None and case.tools else get_tool_definitions())
    row = {
        "schema_version": MS_SWIFT_AGENTIC_STAGE2_SCHEMA_VERSION,
        "stage": "agentic_cold_start",
        "task_type": trace.task_type,
        "messages": messages,
        "tools": json.dumps(tools, ensure_ascii=False, sort_keys=True),
        "images": case.images if case is not None else [],
        "sidewalk_meta": {
            "schema_version": MS_SWIFT_AGENTIC_STAGE2_SCHEMA_VERSION,
            "agent_trace_schema": trace.schema_version,
            "case_id": trace.case_id,
            "task_type": trace.task_type,
            "metrics": dict(trace.metrics),
            "model_run": dict(trace.model_run),
            "artifacts": list(trace.artifacts),
        },
    }
    if supervision.get("pixel_trajectory") is not None:
        row["pixel_trajectory"] = supervision["pixel_trajectory"]
    for key in (
        "trajectory_fpv_t",
        "trajectory_fpv_tv",
        "trajectory_ego_t",
        "trajectory_policy",
        "semantic_segmentation_path",
        "semantic_mask_path",
    ):
        if supervision.get(key) is not None:
            row[key] = supervision[key]
    return row


def write_ms_swift_jsonl(
    traces: Iterable[AgentTrace],
    path: str | Path,
    *,
    cases_by_id: dict[str, AgentEvalCase] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for trace in traces:
            row = trace_to_ms_swift_row(trace, case=(cases_by_id or {}).get(trace.case_id))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_traces_jsonl(traces: Iterable[AgentTrace], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
