"""Export Agentic Stage2 traces to ms-swift agent-template JSONL rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from mcp_server.agent.schemas import (
    MS_SWIFT_AGENTIC_STAGE2_SCHEMA_VERSION,
    AgentEvalCase,
    AgentStage2Case,
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
        # tool_response 进入 ms-swift row 前必须稳定 JSON 化，避免同义空格影响 diff/review。
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def _tool_call_content(content) -> str:
    if isinstance(content, str):
        payload = json.loads(content)
    else:
        payload = content
    # 统一导出为 ms-swift agent-template 期望的 name/arguments 结构。
    call = ToolCall.from_dict(payload if isinstance(payload, dict) else {})
    return json.dumps(call.to_ms_swift_dict(), ensure_ascii=False, sort_keys=True)


def _case_from_trace(trace: AgentTrace) -> AgentStage2Case | None:
    request = trace.request
    if isinstance(request, dict) and request.get("case_id"):
        return AgentStage2Case.from_dict(request)
    return None


def trace_to_ms_swift_stage2_row(trace: AgentTrace, case: AgentStage2Case | None = None) -> dict:
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
        # teacher trace 通常不显式存 system；导出训练 row 时补齐 agent system prompt。
        messages.append({"role": "system", "content": DEFAULT_AGENT_SYSTEM_PROMPT})

    if case is not None and not any(turn.role == "user" for turn in trace.turns):
        # 兼容早期 runner：若 trace 里没有 user turn，则从 case 重建带 <image> 的初始请求。
        image_prefix = "".join("<image>" for _ in case.images)
        messages.append({"role": "user", "content": f"{image_prefix}{case.user_request}"})

    for turn in trace.turns:
        if turn.role == "tool_call":
            # tool_call 是独立 role，content 是 JSON string，不再塞到 assistant content 中。
            messages.append({"role": "tool_call", "content": _tool_call_content(turn.content)})
        elif turn.role == "tool_response":
            # 原始工具返回作为环境上下文，不参与 assistant LM target。
            messages.append({"role": "tool_response", "content": _json_content(turn.content)})
        elif turn.role in {"system", "user", "assistant"}:
            messages.append({"role": turn.role, "content": _text_content(turn.content)})
        else:
            # 未知 role 降级为 assistant 文本，validator 会负责暴露 role sequence 错误。
            messages.append({"role": "assistant", "content": _text_content(turn.content)})

    supervision = dict(trace.supervision)
    if case is not None:
        # case ground_truth 优先级更高，保证 Phase1 adapter 透传的 GT 覆盖 trace 临时值。
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
        # 顶层监督字段给 SideWalk trainer 的 trajectory loss 用，不写入 assistant target。
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
            # FPV/ego trajectory 与 segmentation 监督继续顶层透传，保持 Phase1 loss 接口不变。
            row[key] = supervision[key]
    return row


def trace_to_ms_swift_row(trace: AgentTrace, case: AgentEvalCase | None = None) -> dict:
    """Compatibility wrapper for the Stage2 exporter.

    新代码应调用 ``trace_to_ms_swift_stage2_row``。保留该函数是为了不破坏早期
    eval helper 和测试。
    """

    stage2_case = AgentStage2Case.from_dict(case.to_dict()) if case is not None else None
    return trace_to_ms_swift_stage2_row(trace, stage2_case)


def write_ms_swift_jsonl(
    traces: Iterable[AgentTrace],
    path: str | Path,
    *,
    cases_by_id: dict[str, AgentEvalCase] | None = None,
) -> None:
    """Compatibility wrapper for ``write_ms_swift_stage2_jsonl``."""

    write_ms_swift_stage2_jsonl(traces, path, cases_by_id=cases_by_id)


def write_ms_swift_stage2_jsonl(
    traces: Iterable[AgentTrace],
    path: str | Path,
    *,
    cases_by_id: dict[str, AgentStage2Case] | None = None,
) -> None:
    """Write ms-swift agent-template JSONL for Stage2 cold-start training."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for trace in traces:
            # cases_by_id 用于把 trace 与原始 supervision/images 重新对齐。
            row = trace_to_ms_swift_stage2_row(trace, case=(cases_by_id or {}).get(trace.case_id))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_traces_jsonl(traces: Iterable[AgentTrace], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
