"""Validators for SideWalk Agentic Stage2 cases, traces, and exported rows.

validator 与 metrics 的职责不同 metrics 给出可聚合分数 validator 给出可审计的
错误列表，适合 build 脚本决定样本是否进入 review/train JSONL。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from mcp_server.agent.schemas import (
    AGENT_STAGE2_CASE_SCHEMA_VERSION,
    AGENT_STAGE2_TRACE_SCHEMA_VERSION,
    MS_SWIFT_AGENTIC_STAGE2_SCHEMA_VERSION,
    AgentStage2Case,
    AgentTrace,
    ToolCall,
)
from mcp_server.agent.stage2 import NAVIGATION_READOUT_TOKENS
from mcp_server.agent.tool_parser import parse_finish_text
from mcp_server.agent.tool_schema import validate_tool_call
from mcp_server.eval.metrics import compute_trace_metrics


STAGE2_TRACE_ROLE_SEQUENCE = ("user", "assistant", "tool_call", "tool_response", "user", "assistant")
STAGE2_ROW_ROLE_SEQUENCE = ("system", "user", "assistant", "tool_call", "tool_response", "user", "assistant")
FINAL_COT_REQUIRED_TAGS = ("<Scene>", "<FPVScene>", "<BEVContext>", "<CrossView>", "<Plan>", "<ToolResultUse>", "<finish>")


@dataclass(frozen=True)
class Stage2ValidationResult:
    """Validation result returned by Stage2 validators.

    属性:
        valid: 没有 hard errors 时为 True。
        errors: 会导致样本不可训练或不可回放的问题。
        warnings: 不阻塞训练但值得人工 review 的问题。
        metrics: trace validator 会附带协议级 metrics 便于 report 聚合。
    """

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "metrics": dict(self.metrics),
        }


def validate_stage2_case(case: AgentStage2Case) -> Stage2ValidationResult:
    """Validate one Stage2 case before trace generation.

    输入:
        ``AgentStage2Case``。

    输出:
        ``Stage2ValidationResult``；主要检查 schema、case_id、user_request 和图片字段。
    """

    errors: list[str] = []
    warnings: list[str] = []
    if case.schema_version != AGENT_STAGE2_CASE_SCHEMA_VERSION:
        errors.append(f"schema_version:{case.schema_version}")
    if not case.case_id:
        errors.append("missing_case_id")
    if not case.user_request.strip():
        errors.append("missing_user_request")
    if not case.images:
        warnings.append("missing_images")
    if not isinstance(case.ground_truth, dict):
        errors.append("ground_truth_not_object")
    return Stage2ValidationResult(valid=not errors, errors=errors, warnings=warnings)


def validate_stage2_trace(trace: AgentTrace, case: AgentStage2Case | None = None) -> Stage2ValidationResult:
    """Validate the normalized Stage2 trace contract.

    输入:
        ``AgentTrace``，可选原始 ``AgentStage2Case``。

    输出:
        ``Stage2ValidationResult``。检查固定 role 顺序、tool_call JSON、tool_response
        JSON、readout token 位置、最终 CoT tags 和 ``<finish>`` JSON。
    """

    errors: list[str] = []
    warnings: list[str] = []
    if trace.schema_version != AGENT_STAGE2_TRACE_SCHEMA_VERSION:
        errors.append(f"trace_schema_version:{trace.schema_version}")
    roles = tuple(turn.role for turn in trace.turns)
    if roles != STAGE2_TRACE_ROLE_SEQUENCE:
        errors.append(f"trace_role_sequence:{roles}")

    call = None
    if len(trace.turns) > 2:
        call_payload, call_errors = _json_dict(trace.turns[2].content)
        errors.extend(f"tool_call:{item}" for item in call_errors)
        if call_payload is not None:
            call = ToolCall.from_dict(call_payload)
            is_valid, validation_errors = validate_tool_call(call)
            if not is_valid:
                errors.extend(f"tool_call_schema:{item}" for item in validation_errors)
    if len(trace.turns) > 3:
        response_payload, response_errors = _json_dict(trace.turns[3].content)
        errors.extend(f"tool_response:{item}" for item in response_errors)
        if response_payload is not None:
            if response_payload.get("tool") != (call.tool if call is not None else trace.turns[3].tool):
                errors.append("tool_response_tool_mismatch")
            if "success" not in response_payload:
                errors.append("tool_response_missing_success")

    if len(trace.turns) > 4:
        readout = trace.turns[4].content
        if not isinstance(readout, str) or not _contains_all_readout_tokens(readout):
            errors.append("missing_readout_tokens")
    if len(trace.turns) > 5:
        final_text = trace.turns[5].content
        if not isinstance(final_text, str):
            errors.append("final_assistant_not_text")
        else:
            for tag in FINAL_COT_REQUIRED_TAGS:
                if tag not in final_text:
                    errors.append(f"final_missing_tag:{tag}")
            finish, finish_errors = parse_finish_text(final_text)
            errors.extend(f"finish:{item}" for item in finish_errors)
            if finish is not None and case is not None:
                expected_traj = case.ground_truth.get("pixel_trajectory")
                if expected_traj and finish.get("trajectory") != expected_traj:
                    errors.append("finish_trajectory_mismatch")

    metric_result = compute_trace_metrics(trace, case)
    for key in ("parse_success", "stage2_role_order_valid", "trajectory_readout_order_valid", "final_cot_validity"):
        if metric_result.metrics.get(key) != 1.0:
            errors.append(f"metric_failed:{key}")
    errors.extend(metric_result.errors)
    return Stage2ValidationResult(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        metrics=metric_result.metrics,
    )


def validate_ms_swift_stage2_row(row: dict[str, Any]) -> Stage2ValidationResult:
    """Validate one exported ms-swift Stage2 JSONL row.

    输入:
        exporter 产出的 JSON object。

    输出:
        ``Stage2ValidationResult``。重点检查 ms-swift agent-template 所需字段类型：
        ``tools`` 必须是 JSON 字符串，``tool_call`` / ``tool_response`` content 也必须
        是 JSON 字符串。
    """

    errors: list[str] = []
    warnings: list[str] = []
    if row.get("schema_version") != MS_SWIFT_AGENTIC_STAGE2_SCHEMA_VERSION:
        errors.append(f"row_schema_version:{row.get('schema_version')}")
    messages = row.get("messages")
    if not isinstance(messages, list):
        return Stage2ValidationResult(valid=False, errors=["messages_not_list"])
    roles = tuple(str(message.get("role")) for message in messages if isinstance(message, dict))
    if roles != STAGE2_ROW_ROLE_SEQUENCE:
        errors.append(f"row_role_sequence:{roles}")

    tools = row.get("tools")
    if not isinstance(tools, str):
        errors.append("tools_not_json_string")
    else:
        payload, payload_errors = _json_value(tools)
        errors.extend(f"tools:{item}" for item in payload_errors)
        if payload is not None and not isinstance(payload, list):
            errors.append("tools_json_not_list")

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"message_not_object:{index}")
            continue
        role = message.get("role")
        content = message.get("content")
        if role in {"tool_call", "tool_response"}:
            payload, payload_errors = _json_dict(content)
            errors.extend(f"{role}:{item}" for item in payload_errors)
            if role == "tool_call" and payload is not None:
                call = ToolCall.from_dict(payload)
                is_valid, validation_errors = validate_tool_call(call)
                if not is_valid:
                    errors.extend(f"tool_call_schema:{item}" for item in validation_errors)
        if role == "tool" or (isinstance(content, str) and "<tool_response>" in content):
            errors.append("legacy_tool_response_format")
    if len(messages) > 5 and not _contains_all_readout_tokens(str(messages[5].get("content"))):
        errors.append("row_missing_readout_tokens")
    if len(messages) > 6:
        final_text = messages[6].get("content")
        if not isinstance(final_text, str) or not all(tag in final_text for tag in FINAL_COT_REQUIRED_TAGS):
            errors.append("row_final_cot_invalid")
    else:
        errors.append("row_missing_final_assistant")
    if not row.get("images"):
        warnings.append("row_missing_images")
    return Stage2ValidationResult(valid=not errors, errors=errors, warnings=warnings)


def _json_value(text: Any) -> tuple[Any | None, list[str]]:
    if not isinstance(text, str):
        return None, [f"json_text_type:{type(text).__name__}"]
    try:
        return json.loads(text), []
    except json.JSONDecodeError as exc:
        return None, [f"json_decode_error:{exc.msg}"]


def _json_dict(content: Any) -> tuple[dict[str, Any] | None, list[str]]:
    if isinstance(content, dict):
        return dict(content), []
    payload, errors = _json_value(content)
    if errors:
        return None, errors
    if not isinstance(payload, dict):
        return None, ["json_not_object"]
    return payload, []


def _contains_all_readout_tokens(text: str) -> bool:
    return all(token in text for token in NAVIGATION_READOUT_TOKENS)
