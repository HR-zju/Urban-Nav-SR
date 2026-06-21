"""Stable Agentic Stage2 data schemas.

这些 dataclass 是 RouteServer teacher、rule baseline、外部 LVLM 和本项目
E2E LVLM checkpoint 之间的公共协议。Stage2 直接使用最终 agentic cold-start
格式：工具调用是独立 ``tool_call`` role，工具返回是独立 ``tool_response`` role，
最终导航 readout prompt 在工具返回之后插入 ``[TRAJ_*]`` token。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


AGENT_EVAL_CASE_SCHEMA_VERSION = "sidewalk_agent_stage2_case_v1"
AGENT_TRACE_SCHEMA_VERSION = "sidewalk_agent_trace_v2"
MS_SWIFT_AGENTIC_STAGE2_SCHEMA_VERSION = "sidewalk_agentic_stage2_sft_v1"


def _clean_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop empty optional fields while preserving valid false/zero values."""

    return {
        key: value
        for key, value in payload.items()
        if value is not None and value != {} and value != []
    }


@dataclass(frozen=True)
class ToolCall:
    """One model requested tool call.

    输入格式:
        ``{"tool": "map_tool", "arguments": {"op": "search_poi", ...}}``
        或 ms-swift agent-template 格式
        ``{"name": "map_tool", "arguments": {"op": "search_poi", ...}}``。

    输出格式:
        ``to_dict()`` 返回稳定 JSON，供 executor、metrics、exporter 共用。
        ``to_ms_swift_dict()`` 返回 ``name/arguments`` 形态，供 ``tool_call`` role
        直接序列化为 JSON 字符串。
    """

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ToolCall":
        tool = str(payload.get("tool") or payload.get("name") or "").strip()
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {
                key: value
                for key, value in payload.items()
                if key not in {"tool", "name", "arguments", "call_id", "id"}
            }
        return cls(
            tool=tool,
            arguments=dict(arguments),
            call_id=payload.get("call_id") or payload.get("id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict(
            {
                "tool": self.tool,
                "arguments": dict(self.arguments),
                "call_id": self.call_id,
            }
        )

    def to_ms_swift_dict(self) -> dict[str, Any]:
        return _clean_dict(
            {
                "name": self.tool,
                "arguments": dict(self.arguments),
                "call_id": self.call_id,
            }
        )

    @property
    def op(self) -> str | None:
        op = self.arguments.get("op")
        return str(op).strip() if op is not None else None


@dataclass(frozen=True)
class ToolResponse:
    """One deterministic tool execution result.

    输入:
        ``ToolExecutor.execute()`` 生成的成功/失败结果。

    输出:
        ``to_dict()`` 可直接写入 ``AgentTrace.turns`` 的 ``tool_response`` role。
        exporter 会把该 JSON object 序列化成 JSON 字符串，不再包 ``<tool_response>``
        XML。
    """

    tool: str
    success: bool
    result: Any = field(default_factory=dict)
    error: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict(
            {
                "tool": self.tool,
                "success": bool(self.success),
                "result": self.result,
                "error": self.error,
                "artifacts": list(self.artifacts),
                "call_id": self.call_id,
            }
        )


@dataclass
class AgentTurn:
    """One replayable dialogue turn.

    role:
        ``system`` / ``user`` / ``assistant`` / ``tool_call`` / ``tool_response``。
    content:
        原始文本或 JSON。Stage2 中 ``tool_call`` content 是 JSON object/string，
        形如 ``{"name": "...", "arguments": {...}}``；assistant content 保留
        ``<think>`` 和最终 ``<finish>`` 文本监督。
    metadata:
        解析结果、teacher 来源、时间戳等附加信息。
    """

    role: str
    content: Any
    tool: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentTurn":
        return cls(
            role=str(payload.get("role") or ""),
            content=payload.get("content"),
            tool=payload.get("tool") or payload.get("name"),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict(
            {
                "role": self.role,
                "content": self.content,
                "tool": self.tool,
                "metadata": dict(self.metadata),
            }
        )


@dataclass
class AgentEvalCase:
    """One offline evaluation/training case for Agentic Stage2.

    JSONL 行格式:
        ``schema_version`` 固定为 ``sidewalk_agent_stage2_case_v1``。
        ``observation.images`` 可为 ``{"fpv": "...", "bev": "..."}`` 或图片路径列表。
        ``ground_truth.expected_tool_ops`` 可选，用于 tool-op 指标；Phase1 的
        ``pixel_trajectory``、``trajectory_fpv_t``、``semantic_segmentation_path`` 等
        监督字段继续放在 ``ground_truth`` 中并由 exporter 透传。

    该结构不包含新 teacher LVLM 输出；teacher 只在 runner 阶段生成 AgentTrace。
    """

    case_id: str
    user_request: str
    task_type: str = "sidewalk_navigation_agent"
    schema_version: str = AGENT_EVAL_CASE_SCHEMA_VERSION
    observation: dict[str, Any] = field(default_factory=dict)
    dialogue_history: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    ground_truth: dict[str, Any] = field(default_factory=dict)
    split: str = "eval"
    locale: str = "zh-CN"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentEvalCase":
        return cls(
            case_id=str(payload.get("case_id") or payload.get("id") or ""),
            user_request=str(payload.get("user_request") or payload.get("instruction") or ""),
            task_type=str(payload.get("task_type") or "sidewalk_navigation_agent"),
            schema_version=str(payload.get("schema_version") or AGENT_EVAL_CASE_SCHEMA_VERSION),
            observation=dict(payload.get("observation") or {}),
            dialogue_history=list(payload.get("dialogue_history") or payload.get("messages") or []),
            tools=list(payload.get("tools") or []),
            ground_truth=dict(payload.get("ground_truth") or {}),
            split=str(payload.get("split") or "eval"),
            locale=str(payload.get("locale") or "zh-CN"),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict(
            {
                "schema_version": self.schema_version,
                "case_id": self.case_id,
                "task_type": self.task_type,
                "user_request": self.user_request,
                "observation": dict(self.observation),
                "dialogue_history": list(self.dialogue_history),
                "tools": list(self.tools),
                "ground_truth": dict(self.ground_truth),
                "split": self.split,
                "locale": self.locale,
                "metadata": dict(self.metadata),
            }
        )

    @property
    def images(self) -> list[str]:
        images = self.observation.get("images")
        if isinstance(images, list):
            return [str(item) for item in images if item]
        if isinstance(images, dict):
            ordered = [images.get("fpv"), images.get("front"), images.get("bev")]
            ordered.extend(value for key, value in images.items() if key not in {"fpv", "front", "bev"})
            return [str(item) for item in ordered if item]
        return []


@dataclass
class AgentTrace:
    """Replayable Agentic Stage2 trajectory generated from one ``AgentEvalCase``.

    输入:
        case + runner/model output + optional tool responses。

    输出:
        ``to_dict()`` 是 evaluator 的可回放 trace；exporter 会进一步转成
        ms-swift agent-template 的
        ``messages/tools/images/pixel_trajectory/sidewalk_meta`` JSONL。
    """

    case_id: str
    task_type: str = "sidewalk_navigation_agent"
    schema_version: str = AGENT_TRACE_SCHEMA_VERSION
    request: dict[str, Any] = field(default_factory=dict)
    turns: list[AgentTurn] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    supervision: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    model_run: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_case(cls, case: AgentEvalCase, *, model_run: dict[str, Any] | None = None) -> "AgentTrace":
        return cls(
            case_id=case.case_id,
            task_type=case.task_type,
            request=case.to_dict(),
            tools=list(case.tools),
            supervision=dict(case.ground_truth),
            model_run=dict(model_run or {}),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentTrace":
        return cls(
            case_id=str(payload.get("case_id") or ""),
            task_type=str(payload.get("task_type") or "sidewalk_navigation_agent"),
            schema_version=str(payload.get("schema_version") or AGENT_TRACE_SCHEMA_VERSION),
            request=dict(payload.get("request") or {}),
            turns=[AgentTurn.from_dict(item) for item in payload.get("turns") or []],
            tools=list(payload.get("tools") or []),
            artifacts=list(payload.get("artifacts") or []),
            supervision=dict(payload.get("supervision") or {}),
            metrics=dict(payload.get("metrics") or {}),
            model_run=dict(payload.get("model_run") or {}),
            metadata=dict(payload.get("metadata") or {}),
        )

    def add_turn(
        self,
        role: str,
        content: Any,
        *,
        tool: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.turns.append(AgentTurn(role=role, content=content, tool=tool, metadata=dict(metadata or {})))

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict(
            {
                "schema_version": self.schema_version,
                "case_id": self.case_id,
                "task_type": self.task_type,
                "request": dict(self.request),
                "turns": [turn.to_dict() for turn in self.turns],
                "tools": list(self.tools),
                "artifacts": list(self.artifacts),
                "supervision": dict(self.supervision),
                "metrics": dict(self.metrics),
                "model_run": dict(self.model_run),
                "metadata": dict(self.metadata),
            }
        )


@dataclass(frozen=True)
class AgentMetricResult:
    """Metric output envelope returned by eval.metrics."""

    metrics: dict[str, float | None]
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "errors": list(self.errors),
        }
