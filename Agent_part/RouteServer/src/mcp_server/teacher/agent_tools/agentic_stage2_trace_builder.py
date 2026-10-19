"""Build normalized Agentic Stage2 traces.

该模块是 Stage2 trace 的正式拼装入口。它不负责“选哪个工具”，只负责把已经确定的
``ToolCall`` 和 ``ToolResponse`` 组织成固定因果顺序：

``user -> assistant think -> tool_call -> tool_response -> user trigger -> assistant final + [SYS1_*]``

no-tool replay 使用同一 schema 的短序列：

``user -> assistant think -> user trigger -> assistant final + [SYS1_*]``

重要因果模型:
    SideWalk System1 从 ``[SYS1_i]`` token 的 hidden state 读取连续策略条件。
    对自回归 Transformer hidden state 只能 attend 到当前位置之前的上下文。因此
    V2 ``[SYS1_i]`` 必须位于 ``tool_response``、五段 reasoning 和 ``finish`` 之后。
    ``user_prompt_v1`` 的 ``[TRAJ_i]`` 仅用于旧回放兼容。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp_server.agent.schemas import AgentStage2Case, AgentTrace, ToolCall, ToolResponse
from mcp_server.agent.stage2 import (
    READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2,
    build_agent_state,
    build_agent_think_text,
    build_final_assistant_text,
    build_tool_decision,
    final_navigation_user_content,
    initial_user_content,
    synthetic_tool_response,
)
from mcp_server.agent.tool_parser import parse_agent_action
from mcp_server.agent.tool_schema import get_tool_definitions
from mcp_server.eval.agent_tools.metrics import compute_trace_metrics


@dataclass(frozen=True)
class AgenticStage2TraceBuilderConfig:
    """Configuration for ``AgenticStage2TraceBuilder``.

    属性:
        source: 写入 ``model_run`` 和 final ``<finish>.teacher_source`` 的来源名。
        compute_metrics: True 时在 build 阶段直接填充协议级 metrics。
        readout_placement: 新数据默认使用 assistant suffix V2；V1 仅供旧回放。
    """

    source: str = "agentic_stage2_builder"
    compute_metrics: bool = True
    readout_placement: str = READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2


class AgenticStage2TraceBuilder:
    """Assemble one normalized Stage2 trace from a case and an optional tool interaction.

    输入:
        ``AgentStage2Case``、可选 ``ToolCall``、可选 ``ToolResponse``。tool 路径
        若没有 response 会生成 deterministic synthetic response；no-tool 路径
        则不会生成 ``tool_call`` / ``tool_response`` role。

    输出:
        ``AgentTrace``，其 ``turns`` 满足 Stage2 因果顺序；``supervision`` 继承
        case.ground_truth 用于 exporter 透传 traj/seg loss 字段。
    """

    def __init__(self, config: AgenticStage2TraceBuilderConfig | None = None) -> None:
        self.config = config or AgenticStage2TraceBuilderConfig()

    def build_trace(
        self,
        case: AgentStage2Case,
        *,
        call: ToolCall | None = None,
        response: ToolResponse | None = None,
        tools: list[dict[str, Any]] | None = None,
        model_run: dict[str, Any] | None = None,
        intent: dict[str, Any] | None = None,
        tool_decision_reason: str = "external context is required before final navigation",
        tool_decision_reason_code: str | None = None,
        recommendation: dict[str, Any] | None = None,
        phase1_response: str | None = None,
        teacher_internal: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> AgentTrace:
        """Build a replayable Stage2 trace.

        参数:
            case: Stage2 输入样本。
            call: 模型应该学习生成的工具调用；None 表示 no-tool replay。
            response: 工具返回；tool 路径为空时使用 ``synthetic_tool_response``。
            tools: 学生可见工具 schema 为空时使用默认三类元工具。
            model_run: 记录 runner/teacher 配置的审计信息。
            intent/recommendation: teacher internal context 只用于摘要和合成返回。
            tool_decision_reason_code: 可选离散原因码；为空时由 ``stage2`` helper
                根据 tool 类型推断，保持旧调用兼容。
            phase1_response: 已有 Phase1 CoT 为空时从 case 自动查找，再 fallback。

        返回:
            ``AgentTrace``，包含 messages、tools、supervision、metadata、metrics。
        """

        # trace 从 case 继承 supervision；builder 只负责拼装对话与 teacher metadata。
        trace = AgentTrace.from_case(case, model_run=dict(model_run or {"runner": self.config.source}))
        trace.tools = list(tools or case.tools or get_tool_definitions())
        trace.add_turn("user", initial_user_content(case), metadata={"stage2_step": "initial_request"})

        # 第一轮 assistant 只写 AgentState/ToolDecision；可执行 tool_call 作为独立 role 追加。
        think_text = build_agent_think_text(
            state=build_agent_state(case, intent=intent),
            decision=build_tool_decision(
                call,
                reason=tool_decision_reason,
                reason_code=tool_decision_reason_code,
            ),
        )
        trace.add_turn("assistant", think_text, metadata={"stage2_step": "tool_decision_think"})

        resolved_response = None
        if call is not None:
            # tool 路径：teacher 监督模型生成 tool_call，再把环境返回放入 tool_response role。
            trace.add_turn("tool_call", call.to_ms_swift_dict(), tool=call.tool, metadata={"stage2_step": "tool_call"})
            resolved_response = response or synthetic_tool_response(
                call,
                case,
                source=self.config.source,
                recommendation=recommendation,
            )
            trace.add_turn(
                "tool_response",
                resolved_response.to_dict(),
                tool=call.tool,
                metadata={"stage2_step": "tool_response"},
            )
        trace.add_turn(
            # v2 user turn 只触发最终决策；System1 query 位于 assistant finish 之后。
            "user",
            final_navigation_user_content(
                tool_used=call is not None,
                readout_placement=self.config.readout_placement,
            ),
            metadata={"stage2_step": "trajectory_readout_prompt"},
        )

        # final assistant 复用 Phase1 CoT 时会补 ToolResultUse/finish 字段，保持 LM 与 traj target 对齐。
        final_text = build_final_assistant_text(
            case,
            call=call,
            response=resolved_response,
            source=self.config.source,
            phase1_response=phase1_response,
            readout_placement=self.config.readout_placement,
        )
        trace.add_turn(
            "assistant",
            final_text,
            metadata={
                "stage2_step": "final_navigation",
                "parsed_action": parse_agent_action(final_text).to_dict(),
            },
        )

        if teacher_internal:
            # teacher_internal 只用于审计，不导入 assistant target。
            trace.metadata["teacher_internal"] = dict(teacher_internal)
        if extra_metadata:
            trace.metadata.update(extra_metadata)
        if self.config.compute_metrics:
            # build 阶段先算协议指标，后续 report 可以按 tool/no-tool/failed-tool slice 聚合。
            metric_result = compute_trace_metrics(trace, case)
            trace.metrics = metric_result.metrics
            if metric_result.errors:
                trace.metadata["metric_errors"] = metric_result.errors
        return trace
