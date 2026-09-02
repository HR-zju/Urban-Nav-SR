"""RouteServer compatibility views over the standalone AgentTools schemas.

The replayable case, trace, turn, tool-call, and tool-response data models are
owned by ``urbannav_agent_tools``. RouteServer keeps only the methods that
adapt those provider-neutral values to RouteServer runtime contracts or the
ms-swift wire format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from urbannav_agent_tools.schemas import (
    AGENT_EVAL_CASE_SCHEMA_VERSION,
    AGENT_STAGE2_CASE_SCHEMA_VERSION,
    AGENT_STAGE2_TRACE_SCHEMA_VERSION,
    AGENT_TRACE_SCHEMA_VERSION,
    AgentEvalCase,
    AgentMetricResult,
    AgentStage2Case as _AgentStage2Case,
    AgentTrace as _AgentTrace,
    AgentTurn,
    ToolCall as _ToolCall,
    ToolResponse,
)


MS_SWIFT_AGENTIC_STAGE2_SCHEMA_VERSION = "sidewalk_agentic_stage2_sft_v1"


@dataclass(frozen=True)
class ToolCall(_ToolCall):
    """RouteServer tool call with the legacy ms-swift wire-format view.

    Args:
        tool: Stable RouteServer tool name.
        arguments: JSON-compatible tool arguments.
        call_id: Optional identifier shared with the tool response.

    Returns:
        A standalone AgentTools value with one RouteServer compatibility
        serializer.
    """

    def to_ms_swift_dict(self) -> dict[str, Any]:
        """Return the historical ``name``/``arguments`` representation."""

        return self.to_provider_dict()


@dataclass
class AgentStage2Case(_AgentStage2Case):
    """RouteServer Stage2 case with a runtime observation adapter.

    Args:
        case_id: Stable case identifier.
        user_request: User navigation request.
        task_type: Dataset task type.
        schema_version: Stage2 case schema version.
        observation: Runtime-visible observation payload.
        dialogue_history: Prior dialogue turns.
        tools: Student-visible tool definitions.
        ground_truth: Evaluation or training supervision.
        split: Dataset split.
        locale: Request locale.
        metadata: Non-model audit metadata.
    """

    def to_nav_observation(self):
        """Return the RouteServer ``NavObservation`` compatibility view."""

        from mcp_server.agent.contracts import NavObservation

        return NavObservation.from_case(self)


@dataclass
class AgentTrace(_AgentTrace):
    """RouteServer trace with the unified one-step result adapter.

    Args:
        case_id: Stable source case identifier.
        task_type: Trace task type.
        schema_version: Trace schema version.
        request: Original request/case payload.
        turns: Ordered dialogue and tool turns.
        tools: Explicit tool definitions used for the trace.
        artifacts: External artifact references.
        supervision: Training or evaluation labels.
        metrics: Diagnostic metric values.
        model_run: Model/provider provenance.
        metadata: Additional audit metadata.
    """

    def to_agent_step_result(
        self,
        *,
        case: AgentStage2Case | None = None,
        validation: dict[str, Any] | None = None,
    ):
        """Return the RouteServer ``AgentStepResult`` compatibility view."""

        from mcp_server.agent.contracts import AgentStepResult

        return AgentStepResult.from_trace(self, case=case, validation=validation)


__all__ = [
    "AGENT_EVAL_CASE_SCHEMA_VERSION",
    "AGENT_STAGE2_CASE_SCHEMA_VERSION",
    "AGENT_STAGE2_TRACE_SCHEMA_VERSION",
    "AGENT_TRACE_SCHEMA_VERSION",
    "AgentEvalCase",
    "AgentMetricResult",
    "AgentStage2Case",
    "AgentTrace",
    "AgentTurn",
    "MS_SWIFT_AGENTIC_STAGE2_SCHEMA_VERSION",
    "ToolCall",
    "ToolResponse",
]
