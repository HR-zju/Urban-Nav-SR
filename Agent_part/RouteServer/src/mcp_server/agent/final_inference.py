"""Final RouteServer inference interface for Agentic Stage2.

This module is the application-facing wrapper around the lower-level online
runner.  It accepts RouteServer runtime inputs (images, user request, route
state, and tools), runs the single-tool Stage2 loop, parses ``<finish>``, and
can persist a replayable ms-swift row for audit or future SFT.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp_server.agent.schemas import AgentStage2Case
from mcp_server.agent.tool_executor import ToolExecutor
from mcp_server.agent.tool_parser import parse_finish_text
from mcp_server.agent.tool_runtime import AgentToolRuntime, AgentToolRuntimeConfig, ToolBudget
from mcp_server.agent.tool_schema import get_tool_definitions
from mcp_server.eval.online_agent_runner import ModelAdapter, OnlineAgentRunner, OnlineAgentRunnerConfig


@dataclass(frozen=True)
class RouteServerAgentInferenceRequest:
    """Input payload for one online RouteServer agent inference.

    Args:
        case_id: Stable case/request id.
        user_request: Natural-language user navigation request.
        images: FPV/BEV image paths.  Accepts either an ordered list or a
            mapping such as ``{"fpv": "...", "bev": "..."}``.
        route_state: Current route/session state visible to the model.
        current_location: Optional ``"lon,lat"`` string or location object.
        tools: Optional decoded tool schema.  Defaults to RouteServer Stage2
            tools when omitted.
        locale: Locale string carried into the internal ``AgentStage2Case``.
        metadata: Extra audit metadata.
        supervision: Optional offline labels such as ``pixel_trajectory``.
            Online inference does not require these labels, but preserving them
            lets validation check teacher-forced smoke cases.
    """

    case_id: str
    user_request: str
    images: list[str] | dict[str, str]
    route_state: dict[str, Any] = field(default_factory=dict)
    current_location: Any = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    locale: str = "zh-CN"
    metadata: dict[str, Any] = field(default_factory=dict)
    supervision: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RouteServerAgentInferenceRequest":
        """Build a request from JSON input.

        Args:
            payload: JSON object accepted by the final inference CLI.

        Returns:
            ``RouteServerAgentInferenceRequest``.

        Raises:
            ValueError: If required fields are missing.
        """

        case_id = str(payload.get("case_id") or payload.get("id") or "").strip()
        user_request = str(payload.get("user_request") or payload.get("instruction") or "").strip()
        images = payload.get("images")
        if not case_id:
            raise ValueError("case_id is required.")
        if not user_request:
            raise ValueError("user_request is required.")
        if not isinstance(images, (list, dict)):
            raise ValueError("images must be a list or object.")
        return cls(
            case_id=case_id,
            user_request=user_request,
            images=images,
            route_state=dict(payload.get("route_state") or {}),
            current_location=payload.get("current_location"),
            tools=list(payload.get("tools") or []),
            locale=str(payload.get("locale") or "zh-CN"),
            metadata=dict(payload.get("metadata") or {}),
            supervision=dict(payload.get("supervision") or {}),
        )

    def to_case(self) -> AgentStage2Case:
        """Convert the request into the canonical Stage2 case.

        Returns:
            ``AgentStage2Case`` consumed by ``OnlineAgentRunner``.
        """

        # RouteServer 外部请求先归一化成 AgentStage2Case；在线推理不依赖 GT supervision。
        observation: dict[str, Any] = {
            "images": self.images,
            "route_state": dict(self.route_state),
        }
        if self.current_location is not None:
            observation["current_location"] = self.current_location
        if self.route_state.get("route_session_id") is not None:
            observation["route_session_id"] = self.route_state.get("route_session_id")
        return AgentStage2Case(
            case_id=self.case_id,
            user_request=self.user_request,
            observation=observation,
            tools=list(self.tools or get_tool_definitions()),
            ground_truth=dict(self.supervision),
            locale=self.locale,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the request.

        Returns:
            JSON-serializable request dictionary.
        """

        return {
            "case_id": self.case_id,
            "user_request": self.user_request,
            "images": self.images,
            "route_state": dict(self.route_state),
            "current_location": self.current_location,
            "tools": list(self.tools),
            "locale": self.locale,
            "metadata": dict(self.metadata),
            "supervision": dict(self.supervision),
        }


@dataclass(frozen=True)
class RouteServerAgentInferenceConfig:
    """Configuration for final RouteServer agent inference.

    Args:
        source: Runner/source name written into trace metadata.
        system_prompt: Optional system prompt override.
        max_tool_calls: Runtime tool-call budget.  Stage2 should keep this at
            ``1``.
        allowed_ops_by_tool: Optional per-tool operation allowlist.
        review_artifact_dir: Optional directory where runtime log/trace JSON
            files are written by ``AgentToolRuntime``.
    """

    source: str = "routeserver_final_inference"
    system_prompt: str | None = None
    max_tool_calls: int = 1
    allowed_ops_by_tool: dict[str, tuple[str, ...]] = field(default_factory=dict)
    review_artifact_dir: str | Path | None = None


@dataclass(frozen=True)
class RouteServerAgentInferenceResult:
    """Output payload for one final RouteServer agent inference.

    Args:
        request: Original input request.
        final_assistant_text: Final assistant text containing navigation CoT
            and ``<finish>`` when parsing succeeds.
        finish: Parsed ``<finish>`` JSON object, or ``None``.
        finish_parse_errors: Parser errors for ``<finish>``.
        trajectory: ``finish.trajectory`` if present.
        trace: Normalized replayable AgentTrace dictionary.
        ms_swift_row: ms-swift-compatible replay row.
        validation_report: Trace and row validation results.
        artifacts: Runtime and persistence artifacts.
    """

    request: RouteServerAgentInferenceRequest
    final_assistant_text: str
    finish: dict[str, Any] | None
    finish_parse_errors: list[str]
    trajectory: Any
    trace: dict[str, Any]
    ms_swift_row: dict[str, Any]
    validation_report: dict[str, Any]
    artifacts: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize inference result.

        Returns:
            JSON-ready result dictionary.
        """

        return {
            "schema_version": "routeserver_agentic_stage2_final_inference_result_v1",
            "request": self.request.to_dict(),
            "final_assistant_text": self.final_assistant_text,
            "finish": self.finish,
            "finish_parse_errors": list(self.finish_parse_errors),
            "trajectory": self.trajectory,
            "trace": self.trace,
            "ms_swift_row": self.ms_swift_row,
            "validation_report": self.validation_report,
            "artifacts": list(self.artifacts),
        }


async def run_routeserver_final_inference(
    request: RouteServerAgentInferenceRequest,
    *,
    model_adapter: ModelAdapter,
    executor: ToolExecutor | None = None,
    config: RouteServerAgentInferenceConfig | None = None,
) -> RouteServerAgentInferenceResult:
    """Run one final RouteServer Agentic Stage2 inference.

    Args:
        request: Runtime request containing images, user request, route state,
            and optional tools.
        model_adapter: Model backend, for example ``MsSwiftInferenceAdapter``.
        executor: Optional ``ToolExecutor``.  If omitted, tool calls become
            structured ``missing_tool_executor`` tool responses.
        config: Inference/runtime configuration.

    Returns:
        ``RouteServerAgentInferenceResult`` with parsed finish, replay trace,
        ms-swift row, validation report, and artifacts.
    """

    config = config or RouteServerAgentInferenceConfig()
    # 最终接口仍使用 Stage2 single-tool 协议；多轮工具循环留给后续 Stage3/runtime eval。
    runtime = AgentToolRuntime(
        executor=executor,
        config=AgentToolRuntimeConfig(
            source=config.source,
            budget=ToolBudget(
                max_tool_calls=config.max_tool_calls,
                allowed_ops_by_tool=dict(config.allowed_ops_by_tool),
            ),
            review_artifact_dir=config.review_artifact_dir,
        ),
    )
    # OnlineAgentRunner 做两次模型调用，中间由 AgentToolRuntime 执行工具并补齐 tool_response。
    runner_config = OnlineAgentRunnerConfig(source=config.source)
    if config.system_prompt is not None:
        runner_config = OnlineAgentRunnerConfig(source=config.source, system_prompt=config.system_prompt)
    result = await OnlineAgentRunner(
        model_adapter=model_adapter,
        runtime=runtime,
        config=runner_config,
    ).run_case_with_report(request.to_case())
    # RouteServer 对外返回 parsed finish/trajectory；同时保留原始 assistant 文本供人工审计。
    final_text = _final_assistant_text(result.trace.to_dict())
    finish, finish_errors = parse_finish_text(final_text) if final_text else (None, ["missing_final_assistant"])
    artifacts = list(result.trace.artifacts)
    return RouteServerAgentInferenceResult(
        request=request,
        final_assistant_text=final_text,
        finish=finish,
        finish_parse_errors=finish_errors,
        trajectory=finish.get("trajectory") if isinstance(finish, dict) else None,
        trace=result.trace.to_dict(),
        ms_swift_row=result.ms_swift_row,
        validation_report={
            "trace": result.trace_validation.to_dict(),
            "ms_swift_row": result.row_validation.to_dict(),
            "finish": {
                "valid": finish is not None and not finish_errors,
                "errors": list(finish_errors),
            },
        },
        artifacts=artifacts,
    )


def write_routeserver_final_inference_artifacts(
    result: RouteServerAgentInferenceResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """Persist final inference artifacts to disk.

    Args:
        result: Inference result returned by
            ``run_routeserver_final_inference``.
        output_dir: Destination directory.

    Returns:
        Mapping of artifact names to file paths.
    """

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    # 分开落盘：result 供应用读取，trace/replay row 供问题复现和后续 SFT。
    paths = {
        "result": target / "result.json",
        "trace": target / "agent_trace.json",
        "ms_swift_replay_row": target / "ms_swift_replay_row.json",
        "validation_report": target / "validation_report.json",
        "artifact_index": target / "artifact_index.json",
    }
    _write_json(result.to_dict(), paths["result"])
    _write_json(result.trace, paths["trace"])
    _write_json(result.ms_swift_row, paths["ms_swift_replay_row"])
    _write_json(result.validation_report, paths["validation_report"])
    artifact_index = {
        "schema_version": "routeserver_agentic_stage2_final_inference_artifact_index_v1",
        "case_id": result.request.case_id,
        "files": {name: str(path) for name, path in paths.items() if name != "artifact_index"},
        "runtime_artifacts": list(result.artifacts),
    }
    _write_json(artifact_index, paths["artifact_index"])
    return {name: str(path) for name, path in paths.items()}


def load_routeserver_final_inference_request(path: str | Path) -> RouteServerAgentInferenceRequest:
    """Load one final inference request JSON file.

    Args:
        path: JSON file path.

    Returns:
        Parsed ``RouteServerAgentInferenceRequest``.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return RouteServerAgentInferenceRequest.from_dict(payload)


def _final_assistant_text(trace: dict[str, Any]) -> str:
    for turn in reversed(trace.get("turns") or []):
        if isinstance(turn, dict) and turn.get("role") == "assistant":
            return str(turn.get("content") or "")
    return ""


def _write_json(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
