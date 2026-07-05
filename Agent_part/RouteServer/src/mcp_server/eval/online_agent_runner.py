"""Online Stage2 runner that connects model generation with RouteServer tools.

The offline teacher pipeline already produces replayable Stage2 traces.  This
module is the runtime evaluation layer: it calls a model once for the pre-tool
decision, executes at most one tool through ``AgentToolRuntime``, then calls the
model again for final navigation after ``tool_response`` and ``[TRAJ_i]`` are
visible.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from mcp_server.agent.schemas import AgentStage2Case, AgentTrace
from mcp_server.agent.stage2 import initial_user_content
from mcp_server.agent.tool_runtime import AgentToolRuntime
from mcp_server.agent.tool_schema import get_tool_definitions
from mcp_server.eval.exporters import DEFAULT_AGENT_SYSTEM_PROMPT, trace_to_ms_swift_stage2_row
from mcp_server.eval.runners import AgentRunner
from mcp_server.eval.stage2_validator import (
    Stage2ValidationResult,
    validate_ms_swift_stage2_row,
    validate_stage2_trace,
)
from mcp_server.llm import (
    LLMSettings,
    apply_llm_reasoning_effort,
    close_async_llm_client,
    create_async_llm_client,
    extract_chat_response_text,
)


class ModelAdapter(Protocol):
    """Protocol implemented by real or fake Stage2 model backends.

    Implementations should treat ``messages`` as OpenAI/ms-swift-style chat
    messages, ``tools`` as decoded top-level tool schema, and ``images`` as
    FPV/BEV image paths.  The adapter owns all model-specific formatting.
    """

    async def generate_pre_tool(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        images: list[str],
    ) -> str:
        """Generate the first assistant turn.

        Args:
            messages: System + initial user messages.
            tools: Decoded tool definitions.
            images: Image paths associated with the case.

        Returns:
            Assistant text containing ``<AgentState>/<ToolDecision>`` and
            optionally one sibling ``<tool_call>{...}</tool_call>`` block.
        """

    async def generate_final(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        images: list[str],
    ) -> str:
        """Generate the final assistant navigation turn.

        Args:
            messages: Full prefix through optional ``tool_response`` and final
                readout user turn.
            tools: Decoded tool definitions.
            images: Image paths associated with the case.

        Returns:
            Final assistant text containing navigation CoT and ``<finish>``.
        """


@dataclass(frozen=True)
class OnlineAgentRunResult:
    """Structured output for one online Stage2 run.

    Args:
        trace: Normalized ``AgentTrace`` produced by the online runner.
        ms_swift_row: Same trace exported to ms-swift agent-template row.
        trace_validation: Validation result for ``trace``.
        row_validation: Validation result for ``ms_swift_row``.
    """

    trace: AgentTrace
    ms_swift_row: dict[str, Any]
    trace_validation: Stage2ValidationResult
    row_validation: Stage2ValidationResult

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report dictionary."""

        agent_step_result = self.trace.to_agent_step_result(
            validation=self.trace_validation.to_dict(),
        ).to_dict()
        return {
            "trace": self.trace.to_dict(),
            "ms_swift_row": dict(self.ms_swift_row),
            "trace_validation": self.trace_validation.to_dict(),
            "row_validation": self.row_validation.to_dict(),
            "runtime_state": agent_step_result.get("runtime_state"),
            "dialogue_state": self.trace.metadata.get("dialogue_state") or {},
            "agent_step_result": agent_step_result,
        }


@dataclass(frozen=True)
class OnlineAgentRunnerConfig:
    """Configuration for ``OnlineAgentRunner``.

    Args:
        source: Runner name written into trace metadata.
        system_prompt: System prompt used for the first model call.
    """

    source: str = "online_agent_runner"
    system_prompt: str = DEFAULT_AGENT_SYSTEM_PROMPT


class OnlineAgentRunner(AgentRunner):
    """Run a real or fake model through the Stage2 online tool loop.

    Args:
        model_adapter: Backend responsible for model generation.
        runtime: ``AgentToolRuntime`` that parses tool calls, executes tools,
            appends ``tool_response`` and final readout tokens.
        config: Runner configuration.

    Returns:
        ``run_case`` returns ``AgentTrace`` for compatibility with existing
        evaluator helpers.  ``run_case_with_report`` returns trace, exported
        ms-swift row, and validation results.
    """

    def __init__(
        self,
        *,
        model_adapter: ModelAdapter,
        runtime: AgentToolRuntime,
        config: OnlineAgentRunnerConfig | None = None,
    ) -> None:
        self.model_adapter = model_adapter
        self.runtime = runtime
        self.config = config or OnlineAgentRunnerConfig()

    async def run_case(self, case: AgentStage2Case) -> AgentTrace:
        """Run one case and return only the normalized trace."""

        return (await self.run_case_with_report(case)).trace

    async def run_case_with_report(self, case: AgentStage2Case) -> OnlineAgentRunResult:
        """Run one case and return trace, ms-swift row, and validations.

        Args:
            case: Stage2 evaluation case.

        Returns:
            ``OnlineAgentRunResult`` with replayable artifacts.
        """

        tools = list(case.tools or get_tool_definitions())
        images = case.images
        # 第一次模型调用只看 system/user/images/tools，让模型决定是否需要工具。
        first_messages = build_initial_model_messages(case, system_prompt=self.config.system_prompt)
        pre_tool_output = await self.model_adapter.generate_pre_tool(first_messages, tools, images)

        async def generate_final(prefix_trace: AgentTrace) -> str:
            # 将 runtime prefix 导成 ms-swift row，保证第二次模型调用看到规范 role 序列。
            row = trace_to_ms_swift_stage2_row(prefix_trace, case)
            return await self.model_adapter.generate_final(row["messages"], tools, row.get("images") or images)

        # AgentToolRuntime 负责解析 tool_call、执行工具、追加 tool_response 和 final readout。
        trace = await self.runtime.run_case(
            case,
            first_assistant_output=pre_tool_output,
            final_assistant_generator=generate_final,
            tools=tools,
        )
        trace.model_run["runner"] = self.config.source
        trace.model_run["model_adapter"] = self.model_adapter.__class__.__name__
        # 同一份 trace 同时做 schema validation 和 ms-swift replay row 导出。
        row = trace_to_ms_swift_stage2_row(trace, case)
        trace_validation = validate_stage2_trace(trace, case)
        row_validation = validate_ms_swift_stage2_row(row)
        agent_step_result = trace.to_agent_step_result(
            case=case,
            validation=trace_validation.to_dict(),
        )
        # online report 需要显式暴露 runtime state；真正的对话状态仍由
        # AgentToolRuntime 从 tool_response 中提升，避免 runner 猜测业务含义。
        trace.metadata.setdefault("runtime_state", agent_step_result.runtime_state)
        trace.metadata["agent_step_result_schema"] = agent_step_result.schema_version
        trace.metadata["online_validation"] = {
            "trace": trace_validation.to_dict(),
            "ms_swift_row": row_validation.to_dict(),
        }
        return OnlineAgentRunResult(
            trace=trace,
            ms_swift_row=row,
            trace_validation=trace_validation,
            row_validation=row_validation,
        )


class CallableModelAdapter:
    """ModelAdapter backed by sync or async Python callables.

    Args:
        pre_tool: Callable used for ``generate_pre_tool``.
        final: Callable used for ``generate_final``.

    This adapter is useful for deterministic unit tests and for wrapping an
    in-process model object without committing RouteServer to a specific ML
    framework.
    """

    def __init__(
        self,
        *,
        pre_tool: Callable[[list[dict[str, Any]], list[dict[str, Any]], list[str]], str | Awaitable[str]],
        final: Callable[[list[dict[str, Any]], list[dict[str, Any]], list[str]], str | Awaitable[str]],
    ) -> None:
        self._pre_tool = pre_tool
        self._final = final

    async def generate_pre_tool(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        images: list[str],
    ) -> str:
        """Return pre-tool text from the configured callable."""

        return await _maybe_await(self._pre_tool(messages, tools, images))

    async def generate_final(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        images: list[str],
    ) -> str:
        """Return final text from the configured callable."""

        return await _maybe_await(self._final(messages, tools, images))


@dataclass(frozen=True)
class OpenAICompatibleInferenceConfig:
    """Configuration for OpenAI-compatible Stage2 inference.

    Args:
        settings: Optional LLM settings.  When omitted, settings are loaded from
            environment variables such as ``LLM_PROVIDER`` and ``ARK_API_KEY``.
        case: Optional Stage2 case whose non-supervision runtime context is
            embedded in the live LLM prompt.
        pre_tool_max_tokens: Token budget for the first tool-decision call.
        final_max_tokens: Token budget for the final navigation call.
        temperature: Optional generation temperature override.  If omitted,
            ``LLMSettings.temperature`` is used.
        client_factory: Optional test hook that returns an OpenAI-compatible
            async client for the supplied settings.
    """

    settings: LLMSettings | None = None
    case: AgentStage2Case | None = None
    pre_tool_max_tokens: int = 512
    final_max_tokens: int = 1024
    temperature: float | None = None
    client_factory: Callable[[LLMSettings], Any] | None = None


class OpenAICompatibleStage2Adapter:
    """ModelAdapter backed by an OpenAI-compatible chat-completions API.

    Args:
        config: LLM provider settings and generation controls.

    This adapter is intended for live smoke tests with providers such as Ark /
    Doubao.  It does not use provider function-calling; instead it prompts the
    model to emit the same textual Stage2 protocol that ms-swift trains on.
    """

    def __init__(self, config: OpenAICompatibleInferenceConfig | None = None) -> None:
        """Create the adapter.

        Args:
            config: Optional OpenAI-compatible inference config.
        """

        self.config = config or OpenAICompatibleInferenceConfig()
        self.settings = self.config.settings or LLMSettings.from_env()

    async def generate_pre_tool(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        images: list[str],
    ) -> str:
        """Generate the first Stage2 assistant turn through the LLM API.

        Args:
            messages: System and initial user messages.
            tools: Student-visible tool schema.
            images: FPV/BEV image paths associated with the case.

        Returns:
            Assistant content containing ``<think>`` and optionally one
            sibling ``<tool_call>`` block.
        """

        return await self._generate(messages, tools, images, target="tool_decision")

    async def generate_final(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        images: list[str],
    ) -> str:
        """Generate the final Stage2 navigation turn through the LLM API.

        Args:
            messages: Full prefix through optional tool response and trajectory
                readout prompt.
            tools: Student-visible tool schema.
            images: FPV/BEV image paths associated with the case.

        Returns:
            Assistant content containing final CoT tags and ``<finish>``.
        """

        return await self._generate(messages, tools, images, target="final_navigation")

    def build_chat_messages(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        images: list[str],
        *,
        target: str,
    ) -> list[dict[str, str]]:
        """Build one chat-completions request for a Stage2 generation target.

        Args:
            messages: RouteServer/ms-swift-style conversation prefix.
            tools: Student-visible tool schema.
            images: FPV/BEV image paths.  The API call is text-only; paths are
                included for audit/context but image bytes are not uploaded.
            target: ``tool_decision`` or ``final_navigation``.

        Returns:
            OpenAI-compatible chat messages.
        """

        has_tool_response = any(message.get("role") == "tool_response" for message in messages)
        prompt = {
            "target": target,
            "runtime_context": _stage2_runtime_context(self.config.case),
            "images": list(images),
            "tools": list(tools),
            "conversation_prefix": [_normalize_prompt_message(message) for message in messages],
            "tool_response_available": has_tool_response,
            "output_contract": _stage2_output_contract(target=target, has_tool_response=has_tool_response),
        }
        return [
            {"role": "system", "content": _stage2_llm_system_prompt(target)},
            {
                "role": "user",
                "content": (
                    "Generate only the next assistant message for this RouteServer Stage2 prefix.\n"
                    "Do not wrap the answer in markdown or code fences.\n"
                    f"{json.dumps(prompt, ensure_ascii=False, sort_keys=True)}"
                ),
            },
        ]

    async def _generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        images: list[str],
        *,
        target: str,
    ) -> str:
        """Run one chat-completions request and return assistant text."""

        if not self.settings.api_key:
            raise RuntimeError("OpenAI-compatible LLM backend requires ARK_API_KEY or LLM_API_KEY.")
        client_factory = self.config.client_factory or create_async_llm_client
        client = client_factory(self.settings)
        if client is None:
            raise RuntimeError("OpenAI-compatible LLM backend requires the openai Python package.")
        try:
            payload = {
                "model": self.settings.model,
                "messages": self.build_chat_messages(messages, tools, images, target=target),
                "temperature": self.config.temperature
                if self.config.temperature is not None
                else self.settings.temperature,
                "max_tokens": self.config.pre_tool_max_tokens
                if target == "tool_decision"
                else self.config.final_max_tokens,
            }
            response = await client.chat.completions.create(**apply_llm_reasoning_effort(payload, self.settings))
            text = extract_chat_response_text(response).strip()
            if not text:
                raise RuntimeError(f"OpenAI-compatible LLM returned empty text for target={target}.")
            return text
        finally:
            await close_async_llm_client(client)


@dataclass(frozen=True)
class MsSwiftInferenceConfig:
    """Configuration for real ms-swift subprocess inference.

    Args:
        ms_swift_root: Path to the ms-swift repository.
        model: Base model id or checkpoint path passed to
            ``swift.sidewalk_nav.cli.infer``.
        adapter_dir: Optional LoRA adapter directory.
        checkpoint_dir: Optional SideWalk checkpoint sidecar directory.
        phase: SideWalk phase for ms-swift inference.
        agent_template: ms-swift agent template name, for example ``hermes``.
        python_executable: Python executable used to launch ms-swift.
        work_dir: Optional directory for temporary one-row manifests and model
            outputs.  If omitted, Python's temporary directory is used.
        max_tokens: Generation token budget.
        temperature: Generation temperature.
        torch_dtype: dtype argument accepted by ms-swift infer CLI.
        device_map: Optional device map.
        enable_thinking: Whether to pass ``--enable-thinking``.
        extra_args: Additional CLI arguments appended verbatim.
    """

    ms_swift_root: str | Path
    model: str
    adapter_dir: str | None = None
    checkpoint_dir: str | None = None
    phase: str = "phase0"
    agent_template: str | None = None
    python_executable: str = sys.executable
    work_dir: str | Path | None = None
    max_tokens: int = 512
    temperature: float = 0.0
    torch_dtype: str = "auto"
    device_map: str | None = None
    enable_thinking: bool = False
    extra_args: tuple[str, ...] = field(default_factory=tuple)


class MsSwiftInferenceAdapter:
    """Real model adapter that calls ms-swift text inference via subprocess.

    The adapter writes a one-row ms-swift JSONL manifest for each generation
    request, invokes ``python -m swift.sidewalk_nav.cli.infer``, and returns the
    ``response`` field from the resulting prediction JSONL.
    """

    def __init__(self, config: MsSwiftInferenceConfig) -> None:
        self.config = config

    async def generate_pre_tool(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        images: list[str],
    ) -> str:
        """Generate the first assistant turn with ms-swift."""

        return await self._generate(messages, tools, images, target="tool_decision")

    async def generate_final(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        images: list[str],
    ) -> str:
        """Generate the final assistant turn with ms-swift."""

        return await self._generate(messages, tools, images, target="final_navigation")

    def build_inference_row(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        images: list[str],
        *,
        target: str,
    ) -> dict[str, Any]:
        """Build one ms-swift inference row.

        Args:
            messages: Prefix messages visible to the model.
            tools: Decoded tool definitions.
            images: Image paths.
            target: Stage2 eval target name.

        Returns:
            JSON object accepted by ``swift.sidewalk_nav.cli.infer``.
        """

        return {
            "schema_version": "sidewalk_agentic_stage2_runtime_infer_v1",
            "stage": "agentic_online_runtime",
            "task_type": "sidewalk_navigation_agent",
            "messages": list(messages),
            "tools": json.dumps(list(tools), ensure_ascii=False, sort_keys=True),
            "images": list(images),
            "sidewalk_meta": {
                "prompt_template": "agentic_stage2_cot_trajectory",
                "stage2_eval_target": target,
                "source": "routeserver_online_agent_runner",
            },
        }

    async def _generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        images: list[str],
        *,
        target: str,
    ) -> str:
        # 每次生成写一个 one-row manifest，复用 ms-swift infer.py 的模板/图像处理逻辑。
        row = self.build_inference_row(messages, tools, images, target=target)
        return await asyncio.to_thread(self._run_ms_swift_infer, row, target)

    def _run_ms_swift_infer(self, row: dict[str, Any], target: str) -> str:
        """Run the ms-swift inference subprocess for one row."""

        work_parent = Path(self.config.work_dir) if self.config.work_dir else None
        with tempfile.TemporaryDirectory(prefix=f"routeserver_stage2_{target}_", dir=work_parent) as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.jsonl"
            output_path = tmp_path / "output.jsonl"
            input_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            command = self._command(input_path=input_path, output_path=output_path)
            completed = subprocess.run(
                command,
                cwd=str(self.config.ms_swift_root),
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "ms-swift inference failed "
                    f"(target={target}, code={completed.returncode}):\n"
                    f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
                )
            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        # infer.py 的 contract 是每个输入 row 产生一条 prediction row，这里取第一条响应。
        if not rows:
            raise RuntimeError(f"ms-swift inference produced no rows for target={target}")
        response = rows[0].get("response")
        if response is None:
            raise RuntimeError(f"ms-swift inference row is missing response for target={target}: {rows[0]}")
        return str(response)

    def _command(self, *, input_path: Path, output_path: Path) -> list[str]:
        """Build the ms-swift CLI command without shell interpolation."""

        # 用 argv list 而不是 shell 字符串，避免 checkpoint/path 中特殊字符导致命令注入或转义问题。
        command = [
            self.config.python_executable,
            "-m",
            "swift.sidewalk_nav.cli.infer",
            "--model",
            self.config.model,
            "--dataset",
            str(input_path),
            "--output",
            str(output_path),
            "--phase",
            self.config.phase,
            "--mode",
            "text",
            "--max-tokens",
            str(self.config.max_tokens),
            "--temperature",
            str(self.config.temperature),
            "--torch-dtype",
            self.config.torch_dtype,
            "--prompt-template",
            "agentic_stage2_cot_trajectory",
        ]
        if self.config.adapter_dir:
            command.extend(["--adapter-dir", self.config.adapter_dir])
        if self.config.checkpoint_dir:
            command.extend(["--checkpoint-dir", self.config.checkpoint_dir])
        if self.config.agent_template:
            command.extend(["--agent-template", self.config.agent_template])
        if self.config.device_map:
            command.extend(["--device-map", self.config.device_map])
        if self.config.enable_thinking:
            command.append("--enable-thinking")
        command.extend(self.config.extra_args)
        return command


def _stage2_llm_system_prompt(target: str) -> str:
    """Return the strict system prompt for a live Stage2 LLM backend."""

    return (
        "You are the live RouteServer Stage2 inference backend. "
        "Return only the next assistant message content for the requested target. "
        "Follow the XML-like tag grammar exactly. "
        "Do not use markdown, code fences, prose outside the required tags, or provider function-calling. "
        f"Current target: {target}."
    )


def _stage2_output_contract(*, target: str, has_tool_response: bool) -> dict[str, Any]:
    """Return the JSON-serializable output contract embedded in the LLM prompt."""

    if target == "tool_decision":
        return {
            "allowed_outputs": [
                "no_tool: <think>\\n<AgentState>{json}</AgentState>\\n<ToolDecision>{\"should_call_tool\":false,...}</ToolDecision>\\n</think>, no <tool_call>",
                "tool: <think>\\n<AgentState>{json}</AgentState>\\n<ToolDecision>{\"should_call_tool\":true,\"tool\":\"...\",\"op\":\"...\",...}</ToolDecision>\\n</think>, followed by sibling <tool_call>{...}</tool_call>",
            ],
            "rules": [
                "Always emit <AgentState> and <ToolDecision> as XML-like tags; never write plain labels such as 'AgentState:' or 'ToolDecision:'.",
                "<tool_call> must be outside <think>, immediately after </think>.",
                "Do not emit <finish> in the tool_decision target.",
                "Use only student-visible tool names and ops from tools.",
                "For map_tool.search_poi, include op, query, location, and optional limit.",
                "If runtime_context.current_location is available, copy it exactly as the map_tool.search_poi location value.",
                "Never use placeholders such as 'current longitude,current latitude' or 'lon,lat'.",
                "For map_tool.create_route_session, include op, origin, destination, and mode=foot unless the user asks otherwise.",
                "If the visible route/image context is enough, set should_call_tool=false.",
            ],
            "tool_call_shape": {
                "name": "map_tool",
                "arguments": {"op": "search_poi", "query": "rest area", "location": "lon,lat", "limit": 3},
            },
        }
    return {
        "required_text_shape": (
            "<think>\n"
            "<Scene>{json}</Scene>\n"
            "<FPVScene>{json}</FPVScene>\n"
            "<BEVContext>{json}</BEVContext>\n"
            "<CrossView>{json}</CrossView>\n"
            "<Plan>{json}</Plan>\n"
            + ("<ToolResultUse>{json}</ToolResultUse>\n" if has_tool_response else "")
            + "</think>\n"
            "<finish>{json}</finish>"
        ),
        "rules": [
            "Every JSON block must be valid compact JSON.",
            "<finish> must be outside <think>.",
            "trajectory_grid1000 must contain exactly 16 [x,y] integer points in [0,1000].",
            "trajectory_coord_frame must be \"norm1000\".",
            "Do not emit legacy finish.trajectory or trajectory_norm.",
            "Use route_action=request_user_input only when user_interaction_tool asks for clarification.",
            "If a tool_response is available, include <ToolResultUse> summarizing tool_used, used_for, result_summary, route_update, and navigation_effect.",
            "<ToolResultUse>.tool_used must be true when the prefix contains a tool_response.",
        ],
        "tool_result_use_shape": {
            "tool_used": has_tool_response,
            "used_for": "final_navigation_decision",
            "result_summary": "summarize the tool_response success or failure",
            "route_update": "none",
            "navigation_effect": "state how the tool result affects the local trajectory",
        },
        "finish_shape": {
            "route_action": "continue_navigation",
            "tool_used": has_tool_response,
            "tool": "map_tool" if has_tool_response else None,
            "op": "search_poi" if has_tool_response else None,
            "trajectory_grid1000": [[500, 900], [500, 875], [500, 850], [500, 825], [500, 800], [500, 775], [500, 750], [500, 725], [500, 700], [500, 675], [500, 650], [500, 625], [500, 600], [500, 575], [500, 550], [500, 525]],
            "trajectory_coord_frame": "norm1000",
            "intent": "follow_sidewalk_route",
            "confidence_level": "medium",
            "trajectory_type": "local_navigation",
            "trajectory_validity": "valid",
            "teacher_source": "openai_compatible_live_llm",
        },
    }


def _normalize_prompt_message(message: dict[str, Any]) -> dict[str, Any]:
    """Normalize one RouteServer prefix message for text-only LLM prompting."""

    role = str(message.get("role") or "")
    content = message.get("content")
    if isinstance(content, str):
        normalized_content: Any = content
    else:
        normalized_content = json.loads(json.dumps(content, ensure_ascii=False))
    return {"role": role, "content": normalized_content}


def _stage2_runtime_context(case: AgentStage2Case | None) -> dict[str, Any]:
    """Return non-supervision runtime context for live LLM prompting."""

    if case is None:
        return {}
    observation = dict(case.observation)
    return {
        "case_id": case.case_id,
        "user_request": case.user_request,
        "current_location": observation.get("current_location") or observation.get("origin"),
        "route_session_id": observation.get("route_session_id") or observation.get("session_id"),
        "route_state": observation.get("route_state") or {},
        "observation": observation,
        "metadata": dict(case.metadata),
    }


def build_initial_model_messages(case: AgentStage2Case, *, system_prompt: str = DEFAULT_AGENT_SYSTEM_PROMPT) -> list[dict[str, Any]]:
    """Build the first model-call messages for one Stage2 case.

    Args:
        case: Stage2 input case.
        system_prompt: System prompt for the model.

    Returns:
        Two messages: system prompt and initial user request with image tokens.
    """

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_user_content(case)},
    ]


async def _maybe_await(value: str | Awaitable[str]) -> str:
    """Await a value if needed and coerce it to text."""

    if hasattr(value, "__await__"):
        value = await value
    return str(value)
