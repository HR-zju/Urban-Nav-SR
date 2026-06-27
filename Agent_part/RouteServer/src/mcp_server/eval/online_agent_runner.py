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

        return {
            "trace": self.trace.to_dict(),
            "ms_swift_row": dict(self.ms_swift_row),
            "trace_validation": self.trace_validation.to_dict(),
            "row_validation": self.row_validation.to_dict(),
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
