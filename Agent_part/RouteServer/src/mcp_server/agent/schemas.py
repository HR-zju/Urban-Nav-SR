"""Stable Agentic Stage2 data schemas.

这些 dataclass 是 RouteServer teacher、rule baseline、外部 LVLM 和本项目
E2E LVLM checkpoint 之间的公共协议。Stage2 直接使用最终 agentic cold-start
格式：工具调用是独立 ``tool_call`` role 工具返回是独立 ``tool_response`` role
最终导航 readout prompt 在工具返回之后插入 ``[TRAJ_*]`` token。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


AGENT_STAGE2_CASE_SCHEMA_VERSION = "sidewalk_agent_stage2_case_v1"
AGENT_EVAL_CASE_SCHEMA_VERSION = AGENT_STAGE2_CASE_SCHEMA_VERSION
AGENT_STAGE2_TRACE_SCHEMA_VERSION = "sidewalk_agent_trace_v2"
AGENT_TRACE_SCHEMA_VERSION = AGENT_STAGE2_TRACE_SCHEMA_VERSION
MS_SWIFT_AGENTIC_STAGE2_SCHEMA_VERSION = "sidewalk_agentic_stage2_sft_v1"


def _clean_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop empty optional fields while preserving valid false/zero values."""

    # 注意不能用 truthiness 过滤：False/0 是合法协议值，例如 tool_response.success=false。
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
        ``to_dict()`` 返回稳定 JSON 供 executor、metrics、exporter 共用。
        ``to_ms_swift_dict()`` 返回 ``name/arguments`` 形态，供 ``tool_call`` role
        直接序列化为 JSON 字符串。
    """

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ToolCall":
        # 兼容 RouteServer 内部 tool 字段和 ms-swift/openai 风格 name 字段。
        tool = str(payload.get("tool") or payload.get("name") or "").strip()
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict):
            # 旧样本可能把 op/query 等参数平铺在根对象，这里收束回 arguments。
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
        原始文本或 JSON。Stage2 中 ``tool_call`` content 是 JSON object/string
        形如 ``{"name": "...", "arguments": {...}}`` assistant content 保留
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
    """Backward-compatible offline evaluation/training case container.

    新代码应优先使用 ``AgentStage2Case``。保留本类是为了不破坏已有
    ``AgentEvalCase`` import 和早期 eval helpers 字段与 Stage2 case 完全一致。

    JSONL 行格式:
        ``schema_version`` 固定为 ``sidewalk_agent_stage2_case_v1``。
        ``observation.images`` 可为 ``{"fpv": "...", "bev": "..."}`` 或图片路径列表。
        ``ground_truth.expected_tool_ops`` 可选，用于 tool-op 指标 Phase1 的
        ``pixel_trajectory``、``trajectory_fpv_t``、``semantic_segmentation_path`` 等
        监督字段继续放在 ``ground_truth`` 中并由 exporter 透传。

    该结构不包含新 teacher LVLM 输出 teacher 只在 runner 阶段生成 AgentTrace。
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
            # 固定 FPV/front 优先、BEV 次之，保证 <image> token 顺序稳定。
            ordered = [images.get("fpv"), images.get("front"), images.get("bev")]
            ordered.extend(value for key, value in images.items() if key not in {"fpv", "front", "bev"})
            return [str(item) for item in ordered if item]
        return []

    @property
    def image_roles(self) -> list[str]:
        """Return one explicit modality role for every ordered image.

        Returns:
            Roles aligned with :attr:`images`. List-based inputs must declare
            ``observation.image_roles``; otherwise they are conservatively
            treated as FPV frames instead of guessing that image two is BEV.
        """

        images = self.observation.get("images")
        declared = self.observation.get("image_roles")
        if isinstance(images, list):
            if isinstance(declared, list) and len(declared) == len(self.images):
                return [str(item) for item in declared]
            return ["fpv" for _ in self.images]
        if isinstance(images, dict):
            ordered_keys = [key for key in ("fpv", "front", "bev") if images.get(key)]
            ordered_keys.extend(key for key, value in images.items() if key not in {"fpv", "front", "bev"} and value)
            return [
                "fpv" if key in {"fpv", "front"} else "goal_image" if key in {"goal", "goal_image"} else key
                for key in ordered_keys
            ]
        return []


@dataclass
class AgentStage2Case(AgentEvalCase):
    """Canonical Stage2 case used by Agentic Cold Start data generation.

    输入:
        一个 JSON object 通常来自 ``sidewalk_agent_stage2_cases_review.jsonl``
        或 Phase1 archive 合并后的 Stage2 source。核心字段包括
        ``case_id``、``user_request``、``observation``、``ground_truth``。

    输出:
        ``to_dict()`` 返回 JSONL 可写入对象；``images`` 返回按 FPV、BEV 优先排序
        的图片路径列表。该类不执行 teacher 或 tool 只承载可复现输入和监督字段。

    重要约束:
        ``ground_truth.pixel_trajectory`` / ``trajectory_fpv_t`` /
        ``semantic_segmentation_path`` 等监督字段不进入 LM 文本；它们由 exporter
        透传给 ms-swift SideWalk trainer 的额外 loss。
    """

    schema_version: str = AGENT_STAGE2_CASE_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentStage2Case":
        nav_observation = payload.get("nav_observation") or payload.get("observation_contract")
        observation = dict(payload.get("observation") or {})
        metadata = dict(payload.get("metadata") or {})
        if isinstance(nav_observation, dict):
            # 新 contract 输入先折回旧 observation 字段，保证旧 builder/exporter 不需要
            # 立刻重写；完整 NavObservation 原文放 metadata 便于审计。
            observation = _observation_from_nav_contract(nav_observation, observation)
            metadata.setdefault("nav_observation", nav_observation)
        return cls(
            case_id=str(payload.get("case_id") or payload.get("id") or ""),
            user_request=str(payload.get("user_request") or payload.get("instruction") or ""),
            task_type=str(payload.get("task_type") or "sidewalk_navigation_agent"),
            schema_version=str(payload.get("schema_version") or AGENT_STAGE2_CASE_SCHEMA_VERSION),
            observation=observation,
            dialogue_history=list(payload.get("dialogue_history") or payload.get("messages") or []),
            tools=list(payload.get("tools") or []),
            ground_truth=dict(payload.get("ground_truth") or {}),
            split=str(payload.get("split") or "eval"),
            locale=str(payload.get("locale") or "zh-CN"),
            metadata=metadata,
        )

    def to_nav_observation(self):
        """Return the normalized runtime-visible observation contract.

        Returns:
            ``mcp_server.agent.contracts.NavObservation`` view over this case.
            Supervision fields stay in ``ground_truth`` and are not copied into
            the model-visible observation.
        """

        from mcp_server.agent.contracts import NavObservation

        return NavObservation.from_case(self)

    @classmethod
    def from_nav_observation(
        cls,
        nav_observation: dict[str, Any],
        *,
        ground_truth: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        task_type: str = "sidewalk_navigation_agent",
        split: str = "eval",
        locale: str = "zh-CN",
    ) -> "AgentStage2Case":
        """Build a Stage2 case from a ``NavObservation`` payload.

        Args:
            nav_observation: Runtime-visible observation contract.
            ground_truth: Optional supervision bundle.
            tools: Optional student-visible tools; if omitted, available tools
                are read from ``nav_observation.tool_context.available_tools``.
            task_type: Task type written to the case.
            split: Dataset split.
            locale: Locale code.

        Returns:
            Backward-compatible ``AgentStage2Case``.
        """

        request = nav_observation.get("request") if isinstance(nav_observation.get("request"), dict) else {}
        tool_context = nav_observation.get("tool_context") if isinstance(nav_observation.get("tool_context"), dict) else {}
        return cls(
            case_id=str(
                nav_observation.get("observation_id")
                or (nav_observation.get("runtime_meta") or {}).get("case_id")
                or ""
            ),
            user_request=str(request.get("latest_user_utterance") or request.get("active_instruction") or ""),
            task_type=task_type,
            observation=_observation_from_nav_contract(nav_observation, {}),
            dialogue_history=list(request.get("dialogue_history") or []),
            tools=list(tools if tools is not None else tool_context.get("available_tools") or []),
            ground_truth=dict(ground_truth or {}),
            split=split,
            locale=locale,
            metadata={"nav_observation": dict(nav_observation)},
        )


@dataclass
class AgentTrace:
    """Replayable Agentic Stage2 trajectory generated from one ``AgentStage2Case``.

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
    def from_case(cls, case: AgentStage2Case, *, model_run: dict[str, Any] | None = None) -> "AgentTrace":
        # trace 继承 case 的工具 schema 和监督字段；runner 只负责追加对话 turn。
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
        # role/content 不在这里强校验，validator 统一负责 hard error 和 review warning。
        self.turns.append(AgentTurn(role=role, content=content, tool=tool, metadata=dict(metadata or {})))

    def to_dict(self) -> dict[str, Any]:
        # to_dict 是所有 review/export 的共同入口，保持字段稳定比压缩体积更重要。
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

    def to_agent_step_result(self, *, case: AgentStage2Case | None = None, validation: dict[str, Any] | None = None):
        """Return the unified one-step runtime result contract.

        Args:
            case: Optional source Stage2 case.  When omitted, the trace request
                metadata is used if possible.
            validation: Optional validator result dictionary.

        Returns:
            ``mcp_server.agent.contracts.AgentStepResult``.
        """

        from mcp_server.agent.contracts import AgentStepResult

        return AgentStepResult.from_trace(self, case=case, validation=validation)


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


def _observation_from_nav_contract(nav_observation: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    """Fold ``NavObservation`` into legacy ``AgentStage2Case.observation``.

    Args:
        nav_observation: Observation contract payload.
        base: Existing legacy observation fields.

    Returns:
        Observation dict preserving legacy accessors while keeping the full
        contract under ``nav_observation``.
    """

    observation = dict(base)
    observation["nav_observation"] = dict(nav_observation)
    request = nav_observation.get("request") if isinstance(nav_observation.get("request"), dict) else {}
    sensors = nav_observation.get("sensors") if isinstance(nav_observation.get("sensors"), dict) else {}
    route_state = nav_observation.get("route_state") if isinstance(nav_observation.get("route_state"), dict) else {}
    dialogue_state = nav_observation.get("dialogue_state") if isinstance(nav_observation.get("dialogue_state"), dict) else {}
    tool_context = nav_observation.get("tool_context") if isinstance(nav_observation.get("tool_context"), dict) else {}

    if request.get("active_instruction") is not None:
        observation.setdefault("active_instruction", request.get("active_instruction"))
    if request.get("latest_user_utterance") is not None:
        observation.setdefault("latest_user_utterance", request.get("latest_user_utterance"))
    images = sensors.get("images")
    if images is not None:
        observation.setdefault("images", images)
    elif sensors.get("fpv_images") or sensors.get("bev_images"):
        image_dict: dict[str, Any] = {}
        if sensors.get("fpv_images"):
            image_dict["fpv"] = list(sensors["fpv_images"])[-1]
        if sensors.get("bev_images"):
            image_dict["bev"] = list(sensors["bev_images"])[-1]
        observation.setdefault("images", image_dict)
    for key, value in route_state.items():
        observation.setdefault(key, value)
    if dialogue_state:
        observation.setdefault("dialogue_state", dialogue_state)
    if tool_context:
        observation.setdefault("tool_context", tool_context)
    return observation
