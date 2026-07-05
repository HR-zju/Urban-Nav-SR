"""SpotAgent-style tool-use runner for UrbanNav Module 1."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
import json
from typing import Any

from mcp_server.agent.schemas import AgentTrace, ToolResponse
from mcp_server.agent.tool_parser import parse_agent_action
from mcp_server.agent.tool_schema import MAP_TOOL_NAME, get_tool_definitions
from mcp_server.eval.urbannav_replay_tool_executor import (
    ReplayUrbanNavToolExecutor,
    UrbanNavReplayToolState,
    build_prediction_from_tool_state,
)
from mcp_server.eval.urbannav_runners import AsyncUrbanNavRunner, ReplayUrbanNavBackendAdapter, _effective_instruction
from mcp_server.eval.urbannav_schema import UrbanNavCase, UrbanNavPrediction
from mcp_server.llm import (
    LLMSettings,
    apply_llm_reasoning_effort,
    close_async_llm_client,
    create_async_llm_client,
    extract_chat_response_text,
)


LLMCompleteFn = Callable[[list[dict[str, str]]], Awaitable[str]]


@dataclass(frozen=True)
class UrbanNavToolAgentConfig:
    """Configuration for the UrbanNav tool-agent runner.

    Args:
        max_turns: Maximum assistant turns, including final ``<finish>``.
        candidate_limit: Maximum replay candidates exposed per search.
        mode: Route mode recorded in route summaries.
        max_tokens: Maximum LLM output tokens per assistant turn.
        temperature: Sampling temperature for the tool-agent loop.
        fail_on_tool_error: Whether a failed tool response immediately ends
            the case as ``prediction_exception``.  Defaults to false so the
            model may observe the failure and recover in the next turn.
    """

    max_turns: int = 0
    candidate_limit: int = 10
    mode: str = "foot"
    max_tokens: int = 768
    temperature: float = 0.0
    fail_on_tool_error: bool = False


class UrbanNavToolAgentRunner(AsyncUrbanNavRunner):
    """Run UrbanNav cases through explicit model-emitted map tool calls.

    Args:
        backend: Replay backend adapter.  The runner currently supports only
            replay execution so benchmark inputs remain frozen and repeatable.
        settings: OpenAI-compatible LLM settings.
        config: Tool-agent runner config.
        llm_complete: Optional async test double.  When omitted, the runner
            calls ``settings.base_url`` through the OpenAI-compatible client.

    The runner is deliberately not a fallback system.  It never searches,
    routes, or chooses a destination unless the model emits a valid
    ``<tool_call>`` or ``<finish>`` that references prior tool ids.
    """

    def __init__(
        self,
        *,
        backend: ReplayUrbanNavBackendAdapter,
        settings: LLMSettings | None = None,
        config: UrbanNavToolAgentConfig | None = None,
        llm_complete: LLMCompleteFn | None = None,
    ) -> None:
        """Initialize the tool-agent runner.

        Args:
            backend: Replay backend adapter.
            settings: Optional LLM settings.
            config: Optional runner config.
            llm_complete: Optional async completion callable for tests.

        Returns:
            None.
        """

        self.backend = backend
        self.settings = replace(settings or LLMSettings.from_env(), response_format=False)
        self.config = config or UrbanNavToolAgentConfig()
        self.llm_complete = llm_complete
        self.executor = ReplayUrbanNavToolExecutor(
            backend=backend,
            candidate_limit=self.config.candidate_limit,
            mode=self.config.mode,
        )
        self._client: Any | None = None

    async def close(self) -> None:
        """Close the underlying LLM client.

        Args:
            None.

        Returns:
            None.
        """

        if self._client is not None:
            await close_async_llm_client(self._client)
            self._client = None

    async def run_case_async(self, case: UrbanNavCase) -> UrbanNavPrediction:
        """Run one UrbanNav case through the bounded tool loop.

        Args:
            case: Frozen benchmark case.

        Returns:
            Prediction row, or a ``prediction_exception`` row when the model
            output/tool sequence is invalid.
        """

        state = self.executor.new_state(case)
        trace = self._new_trace(case)
        if state is None:
            return self._prediction_exception(case=case, trace=trace, detail="missing_origin")

        messages = self._initial_messages(case)
        trace.add_turn("system", messages[0]["content"])
        trace.add_turn("user", messages[1]["content"])

        max_turns = self._max_turns_for_case(case)
        trace.metadata["max_turns_effective"] = max_turns
        for turn_index in range(1, max_turns + 1):
            assistant_text = await self._complete(messages)
            if not assistant_text.strip():
                return self._prediction_exception(case=case, trace=trace, detail="empty_model_response")
            trace.add_turn("assistant", assistant_text, metadata={"turn_index": turn_index})
            action = parse_agent_action(assistant_text)
            if action.action_type == "invalid":
                return self._prediction_exception(
                    case=case,
                    trace=trace,
                    detail="invalid_agent_action:" + ",".join(action.errors),
                )
            if action.action_type == "finish":
                trace.metadata["finish_turn_index"] = turn_index
                trace.metrics = self.executor.tool_metrics(state)
                diagnostic = self._trace_payload(trace=trace, state=state, finish=action.finish or {})
                return build_prediction_from_tool_state(
                    case=case,
                    state=state,
                    finish=action.finish or {},
                    trace=diagnostic,
                    executor=self.executor,
                )

            assert action.tool_call is not None
            trace.add_turn(
                "tool_call",
                action.tool_call.to_ms_swift_dict(),
                tool=action.tool_call.tool,
                metadata={"turn_index": turn_index},
            )
            response = self.executor.execute(case=case, state=state, call=action.tool_call)
            trace.add_turn(
                "tool_response",
                response.to_dict(),
                tool=response.tool,
                metadata={"turn_index": turn_index},
            )
            if not response.success and self.config.fail_on_tool_error:
                return self._prediction_exception(
                    case=case,
                    trace=trace,
                    detail=f"tool_execution_failed:{_tool_error_code(response)}",
                )
            messages.append({"role": "assistant", "content": assistant_text})
            messages.append({"role": "user", "content": _tool_response_message(response)})

        return self._prediction_exception(case=case, trace=trace, detail="max_turns_exceeded_without_finish")

    async def _complete(self, messages: list[dict[str, str]]) -> str:
        if self.llm_complete is not None:
            return await self.llm_complete(messages)
        client = self._client or create_async_llm_client(self.settings)
        self._client = client
        if client is None:
            raise RuntimeError("openai_async_client_unavailable")
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        payload = apply_llm_reasoning_effort(payload, self.settings)
        response = await client.chat.completions.create(**payload)
        return extract_chat_response_text(response)

    def _new_trace(self, case: UrbanNavCase) -> AgentTrace:
        tools = [tool for tool in get_tool_definitions() if tool.get("function", {}).get("name") == MAP_TOOL_NAME]
        return AgentTrace(
            case_id=case.case_id,
            task_type="urbannav_module1_tool_agent",
            request=case.to_dict(),
            tools=tools,
            model_run={
                "runner": "urbannav_tool_agent",
                "provider": self.settings.provider,
                "model": self.settings.model,
                "base_url": self.settings.base_url,
            },
            metadata={
                "max_turns": self.config.max_turns,
                "candidate_limit": self.config.candidate_limit,
                "mode": self.config.mode,
            },
        )

    def _initial_messages(self, case: UrbanNavCase) -> list[dict[str, str]]:
        instruction = _effective_instruction(case)
        payload = {
            "case_id": case.case_id,
            "instruction": instruction,
            "category": case.category,
            "subcategory": case.subcategory,
            "city": case.city,
            "scene_id": case.scene_id,
            "origin": case.origin.to_dict() if case.origin else None,
            "metadata": _public_case_metadata(case),
            "tool_policy": _tool_policy_for_case(case),
        }
        return [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": "UrbanNav case JSON:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ]

    def _trace_payload(
        self,
        *,
        trace: AgentTrace,
        state: UrbanNavReplayToolState,
        finish: dict[str, Any],
    ) -> dict[str, Any]:
        trace.metrics = self.executor.tool_metrics(state)
        return {
            "runner": "urbannav_tool_agent",
            "agent_trace": trace.to_dict(),
            "tool_metrics": self.executor.tool_metrics(state),
            "finish": dict(finish),
        }

    def _prediction_exception(self, *, case: UrbanNavCase, trace: AgentTrace, detail: str) -> UrbanNavPrediction:
        return UrbanNavPrediction(
            case_id=case.case_id,
            trace={
                "runner": "urbannav_tool_agent",
                "case_id": case.case_id,
                "instruction": case.instruction,
                "category": case.category,
                "subcategory": case.subcategory,
                "city": case.city,
                "error": "prediction_exception",
                "detail": detail,
                "agent_trace": trace.to_dict(),
            },
        )

    def _max_turns_for_case(self, case: UrbanNavCase) -> int:
        configured = int(self.config.max_turns or 0)
        if configured > 0:
            return max(1, configured)
        category = str(case.category or "").strip()
        if category == "multi_stop_route":
            goal_count = _goal_count(case)
            return 10 if goal_count >= 3 else 8
        if category.startswith("route_update") or _public_case_metadata(case).get("route_update"):
            return 8
        return 6


def _system_prompt() -> str:
    """Return the fixed UrbanNav tool-agent system prompt.

    Args:
        None.

    Returns:
        Prompt text.
    """

    return (
        "You are an UrbanNav tool-use agent. You must solve the navigation request by emitting explicit "
        "map_tool calls and then a final finish JSON.\n"
        "Hard output contract:\n"
        "- The entire assistant message must be exactly one action block and nothing else.\n"
        "- The only valid action blocks are <tool_call>{...}</tool_call> and <finish>{...}</finish>.\n"
        "- Do not output <think>, reasoning, explanations, Markdown fences, comments, or any text outside the action block.\n"
        "- The JSON inside the action block must be valid JSON with double quotes and no trailing commas.\n"
        "- The tool_call JSON shape is exactly {\"name\":\"map_tool\",\"arguments\":{\"op\":\"...\", ...}}.\n"
        "Allowed map_tool operations: search_destination, plan_route, create_route_session, get_route_session, add_via, replace_destination, replan_from_current_pose.\n"
        "ID and field rules:\n"
        "- tool_response is the only source of truth. Use only ids returned in previous tool_response messages.\n"
        "- Copy ids exactly. Prefer full candidate_id values such as search_1_poi_1.\n"
        "- Never invent, shorten, rewrite, or guess candidate ids, destination ids, route ids, session ids, or coordinates.\n"
        "- In tool calls, use canonical fields only: destination_id, route_id, session_id, stop_index, current_pose, query, origin, mode.\n"
        "- Alias fields are accepted by the executor only as a fallback; do not prefer aliases in your output.\n"
        "Candidate selection rules:\n"
        "- After search_destination returns candidates, select exactly one returned candidate before planning.\n"
        "- Prefer explicit user constraints. If candidates are equally valid, choose the top-ranked returned candidate.\n"
        "- Do not finish immediately after search_destination. Finish only after a tool_response has returned route_id.\n"
        "Workflow rules:\n"
        "- Follow the case JSON tool_policy exactly. It defines task_mode, required_sequence, recommended_update_op, and finish_shape.\n"
        "- For single_destination tasks: call search_destination, then plan_route with destination_id, then finish.\n"
        "- For multi_stop_route tasks: preserve the ordered goals. For each stop i, call search_destination with stop_index=i, then plan_route with destination_id and stop_index=i. Do not use add_via, replace_destination, or replan_from_current_pose.\n"
        "- For route_update tasks: first call get_route_session. Then follow tool_policy.recommended_update_op exactly. If the update needs a new place, call search_destination before add_via or replace_destination. Do not include stop_index in route-update tool calls.\n"
        "- Do not call create_route_session unless tool_policy explicitly requires it or a tool_response route/session state requires it.\n"
        "Finish rules:\n"
        "- Single-destination finish JSON: {\"chosen_destination_id\":\"search_1_poi_1\",\"route_id\":\"route_1\"}.\n"
        "- Multi-stop finish JSON: {\"destination_sequence_ids\":[\"search_1_poi_1\",\"search_2_poi_1\"],\"route_id\":\"route_2\"}.\n"
        "- Route-update finish JSON follows tool_policy.finish_shape and must include the updated route_id.\n"
    )


def _public_case_metadata(case: UrbanNavCase) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    route_update = case.metadata.get("route_update")
    if isinstance(route_update, dict):
        metadata["route_update"] = route_update
    target_types = case.metadata.get("target_types")
    if isinstance(target_types, list):
        metadata["target_types"] = target_types
    return metadata


def _tool_policy_for_case(case: UrbanNavCase) -> dict[str, Any]:
    category = str(case.category or "").strip()
    if category == "multi_stop_route":
        goal_count = _goal_count(case)
        return {
            "task_mode": "ordered_multi_stop",
            "goal_count": goal_count,
            "required_sequence": [
                "search_destination(stop_index=i)",
                "plan_route(destination_id=..., stop_index=i)",
            ],
            "required_stop_index": list(range(1, goal_count + 1)) if goal_count else [],
            "forbidden_update_ops": ["add_via", "replace_destination", "replan_from_current_pose"],
            "finish_shape": {
                "destination_sequence_ids": ["search_1_poi_1", "search_2_poi_1"],
                "route_id": "last route_id",
            },
        }
    update = _route_update_metadata_from_case(case)
    if update:
        update_type = str(update.get("update_type") or "").strip()
        recommended_op = _recommended_route_update_op(update_type)
        return {
            "task_mode": "route_update",
            "update_type": update_type,
            "recommended_update_op": recommended_op,
            "required_sequence": ["get_route_session", "search_destination(update target)", recommended_op],
            "op_rule": _route_update_op_rule(update_type, recommended_op),
            "finish_shape": {"chosen_destination_id": "updated candidate id", "route_id": "updated route_id"},
        }
    return {
        "task_mode": "single_destination",
        "required_sequence": ["search_destination", "plan_route"],
        "finish_shape": {"chosen_destination_id": "candidate id", "route_id": "route_id"},
    }


def _goal_count(case: UrbanNavCase) -> int:
    metadata = case.metadata if isinstance(case.metadata, dict) else {}
    value = metadata.get("goal_count")
    try:
        count = int(value)
        if count > 0:
            return count
    except Exception:
        pass
    ordered = metadata.get("ordered_target_types")
    if isinstance(ordered, list) and ordered:
        return len(ordered)
    return 2


def _route_update_metadata_from_case(case: UrbanNavCase) -> dict[str, Any]:
    metadata = case.metadata if isinstance(case.metadata, dict) else {}
    update = metadata.get("route_update")
    return dict(update) if isinstance(update, dict) else {}


def _recommended_route_update_op(update_type: str) -> str:
    if update_type == "replace_destination":
        return "replace_destination"
    if update_type in {"add_via", "insert_before_final"}:
        return "add_via"
    return "replan_from_current_pose"


def _route_update_op_rule(update_type: str, recommended_op: str) -> str:
    if recommended_op == "replace_destination":
        return "Use replace_destination only for the update operation. Do not use add_via."
    if recommended_op == "add_via":
        return "Use add_via only for the update operation, then finish with the updated route_id."
    return "Use replan_from_current_pose only; keep the destination unchanged."


def _tool_response_message(response: ToolResponse) -> str:
    return "Tool response JSON:\n" + json.dumps(response.to_dict(), ensure_ascii=False, sort_keys=True)


def _tool_error_code(response: ToolResponse) -> str:
    error = response.error if isinstance(response.error, dict) else {}
    return str(error.get("code") or "unknown")
