"""Deterministic terminal-decision validation for navigation episodes."""

from __future__ import annotations

from typing import Any

from mcp_server.runtime.interfaces import (
    System2Action,
    System2ActionType,
    TerminalValidationResult,
)


class RuntimeTerminalValidator:
    r"""Validate completion from authoritative state and accept typed aborts.

    Successful completion follows

    \[
    S_t=[a_t^{S2}=\texttt{complete\_task}]\land
    \operatorname{ArrivalVerifier}(o_t, r_t).
    \]

    Model-declared ``evidence_refs`` are audit metadata only. The default
    verifier accepts explicit boolean arrival/goal/route-completion fields from
    environment metadata or successful typed tool responses.
    """

    async def validate(self, action: System2Action, state: dict[str, Any]) -> TerminalValidationResult:
        """Validate one terminal request.

        Args:
            action: Canonical ``complete_task`` or ``abort_task`` action.
            state: Current event-sourced graph state.

        Returns:
            Deterministic validation result.

        Raises:
            ValueError: If a non-terminal action is passed.
        """

        value = System2Action.model_validate(action)
        if value.terminal is None or value.action_type not in {
            System2ActionType.COMPLETE_TASK,
            System2ActionType.ABORT_TASK,
        }:
            raise ValueError("terminal validator requires complete_task or abort_task")
        if value.action_type == System2ActionType.ABORT_TASK:
            return TerminalValidationResult(
                action_type=value.action_type,
                accepted=True,
                reason_code=value.terminal.reason_code,
                evidence_refs=[],
                metadata={
                    "claimed_evidence_refs": list(value.terminal.evidence_refs),
                    "retryable": value.terminal.retryable,
                },
            )

        evidence_refs, checked = _arrival_evidence(state)
        return TerminalValidationResult(
            action_type=value.action_type,
            accepted=bool(evidence_refs),
            reason_code="arrival_confirmed" if evidence_refs else "arrival_not_verified",
            evidence_refs=evidence_refs,
            metadata={
                "claimed_evidence_refs": list(value.terminal.evidence_refs),
                "checked_evidence": checked,
            },
        )


def _arrival_evidence(state: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Collect explicit true arrival indicators from trusted runtime fields.

    Args:
        state: Current graph state.

    Returns:
        ``(positive_refs, checked_refs)``. Only values that are exactly
        ``True`` become positive completion evidence.
    """

    sources: list[tuple[str, dict[str, Any]]] = []
    observation_info = state.get("observation_info")
    if isinstance(observation_info, dict):
        sources.append(("observation_info", observation_info))
    environment_result = state.get("environment_result")
    if isinstance(environment_result, dict) and isinstance(environment_result.get("info"), dict):
        sources.append(("environment_result.info", environment_result["info"]))
    tool_response = state.get("last_tool_response")
    if (
        isinstance(tool_response, dict)
        and tool_response.get("success") is True
        and isinstance(tool_response.get("result"), dict)
    ):
        sources.append(("tool_response.result", tool_response["result"]))

    positive: list[str] = []
    checked: list[str] = []
    for source_name, payload in sources:
        for key in ("arrival_success", "goal_reached", "route_completed"):
            ref = f"{source_name}:{key}"
            checked.append(ref)
            if payload.get(key) is True:
                positive.append(ref)
    return positive, checked
