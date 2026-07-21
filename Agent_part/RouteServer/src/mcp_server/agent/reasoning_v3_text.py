"""Deterministic renderer and parser for canonical Reasoning V3.3 text."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from pydantic import ValidationError

from mcp_server.contracts.reasoning_v3 import (
    REASONING_V3_SECTION_NAMES,
    DecisionV3,
    FinishV3,
    InteractionAndRuleGroundingV3,
    LocalTraversabilityV3,
    NavigationContextGroundingV3,
    ReasoningTargetV3,
    ReasoningTraceV3,
    SceneGroundingV3,
    TaskGroundingV3,
)


_THINK_RE = re.compile(r"<think>\s*(?P<body>.*?)\s*</think>", re.DOTALL)
_TAG_RE = re.compile(r"<(?P<tag>[A-Za-z][A-Za-z0-9]*)>\s*(?P<body>\{.*?\})\s*</(?P=tag)>", re.DOTALL)
_READOUT_RE = re.compile(r"<readout>\s*(?P<body>.*?)\s*</readout>", re.DOTALL)


@dataclass(frozen=True)
class ReasoningTextParseResultV3:
    """Structured result of parsing one V3.3 assistant target.

    Args:
        target: Parsed typed Full-CoT target, if valid.
        errors: Hard parser or schema failures.
        warnings: Non-fatal parser findings.
        tool_result_use: Rejected legacy extension retained only for diagnostics.
    """

    target: ReasoningTargetV3 | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tool_result_use: dict[str, Any] | None = None

    @property
    def valid(self) -> bool:
        """Return whether parsing produced a typed target.

        Returns:
            ``True`` when a target exists and no errors were recorded.
        """

        return self.target is not None and not self.errors


def render_reasoning_target_v3(
    target: ReasoningTargetV3,
    *,
    readout_tokens: Iterable[str] = (),
    tool_result_use: dict[str, Any] | None = None,
) -> str:
    """Render one V3.3 target using fixed section order and compact JSON.

    Args:
        target: Validated canonical Full-CoT target.
        readout_tokens: Optional ordered latent System1 query anchors.
        tool_result_use: Unsupported legacy extension; any value is rejected.

    Returns:
        Strict tagged Full-CoT assistant text.

    Raises:
        ValueError: If an untyped tool extension is provided or readout tokens
            are attached to a non-navigation finish.
    """

    if tool_result_use is not None:
        raise ValueError("ToolResultUse is disabled until a typed Reasoning V3 contract is implemented")
    tokens = [str(token) for token in readout_tokens if str(token)]
    if tokens and target.finish.decision_type != "navigate":
        raise ValueError("System1 readout is legal only for decision_type=navigate")

    reasoning = target.reasoning
    sections = [
        _render_tag("TaskGrounding", reasoning.task_grounding.model_dump(mode="json", exclude_none=True)),
        _render_tag("SceneGrounding", reasoning.scene_grounding.model_dump(mode="json", exclude_none=True)),
        _render_tag("LocalTraversability", reasoning.local_traversability.model_dump(mode="json", exclude_none=True)),
    ]
    sections.extend(
        [
            _render_tag(
                "NavigationContextGrounding",
                reasoning.navigation_context_grounding.model_dump(mode="json", exclude_none=True),
            ),
            _render_tag(
                "InteractionAndRuleGrounding",
                reasoning.interaction_and_rule_grounding.model_dump(mode="json", exclude_none=True),
            ),
            _render_tag("Decision", reasoning.decision.model_dump(mode="json", exclude_none=True)),
        ]
    )
    text = "<think>\n" + "\n".join(sections) + "\n</think>\n"
    text += _render_tag("finish", target.finish.model_dump(mode="json", exclude_none=True))
    if tokens:
        text += "\n<readout>" + " ".join(tokens) + "</readout>"
    return text


def parse_reasoning_target_v3(text: str) -> ReasoningTextParseResultV3:
    """Parse strict V3.3 Full-CoT text without accepting legacy sections.

    Args:
        text: Raw assistant text containing six ordered sections and finish.

    Returns:
        Typed parse result with bounded machine-readable errors.
    """

    think_match = _THINK_RE.search(text)
    if think_match is None:
        return ReasoningTextParseResultV3(target=None, errors=["missing_think"])

    errors: list[str] = []
    raw_sections = list(_TAG_RE.finditer(think_match.group("body")))
    tag_names = [match.group("tag") for match in raw_sections]
    names = [name for name in tag_names if name != "ToolResultUse"]
    if names != list(REASONING_V3_SECTION_NAMES):
        errors.append(f"reasoning_section_order:{names}")
    tool_matches = [match for match in raw_sections if match.group("tag") == "ToolResultUse"]
    if tool_matches:
        errors.append("tool_result_use_not_supported_in_full_cot_v3_3")
    if _TAG_RE.sub("", think_match.group("body")).strip():
        errors.append("unexpected_text_inside_think")

    payloads: dict[str, dict[str, Any]] = {}
    tool_result_use: dict[str, Any] | None = None
    for match in raw_sections:
        payload, payload_errors = _parse_json_object(match.group("body"))
        errors.extend(f"{match.group('tag')}:{item}" for item in payload_errors)
        if payload is None:
            continue
        if match.group("tag") == "ToolResultUse":
            tool_result_use = payload
        else:
            payloads[match.group("tag")] = payload

    finish_payload, finish_errors = _tag_payload(text, "finish")
    errors.extend(f"finish:{item}" for item in finish_errors)
    readout_match = _READOUT_RE.search(text)
    if readout_match and finish_payload and finish_payload.get("decision_type") != "navigate":
        errors.append("non_navigation_has_system1_readout")
    if errors:
        return ReasoningTextParseResultV3(target=None, errors=errors, tool_result_use=tool_result_use)

    try:
        trace = ReasoningTraceV3(
            task_grounding=TaskGroundingV3.model_validate(payloads["TaskGrounding"]),
            scene_grounding=SceneGroundingV3.model_validate(payloads["SceneGrounding"]),
            local_traversability=LocalTraversabilityV3.model_validate(payloads["LocalTraversability"]),
            navigation_context_grounding=NavigationContextGroundingV3.model_validate(
                payloads["NavigationContextGrounding"]
            ),
            interaction_and_rule_grounding=InteractionAndRuleGroundingV3.model_validate(
                payloads["InteractionAndRuleGrounding"]
            ),
            decision=DecisionV3.model_validate(payloads["Decision"]),
        )
        target = ReasoningTargetV3(reasoning=trace, finish=FinishV3.model_validate(finish_payload))
    except (KeyError, ValidationError, ValueError) as exc:
        return ReasoningTextParseResultV3(
            target=None,
            errors=[f"reasoning_schema:{_validation_error_text(exc)}"],
            tool_result_use=tool_result_use,
        )
    return ReasoningTextParseResultV3(target=target, tool_result_use=tool_result_use)


def _render_tag(tag: str, payload: dict[str, Any]) -> str:
    return f"<{tag}>" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + f"</{tag}>"


def _tag_payload(text: str, tag: str) -> tuple[dict[str, Any] | None, list[str]]:
    pattern = re.compile(rf"<{re.escape(tag)}>\s*(?P<body>\{{.*?\}})\s*</{re.escape(tag)}>", re.DOTALL)
    matches = list(pattern.finditer(text))
    if not matches:
        return None, ["missing"]
    if len(matches) > 1:
        return None, ["duplicate"]
    return _parse_json_object(matches[0].group("body"))


def _parse_json_object(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, [f"invalid_json:{exc.msg}"]
    if not isinstance(payload, dict):
        return None, ["not_object"]
    return payload, []


def _validation_error_text(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return ";".join(
            f"{'.'.join(str(part) for part in item['loc'])}:{item['msg']}"
            for item in exc.errors()
        )
    return str(exc)


__all__ = [
    "ReasoningTextParseResultV3",
    "parse_reasoning_target_v3",
    "render_reasoning_target_v3",
]
