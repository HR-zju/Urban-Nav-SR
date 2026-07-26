"""Strict renderer and parser for canonical Reasoning V4 assistant text."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable

from pydantic import BaseModel, ValidationError

from mcp_server.contracts.reasoning_v4 import (
    DecisionV4,
    FinishV4,
    InteractionConstraintGroundingV4,
    LocalTraversabilityV4,
    NavigationContextGroundingV4,
    REASONING_V4_SECTION_NAMES,
    ReasoningTargetV4,
    ReasoningTraceV4,
    SceneGroundingV4,
    TaskGroundingV4,
)


@dataclass(frozen=True)
class ReasoningTextParseResultV4:
    """Return a typed target together with bounded parser diagnostics.

    Args:
        target: Parsed target when the document is valid.
        errors: Hard wire, JSON, or schema failures.
        warnings: Non-fatal parse diagnostics.
    """

    target: ReasoningTargetV4 | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Return whether a typed target was parsed without errors."""

        return self.target is not None and not self.errors


_THINK_RE = re.compile(r"<think>\s*(?P<body>.*?)\s*</think>", re.DOTALL)
_TAG_RE = re.compile(
    r"<(?P<tag>[A-Za-z][A-Za-z0-9]*)>\s*(?P<body>\{.*?\})\s*</(?P=tag)>",
    re.DOTALL,
)
_FINISH_RE = re.compile(r"<finish>\s*(?P<body>\{.*?\})\s*</finish>", re.DOTALL)
_READOUT_RE = re.compile(r"<readout>\s*(?P<body>.*?)\s*</readout>", re.DOTALL)


def render_reasoning_target_v4(
    target: ReasoningTargetV4,
    *,
    readout_tokens: Iterable[str] = (),
) -> str:
    """Render one V4 target in the fixed six-section wire order.

    Args:
        target: Typed V4 reasoning target.
        readout_tokens: Optional ordered latent System1 query anchors.

    Returns:
        Canonical tagged assistant text.

    Raises:
        ValueError: If readout is requested for a non-navigation decision.
    """

    tokens = [str(token) for token in readout_tokens if str(token)]
    if tokens and target.finish.decision_type != "navigate":
        raise ValueError("System1 readout is legal only for navigate")
    trace = target.reasoning
    sections = (
        ("TaskGrounding", trace.task_grounding),
        ("SceneGrounding", trace.scene_grounding),
        ("LocalTraversability", trace.local_traversability),
        ("NavigationContextGrounding", trace.navigation_context_grounding),
        (
            "InteractionConstraintGrounding",
            trace.interaction_constraint_grounding,
        ),
        ("Decision", trace.decision),
    )
    body = "\n".join(_tag(name, value) for name, value in sections)
    text = f"<think>\n{body}\n</think>\n{_tag('finish', target.finish)}"
    if tokens:
        text += "\n<readout>" + " ".join(tokens) + "</readout>"
    return text


def render_reasoning_section_v4(section_name: str, value: BaseModel) -> str:
    """Render one standalone V4 section-view target.

    Args:
        section_name: One canonical V4 section tag.
        value: Typed section payload.

    Returns:
        A single canonical tagged JSON object.

    Raises:
        ValueError: If the section name is not part of V4.
    """

    if section_name not in REASONING_V4_SECTION_NAMES:
        raise ValueError(f"unsupported Reasoning V4 section: {section_name}")
    return _tag(section_name, value)


def parse_reasoning_target_v4(text: str) -> ReasoningTextParseResultV4:
    """Strictly parse one full Reasoning V4 assistant document.

    Args:
        text: Raw assistant response.

    Returns:
        Typed parse result and bounded errors.
    """

    value = str(text).strip()
    errors: list[str] = []
    think_matches = list(_THINK_RE.finditer(value))
    finish_matches = list(_FINISH_RE.finditer(value))
    readout_matches = list(_READOUT_RE.finditer(value))
    if len(think_matches) != 1:
        errors.append(f"think_section_count:{len(think_matches)}")
    if len(finish_matches) != 1:
        errors.append(f"finish_section_count:{len(finish_matches)}")
    if len(readout_matches) > 1:
        errors.append(f"readout_section_count:{len(readout_matches)}")

    matches: list[re.Match[str]] = []
    if len(think_matches) == 1:
        matches = list(_TAG_RE.finditer(think_matches[0].group("body")))
        names = [item.group("tag") for item in matches]
        if names != list(REASONING_V4_SECTION_NAMES):
            errors.append(f"invalid_reasoning_section_order:{names}")
        if _TAG_RE.sub("", think_matches[0].group("body")).strip():
            errors.append("unexpected_text_inside_think")

    if len(think_matches) == 1 and len(finish_matches) == 1 and len(readout_matches) <= 1:
        blocks = [think_matches[0], finish_matches[0], *readout_matches]
        if blocks != sorted(blocks, key=lambda item: item.start()):
            errors.append("invalid_document_order")
        elif _has_unexpected_document_text(value, blocks):
            errors.append("unexpected_text_outside_document")
    if errors:
        return ReasoningTextParseResultV4(target=None, errors=errors)

    try:
        payloads = {
            item.group("tag"): json.loads(item.group("body"))
            for item in matches
        }
        trace = ReasoningTraceV4(
            task_grounding=TaskGroundingV4.model_validate(
                payloads["TaskGrounding"]
            ),
            scene_grounding=SceneGroundingV4.model_validate(
                payloads["SceneGrounding"]
            ),
            local_traversability=LocalTraversabilityV4.model_validate(
                payloads["LocalTraversability"]
            ),
            navigation_context_grounding=NavigationContextGroundingV4.model_validate(
                payloads["NavigationContextGrounding"]
            ),
            interaction_constraint_grounding=InteractionConstraintGroundingV4.model_validate(
                payloads["InteractionConstraintGrounding"]
            ),
            decision=DecisionV4.model_validate(payloads["Decision"]),
        )
        finish = FinishV4.model_validate(json.loads(finish_matches[0].group("body")))
        target = ReasoningTargetV4(reasoning=trace, finish=finish)
    except (
        KeyError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        ValidationError,
    ) as exc:
        return ReasoningTextParseResultV4(
            target=None,
            errors=[f"reasoning_v4_schema:{exc}"],
        )

    if readout_matches and target.finish.decision_type != "navigate":
        return ReasoningTextParseResultV4(
            target=None,
            errors=["readout_requires_navigate"],
        )
    return ReasoningTextParseResultV4(target=target)


def _has_unexpected_document_text(
    value: str,
    blocks: list[re.Match[str]],
) -> bool:
    """Return whether non-whitespace text exists outside canonical blocks."""

    cursor = 0
    for block in blocks:
        if value[cursor:block.start()].strip():
            return True
        cursor = block.end()
    return bool(value[cursor:].strip())


def _tag(name: str, value: BaseModel) -> str:
    """Render one compact JSON object inside a canonical tag."""

    payload = json.dumps(
        value.model_dump(mode="json", exclude_none=True),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"<{name}>{payload}</{name}>"
