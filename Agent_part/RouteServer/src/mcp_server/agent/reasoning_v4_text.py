"""Strict tagged-text codec for canonical Reasoning V4.5."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from mcp_server.contracts.reasoning_v4 import (
    DecisionV4,
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
    """Return a typed value with bounded parser diagnostics.

    Args:
        target: Full target when full-document parsing succeeds.
        section: Standalone section when section parsing succeeds.
        errors: Hard wire, JSON, or schema failures.
        warnings: Non-fatal diagnostics.
    """

    target: ReasoningTargetV4 | None = None
    section: BaseModel | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Return whether exactly one typed value was parsed.

        Returns:
            True when no error exists and target or section is available.
        """

        return (
            not self.errors
            and (self.target is not None or self.section is not None)
        )


_THINK_RE = re.compile(
    r"\A<think>\s*(?P<body>.*?)\s*</think>\Z",
    re.DOTALL,
)
_TAG_RE = re.compile(
    r"<(?P<tag>[A-Za-z][A-Za-z0-9]*)>"
    r"\s*(?P<body>\{.*?\})\s*</(?P=tag)>",
    re.DOTALL,
)
_SECTION_MODELS: dict[str, type[BaseModel]] = {
    "TaskGrounding": TaskGroundingV4,
    "SceneGrounding": SceneGroundingV4,
    "LocalTraversability": LocalTraversabilityV4,
    "NavigationContextGrounding": NavigationContextGroundingV4,
    "InteractionConstraintGrounding": (
        InteractionConstraintGroundingV4
    ),
    "Decision": DecisionV4,
}


def render_reasoning_target_v4(target: ReasoningTargetV4) -> str:
    """Render one target in the exact six-section wire order.

    Args:
        target: Typed V4.5 reasoning target.

    Returns:
        Canonical ``<think>`` document with no Finish or free text.
    """

    trace = target.reasoning
    sections = (
        ("TaskGrounding", trace.task_grounding),
        ("SceneGrounding", trace.scene_grounding),
        ("LocalTraversability", trace.local_traversability),
        (
            "NavigationContextGrounding",
            trace.navigation_context_grounding,
        ),
        (
            "InteractionConstraintGrounding",
            trace.interaction_constraint_grounding,
        ),
        ("Decision", trace.decision),
    )
    body = "\n".join(_tag(name, value) for name, value in sections)
    return f"<think>\n{body}\n</think>"


def render_reasoning_section_v4(
    section_name: str,
    value: BaseModel,
) -> str:
    """Render one standalone section.

    Args:
        section_name: Canonical section tag.
        value: Typed section value.

    Returns:
        One tagged compact JSON object.

    Raises:
        ValueError: If the name or model type is incompatible.
    """

    expected = _SECTION_MODELS.get(section_name)
    if expected is None:
        raise ValueError(
            f"unsupported Reasoning V4 section: {section_name}"
        )
    if not isinstance(value, expected):
        raise ValueError(
            "section/type mismatch: "
            f"{section_name}/{type(value).__name__}"
        )
    return _tag(section_name, value)


def parse_reasoning_target_v4(
    text: str,
) -> ReasoningTextParseResultV4:
    """Strictly parse one full six-section V4.5 document.

    Args:
        text: Raw assistant response.

    Returns:
        Typed target or bounded errors.
    """

    value = str(text).strip()
    think = _THINK_RE.fullmatch(value)
    if think is None:
        return ReasoningTextParseResultV4(
            errors=["invalid_think_document"],
        )
    matches = list(_TAG_RE.finditer(think.group("body")))
    names = [item.group("tag") for item in matches]
    if names != list(REASONING_V4_SECTION_NAMES):
        return ReasoningTextParseResultV4(
            errors=[f"invalid_reasoning_section_order:{names}"],
        )
    if _TAG_RE.sub("", think.group("body")).strip():
        return ReasoningTextParseResultV4(
            errors=["unexpected_text_inside_think"],
        )
    try:
        sections = {
            item.group("tag"): _SECTION_MODELS[
                item.group("tag")
            ].model_validate(json.loads(item.group("body")))
            for item in matches
        }
        target = ReasoningTargetV4(
            reasoning=ReasoningTraceV4(
                task_grounding=sections["TaskGrounding"],
                scene_grounding=sections["SceneGrounding"],
                local_traversability=sections[
                    "LocalTraversability"
                ],
                navigation_context_grounding=sections[
                    "NavigationContextGrounding"
                ],
                interaction_constraint_grounding=sections[
                    "InteractionConstraintGrounding"
                ],
                decision=sections["Decision"],
            )
        )
    except (
        KeyError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        ValidationError,
    ) as exc:
        return ReasoningTextParseResultV4(
            errors=[f"reasoning_v4_schema:{exc}"],
        )
    return ReasoningTextParseResultV4(target=target)


def parse_reasoning_section_v4(
    text: str,
    section_name: str,
) -> ReasoningTextParseResultV4:
    """Strictly parse one standalone V4.5 section.

    Args:
        text: Raw tagged section.
        section_name: Expected canonical section name.

    Returns:
        Typed section or bounded errors.
    """

    model_type = _SECTION_MODELS.get(section_name)
    if model_type is None:
        return ReasoningTextParseResultV4(
            errors=[f"unsupported_section:{section_name}"],
        )
    value = str(text).strip()
    matches = list(_TAG_RE.finditer(value))
    if len(matches) != 1:
        return ReasoningTextParseResultV4(
            errors=[f"section_count:{len(matches)}"],
        )
    match = matches[0]
    if (
        match.group("tag") != section_name
        or match.start() != 0
        or match.end() != len(value)
    ):
        return ReasoningTextParseResultV4(
            errors=["invalid_section_document"],
        )
    try:
        section = model_type.model_validate(
            json.loads(match.group("body"))
        )
    except (
        json.JSONDecodeError,
        TypeError,
        ValueError,
        ValidationError,
    ) as exc:
        return ReasoningTextParseResultV4(
            errors=[f"reasoning_v4_section_schema:{exc}"],
        )
    return ReasoningTextParseResultV4(section=section)


def _tag(name: str, value: BaseModel) -> str:
    """Render one compact JSON object inside a canonical tag."""

    payload = json.dumps(
        value.model_dump(mode="json", exclude_none=True),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"<{name}>{payload}</{name}>"


__all__ = [
    "ReasoningTextParseResultV4",
    "parse_reasoning_section_v4",
    "parse_reasoning_target_v4",
    "render_reasoning_section_v4",
    "render_reasoning_target_v4",
]
