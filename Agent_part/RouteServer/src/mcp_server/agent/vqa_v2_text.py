"""Strict minimal-answer codec for UrbanNav Navigation VQA V2."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from mcp_server.contracts.vqa_v2 import (
    VQAAnswerV2,
    VQAQuestionV2,
    VQA_ANSWER_ENVELOPE_BY_TASK,
    VQA_V2_ANSWER_TAG,
    build_vqa_answer_v2,
    validate_answer_for_question_v2,
)


_ANSWER_PATTERN = re.compile(
    rf"<{VQA_V2_ANSWER_TAG}>\s*(?P<body>.*?)\s*"
    rf"</{VQA_V2_ANSWER_TAG}>",
    flags=re.DOTALL,
)


@dataclass(frozen=True)
class VQAParseResultV2:
    """Represent strict parsing of one task-specific answer.

    Args:
        answer: Typed answer when parsing succeeds.
        errors: Deterministic parse or validation errors.
    """

    answer: VQAAnswerV2 | None
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Return whether parsing produced one valid typed answer."""

        return self.answer is not None and not self.errors


def render_vqa_answer_v2(answer: VQAAnswerV2) -> str:
    """Render only the task payload inside the canonical answer tag.

    Args:
        answer: Typed annotation answer envelope.

    Returns:
        Compact assistant target without task id, provenance, or rationale.
    """

    payload = answer.payload.model_dump(mode="json")
    body = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        f"<{VQA_V2_ANSWER_TAG}>{body}"
        f"</{VQA_V2_ANSWER_TAG}>"
    )


def parse_vqa_answer_v2(
    text: str,
    *,
    expected_task_id: str,
    question: VQAQuestionV2 | None = None,
) -> VQAParseResultV2:
    """Parse exactly one minimal answer under an expected task schema.

    Args:
        text: Raw model or fixture response.
        expected_task_id: Question-owned atomic task id.
        question: Optional exact question for candidate consistency checks.

    Returns:
        Typed answer or deterministic errors.
    """

    if expected_task_id not in VQA_ANSWER_ENVELOPE_BY_TASK:
        return VQAParseResultV2(
            answer=None,
            errors=[f"unsupported_task:{expected_task_id}"],
        )
    value = str(text or "").strip()
    matches = list(_ANSWER_PATTERN.finditer(value))
    if len(matches) != 1:
        return VQAParseResultV2(
            answer=None,
            errors=[f"vqa_answer_count:{len(matches)}"],
        )
    match = matches[0]
    if value[: match.start()].strip() or value[
        match.end() :
    ].strip():
        return VQAParseResultV2(
            answer=None,
            errors=["unexpected_text_outside_vqa_answer"],
        )
    try:
        payload = json.loads(match.group("body"))
    except json.JSONDecodeError as exc:
        return VQAParseResultV2(
            answer=None,
            errors=[f"invalid_json:{exc.msg}"],
        )
    if not isinstance(payload, dict):
        return VQAParseResultV2(
            answer=None,
            errors=["answer_payload_must_be_object"],
        )
    try:
        answer = build_vqa_answer_v2(
            expected_task_id,
            payload,
        )
        if question is not None:
            validate_answer_for_question_v2(question, answer)
    except (TypeError, ValueError) as exc:
        return VQAParseResultV2(
            answer=None,
            errors=[f"answer_schema:{exc}"],
        )
    return VQAParseResultV2(answer=answer)


def answer_payload_json_schema_v2(task_id: str) -> dict[str, Any]:
    """Return the task-specific assistant payload JSON schema.

    Args:
        task_id: Exact VQA task identifier.

    Returns:
        JSON schema for the minimal payload only.

    Raises:
        ValueError: If the task id is unknown.
    """

    try:
        envelope = VQA_ANSWER_ENVELOPE_BY_TASK[task_id]
    except KeyError as exc:
        raise ValueError(f"unsupported VQA V2 task: {task_id}") from exc
    payload_field = envelope.model_fields["payload"]
    payload_type = payload_field.annotation
    if not isinstance(payload_type, type) or not issubclass(
        payload_type,
        BaseModel,
    ):
        raise TypeError(
            f"VQA answer payload is not a model: {task_id}"
        )
    return payload_type.model_json_schema()


__all__ = [
    "VQAParseResultV2",
    "answer_payload_json_schema_v2",
    "parse_vqa_answer_v2",
    "render_vqa_answer_v2",
]
