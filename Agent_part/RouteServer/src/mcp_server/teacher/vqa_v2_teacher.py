"""Visible-input prompt construction and parsing for VQA V2 teachers."""

from __future__ import annotations

import json
import re

from mcp_server.agent.vqa_v2_text import parse_vqa_answer_v2
from mcp_server.contracts.vqa_dataset_v2 import VQASourceV2
from mcp_server.contracts.vqa_v2 import VQAAnswerV2, VQAQuestionV2
from mcp_server.teacher.vqa_v2_profiles import VQATeacherProfileV2


def build_vqa_teacher_question_v2(
    source: VQASourceV2,
    question: VQAQuestionV2,
    profile: VQATeacherProfileV2,
    *,
    previous_answer: str | None = None,
    previous_errors: list[str] | None = None,
) -> str:
    """Build a weak-label request from student-visible input only.

    Args:
        source: Frozen VQA source.
        question: Exact answer-independent question.
        profile: Matching task-specific teacher profile.
        previous_answer: Optional invalid response for repair.
        previous_errors: Optional deterministic parser errors.

    Returns:
        Canonical provider user prompt.

    Raises:
        ValueError: If the question and profile task ids differ.
    """

    if profile.task_id != question.task_id:
        raise ValueError("VQA teacher profile/question task mismatch")
    observation = source.observation
    visible_input = {
        "source_group_id": observation.source_group_id,
        "image_order": [
            {"ordinal": index + 1, "role": role}
            for index, role in enumerate(observation.image_roles)
        ],
        "instruction": observation.instruction,
        "travel_mode": observation.travel_mode,
        "model_visible_context": observation.model_visible_context,
        "input_modalities": observation.input_modalities.model_dump(
            mode="json"
        ),
        "temporal_evidence_eligible": (
            observation.temporal_evidence_eligible
        ),
        "evidence_catalog": [
            item.model_dump(mode="json", exclude_none=True)
            for item in observation.evidence_catalog
        ],
    }
    payload = {
        "question": question.model_dump(mode="json"),
        "complete_visible_input": visible_input,
        "answer_payload_json_schema": profile.output_json_schema,
    }
    parts = [
        "Answer the requested atomic Navigation VQA question. The "
        "ordered images are attached in image_order.",
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    ]
    if previous_answer is not None or previous_errors:
        parts.extend(
            [
                "The previous answer was invalid. Repair only the answer "
                "payload using these diagnostics.",
                json.dumps(
                    {
                        "previous_answer": str(
                            previous_answer or ""
                        )[:8000],
                        "errors": [
                            str(item)[:500]
                            for item in (previous_errors or [])[:16]
                        ],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            ]
        )
    return "\n\n".join(parts)


def parse_vqa_teacher_answer_v2(
    text: str,
    *,
    question: VQAQuestionV2,
) -> tuple[VQAAnswerV2 | None, list[str]]:
    """Parse tagged output or one whole-response JSON object.

    Args:
        text: Raw provider response.
        question: Exact question that determines the answer schema.

    Returns:
        Typed answer and deterministic errors.
    """

    value = str(text or "").strip()
    if re.fullmatch(
        r"```(?:json)?\s*.*?\s*```",
        value,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        value = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            value,
            flags=re.IGNORECASE,
        ).strip()
    spaced_wrapper = re.fullmatch(
        r"<\s*VQAAnswer\s*>(?P<body>.*?)"
        r"<\s*/\s*VQAAnswer\s*>",
        value,
        flags=re.DOTALL,
    )
    if spaced_wrapper is not None:
        value = (
            "<VQAAnswer>"
            + spaced_wrapper.group("body").strip()
            + "</VQAAnswer>"
        )
    if not value.startswith("<VQAAnswer>"):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            value = (
                "<VQAAnswer>"
                + json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "</VQAAnswer>"
            )
    parsed = parse_vqa_answer_v2(
        value,
        expected_task_id=question.task_id,
        question=question,
    )
    return parsed.answer, list(parsed.errors)


__all__ = [
    "build_vqa_teacher_question_v2",
    "parse_vqa_teacher_answer_v2",
]
