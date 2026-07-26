"""Canonical positive student prompt policy for Navigation VQA V2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from mcp_server.agent.vqa_v2_text import (
    answer_payload_json_schema_v2,
)
from mcp_server.contracts.reasoning_dataset_v4 import NavObservationV4
from mcp_server.contracts.vqa_v2 import (
    VQAQuestionV2,
    VQA_V2_ANSWER_TAG,
    VQA_V2_CONTRACT_SCHEMA_DIGEST,
    VQA_V2_SCHEMA_VERSION,
)


VQA_V2_STUDENT_PROMPT_PROFILE_ID = (
    "urbannav_navigation_vqa_student_v2_0"
)

VQA_V2_STUDENT_SYSTEM_PROMPT = """\
You are the local visual-question answering component of an assistive urban
navigation model. Answer the one atomic question from the complete
prompt-visible observation. Use FPV for immediate visible safety and use
instruction, goal, waypoint, or BEV context only for the task information they
actually provide. Direction labels are short-horizon motion candidates from the
named DirectionProfile; they are not polygons or semantic regions.

Return exactly one <VQAAnswer> JSON object matching the requested payload
schema. Give only the answer payload: no task id, copied conditioned direction,
provenance, rationale, markdown, Finish, trajectory, or text outside the tag.
When evidence is insufficient, use the task schema's unknown or
insufficient-evidence value."""


@dataclass(frozen=True)
class VQAStudentPromptProfileV2:
    """Describe the immutable VQA student prompt policy.

    Args:
        profile_id: Stable prompt identifier.
        system_prompt: Exact system policy.
        policy_digest: Digest over contract and prompt behavior.
    """

    profile_id: str
    system_prompt: str
    policy_digest: str


def build_vqa_student_user_prompt_v2(
    observation: NavObservationV4,
    question: VQAQuestionV2,
) -> str:
    """Build a student prompt from visible input and one atomic question.

    Args:
        observation: Frozen student-visible observation.
        question: Answer-independent atomic question.

    Returns:
        Canonical multimodal user text.
    """

    lines = ["<image>" * len(observation.images)]
    lines.append(
        "Image order: "
        + json.dumps(
            [
                {"ordinal": index + 1, "role": role}
                for index, role in enumerate(
                    observation.image_roles
                )
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    if observation.instruction is not None:
        lines.append(f"Instruction: {observation.instruction}")
    if observation.model_visible_context:
        lines.append(
            "Visible context: "
            + json.dumps(
                observation.model_visible_context,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    question_evidence = set(question.evidence_refs)
    visible_evidence = [
        item.model_dump(mode="json", exclude_none=True)
        for item in observation.evidence_catalog
        if item.ref in question_evidence
    ]
    if visible_evidence:
        lines.append(
            "Prompt-visible evidence catalog: "
            + json.dumps(
                visible_evidence,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    if question.target_refs:
        lines.append(
            "Question target refs: "
            + ", ".join(question.target_refs)
        )
    if question.direction_profile_id is not None:
        lines.append(
            f"Direction profile: {question.direction_profile_id}"
        )
    if question.conditioned_direction_label is not None:
        lines.append(
            "Conditioned direction: "
            f"{question.conditioned_direction_label}"
        )
    if question.candidate_direction_labels:
        lines.append(
            "Candidate directions in presentation order: "
            + ", ".join(question.candidate_direction_labels)
        )
    if question.visible_candidates:
        lines.append(
            "Prompt-visible candidate set: "
            + json.dumps(
                {
                    "candidate_set_id": question.candidate_set_id,
                    "candidates": [
                        item.model_dump(mode="json")
                        for item in question.visible_candidates
                    ],
                    "target_candidate_id": (
                        question.target_candidate_id
                    ),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    lines.append(f"Question: {question.question_text}")
    lines.append(
        "Answer payload JSON schema: "
        + json.dumps(
            answer_payload_json_schema_v2(question.task_id),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    lines.append(
        f"Return exactly <{VQA_V2_ANSWER_TAG}>"
        f"{{...}}</{VQA_V2_ANSWER_TAG}>."
    )
    return "\n".join(lines)


def _student_policy_digest() -> str:
    """Hash behavior-bearing student policy inputs."""

    payload = {
        "profile_id": VQA_V2_STUDENT_PROMPT_PROFILE_ID,
        "schema_version": VQA_V2_SCHEMA_VERSION,
        "contract_schema_digest": VQA_V2_CONTRACT_SCHEMA_DIGEST,
        "system_prompt": VQA_V2_STUDENT_SYSTEM_PROMPT,
        "builder": "build_vqa_student_user_prompt_v2",
    }
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


STUDENT_VQA_PROFILE_V2 = VQAStudentPromptProfileV2(
    profile_id=VQA_V2_STUDENT_PROMPT_PROFILE_ID,
    system_prompt=VQA_V2_STUDENT_SYSTEM_PROMPT,
    policy_digest=_student_policy_digest(),
)


__all__ = [
    "STUDENT_VQA_PROFILE_V2",
    "VQAStudentPromptProfileV2",
    "VQA_V2_STUDENT_PROMPT_PROFILE_ID",
    "VQA_V2_STUDENT_SYSTEM_PROMPT",
    "build_vqa_student_user_prompt_v2",
]
