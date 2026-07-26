"""Versioned task-specific weak-label teacher profiles for VQA V2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from mcp_server.agent.vqa_v2_text import (
    answer_payload_json_schema_v2,
)
from mcp_server.contracts.vqa_v2 import (
    VQA_V2_CONTRACT_SCHEMA_DIGEST,
    VQA_V2_SCHEMA_VERSION,
)


LVLM_VQA_PRODUCER_ID = "lvlm_visible_vqa_teacher"
LVLM_VQA_PRODUCER_VERSION = "lvlm_visible_vqa_teacher_v2_0"

VQA_V2_WEAK_TASK_IDS = (
    "a1_goal_understanding",
    "a2_explicit_requirement",
    "a3_task_ambiguity",
    "b1_direction_affordance",
    "b2_direction_clearance",
    "b3_direction_surface_effect",
    "b4_conditional_affordance",
    "b5_direction_comparison",
    "b6_unavailable_directions",
    "c1_goal_to_direction",
    "d1_agent_relevance",
    "d2_static_interaction_relation",
    "d3_temporal_interaction_relation",
    "d4_applicable_constraint",
    "d5_appropriate_response",
    "d6_traffic_control_applicability",
    "e4_alternative_direction",
)

_COMMON_POLICY = """\
You are an offline weak-label teacher for atomic visual questions in an
assistive urban navigation model. Use only the complete prompt-visible images,
instruction, goal, waypoint, BEV context, question direction, and candidate
list supplied by the request. FPV is authoritative for immediate visible
safety. A direction label is a short-horizon motion candidate under the named
DirectionProfile, not a polygon or semantic region.

Describe only evidence visible to the student. A single FPV may support static
occupancy, proximity, obstruction, surface, and coarse clearance. Temporal
relations require the request to declare distinct temporal evidence. Use
unknown or insufficient-evidence values instead of inventing a match, motion,
rule, negative example, route, calibration, tracking fact, or geometry.

Return exactly one <VQAAnswer> JSON payload matching the task schema. Return no
task id, copied conditioned direction, rationale, markdown, provenance, or text
outside the tag."""

_TASK_POLICY = {
    "a1_goal_understanding": (
        "Extract only the goal explicitly present in instruction or visible "
        "goal context. Do not infer a runtime phase or hidden objective."
    ),
    "a2_explicit_requirement": (
        "Return only explicitly stated task requirements as compact "
        "lower-snake-case codes."
    ),
    "a3_task_ambiguity": (
        "Judge ambiguity only from prompt-visible task language. Do not "
        "treat missing scene facts as instruction ambiguity."
    ),
    "b1_direction_affordance": (
        "Judge current visible affordance along the conditioned direction."
    ),
    "b2_direction_clearance": (
        "Judge semantic visible clearance, not metric distance."
    ),
    "b3_direction_surface_effect": (
        "Name visible support with a lower-snake-case code and its coarse "
        "navigation effect."
    ),
    "b4_conditional_affordance": (
        "Separate immediate usability, a visible required condition, and "
        "the appropriate behavior."
    ),
    "b5_direction_comparison": (
        "Choose only from the supplied candidate directions. Candidate "
        "presentation order carries no preference."
    ),
    "b6_unavailable_directions": (
        "Return only unusable members of the supplied candidate set. An "
        "empty list is allowed; this is not exhaustive free-space truth."
    ),
    "c1_goal_to_direction": (
        "Ground only a prompt-visible goal to supplied direction candidates."
    ),
    "d1_agent_relevance": (
        "Judge whether any visible agent affects the conditioned direction "
        "without creating an instance or track identity."
    ),
    "d2_static_interaction_relation": (
        "Use static visible relations only."
    ),
    "d3_temporal_interaction_relation": (
        "Use ordered distinct FPV evidence for temporal claims; otherwise "
        "return unknown."
    ),
    "d4_applicable_constraint": (
        "Return one currently visible applicable constraint. Constraint "
        "codes are open lower-snake-case identifiers."
    ),
    "d5_appropriate_response": (
        "Choose the immediate response supported by visible evidence."
    ),
    "d6_traffic_control_applicability": (
        "Seeing a control is insufficient: judge whether it governs the "
        "conditioned direction."
    ),
    "e4_alternative_direction": (
        "Choose only from supplied candidate directions, or report no safe "
        "alternative/insufficient evidence."
    ),
}


@dataclass(frozen=True)
class VQATeacherProfileV2:
    """Describe one immutable task-specific teacher policy.

    Args:
        profile_id: Stable prompt profile id.
        task_id: Atomic VQA task.
        output_json_schema: Minimal assistant payload schema.
        system_prompt: Exact teacher policy.
        policy_digest: Digest over contract, task, schema, and prompt.
    """

    profile_id: str
    task_id: str
    output_json_schema: dict[str, Any]
    system_prompt: str
    policy_digest: str


def _profile(task_id: str) -> VQATeacherProfileV2:
    """Build one teacher profile from canonical task policy."""

    output_schema = answer_payload_json_schema_v2(task_id)
    system_prompt = (
        _COMMON_POLICY
        + "\n\nTASK RESPONSIBILITY\n"
        + _TASK_POLICY[task_id]
    )
    profile_id = f"urbannav_vqa_teacher_{task_id}_v2_0"
    payload = {
        "profile_id": profile_id,
        "task_id": task_id,
        "schema_version": VQA_V2_SCHEMA_VERSION,
        "contract_schema_digest": VQA_V2_CONTRACT_SCHEMA_DIGEST,
        "producer_id": LVLM_VQA_PRODUCER_ID,
        "producer_version": LVLM_VQA_PRODUCER_VERSION,
        "output_json_schema": output_schema,
        "system_prompt": system_prompt,
    }
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return VQATeacherProfileV2(
        profile_id=profile_id,
        task_id=task_id,
        output_json_schema=output_schema,
        system_prompt=system_prompt,
        policy_digest=hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest(),
    )


VQA_V2_TEACHER_PROFILES: Mapping[str, VQATeacherProfileV2] = (
    MappingProxyType(
        {
            task_id: _profile(task_id)
            for task_id in VQA_V2_WEAK_TASK_IDS
        }
    )
)


def get_vqa_teacher_profile_v2(task_id: str) -> VQATeacherProfileV2:
    """Resolve one active weak-label teacher profile.

    Args:
        task_id: Exact atomic task id.

    Returns:
        Immutable teacher profile.

    Raises:
        ValueError: If the task has no LVLM weak-label producer.
    """

    try:
        return VQA_V2_TEACHER_PROFILES[task_id]
    except KeyError as exc:
        raise ValueError(
            f"task has no VQA V2 weak-label profile: {task_id}"
        ) from exc


__all__ = [
    "LVLM_VQA_PRODUCER_ID",
    "LVLM_VQA_PRODUCER_VERSION",
    "VQATeacherProfileV2",
    "VQA_V2_TEACHER_PROFILES",
    "VQA_V2_WEAK_TASK_IDS",
    "get_vqa_teacher_profile_v2",
]
