"""Visible-FPV prompt construction for language-goal annotation V1.

The future trajectory candidate only guides the offline teacher toward a
semantically compatible destination. The student-visible annotation is
accepted only when the LVLM grounds it in the current FPV image.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

from mcp_server.contracts.language_waypoint_benchmark import (
    LANGUAGE_GOAL_ANNOTATION_SCHEMA_VERSION,
    LanguageGoalAnnotationV1,
    LanguageGoalCandidateV1,
)
from mcp_server.contracts.reasoning_dataset_v4 import ReasoningSourceV4


LANGUAGE_GOAL_LVLM_PROFILE_ID_V1 = "urbannav_language_goal_visible_fpv_v1"
LANGUAGE_GOAL_LVLM_PRODUCER_VERSION_V1 = "urbannav_language_goal_teacher_v1_0"

LANGUAGE_GOAL_LVLM_SYSTEM_PROMPT_V1 = """\
You are an offline annotation teacher for an urban pedestrian navigation
benchmark. You receive exactly one current first-person image, an optional
existing dataset text, and a teacher-only coarse target relation. The target
relation helps choose the intended navigation semantics, but it is not
student-visible evidence.

Create an instruction only when a real, visible landmark, entrance, crossing,
or local corridor in the current image can ground the wording. Existing text
may be a caption, a stale instruction, or a valid instruction; inspect and
repair it rather than copying it blindly. Never mention coordinates, metres,
degrees, trajectories, waypoints, future frames, ground truth, candidates, or
the annotation process. Do not invent a landmark, traffic control, obstacle,
or accessibility property that is not visible in the image.

When no reliable visible referent is compatible with the target relation,
return validator_status="rejected" or validator_status="pending_human". For
accepted output, use target_ref exactly "fpv.current", set
visibility_at_anchor=true, and cite fpv.current in evidence_refs. A bounding
box is normalized [x1,y1,x2,y2] in [0,1], required for a discrete landmark,
and may be null only for a whole visible corridor or crossing region.

Return exactly one JSON object matching the supplied schema. Do not emit
markdown, explanations, or extra keys."""


@dataclass(frozen=True)
class LanguageGoalTeacherProfileV1:
    """Freeze the V1 language annotation teacher contract.

    Args:
        profile_id: Stable prompt-profile identifier.
        producer_version: Exact offline teacher implementation version.
        system_prompt: Shared visible-evidence policy.
        output_json_schema: Strict model response schema before provenance is
            injected by the runner.
        policy_digest: SHA-256 over the frozen profile payload.
    """

    profile_id: str
    producer_version: str
    system_prompt: str
    output_json_schema: dict[str, Any]
    policy_digest: str


def language_goal_teacher_profile_v1() -> LanguageGoalTeacherProfileV1:
    """Return the immutable language-goal teacher profile.

    Returns:
        Versioned prompt and response schema used by the annotation runner.
    """

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "instruction_text",
            "instruction_type",
            "target_ref",
            "target_bbox_anchor_xyxy",
            "spatial_relation",
            "visibility_at_anchor",
            "evidence_refs",
            "confidence",
            "validator_status",
        ],
        "properties": {
            "instruction_text": {"type": ["string", "null"], "maxLength": 512},
            "instruction_type": {
                "type": ["string", "null"],
                "enum": [
                    "landmark_object_goal",
                    "visible_corridor_goal",
                    "crossing_entrance_region",
                    "accessibility_constraint",
                    "social_rule_constraint",
                    "arrival_stop",
                    None,
                ],
            },
            "target_ref": {
                "type": ["string", "null"],
                "enum": ["fpv.current", None],
            },
            "target_bbox_anchor_xyxy": {
                "type": ["array", "null"],
                "minItems": 4,
                "maxItems": 4,
                "items": {"type": "number"},
            },
            "spatial_relation": {"type": ["string", "null"], "maxLength": 160},
            "visibility_at_anchor": {"type": "boolean"},
            "evidence_refs": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 4,
            },
            "confidence": {
                "type": ["string", "null"],
                "enum": ["high", "medium", "low", None],
            },
            "validator_status": {
                "type": "string",
                "enum": ["accepted", "rejected", "pending_human"],
            },
        },
    }
    payload = {
        "profile_id": LANGUAGE_GOAL_LVLM_PROFILE_ID_V1,
        "producer_version": LANGUAGE_GOAL_LVLM_PRODUCER_VERSION_V1,
        "schema_version": LANGUAGE_GOAL_ANNOTATION_SCHEMA_VERSION,
        "system_prompt": LANGUAGE_GOAL_LVLM_SYSTEM_PROMPT_V1,
        "output_json_schema": schema,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return LanguageGoalTeacherProfileV1(
        profile_id=LANGUAGE_GOAL_LVLM_PROFILE_ID_V1,
        producer_version=LANGUAGE_GOAL_LVLM_PRODUCER_VERSION_V1,
        system_prompt=LANGUAGE_GOAL_LVLM_SYSTEM_PROMPT_V1,
        output_json_schema=schema,
        policy_digest=digest,
    )


def build_language_goal_teacher_question_v1(
    source: ReasoningSourceV4,
    candidate: LanguageGoalCandidateV1,
    profile: LanguageGoalTeacherProfileV1,
    *,
    previous_answer: str | None = None,
    previous_errors: Iterable[str] = (),
) -> str:
    """Build a request from current FPV and a coarse target relation.

    Args:
        source: Frozen source record providing the anchor image and text.
        candidate: Teacher-only future target chosen from a trusted sidecar.
        profile: Frozen LVLM prompt profile.
        previous_answer: Invalid prior response used only for repair.
        previous_errors: Deterministic validation findings for repair.

    Returns:
        JSON-only user prompt for an OpenAI-compatible VLM.

    Raises:
        ValueError: If candidate/source identities do not match or FPV is absent.
    """

    if candidate.parent_sample_id != source.observation.source_group_id:
        raise ValueError("candidate parent does not match source_group_id")
    if "fpv.current" not in source.observation.image_roles:
        raise ValueError("source has no fpv.current image")
    visible_input = {
        "source_group_id": source.observation.source_group_id,
        "image_order": [{"ordinal": 1, "role": "fpv.current"}],
        "existing_dataset_text": source.observation.instruction,
        "travel_mode": source.observation.travel_mode,
        "visible_evidence_catalog": [{"ref": "fpv.current", "modality": "fpv"}],
    }
    payload = {
        "visible_input": visible_input,
        "teacher_only_target_relation": _candidate_relation_v1(candidate),
        "output_json_schema": profile.output_json_schema,
    }
    parts = [
        "Produce one language-goal annotation from this current FPV image.",
        "Do not reveal teacher_only_target_relation in the instruction or response text.",
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    ]
    errors = [str(value)[:500] for value in previous_errors if str(value)]
    if previous_answer is not None or errors:
        parts.extend(
            [
                "The previous response was invalid. Repair it without adding unsupported visual facts.",
                json.dumps(
                    {
                        "previous_answer": str(previous_answer or "")[:12000],
                        "validator_errors": errors[:16],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
    return "\n\n".join(parts)


def parse_language_goal_teacher_answer_v1(
    text: str,
    *,
    source: ReasoningSourceV4,
    candidate: LanguageGoalCandidateV1,
    teacher_model: str | None,
    profile: LanguageGoalTeacherProfileV1,
) -> tuple[LanguageGoalAnnotationV1 | None, list[str]]:
    """Parse and validate one LVLM language annotation response.

    Args:
        text: Raw model response.
        source: Frozen source row used for identity and provenance.
        candidate: Teacher-only target used to attach private metadata.
        teacher_model: Provider-reported model identifier.
        profile: Exact prompt profile used for this response.

    Returns:
        Typed annotation or deterministic errors.
    """

    value = _strip_json_fence_v1(str(text or "").strip())
    if not value:
        return None, ["empty_teacher_answer"]
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        return None, [f"invalid_json:{exc.msg}"]
    if not isinstance(payload, dict):
        return None, ["teacher_payload_must_be_object"]
    allowed = set(profile.output_json_schema["properties"])
    missing = sorted(allowed - set(payload))
    extras = sorted(set(payload) - allowed)
    if missing or extras:
        errors: list[str] = []
        if missing:
            errors.append("teacher_payload_missing:" + ",".join(missing))
        if extras:
            errors.append("teacher_payload_extra:" + ",".join(extras))
        return None, errors
    status = payload.get("validator_status")
    if status not in {"accepted", "rejected", "pending_human"}:
        return None, ["invalid_validator_status"]
    errors = _validate_visible_payload_v1(payload, status=str(status))
    if errors:
        return None, errors
    try:
        annotation = LanguageGoalAnnotationV1(
            annotation_id=_annotation_id_v1(candidate=candidate, profile=profile),
            parent_sample_id=source.observation.source_group_id,
            instruction_text=_optional_text_v1(payload.get("instruction_text")),
            instruction_type=payload.get("instruction_type"),
            target_ref=payload.get("target_ref"),
            target_bbox_anchor_xyxy=_normalized_bbox_v1(payload.get("target_bbox_anchor_xyxy")),
            spatial_relation=_optional_text_v1(payload.get("spatial_relation")),
            target_route_distance_m=math.hypot(
                candidate.candidate_xy_flu_m.x_m,
                candidate.candidate_xy_flu_m.y_m,
            ),
            target_time_offset_s=candidate.candidate_time_s,
            visibility_at_anchor=bool(payload.get("visibility_at_anchor")),
            evidence_refs=tuple(str(item) for item in payload.get("evidence_refs", [])),
            teacher_model=teacher_model or "unknown_teacher_model",
            prompt_version=profile.profile_id,
            confidence=payload.get("confidence"),
            validator_status=status,
            provenance="lvlm_annotation",
        )
    except (TypeError, ValueError) as exc:
        return None, [f"annotation_schema:{exc}"]
    return annotation, []


def _candidate_relation_v1(candidate: LanguageGoalCandidateV1) -> dict[str, str | float]:
    """Convert private target geometry to a coarse non-coordinate hint."""

    forward = candidate.candidate_xy_flu_m.x_m
    left = candidate.candidate_xy_flu_m.y_m
    angle_deg = math.degrees(math.atan2(left, forward))
    if abs(angle_deg) <= 12.0:
        relation = "ahead"
    elif abs(angle_deg) <= 45.0:
        relation = "ahead_left" if angle_deg > 0.0 else "ahead_right"
    else:
        relation = "left" if angle_deg > 0.0 else "right"
    distance_m = math.hypot(forward, left)
    distance_band = "near" if distance_m < 8.0 else "mid_range" if distance_m < 20.0 else "far"
    return {
        "desired_relation": relation,
        "distance_band": distance_band,
        "time_horizon_s": round(candidate.candidate_time_s, 3),
    }


def _validate_visible_payload_v1(payload: dict[str, Any], *, status: str) -> list[str]:
    """Apply anti-leakage and current-FPV grounding checks."""

    if status != "accepted":
        forbidden = [
            key
            for key in (
                "instruction_text",
                "instruction_type",
                "target_ref",
                "target_bbox_anchor_xyxy",
                "spatial_relation",
                "confidence",
            )
            if payload.get(key) is not None and payload.get(key) != ""
        ]
        if payload.get("visibility_at_anchor") or payload.get("evidence_refs"):
            forbidden.append("visible_evidence")
        return ["nonaccepted_payload_must_not_carry_instruction:" + ",".join(forbidden)] if forbidden else []
    instruction = _optional_text_v1(payload.get("instruction_text"))
    if not instruction or not 3 <= len(instruction) <= 512:
        return ["accepted_instruction_text_missing_or_length_invalid"]
    lower = instruction.lower()
    forbidden_terms = (
        "trajectory",
        "waypoint",
        "ground truth",
        "candidate",
        "odometry",
        "metre",
        "meter",
        "degree",
    )
    leaked = [term for term in forbidden_terms if term in lower]
    if leaked or re.search(r"\b\d+(?:\.\d+)?\s*(?:m|meters?|metres?|deg|degrees?)\b", lower):
        return ["instruction_contains_teacher_private_geometry:" + ",".join(leaked or ["numeric_geometry"])]
    if payload.get("instruction_type") is None:
        return ["accepted_instruction_type_missing"]
    if payload.get("target_ref") != "fpv.current" or not payload.get("visibility_at_anchor"):
        return ["accepted_target_must_be_visible_in_current_fpv"]
    evidence = payload.get("evidence_refs")
    if not isinstance(evidence, list) or "fpv.current" not in evidence:
        return ["accepted_annotation_requires_fpv_current_evidence"]
    if payload.get("confidence") not in {"high", "medium"}:
        return ["accepted_annotation_confidence_too_low"]
    try:
        _normalized_bbox_v1(payload.get("target_bbox_anchor_xyxy"))
    except ValueError as exc:
        return [str(exc)]
    return []


def _normalized_bbox_v1(value: Any) -> tuple[float, float, float, float] | None:
    """Validate an optional normalized image bounding box."""

    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("target_bbox_anchor_xyxy_must_have_four_values")
    box = tuple(float(item) for item in value)
    if not all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in box):
        raise ValueError("target_bbox_anchor_xyxy_must_be_normalized")
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("target_bbox_anchor_xyxy_must_be_ordered")
    return box


def _annotation_id_v1(
    *,
    candidate: LanguageGoalCandidateV1,
    profile: LanguageGoalTeacherProfileV1,
) -> str:
    """Return a stable id tied to candidate and prompt version."""

    digest = hashlib.sha256(
        f"{candidate.candidate_id}\0{profile.profile_id}\0{profile.policy_digest}".encode("utf-8")
    ).hexdigest()[:24]
    return f"language_goal_annotation:{digest}"


def _optional_text_v1(value: Any) -> str | None:
    """Normalize optional short teacher text."""

    text = str(value or "").strip()
    return text or None


def _strip_json_fence_v1(value: str) -> str:
    """Strip one whole-response JSON fence when present."""

    match = re.fullmatch(r"```(?:json)?\s*(?P<body>.*?)\s*```", value, flags=re.DOTALL | re.IGNORECASE)
    return match.group("body").strip() if match else value


__all__ = [
    "LANGUAGE_GOAL_LVLM_PRODUCER_VERSION_V1",
    "LANGUAGE_GOAL_LVLM_PROFILE_ID_V1",
    "LANGUAGE_GOAL_LVLM_SYSTEM_PROMPT_V1",
    "LanguageGoalTeacherProfileV1",
    "build_language_goal_teacher_question_v1",
    "language_goal_teacher_profile_v1",
    "parse_language_goal_teacher_answer_v1",
]
