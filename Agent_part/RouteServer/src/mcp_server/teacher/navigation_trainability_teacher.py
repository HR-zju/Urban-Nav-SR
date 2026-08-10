"""Prompt profiles and parsers for post-annotation data suitability filtering."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from mcp_server.contracts.navigation_trainability import (
    MotionEpisodeSummaryV1,
    MotionSceneAssessmentV1,
    SceneSuitabilityAssessmentV1,
)
from mcp_server.contracts.reasoning_dataset_v4 import ReasoningSourceV4


SCENE_SUITABILITY_LVLM_PROFILE_ID_V1 = "urbannav_scene_suitability_visible_fpv_v1"
MOTION_SCENE_LVLM_PROFILE_ID_V1 = "urbannav_motion_scene_consistency_temporal_fpv_v1"
NAVIGATION_TRAINABILITY_LVLM_PRODUCER_VERSION_V1 = "urbannav_navigation_trainability_teacher_v1_0"

SCENE_SUITABILITY_LVLM_SYSTEM_PROMPT_V1 = """\
You are an offline data-quality teacher for an outdoor pedestrian-assistance
navigation corpus. You receive exactly one current FPV image. Classify only
whether the visible environment is suitable for the current outdoor pedestrian
navigation data pool; do not choose a driving action or infer future motion.

Reject the core outdoor pool for metro/train interiors, indoor narrow corridors,
other indoor spaces, vehicle interiors, severe camera occlusion, and scenes
that are not pedestrian navigation. Outdoor scenes remain suitable only when a
person can plausibly navigate the visible support area. Do not reject merely
because the scene is visually simple. Use unknown only when the visible image
does not support a reliable classification.

Return strict JSON using only the supplied evidence references. Do not mention
trajectory, waypoint, ground truth, metadata, or the annotation process."""

MOTION_SCENE_LVLM_SYSTEM_PROMPT_V1 = """\
You are an offline quality teacher comparing a short ordered FPV sequence with
a teacher-only recorded-motion summary. The summary says what was recorded but
is not visual evidence. Determine whether visible events plausibly explain the
recorded motion. Cite only image roles as evidence.

For a stationary sequence, visible explanations can include a red light,
crossing wait, yielding to a pedestrian, a local obstacle, or visible goal
arrival. If a corridor looks clear and no such cue is visible, use
likely_unexplained_stop rather than inventing a reason. For moving sequences,
dense nearby crowds are not automatically bad; use likely_unsafe_dense_crowd_motion
or likely_clearance_conflict only when the visible near-field corridor lacks
adequate safe clearance or the recorded motion conflicts with people/obstacles.
Use visual_evidence_insufficient when the camera cannot establish the reason.

Do not predict a navigation action, do not expose coordinates, and do not claim
unseen entities or tracked motion. Return exactly one JSON object."""


@dataclass(frozen=True)
class NavigationTrainabilityTeacherProfileV1:
    """Freeze one independently cacheable LVLM filtering profile.

    Args:
        profile_id: Stable prompt profile identity.
        producer_version: Exact teacher implementation version.
        system_prompt: Shared profile safety and evidence policy.
        output_json_schema: Strict provider response schema.
        policy_digest: SHA-256 digest of profile contents.
    """

    profile_id: str
    producer_version: str
    system_prompt: str
    output_json_schema: dict[str, Any]
    policy_digest: str


def scene_suitability_teacher_profile_v1() -> NavigationTrainabilityTeacherProfileV1:
    """Return the current-FPV outdoor-scene suitability profile."""

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["scene_class", "confidence", "evidence_refs", "reason_codes"],
        "properties": {
            "scene_class": {
                "type": "string",
                "enum": [
                    "outdoor_pedestrian_supported", "outdoor_but_unsuitable", "indoor_transit",
                    "indoor_narrow_corridor", "indoor_other", "vehicle_interior", "camera_occluded",
                    "non_pedestrian_or_irrelevant", "unknown",
                ],
            },
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "evidence_refs": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5},
            "reason_codes": {
                "type": "array",
                "items": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"},
                "maxItems": 8,
            },
        },
    }
    return _profile_v1(
        profile_id=SCENE_SUITABILITY_LVLM_PROFILE_ID_V1,
        system_prompt=SCENE_SUITABILITY_LVLM_SYSTEM_PROMPT_V1,
        schema=schema,
    )


def motion_scene_teacher_profile_v1() -> NavigationTrainabilityTeacherProfileV1:
    """Return the temporal FPV motion-consistency profile."""

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "consistency", "crowd_density_near_path", "corridor_occupancy", "visible_stop_cue",
            "confidence", "evidence_refs", "reason_codes",
        ],
        "properties": {
            "consistency": {
                "type": "string",
                "enum": [
                    "consistent", "likely_unexplained_stop", "likely_unsafe_dense_crowd_motion",
                    "likely_clearance_conflict", "visual_evidence_insufficient", "unknown",
                ],
            },
            "crowd_density_near_path": {"type": "string", "enum": ["none", "sparse", "moderate", "dense", "unknown"]},
            "corridor_occupancy": {"type": "string", "enum": ["clear", "shared", "blocked", "unknown"]},
            "visible_stop_cue": {
                "type": "string",
                "enum": ["red_light", "crossing_wait", "pedestrian_yield", "obstacle", "goal_arrival", "none", "unknown"],
            },
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "evidence_refs": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 8},
            "reason_codes": {
                "type": "array",
                "items": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"},
                "maxItems": 8,
            },
        },
    }
    return _profile_v1(
        profile_id=MOTION_SCENE_LVLM_PROFILE_ID_V1,
        system_prompt=MOTION_SCENE_LVLM_SYSTEM_PROMPT_V1,
        schema=schema,
    )


def build_scene_suitability_question_v1(
    source: ReasoningSourceV4,
    profile: NavigationTrainabilityTeacherProfileV1,
    *,
    previous_answer: str | None = None,
    previous_errors: Iterable[str] = (),
) -> str:
    """Build a current-FPV scene-suitability request.

    Args:
        source: Canonical source providing the current FPV image.
        profile: Frozen scene profile.
        previous_answer: Optional invalid provider response for repair.
        previous_errors: Deterministic validation errors for repair.

    Returns:
        JSON-only prompt that excludes trajectories and annotations.
    """

    return _build_question_v1(
        source=source,
        profile=profile,
        image_roles=("fpv.current",),
        teacher_context=None,
        previous_answer=previous_answer,
        previous_errors=previous_errors,
        task="Classify the current scene for outdoor pedestrian-navigation dataset suitability.",
    )


def build_motion_scene_question_v1(
    source: ReasoningSourceV4,
    motion_summary: MotionEpisodeSummaryV1,
    profile: NavigationTrainabilityTeacherProfileV1,
    *,
    temporal_image_roles: tuple[str, ...],
    previous_answer: str | None = None,
    previous_errors: Iterable[str] = (),
) -> str:
    """Build a temporal FPV motion-consistency request.

    Args:
        source: Canonical source providing ordered FPV images.
        motion_summary: Teacher-only deterministic episode summary.
        profile: Frozen temporal profile.
        temporal_image_roles: Ordered history/current FPV roles sent to LVLM.
        previous_answer: Optional invalid provider response for repair.
        previous_errors: Deterministic validation errors for repair.

    Returns:
        JSON-only prompt.  The motion summary is declared teacher-only and is
        excluded from the evidence-reference vocabulary.
    """

    return _build_question_v1(
        source=source,
        profile=profile,
        image_roles=temporal_image_roles,
        teacher_context={"teacher_only_recorded_motion_summary": motion_summary.model_dump(mode="json")},
        previous_answer=previous_answer,
        previous_errors=previous_errors,
        task="Assess whether visible temporal FPV evidence explains the recorded motion summary.",
    )


def parse_scene_suitability_answer_v1(
    text: str,
    *,
    source: ReasoningSourceV4,
    teacher_model: str | None,
    profile: NavigationTrainabilityTeacherProfileV1,
) -> tuple[SceneSuitabilityAssessmentV1 | None, list[str]]:
    """Parse one scene-suitability response into its typed assessment."""

    payload, errors = _parse_payload_v1(text, profile)
    if payload is None:
        return None, errors
    try:
        assessment = SceneSuitabilityAssessmentV1(
            scene_class=payload["scene_class"],
            confidence=payload["confidence"],
            evidence_refs=tuple(str(value) for value in payload["evidence_refs"]),
            reason_codes=tuple(str(value) for value in payload["reason_codes"]),
            teacher_model=teacher_model or "unknown_teacher_model",
            prompt_profile_id=profile.profile_id,
            prompt_profile_digest=profile.policy_digest,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return None, [f"scene_suitability_schema:{exc}"]
    errors = _validate_evidence_refs_v1(assessment.evidence_refs, source, allowed_roles=("fpv.current",))
    return (assessment, []) if not errors else (None, errors)


def parse_motion_scene_answer_v1(
    text: str,
    *,
    source: ReasoningSourceV4,
    teacher_model: str | None,
    profile: NavigationTrainabilityTeacherProfileV1,
    temporal_image_roles: tuple[str, ...],
) -> tuple[MotionSceneAssessmentV1 | None, list[str]]:
    """Parse one temporal motion-consistency response into a typed assessment."""

    payload, errors = _parse_payload_v1(text, profile)
    if payload is None:
        return None, errors
    try:
        assessment = MotionSceneAssessmentV1(
            consistency=payload["consistency"],
            crowd_density_near_path=payload["crowd_density_near_path"],
            corridor_occupancy=payload["corridor_occupancy"],
            visible_stop_cue=payload["visible_stop_cue"],
            confidence=payload["confidence"],
            evidence_refs=tuple(str(value) for value in payload["evidence_refs"]),
            reason_codes=tuple(str(value) for value in payload["reason_codes"]),
            teacher_model=teacher_model or "unknown_teacher_model",
            prompt_profile_id=profile.profile_id,
            prompt_profile_digest=profile.policy_digest,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return None, [f"motion_scene_schema:{exc}"]
    errors = _validate_evidence_refs_v1(assessment.evidence_refs, source, allowed_roles=temporal_image_roles)
    return (assessment, []) if not errors else (None, errors)


def _profile_v1(
    *,
    profile_id: str,
    system_prompt: str,
    schema: dict[str, Any],
) -> NavigationTrainabilityTeacherProfileV1:
    """Create one digest-stamped profile from immutable prompt inputs."""

    payload = {
        "profile_id": profile_id,
        "producer_version": NAVIGATION_TRAINABILITY_LVLM_PRODUCER_VERSION_V1,
        "system_prompt": system_prompt,
        "output_json_schema": schema,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return NavigationTrainabilityTeacherProfileV1(
        profile_id=profile_id,
        producer_version=NAVIGATION_TRAINABILITY_LVLM_PRODUCER_VERSION_V1,
        system_prompt=system_prompt,
        output_json_schema=schema,
        policy_digest=digest,
    )


def _build_question_v1(
    *,
    source: ReasoningSourceV4,
    profile: NavigationTrainabilityTeacherProfileV1,
    image_roles: tuple[str, ...],
    teacher_context: dict[str, Any] | None,
    previous_answer: str | None,
    previous_errors: Iterable[str],
    task: str,
) -> str:
    """Build one profile request and restrict evidence refs to sent images."""

    source_roles = set(source.observation.image_roles)
    if "fpv.current" not in image_roles or not set(image_roles).issubset(source_roles):
        raise ValueError("requested_trainability_image_roles_are_not_available")
    payload: dict[str, Any] = {
        "source_group_id": source.observation.source_group_id,
        "image_order": [{"ordinal": index + 1, "role": role} for index, role in enumerate(image_roles)],
        "visible_evidence_catalog": [{"ref": role, "modality": "fpv"} for role in image_roles],
        "output_json_schema": profile.output_json_schema,
    }
    if teacher_context is not None:
        payload.update(teacher_context)
    parts = [task, "Return strict JSON and cite only visible_evidence_catalog references.", "FROZEN_INPUT=" + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))]
    errors = [str(value)[:500] for value in previous_errors if str(value)]
    if previous_answer is not None or errors:
        parts.extend(
            [
                "The previous response was invalid. Repair it without inventing visual evidence.",
                "VALIDATION_ERRORS=" + json.dumps(errors[:16], ensure_ascii=False, separators=(",", ":")),
                "PREVIOUS_RESPONSE=" + str(previous_answer or "")[:12000],
            ]
        )
    return "\n\n".join(parts)


def _parse_payload_v1(
    text: str,
    profile: NavigationTrainabilityTeacherProfileV1,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Parse strict JSON and reject missing or provider-invented keys."""

    body = _strip_json_fence_v1(str(text or "").strip())
    if not body:
        return None, ["empty_teacher_answer"]
    try:
        payload = json.loads(body)
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
    return payload, []


def _validate_evidence_refs_v1(
    refs: tuple[str, ...],
    source: ReasoningSourceV4,
    *,
    allowed_roles: tuple[str, ...],
) -> list[str]:
    """Ensure model evidence is both source-known and profile-visible."""

    source_roles = set(source.observation.image_roles)
    invalid = [ref for ref in refs if ref not in source_roles or ref not in allowed_roles]
    return ["assessment_contains_nonvisible_evidence_refs:" + ",".join(sorted(set(invalid)))] if invalid else []


def _strip_json_fence_v1(value: str) -> str:
    """Strip one whole-response JSON code fence when supplied by a provider."""

    match = re.fullmatch(r"```(?:json)?\s*(?P<body>.*?)\s*```", value, flags=re.DOTALL | re.IGNORECASE)
    return match.group("body").strip() if match else value


__all__ = [
    "MOTION_SCENE_LVLM_PROFILE_ID_V1",
    "MOTION_SCENE_LVLM_SYSTEM_PROMPT_V1",
    "NAVIGATION_TRAINABILITY_LVLM_PRODUCER_VERSION_V1",
    "SCENE_SUITABILITY_LVLM_PROFILE_ID_V1",
    "SCENE_SUITABILITY_LVLM_SYSTEM_PROMPT_V1",
    "NavigationTrainabilityTeacherProfileV1",
    "build_motion_scene_question_v1",
    "build_scene_suitability_question_v1",
    "motion_scene_teacher_profile_v1",
    "parse_motion_scene_answer_v1",
    "parse_scene_suitability_answer_v1",
    "scene_suitability_teacher_profile_v1",
]
