"""Versioned two-stage LVLM review profiles for Reasoning V4.6."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from pydantic import BaseModel

from mcp_server.contracts.reasoning_dataset_v4 import (
    ACTION_REVIEW_FILTER_VERSION_V4,
    ActionBlindReviewV4,
    ActionReviewDiagnosisPayloadV4,
)


REASONING_V4_REVIEW_ALL_VISIBLE_POLICY_ID = (
    "reasoning_v4_review_all_visible_v1"
)

_COMMON_POLICY = """\
You are an offline quality-control reviewer for Reasoning V4.6 urban pedestrian
navigation annotations. This is annotation diagnosis, not student training and
not a navigation-target generator.

FPV is authoritative for visible walkable support, obstacles, clearance,
traffic controls, and immediate interaction risk. History FPV frames are visual
context only; do not create temporal-scope labels, tracks, TTC, or calibrated
confidence. A prompt-visible BEV route overlay may support route direction and
upcoming topology when its orientation is clear, but cannot prove
traversability, clearance, or collision safety. Do not use GPS, route matching,
route theta, pose, pixel-goal geometry, or waypoint geometry.

Never output corrected action labels, corrected grounding values, suggested
values, replacement values, or a new FullCoT. Return exactly one JSON object
matching output_json_schema, with no markdown or surrounding prose."""

_STAGE_POLICIES = {
    "blind_visual": """\
You are blind to the teacher Decision proposal, private strong direction,
trajectory sidecars, deterministic action audit, composer candidate, and human
decisions. Assess only evidence sufficiency and visible findings. Do not choose
an error source or repair action. visual_findings and ambiguous_evidence must be
concise lower_snake_case codes. relevant_components may name only the five
Grounding sections.""",
    "comparative_diagnosis": """\
Compare the blind visual assessment with the normalized Grounding sections,
teacher Decision proposal, frozen strong direction stamp, and deterministic
audit facts. The audit is a review trigger, not truth. The strong label is
supervision, not infallible.

verdict has exactly four values: teacher_proposal_error, grounding_error,
strong_label_error, unresolved. A constraint issue is grounding_error with
interaction_constraint_error. Insufficient evidence is unresolved with a
reason code. Multiple affected components are scope, not a new verdict.
Possible coarse/fine equivalence remains unresolved. Do not introduce temporal
scope or any fifth verdict.

confidence is an uncalibrated self-score used only to order the human queue.
repair_action is advisory and never changes training eligibility. Human review
is always required.""",
}

_STAGE_MODELS: dict[str, type[BaseModel]] = {
    "blind_visual": ActionBlindReviewV4,
    "comparative_diagnosis": ActionReviewDiagnosisPayloadV4,
}


@dataclass(frozen=True)
class ReasoningV4ReviewProfile:
    """Describe one executable LVLM review stage.

    Args:
        profile_id: Stable profile identity.
        stage_name: Blind or comparative stage.
        model_type: Pydantic model parsed from the provider response.
        output_json_schema: Exact provider output schema.
        system_prompt: Exact provider-side policy.
        policy_digest: Digest over all behavior-defining components.
        derivation_version: Exact review derivation version.
        image_policy_id: Exact all-visible reviewer image policy.
    """

    profile_id: str
    stage_name: str
    model_type: type[BaseModel]
    output_json_schema: dict[str, Any]
    system_prompt: str
    policy_digest: str
    derivation_version: str
    image_policy_id: str


def _profile(stage_name: str) -> ReasoningV4ReviewProfile:
    """Build one immutable review-stage profile."""

    model_type = _STAGE_MODELS[stage_name]
    output_json_schema = json.loads(
        json.dumps(
            model_type.model_json_schema(),
            ensure_ascii=False,
        )
    )
    profile_id = f"reasoning_v4_review_{stage_name}_v1"
    derivation_version = f"reasoning_v4_6_review_{stage_name}_v1"
    system_prompt = (
        f"{_COMMON_POLICY}\n\nREVIEW STAGE — {stage_name}\n"
        f"{_STAGE_POLICIES[stage_name]}"
    )
    payload = {
        "profile_id": profile_id,
        "filter_version": ACTION_REVIEW_FILTER_VERSION_V4,
        "stage_name": stage_name,
        "derivation_version": derivation_version,
        "image_policy_id": REASONING_V4_REVIEW_ALL_VISIBLE_POLICY_ID,
        "output_json_schema": output_json_schema,
        "system_prompt": system_prompt,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return ReasoningV4ReviewProfile(
        profile_id=profile_id,
        stage_name=stage_name,
        model_type=model_type,
        output_json_schema=output_json_schema,
        system_prompt=system_prompt,
        policy_digest=digest,
        derivation_version=derivation_version,
        image_policy_id=REASONING_V4_REVIEW_ALL_VISIBLE_POLICY_ID,
    )


REASONING_V4_REVIEW_PROFILES: Mapping[
    str,
    ReasoningV4ReviewProfile,
] = MappingProxyType(
    {
        stage_name: _profile(stage_name)
        for stage_name in _STAGE_MODELS
    }
)


def get_reasoning_v4_review_profile(
    stage_name: str,
) -> ReasoningV4ReviewProfile:
    """Resolve one canonical review-stage profile.

    Args:
        stage_name: ``blind_visual`` or ``comparative_diagnosis``.

    Returns:
        Immutable review profile.

    Raises:
        ValueError: If the stage name is unsupported.
    """

    try:
        return REASONING_V4_REVIEW_PROFILES[str(stage_name)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported Reasoning V4 review stage: {stage_name!r}"
        ) from exc


__all__ = [
    "REASONING_V4_REVIEW_ALL_VISIBLE_POLICY_ID",
    "REASONING_V4_REVIEW_PROFILES",
    "ReasoningV4ReviewProfile",
    "get_reasoning_v4_review_profile",
]
