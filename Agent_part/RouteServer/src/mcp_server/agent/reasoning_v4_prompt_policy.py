"""Versioned compact FullCoT student prompt policy for Reasoning V4.6."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from mcp_server.contracts.direction_profile import (
    DIRECTION_PROFILE_REGISTRY_V1,
    DIRECTION_PROFILE_REGISTRY_V2,
    approved_affordance_probe_profiles,
    approved_stage_a_direction_profiles,
)
from mcp_server.contracts.reasoning_v4 import (
    REASONING_TRAINING_VIEW_V4_SCHEMA_VERSION,
    REASONING_TRACE_V4_SCHEMA_VERSION,
    REASONING_V4_COMPACT_FULL_COT_RENDERER_ID,
    REASONING_V4_EVIDENCE_POLICY_VERSION,
    REASONING_V4_PARSER_ID,
    REASONING_V4_RENDERER_ID,
    REASONING_V4_SECTION_DEFINITION_VERSION,
    REASONING_V4_VALIDATOR_ID,
    REASONING_V4_WIRE_FORMAT_VERSION,
    ReasoningTargetV4,
    reasoning_v4_contract_schema_digest,
)


_SYSTEM_PROMPT = """\
You are the System2 policy of a single-turn urban pedestrian navigation model.
Read the complete prompt-visible multimodal input and produce compact,
short-horizon navigation reasoning.

The six sections independently use the same complete input:
1. TaskGrounding normalizes the task into objective, explicit requirements, and
   material ambiguities.
2. SceneGrounding summarizes scene type, walkable support, key entity
   categories, and traffic context without object IDs or boxes. A crosswalk is
   walkable support, not an entity.
3. LocalTraversability judges physical FPV executability only for public
   affordance_probe_labels. limiting_factors contain physical causes only.
4. NavigationContextGrounding gives only a visible moving task direction, its
   minimal visible source, or a fail-closed status with source=none.
5. InteractionConstraintGrounding records current high-level restrictions that
   require slow, yield, stop, or avoidance.
6. Decision gives one action tuple: selected_action_label, speed, response.

FPV is authoritative for visible support, obstacles, clearance, controls, and
immediate interaction risk. History FPV frames are input context only; do not
output temporal labels, tracks, TTC, or confidence. A prompt-visible BEV route
overlay may support route direction and upcoming topology when its orientation
is clear. The absence of local safety authority only prevents BEV from proving
traversability, clearance, or collision safety. Never derive direction from GPS,
route matching, route theta, pose, pixel-goal geometry, or waypoint geometry.
Set task_direction.source=bev_route only when that oriented overlay supports the
direction. Set source=instruction only when the instruction itself explicitly
states a direction. Generic route-following text is not directional evidence.

Use lower_snake_case open codes where the schema uses ReasonCode. Use only the
provided action_labels and affordance_probe_labels for direction fields. Do not
output evidence refs, source alignments, object/constraint IDs, geometry,
route-progress, motion, social_action, fallback, critical uncertainties,
schema bookkeeping, Finish, or readout tokens.

Return exactly:
<think>
<TaskGrounding>{"objective":"...","requirements":[],"ambiguities":[]}</TaskGrounding>
<SceneGrounding>{"scene_type":"...","walkable_support":"...","key_entities":[],"traffic_context":"..."}</SceneGrounding>
<LocalTraversability>{"direction_affordances":[]}</LocalTraversability>
<NavigationContextGrounding>{"task_direction":{"status":"...","source":"...","direction":"..."}}</NavigationContextGrounding>
<InteractionConstraintGrounding>{"constraints":[]}</InteractionConstraintGrounding>
<Decision>{"selected_action_label":"...","speed":"...","response":"..."}</Decision>
</think>

Preserve this exact order. Omit nullable direction and use source=none when
status is not aligned. The stop action uses speed=stop and response=stop. Moving
actions never use stop speed or response. Only avoid may name affected direction
labels; slow, yield, and stop are global and must use an empty list. Add no
prose, markdown, comments, or extra tags."""


@dataclass(frozen=True)
class ReasoningV4StudentProfile:
    """Describe one cross-repository V4.6 student policy.

    Args:
        profile_id: Stable profile identity.
        view_type: Exact supported view type.
        section_name: Always ``None`` for compact FullCoT.
        system_prompt: Exact system prompt.
        policy_digest: Digest over all behavior-defining components.
    """

    profile_id: str
    view_type: str
    section_name: str | None
    system_prompt: str
    policy_digest: str


def _full_cot_profile(
    direction_registry_version: str,
) -> ReasoningV4StudentProfile:
    """Build one V4.6 student prompt bound to a direction registry.

    Args:
        direction_registry_version: Exact direction-registry snapshot.

    Returns:
        Immutable student prompt profile.
    """

    legacy_registry = (
        direction_registry_version == DIRECTION_PROFILE_REGISTRY_V1
    )
    profile_id = (
        "urbannav_reasoning_v4_6_compact_full_cot_sft_v1"
        if legacy_registry
        else "urbannav_reasoning_v4_6_compact_full_cot_sft_v2"
    )
    payload = {
        "profile_id": profile_id,
        "trace_schema_version": REASONING_TRACE_V4_SCHEMA_VERSION,
        "training_view_schema_version": (
            REASONING_TRAINING_VIEW_V4_SCHEMA_VERSION
        ),
        "wire_format_version": REASONING_V4_WIRE_FORMAT_VERSION,
        "section_definition_version": (
            REASONING_V4_SECTION_DEFINITION_VERSION
        ),
        "evidence_policy_version": (
            REASONING_V4_EVIDENCE_POLICY_VERSION
        ),
        "renderer_id": REASONING_V4_RENDERER_ID,
        "compact_renderer_id": (
            REASONING_V4_COMPACT_FULL_COT_RENDERER_ID
        ),
        "parser_id": REASONING_V4_PARSER_ID,
        "validator_id": REASONING_V4_VALIDATOR_ID,
        "contract_schema_digest": (
            reasoning_v4_contract_schema_digest()
        ),
        "target_schema": ReasoningTargetV4.model_json_schema(),
        "action_profiles": approved_stage_a_direction_profiles(
            direction_registry_version
        ),
        "affordance_probe_profiles": (
            approved_affordance_probe_profiles()
        ),
        "view_type": "full_cot_sft",
        "section_name": None,
        "system_prompt": _SYSTEM_PROMPT,
        "user_prompt_builder": "build_reasoning_v4_user_prompt_v4_6",
    }
    if not legacy_registry:
        payload["direction_registry_version"] = (
            direction_registry_version
        )
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return ReasoningV4StudentProfile(
        profile_id=profile_id,
        view_type="full_cot_sft",
        section_name=None,
        system_prompt=_SYSTEM_PROMPT,
        policy_digest=digest,
    )


STUDENT_FULL_COT_PROFILE_V4 = _full_cot_profile(
    DIRECTION_PROFILE_REGISTRY_V1
)
STUDENT_FULL_COT_PROFILE_V4_REGISTRY_V2 = _full_cot_profile(
    DIRECTION_PROFILE_REGISTRY_V2
)
PROMPT_PROFILE_REGISTRY_V4: Mapping[
    str,
    ReasoningV4StudentProfile,
] = MappingProxyType(
    {
        STUDENT_FULL_COT_PROFILE_V4.profile_id: (
            STUDENT_FULL_COT_PROFILE_V4
        ),
        STUDENT_FULL_COT_PROFILE_V4_REGISTRY_V2.profile_id: (
            STUDENT_FULL_COT_PROFILE_V4_REGISTRY_V2
        ),
    }
)


def get_prompt_profile_v4(
    profile_id: str,
) -> ReasoningV4StudentProfile:
    """Resolve the sole supported V4.6 student profile.

    Args:
        profile_id: Exact profile identity.

    Returns:
        Immutable student profile.

    Raises:
        ValueError: If the profile is unsupported.
    """

    try:
        return PROMPT_PROFILE_REGISTRY_V4[str(profile_id)]
    except KeyError as exc:
        raise ValueError(
            "unsupported Reasoning V4.6 prompt profile: "
            f"{profile_id!r}"
        ) from exc


def student_prompt_profile_for_direction_registry_v4(
    direction_registry_version: str,
) -> ReasoningV4StudentProfile:
    """Resolve the student prompt bound to a direction registry.

    Args:
        direction_registry_version: Exact canonical registry snapshot.

    Returns:
        Immutable student prompt profile.

    Raises:
        ValueError: If the registry has no compatible prompt profile.
    """

    mapping = {
        DIRECTION_PROFILE_REGISTRY_V1: STUDENT_FULL_COT_PROFILE_V4,
        DIRECTION_PROFILE_REGISTRY_V2: (
            STUDENT_FULL_COT_PROFILE_V4_REGISTRY_V2
        ),
    }
    try:
        return mapping[str(direction_registry_version)]
    except KeyError as exc:
        raise ValueError(
            "unsupported Reasoning V4.6 direction registry: "
            f"{direction_registry_version!r}"
        ) from exc


__all__ = [
    "PROMPT_PROFILE_REGISTRY_V4",
    "STUDENT_FULL_COT_PROFILE_V4",
    "STUDENT_FULL_COT_PROFILE_V4_REGISTRY_V2",
    "ReasoningV4StudentProfile",
    "get_prompt_profile_v4",
    "student_prompt_profile_for_direction_registry_v4",
]
