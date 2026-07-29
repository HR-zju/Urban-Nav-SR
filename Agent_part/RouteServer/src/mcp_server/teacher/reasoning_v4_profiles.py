"""Versioned independent-section teacher profiles for Reasoning V4.5."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from pydantic import BaseModel

from mcp_server.contracts.direction_profile import (
    approved_affordance_probe_profiles,
    approved_stage_a_direction_profiles,
)
from mcp_server.contracts.reasoning_v4 import (
    DecisionProposalV4,
    InteractionConstraintGroundingV4,
    LocalTraversabilityV4,
    NavigationContextGroundingV4,
    REASONING_TRACE_V4_SCHEMA_VERSION,
    REASONING_V4_EVIDENCE_POLICY_VERSION,
    REASONING_V4_PARSER_ID,
    REASONING_V4_RENDERER_ID,
    REASONING_V4_SECTION_DEFINITION_VERSION,
    REASONING_V4_VALIDATOR_ID,
    REASONING_V4_WIRE_FORMAT_VERSION,
    SceneGroundingV4,
    TaskGroundingV4,
    reasoning_v4_contract_schema_digest,
)


_COMMON_POLICY = """\
You are an offline annotation teacher for a single-turn urban pedestrian
navigation model. Use the complete prompt-visible multimodal input. Each section
independently reads the same input and answers only its own responsibility.

FPV is authoritative for visible walkable support, obstacles, clearance,
traffic controls, and immediate interaction risk. History FPV frames are extra
visual context only; do not create temporal labels, tracks, TTC, or calibrated
confidence. A prompt-visible BEV route overlay may support route direction and
upcoming topology when its orientation is clear. The absence of local safety
authority only prevents BEV from proving traversability, clearance, or collision
safety. Do not derive direction from GPS, route matching, route theta, pose,
pixel-goal geometry, or waypoint geometry.

Return exactly one JSON object matching the requested schema. Do not return
markdown, XML tags, explanations, another section, Finish, or readout tokens."""


_SECTION_POLICIES = {
    "TaskGrounding": """\
Responsibility: normalize an instruction not already covered by the deterministic
task registry.
Fields: objective, explicit requirements, and material ambiguities as concise
lower_snake_case codes.
Boundaries: do not add generic safety rules; do not describe the scene; use the
same taxonomy for semantically identical instructions.
Example:
{"objective":"follow_route","requirements":["route_adherence"],"ambiguities":[]}""",
    "SceneGrounding": """\
Responsibility: summarize the visible navigation scene.
Fields: scene_type, walkable_support, key_entities, and traffic_context as
coarse lower_snake_case categories.
Boundaries: categories are not instance IDs or boxes; include only entities
material to local navigation; do not decide traversability or action.
Example:
{"scene_type":"urban_sidewalk","walkable_support":"paved_sidewalk","key_entities":["pedestrian"],"traffic_context":"no_active_control_visible"}""",
    "LocalTraversability": """\
Responsibility: judge physical FPV executability for public
affordance_probe_labels.
Fields per probe: direction_label, affordance, clearance, and physical
limiting_factors.
Boundaries: never emit stop or labels outside the probe vocabulary; traffic
rules and navigation intent are not physical limiting factors; use unknown when
visibility is insufficient.
Example:
{"direction_affordances":[{"direction_label":"straight","affordance":"conditional","clearance":"narrow","limiting_factors":["dynamic_occupancy"]}]}""",
    "NavigationContextGrounding": """\
Responsibility: infer only the prompt-visible short-horizon task direction.
Fields: task_direction.status and optional task_direction.direction.
Boundaries: direction must be a moving action_label; generic route-following
text alone does not determine left/right; a clearly oriented visible BEV route
overlay may support direction but never local safety.
Example:
{"task_direction":{"status":"aligned","direction":"left_20"}}""",
    "InteractionConstraintGrounding": """\
Responsibility: record current high-level behavior restrictions that alter the
short action.
Fields per constraint: kind, status, effect, affected_direction_labels.
Boundaries: default to an empty list; stop is global and uses an empty affected
list; avoid names a non-empty affected subset; do not add generic caution.
Example:
{"constraints":[{"kind":"pedestrian_crossing","status":"applicable","effect":"yield","affected_direction_labels":[]}]}""",
    "Decision": """\
Responsibility: make a blind short-horizon proposal from the complete visible
input and the five validated grounding sections.
Fields: selected_action_label, speed, response.
Boundaries: use only action_labels; stop requires stop/stop; moving actions
cannot use stop speed or response. Private strong direction and trajectory
labels are unavailable and will be audited later.
Example:
{"selected_action_label":"straight","speed":"slow","response":"yield"}""",
}

_SECTION_MODELS: dict[str, type[BaseModel]] = {
    "TaskGrounding": TaskGroundingV4,
    "SceneGrounding": SceneGroundingV4,
    "LocalTraversability": LocalTraversabilityV4,
    "NavigationContextGrounding": NavigationContextGroundingV4,
    "InteractionConstraintGrounding": (
        InteractionConstraintGroundingV4
    ),
    "Decision": DecisionProposalV4,
}


@dataclass(frozen=True)
class ReasoningV4TeacherProfile:
    """Describe one executable section annotation policy.

    Args:
        profile_id: Stable profile written to stage provenance.
        section_name: Canonical section name.
        model_type: Pydantic model accepted from the teacher.
        output_json_schema: Exact section JSON schema.
        system_prompt: Exact provider-side policy.
        policy_digest: Digest over all behavior-defining components.
        derivation_version: Exact annotation derivation version.
    """

    profile_id: str
    section_name: str
    model_type: type[BaseModel]
    output_json_schema: dict[str, Any]
    system_prompt: str
    policy_digest: str
    derivation_version: str


def _profile(section_name: str) -> ReasoningV4TeacherProfile:
    """Build one immutable section profile."""

    model_type = _SECTION_MODELS[section_name]
    output_json_schema = json.loads(
        json.dumps(
            model_type.model_json_schema(),
            ensure_ascii=False,
        )
    )
    system_prompt = (
        f"{_COMMON_POLICY}\n\nSECTION — {section_name}\n"
        f"{_SECTION_POLICIES[section_name]}"
    )
    profile_id = (
        f"reasoning_v4_teacher_{section_name.lower()}_v8"
    )
    derivation_version = (
        f"reasoning_v4_5_{section_name.lower()}_derivation_v1"
    )
    payload = {
        "profile_id": profile_id,
        "derivation_version": derivation_version,
        "trace_schema_version": REASONING_TRACE_V4_SCHEMA_VERSION,
        "contract_schema_digest": (
            reasoning_v4_contract_schema_digest()
        ),
        "action_profiles": approved_stage_a_direction_profiles(),
        "affordance_probe_profiles": (
            approved_affordance_probe_profiles()
        ),
        "wire_format_version": REASONING_V4_WIRE_FORMAT_VERSION,
        "section_definition_version": (
            REASONING_V4_SECTION_DEFINITION_VERSION
        ),
        "evidence_policy_version": (
            REASONING_V4_EVIDENCE_POLICY_VERSION
        ),
        "parser_id": REASONING_V4_PARSER_ID,
        "renderer_id": REASONING_V4_RENDERER_ID,
        "validator_id": REASONING_V4_VALIDATOR_ID,
        "section_name": section_name,
        "section_schema": output_json_schema,
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
    return ReasoningV4TeacherProfile(
        profile_id=profile_id,
        section_name=section_name,
        model_type=model_type,
        output_json_schema=output_json_schema,
        system_prompt=system_prompt,
        policy_digest=digest,
        derivation_version=derivation_version,
    )


REASONING_V4_TEACHER_PROFILES: Mapping[
    str,
    ReasoningV4TeacherProfile,
] = MappingProxyType(
    {
        section_name: _profile(section_name)
        for section_name in _SECTION_MODELS
    }
)


def get_reasoning_v4_teacher_profile(
    section_name: str,
) -> ReasoningV4TeacherProfile:
    """Resolve the canonical teacher profile.

    Args:
        section_name: Canonical six-section name.

    Returns:
        Immutable profile descriptor.

    Raises:
        ValueError: If the section is unsupported.
    """

    try:
        return REASONING_V4_TEACHER_PROFILES[
            str(section_name)
        ]
    except KeyError as exc:
        raise ValueError(
            "unsupported Reasoning V4 teacher section: "
            f"{section_name!r}"
        ) from exc


__all__ = [
    "REASONING_V4_TEACHER_PROFILES",
    "ReasoningV4TeacherProfile",
    "get_reasoning_v4_teacher_profile",
]
