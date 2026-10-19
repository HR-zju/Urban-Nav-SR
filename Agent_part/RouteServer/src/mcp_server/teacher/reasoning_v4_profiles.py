"""Versioned independent-section teacher profiles for Reasoning V4.6."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from pydantic import BaseModel

from mcp_server.contracts.direction_profile import (
    DIRECTION_PROFILE_REGISTRY_V1,
    DIRECTION_PROFILE_REGISTRY_V2,
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
from mcp_server.teacher.reasoning_v4_image_policy import (
    REASONING_V4_ALL_VISIBLE_IMAGES_POLICY_ID,
    REASONING_V4_SECTION_IMAGE_POLICIES,
    SectionImagePolicy,
    get_reasoning_v4_section_image_policy,
)


_COMMON_POLICY = """\
You are an offline annotation teacher for a single-turn urban pedestrian
navigation model. Use only the exact section-visible images and structured
context attached to the current request. Different sections may receive
different image subsets. Never infer an omitted image modality or cite an image
role absent from image_order.

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
material to local navigation; a crosswalk is walkable_support and must not be
repeated as an entity; do not decide traversability or action. Prefer vehicle,
micromobility, and bicycle over make/model, parked-state, or plural variants.
traffic_context must be traffic_control_present, no_active_control_visible, or
unknown.
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
Fields: task_direction.status, task_direction.source, and optional
task_direction.direction.
Boundaries: direction must be a moving action_label; generic route-following
text alone does not determine left/right; a clearly oriented visible BEV route
overlay may support direction but never local safety. Use source=bev_route only
for that visible overlay, source=instruction only for an explicit directional
instruction. Use source=visual_goal when the instruction names a goal and the
current FPV unambiguously locates that goal; its direction comes from the image.
Resolve only the instructed goal, not any nearby object of the same category.
Use not_visible or insufficient_evidence for absent or ambiguous goals, with
source=none whenever status is not aligned.
Example:
{"task_direction":{"status":"aligned","source":"bev_route","direction":"left_20"}}""",
    "InteractionConstraintGrounding": """\
Responsibility: record current high-level behavior restrictions that alter the
short action.
Fields per constraint: kind, status, effect, affected_direction_labels.
Boundaries: default to an empty list; slow, yield, and stop are global and use
an empty affected list; only avoid names a non-empty subset. That subset means
directions requiring a local detour, not forbidden directions: selecting one
requires slow/detour, while stop safely satisfies avoid. Prefer the coarse kinds
crossing_interaction, social_interaction, local_obstacle, and traffic_control.
For traffic_control, establish which signal applies to the pedestrian's current
path and its current state. Distinguish pedestrian, cycle and vehicle signals.
A red signal for another movement does not establish a stop for this path.
History provides context; current applicability is determined at the anchor.
Use stop only with visible evidence of a current applicable stopping requirement;
retain it when supported even if another section suggests moving. Omit an
unsupported traffic restriction rather than infer it from a crossing instruction.
Do not add generic caution.
Example:
{"constraints":[{"kind":"crossing_interaction","status":"applicable","effect":"yield","affected_direction_labels":[]}]}""",
    "Decision": """\
Responsibility: make a visual grounding-conditioned short-horizon proposal from
the attached current FPV, optional current BEV, and the five validated grounding
sections.
Fields: selected_action_label, speed, response.
Boundaries: use only action_labels; stop requires stop/stop; moving actions
cannot use stop speed or response. Images may verify current geometry and route
alignment, but do not introduce an entity, traversability judgment, navigation
direction, or interaction constraint absent from the validated groundings.
Private strong direction and trajectory labels are unavailable and will be
audited later.
An applicable global stop constraint requires a stop/stop/stop proposal. If the
groundings support yielding rather than stopping, use the supported moving tuple.
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
        section_image_policy_id: Exact image-routing policy identity.
        section_image_policy_digest: Digest over image-routing behavior.
    """

    profile_id: str
    section_name: str
    model_type: type[BaseModel]
    output_json_schema: dict[str, Any]
    system_prompt: str
    policy_digest: str
    derivation_version: str
    section_image_policy_id: str
    section_image_policy_digest: str


def _profile(
    section_name: str,
    *,
    image_policy: SectionImagePolicy,
    direction_registry_version: str,
) -> ReasoningV4TeacherProfile:
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
    policy_slug = image_policy.policy_id.removeprefix(
        "reasoning_v4_section_images_"
    ).removesuffix("_v1")
    legacy_registry = (
        direction_registry_version == DIRECTION_PROFILE_REGISTRY_V1
    )
    profile_id = (
        f"reasoning_v4_teacher_{section_name.lower()}_"
        f"{policy_slug}_v{'10' if legacy_registry else '11'}"
    )
    derivation_version = (
        f"reasoning_v4_6_{section_name.lower()}_derivation_"
        f"v{'2' if legacy_registry else '3'}"
    )
    if section_name in {"NavigationContextGrounding", "InteractionConstraintGrounding", "Decision"}:
        profile_id += "_visual_goal_stop_v1"
        derivation_version += "_visual_goal_stop_v1"
    payload = {
        "profile_id": profile_id,
        "derivation_version": derivation_version,
        "trace_schema_version": REASONING_TRACE_V4_SCHEMA_VERSION,
        "contract_schema_digest": (
            reasoning_v4_contract_schema_digest()
        ),
        "action_profiles": approved_stage_a_direction_profiles(
            direction_registry_version
        ),
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
        "section_image_policy_id": image_policy.policy_id,
        "section_image_policy_digest": image_policy.policy_digest,
        "section_image_selectors": list(
            image_policy.selectors[section_name]
        ),
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
    return ReasoningV4TeacherProfile(
        profile_id=profile_id,
        section_name=section_name,
        model_type=model_type,
        output_json_schema=output_json_schema,
        system_prompt=system_prompt,
        policy_digest=digest,
        derivation_version=derivation_version,
        section_image_policy_id=image_policy.policy_id,
        section_image_policy_digest=image_policy.policy_digest,
    )


REASONING_V4_TEACHER_PROFILE_REGISTRY_BY_DIRECTION: Mapping[
    str,
    Mapping[str, Mapping[str, ReasoningV4TeacherProfile]],
] = MappingProxyType(
    {
        registry_version: MappingProxyType(
            {
                policy_id: MappingProxyType(
                    {
                        section_name: _profile(
                            section_name,
                            image_policy=image_policy,
                            direction_registry_version=(
                                registry_version
                            ),
                        )
                        for section_name in _SECTION_MODELS
                    }
                )
                for policy_id, image_policy in (
                    REASONING_V4_SECTION_IMAGE_POLICIES.items()
                )
            }
        )
        for registry_version in (
            DIRECTION_PROFILE_REGISTRY_V1,
            DIRECTION_PROFILE_REGISTRY_V2,
        )
    }
)
REASONING_V4_TEACHER_PROFILE_REGISTRY: Mapping[
    str,
    Mapping[str, ReasoningV4TeacherProfile],
] = REASONING_V4_TEACHER_PROFILE_REGISTRY_BY_DIRECTION[
    DIRECTION_PROFILE_REGISTRY_V1
]

REASONING_V4_TEACHER_PROFILES = (
    REASONING_V4_TEACHER_PROFILE_REGISTRY[
        REASONING_V4_ALL_VISIBLE_IMAGES_POLICY_ID
    ]
)


def get_reasoning_v4_teacher_profiles(
    section_image_policy_id: str,
    *,
    direction_registry_version: str = DIRECTION_PROFILE_REGISTRY_V1,
) -> Mapping[str, ReasoningV4TeacherProfile]:
    """Resolve section profiles for image and direction policies.

    Args:
        section_image_policy_id: Exact registered image-policy identity.
        direction_registry_version: Exact direction-registry snapshot.

    Returns:
        Immutable section-to-profile mapping.

    Raises:
        ValueError: If either policy is unsupported.
    """

    get_reasoning_v4_section_image_policy(section_image_policy_id)
    try:
        return REASONING_V4_TEACHER_PROFILE_REGISTRY_BY_DIRECTION[
            str(direction_registry_version)
        ][str(section_image_policy_id)]
    except KeyError as exc:
        raise ValueError(
            "unsupported Reasoning V4.6 direction registry: "
            f"{direction_registry_version!r}"
        ) from exc


def get_reasoning_v4_teacher_profile(
    section_name: str,
    *,
    section_image_policy_id: str = (
        REASONING_V4_ALL_VISIBLE_IMAGES_POLICY_ID
    ),
    direction_registry_version: str = DIRECTION_PROFILE_REGISTRY_V1,
) -> ReasoningV4TeacherProfile:
    """Resolve the canonical teacher profile.

    Args:
        section_name: Canonical six-section name.
        section_image_policy_id: Registered annotation image policy.
        direction_registry_version: Exact direction-registry snapshot.

    Returns:
        Immutable profile descriptor.

    Raises:
        ValueError: If the section is unsupported.
    """

    try:
        return get_reasoning_v4_teacher_profiles(
            section_image_policy_id,
            direction_registry_version=direction_registry_version,
        )[str(section_name)]
    except KeyError as exc:
        raise ValueError(
            "unsupported Reasoning V4 teacher section: "
            f"{section_name!r}"
        ) from exc


__all__ = [
    "REASONING_V4_TEACHER_PROFILES",
    "REASONING_V4_TEACHER_PROFILE_REGISTRY",
    "REASONING_V4_TEACHER_PROFILE_REGISTRY_BY_DIRECTION",
    "ReasoningV4TeacherProfile",
    "get_reasoning_v4_teacher_profile",
    "get_reasoning_v4_teacher_profiles",
]
