"""Versioned student prompt policies for Reasoning V4 SFT views."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from mcp_server.contracts.reasoning_v4 import (
    DecisionV4,
    FinishV4,
    InteractionConstraintGroundingV4,
    LocalTraversabilityV4,
    NavigationContextGroundingV4,
    REASONING_TRAINING_VIEW_V4_SCHEMA_VERSION,
    REASONING_TRACE_V4_SCHEMA_VERSION,
    REASONING_V4_EVIDENCE_POLICY_VERSION,
    REASONING_V4_PARSER_ID,
    REASONING_V4_RENDERER_ID,
    REASONING_V4_SECTION_DEFINITION_VERSION,
    REASONING_V4_VALIDATOR_ID,
    REASONING_V4_WIRE_FORMAT_VERSION,
    ReasoningTargetV4,
    SceneGroundingV4,
    TaskGroundingV4,
    reasoning_v4_contract_schema_digest,
)


_BASE_POLICY = """\
You are the single-turn local navigation reasoning model for an urban pedestrian
assistance robot. Use only the supplied instruction, ordered images, optional
prompt-visible goal, waypoint, or BEV context, and evidence catalog. FPV is the
only authority for immediate local safety. Non-FPV navigation context may express
task direction but cannot prove unseen traversability.

Never use or claim GPS, pose, calibration, route matching, route theta, future
trajectory, runtime task phase, masks, boxes, instance tracking, TTC, or calibrated
confidence. Raw evidence_refs must copy exact evidence_catalog refs. Semantic refs
are sample-local links, not detector or tracker truth. Use empty lists or explicit
unknown states when evidence is insufficient.

The field responsibilities are fixed:
- TaskGrounding normalizes the visible task goal, explicit requirements, and
  material ambiguity; it does not describe the scene or choose an action.
- SceneGrounding records directly visible navigation-relevant entities and traffic
  controls; it does not claim traversability or interaction policy.
- LocalTraversability describes FPV-visible local regions, coarse surface,
  traversability state, clearance category, and visible limiting factors.
- NavigationContextGrounding aligns actual task/goal/waypoint/BEV sources with the
  FPV scene and proposes a coarse task direction without deriving local safety from
  non-FPV context.
- InteractionConstraintGrounding records only evidence-supported social, traffic,
  accessibility, or local-safety constraints that affect local candidate regions.
- Decision selects one short-horizon semantic action using raw evidence and
  resolvable semantic support. Stage-A has no reliable arrival state, so it must
  not emit decision_type=finish.

Do not output tool calls, dialogue turns, agent-loop state, or private supervision."""

_FULL_OUTPUT = """\
Return exactly one canonical document: a <think> block containing
TaskGrounding, SceneGrounding, LocalTraversability,
NavigationContextGrounding, InteractionConstraintGrounding, and Decision in that
order, followed by one <finish> JSON projection. If the prompt requests System1
readout anchors, append exactly one <readout> block after <finish>."""

_GROUNDING_OUTPUT = """\
Return exactly one tagged JSON object for the requested grounding section and no
other section, Finish, readout, markdown, or prose."""

_DECISION_OUTPUT = """\
The five grounding sections supplied in the prompt are validated conditioning
context. Return exactly one <Decision> JSON object. Resolve selected_region_ref and
support_refs against that context, while evidence_refs must still cite raw catalog
entries. Do not output Finish, direction classes, trajectory, markdown, or prose."""

_SECTION_MODEL_TYPES = (
    ("TaskGrounding", TaskGroundingV4),
    ("SceneGrounding", SceneGroundingV4),
    ("LocalTraversability", LocalTraversabilityV4),
    ("NavigationContextGrounding", NavigationContextGroundingV4),
    (
        "InteractionConstraintGrounding",
        InteractionConstraintGroundingV4,
    ),
    ("Decision", DecisionV4),
    ("finish", FinishV4),
)

_ENUM_GUIDANCE = {
    "TaskGroundingV4.objective": (
        "task specification, never the current action: approach_goal=approach "
        "an explicit target; follow_instruction=execute visible language; "
        "follow_route=follow explicit route/BEV context; "
        "follow_waypoint=follow an explicit waypoint; "
        "local_navigation=no explicit long-range target; unknown=insufficient "
        "task evidence"
    ),
    "TaskGoalV4.goal_type": (
        "which visible input defines the goal; none is legal only when no "
        "explicit goal exists"
    ),
    "TaskGoalV4.specification": (
        "specified=clear input definition; ambiguous=conflicting definitions; "
        "underspecified=incomplete definition; absent=no explicit goal"
    ),
    "TaskRequirementV4.polarity": (
        "required/preferred are positive constraints; avoid/forbidden are "
        "negative constraints"
    ),
    "SceneEntityV4.lateral_position": (
        "coarse image-relative horizontal position, not metric geometry"
    ),
    "SceneEntityV4.depth_position": (
        "coarse visual depth only; never metric distance"
    ),
    "TrafficControlV4.lateral_position": (
        "coarse image-relative horizontal position, not metric geometry"
    ),
    "TraversabilityRegionV4.lateral_position": (
        "coarse image-relative region position; do not force a fixed three-way "
        "partition"
    ),
    "TraversabilityRegionV4.state": (
        "direct=immediately usable; conditional=usable with behavioral "
        "constraint; blocked=visibly obstructed; forbidden=not a legal walking "
        "region; unknown=insufficient FPV evidence"
    ),
    "TraversabilityRegionV4.clearance": (
        "sufficient/narrow/insufficient/unknown are coarse visual categories, "
        "not measured widths"
    ),
    "GoalAlignmentV4.status": (
        "resolved=one target; ambiguous=at least two candidates; "
        "not_visible=defined goal not grounded in FPV; "
        "insufficient_evidence=cannot judge"
    ),
    "NavigationSourceAlignmentV4.local_effect": (
        "semantic local implication of one actual task/navigation source"
    ),
    "NavigationSourceAlignmentV4.status": (
        "supports/conflicts compare the source with visible context; "
        "not_visible/insufficient do not claim alignment"
    ),
    "TaskDirectionV4.desired_motion": (
        "coarse task-requested direction before local safety modifies action"
    ),
    "TaskDirectionV4.status": (
        "aligned=task sources agree; conflict=sources disagree; "
        "not_visible=required visual grounding absent; "
        "insufficient_evidence=cannot infer direction"
    ),
    "InteractionConstraintV4.kind": (
        "source family: social, traffic, accessibility, or local_safety"
    ),
    "InteractionConstraintV4.relation": (
        "observable source-to-candidate-region relation; crossing/approaching "
        "requires distinct multi-frame FPV"
    ),
    "InteractionConstraintV4.temporal_basis": (
        "single_frame, distinct multi_frame, visible static_control, or explicit "
        "instruction evidence"
    ),
    "InteractionConstraintV4.status": (
        "applicable=evidence supports current use; uncertain=evidence is not "
        "strong enough for a definite constraint"
    ),
    "InteractionConstraintV4.effect": (
        "behavioral consequence of this constraint, not a duplicated scene fact"
    ),
    "DecisionV4.decision_type": (
        "navigate=produce local motion; stop=hold without task completion; "
        "request_user=material task ambiguity; request_replan=no safe/current "
        "navigation solution; finish is forbidden in Stage-A"
    ),
    "DecisionV4.motion": (
        "coarse semantic action only; fine direction remains private sidecar "
        "supervision projected into Finish"
    ),
    "DecisionV4.speed": (
        "normal/slow/stop are semantic speed modes, not metric velocity"
    ),
    "DecisionV4.social_action": (
        "interaction behavior; use none only when no social action is needed and "
        "unknown only when evidence cannot resolve it"
    ),
    "DecisionV4.fallback": (
        "conservative response if evidence worsens or the selected action fails"
    ),
    "FinishV4.decision_type": "must exactly equal Decision.decision_type",
    "FinishV4.motion": "must exactly equal Decision.motion",
    "FinishV4.speed": "must exactly equal Decision.speed",
    "FinishV4.social_action": "must exactly equal Decision.social_action",
}


def _canonical_field_contract_text() -> str:
    """Render field skeletons and closed enums from the canonical schema."""

    schema = ReasoningTargetV4.model_json_schema()
    definitions = schema.get("$defs", {})
    lines = [
        "Canonical tagged object skeletons (* means schema-required; preserve "
        "every exact field name):"
    ]
    for section_name, model_type in _SECTION_MODEL_TYPES:
        definition = definitions.get(model_type.__name__, {})
        lines.append(
            f"- {section_name} -> {model_type.__name__}"
        )
    lines.append("Canonical nested object fields:")
    for model_name, definition in sorted(definitions.items()):
        properties = definition.get("properties") or {}
        required = set(definition.get("required") or ())
        fields = [
            f"{field_name}{'*' if field_name in required else ''}"
            for field_name in properties
        ]
        lines.append(f"- {model_name}: {', '.join(fields)}")
    lines.append(
        "Closed enum vocabulary (all ReasonCode/category/description fields "
        "remain constrained open text):"
    )
    for model_name, definition in sorted(definitions.items()):
        properties = definition.get("properties")
        if not isinstance(properties, dict):
            continue
        for field_name, field_schema in properties.items():
            values = _schema_enum_values(field_schema)
            if values:
                line = (
                    f"- {model_name}.{field_name}: "
                    + " | ".join(str(value) for value in values)
                )
                guidance = _ENUM_GUIDANCE.get(
                    f"{model_name}.{field_name}"
                )
                if guidance:
                    line += f". Meaning: {guidance}"
                lines.append(line)
    return "\n".join(lines)


def _schema_enum_values(value: Any) -> list[Any]:
    """Extract one direct or nullable enum list from JSON Schema."""

    if not isinstance(value, dict):
        return []
    direct = value.get("enum")
    if isinstance(direct, list):
        return direct
    for branch in value.get("anyOf", []):
        if isinstance(branch, dict) and isinstance(
            branch.get("enum"),
            list,
        ):
            return branch["enum"]
    return []


_FIELD_CONTRACT = _canonical_field_contract_text()


@dataclass(frozen=True)
class ReasoningV4StudentProfile:
    """Describe one cross-repository V4 student prompt profile."""

    profile_id: str
    view_type: str
    section_name: str | None
    system_prompt: str
    policy_digest: str


def _student_profile(
    *,
    profile_id: str,
    view_type: str,
    section_name: str | None,
    output_policy: str,
) -> ReasoningV4StudentProfile:
    """Create one deterministic student profile and its parity digest."""

    system_prompt = (
        f"{_BASE_POLICY}\n\n{_FIELD_CONTRACT}\n\n{output_policy}"
    )
    components = (
        profile_id,
        REASONING_TRACE_V4_SCHEMA_VERSION,
        REASONING_TRAINING_VIEW_V4_SCHEMA_VERSION,
        REASONING_V4_WIRE_FORMAT_VERSION,
        REASONING_V4_SECTION_DEFINITION_VERSION,
        REASONING_V4_EVIDENCE_POLICY_VERSION,
        REASONING_V4_RENDERER_ID,
        REASONING_V4_PARSER_ID,
        REASONING_V4_VALIDATOR_ID,
        view_type,
        section_name or "",
        system_prompt,
        reasoning_v4_contract_schema_digest(),
        json.dumps(
            ReasoningTargetV4.model_json_schema(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "build_reasoning_v4_user_prompt",
    )
    digest = hashlib.sha256("\n".join(components).encode("utf-8")).hexdigest()
    return ReasoningV4StudentProfile(
        profile_id=profile_id,
        view_type=view_type,
        section_name=section_name,
        system_prompt=system_prompt,
        policy_digest=digest,
    )


STUDENT_FULL_COT_PROFILE_V4 = _student_profile(
    profile_id="urbannav_reasoning_v4_0_full_cot_sft_v1",
    view_type="full_cot_sft",
    section_name=None,
    output_policy=_FULL_OUTPUT,
)
STUDENT_DECISION_PROFILE_V4 = _student_profile(
    profile_id="urbannav_reasoning_v4_0_decision_section_sft_v1",
    view_type="decision_section_sft",
    section_name="Decision",
    output_policy=_DECISION_OUTPUT,
)
STUDENT_GROUNDING_PROFILES_V4: Mapping[
    str, ReasoningV4StudentProfile
] = MappingProxyType(
    {
        section_name: _student_profile(
            profile_id=(
                "urbannav_reasoning_v4_0_"
                f"{section_name.lower()}_section_sft_v1"
            ),
            view_type="grounding_section_sft",
            section_name=section_name,
            output_policy=(
                f"Requested grounding section: {section_name}.\n"
                f"{_GROUNDING_OUTPUT}"
            ),
        )
        for section_name in (
            "TaskGrounding",
            "SceneGrounding",
            "LocalTraversability",
            "NavigationContextGrounding",
            "InteractionConstraintGrounding",
        )
    }
)

PROMPT_PROFILE_REGISTRY_V4: Mapping[
    str, ReasoningV4StudentProfile
] = MappingProxyType(
    {
        STUDENT_FULL_COT_PROFILE_V4.profile_id: STUDENT_FULL_COT_PROFILE_V4,
        STUDENT_DECISION_PROFILE_V4.profile_id: STUDENT_DECISION_PROFILE_V4,
        **{
            profile.profile_id: profile
            for profile in STUDENT_GROUNDING_PROFILES_V4.values()
        },
    }
)


def get_prompt_profile_v4(
    profile_id: str,
) -> ReasoningV4StudentProfile:
    """Resolve one supported V4 student prompt profile.

    Args:
        profile_id: Stable profile id carried by a training view.

    Returns:
        Registered immutable profile.

    Raises:
        ValueError: If the profile is unsupported.
    """

    try:
        return PROMPT_PROFILE_REGISTRY_V4[str(profile_id)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported Reasoning V4 prompt profile: {profile_id!r}"
        ) from exc
