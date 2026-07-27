"""Versioned student prompt policies for Reasoning V4.2 SFT views."""

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
    STAGE_A_DECISION_TYPES_V4,
    SceneGroundingV4,
    TaskGroundingV4,
    reasoning_v4_contract_schema_digest,
)


_BASE_POLICY = """\
You are the System2 policy of a single-turn urban pedestrian navigation model.
Use the complete prompt-visible multimodal observation to produce grounded,
short-horizon navigation reasoning.

Evidence contract:
- Use only the instruction, ordered FPV images, optional navigation context,
  evidence catalog, and public direction_output_space in the current prompt.
- FPV is the only authority for immediate visibility, occupancy, clearance,
  local traversability, traffic controls, and interaction risk.
- Raw evidence_refs must exactly match evidence-catalog refs.
- Public direction labels define the answer vocabulary, not the selected action.
- category, support_type, source_type, source_state, uncertainty_code, and other
  ReasonCode fields are open lower_snake_case descriptions.
- evidence_status describes support/ambiguity/insufficiency. It is not calibrated
  confidence, a probability, a loss weight, or a runtime score.

Grounding contract:
- The five grounding sections independently read the same complete visible input.
- A prompt-visible BEV route overlay may support route direction and upcoming
  topology when its orientation is clear. The absence of local safety authority
  only prevents BEV from proving traversability, clearance, or collision safety.
- Do not infer a direction from route matching, route theta, GPS, pose,
  pixel-goal geometry, or waypoint geometry.
- Do not invent masks, boxes, tracks, TTC, metric clearance, hidden object IDs,
  geometric areas, route segments, trajectory corridors, or numeric confidence.
- Scene refs are neutral counters such as entity.e01 and control.c01; semantics
  belong in open category fields.
- supported scene categories use one category and no candidates; ambiguous uses
  no category and at least two candidates; insufficient uses neither.
- LocalTraversability judges public direction labels directly. It does not create
  geometric areas or choose the strong direction target.
- source_alignments distinguish task_intent, route_intent, and
  local_safety_bias. task_direction contains only task/route movement labels,
  never stop.
- Pixel-goal and waypoint direction alignments remain insufficient until their
  frame and derivation contracts are frozen.
- Interaction constraints use open source_type/source_state. An empty
  affected_direction_labels list means global; only an affected subset is
  enumerated. Global stop constraints use effect=stop, never all_moving or an
  expansion of every non-stop label. Dynamic crossing/approach claims require
  distinct ordered FPV images.

Decision contract:
- Stage-A allows only decision_type=navigate or stop.
- navigation_intent_direction_label is the non-stop task/route direction.
- selected_action_label is the immediate action and may be stop.
- navigate uses speed=normal/slow. selected_action_label=stop requires
  decision_type=stop, motion=stop, speed=stop, social_action=wait, and
  fallback=stop.
- A stop action must cite an applicable global stop/yield constraint, all
  blocked task-intent directions, or a material stop-policy uncertainty. Do not
  stop only because a destination is absent when local continuation is visible.
- Critical uncertainties are ordinary weak language supervision with a closed
  resolution policy; never emit numeric confidence or cost.
- Decision.motion is derived only from selected_action_label.
- Finish.demonstrated_direction.label is the deterministic trajectory-derived
  label. It is distinct from navigation intent and the immediate safe action.
- Private direction supervision and continuous trajectories must never appear
  in the student prompt or reasoning evidence."""

_FULL_OUTPUT = """\
Return exactly:
<think>
<TaskGrounding>{...}</TaskGrounding>
<SceneGrounding>{...}</SceneGrounding>
<LocalTraversability>{...}</LocalTraversability>
<NavigationContextGrounding>{...}</NavigationContextGrounding>
<InteractionConstraintGrounding>{...}</InteractionConstraintGrounding>
<Decision>{...}</Decision>
</think>
<finish>{...}</finish>

Use canonical JSON field names and section order. Finish must project its
Decision fields exactly and carry demonstrated_direction separately. Do not add
prose, markdown, comments, or extra tags. Append one
<readout> block only when the prompt explicitly supplies System1 readout tokens;
the block must contain exactly those tokens and no natural language."""

_GROUNDING_OUTPUT = """\
Return exactly one tagged JSON object for the requested grounding section.
Do not output another section, Decision, finish, readout, markdown, or prose."""

_DECISION_OUTPUT = """\
The prompt includes the original visible input and five validated grounding
sections. Return exactly one canonical <Decision>{...}</Decision> object.
support_refs must resolve to supplied semantic objects; evidence_refs must cite
raw catalog evidence. Distinguish navigation intent from the immediate selected
action. The student prompt never contains a private demonstrated direction or
trajectory target."""

_SECTION_MODEL_TYPES = (
    ("TaskGrounding", TaskGroundingV4),
    ("SceneGrounding", SceneGroundingV4),
    ("LocalTraversability", LocalTraversabilityV4),
    (
        "NavigationContextGrounding",
        NavigationContextGroundingV4,
    ),
    (
        "InteractionConstraintGrounding",
        InteractionConstraintGroundingV4,
    ),
    ("Decision", DecisionV4),
    ("finish", FinishV4),
)

_STAGE_A_ENUM_OVERRIDES = {
    "DecisionV4.decision_type": list(STAGE_A_DECISION_TYPES_V4),
    "FinishV4.decision_type": list(STAGE_A_DECISION_TYPES_V4),
}


def _canonical_field_contract_text() -> str:
    """Render exact field names and closed enums from the schema."""

    definitions = ReasoningTargetV4.model_json_schema().get(
        "$defs",
        {},
    )
    lines = [
        "Canonical section models:",
        *(
            f"- {section_name}: {model_type.__name__}"
            for section_name, model_type in _SECTION_MODEL_TYPES
        ),
        "Canonical nested fields (* is required):",
    ]
    for model_name, definition in sorted(definitions.items()):
        properties = definition.get("properties") or {}
        required = set(definition.get("required") or ())
        fields = [
            f"{name}{'*' if name in required else ''}"
            for name in properties
        ]
        lines.append(f"- {model_name}: {', '.join(fields)}")
    lines.append(
        "Closed enums (ReasonCode and description fields remain open):"
    )
    for model_name, definition in sorted(definitions.items()):
        properties = definition.get("properties")
        if not isinstance(properties, dict):
            continue
        for field_name, field_schema in properties.items():
            qualified = f"{model_name}.{field_name}"
            values = _STAGE_A_ENUM_OVERRIDES.get(
                qualified,
                _schema_enum_values(field_schema),
            )
            if values:
                lines.append(
                    f"- {qualified}: "
                    + " | ".join(str(value) for value in values)
                )
    return "\n".join(lines)


def _schema_enum_values(value: Any) -> list[Any]:
    """Extract one direct or nullable enum list from JSON Schema."""

    if not isinstance(value, dict):
        return []
    if isinstance(value.get("enum"), list):
        return value["enum"]
    for branch in value.get("anyOf", []):
        if (
            isinstance(branch, dict)
            and isinstance(branch.get("enum"), list)
        ):
            return branch["enum"]
    return []


_FIELD_CONTRACT = _canonical_field_contract_text()


@dataclass(frozen=True)
class ReasoningV4StudentProfile:
    """Describe one cross-repository V4.2 student prompt profile."""

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
    """Create one deterministic student profile and parity digest."""

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
    digest = hashlib.sha256(
        "\n".join(components).encode("utf-8")
    ).hexdigest()
    return ReasoningV4StudentProfile(
        profile_id=profile_id,
        view_type=view_type,
        section_name=section_name,
        system_prompt=system_prompt,
        policy_digest=digest,
    )


STUDENT_FULL_COT_PROFILE_V4 = _student_profile(
    profile_id="urbannav_reasoning_v4_2_full_cot_sft_v1",
    view_type="full_cot_sft",
    section_name=None,
    output_policy=_FULL_OUTPUT,
)
STUDENT_DECISION_PROFILE_V4 = _student_profile(
    profile_id="urbannav_reasoning_v4_2_decision_section_sft_v1",
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
                "urbannav_reasoning_v4_2_"
                f"{section_name.lower()}_section_sft_v1"
            ),
            view_type="grounding_section_sft",
            section_name=section_name,
            output_policy=(
                f"Requested section: {section_name}.\n"
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
        STUDENT_FULL_COT_PROFILE_V4.profile_id: (
            STUDENT_FULL_COT_PROFILE_V4
        ),
        STUDENT_DECISION_PROFILE_V4.profile_id: (
            STUDENT_DECISION_PROFILE_V4
        ),
        **{
            profile.profile_id: profile
            for profile in STUDENT_GROUNDING_PROFILES_V4.values()
        },
    }
)


def get_prompt_profile_v4(
    profile_id: str,
) -> ReasoningV4StudentProfile:
    """Resolve one supported V4.2 student prompt profile."""

    try:
        return PROMPT_PROFILE_REGISTRY_V4[str(profile_id)]
    except KeyError as exc:
        raise ValueError(
            "unsupported Reasoning V4.2 prompt profile: "
            f"{profile_id!r}"
        ) from exc
