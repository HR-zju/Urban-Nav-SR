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
    STAGE_A_DECISION_TYPES_V4,
    SceneGroundingV4,
    TaskGroundingV4,
    reasoning_v4_contract_schema_digest,
)


_BASE_POLICY = """\
You are the System2 reasoning policy of an urban pedestrian navigation assistance
system for a mobile robot. The system supports people with disabilities, including
blind or low-vision users and people with mobility impairments, by providing safe,
predictable, accessible, and socially considerate local navigation.

Transform the complete prompt-visible multimodal observation into one grounded
short-horizon navigation decision. Read all supplied information, including the
active instruction, ordered FPV observations, optional BEV, goal, pixel-goal, or
waypoint context, and the evidence catalog.

Evidence principles:
- Use only evidence explicitly available in the current prompt.
- FPV is authoritative for immediate local safety, visible walkable support,
  obstacles, clearance, occlusion, traffic controls, and local interaction risk.
- Instructions, goals, BEV, and waypoints provide task direction, but cannot prove
  that an unseen or visibly blocked region is traversable.
- Raw evidence_refs must exactly match entries in the evidence catalog.
- Semantic refs are sample-local reasoning links. They organize task objects,
  visible entities, controls, regions, alignments, and constraints, but do not
  create evidence by themselves.
- When evidence is insufficient, use the defined unknown state, an empty list, or
  a conservative decision. Do not fabricate missing entities, regions, relations,
  or navigation context.

Section independence:
- Every grounding section independently reads the complete prompt-visible input.
- Earlier sections may provide useful structured context, but they are not the
  sole source of evidence for later sections.
- Every substantive claim must remain directly supportable by raw prompt-visible
  evidence.
- Avoid repeating information that belongs to another section.

Temporal evidence:
- A single FPV image may support visible occupancy, proximity, obstruction,
  clearance, and other static local relations.
- Use ordered historical FPV observations when claiming temporal relations such
  as crossing, approaching, moving away, or persistent path interaction.
- When temporal evidence does not clearly support a dynamic relation, use a static
  relation or an uncertain constraint rather than inventing motion.

The section responsibilities are fixed:

- TaskGrounding normalizes the supplied task into an objective, goal specification,
  explicit user requirements, and material task ambiguity. It does not determine
  whether the goal is currently visible, describe the scene, or choose an action.

- SceneGrounding records directly visible navigation-relevant entities and traffic
  controls in FPV. It describes observable scene facts only and does not determine
  traversability, path interaction, or navigation policy.

- LocalTraversability identifies FPV-visible local regions and describes their
  surface type, lateral position, traversability state, clearance category, and
  visible limiting factors. It does not select the final route or action.

- NavigationContextGrounding aligns the normalized task and any prompt-visible
  goal, pixel-goal, waypoint, or BEV source with current FPV entities and local
  regions. It determines coarse task direction and candidate regions, but does
  not override FPV safety evidence or apply social and traffic constraints.

- InteractionConstraintGrounding records only evidence-supported social, traffic,
  accessibility, or local-safety constraints that affect current candidate
  regions. Seeing an entity or traffic control alone is not sufficient. A
  constraint must identify its source, the affected candidate region, the
  source-to-path relation, and its behavioral effect.

- Decision selects one short-horizon semantic action by jointly considering the
  complete raw input and all structured grounding sections. It must select a
  resolvable local region for navigation and must not violate visible
  traversability or applicable constraints.

Stage-A is single-turn local E2E navigation. The only allowed decision_type values
are navigate and stop. Do not output task completion, tool calls, user-interaction
actions, replanning actions, dialogue turns, agent-loop state, or private
supervision.

Continuous trajectory targets, future poses, direction-label metadata,
segmentation annotations, and other sidecar supervision must not appear in the
reasoning text."""

_FULL_OUTPUT = """\
Return exactly one canonical document with the following order:

<think>
<TaskGrounding>{...}</TaskGrounding>
<SceneGrounding>{...}</SceneGrounding>
<LocalTraversability>{...}</LocalTraversability>
<NavigationContextGrounding>{...}</NavigationContextGrounding>
<InteractionConstraintGrounding>{...}</InteractionConstraintGrounding>
<Decision>{...}</Decision>
</think>
<finish>{...}</finish>

Do not output prose, headings, markdown, comments, or any additional tags.

The finish JSON is a compact runtime projection of Decision and must exactly match
Decision for decision_type, motion, speed, and social_action.

Append exactly one <readout> block after <finish> only when all of the following
conditions hold:
1. the active prompt explicitly requests System1 readout anchors;
2. Decision.decision_type is navigate; and
3. the sample is eligible for System1 trajectory supervision.

The readout block must use the exact requested System1 tokens and contain no
natural-language content."""

_GROUNDING_OUTPUT = """\
Return exactly one tagged JSON object for the requested grounding section and
nothing else.

The requested section must independently use the complete prompt-visible
instruction, ordered FPV observations, optional multimodal navigation context,
and evidence catalog. Do not assume that another grounding section has already
extracted the required information.

Do not output any other section, Decision, finish, readout, markdown, or prose."""

_DECISION_OUTPUT = """\
The prompt contains the complete original multimodal observation, evidence catalog,
and five validated grounding sections.

Read the original input directly. Treat the grounding sections as structured
supporting context, not as replacements for raw evidence.

Return exactly one <Decision> JSON object.

Requirements:
- selected_region_ref must resolve to a valid local region when decision_type is
  navigate;
- support_refs must resolve to relevant task, entity, control, region, alignment,
  ambiguity, requirement, or constraint objects in the grounding context;
- evidence_refs must exactly cite raw evidence-catalog entries supporting the
  final decision;
- the selected region must not be blocked or forbidden;
- the decision must remain compatible with applicable interaction constraints;
- the only allowed decision_type values in Stage-A are navigate and stop.

Do not output finish, direction classes, trajectories, readout tokens, markdown,
or prose."""

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
        "Stage-A subset only: navigate=move through one executable selected "
        "region; stop=hold position without claiming task completion"
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

_STAGE_A_ENUM_OVERRIDES = {
    "DecisionV4.decision_type": list(STAGE_A_DECISION_TYPES_V4),
    "FinishV4.decision_type": list(STAGE_A_DECISION_TYPES_V4),
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
            qualified_name = f"{model_name}.{field_name}"
            values = _STAGE_A_ENUM_OVERRIDES.get(
                qualified_name,
                _schema_enum_values(field_schema),
            )
            if values:
                line = (
                    f"- {model_name}.{field_name}: "
                    + " | ".join(str(value) for value in values)
                )
                guidance = _ENUM_GUIDANCE.get(qualified_name)
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
    profile_id="urbannav_reasoning_v4_0_full_cot_sft_v2",
    view_type="full_cot_sft",
    section_name=None,
    output_policy=_FULL_OUTPUT,
)
STUDENT_DECISION_PROFILE_V4 = _student_profile(
    profile_id="urbannav_reasoning_v4_0_decision_section_sft_v2",
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
                f"{section_name.lower()}_section_sft_v2"
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
