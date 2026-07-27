"""Versioned teacher prompt profiles for independent Reasoning V4 sections."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from pydantic import BaseModel

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
    STAGE_A_DECISION_TYPES_V4,
    SceneGroundingV4,
    TaskGroundingV4,
    reasoning_v4_contract_schema_digest,
)


_COMMON_POLICY = """\
You are an offline annotation teacher for the System2 policy of an urban
pedestrian navigation assistance robot. The system supports blind or low-vision
users and people with mobility impairments through safe, predictable,
accessible, and socially considerate local navigation.

Use the complete prompt-visible multimodal observation supplied in the user
message. FPV is authoritative for immediate local safety, visible walkable
support, obstacles, clearance, traffic controls, and interaction risk.
Instructions, goals, BEV, and waypoints provide task direction but cannot prove
that an unseen or visibly blocked region is traversable.

Raw evidence_refs must exactly match evidence_catalog entries. The public
direction_output_space defines legal direction labels but never reveals the
selected action. Categories, support types, source types, source states,
uncertainty codes, and reason codes are open lower_snake_case strings.
evidence_status is categorical support metadata, not calibrated confidence,
probability, a loss weight, or a runtime score.

Every grounding section independently reads the complete visible input. A single
FPV image can support static occupancy, proximity, obstruction, and clearance.
Claims about crossing, approaching, moving away, or persistent interaction should
use ordered historical FPV evidence; otherwise use a static relation or an
uncertain constraint.

Do not invent masks, boxes, tracks, TTC, metric geometry, hidden object identity,
geometric areas, route segments, trajectory corridors, or numeric confidence.
A prompt-visible BEV route overlay may support route direction and upcoming
topology when its orientation is clear. The absence of local safety authority
only prevents BEV from proving traversability, clearance, or collision safety.
Pixel-goal and waypoint values cannot derive direction labels in V4.2 until
their frame and derivation contracts are frozen.

Return exactly one JSON object for the requested section. Do not return markdown,
XML tags, explanations, a full six-section trace, Finish, or readout tokens."""


_SECTION_POLICIES = {
    "TaskGrounding": """\
Normalize the supplied task into its objective, goal specification, explicit user
requirements, and material task ambiguity. Base every item on the actual visible
task sources. Leave scene description, goal visibility, traversability, and action
selection to their responsible sections.""",
    "SceneGrounding": """\
Record only directly visible navigation-relevant entities and traffic controls.
Use neutral refs such as entity.e01 and control.c01. Put semantics in open
category/category_candidates fields and cite FPV evidence. For
evidence_status=supported, set one category and category_candidates=[]; for
ambiguous, set category=null and at least two candidates; for insufficient, set
category=null and candidates=[]. Apply the same rule to traffic controls. Do not
turn visible facts into traversability, interaction, or policy.""",
    "LocalTraversability": """\
Judge immediate FPV support for public direction labels. Emit at most one
direction_affordance per label with an open support_type, affordance, coarse
clearance, visible limiting factors, and evidence status. Do not create geometric
areas and do not select the private strong direction.""",
    "NavigationContextGrounding": """\
Interpret the task and each actually available goal, waypoint, or BEV source
against current visible evidence. Use open target descriptions and public
direction labels. Do not create entity or geometric-area refs here.

Keep status-dependent fields exact. A resolved goal has target_description,
no candidate_target_descriptions, and may have supported directions. An
ambiguous goal has at least two candidate_target_descriptions and no
target_description. A not_visible or insufficient_evidence goal has no target
description, no candidates, and no supported directions.

For every source_alignment, copy source_ref exactly from evidence_catalog and
also include that same ref in the alignment's evidence_refs. Set alignment_role
to task_intent, route_intent, or local_safety_bias. status=supports requires one
or more supported directions; every other status requires none. Before returning,
audit every source_alignment: status=supports with
supported_direction_labels=[] is invalid, so change its status to
insufficient_evidence when no label is supported. A prompt-visible BEV route
overlay may use route_intent and supported movement labels only when its
orientation is clear. Pixel-goal and waypoint source alignments must remain
insufficient because their geometry is not frozen. task_direction contains only
task_intent/route_intent movement labels, never stop; it must not absorb
local_safety_bias labels.""",
    "InteractionConstraintGrounding": """\
Record only task, traffic, social, accessibility, or local-safety constraints that
change the current action. Use neutral constraint refs, open source_type and
source_state, a closed relation, and one of slow/yield/stop/avoid. An empty
affected_direction_labels means a global current-action constraint. A global
red pedestrian signal uses effect=stop and an empty list; never enumerate all
non-stop labels and never emit affected_direction_scope. Use a non-empty list
only for an affected subset that leaves other directions viable; effect=avoid
requires such a subset. Do not cite other section IDs as sources. Crossing or
approach relations require distinct historical FPV; otherwise use a static
relation, uncertain status, or omit the constraint. A navigation instruction or
a visibly clear/supportive path is context, not by itself a slow constraint;
omit records that do not actually restrict behavior.

The default output is {"constraints":[]}. Add an item only when you can name and
cite the specific visible or explicit trigger that makes the current action
slower, yielding, stopped, or redirected than it would otherwise be. Effects are
restrictions, never permissions or generic caution. In particular, do not emit
source_type=instruction with source_state=active_navigation_instruction, and do
not emit clear_walkable_path, visible_sidewalk, route_following, safety_priority,
or similar context as a constraint.""",
    "Decision": """\
Choose one short-horizon semantic decision using the complete raw observation and
the five validated grounding sections. Output navigation_intent_direction_label
for the non-stop task/route direction and selected_action_label for the immediate
action. A teacher_only_action_constraint may supply the private trajectory-derived
demonstrated direction as an ordinary-navigation prior. It is not current-action
truth, student input, evidence, or rationale. An applicable global stop constraint
must select stop even when the demonstrated direction is a movement label, without
rewriting task/route intent.

Output decision_type, navigation_intent_direction_label,
selected_action_label, speed, social_action, fallback,
critical_uncertainties, support_refs, and evidence_refs. Do not output motion;
the deterministic composer derives motion and decision_type consistency from
selected_action_label. Stage-A allows only navigate or stop. selected_action=stop
requires decision_type=stop, speed=stop, social_action=wait, and fallback=stop.
navigation_intent_direction_label must never be stop. A stop action must cite
an applicable global stop/yield constraint, all blocked task-intent directions,
or a material stop-policy uncertainty. Do not stop merely because a destination
is absent when the visible short-horizon path and instruction permit
continuing.

Honor every applicable constraint whose affected_direction_labels is empty or
contains selected_action_label: slow requires speed=slow or stop; yield requires
social_action=yield or a stop action; stop requires the exact stop tuple; avoid
requires selecting outside the affected subset. Include the corresponding
constraint_ref in support_refs when it changes the decision.

Run this final audit before returning: for each applicable global constraint and
each applicable subset constraint covering selected_action_label, verify the
chosen speed/social_action/selected_action_label satisfies its effect. If any
slow constraint applies, speed=normal is invalid. If any avoid constraint covers
the selected label, that label is invalid. Revise the action, not the grounding,
until all applicable effects are satisfied.""",
}

_SECTION_MODELS: dict[str, type[BaseModel]] = {
    "TaskGrounding": TaskGroundingV4,
    "SceneGrounding": SceneGroundingV4,
    "LocalTraversability": LocalTraversabilityV4,
    "NavigationContextGrounding": NavigationContextGroundingV4,
    "InteractionConstraintGrounding": InteractionConstraintGroundingV4,
    "Decision": DecisionProposalV4,
}


@dataclass(frozen=True)
class ReasoningV4TeacherProfile:
    """Describe one executable, versioned section annotation policy.

    Args:
        profile_id: Stable identifier written into annotation artifacts.
        section_name: Canonical Reasoning V4 section name.
        model_type: Pydantic model accepted from the teacher.
        output_json_schema: Section schema projected to the active Stage-A policy.
        system_prompt: Exact provider-side policy text.
        policy_digest: Digest over schema, section, model schema, and prompt.
    """

    profile_id: str
    section_name: str
    model_type: type[BaseModel]
    output_json_schema: dict[str, Any]
    system_prompt: str
    policy_digest: str


def _section_output_schema(
    section_name: str,
    model_type: type[BaseModel],
) -> dict[str, Any]:
    """Project the canonical model schema to the active Stage-A subset."""

    schema = json.loads(
        json.dumps(
            model_type.model_json_schema(),
            ensure_ascii=False,
        )
    )
    if section_name == "Decision":
        schema["properties"]["decision_type"]["enum"] = list(
            STAGE_A_DECISION_TYPES_V4
        )
    return schema


def _profile(section_name: str) -> ReasoningV4TeacherProfile:
    """Build one immutable section profile from the canonical policy."""

    model_type = _SECTION_MODELS[section_name]
    output_json_schema = _section_output_schema(
        section_name,
        model_type,
    )
    system_prompt = (
        f"{_COMMON_POLICY}\n\nSECTION RESPONSIBILITY — {section_name}\n"
        f"{_SECTION_POLICIES[section_name]}"
    )
    payload = {
        "profile_id": f"reasoning_v4_teacher_{section_name.lower()}_v4",
        "trace_schema_version": REASONING_TRACE_V4_SCHEMA_VERSION,
        "contract_schema_digest": reasoning_v4_contract_schema_digest(),
        "wire_format_version": REASONING_V4_WIRE_FORMAT_VERSION,
        "section_definition_version": (
            REASONING_V4_SECTION_DEFINITION_VERSION
        ),
        "evidence_policy_version": REASONING_V4_EVIDENCE_POLICY_VERSION,
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
        profile_id=payload["profile_id"],
        section_name=section_name,
        model_type=model_type,
        output_json_schema=output_json_schema,
        system_prompt=system_prompt,
        policy_digest=digest,
    )


REASONING_V4_TEACHER_PROFILES: Mapping[str, ReasoningV4TeacherProfile] = (
    MappingProxyType(
        {
            section_name: _profile(section_name)
            for section_name in _SECTION_MODELS
        }
    )
)


def get_reasoning_v4_teacher_profile(
    section_name: str,
) -> ReasoningV4TeacherProfile:
    """Resolve the canonical teacher profile for one V4 section.

    Args:
        section_name: Canonical six-section name.

    Returns:
        Immutable profile descriptor.

    Raises:
        ValueError: If the section name is unsupported.
    """

    try:
        return REASONING_V4_TEACHER_PROFILES[str(section_name)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported Reasoning V4 teacher section: {section_name!r}"
        ) from exc
