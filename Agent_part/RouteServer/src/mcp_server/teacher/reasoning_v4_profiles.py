"""Versioned teacher prompt profiles for independent Reasoning V4 sections."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from pydantic import BaseModel

from mcp_server.contracts.reasoning_v4 import (
    DecisionV4,
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
You are an offline annotation teacher for the System2 policy of an urban pedestrian
navigation assistance robot. The system supports blind or low-vision users and
people with mobility impairments through safe, predictable, accessible, and
socially considerate local navigation.

Use the complete prompt-visible multimodal observation supplied in the user
message. FPV is authoritative for immediate local safety, visible walkable
support, obstacles, clearance, traffic controls, and interaction risk.
Instructions, goals, BEV, and waypoints provide task direction but cannot prove
that an unseen or visibly blocked region is traversable.

Raw evidence_refs must exactly match evidence_catalog entries. Semantic refs are
sample-local links that organize reasoning but do not create evidence. Categories
and reason codes are constrained open lower-snake-case strings rather than a
hidden class choice. When evidence is insufficient, use the schema's unknown
state, an empty list, or a conservative judgment instead of fabricating facts.

Every grounding section independently reads the complete visible input. A single
FPV image can support static occupancy, proximity, obstruction, and clearance.
Claims about crossing, approaching, moving away, or persistent interaction should
use ordered historical FPV evidence; otherwise use a static relation or an
uncertain constraint.

Only prompt-visible evidence may appear in the reasoning text; private sidecar
supervision is outside the teacher boundary.

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
Every item must cite FPV evidence. Describe observable scene facts without turning
them into traversability, source-to-path relations, or navigation policy.""",
    "LocalTraversability": """\
Describe visible local regions and their immediate traversability from FPV. Use
sample-local semantic region refs based on the areas actually observed; do not
force every scene into a fixed left/center/right partition. Record surface,
lateral position, coarse clearance, state, and visible limiting factors without
selecting the final action. A conditional, blocked, or forbidden region needs a
concise lower-snake-case limiting factor code.""",
    "NavigationContextGrounding": """\
Interpret the task and each actually available goal, waypoint, or BEV source
against the current FPV scene. This section may propose semantic region refs, but
FPV safety evidence remains authoritative. Produce goal/source alignments, coarse
task direction, and candidate regions; use insufficient or ambiguous states when
the visible sources do not resolve them.""",
    "InteractionConstraintGrounding": """\
Record only task, traffic, social, accessibility, or local-safety constraints that
affect current candidate regions. Seeing an entity or control alone is not a
constraint. Each record must establish a source, affected candidate region,
source-to-path relation, and behavioral effect. Use historical FPV for supported
crossing or approaching claims; otherwise choose a static relation, mark the
constraint uncertain, or omit it.""",
    "Decision": """\
Choose one short-horizon semantic decision using the complete raw observation and
the five validated grounding sections supplied in the user message. Read the
original input directly; the sections are structured support rather than a
replacement for raw evidence. Resolve selected_region_ref and support_refs, cite
the raw evidence supporting the action, and respect visible traversability and
applicable constraints. Stage-A allows only decision_type=navigate or
decision_type=stop. Fine direction and continuous trajectory targets are attached
later as private supervision.""",
}

_SECTION_MODELS: dict[str, type[BaseModel]] = {
    "TaskGrounding": TaskGroundingV4,
    "SceneGrounding": SceneGroundingV4,
    "LocalTraversability": LocalTraversabilityV4,
    "NavigationContextGrounding": NavigationContextGroundingV4,
    "InteractionConstraintGrounding": InteractionConstraintGroundingV4,
    "Decision": DecisionV4,
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
        "profile_id": f"reasoning_v4_teacher_{section_name.lower()}_v2",
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
