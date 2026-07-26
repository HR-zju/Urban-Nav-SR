"""Versioned teacher prompt profiles for independent Reasoning V4 sections."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

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
    SceneGroundingV4,
    TaskGroundingV4,
    reasoning_v4_contract_schema_digest,
)


_COMMON_POLICY = """\
You are an offline annotation teacher for single-turn urban pedestrian navigation.
Use only the complete prompt-visible observation and ordered images supplied in the
user message. FPV is the only authority for immediate local safety. Instruction,
goal, waypoint, and BEV context may express task direction, but cannot prove unseen
free space. Do not use GPS, pose, calibration, route theta, future trajectory,
runtime task phase, tracking truth, segmentation truth, bounding-box truth, or
calibrated confidence.

Raw evidence_refs must be copied exactly from evidence_catalog. Locally created
semantic refs must use the namespaces required by the schema. Categories and
reason codes are constrained open lower-snake-case strings, not a hidden class
choice. Do not invent facts merely to fill a field; use an empty list or an
explicit unknown status where the schema permits it.

Return exactly one JSON object for the requested section. Do not return markdown,
XML tags, explanations, a full six-section trace, Finish, direction labels, or
private supervision."""


_SECTION_POLICIES = {
    "TaskGrounding": """\
Normalize only what the prompt-visible task sources ask the robot to do. A missing
instruction is not a generic instruction. Requirements must be task-specific and
directly evidenced. Do not describe visual entities, paths, or interactions here.""",
    "SceneGrounding": """\
Record only directly visible navigation-relevant entities and traffic controls.
Every item must cite FPV evidence. Do not claim masks, boxes, instance identity,
tracking, motion, path blockage, traversability, or interaction policy.""",
    "LocalTraversability": """\
Describe visible local regions and their immediate traversability from FPV. Use
sample-local semantic region refs based on the areas actually observed; do not
force every scene into a fixed left/center/right partition. Do not infer GPS
geometry, route matching, future motion, or precise metric clearance. A
conditional, blocked, or forbidden region needs a concise lower-snake-case
limiting factor code.""",
    "NavigationContextGrounding": """\
Interpret the task and each actually available goal, waypoint, or BEV source
against the current FPV scene. This section may propose semantic region refs, but
must not claim local safety from non-FPV context. If sources do not resolve a
direction, use the schema's insufficient or ambiguous status.""",
    "InteractionConstraintGrounding": """\
Record only task, traffic, social, accessibility, or local-safety constraints that
can affect the current local choice. LVLM judgments are weak labels. Dynamic
crossing or approaching relations require at least two distinct FPV evidence refs;
otherwise use a static or uncertain relation, or omit the constraint. Do not claim
tracking identity, TTC, calibrated risk, or normative rules not supported by the
visible input or explicit instruction.""",
    "Decision": """\
Choose one short-horizon semantic decision using the complete raw observation and
the five validated grounding sections supplied in the user message. Resolve
selected_region_ref and support_refs against those sections. Decision evidence_refs
must still cite raw evidence_catalog entries. This Stage-A dataset has no reliable
arrival state, so never emit decision_type=finish. Do not output a direction class
or continuous trajectory; deterministic direction supervision and System1 targets
are attached later by the exporter. If no safe executable region exists, prefer a
stop, request_user, or request_replan decision justified by the validated inputs.""",
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
        system_prompt: Exact provider-side policy text.
        policy_digest: Digest over schema, section, model schema, and prompt.
    """

    profile_id: str
    section_name: str
    model_type: type[BaseModel]
    system_prompt: str
    policy_digest: str


def _profile(section_name: str) -> ReasoningV4TeacherProfile:
    """Build one immutable section profile from the canonical policy."""

    model_type = _SECTION_MODELS[section_name]
    system_prompt = (
        f"{_COMMON_POLICY}\n\nSECTION RESPONSIBILITY — {section_name}\n"
        f"{_SECTION_POLICIES[section_name]}"
    )
    payload = {
        "profile_id": f"reasoning_v4_teacher_{section_name.lower()}_v1",
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
        "section_schema": model_type.model_json_schema(),
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
