"""Versioned prompt policy profiles for canonical Reasoning V3.4 minimal.

The policy text is split into semantic components so the offline annotation
teacher and the trainable student share navigation principles without sharing
their output authority. The teacher emits observations only; the student emits
the composed Full-CoT target.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from mcp_server.contracts.reasoning_v3 import REASONING_V3_CONTRACT_SCHEMA_DIGEST
from mcp_server.contracts.reasoning_v3_observation import (
    DecisionRationaleObservationV3,
    ReasoningObservationV3,
)


URBANNAV_ROLE_AND_ACCESSIBILITY = (
    "You are an urban pedestrian-assistance navigation model for a mobile robot. "
    "Support blind or low-vision users, people with mobility impairments, and other "
    "pedestrians who need accessible, predictable, and socially considerate guidance. "
    "Preserve the user's stated destination, route preferences, accessibility needs, "
    "and current request; do not silently replace or relax them."
)

EVIDENCE_AND_MODALITY_ROLES = (
    "Use only prompt-visible instruction, ordered FPV images, optional BEV or goal images, "
    "and prompt-visible text/tool context. FPV supports only visible semantic observations; "
    "BEV, route text, waypoint text, goal text, and tool text are context, not local geometry "
    "or free-space ground truth. Account for every supplied non-FPV source as used or ignored. "
    "Never use GPS, pose, calibration, route matching, route theta, projected trajectories, "
    "private action labels, or persistent tracks as visual evidence. Preserve supplied evidence "
    "references exactly."
)

SAFETY_AND_SOCIAL_POLICY = (
    "Extract explicit accessibility and social constraints without inventing policy labels. "
    "Do not assume an unmentioned disability, preference, capability, right-of-way, speed, "
    "social action, or fallback. Interaction and traffic-rule outputs are LVLM weak annotations "
    "conditioned on visible evidence and the frozen canonical direction; they never create or "
    "override the action ground truth."
)

UNCERTAINTY_POLICY = (
    "Use unknown or an empty relation list when visible evidence is insufficient. A single "
    "frame cannot establish motion or tracking; multiple FPV frames permit only an apparent "
    "multi-frame relation, not a persistent track or TTC truth. Local entity, control, and "
    "region references are sample-local relation keys, never instance IDs, masks, boxes, "
    "tracks, or calibrated detections. Do not guess merely to fill the schema."
)

OUTPUT_DISCIPLINE = (
    "Return only the exact structured object requested by the active profile. Use valid "
    "JSON and resolvable evidence references. Continuous trajectories, pixel coordinates, "
    "GPS, pose, route geometry, masks, boxes, tracks, confidence scores, private teacher "
    "labels, provenance, and dataset policy must never be written as visual facts."
)

SEMANTIC_OBSERVATION_CONTRACT = (
    "You are the first offline multimodal annotation teacher. Extract only explicit requirements, "
    "referents, and material ambiguity from the visible task text. Describe visible scene entities "
    "and traffic controls, coarse semantic traversability, and used/ignored navigation context. "
    "Copy exact instruction spans. Do not classify instruction type or local objective. Do not "
    "predict action, speed, social action, execution corridor, arrival, confidence, or continuous "
    "System1 output. These outputs are LVLM weak labels rather than dataset ground truth."
)

DECISION_RATIONALE_CONTRACT = (
    "You are the second offline multimodal annotation teacher. A profile-stamped canonical direction "
    "from an approved dataset adapter is frozen. Using the visible images, validated observations, "
    "and that direction, annotate only path-relevant interactions or visible traffic rules and cite "
    "supporting evidence. These relations and the rationale remain LVLM weak labels. Never alter the "
    "direction, derive speed/social/fallback fields, or invent a corridor, tracker, TTC, mask, box, "
    "confidence, or hidden geometric fact. If visible facts do not support the direction, mark the "
    "rationale unsupported."
)

FULL_V3_CHAIN_CONTRACT = (
    "Produce one canonical Reasoning V3.4 minimal chain in this order: TaskGrounding, "
    "SceneGrounding, LocalTraversability, NavigationContextGrounding, "
    "InteractionAndRuleGrounding, Decision, then finish. The first five sections contain only "
    "input references or student-inferable LVLM weak annotations. NavigationContextGrounding only "
    "accounts for visible context sources; it performs no route matching, direction fusion, or "
    "corridor projection. Decision copies exactly one approved profile-stamped direction label and "
    "adds only an evidence-supported weak rationale. The finish payload is a deterministic projection "
    "of that direction label. A System1 readout, when requested, is a latent query anchor rather than "
    "a language target."
)


@dataclass(frozen=True)
class PromptProfileV3:
    """Describe one versioned Reasoning V3 prompt contract.

    Args:
        profile_id: Stable profile identifier stored with generated artifacts.
        output_schema_version: Schema emitted under this profile.
        system_prompt: Fully composed system prompt text.
        user_prompt_builder_id: Stable identifier of the model-visible user builder.
        parser_id: Stable identifier of the typed output parser.
        validator_id: Stable identifier of the semantic validator.
        output_contract_digest: Semantic JSON-Schema digest. Empty only for
            teacher profiles whose separate observation contract is tracked by
            their own parser and validator tests.
    """

    profile_id: str
    output_schema_version: str
    system_prompt: str
    user_prompt_builder_id: str
    parser_id: str
    validator_id: str
    output_contract_digest: str = ""

    @property
    def policy_digest(self) -> str:
        """Return a stable SHA-256 digest for cross-repository drift checks.

        Returns:
            Hex digest over the profile id, schema version, and system prompt.
        """

        components = (
            self.profile_id,
            self.output_schema_version,
            self.system_prompt,
            self.user_prompt_builder_id,
            self.parser_id,
            self.validator_id,
        )
        if self.output_contract_digest:
            components += (self.output_contract_digest,)
        payload = "\n".join(components)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compose_prompt_policy_v3(*parts: str) -> str:
    """Compose non-empty prompt components with stable paragraph boundaries.

    Args:
        *parts: Prompt-policy paragraphs in semantic order.

    Returns:
        A normalized system prompt.
    """

    return "\n\n".join(str(part).strip() for part in parts if str(part).strip())


def _semantic_schema_digest(model: type) -> str:
    """Hash a teacher output schema while ignoring documentation-only keys."""

    def normalize(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: normalize(item)
                for key, item in sorted(value.items())
                if key not in {"title", "description"}
            }
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    payload = json.dumps(
        normalize(model.model_json_schema()),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


TEACHER_OBSERVATION_PROFILE_V3 = PromptProfileV3(
    profile_id="urbannav_reasoning_v3_4_minimal_observation_v3",
    output_schema_version="urbannav_reasoning_observation_v3_4",
    system_prompt=compose_prompt_policy_v3(
        URBANNAV_ROLE_AND_ACCESSIBILITY,
        EVIDENCE_AND_MODALITY_ROLES,
        SAFETY_AND_SOCIAL_POLICY,
        UNCERTAINTY_POLICY,
        SEMANTIC_OBSERVATION_CONTRACT,
        OUTPUT_DISCIPLINE,
    ),
    user_prompt_builder_id="build_reasoning_v3_teacher_question",
    parser_id="parse_reasoning_v3_teacher_observation",
    validator_id="validate_reasoning_observation_v3",
    output_contract_digest=_semantic_schema_digest(ReasoningObservationV3),
)

TEACHER_DECISION_RATIONALE_PROFILE_V3 = PromptProfileV3(
    profile_id="urbannav_reasoning_v3_4_minimal_rationale_v2",
    output_schema_version="urbannav_decision_rationale_observation_v3_4",
    system_prompt=compose_prompt_policy_v3(
        URBANNAV_ROLE_AND_ACCESSIBILITY,
        EVIDENCE_AND_MODALITY_ROLES,
        SAFETY_AND_SOCIAL_POLICY,
        UNCERTAINTY_POLICY,
        DECISION_RATIONALE_CONTRACT,
        OUTPUT_DISCIPLINE,
    ),
    user_prompt_builder_id="build_reasoning_v3_decision_rationale_question",
    parser_id="parse_reasoning_v3_decision_rationale",
    validator_id="validate_decision_rationale_v3",
    output_contract_digest=_semantic_schema_digest(DecisionRationaleObservationV3),
)

STUDENT_FULL_COT_PROFILE_V3 = PromptProfileV3(
    profile_id="urbannav_reasoning_v3_4_minimal_full_cot_v1",
    output_schema_version="urbannav_reasoning_trace_v3_4",
    system_prompt=compose_prompt_policy_v3(
        URBANNAV_ROLE_AND_ACCESSIBILITY,
        EVIDENCE_AND_MODALITY_ROLES,
        SAFETY_AND_SOCIAL_POLICY,
        UNCERTAINTY_POLICY,
        FULL_V3_CHAIN_CONTRACT,
        OUTPUT_DISCIPLINE,
    ),
    user_prompt_builder_id="build_reasoning_v3_full_prompt",
    parser_id="parse_reasoning_target_v3",
    validator_id="validate_reasoning_target_v3",
    output_contract_digest=REASONING_V3_CONTRACT_SCHEMA_DIGEST,
)

PROMPT_PROFILE_REGISTRY_V3: Mapping[str, PromptProfileV3] = MappingProxyType(
    {
        TEACHER_OBSERVATION_PROFILE_V3.profile_id: TEACHER_OBSERVATION_PROFILE_V3,
        TEACHER_DECISION_RATIONALE_PROFILE_V3.profile_id: TEACHER_DECISION_RATIONALE_PROFILE_V3,
        STUDENT_FULL_COT_PROFILE_V3.profile_id: STUDENT_FULL_COT_PROFILE_V3,
    }
)


def get_prompt_profile_v3(profile_id: str) -> PromptProfileV3:
    """Resolve one supported Reasoning V3.4 prompt profile.

    Args:
        profile_id: Stable profile identifier carried by an artifact.

    Returns:
        Registered immutable prompt profile.

    Raises:
        ValueError: If the profile identifier is missing or unsupported.
    """

    try:
        return PROMPT_PROFILE_REGISTRY_V3[str(profile_id)]
    except KeyError as exc:
        raise ValueError(f"unsupported Reasoning V3.4 prompt profile: {profile_id or '<missing>'}") from exc


__all__ = [
    "EVIDENCE_AND_MODALITY_ROLES",
    "FULL_V3_CHAIN_CONTRACT",
    "DECISION_RATIONALE_CONTRACT",
    "OUTPUT_DISCIPLINE",
    "PROMPT_PROFILE_REGISTRY_V3",
    "PromptProfileV3",
    "SAFETY_AND_SOCIAL_POLICY",
    "STUDENT_FULL_COT_PROFILE_V3",
    "SEMANTIC_OBSERVATION_CONTRACT",
    "TEACHER_DECISION_RATIONALE_PROFILE_V3",
    "TEACHER_OBSERVATION_PROFILE_V3",
    "UNCERTAINTY_POLICY",
    "URBANNAV_ROLE_AND_ACCESSIBILITY",
    "compose_prompt_policy_v3",
    "get_prompt_profile_v3",
]
