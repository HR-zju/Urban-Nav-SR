"""Focused test helpers for the two-stage V3.3 Full-CoT pipeline."""

from __future__ import annotations

import json

from mcp_server.agent.reasoning_v3_prompt_policy import (
    TEACHER_DECISION_RATIONALE_PROFILE_V3,
    TEACHER_OBSERVATION_PROFILE_V3,
)
from mcp_server.contracts.reasoning_dataset_v3 import ReasoningAnnotationV3, ReasoningSourceV3
from mcp_server.data.reasoning_v3_composer import (
    REASONING_V3_COMPOSER_VERSION,
    compose_reasoning_target_v3,
    derive_decision_outcome_v3,
)
from mcp_server.teacher.reasoning_v3_batch import (
    REASONING_V3_RATIONALE_PROMPT_VERSION,
    REASONING_V3_TEACHER_PROMPT_VERSION,
)
from mcp_server.teacher.reasoning_v3_fewshot import REASONING_V3_FEWSHOT_VERSION
from mcp_server.teacher.reasoning_v3_teacher import (
    build_reasoning_v3_teacher_question,
    parse_reasoning_v3_decision_rationale,
    parse_reasoning_v3_teacher_observation,
)


def observation_payload(
    source: ReasoningSourceV3,
    *,
    corridor: str,
    direction: str,
) -> dict:
    """Build a valid positive observation payload from the teacher skeleton.

    Args:
        source: Frozen V3.3 source.
        corridor: Directly traversable FPV corridor selected by the observation.
        direction: Coarse directive assigned to every real navigation source.

    Returns:
        JSON-serializable teacher observation payload.
    """

    prompt = build_reasoning_v3_teacher_question(source)
    prefix = "OUTPUT_OBSERVATION_SKELETON="
    line = next(item for item in prompt.splitlines() if item.startswith(prefix))
    payload = json.loads(line[len(prefix) :])
    payload["task_semantic"] = {
        "applicable": True,
        "instruction_type": "directional",
        "caption_role": "not_caption",
        "inferred_objective": "follow_instruction",
        "objective_evidence_refs": ["instruction.active"],
        "requirements": [],
        "referents": [],
        "ambiguities": [],
        "confidence": "high",
    }
    scene = payload["scene_grounding"]
    scene["scene_facts"] = [
        {
            "fact_code": "direct_sidewalk_visible",
            "evidence_refs": ["fpv.current", corridor],
        }
    ]
    scene.update({"visibility": "clear", "confidence": "high"})

    local = payload["local_traversability"]
    local["regions"] = [
        {
            "region_ref": corridor,
            "surface": "sidewalk",
            "state": "direct",
            "required_condition": None,
            "blocker_refs": [],
            "reason": None,
            "severity": "unknown",
            "evidence_refs": [corridor, "fpv.current"],
        }
    ]
    local["preferred_clear_corridor_ref"] = corridor
    local.update({"visibility": "clear", "local_risk": "low", "confidence": "high"})

    context_by_ref = {item["source_ref"]: item for item in scene["context_observations"]}
    relations: list[dict] = []
    navigation = payload["navigation_context_grounding"]
    for grounding in navigation["source_groundings"]:
        source_ref = grounding["source_ref"]
        grounding.update(
            {
                "directive": direction,
                "target_corridor_ref": None if direction == "stop" else corridor,
                "status": "aligned",
                "reliability": "high",
                "ignored_reason": None,
                "evidence_refs": [
                    source_ref,
                    "fpv.current",
                    *([] if direction == "stop" else [corridor]),
                ],
            }
        )
        if source_ref in context_by_ref:
            context_by_ref[source_ref].update(
                {
                    "usage": "used",
                    "facts": ["route_topology_visible"],
                    "ignored_reason": None,
                    "evidence_refs": [source_ref],
                }
            )
            relations.append(
                {
                    "source_ref": source_ref,
                    "relation": _relation_for_source(grounding["source_type"]),
                    "target_ref": None if direction == "stop" else corridor,
                    "status": "aligned" if direction != "stop" else "context_only",
                    "evidence_refs": [
                        source_ref,
                        "fpv.current",
                        *([] if direction == "stop" else [corridor]),
                    ],
                    "confidence": "high",
                }
            )
    scene["cross_modal_relations"] = relations
    return payload


def decision_rationale_payload(source: ReasoningSourceV3, observation: object) -> dict:
    """Build a valid rationale for the deterministic test outcome.

    Args:
        source: Frozen V3.3 source.
        observation: Parsed first-stage observation.

    Returns:
        JSON-serializable decision-rationale payload.
    """

    outcome = derive_decision_outcome_v3(source, observation)  # type: ignore[arg-type]
    if outcome.motion != "stop" and outcome.selected_corridor_ref:
        status = "supported"
        reasons = [
            {
                "reason": "direct_corridor_available",
                "evidence_refs": ["fpv.current", outcome.selected_corridor_ref],
            }
        ]
        confidence = "high"
    else:
        status = "unsupported"
        reasons = []
        confidence = "low"
    return {
        "schema_version": "urbannav_decision_rationale_observation_v3_3",
        "outcome_motion": outcome.motion,
        "status": status,
        "supporting_reasons": reasons,
        "rejected_alternatives": [],
        "confidence": confidence,
    }


def accepted_annotation(
    source: ReasoningSourceV3,
    *,
    corridor: str,
    direction: str,
) -> ReasoningAnnotationV3:
    """Compose one accepted annotation from a valid test observation.

    Args:
        source: Frozen V3.3 source.
        corridor: Directly traversable FPV corridor.
        direction: Coarse navigation directive.

    Returns:
        Accepted annotation bound to ``source``.
    """

    parsed = parse_reasoning_v3_teacher_observation(
        json.dumps(observation_payload(source, corridor=corridor, direction=direction))
    )
    assert parsed.observation is not None, parsed.errors
    rationale_parsed = parse_reasoning_v3_decision_rationale(
        json.dumps(decision_rationale_payload(source, parsed.observation))
    )
    assert rationale_parsed.rationale is not None, rationale_parsed.errors
    composition = compose_reasoning_target_v3(source, parsed.observation, rationale_parsed.rationale)
    return ReasoningAnnotationV3(
        source_group_id=source.model_input.source_group_id,
        source_fingerprint=source.source_fingerprint,
        status="accepted",
        observation=parsed.observation,
        decision_rationale=rationale_parsed.rationale,
        target=composition.target,
        section_loss_mask=composition.section_loss_mask,
        attempts=1,
        label_provenance=composition.label_provenance,
        metadata={
            "teacher_prompt_version": REASONING_V3_TEACHER_PROMPT_VERSION,
            "teacher_prompt_policy_digest": TEACHER_OBSERVATION_PROFILE_V3.policy_digest,
            "rationale_prompt_version": REASONING_V3_RATIONALE_PROMPT_VERSION,
            "rationale_prompt_policy_digest": TEACHER_DECISION_RATIONALE_PROFILE_V3.policy_digest,
            "fewshot_registry_version": REASONING_V3_FEWSHOT_VERSION,
            "composer_version": REASONING_V3_COMPOSER_VERSION,
        },
    )


def _relation_for_source(source_type: str) -> str:
    """Map a navigation source type to its cross-modal relation enum."""

    return {
        "goal": "goal_to_corridor",
        "waypoint": "waypoint_to_corridor",
        "tool": "tool_route_to_corridor",
    }.get(source_type, "route_to_corridor")
