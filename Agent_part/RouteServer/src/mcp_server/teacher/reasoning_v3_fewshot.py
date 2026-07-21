"""Versioned, retrieval-scored few-shot examples for Reasoning V3.3 teachers."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from mcp_server.contracts.reasoning_dataset_v3 import ReasoningSourceV3


REASONING_V3_FEWSHOT_VERSION = "urbannav_reasoning_v3_3_fewshot_v1"
FewShotStageV3 = Literal["semantic_observation", "decision_rationale"]


@dataclass(frozen=True)
class ReasoningFewShotV3:
    """Describe one compact teacher demonstration.

    Args:
        example_id: Stable versioned example identifier.
        stage: Teacher stage that may retrieve this example.
        dataset_tags: Preferred dataset families; ``generic`` matches all.
        modality_tags: Modalities whose presence increases retrieval score.
        input_summary: Compact prompt-visible example input.
        output: Typed target fragment for the corresponding stage.
    """

    example_id: str
    stage: FewShotStageV3
    dataset_tags: tuple[str, ...]
    modality_tags: tuple[str, ...]
    input_summary: dict[str, Any]
    output: dict[str, Any]


_EXAMPLES = (
    ReasoningFewShotV3(
        example_id="task_compound_accessible_01",
        stage="semantic_observation",
        dataset_tags=("generic",),
        modality_tags=("instruction", "fpv"),
        input_summary={
            "instruction": "Use the ramp on the right and avoid the stairs.",
            "visible_facts": ["ramp and stairs are both visible"],
        },
        output={
            "applicable": True,
            "instruction_type": "compound",
            "caption_role": "not_caption",
            "inferred_objective": "follow_instruction",
            "objective_evidence_refs": ["instruction.active", "fpv.current"],
            "requirements": [
                {
                    "code": "accessible_route_required",
                    "polarity": "required",
                    "text_span": "Use the ramp",
                    "target_ref": "corridor.right",
                    "evidence_refs": ["instruction.active", "fpv.current", "corridor.right"],
                    "confidence": "high",
                },
                {
                    "code": "avoid_stairs",
                    "polarity": "avoid",
                    "text_span": "avoid the stairs",
                    "target_ref": None,
                    "evidence_refs": ["instruction.active", "fpv.current"],
                    "confidence": "high",
                },
            ],
            "referents": [
                {
                    "phrase": "the ramp on the right",
                    "resolution": "resolved",
                    "resolved_ref": "corridor.right",
                    "evidence_refs": ["instruction.active", "fpv.current", "corridor.right"],
                    "confidence": "high",
                }
            ],
            "ambiguities": [],
            "confidence": "high",
        },
    ),
    ReasoningFewShotV3(
        example_id="task_egowalk_scene_caption_01",
        stage="semantic_observation",
        dataset_tags=("egowalk",),
        modality_tags=("instruction", "fpv"),
        input_summary={
            "caption": "A cyclist passes a row of parked vehicles.",
            "note": "The caption describes the scene and contains no user command.",
        },
        output={
            "applicable": False,
            "instruction_type": "unknown",
            "caption_role": "scene_description",
            "inferred_objective": "unknown",
            "objective_evidence_refs": [],
            "requirements": [],
            "referents": [],
            "ambiguities": [],
            "confidence": "unknown",
        },
    ),
    ReasoningFewShotV3(
        example_id="task_ambiguous_referent_01",
        stage="semantic_observation",
        dataset_tags=("generic",),
        modality_tags=("instruction", "fpv"),
        input_summary={
            "instruction": "Take the opening beside it.",
            "visible_facts": ["multiple possible referents are visible"],
        },
        output={
            "applicable": True,
            "instruction_type": "directional",
            "caption_role": "not_caption",
            "inferred_objective": "unknown",
            "objective_evidence_refs": [],
            "requirements": [],
            "referents": [
                {
                    "phrase": "it",
                    "resolution": "ambiguous",
                    "resolved_ref": None,
                    "evidence_refs": ["instruction.active", "fpv.current"],
                    "confidence": "low",
                }
            ],
            "ambiguities": [
                {
                    "type": "referent",
                    "text_span": "it",
                    "evidence_refs": ["instruction.active", "fpv.current"],
                }
            ],
            "confidence": "low",
        },
    ),
    ReasoningFewShotV3(
        example_id="decision_direct_corridor_01",
        stage="decision_rationale",
        dataset_tags=("generic",),
        modality_tags=("fpv",),
        input_summary={
            "frozen_outcome": {"motion": "go_straight", "selected_corridor_ref": "corridor.center"},
            "validated_facts": ["corridor.center is direct"],
        },
        output={
            "outcome_motion": "go_straight",
            "status": "supported",
            "supporting_reasons": [
                {
                    "reason": "direct_corridor_available",
                    "evidence_refs": ["fpv.current", "corridor.center"],
                }
            ],
            "rejected_alternatives": [],
            "confidence": "high",
        },
    ),
    ReasoningFewShotV3(
        example_id="decision_wait_for_crossing_01",
        stage="decision_rationale",
        dataset_tags=("generic",),
        modality_tags=("fpv", "temporal_fpv"),
        input_summary={
            "frozen_outcome": {"motion": "stop", "social_action": "wait"},
            "validated_facts": ["entity.person_01 crosses corridor.center over multiple frames"],
        },
        output={
            "outcome_motion": "stop",
            "status": "supported",
            "supporting_reasons": [
                {
                    "reason": "interaction_requires_wait",
                    "evidence_refs": ["fpv.history_00", "fpv.current", "corridor.center"],
                }
            ],
            "rejected_alternatives": [
                {
                    "motion": "go_straight",
                    "reason": "interaction_requires_wait",
                    "evidence_refs": ["fpv.history_00", "fpv.current", "corridor.center"],
                }
            ],
            "confidence": "high",
        },
    ),
    ReasoningFewShotV3(
        example_id="decision_unsupported_stop_01",
        stage="decision_rationale",
        dataset_tags=("generic",),
        modality_tags=("fpv",),
        input_summary={
            "frozen_outcome": {"motion": "stop"},
            "validated_facts": ["no goal-arrival, blockage, interaction, or stop instruction is supported"],
        },
        output={
            "outcome_motion": "stop",
            "status": "unsupported",
            "supporting_reasons": [],
            "rejected_alternatives": [],
            "confidence": "low",
        },
    ),
)

FEWSHOT_REGISTRY_V3 = MappingProxyType({item.example_id: item for item in _EXAMPLES})


def select_reasoning_v3_few_shots(
    source: ReasoningSourceV3,
    *,
    stage: FewShotStageV3,
    limit: int = 3,
) -> list[ReasoningFewShotV3]:
    """Select compact examples by dataset and modality overlap.

    Args:
        source: Frozen source used only for retrieval metadata.
        stage: Semantic-observation or decision-rationale teacher stage.
        limit: Maximum returned examples.

    Returns:
        Deterministically ranked few-shot examples.
    """

    dataset = source.model_input.dataset_source.strip().lower().replace("_", "-")
    modalities = set(source.model_input.available_modalities)

    def score(item: ReasoningFewShotV3) -> tuple[int, str]:
        dataset_score = 4 if dataset in item.dataset_tags else (1 if "generic" in item.dataset_tags else 0)
        modality_score = len(modalities & set(item.modality_tags))
        return dataset_score + modality_score, item.example_id

    candidates = [item for item in _EXAMPLES if item.stage == stage]
    ranked = sorted(candidates, key=lambda item: (-score(item)[0], score(item)[1]))
    return ranked[: max(0, int(limit))]


__all__ = [
    "FEWSHOT_REGISTRY_V3",
    "REASONING_V3_FEWSHOT_VERSION",
    "ReasoningFewShotV3",
    "select_reasoning_v3_few_shots",
]
