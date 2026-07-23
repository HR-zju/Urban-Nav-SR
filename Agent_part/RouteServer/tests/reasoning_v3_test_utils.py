"""Shared fixtures for the minimal Reasoning V3.4 pipeline."""

from __future__ import annotations

import json
from typing import Any

from mcp_server.agent.reasoning_v3_prompt_policy import (
    TEACHER_DECISION_RATIONALE_PROFILE_V3,
    TEACHER_OBSERVATION_PROFILE_V3,
)
from mcp_server.contracts.reasoning_dataset_v3 import ReasoningAnnotationV3, ReasoningSourceV3
from mcp_server.contracts.reasoning_v3_observation import (
    DECISION_RATIONALE_OBSERVATION_V3_SCHEMA_VERSION,
    REASONING_OBSERVATION_V3_SCHEMA_VERSION,
)
from mcp_server.data.reasoning_source import ReasoningSourceBuildConfig
from mcp_server.data.reasoning_v3_composer import (
    REASONING_V3_COMPOSER_VERSION,
    compose_reasoning_target_v3,
    derive_decision_outcome_v3,
)
from mcp_server.data.reasoning_v3_conditions import build_navigation_context_sources_v3
from mcp_server.data.reasoning_v3_source import build_reasoning_source_v3
from mcp_server.teacher.reasoning_v3_batch import (
    REASONING_V3_RATIONALE_PROMPT_VERSION,
    REASONING_V3_TEACHER_PROMPT_VERSION,
)
from mcp_server.teacher.reasoning_v3_fewshot import REASONING_V3_FEWSHOT_VERSION
from mcp_server.teacher.reasoning_v3_teacher import (
    parse_reasoning_v3_decision_rationale,
    parse_reasoning_v3_teacher_observation,
)


def canonical_direction_record(
    *,
    profile_id: str = "fpv_degree10_terminal_centroid_v2",
    label_index: int = 12,
    name: str = "right_30",
    choice: str | None = "M",
    angle_deg: float | None = 30.0,
) -> dict[str, Any]:
    """Return one complete record from an approved direction adapter."""

    if profile_id == "fpv_degree10_terminal_centroid_v2":
        profile_version = "2.0"
        bin_mode = "degree10_with_stop"
        derivation_version = "pixel_trajectory_terminal_centroid_rule_v2"
    elif profile_id == "ego_degree10_12way_terminal_centroid_v2":
        profile_version = "2.0"
        bin_mode = "degree10_12way_with_stop"
        derivation_version = "ego_x_right_y_front_terminal_centroid_rule_v2"
    else:
        raise ValueError(f"unsupported test direction profile: {profile_id}")
    payload: dict[str, Any] = {
        "profile_id": profile_id,
        "profile_version": profile_version,
        "label_index": label_index,
        "name": name,
        "bin_mode": bin_mode,
        "derivation_version": derivation_version,
        "source": "direction_choice_adapter",
    }
    if choice is not None:
        payload["choice"] = choice
    if angle_deg is not None:
        payload["angle_deg"] = angle_deg
    return payload


def build_test_source(
    *,
    sample_id: str = "reasoning-v3-test",
    dataset_source: str = "sidewalk",
    include_direction: bool = True,
    include_trajectory: bool = True,
    include_bev: bool = True,
    extra_row: dict[str, Any] | None = None,
) -> ReasoningSourceV3:
    """Build one frozen V3.4 source with explicit supervision sidecars."""

    images = ["current.jpg"]
    image_roles = ["fpv.current"]
    if include_bev:
        images.append("bev.jpg")
        image_roles.append("bev.current")
    row: dict[str, Any] = {
        "id": sample_id,
        "dataset_source": dataset_source,
        "images": images,
        "image_roles": image_roles,
        "instruction": "Turn right toward the entrance.",
    }
    if include_trajectory:
        row["trajectory_ego_t"] = [[0.0, 0.0, 0.0], [2.0, -1.0, -0.2]]
    if include_direction:
        row["canonical_direction"] = canonical_direction_record()
    if extra_row:
        row.update(extra_row)
    return build_reasoning_source_v3(
        row,
        index=0,
        config=ReasoningSourceBuildConfig(dataset_source=dataset_source),
    )


def observation_payload(
    source: ReasoningSourceV3,
    *,
    corridor: str = "corridor.right",
) -> dict[str, Any]:
    """Build one schema-valid Pass1 response from model-visible references."""

    fpv_refs = [
        entry.ref for entry in source.model_input.evidence_catalog if entry.modality == "fpv"
    ]
    current_fpv = "fpv.current" if "fpv.current" in fpv_refs else fpv_refs[-1]
    context_sources = build_navigation_context_sources_v3(source)
    return {
        "schema_version": REASONING_OBSERVATION_V3_SCHEMA_VERSION,
        "task_semantic": {
            "applicable": True,
            "requirements": [],
            "referents": [],
            "ambiguities": [],
        },
        "scene_grounding": {
            "applicable": True,
            "entities": [],
            "traffic_controls": [],
        },
        "local_traversability": {
            "applicable": True,
            "regions": [
                {
                    "region_ref": corridor,
                    "surface": "sidewalk",
                    "state": "direct",
                    "blocker_refs": [],
                    "evidence_refs": [corridor, current_fpv],
                }
            ],
        },
        "navigation_context_grounding": {
            "applicable": bool(context_sources),
            "source_groundings": [
                {
                    "source_type": item.source_type,
                    "source_ref": item.source_ref,
                    "usage": "used",
                    "relation": "visible_navigation_context",
                    "evidence_refs": [item.source_ref],
                }
                for item in context_sources
            ],
        },
    }


def decision_rationale_payload(
    source: ReasoningSourceV3,
    observation: object,
    *,
    corridor: str = "corridor.right",
) -> dict[str, Any]:
    """Build a Pass2 response that copies the complete direction identity."""

    outcome = derive_decision_outcome_v3(source, observation)  # type: ignore[arg-type]
    return {
        "schema_version": DECISION_RATIONALE_OBSERVATION_V3_SCHEMA_VERSION,
        "interaction_and_rule_grounding": {
            "applicable": False,
            "interactions": [],
            "traffic_rules": [],
        },
        "direction_profile_id": outcome.profile_id,
        "direction_profile_version": outcome.profile_version,
        "direction_label_index": outcome.label_index,
        "direction_label_name": outcome.label_name,
        "direction_choice": outcome.choice,
        "direction_angle_deg": outcome.angle_deg,
        "direction_source": outcome.source,
        "rationale_status": "supported",
        "supporting_reasons": [
            {
                "reason": "visible_path_supports_direction",
                "evidence_refs": ["fpv.current", corridor],
            }
        ],
    }


def accepted_annotation(
    source: ReasoningSourceV3,
    *,
    corridor: str = "corridor.right",
) -> ReasoningAnnotationV3:
    """Compose one accepted V3.4 annotation from both validated passes."""

    observation_result = parse_reasoning_v3_teacher_observation(
        json.dumps(observation_payload(source, corridor=corridor))
    )
    assert observation_result.observation is not None, observation_result.errors
    rationale_result = parse_reasoning_v3_decision_rationale(
        json.dumps(
            decision_rationale_payload(
                source,
                observation_result.observation,
                corridor=corridor,
            )
        )
    )
    assert rationale_result.rationale is not None, rationale_result.errors
    composition = compose_reasoning_target_v3(
        source,
        observation_result.observation,
        rationale_result.rationale,
    )
    return ReasoningAnnotationV3(
        source_group_id=source.model_input.source_group_id,
        source_fingerprint=source.source_fingerprint,
        status="accepted",
        observation=observation_result.observation,
        decision_rationale=rationale_result.rationale,
        target=composition.target,
        section_loss_mask=composition.section_loss_mask,
        attempts=2,
        label_provenance=composition.label_provenance,
        metadata={
            "teacher_prompt_version": REASONING_V3_TEACHER_PROMPT_VERSION,
            "teacher_prompt_policy_digest": TEACHER_OBSERVATION_PROFILE_V3.policy_digest,
            "rationale_prompt_version": REASONING_V3_RATIONALE_PROMPT_VERSION,
            "rationale_prompt_policy_digest":
                TEACHER_DECISION_RATIONALE_PROFILE_V3.policy_digest,
            "fewshot_registry_version": REASONING_V3_FEWSHOT_VERSION,
            "composer_version": REASONING_V3_COMPOSER_VERSION,
            "annotation_mode": "full_cot",
        },
    )
