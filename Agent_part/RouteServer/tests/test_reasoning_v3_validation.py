from __future__ import annotations

import json

from mcp_server.agent.reasoning_v3_observation_validation import validate_reasoning_observation_v3
from mcp_server.agent.reasoning_v3_text import render_reasoning_target_v3
from mcp_server.agent.reasoning_v3_validation import validate_reasoning_target_v3
from mcp_server.contracts.reasoning_v3 import LabelProvenanceEntryV3, SectionLossMaskV3
from mcp_server.data.reasoning_source import ReasoningSourceBuildConfig
from mcp_server.data.reasoning_v3_conditions import build_navigation_context_sources_v3
from mcp_server.data.reasoning_v3_source import build_reasoning_source_v3
from mcp_server.teacher.reasoning_v3_teacher import parse_reasoning_v3_teacher_observation

from reasoning_v3_test_utils import accepted_annotation, observation_payload


def test_v33_validation_accepts_bev_accounted_positive_context() -> None:
    """A BEV row is valid only when scene and navigation ledgers both account for it."""

    source = _source("foxglove", include_bev=True)
    annotation = accepted_annotation(source, corridor="corridor.right", direction="right")
    assert annotation.target is not None
    result = validate_reasoning_target_v3(
        annotation.target,
        instruction=source.model_input.instruction,
        evidence_catalog=source.model_input.evidence_catalog,
        available_modalities=source.model_input.available_modalities,
        label_provenance={**source.label_provenance, **annotation.label_provenance},
        section_loss_mask=annotation.section_loss_mask,
        source_section_loss_mask=source.section_loss_mask,
        fixed_labels=source.teacher_supervision_context["fixed_labels"],
        dataset_source=source.model_input.dataset_source,
        assistant_text=render_reasoning_target_v3(annotation.target),
    )

    assert result.valid, result.errors
    scene_refs = {
        item.source_ref for item in annotation.target.reasoning.scene_grounding.context_observations
    }
    navigation_refs = {
        item.source_ref
        for item in annotation.target.reasoning.navigation_context_grounding.source_groundings
    }
    assert "bev.current" in scene_refs
    assert "bev.current" in navigation_refs


def test_observation_validator_rejects_silently_omitted_bev() -> None:
    """BEV cannot disappear merely because the teacher found no useful relation."""

    source = _source("foxglove", include_bev=True)
    payload = observation_payload(source, corridor="corridor.right", direction="right")
    payload["scene_grounding"]["context_observations"] = []
    parsed = parse_reasoning_v3_teacher_observation(json.dumps(payload))
    assert parsed.observation is not None, parsed.errors

    result = validate_reasoning_observation_v3(
        parsed.observation,
        instruction=source.model_input.instruction,
        evidence_catalog=source.model_input.evidence_catalog,
        available_modalities=source.model_input.available_modalities,
        navigation_context_source_refs=[
            item.source_ref for item in build_navigation_context_sources_v3(source)
        ],
    )

    assert "required_context_silently_omitted:bev.current" in result.errors


def test_observation_validator_rejects_dynamic_relation_from_one_frame() -> None:
    """Crossing motion requires temporal frames or tracked metadata."""

    source = _source("sidewalk", include_bev=False)
    payload = observation_payload(source, corridor="corridor.center", direction="straight")
    payload["scene_grounding"]["entities"] = [
        {
            "entity_ref": "entity.person_01",
            "type": "pedestrian",
            "lateral_position": "center",
            "depth_position": "near",
            "motion_state": "crossing_path",
            "evidence_refs": ["fpv.current"],
            "confidence": "high",
        }
    ]
    payload["interaction_and_rule_grounding"] = {
        "applicable": True,
        "interactions": [
            {
                "actor_ref": "entity.person_01",
                "relation": "crossing_path",
                "affected_corridor_ref": "corridor.center",
                "temporal_basis": "single_frame",
                "required_behavior": "yield",
                "rule": "pedestrian_priority",
                "evidence_refs": ["fpv.current", "corridor.center"],
                "confidence": "high",
            }
        ],
        "traffic_rules": [],
    }
    parsed = parse_reasoning_v3_teacher_observation(json.dumps(payload))
    assert parsed.observation is not None, parsed.errors

    result = validate_reasoning_observation_v3(
        parsed.observation,
        instruction=source.model_input.instruction,
        evidence_catalog=source.model_input.evidence_catalog,
        available_modalities=source.model_input.available_modalities,
        navigation_context_source_refs=[
            item.source_ref for item in build_navigation_context_sources_v3(source)
        ],
    )

    assert "dynamic_entity_missing_temporal_evidence:entity.person_01" in result.errors
    assert "dynamic_interaction_without_temporal_basis:entity.person_01" in result.errors


def test_target_validator_rejects_section_mask_escalation() -> None:
    """Composer/export cannot enable a section disabled by source policy."""

    source = _source("sidewalk", include_bev=False)
    annotation = accepted_annotation(source, corridor="corridor.center", direction="straight")
    assert annotation.target is not None
    source_mask = source.section_loss_mask.model_copy(update={"scene_grounding": False})
    final_mask = annotation.section_loss_mask.model_copy(update={"scene_grounding": True})

    result = validate_reasoning_target_v3(
        annotation.target,
        instruction=source.model_input.instruction,
        evidence_catalog=source.model_input.evidence_catalog,
        available_modalities=source.model_input.available_modalities,
        label_provenance={**source.label_provenance, **annotation.label_provenance},
        section_loss_mask=final_mask,
        source_section_loss_mask=source_mask,
        fixed_labels=source.teacher_supervision_context["fixed_labels"],
        dataset_source=source.model_input.dataset_source,
    )

    assert "section_loss_mask_escalated:scene_grounding" in result.errors


def test_target_validator_rejects_frodobots_trajectory_provenance() -> None:
    """FrodoBots can never acquire strong trajectory or FPV-projection labels."""

    source = _source("frodobots", include_bev=True)
    annotation = accepted_annotation(source, corridor="corridor.left", direction="left")
    assert annotation.target is not None
    provenance = {
        **source.label_provenance,
        **annotation.label_provenance,
        "demonstrated_direction": LabelProvenanceEntryV3(
            source="gt_trajectory",
            strength="strong",
            generator_version="invalid_test",
        ),
    }

    result = validate_reasoning_target_v3(
        annotation.target,
        instruction=source.model_input.instruction,
        evidence_catalog=source.model_input.evidence_catalog,
        available_modalities=source.model_input.available_modalities,
        label_provenance=provenance,
        dataset_source="frodobots",
    )

    assert "frodobots_forbidden_strong_provenance:demonstrated_direction:gt_trajectory" in result.errors


def _source(dataset: str, *, include_bev: bool):
    """Build one directional source with optional BEV context."""

    direction = "left" if dataset == "frodobots" else ("right" if include_bev else "straight")
    instruction = {
        "left": "Turn left at the next opening.",
        "right": "Turn right at the next opening.",
        "straight": "Continue straight.",
    }[direction]
    images = ["current.jpg"]
    roles = ["fpv.current"]
    row = {
        "id": f"{dataset}-validation",
        "dataset_source": dataset,
        "images": images,
        "image_roles": roles,
        "instruction": instruction,
    }
    if include_bev:
        row["images"] = ["current.jpg", "bev.jpg"]
        row["image_roles"] = ["fpv.current", "bev.current"]
    if dataset == "frodobots":
        row["bev_direction"] = "turn_left"
    return build_reasoning_source_v3(
        row,
        index=0,
        config=ReasoningSourceBuildConfig(dataset_source=dataset),
    )
