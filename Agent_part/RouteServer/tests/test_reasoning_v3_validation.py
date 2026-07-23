from __future__ import annotations

import json

from mcp_server.agent.reasoning_v3_observation_validation import (
    validate_decision_rationale_v3,
    validate_reasoning_observation_v3,
)
from mcp_server.agent.reasoning_v3_text import render_reasoning_target_v3
from mcp_server.agent.reasoning_v3_validation import validate_reasoning_target_v3
from mcp_server.contracts.reasoning_v3 import LabelProvenanceEntryV3, SectionLossMaskV3
from mcp_server.data.reasoning_v3_composer import derive_decision_outcome_v3
from mcp_server.data.reasoning_v3_conditions import build_navigation_context_sources_v3
from mcp_server.teacher.reasoning_v3_teacher import (
    parse_reasoning_v3_decision_rationale,
    parse_reasoning_v3_teacher_observation,
)
from reasoning_v3_test_utils import (
    accepted_annotation,
    build_test_source,
    decision_rationale_payload,
    observation_payload,
)


def test_v3_4_target_validation_accepts_minimal_evidence_closed_trace() -> None:
    source = build_test_source()
    annotation = accepted_annotation(source)
    assert annotation.target is not None

    result = validate_reasoning_target_v3(
        annotation.target,
        instruction=source.model_input.instruction,
        evidence_catalog=source.model_input.evidence_catalog,
        available_modalities=source.model_input.available_modalities,
        label_provenance=annotation.label_provenance,
        section_loss_mask=annotation.section_loss_mask,
        source_section_loss_mask=source.section_loss_mask,
        fixed_labels=source.teacher_supervision_context["fixed_labels"],
        dataset_source=source.model_input.dataset_source,
        assistant_text=render_reasoning_target_v3(annotation.target),
    )

    assert result.valid, result.errors


def test_v3_4_target_validation_rejects_direction_identity_drift() -> None:
    source = build_test_source()
    annotation = accepted_annotation(source)
    assert annotation.target is not None
    trace = annotation.target.reasoning
    drifted_decision = trace.decision.model_copy(update={"label_index": 11})
    drifted_finish = annotation.target.finish.model_copy(update={"label_index": 11})
    drifted_target = annotation.target.model_copy(
        update={
            "reasoning": trace.model_copy(update={"decision": drifted_decision}),
            "finish": drifted_finish,
        }
    )

    result = validate_reasoning_target_v3(
        drifted_target,
        instruction=source.model_input.instruction,
        evidence_catalog=source.model_input.evidence_catalog,
        label_provenance=annotation.label_provenance,
        fixed_labels=source.teacher_supervision_context["fixed_labels"],
        dataset_source=source.model_input.dataset_source,
    )

    assert any("decision_does_not_copy_canonical_direction" in error for error in result.errors)


def test_v3_4_target_validation_enforces_provenance_semantics() -> None:
    source = build_test_source()
    annotation = accepted_annotation(source)
    assert annotation.target is not None
    direction = annotation.label_provenance["decision.direction"]
    assert direction.source_kind == "deterministic_adapter"
    assert direction.supervision == "strong"
    assert direction.student_inferable is True
    assert direction.loss_channel == "direction_choice"
    for field_path, entry in annotation.label_provenance.items():
        if field_path in {"decision.direction", "system1.trajectory"}:
            continue
        assert entry.source_kind == "lvlm_weak"
        assert entry.supervision == "weak"
        assert entry.student_inferable is True
        assert entry.loss_channel == "none"

    invalid = dict(annotation.label_provenance)
    invalid["decision.direction"] = direction.model_copy(update={"student_inferable": False})
    result = validate_reasoning_target_v3(
        annotation.target,
        instruction=source.model_input.instruction,
        evidence_catalog=source.model_input.evidence_catalog,
        label_provenance=invalid,
        fixed_labels=source.teacher_supervision_context["fixed_labels"],
    )
    assert "canonical_direction_provenance_is_not_strong_adapter_output" in result.errors


def test_v3_4_reasoning_decision_lm_mask_is_never_authorized() -> None:
    source = build_test_source()
    annotation = accepted_annotation(source)
    assert annotation.target is not None
    assert not any(source.section_loss_mask.model_dump(mode="python").values())
    assert annotation.section_loss_mask is not None
    assert annotation.section_loss_mask.decision is False

    result = validate_reasoning_target_v3(
        annotation.target,
        instruction=source.model_input.instruction,
        evidence_catalog=source.model_input.evidence_catalog,
        label_provenance=annotation.label_provenance,
        section_loss_mask=SectionLossMaskV3(decision=True),
        source_section_loss_mask=source.section_loss_mask,
        fixed_labels=source.teacher_supervision_context["fixed_labels"],
    )
    assert "decision_language_loss_is_forbidden_in_v3_4" in result.errors


def test_v3_4_observation_validator_requires_exact_navigation_source_coverage() -> None:
    source = build_test_source()
    payload = observation_payload(source)
    payload["navigation_context_grounding"]["source_groundings"].pop()
    observation = parse_reasoning_v3_teacher_observation(json.dumps(payload)).observation
    assert observation is not None

    result = validate_reasoning_observation_v3(
        observation,
        instruction=source.model_input.instruction,
        evidence_catalog=source.model_input.evidence_catalog,
        navigation_context_source_refs=[
            item.source_ref for item in build_navigation_context_sources_v3(source)
        ],
    )

    assert any("navigation_context_source_coverage_mismatch" in error for error in result.errors)


def test_v3_4_dynamic_interaction_requires_two_visible_fpv_frames() -> None:
    source = build_test_source(include_bev=False)
    observation_data = observation_payload(source)
    observation_data["scene_grounding"]["entities"] = [
        {
            "entity_ref": "entity.person_01",
            "category": "pedestrian",
            "lateral_position": "center",
            "depth_position": "near",
            "evidence_refs": ["fpv.current"],
        }
    ]
    observation = parse_reasoning_v3_teacher_observation(
        json.dumps(observation_data)
    ).observation
    assert observation is not None
    rationale_data = decision_rationale_payload(source, observation)
    rationale_data["interaction_and_rule_grounding"] = {
        "applicable": True,
        "interactions": [
            {
                "actor_ref": "entity.person_01",
                "relation": "crossing_path",
                "temporal_basis": "single_frame",
                "suggested_behavior": "yield",
                "evidence_refs": ["entity.person_01", "fpv.current"],
            }
        ],
        "traffic_rules": [],
    }
    rationale = parse_reasoning_v3_decision_rationale(
        json.dumps(rationale_data)
    ).rationale
    assert rationale is not None

    result = validate_decision_rationale_v3(
        rationale,
        expected_outcome=derive_decision_outcome_v3(source).model_dump(),
        observation=observation,
        evidence_catalog=source.model_input.evidence_catalog,
        fixed_labels=source.teacher_supervision_context["fixed_labels"],
    )

    assert any(
        "dynamic_interaction_requires_visible_multiframe_evidence" in error
        for error in result.errors
    )


def test_v3_4_provenance_contract_rejects_wrong_loss_routing() -> None:
    invalid_payloads = (
        {
            "source": "weak_teacher",
            "source_kind": "lvlm_weak",
            "supervision": "audit_only",
            "student_inferable": True,
            "loss_channel": "weak_lm",
            "generator_version": "test",
        },
        {
            "source": "weak_teacher",
            "source_kind": "lvlm_weak",
            "supervision": "weak",
            "student_inferable": True,
            "loss_channel": "direction_choice",
            "generator_version": "test",
        },
    )
    for payload in invalid_payloads:
        try:
            LabelProvenanceEntryV3.model_validate(payload)
        except ValueError:
            continue
        raise AssertionError(f"invalid provenance unexpectedly accepted: {payload}")
