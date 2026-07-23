from __future__ import annotations

import json

from mcp_server.agent.reasoning_v3_prompt_policy import (
    TEACHER_DECISION_RATIONALE_PROFILE_V3,
    TEACHER_OBSERVATION_PROFILE_V3,
)
from mcp_server.data.reasoning_v3_composer import derive_decision_outcome_v3
from mcp_server.teacher.reasoning_v3_profiles import (
    TEACHER_DECISION_RATIONALE_RUNTIME_PROFILE_V3,
    TEACHER_OBSERVATION_RUNTIME_PROFILE_V3,
)
from mcp_server.teacher.reasoning_v3_teacher import (
    build_reasoning_v3_decision_rationale_question,
    build_reasoning_v3_teacher_question,
    parse_reasoning_v3_decision_rationale,
    parse_reasoning_v3_teacher_observation,
)
from reasoning_v3_test_utils import (
    build_test_source,
    decision_rationale_payload,
    observation_payload,
)


def test_v3_4_pass1_prompt_contains_only_visible_input_and_four_observation_sections() -> None:
    source = build_test_source()
    prompt = build_reasoning_v3_teacher_question(source)
    visible = _json_block(prompt, "MODEL_VISIBLE_INPUT", "OUTPUT_SKELETON")
    skeleton = _json_block(prompt, "OUTPUT_SKELETON", "OUTPUT_JSON_SCHEMA")

    assert set(skeleton) == {
        "schema_version",
        "task_semantic",
        "scene_grounding",
        "local_traversability",
        "navigation_context_grounding",
    }
    assert skeleton["schema_version"] == "urbannav_reasoning_observation_v3_4"
    assert "decision" not in skeleton
    assert "interaction_and_rule_grounding" not in skeleton
    serialized_visible = json.dumps(visible, sort_keys=True)
    for private_token in (
        "canonical_direction",
        "trajectory_ego_t",
        "route_theta",
        "teacher_supervision_context",
        "label_provenance",
    ):
        assert private_token not in serialized_visible

    parsed = parse_reasoning_v3_teacher_observation(
        json.dumps(observation_payload(source))
    )
    assert parsed.valid, parsed.errors


def test_v3_4_pass1_parser_rejects_removed_semantic_and_geometry_fields() -> None:
    source = build_test_source()
    for section, field, value in (
        ("task_semantic", "instruction_type", "directional"),
        ("task_semantic", "local_objective", "follow_instruction"),
        ("scene_grounding", "confidence", "high"),
        ("scene_grounding", "bbox", [0, 0, 1, 1]),
        ("scene_grounding", "mask", "rle"),
        ("scene_grounding", "track", "track.01"),
        ("local_traversability", "route_theta", 20.0),
    ):
        payload = observation_payload(source)
        payload[section][field] = value
        parsed = parse_reasoning_v3_teacher_observation(json.dumps(payload))
        assert not parsed.valid
        assert any(
            field in error and "Extra inputs are not permitted" in error
            for error in parsed.errors
        )


def test_v3_4_pass2_prompt_exposes_only_validated_observation_and_frozen_direction() -> None:
    source = build_test_source()
    parsed = parse_reasoning_v3_teacher_observation(
        json.dumps(observation_payload(source))
    )
    assert parsed.observation is not None, parsed.errors
    outcome = derive_decision_outcome_v3(source, parsed.observation)
    prompt = build_reasoning_v3_decision_rationale_question(
        source,
        observation=parsed.observation,
        expected_outcome=outcome.model_dump(),
    )
    visible = _json_block(
        prompt,
        "VISIBLE_INPUT_AND_FROZEN_DIRECTION",
        "OUTPUT_SKELETON",
    )
    skeleton = _json_block(prompt, "OUTPUT_SKELETON", "OUTPUT_JSON_SCHEMA")

    assert visible["frozen_canonical_direction"] == outcome.model_dump()
    assert "trajectory_ego_t" not in json.dumps(visible, sort_keys=True)
    assert {
        "profile_id": skeleton["direction_profile_id"],
        "profile_version": skeleton["direction_profile_version"],
        "label_index": skeleton["direction_label_index"],
        "label_name": skeleton["direction_label_name"],
        "choice": skeleton["direction_choice"],
        "angle_deg": skeleton["direction_angle_deg"],
        "source": skeleton["direction_source"],
    } == {
        **outcome.model_dump(),
    }

    rationale = parse_reasoning_v3_decision_rationale(
        json.dumps(decision_rationale_payload(source, parsed.observation))
    )
    assert rationale.valid, rationale.errors


def test_v3_4_pass2_parser_rejects_ttc_confidence_and_direction_drift() -> None:
    source = build_test_source()
    observation = parse_reasoning_v3_teacher_observation(
        json.dumps(observation_payload(source))
    ).observation
    assert observation is not None

    for field, value in (("ttc", 1.0), ("confidence", "high")):
        payload = decision_rationale_payload(source, observation)
        payload[field] = value
        parsed = parse_reasoning_v3_decision_rationale(json.dumps(payload))
        assert not parsed.valid
        assert any(
            field in error and "Extra inputs are not permitted" in error
            for error in parsed.errors
        )

    payload = decision_rationale_payload(source, observation)
    payload["direction_label_index"] += 1
    parsed = parse_reasoning_v3_decision_rationale(json.dumps(payload))
    assert parsed.valid
    checked = TEACHER_DECISION_RATIONALE_RUNTIME_PROFILE_V3.validator(
        parsed.rationale,
        expected_outcome=derive_decision_outcome_v3(source).model_dump(),
        observation=observation,
        evidence_catalog=source.model_input.evidence_catalog,
        fixed_labels=source.teacher_supervision_context["fixed_labels"],
    )
    assert any("rationale_direction_identity_mismatch" in error for error in checked.errors)


def test_v3_4_runtime_profiles_bind_versioned_builder_parser_and_validator() -> None:
    observation_runtime = TEACHER_OBSERVATION_RUNTIME_PROFILE_V3
    rationale_runtime = TEACHER_DECISION_RATIONALE_RUNTIME_PROFILE_V3

    assert observation_runtime.profile == TEACHER_OBSERVATION_PROFILE_V3
    assert rationale_runtime.profile == TEACHER_DECISION_RATIONALE_PROFILE_V3
    for runtime in (observation_runtime, rationale_runtime):
        assert runtime.profile.user_prompt_builder_id == runtime.user_prompt_builder.__name__
        assert runtime.profile.parser_id == runtime.parser.__name__
        assert runtime.profile.validator_id == runtime.validator.__name__
        assert "_v3_4_" in runtime.profile.profile_id


def _json_block(prompt: str, heading: str, next_heading: str) -> dict:
    prefix = heading + ":\n"
    suffix = "\n\n" + next_heading + ":"
    return json.loads(prompt.split(prefix, 1)[1].split(suffix, 1)[0])
