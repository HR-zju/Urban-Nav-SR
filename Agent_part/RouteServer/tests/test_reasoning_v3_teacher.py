from __future__ import annotations

import json

from mcp_server.agent.reasoning_v3_prompt_policy import (
    TEACHER_DECISION_RATIONALE_PROFILE_V3,
    TEACHER_OBSERVATION_PROFILE_V3,
)
from mcp_server.data.reasoning_source import ReasoningSourceBuildConfig
from mcp_server.data.reasoning_v3_source import build_reasoning_source_v3
from mcp_server.teacher.reasoning_v3_teacher import (
    build_reasoning_v3_decision_rationale_question,
    build_reasoning_v3_teacher_question,
    parse_reasoning_v3_decision_rationale,
    parse_reasoning_v3_teacher_observation,
)
from mcp_server.teacher.reasoning_v3_profiles import (
    TEACHER_DECISION_RATIONALE_RUNTIME_PROFILE_V3,
    TEACHER_OBSERVATION_RUNTIME_PROFILE_V3,
)
from mcp_server.data.reasoning_v3_composer import derive_decision_outcome_v3
from reasoning_v3_test_utils import decision_rationale_payload, observation_payload


def test_v3_teacher_prompt_is_observation_only_and_composes_modalities() -> None:
    source = _source()

    prompt = build_reasoning_v3_teacher_question(source)
    skeleton = _prompt_json(prompt, "OUTPUT_OBSERVATION_SKELETON")
    frozen = _prompt_json(prompt, "FROZEN_MODEL_INPUT")

    assert "FIXED_TEACHER_LABELS" not in prompt
    assert "demonstrated_direction" not in prompt
    assert "trajectory_ego_t" not in prompt
    assert "dataset_source" not in frozen
    assert "source_group_id" not in frozen
    assert _prompt_value(prompt, "PROMPT_PROFILE_ID") == TEACHER_OBSERVATION_PROFILE_V3.profile_id
    assert "reasoning" not in skeleton
    assert "task_semantic" in skeleton
    assert "decision" not in skeleton
    assert "finish" not in skeleton
    assert skeleton["local_traversability"]["regions"] == []
    assert skeleton["local_traversability"]["preferred_clear_corridor_ref"] is None
    assert {
        item["source_ref"]: {"source_type": item["source_type"], "role": item["role"]}
        for item in frozen["navigation_context_sources"]
    } == {
        "instruction.active": {"source_type": "instruction", "role": "unknown"},
        "bev.current": {"source_type": "bev", "role": "route_topology"},
    }
    assert {item["source_ref"] for item in skeleton["navigation_context_grounding"]["source_groundings"]} == {
        "instruction.active",
        "bev.current",
    }
    assert skeleton["scene_grounding"]["context_observations"][0]["source_ref"] == "bev.current"

    parsed = parse_reasoning_v3_teacher_observation(
        json.dumps(observation_payload(source, corridor="corridor.right", direction="right"))
    )
    assert parsed.valid, parsed.errors
    runtime = TEACHER_OBSERVATION_RUNTIME_PROFILE_V3
    assert runtime.profile.user_prompt_builder_id == runtime.user_prompt_builder.__name__
    assert runtime.profile.parser_id == runtime.parser.__name__
    assert runtime.profile.validator_id == runtime.validator.__name__


def test_v3_teacher_defers_generic_instruction_role_to_lvlm() -> None:
    source = build_reasoning_source_v3(
        {
            "id": "teacher-2",
            "dataset_source": "egowalk",
            "images": ["current.jpg"],
            "image_roles": ["fpv.current"],
            "instruction": "Continue safe pedestrian navigation.",
            "trajectory_ego_t": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        },
        index=0,
        config=ReasoningSourceBuildConfig(dataset_source="egowalk"),
    )

    prompt = build_reasoning_v3_teacher_question(source)
    skeleton = _prompt_json(prompt, "OUTPUT_OBSERVATION_SKELETON")
    frozen = _prompt_json(prompt, "FROZEN_MODEL_INPUT")

    assert frozen["navigation_context_sources"] == [
        {"source_type": "instruction", "source_ref": "instruction.active", "role": "unknown"}
    ]
    assert skeleton["navigation_context_grounding"]["applicable"] is True
    assert skeleton["navigation_context_grounding"]["source_groundings"][0]["source_ref"] == "instruction.active"
    assert skeleton["local_traversability"]["applicable"] is True


def test_v3_teacher_rejects_full_target_in_observation_parser() -> None:
    source = _source()
    payload = observation_payload(source, corridor="corridor.right", direction="right")
    payload["decision"] = {"motion": "turn_right"}

    parsed = parse_reasoning_v3_teacher_observation(json.dumps(payload))

    assert not parsed.valid
    assert any("extra_forbidden" in error for error in parsed.errors)


def test_v3_rationale_prompt_exposes_outcome_without_raw_trajectory() -> None:
    """The hindsight stage sees frozen facts and outcome, never raw GT geometry."""

    source = _source()
    parsed = parse_reasoning_v3_teacher_observation(
        json.dumps(observation_payload(source, corridor="corridor.right", direction="right"))
    )
    assert parsed.observation is not None, parsed.errors
    outcome = derive_decision_outcome_v3(source, parsed.observation)
    prompt = build_reasoning_v3_decision_rationale_question(
        source,
        observation=parsed.observation,
        expected_outcome=outcome.model_dump(),
    )

    assert _prompt_value(prompt, "PROMPT_PROFILE_ID") == TEACHER_DECISION_RATIONALE_PROFILE_V3.profile_id
    assert "FROZEN_OUTCOME=" in prompt
    assert "trajectory_ego_t" not in prompt
    assert "trusted_trajectory_summary" not in prompt
    rationale = parse_reasoning_v3_decision_rationale(
        json.dumps(decision_rationale_payload(source, parsed.observation))
    )
    assert rationale.valid, rationale.errors
    runtime = TEACHER_DECISION_RATIONALE_RUNTIME_PROFILE_V3
    assert runtime.profile.user_prompt_builder_id == runtime.user_prompt_builder.__name__
    assert runtime.profile.parser_id == runtime.parser.__name__
    assert runtime.profile.validator_id == runtime.validator.__name__


def _source():
    return build_reasoning_source_v3(
        {
            "id": "teacher-1",
            "dataset_source": "sidewalk",
            "images": ["current.jpg", "bev.jpg"],
            "image_roles": ["fpv.current", "bev.current"],
            "instruction": "Turn right toward the entrance.",
            "trajectory_ego_t": [[0.0, 0.0, 0.0], [2.0, -1.0, -0.2]],
            "trajectory_fpv_t": [[0.5, 0.9], [0.8, 0.7]],
            "trajectory_fpv_coord_frame": "normalized_0_1",
        },
        index=0,
        config=ReasoningSourceBuildConfig(dataset_source="sidewalk"),
    )


def _prompt_json(prompt: str, key: str) -> dict:
    prefix = key + "="
    line = next(item for item in prompt.splitlines() if item.startswith(prefix))
    return json.loads(line[len(prefix) :])


def _prompt_value(prompt: str, key: str) -> str:
    prefix = key + "="
    return next(item[len(prefix) :] for item in prompt.splitlines() if item.startswith(prefix))
