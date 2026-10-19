#!/usr/bin/env python3
"""Export deterministic Navigation VQA V2 schemas and parity fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from pydantic import TypeAdapter

from mcp_server.agent.vqa_v2_prompt_policy import (
    STUDENT_VQA_PROFILE_V2,
)
from mcp_server.agent.vqa_v2_text import render_vqa_answer_v2
from mcp_server.contracts.direction_profile import (
    DirectionLabelStamp,
    resolve_stage_a_direction_profile,
)
from mcp_server.contracts.vqa_dataset_v2 import (
    VQAAnnotationV2,
    VQACapabilityConfigV2,
    VQA_EXTERNAL_INTERFACE_MODEL_BY_TASK,
    VQASourceV2,
    VQATrainingViewV2,
)
from mcp_server.contracts.vqa_v2 import (
    VQAAnswerV2,
    VQAQuestionV2,
    VQA_V2_CONTRACT_SCHEMA_DIGEST,
    VQA_V2_PARSER_ID,
    VQA_V2_RENDERER_ID,
    VQA_V2_SCHEMA_VERSION,
    VQA_V2_TASK_IDS,
    VQA_V2_TASK_REGISTRY_VERSION,
    VQA_V2_WIRE_FORMAT_VERSION,
    build_vqa_answer_v2,
)
from mcp_server.teacher.vqa_v2_profiles import (
    VQA_V2_TEACHER_PROFILES,
)


ARTIFACT_SCHEMA_VERSION = (
    "urbannav_navigation_vqa_contract_artifacts_v2"
)
ARTIFACT_FILENAMES = (
    "direction_profile.schema.json",
    "vqa_v2.schema.json",
    "vqa_source_v2.schema.json",
    "vqa_annotation_v2.schema.json",
    "vqa_training_view_v2.schema.json",
    "vqa_capabilities_v2.schema.json",
    "vqa_v2_stage_a_capabilities.json",
    "vqa_v2_golden_valid.json",
    "vqa_v2_golden_invalid.json",
    "vqa_v2_interface_golden_valid.json",
    "vqa_v2_interface_golden_invalid.json",
    "contract_digest.json",
)

_VALID_PAYLOAD_BY_TASK: dict[str, dict[str, Any]] = {
    "a1_goal_understanding": {
        "goal": "reach the visible entrance",
        "status": "resolved",
    },
    "a2_explicit_requirement": {
        "requirements": ["avoid_stairs"],
        "status": "resolved",
    },
    "a3_task_ambiguity": {
        "ambiguities": [],
        "status": "clear",
    },
    "b1_direction_affordance": {"affordance": "direct"},
    "b2_direction_clearance": {"clearance": "sufficient"},
    "b3_direction_surface_effect": {
        "surface": "sidewalk",
        "effect": "proceed",
    },
    "b4_conditional_affordance": {
        "usability": "immediate",
        "condition_code": None,
        "effect": "proceed",
    },
    "b5_direction_comparison": {
        "preferred_direction_label": "straight",
        "reason_codes": ["better_clearance"],
    },
    "b6_unavailable_directions": {
        "direction_labels": ["left_10"],
        "reason_codes": ["visible_obstruction"],
    },
    "c1_goal_to_direction": {
        "direction_label": "straight",
        "candidate_direction_labels": [],
        "status": "resolved",
    },
    "c2_pixel_goal_to_direction": {
        "direction_label": "left_10",
        "status": "aligned",
    },
    "c2_waypoint_to_direction": {
        "direction_label": "right_10",
        "status": "aligned",
    },
    "c3_bev_route_alignment": {
        "direction_label": "right_20",
        "alignment_status": "aligned",
    },
    "c4_bev_object_anchoring": {
        "bev_ref": "bev.building_a",
        "fpv_entity_ref": "fpv.entity_e2",
        "direction_label": "right_30",
        "status": "matched",
    },
    "c5_outside_view_direction": {
        "outside_view_information": ["upcoming_turn"],
        "current_direction_label": "straight",
        "future_direction_label": "left_40",
    },
    "c6_bev_fpv_executability": {
        "direction_label": "right_20",
        "status": "executable",
    },
    "d1_agent_relevance": {
        "relevance": "near_candidate_path"
    },
    "d2_static_interaction_relation": {
        "relation": "near_candidate_path"
    },
    "d3_temporal_interaction_relation": {
        "relation": "approaches_candidate_path"
    },
    "d4_applicable_constraint": {
        "constraint_code": "yield_to_pedestrian",
        "kind": "social",
        "effect": "yield",
        "affected_direction_labels": [],
    },
    "d5_appropriate_response": {"effect": "slow"},
    "d6_traffic_control_applicability": {
        "applicability": "applies",
        "effect": "wait",
        "controlled_direction_labels": [],
    },
    "e0_demonstrated_direction_selection": {
        "direction_label": "straight"
    },
    "e1_candidate_trajectory_preference": {
        "preferred_candidate": "candidate_b",
        "reason_codes": ["better_clearance"],
    },
    "e2_candidate_direction_classification": {
        "direction_label": "right_20"
    },
    "e3_trajectory_compatibility": {
        "status": "incompatible",
        "reason_codes": ["blocked_direction"],
    },
    "e4_alternative_direction": {
        "alternative_direction_label": "left_10",
        "reason_codes": ["visible_obstruction"],
        "status": "resolved",
    },
}


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse contract export arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def build_vqa_v2_contract_artifacts() -> dict[str, Any]:
    """Build canonical schemas and strict text fixtures in memory.

    Returns:
        Stable filename to JSON-compatible artifact mapping.
    """

    valid, invalid = _answer_golden_fixtures()
    interface_valid, interface_invalid = (
        _interface_golden_fixtures()
    )
    route_root = Path(__file__).resolve().parents[2]
    capability_payload = json.loads(
        (
            route_root
            / "config/vqa_v2_stage_a_capabilities.json"
        ).read_text(encoding="utf-8")
    )
    VQACapabilityConfigV2.model_validate(capability_payload)
    artifacts: dict[str, Any] = {
        "direction_profile.schema.json": (
            DirectionLabelStamp.model_json_schema()
        ),
        "vqa_v2.schema.json": TypeAdapter(
            VQAAnswerV2
        ).json_schema(),
        "vqa_source_v2.schema.json": (
            VQASourceV2.model_json_schema()
        ),
        "vqa_annotation_v2.schema.json": (
            VQAAnnotationV2.model_json_schema()
        ),
        "vqa_training_view_v2.schema.json": (
            VQATrainingViewV2.model_json_schema()
        ),
        "vqa_capabilities_v2.schema.json": (
            VQACapabilityConfigV2.model_json_schema()
        ),
        "vqa_v2_stage_a_capabilities.json": capability_payload,
        "vqa_v2_golden_valid.json": valid,
        "vqa_v2_golden_invalid.json": invalid,
        "vqa_v2_interface_golden_valid.json": interface_valid,
        "vqa_v2_interface_golden_invalid.json": (
            interface_invalid
        ),
    }
    artifacts["contract_digest.json"] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "vqa_schema_version": VQA_V2_SCHEMA_VERSION,
        "vqa_contract_schema_digest": (
            VQA_V2_CONTRACT_SCHEMA_DIGEST
        ),
        "protocol_ids": {
            "wire_format_version": VQA_V2_WIRE_FORMAT_VERSION,
            "task_registry_version": (
                VQA_V2_TASK_REGISTRY_VERSION
            ),
            "renderer_id": VQA_V2_RENDERER_ID,
            "parser_id": VQA_V2_PARSER_ID,
        },
        "student_prompt_policy_digest": (
            STUDENT_VQA_PROFILE_V2.policy_digest
        ),
        "teacher_prompt_policy_digests": {
            task_id: profile.policy_digest
            for task_id, profile in sorted(
                VQA_V2_TEACHER_PROFILES.items()
            )
        },
        "canonical_source_sha256": _canonical_source_hashes(),
        "artifact_sha256": {
            filename: _payload_sha256(payload)
            for filename, payload in artifacts.items()
        },
    }
    return artifacts


def _answer_golden_fixtures(
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one valid and one invalid assistant fixture per task."""

    valid_cases: list[dict[str, Any]] = []
    invalid_cases: list[dict[str, Any]] = []
    for task_id in VQA_V2_TASK_IDS:
        question = _fixture_question(task_id)
        payload = _VALID_PAYLOAD_BY_TASK[task_id]
        answer = build_vqa_answer_v2(task_id, payload)
        assistant_text = render_vqa_answer_v2(answer)
        valid_cases.append(
            {
                "case_id": f"{task_id}_valid",
                "task_id": task_id,
                "question": question.model_dump(mode="json"),
                "answer": answer.model_dump(mode="json"),
                "canonical_assistant_text": assistant_text,
            }
        )
        invalid_payload = {
            **payload,
            "unexpected_field": "must_be_rejected",
        }
        invalid_cases.append(
            {
                "case_id": f"{task_id}_extra_field",
                "entrypoint": "assistant_text",
                "task_id": task_id,
                "question": question.model_dump(mode="json"),
                "value": (
                    "<VQAAnswer>"
                    + json.dumps(
                        invalid_payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "</VQAAnswer>"
                ),
                "expected_error_contains": "extra_forbidden",
            }
        )
    reference_question = _fixture_question(
        "b1_direction_affordance"
    )
    reference_answer = build_vqa_answer_v2(
        reference_question.task_id,
        _VALID_PAYLOAD_BY_TASK[reference_question.task_id],
    )
    invalid_cases.extend(
        [
            {
                "case_id": "outside_text",
                "entrypoint": "assistant_text",
                "task_id": reference_question.task_id,
                "question": reference_question.model_dump(mode="json"),
                "value": (
                    render_vqa_answer_v2(reference_answer) + "\nextra"
                ),
                "expected_error_contains": (
                    "unexpected_text_outside_vqa_answer"
                ),
            },
            {
                "case_id": "region_question_rejected",
                "entrypoint": "question",
                "value": {
                    **reference_question.model_dump(mode="json"),
                    "question_text": (
                        "Is region_r1 traversable along straight?"
                    ),
                },
                "expected_error_contains": (
                    "must not use region terminology"
                ),
            },
        ]
    )
    return (
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "cases": valid_cases,
        },
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "cases": invalid_cases,
        },
    )


def _fixture_question(task_id: str) -> VQAQuestionV2:
    """Build a realistic answer-independent question fixture."""

    conditioned_tasks = {
        "b1_direction_affordance",
        "b2_direction_clearance",
        "b3_direction_surface_effect",
        "b4_conditional_affordance",
        "d1_agent_relevance",
        "d2_static_interaction_relation",
        "d3_temporal_interaction_relation",
        "d4_applicable_constraint",
        "d5_appropriate_response",
        "d6_traffic_control_applicability",
    }
    candidates_by_task = {
        "b5_direction_comparison": ["left_10", "straight"],
        "b6_unavailable_directions": [
            "left_10",
            "straight",
            "right_10",
        ],
        "c1_goal_to_direction": ["left_10", "straight"],
        "c2_pixel_goal_to_direction": ["left_10"],
        "c2_waypoint_to_direction": ["right_10"],
        "c3_bev_route_alignment": ["right_20"],
        "c4_bev_object_anchoring": ["right_30"],
        "c5_outside_view_direction": ["straight", "left_40"],
        "c6_bev_fpv_executability": ["right_20"],
        "e0_demonstrated_direction_selection": ["straight"],
        "e2_candidate_direction_classification": ["right_20"],
        "e4_alternative_direction": ["left_10", "right_10"],
    }
    candidates = list(candidates_by_task.get(task_id, []))
    conditioned = "straight" if task_id in conditioned_tasks else None
    evidence_refs = ["fpv.current"]
    target_refs: list[str] = []
    if task_id == "c1_goal_to_direction":
        evidence_refs.append("goal.image_1")
        target_refs.append("goal.image_1")
    elif task_id == "c2_pixel_goal_to_direction":
        evidence_refs.append("goal.pixel_1")
        target_refs.append("goal.pixel_1")
    elif task_id == "c2_waypoint_to_direction":
        evidence_refs.append("waypoint.active")
        target_refs.append("waypoint.active")
    elif task_id in {
        "c3_bev_route_alignment",
        "c4_bev_object_anchoring",
        "c5_outside_view_direction",
        "c6_bev_fpv_executability",
    }:
        evidence_refs.append("bev.current")
        target_refs.append("bev.current")
    visible_candidates = (
        [
            {
                "candidate_id": "candidate_a",
                "prompt_visible_representation_ref": (
                    "fpv.candidate_a"
                ),
            },
            {
                "candidate_id": "candidate_b",
                "prompt_visible_representation_ref": (
                    "fpv.candidate_b"
                ),
            },
        ]
        if task_id
        in {
            "e1_candidate_trajectory_preference",
            "e2_candidate_direction_classification",
            "e3_trajectory_compatibility",
        }
        else []
    )
    evidence_refs.extend(
        [
            item["prompt_visible_representation_ref"]
            for item in visible_candidates
        ]
    )
    has_direction = conditioned is not None or bool(candidates)
    return VQAQuestionV2(
        task_id=task_id,
        template_id=f"{task_id}_golden_v1",
        question_text=f"Synthetic contract question for {task_id}?",
        direction_profile_id=(
            "fpv_degree10_terminal_centroid_v2"
            if has_direction
            else None
        ),
        conditioned_direction_label=conditioned,
        candidate_direction_labels=candidates,
        target_refs=target_refs,
        evidence_refs=evidence_refs,
        candidate_set_id=(
            "candidate-set-1" if visible_candidates else None
        ),
        visible_candidates=visible_candidates,
        target_candidate_id=(
            "candidate_b"
            if task_id
            == "e2_candidate_direction_classification"
            else "candidate_a"
            if task_id == "e3_trajectory_compatibility"
            else None
        ),
    )


def _interface_golden_fixtures(
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build valid and provenance-invalid fixtures for dormant interfaces."""

    source_id = "vqa-v2-interface-fixture"
    fingerprint = "a" * 64
    base = {
        "source_group_id": source_id,
        "producer_version": "fixture_producer_v1",
        "input_fingerprint": fingerprint,
        "qc_status": "accepted",
    }
    candidate_set = {
        "candidate_set_id": "candidate-set-1",
        "candidate_producer_id": "fixture_candidate_producer",
        "candidate_producer_version": "fixture_candidate_producer_v1",
        "candidates": [
            {
                "candidate_id": "candidate_a",
                "prompt_visible_representation_ref": "fpv.candidate_a",
                "direction": _direction_stamp(
                    "straight",
                    source="candidate_overlay_a",
                    derivation_version="candidate_overlay_rule_v1",
                ),
            },
            {
                "candidate_id": "candidate_b",
                "prompt_visible_representation_ref": "fpv.candidate_b",
                "direction": _direction_stamp(
                    "right_20",
                    source="candidate_overlay_b",
                    derivation_version="candidate_overlay_rule_v1",
                ),
            },
        ],
    }
    payloads: dict[str, dict[str, Any]] = {
        "c2_pixel_goal_to_direction": {
            **base,
            "producer_id": "fixture_pixel_goal_producer",
            "source_kind": "pixel_goal_direction",
            "evidence_refs": ["fpv.current", "goal.pixel_1"],
            "source_field_refs": ["pixel_goal.xy"],
            "coordinate_frame": "fpv_pixel_xy_top_left",
            "image_width": 1920,
            "image_height": 1080,
            "coordinate_convention_version": "pixel_goal_xy_v1",
            "pixel_goal_source_ref": "goal.pixel_1",
            "direction": _direction_stamp(
                "left_10",
                source="pixel_goal_annotation",
                derivation_version="pixel_goal_direction_rule_v1",
            ),
            "answer": _answer_dump("c2_pixel_goal_to_direction"),
        },
        "c2_waypoint_to_direction": {
            **base,
            "producer_id": "fixture_waypoint_producer",
            "source_kind": "waypoint_direction",
            "evidence_refs": ["fpv.current", "waypoint.active"],
            "source_field_refs": ["waypoint.xy"],
            "coordinate_frame": "ego_x_right_y_front",
            "coordinate_unit": "meter",
            "transform_version": "waypoint_transform_v1",
            "waypoint_source_ref": "waypoint.active",
            "direction": _direction_stamp(
                "right_10",
                source="waypoint_annotation",
                derivation_version="waypoint_direction_rule_v1",
            ),
            "answer": _answer_dump("c2_waypoint_to_direction"),
        },
        "c3_bev_route_alignment": {
            **base,
            "producer_id": "fixture_bev_alignment_producer",
            "source_kind": "bev_route_direction",
            "evidence_refs": ["fpv.current", "bev.current"],
            "source_field_refs": ["independent_bev_direction_record"],
            "independent_source": True,
            "bev_evidence_ref": "bev.current",
            "fpv_evidence_ref": "fpv.current",
            "direction": _direction_stamp(
                "right_20",
                source="independent_bev_annotation",
                derivation_version="bev_alignment_rule_v1",
            ),
            "answer": _answer_dump("c3_bev_route_alignment"),
        },
        "c4_bev_object_anchoring": {
            **base,
            "producer_id": "fixture_cross_view_producer",
            "source_kind": "cross_view_entity_binding",
            "evidence_refs": [
                "fpv.current",
                "bev.current",
                "bev.building_a",
                "fpv.entity_e2",
            ],
            "source_field_refs": ["cross_view_match_record"],
            "bev_ref": "bev.building_a",
            "fpv_entity_ref": "fpv.entity_e2",
            "direction": _direction_stamp(
                "right_30",
                source="cross_view_annotation",
                derivation_version="cross_view_direction_rule_v1",
            ),
            "answer": _answer_dump("c4_bev_object_anchoring"),
        },
        "c5_outside_view_direction": {
            **base,
            "producer_id": "fixture_future_direction_producer",
            "source_kind": "outside_view_direction",
            "evidence_refs": [
                "fpv.current",
                "bev.current",
                "bev.transition_1",
            ],
            "source_field_refs": [
                "current_direction_record",
                "future_direction_record",
            ],
            "current_direction": _direction_stamp(
                "straight",
                source="current_direction_annotation",
                derivation_version="current_direction_rule_v1",
            ),
            "future_direction": _direction_stamp(
                "left_40",
                source="future_direction_annotation",
                derivation_version="future_direction_rule_v1",
            ),
            "future_horizon": 3.0,
            "future_horizon_unit": "second",
            "transition_ref": "bev.transition_1",
            "future_not_yet_executed": True,
            "answer": _answer_dump("c5_outside_view_direction"),
        },
        "c6_bev_fpv_executability": {
            **base,
            "producer_id": "fixture_executability_producer",
            "source_kind": "bev_fpv_executability",
            "evidence_refs": ["fpv.current", "bev.current"],
            "source_field_refs": [
                "bev_alignment_record",
                "fpv_executability_record",
            ],
            "bev_direction": _direction_stamp(
                "right_20",
                source="independent_bev_annotation",
                derivation_version="bev_alignment_rule_v1",
            ),
            "bev_alignment_evidence_ref": "bev.current",
            "fpv_executability_evidence_ref": "fpv.current",
            "local_executability_producer_id": (
                "fixture_local_executability"
            ),
            "local_executability_producer_version": (
                "fixture_local_executability_v1"
            ),
            "answer": _answer_dump("c6_bev_fpv_executability"),
        },
        "e1_candidate_trajectory_preference": {
            **base,
            "producer_id": "fixture_preference_producer",
            "source_kind": "candidate_trajectory_preference",
            "evidence_refs": [
                "fpv.current",
                "fpv.candidate_a",
                "fpv.candidate_b",
            ],
            "source_field_refs": ["candidate_preference_label"],
            "candidate_set": candidate_set,
            "answer": _answer_dump(
                "e1_candidate_trajectory_preference"
            ),
        },
        "e2_candidate_direction_classification": {
            **base,
            "producer_id": "fixture_candidate_direction_producer",
            "source_kind": "candidate_direction",
            "evidence_refs": [
                "fpv.current",
                "fpv.candidate_a",
                "fpv.candidate_b",
            ],
            "source_field_refs": ["candidate_direction_label"],
            "candidate_set": candidate_set,
            "target_candidate_id": "candidate_b",
            "answer": _answer_dump(
                "e2_candidate_direction_classification"
            ),
        },
        "e3_trajectory_compatibility": {
            **base,
            "producer_id": "fixture_compatibility_producer",
            "source_kind": "trajectory_compatibility",
            "evidence_refs": [
                "fpv.current",
                "fpv.candidate_a",
                "fpv.candidate_b",
            ],
            "source_field_refs": ["candidate_compatibility_label"],
            "candidate_set": candidate_set,
            "target_candidate_id": "candidate_a",
            "constraint_evidence_refs": ["fpv.current"],
            "answer": _answer_dump("e3_trajectory_compatibility"),
        },
    }
    valid_cases: list[dict[str, Any]] = []
    invalid_cases: list[dict[str, Any]] = []
    for task_id, payload in payloads.items():
        model_type = VQA_EXTERNAL_INTERFACE_MODEL_BY_TASK[task_id]
        validated = model_type.model_validate(payload)
        value = validated.model_dump(mode="json")
        valid_cases.append(
            {
                "case_id": f"{task_id}_interface_valid",
                "task_id": task_id,
                "interface_model": model_type.__name__,
                "value": value,
            }
        )
        invalid_value = json.loads(json.dumps(value))
        del invalid_value["producer_version"]
        invalid_cases.append(
            {
                "case_id": f"{task_id}_missing_producer_version",
                "task_id": task_id,
                "interface_model": model_type.__name__,
                "value": invalid_value,
                "expected_error_contains": "producer_version",
            }
        )
    forbidden_c3 = json.loads(
        json.dumps(payloads["c3_bev_route_alignment"])
    )
    forbidden_c3["direction"]["source"] = "route_theta"
    invalid_cases.append(
        {
            "case_id": "c3_route_theta_forbidden",
            "task_id": "c3_bev_route_alignment",
            "interface_model": (
                VQA_EXTERNAL_INTERFACE_MODEL_BY_TASK[
                    "c3_bev_route_alignment"
                ].__name__
            ),
            "value": forbidden_c3,
            "expected_error_contains": "forbidden",
        }
    )
    hidden_c4 = json.loads(
        json.dumps(payloads["c4_bev_object_anchoring"])
    )
    hidden_c4["evidence_refs"].remove("fpv.entity_e2")
    invalid_cases.append(
        {
            "case_id": "c4_hidden_entity_id_forbidden",
            "task_id": "c4_bev_object_anchoring",
            "interface_model": (
                VQA_EXTERNAL_INTERFACE_MODEL_BY_TASK[
                    "c4_bev_object_anchoring"
                ].__name__
            ),
            "value": hidden_c4,
            "expected_error_contains": "prompt-visible evidence",
        }
    )
    return (
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "cases": valid_cases,
        },
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "cases": invalid_cases,
        },
    )


def _answer_dump(task_id: str) -> dict[str, Any]:
    """Return one typed answer envelope for an interface fixture."""

    return build_vqa_answer_v2(
        task_id,
        _VALID_PAYLOAD_BY_TASK[task_id],
    ).model_dump(mode="json")


def _direction_stamp(
    label_name: str,
    *,
    source: str,
    derivation_version: str,
) -> dict[str, Any]:
    """Build one internally consistent neutral direction stamp."""

    profile_id = "fpv_degree10_terminal_centroid_v2"
    profile = resolve_stage_a_direction_profile(profile_id)
    labels = list(profile["labels"])
    index = labels.index(label_name)
    return {
        "profile_id": profile_id,
        "profile_version": str(profile["profile_version"]),
        "label_index": index,
        "label_name": label_name,
        "choice": profile["choices"][index],
        "angle_deg": profile["angles"][index],
        "source": source,
        "bin_mode": str(profile["bin_mode"]),
        "derivation_version": derivation_version,
    }


def export_vqa_v2_contract_artifacts(
    output_dir: str | Path,
    *,
    check: bool = False,
) -> dict[str, Any]:
    """Write or verify committed VQA V2 artifacts.

    Args:
        output_dir: Destination artifact directory.
        check: Compare without writing when true.

    Returns:
        Written/stale filenames and content hashes.

    Raises:
        ValueError: If check mode finds missing or stale artifacts.
    """

    destination = Path(output_dir).expanduser()
    artifacts = build_vqa_v2_contract_artifacts()
    written: list[str] = []
    stale: list[str] = []
    for filename in ARTIFACT_FILENAMES:
        path = destination / filename
        text = _json_text(artifacts[filename])
        if check:
            if not path.is_file() or path.read_text(
                encoding="utf-8"
            ) != text:
                stale.append(filename)
            continue
        destination.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(filename)
    if stale:
        raise ValueError(
            "VQA V2 contract artifacts are stale or missing: "
            f"{stale}"
        )
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "output_dir": str(destination),
        "checked": check,
        "written": written,
        "stale": stale,
        "artifact_sha256": {
            filename: _payload_sha256(artifacts[filename])
            for filename in ARTIFACT_FILENAMES
        },
    }


def _canonical_source_hashes() -> dict[str, str]:
    """Hash behavior-bearing canonical VQA sources."""

    route_root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "src/mcp_server/contracts/direction_profile.py",
        "src/mcp_server/contracts/vqa_v2.py",
        "src/mcp_server/contracts/vqa_dataset_v2.py",
        "src/mcp_server/agent/vqa_v2_text.py",
        "src/mcp_server/agent/vqa_v2_prompt_policy.py",
        "src/mcp_server/data/vqa/source.py",
        "src/mcp_server/data/vqa/eligibility.py",
        "src/mcp_server/data/vqa/questions.py",
        "src/mcp_server/data/vqa/producers.py",
        "src/mcp_server/data/vqa/views.py",
        "src/mcp_server/data/vqa/audit.py",
        "src/mcp_server/teacher/vqa_v2_profiles.py",
        "src/mcp_server/teacher/vqa_v2_teacher.py",
        "src/mcp_server/teacher/vqa_v2_batch.py",
        "config/vqa_v2_stage_a_capabilities.json",
    )
    return {
        relative: hashlib.sha256(
            (route_root / relative).read_bytes()
        ).hexdigest()
        for relative in relative_paths
    }


def _json_text(payload: Any) -> str:
    """Serialize a stable human-readable artifact."""

    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


def _payload_sha256(payload: Any) -> str:
    """Hash one canonical compact JSON payload."""

    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    """Export requested VQA artifacts and print the report."""

    args = parse_args(argv)
    report = export_vqa_v2_contract_artifacts(
        args.output_dir,
        check=args.check,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
