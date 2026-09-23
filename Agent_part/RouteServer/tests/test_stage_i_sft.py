from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from mcp_server.agent.reasoning_v4_prompt_policy import get_prompt_profile_v4
from mcp_server.agent.reasoning_v4_text import render_reasoning_target_v4
from mcp_server.contracts.reasoning_dataset_v4 import (
    DirectionContextV4,
    InputModalitiesV4,
    NavObservationV4,
    REASONING_SOURCE_V4_SCHEMA_VERSION,
    ReasoningSourceV4,
    ReasoningTrainingViewV4,
    reasoning_source_fingerprint_v4,
)
from mcp_server.contracts.reasoning_v4 import (
    DecisionV4,
    DirectionAffordanceV4,
    InteractionConstraintGroundingV4,
    LocalTraversabilityV4,
    NavigationContextGroundingV4,
    ReasoningTargetV4,
    ReasoningTraceV4,
    SceneGroundingV4,
    TaskDirectionV4,
    TaskGroundingV4,
)
from mcp_server.contracts.stage_i_sft import (
    StageIFullCoTViewV2,
    StageIProjectionDropV1,
    parse_stage_i_training_view_v1,
    parse_stage_i_training_view_v2,
    stage_i_projection_digest_v1,
)
from mcp_server.data.stage_i_batch import (
    project_stage_i_batch_v1,
    project_stage_i_batch_v2,
)
from mcp_server.data.openloop_benchmark import (
    build_stage_i_openloop_reference_v1, )
from mcp_server.data.openloop_lineage import (
    build_openloop_split_lineage_v1,
    freeze_openloop_stage_i_views_v1,
)
from mcp_server.data.stage_i_provenance import (
    build_stage_i_instruction_provenance_v1,
    resolve_stage_i_instruction_role_v1,
)
from mcp_server.data.stage_i_portable import relocate_stage_i_training_views_v1
from mcp_server.data.stage_i_migration_v2 import (
    extract_stage_i_full_cot_manifest_v1,
    upgrade_stage_i_full_cot_manifest_v2,
)
from mcp_server.data.stage_i_views import (
    project_stage_i_reasoning_dropout_from_full_v2,
    project_stage_i_reasoning_view_v1,
    project_stage_i_reasoning_view_with_audit_v1,
    project_stage_i_vqa_view_v1,
    project_stage_i_vqa_view_with_audit_v1,
    project_stage_i_reasoning_view_v2,
)
from mcp_server.data.vqa_v2_eligibility import load_vqa_capabilities_v2
from mcp_server.data.vqa_v2_producers import CanonicalDemonstratedDirectionProducerV2
from mcp_server.data.vqa_v2_questions import build_vqa_questions_v2
from mcp_server.data.vqa_v2_source import build_vqa_source_v2
from tools.model_processing.build_stage_i_instruction_provenance import (
    build_policy_registry, )


def _direction(label: str = "straight",
               *,
               profile_id: str = "fpv_degree10_terminal_centroid_v2") -> dict:
    if profile_id == "ego_degree10_12way_terminal_centroid_v2":
        labels = [
            "stop",
            "straight",
            "left_10",
            "right_10",
            "left_20",
            "right_20",
            "left_30",
            "right_30",
            "left_40",
            "right_40",
            "left_50plus",
            "right_50plus",
        ]
        angles = [
            None, 0.0, -10.0, 10.0, -20.0, 20.0, -30.0, 30.0, -40.0, 40.0,
            -50.0, 50.0
        ]
        bin_mode = "degree10_12way_with_stop"
        derivation = "ego_x_right_y_front_terminal_centroid_rule_v2"
    else:
        labels = [
            *[f"left_{angle}" for angle in range(90, 0, -10)],
            "straight",
            *[f"right_{angle}" for angle in range(10, 100, 10)],
            "stop",
        ]
        angles = [*[float(angle) for angle in range(-90, 91, 10)], None]
        bin_mode = "degree10_with_stop"
        derivation = "pixel_trajectory_terminal_centroid_rule_v2"
    index = labels.index(label)
    return {
        "profile_id": profile_id,
        "profile_version": "2.0",
        "label_index": index,
        "label_name": label,
        "choice": chr(ord("A") + index),
        "angle_deg": angles[index],
        "source": "finish_stop_rule" if label == "stop" else derivation,
        "bin_mode": bin_mode,
        "derivation_version": derivation,
    }


def _target(label: str = "straight") -> ReasoningTargetV4:
    stop = label == "stop"
    return ReasoningTargetV4(reasoning=ReasoningTraceV4(
        task_grounding=TaskGroundingV4(
            objective="follow_route",
            requirements=["route_adherence"],
        ),
        scene_grounding=SceneGroundingV4(
            scene_type="urban_sidewalk",
            walkable_support="paved_sidewalk",
            key_entities=[],
            traffic_context="none_visible",
        ),
        local_traversability=LocalTraversabilityV4(direction_affordances=[
            DirectionAffordanceV4(
                direction_label="straight",
                affordance="direct",
                clearance="sufficient",
            )
        ]),
        navigation_context_grounding=NavigationContextGroundingV4(
            task_direction=TaskDirectionV4(
                status="not_visible",
                source="none",
            )),
        interaction_constraint_grounding=InteractionConstraintGroundingV4(),
        decision=DecisionV4(
            selected_action_label=label,
            speed="stop" if stop else "normal",
            response="stop" if stop else "proceed",
        ),
    ))


def _formal_reasoning_view(
    label: str = "straight",
    *,
    profile_id: str = "fpv_degree10_terminal_centroid_v2",
) -> ReasoningTrainingViewV4:
    target = _target(label)
    profile = get_prompt_profile_v4(
        "urbannav_reasoning_v4_6_compact_full_cot_sft_v1")
    return ReasoningTrainingViewV4.model_validate({
        "sample_id":
        f"sidewalk:case-{label}:formal",
        "source_group_id":
        f"sidewalk:case-{label}",
        "split_group_id":
        "clip-1",
        "dataset_source":
        "sidewalk",
        "release_track":
        "formal_training",
        "audit_id":
        "b" * 64,
        "review_status":
        "not_required",
        "artifact_status":
        "training_ready",
        "artifact_training_eligible":
        True,
        "formal_training_eligible":
        True,
        "images": ["/data/current.jpg"],
        "image_roles": ["fpv.current"],
        "instruction":
        "Follow the route.",
        "model_visible_context": {},
        "evidence_catalog": [
            {
                "ref": "instruction.active",
                "modality": "instruction"
            },
            {
                "ref": "fpv.current",
                "modality": "fpv"
            },
        ],
        "input_modalities":
        InputModalitiesV4(instruction=True).model_dump(mode="json"),
        "direction_context":
        DirectionContextV4(
            action_profile_id=profile_id,
            action_profile_version="2.0",
            affordance_probe_profile_id=(
                "ego_12way_affordance7_v1"
                if profile_id == "ego_degree10_12way_terminal_centroid_v2" else
                "fpv_degree10_affordance7_v1"),
            affordance_probe_profile_version="1.0",
        ).model_dump(mode="json"),
        "assistant_text":
        render_reasoning_target_v4(target),
        "reasoning_target":
        target.model_dump(mode="json", exclude_none=True),
        "prompt_profile_id":
        profile.profile_id,
        "prompt_policy_digest":
        profile.policy_digest,
        "direction_supervision":
        _direction(label, profile_id=profile_id),
        "metadata": {
            "source_fingerprint": "a" * 64,
            "action_reconciliation_audit_id": "b" * 64,
        },
    })


def _vqa_source():
    observation = NavObservationV4(
        source_group_id="sidewalk:vqa-case",
        dataset_source="sidewalk",
        source_record_id="vqa-case",
        split_group_id="clip-vqa",
        images=["current.jpg"],
        image_roles=["fpv.current"],
        instruction="Continue straight.",
        evidence_catalog=[
            {
                "ref": "instruction.active",
                "modality": "instruction"
            },
            {
                "ref": "fpv.current",
                "modality": "fpv"
            },
            {
                "ref": "fpv.obstacle_a",
                "modality": "fpv"
            },
        ],
        input_modalities=InputModalitiesV4(instruction=True),
        direction_context=DirectionContextV4(
            action_profile_id="fpv_degree10_terminal_centroid_v2",
            action_profile_version="2.0",
            affordance_probe_profile_id="fpv_degree10_affordance7_v1",
            affordance_probe_profile_version="1.0",
        ),
    )
    payload = {
        "schema_version": REASONING_SOURCE_V4_SCHEMA_VERSION,
        "observation": observation.model_dump(mode="json"),
        "direction_supervision": _direction(),
        "trajectory_ego_t": None,
        "trajectory_policy": None,
        "action_mask": None,
        "warnings": [],
    }
    payload["source_fingerprint"] = reasoning_source_fingerprint_v4(payload)
    return build_vqa_source_v2(ReasoningSourceV4.model_validate(payload))


def test_full_cot_projection_masks_only_final_action_value() -> None:
    view = project_stage_i_reasoning_view_v1(
        _formal_reasoning_view(),
        task_type="full_cot",
        modality_profile_id="fpv_instruction_v1",
        instruction_training_role="primary",
    )

    assert view.answerability.full_cot
    assert view.training_selection.instruction
    assert view.modality_state.bev == "native_missing"
    zero_spans = [span for span in view.loss_spans if span.scale == 0.0]
    assert len(zero_spans) == 1
    span = zero_spans[0]
    assert view.assistant_text[span.start:span.end] == '"straight"'
    assert parse_stage_i_training_view_v1(view.model_dump(mode="json")) == view
    reference = build_stage_i_openloop_reference_v1(
        view.model_dump(mode="json"))
    assert reference.label_provenance == "lvlm_weak"
    assert reference.field_provenance[
        "decision.selected_action_label"] == "deterministic_gt"
    assert "assistant_text" not in reference.target


def test_full_cot_projection_rejects_native_modality_dropout() -> None:
    with pytest.raises(ValueError, match="dropped_native_instruction"):
        project_stage_i_reasoning_view_v1(
            _formal_reasoning_view(),
            task_type="full_cot",
            modality_profile_id="fpv_current_v1",
            instruction_training_role="unknown",
        )


def test_full_cot_projection_drop_has_structured_reason() -> None:
    outcome = project_stage_i_reasoning_view_with_audit_v1(
        _formal_reasoning_view(),
        task_type="full_cot",
        modality_profile_id="fpv_current_v1",
        instruction_training_role="unknown",
    )

    assert isinstance(outcome, StageIProjectionDropV1)
    assert str(outcome.reason_code) == "full_cot_native_modality_dropout"
    assert len(outcome.drop_digest) == 64


def test_decision_only_projection_has_no_full_cot_prefix_or_system1() -> None:
    view = project_stage_i_reasoning_view_v1(
        _formal_reasoning_view(label="stop"),
        task_type="decision",
        modality_profile_id="fpv_current_v1",
        instruction_training_role="unknown",
        allow_fpv_only_decision=True,
    )
    serialized = view.model_dump(mode="json", exclude_none=True)

    assert view.decision_authority == "fpv_behavior_cloning"
    assert view.assistant_text.startswith("<Decision>")
    assert "<TaskGrounding>" not in view.assistant_text
    assert all(span.scale == 1.0 for span in view.loss_spans)
    assert "trajectory_ego_t" not in serialized
    assert "action_mask" not in serialized
    reference = build_stage_i_openloop_reference_v1(serialized)
    assert reference.direction_stamp is not None
    assert reference.direction_stamp.label_name == "stop"


def test_history_projection_preserves_approved_source_time_and_target():
    from urbannav_contracts.history_observation import HISTORY_4S_2HZ_PROFILE, history_image_roles
    from swift.sidewalk_nav.data.stage_i.stage_i_manifest import build_stage_i_manifest_row

    source = _formal_reasoning_view().model_dump(mode="json", exclude_none=True)
    source["images"] = [f"/data/{i}.jpg" for i in range(9)]
    source["image_roles"] = history_image_roles(include_bev=False)
    source["input_modalities"]["history_fpv"] = True
    source["evidence_catalog"].extend({"ref": role, "modality": "fpv"} for role in source["image_roles"][:-1])
    times = list(range(0, 4_000_000_001, 500_000_000))
    source["metadata"].update({
        "history_observation": {"observation_profile": HISTORY_4S_2HZ_PROFILE,
                                "source_parent_id": source["source_group_id"],
                                "source_ns": times, "target_ns": times, "video_ordinals": list(range(9)),
                                "fpv_paths": source["images"], "bev": None},
        "history_max_lag_ns": 0,
    })
    full = project_stage_i_reasoning_view_v2(
        source, task_type="full_cot", modality_profile_id="fpv_temporal_instruction_v1",
        instruction_training_role="primary")
    assert full.metadata["history_observation"] == source["metadata"]["history_observation"]
    assert full.assistant_text == source["assistant_text"]
    physical = build_stage_i_manifest_row(full.model_dump(mode="json", exclude_none=True))
    assert physical["observation_profile"] == HISTORY_4S_2HZ_PROFILE
    assert physical["frame_offsets_sec"] == [-4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, 0]
    dropout = project_stage_i_reasoning_view_v2(
        source, task_type="reasoning_dropout", modality_profile_id="fpv_temporal_instruction_v1",
        instruction_training_role="primary")
    assert dropout.metadata["history_observation"] == full.metadata["history_observation"]
    source["metadata"]["history_observation"]["source_parent_id"] = "other"
    with pytest.raises(ValueError, match="annotated source parent"):
        project_stage_i_reasoning_view_v2(
            source, task_type="full_cot", modality_profile_id="fpv_temporal_instruction_v1",
            instruction_training_role="primary")


def test_vqa_history_time_changes_fingerprint_and_reaches_stage_i(tmp_path):
    from urbannav_contracts.history_observation import HISTORY_4S_2HZ_PROFILE, history_image_roles
    from mcp_server.data.stage_i.views import project_stage_i_vqa_view_v2
    from swift.sidewalk_nav.data.stage_i.stage_i_manifest import build_stage_i_manifest_row
    from mcp_server.data.vqa.source import build_vqa_source_manifest_v2

    observation = _vqa_source().observation.model_dump(mode="json")
    observation["images"] = [f"/data/{i}.jpg" for i in range(9)]
    observation["image_roles"] = history_image_roles(include_bev=False)
    observation["input_modalities"]["history_fpv"] = True
    observation["evidence_catalog"].extend({"ref": role, "modality": "fpv"}
                                          for role in observation["image_roles"][:-1])
    observation = NavObservationV4.model_validate(observation).model_dump(mode="json")
    payload = {"schema_version": REASONING_SOURCE_V4_SCHEMA_VERSION, "observation": observation,
               "direction_supervision": _direction(), "trajectory_ego_t": None,
               "trajectory_policy": None, "action_mask": None, "warnings": []}
    payload["source_fingerprint"] = reasoning_source_fingerprint_v4(payload)
    source = ReasoningSourceV4.model_validate(payload)
    old = build_vqa_source_v2(source)
    times = list(range(0, 4_000_000_001, 500_000_000))
    history = {"observation_profile": HISTORY_4S_2HZ_PROFILE, "source_parent_id": observation["source_group_id"],
               "source_ns": times, "target_ns": times, "video_ordinals": list(range(9)),
               "fpv_paths": observation["images"], "bev": None}
    current = build_vqa_source_v2(source, history_observation=history, history_max_lag_ns=0)
    assert old.source_fingerprint != current.source_fingerprint
    assert current.observation.images == old.observation.images
    # canonical JSONL 是批量构造的真实边界；时间合同随 evidence 一起进入新 source 身份。
    payload["observation"] = current.observation.model_dump(mode="json")
    payload.pop("source_fingerprint")
    payload["source_fingerprint"] = reasoning_source_fingerprint_v4(payload)
    input_path, output_path = tmp_path / "reasoning.jsonl", tmp_path / "vqa.jsonl"
    input_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    report = build_vqa_source_manifest_v2(input_path, output_path)
    assert report["counts"]["output_rows"] == 1
    assert json.loads(output_path.read_text())["source_fingerprint"] == current.source_fingerprint
    attributes = next(item["attributes"] for item in payload["observation"]["evidence_catalog"]
                      if item["ref"] == "fpv.current")
    attributes["history_observation"]["source_parent_id"] = "other"
    payload.pop("source_fingerprint")
    payload["source_fingerprint"] = reasoning_source_fingerprint_v4(payload)
    input_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="visible parent"):
        build_vqa_source_manifest_v2(input_path, tmp_path / "invalid.jsonl")
    capabilities = load_vqa_capabilities_v2()
    annotations = []
    for item in (old, current):
        question = next(q for q in build_vqa_questions_v2(item, capabilities).questions
                        if q.task_id == "e0_demonstrated_direction_selection")
        annotations.append(CanonicalDemonstratedDirectionProducerV2().produce(item, question, capabilities).annotation)
    kwargs = dict(capabilities=capabilities, modality_profile_id="fpv_temporal_instruction_v1",
                  instruction_training_role="primary")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        project_stage_i_vqa_view_v2(current, annotations[0], **kwargs)
    view = project_stage_i_vqa_view_v2(current, annotations[1], **kwargs)
    row = build_stage_i_manifest_row(view.model_dump(mode="json", exclude_none=True))
    assert row["observation_profile"] == HISTORY_4S_2HZ_PROFILE
    assert row["frame_timestamps_ns"] == times


@pytest.mark.parametrize("bev_profile", ["causal_map_v1", "recorded_bev_v1"])
def test_materialized_history_builds_fresh_reasoning_source(tmp_path, bev_profile):
    """Package inputs preserve time and private direction, while rebuilding evidence."""
    from urbannav_contracts.history_observation import HISTORY_4S_2HZ_PROFILE, history_image_roles
    from mcp_server.data.canonical.reasoning_v4_source import build_reasoning_source_manifest_v4
    from mcp_server.data.legacy.reasoning_v2.source import ReasoningSourceBuildConfig

    paths = [f"{i}.jpg" for i in range(9)]
    for path in paths + ["bev.jpg"]:
        (tmp_path / path).write_bytes(b"fixture")
    times = list(range(0, 4_000_000_001, 500_000_000))
    history = {"observation_profile": HISTORY_4S_2HZ_PROFILE, "source_parent_id": "sidewalk:parent",
               "source_ns": times, "target_ns": times, "video_ordinals": list(range(9)),
               "fpv_paths": paths, "bev": {"bev_image_path": "bev.jpg", "input_profile": bev_profile,
                   "draw_trajectory": bev_profile == "recorded_bev_v1",
                   "bev_route_scope": "global_full" if bev_profile == "recorded_bev_v1" else "none",
                   "bev_alignment": "ego_aligned", "timestamp": times[-1], "age_ns": 0}}
    original = {"images": ["old.jpg"], "direction_supervision": _direction(),
                "direction_raw_angle_deg": 1.25, "direction_displacement": 0.4,
                "instruction": "Continue straight.", "route_context": "old complete route",
                "messages": [{"role": "assistant", "content": "old annotation"}]}
    record = {"source_parent_id": "sidewalk:parent", "split_group_id": "clip", "split": "train",
              "observation": history, "source_row": original, "trajectory_sidecar": None}
    input_path, output_path = tmp_path / "observations.jsonl", tmp_path / "reasoning.jsonl"
    input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    report = build_reasoning_source_manifest_v4(input_path, output_path,
        config=ReasoningSourceBuildConfig(dataset_source="sidewalk", media_root=tmp_path),
        history_max_lag_ns=0)
    assert report["counts"]["output_rows"] == 1
    result = ReasoningSourceV4.model_validate(json.loads(output_path.read_text()))
    assert result.observation.image_roles == history_image_roles(include_bev=True)
    assert result.observation.source_group_id == record["source_parent_id"]
    assert result.observation.split_group_id == "clip"
    bev = next(entry for entry in result.observation.evidence_catalog if entry.ref == "bev.current")
    assert bev.bev_route_evidence.route_overlay_visible == (bev_profile == "recorded_bev_v1")
    vqa = build_vqa_source_v2(result)
    current = next(entry for entry in vqa.observation.evidence_catalog if entry.ref == "fpv.current")
    assert current.attributes["history_observation"]["source_ns"] == times
    assert current.attributes["history_observation"]["fpv_paths"] == result.observation.images[:9]
    assert result.direction_supervision.model_dump(mode="json") == _direction()
    assert result.direction_geometry.raw_angle_deg == 1.25
    assert result.direction_geometry.displacement == 0.4
    assert result.direction_geometry.angle_valid is True
    assert result.trajectory_ego_t is None
    assert "old annotation" not in output_path.read_text()


def test_v2_full_cot_supervises_decision_and_dropout_is_parent_linked(
) -> None:
    """V2 fixes the V1 action mask without creating a second action teacher."""

    full = project_stage_i_reasoning_view_v2(
        _formal_reasoning_view(label="left_20"),
        task_type="full_cot",
        modality_profile_id="fpv_instruction_v1",
        instruction_training_role="primary",
    )
    assert full.task_type == "full_cot"
    assert len(full.loss_spans) == 1
    assert full.loss_spans[0].scale == 1.0
    assert full.assistant_text[full.loss_spans[0].start:full.loss_spans[0].
                               end] == full.assistant_text
    assert '"selected_action_label":"left_20"' in full.assistant_text

    dropout = project_stage_i_reasoning_view_v2(
        _formal_reasoning_view(label="left_20"),
        task_type="reasoning_dropout",
        modality_profile_id="fpv_instruction_v1",
        instruction_training_role="primary",
    )
    assert dropout.task_type == "reasoning_dropout"
    assert dropout.assistant_text.startswith("<Decision>")
    assert "<TaskGrounding>" not in dropout.assistant_text
    assert dropout.dropout_link.dropout_reason == "assistant_reasoning_prefix"
    assert dropout.dropout_link.parent_full_cot_sample_id.endswith(
        ":stage_i:fpv_instruction_v1")
    assert all(span.scale == 1.0 for span in dropout.loss_spans)


def test_v2_full_cot_has_a_deterministic_no_cot_projection_primitive() -> None:
    """The epoch compiler may derive Decision only after routing a FullCoT parent."""

    full = project_stage_i_reasoning_view_v2(
        _formal_reasoning_view(label="right_20"),
        task_type="full_cot",
        modality_profile_id="fpv_instruction_v1",
        instruction_training_role="primary",
    )
    dropout = project_stage_i_reasoning_dropout_from_full_v2(full)
    assert dropout.dropout_link.parent_full_cot_sample_id == full.sample_id
    assert dropout.dropout_link.parent_full_cot_assistant_sha256 == hashlib.sha256(
        full.assistant_text.encode("utf-8")).hexdigest()
    assert dropout.decision.selected_action_label == "right_20"

    with pytest.raises(ValueError, match="requires a V2 FullCoT parent"):
        project_stage_i_reasoning_dropout_from_full_v2(
            dropout.model_dump(mode="json"))


def test_v2_waypoint_profile_retains_independent_visible_condition() -> None:
    """V2 waypoint input reaches the prompt contract without target leakage."""

    payload = _formal_reasoning_view(label="left_20").model_dump(mode="json")
    payload["input_modalities"]["waypoint"] = True
    payload["evidence_catalog"].append({
        "ref": "waypoint.active",
        "modality": "waypoint",
        "summary": "independent planner route",
    })
    payload["model_visible_context"] = {
        "waypoint": {
            "goal_geometry": {
                "schema_version": "urbannav_goal_geometry_v2",
                "coordinate_frame": "current_ego_flu",
                "points": [{"x": 8.0, "y": 1.0, "kind": "goal"}],
            },
            "waypoint_input": {
                "available": True,
                "independent_of_future_target": True,
                "source": "independent_gps_route",
            },
        }
    }
    source = ReasoningTrainingViewV4.model_validate(payload)
    projected = project_stage_i_reasoning_view_v2(
        source,
        task_type="full_cot",
        modality_profile_id="fpv_instruction_waypoint_v2",
        instruction_training_role="primary",
    )

    assert projected.training_selection.waypoint is True
    assert projected.input_modalities.waypoint is True
    assert "waypoint" in projected.model_visible_context
    assert any(item.modality == "waypoint" for item in projected.evidence_catalog)


def test_v2_batch_materializes_parents_but_not_dropout_rows(
        tmp_path: Path) -> None:
    """The epoch compiler, not the source pool, owns dropout selection."""

    parent = _formal_reasoning_view(label="left_20")
    parent_path = tmp_path / "reasoning.jsonl"
    parent_path.write_text(
        json.dumps(parent.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    provenance = build_stage_i_instruction_provenance_v1(
        source_group_id=parent.source_group_id,
        source_fingerprint=str(parent.metadata["source_fingerprint"]),
        instruction=parent.instruction,
        training_role="primary",
        provenance_source="explicit_source_registry",
        policy_id="stage_i_test_instruction_policy",
        policy_version="1.0",
    )
    provenance_path = tmp_path / "provenance.jsonl"
    provenance_path.write_text(
        json.dumps(provenance.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )

    report = project_stage_i_batch_v2(
        output_view_jsonl=tmp_path / "views.jsonl",
        output_drop_jsonl=tmp_path / "drops.jsonl",
        instruction_provenance_jsonl=provenance_path,
        reasoning_view_jsonls=[parent_path],
        modality_profile_ids=["fpv_instruction_v1"],
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "views.jsonl").read_text().splitlines()
    ]

    assert report["training_contract"] == "v2"
    assert report["counts"] == {
        "mode:fpv_instruction_v1": 1,
        "task:full_cot": 1,
        "views": 1,
    }
    assert report["reasoning_dropout_policy"] == {
        "materialized_by_batch_projector": False,
        "source": "deterministic_projection_after_epoch_route",
        "eligible_parent_count": 1,
    }
    assert len(rows) == 1
    assert isinstance(parse_stage_i_training_view_v2(rows[0]),
                      StageIFullCoTViewV2)
    assert all(row["task_type"] != "reasoning_dropout" for row in rows)


def test_mixed_v1_source_selection_is_validated_and_full_cot_only(
        tmp_path: Path) -> None:
    """Formal mixed sources become pure FullCoT before V2 migration."""

    full = project_stage_i_reasoning_view_v1(
        _formal_reasoning_view(label="left_20"),
        task_type="full_cot",
        modality_profile_id="fpv_instruction_v1",
        instruction_training_role="primary",
    )
    decision = project_stage_i_reasoning_view_v1(
        _formal_reasoning_view(label="stop"),
        task_type="decision",
        modality_profile_id="fpv_current_v1",
        instruction_training_role="unknown",
        allow_fpv_only_decision=True,
    )
    source = tmp_path / "mixed_v1.jsonl"
    source.write_text(
        "".join(
            json.dumps(view.model_dump(mode="json", exclude_none=True)) + "\n"
            for view in (decision, full)),
        encoding="utf-8",
    )
    selected = tmp_path / "full_v1.jsonl"
    report = extract_stage_i_full_cot_manifest_v1(
        source,
        selected,
        report_json=tmp_path / "selection.json",
    )
    rows = [json.loads(line) for line in selected.read_text().splitlines()]

    assert len(rows) == 1
    assert rows[0]["sample_id"] == full.sample_id
    assert parse_stage_i_training_view_v1(rows[0]).task_type == "full_cot"
    assert report["counts"] == {
        "input_rows": 2,
        "output_full_cot_rows": 1,
        "skipped_non_full_cot": 1,
        "task:decision": 1,
        "task:full_cot": 1,
    }
    assert report["selection_contract"]["payload_mutation"] is False


def test_v1_full_cot_migration_is_fail_closed_and_parent_only(
        tmp_path: Path) -> None:
    """Existing formal FullCoT can enter V2 without promoting V1 Decision."""

    full = project_stage_i_reasoning_view_v1(
        _formal_reasoning_view(label="right_20"),
        task_type="full_cot",
        modality_profile_id="fpv_instruction_v1",
        instruction_training_role="primary",
    )
    source = tmp_path / "full_v1.jsonl"
    source.write_text(
        json.dumps(full.model_dump(mode="json", exclude_none=True)) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "full_v2.jsonl"
    report = upgrade_stage_i_full_cot_manifest_v2(source, output)
    upgraded = parse_stage_i_training_view_v2(json.loads(output.read_text()))

    assert isinstance(upgraded, StageIFullCoTViewV2)
    assert upgraded.images == full.images
    assert upgraded.assistant_text == full.assistant_text
    assert upgraded.metadata[
        "source_stage_i_projection_digest_v1"] == full.projection_digest
    assert report["migration_contract"][
        "reasoning_dropout_materialized"] is False

    image = tmp_path / "new-media" / "current.jpg"
    image.parent.mkdir()
    image.write_bytes(b"stage-i-v2-path-map")
    mapped_payload = full.model_dump(mode="json", exclude_none=True)
    mapped_payload["images"] = ["/retired/stage-i/current.jpg"]
    mapped_payload.pop("projection_digest")
    mapped_payload["projection_digest"] = stage_i_projection_digest_v1(
        mapped_payload)
    source.write_text(json.dumps(mapped_payload) + "\n", encoding="utf-8")
    mapped_output = tmp_path / "mapped_v2.jsonl"
    mapped_report = upgrade_stage_i_full_cot_manifest_v2(
        source,
        mapped_output,
        image_prefix_maps=[("/retired/stage-i", image.parent)],
    )
    mapped = parse_stage_i_training_view_v2(
        json.loads(mapped_output.read_text()))
    assert mapped.images == [str(image.resolve())]
    assert mapped_report["migration_contract"][
        "rewritten_image_references"] == 1

    decision = project_stage_i_reasoning_view_v1(
        _formal_reasoning_view(),
        task_type="decision",
        modality_profile_id="fpv_current_v1",
        instruction_training_role="unknown",
        allow_fpv_only_decision=True,
    )
    source.write_text(
        json.dumps(decision.model_dump(mode="json", exclude_none=True)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only V1 FullCoT"):
        upgrade_stage_i_full_cot_manifest_v2(source,
                                             tmp_path / "invalid.jsonl")


def test_selected_instruction_must_be_primary() -> None:
    with pytest.raises(ValueError, match="trusted primary"):
        project_stage_i_reasoning_view_v1(
            _formal_reasoning_view(),
            task_type="decision",
            modality_profile_id="fpv_instruction_v1",
            instruction_training_role="audit_only",
            allow_fpv_only_decision=True,
        )


def test_decision_projection_drops_bev_route_grounding_with_bev() -> None:
    payload = _formal_reasoning_view().model_dump(mode="json")
    payload["images"].append("/data/bev.jpg")
    payload["image_roles"].append("bev.current")
    payload["evidence_catalog"].append({
        "ref": "bev.current",
        "modality": "bev",
        "attributes": {},
        "bev_route_evidence": {
            "frame": "ego",
            "ego_forward": "up",
            "route_overlay_visible": True,
            "route_direction_authority": "visual_conditional",
            "route_progress_authority": "visual_conditional",
            "route_terminal_marker_visible": False,
            "route_endpoint_semantics": "unknown",
            "local_safety_authority": False,
            "producer_contract_id": "test_bev_overlay",
            "producer_contract_version": "1.0",
        },
    })
    payload["input_modalities"]["bev"] = True
    payload["reasoning_target"]["reasoning"][
        "navigation_context_grounding"] = {
            "task_direction": {
                "status": "aligned",
                "source": "bev_route",
                "direction": "straight",
            }
        }
    target = ReasoningTargetV4.model_validate(payload["reasoning_target"])
    payload["assistant_text"] = render_reasoning_target_v4(target)
    parent = ReasoningTrainingViewV4.model_validate(payload)

    view = project_stage_i_reasoning_view_v1(
        parent,
        task_type="decision",
        modality_profile_id="fpv_current_v1",
        instruction_training_role="unknown",
        allow_fpv_only_decision=True,
    )
    serialized = view.model_dump(mode="json", exclude_none=True)

    assert view.decision_authority == "fpv_behavior_cloning"
    assert not view.training_selection.bev
    assert all(item.modality != "bev" for item in view.evidence_catalog)
    assert "bev_route" not in str(serialized)


def test_decision_projection_rejects_non_native_mode_selection() -> None:
    with pytest.raises(ValueError, match="not native: bev"):
        project_stage_i_reasoning_view_v1(
            _formal_reasoning_view(),
            task_type="decision",
            modality_profile_id="fpv_bev_v1",
            instruction_training_role="unknown",
            allow_fpv_only_decision=True,
        )


def test_vqa_projection_keeps_annotation_evidence() -> None:
    source = _vqa_source()
    capabilities = load_vqa_capabilities_v2()
    question = next(
        item for item in build_vqa_questions_v2(source, capabilities).questions
        if item.task_id == "e0_demonstrated_direction_selection")
    produced = CanonicalDemonstratedDirectionProducerV2().produce(
        source, question, capabilities)
    assert produced.annotation is not None

    view = project_stage_i_vqa_view_v1(
        source,
        produced.annotation,
        capabilities=capabilities,
        modality_profile_id="fpv_instruction_v1",
        instruction_training_role="primary",
    )

    assert view.answerability.vqa
    assert view.instruction == "Continue straight."
    assert view.modality_state.instruction == "present"
    assert "fpv.obstacle_a" in {
        str(item.ref)
        for item in view.evidence_catalog
    }
    assert set(view.annotation_evidence_refs).issubset(
        {str(item.ref)
         for item in view.evidence_catalog})
    reference = build_stage_i_openloop_reference_v1(
        view.model_dump(mode="json"))
    assert reference.label_provenance == "deterministic_gt"
    assert reference.target["question"]["task_id"] == (
        "e0_demonstrated_direction_selection")


def test_vqa_projection_rejects_removed_annotation_evidence() -> None:
    source = _vqa_source()
    capabilities = load_vqa_capabilities_v2()
    question = next(
        item for item in build_vqa_questions_v2(source, capabilities).questions
        if item.task_id == "e0_demonstrated_direction_selection")
    produced = CanonicalDemonstratedDirectionProducerV2().produce(
        source, question, capabilities)
    assert produced.annotation is not None
    # Legacy annotations cited the complete source envelope.  Preserve this
    # regression fixture to prove the projector still fails closed instead of
    # silently treating a minimal-evidence template as already re-annotated.
    legacy_question = produced.annotation.question.model_copy(
        update={
            "evidence_refs":
            [str(item.ref) for item in source.observation.evidence_catalog]
        })
    legacy_annotation = produced.annotation.model_copy(
        update={
            "question": legacy_question,
            "evidence_refs": list(legacy_question.evidence_refs),
        })

    with pytest.raises(ValueError, match="removed annotation evidence"):
        project_stage_i_vqa_view_v1(
            source,
            legacy_annotation,
            capabilities=capabilities,
            modality_profile_id="fpv_current_v1",
            instruction_training_role="unknown",
        )

    outcome = project_stage_i_vqa_view_with_audit_v1(
        source,
        legacy_annotation,
        capabilities=capabilities,
        modality_profile_id="fpv_current_v1",
        instruction_training_role="unknown",
    )
    assert isinstance(outcome, StageIProjectionDropV1)
    assert str(outcome.reason_code) == "vqa_annotation_evidence_removed"


def test_projection_digest_rejects_mutation() -> None:
    view = project_stage_i_reasoning_view_v1(
        _formal_reasoning_view(),
        task_type="decision",
        modality_profile_id="fpv_current_v1",
        instruction_training_role="unknown",
        allow_fpv_only_decision=True,
    )
    payload = copy.deepcopy(view.model_dump(mode="json"))
    payload["assistant_text"] = payload["assistant_text"].replace(
        "straight", "straighx")
    with pytest.raises(ValidationError, match="projection_digest"):
        parse_stage_i_training_view_v1(payload)


def test_stage_i_portable_relocation_rehashes_and_validates(
    tmp_path: Path, ) -> None:
    """A stale explicit prefix is repaired without basename guessing."""

    media_source = tmp_path / "current-media" / "clip" / "current.jpg"
    media_source.parent.mkdir(parents=True)
    media_source.write_bytes(b"portable-stage-i-image")
    view = project_stage_i_reasoning_view_v1(
        _formal_reasoning_view(),
        task_type="decision",
        modality_profile_id="fpv_current_v1",
        instruction_training_role="unknown",
        allow_fpv_only_decision=True,
    ).model_dump(mode="json", exclude_none=True)
    stale_root = Path("/retired/l20x/stage-i-media")
    view["images"] = [str(stale_root / "clip" / "current.jpg")]
    view.pop("projection_digest")
    view["projection_digest"] = stage_i_projection_digest_v1({
        key: value
        for key, value in view.items() if key != "projection_digest"
    })
    input_jsonl = tmp_path / "input.jsonl"
    input_jsonl.write_text(json.dumps(view) + "\n", encoding="utf-8")

    output_root = tmp_path / "portable"
    report = relocate_stage_i_training_views_v1(
        input_jsonl,
        output_root,
        source_prefix_maps=[(stale_root, media_source.parents[1])],
    )
    relocated = json.loads(
        Path(report["output_jsonl"]).read_text(encoding="utf-8"))

    parsed = parse_stage_i_training_view_v1(relocated)
    assert parsed.images[0].startswith("media/")
    assert not Path(parsed.images[0]).is_absolute()
    assert (output_root /
            parsed.images[0]).read_bytes() == media_source.read_bytes()
    assert report["row_count"] == 1
    assert report["prefix_repaired_reference_count"] == 1
    assert report["root_image_dir_for_consumers"] == str(output_root.resolve())


def test_stage_i_portable_relocation_rejects_unmapped_missing_media(
    tmp_path: Path, ) -> None:
    """Missing media fails closed instead of searching for a matching name."""

    view = project_stage_i_reasoning_view_v1(
        _formal_reasoning_view(),
        task_type="decision",
        modality_profile_id="fpv_current_v1",
        instruction_training_role="unknown",
        allow_fpv_only_decision=True,
    ).model_dump(mode="json", exclude_none=True)
    input_jsonl = tmp_path / "input.jsonl"
    input_jsonl.write_text(json.dumps(view) + "\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError,
                       match="cannot resolve Stage I image"):
        relocate_stage_i_training_views_v1(input_jsonl, tmp_path / "portable")


def test_instruction_provenance_requires_exact_source_revision_and_text(
) -> None:
    view = _formal_reasoning_view()
    source_fingerprint = str(view.metadata["source_fingerprint"])
    entry = build_stage_i_instruction_provenance_v1(
        source_group_id=view.source_group_id,
        source_fingerprint=source_fingerprint,
        instruction=view.instruction,
        training_role="primary",
        provenance_source="explicit_source_registry",
        policy_id="stage_i_test_instruction_policy",
        policy_version="1.0",
    )
    registry = {(entry.source_group_id, entry.source_fingerprint): entry}

    assert resolve_stage_i_instruction_role_v1(
        source_group_id=view.source_group_id,
        source_fingerprint=source_fingerprint,
        instruction=view.instruction,
        registry=registry,
    ) == "primary"
    with pytest.raises(ValueError, match="text digest"):
        resolve_stage_i_instruction_role_v1(
            source_group_id=view.source_group_id,
            source_fingerprint=source_fingerprint,
            instruction="Changed instruction.",
            registry=registry,
        )
    with pytest.raises(ValueError, match="provenance is missing"):
        resolve_stage_i_instruction_role_v1(
            source_group_id=view.source_group_id,
            source_fingerprint="f" * 64,
            instruction=view.instruction,
            registry=registry,
        )


def test_instruction_policy_builds_primary_registry_and_fails_closed(
    tmp_path: Path, ) -> None:
    view = _formal_reasoning_view()
    view_path = tmp_path / "reasoning.jsonl"
    view_path.write_text(
        json.dumps(view.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    instruction = str(view.instruction)
    instruction_digest = hashlib.sha256(
        instruction.encode("utf-8")).hexdigest()
    policy_payload = {
        "schema_version":
        "urbannav_stage_i_instruction_policy_v1",
        "policy_id":
        "stage_i_test_exact_instruction_policy",
        "policy_version":
        "1.0",
        "decision_authority":
        "test_fixture",
        "unmatched_instruction_policy":
        "fail_closed",
        "decisions": [{
            "instruction":
            instruction,
            "instruction_sha256":
            instruction_digest,
            "training_role":
            "primary",
            "provenance_source":
            "test_visible_task_contract",
            "rationale":
            "The fixture instruction is prompt-visible.",
        }],
    }
    digest_payload = json.dumps(
        policy_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    policy_payload["policy_digest"] = hashlib.sha256(
        digest_payload.encode("utf-8")).hexdigest()
    policy_path = tmp_path / "instruction-policy.json"
    policy_path.write_text(
        json.dumps(policy_payload),
        encoding="utf-8",
    )
    registry_path = tmp_path / "instruction-registry.jsonl"
    report = build_policy_registry(
        policy_path,
        registry_path,
        reasoning_view_jsonls=[view_path],
        report_json=tmp_path / "report.json",
    )

    assert report["mode"] == "exact_policy"
    assert report["training_role_counts"] == {"primary": 1}
    projected = project_stage_i_batch_v1(
        output_view_jsonl=tmp_path / "views.jsonl",
        output_drop_jsonl=tmp_path / "drops.jsonl",
        instruction_provenance_jsonl=registry_path,
        reasoning_view_jsonls=[view_path],
        modality_profile_ids=["fpv_instruction_v1"],
    )
    assert projected["counts"]["task:full_cot"] == 1

    changed = view.model_copy(
        update={
            "sample_id": "sidewalk:changed:formal",
            "source_group_id": "sidewalk:changed",
            "instruction": "A new unreviewed instruction.",
            "metadata": {
                **view.metadata,
                "source_fingerprint": "c" * 64,
            },
        })
    changed_path = tmp_path / "changed.jsonl"
    changed_path.write_text(
        json.dumps(changed.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no exact decision"):
        build_policy_registry(
            policy_path,
            tmp_path / "rejected.jsonl",
            reasoning_view_jsonls=[changed_path],
        )


def test_batch_projector_requires_registry_and_emits_drop_audit(
        tmp_path: Path) -> None:
    view = _formal_reasoning_view()
    parent_path = tmp_path / "reasoning.jsonl"
    parent_path.write_text(
        json.dumps(view.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    empty_registry = tmp_path / "empty-provenance.jsonl"
    empty_registry.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance is missing"):
        project_stage_i_batch_v1(
            output_view_jsonl=tmp_path / "rejected-views.jsonl",
            output_drop_jsonl=tmp_path / "rejected-drops.jsonl",
            instruction_provenance_jsonl=empty_registry,
            reasoning_view_jsonls=[parent_path],
            modality_profile_ids=["fpv_instruction_v1"],
        )

    entry = build_stage_i_instruction_provenance_v1(
        source_group_id=view.source_group_id,
        source_fingerprint=str(view.metadata["source_fingerprint"]),
        instruction=view.instruction,
        training_role="primary",
        provenance_source="explicit_source_registry",
        policy_id="stage_i_test_instruction_policy",
        policy_version="1.0",
    )
    registry_path = tmp_path / "provenance.jsonl"
    registry_path.write_text(
        json.dumps(entry.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    report = project_stage_i_batch_v1(
        output_view_jsonl=tmp_path / "views.jsonl",
        output_drop_jsonl=tmp_path / "drops.jsonl",
        instruction_provenance_jsonl=registry_path,
        reasoning_view_jsonls=[parent_path],
        modality_profile_ids=["fpv_instruction_v1"],
    )

    assert report["counts"]["views"] == 1
    assert report["counts"]["drops"] == 1
    assert report["counts"]["drop:decision_no_visible_authority"] == 1


def test_all_stage_i_task_views_build_ms_swift_physical_rows() -> None:
    ms_root = Path(__file__).resolve().parents[4] / "ms-swift"
    if not ms_root.is_dir():
        pytest.skip("sibling ms-swift checkout is unavailable")
    sys.path.insert(0, str(ms_root))
    try:
        adapter = importlib.import_module(
            "swift.sidewalk_nav.data.stage_i.stage_i_manifest")
        full_cot = project_stage_i_reasoning_view_v1(
            _formal_reasoning_view(),
            task_type="full_cot",
            modality_profile_id="fpv_instruction_v1",
            instruction_training_role="primary",
        )
        decision = project_stage_i_reasoning_view_v1(
            _formal_reasoning_view(),
            task_type="decision",
            modality_profile_id="fpv_current_v1",
            instruction_training_role="unknown",
            allow_fpv_only_decision=True,
        )
        decision_12way = project_stage_i_reasoning_view_v1(
            _formal_reasoning_view(
                profile_id="ego_degree10_12way_terminal_centroid_v2"),
            task_type="decision",
            modality_profile_id="fpv_current_v1",
            instruction_training_role="unknown",
            allow_fpv_only_decision=True,
        )
        source = _vqa_source()
        capabilities = load_vqa_capabilities_v2()
        question = next(
            item
            for item in build_vqa_questions_v2(source, capabilities).questions
            if item.task_id == "e0_demonstrated_direction_selection")
        produced = CanonicalDemonstratedDirectionProducerV2().produce(
            source,
            question,
            capabilities,
        )
        assert produced.annotation is not None
        vqa = project_stage_i_vqa_view_v1(
            source,
            produced.annotation,
            capabilities=capabilities,
            modality_profile_id="fpv_instruction_v1",
            instruction_training_role="primary",
        )

        rows = [
            adapter.build_stage_i_manifest_row(view.model_dump(mode="json"))
            for view in (full_cot, vqa, decision, decision_12way)
        ]
        assert [row["task_type"] for row in rows] == [
            "full_cot",
            "vqa",
            "decision",
            "decision",
        ]
        assert rows[-1]["direction_profile_id"] == (
            "ego_degree10_12way_terminal_centroid_v2")
        assert all(row["loss_mask"]["lm"] for row in rows)
    finally:
        sys.path.remove(str(ms_root))


def test_split_lineage_uses_explicit_nested_fields_and_freezes_views(
    tmp_path: Path, ) -> None:

    def write_source(name: str, row_id: str, clip_id: str) -> Path:
        path = tmp_path / name
        path.write_text(
            json.dumps({
                "id": row_id,
                "source_dataset": "sidewalk_main_multiframe",
                "sidewalk_meta": {
                    "experiment_group": {
                        "clip_id": clip_id
                    }
                },
            }) + "\n",
            encoding="utf-8",
        )
        return path

    write_source("train.jsonl", "train-row", "clip-train")
    write_source("validation.jsonl", "validation-row", "clip-1")
    write_source("test.jsonl", "test-row", "clip-test")
    spec = {
        "schema_version":
        "urbannav_openloop_split_source_spec_v1",
        "manifests": [{
            "manifest_id":
            f"legacy-{split}",
            "path":
            f"{split}.jsonl",
            "split":
            split,
            "parent_field":
            "id",
            "split_group_field": ("sidewalk_meta.experiment_group.clip_id"),
            "dataset_field":
            "source_dataset",
            "dataset_aliases": {
                "sidewalk_main_multiframe": "sidewalk"
            },
        } for split in ("train", "validation", "test")],
    }
    spec_path = tmp_path / "split-spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    report = build_openloop_split_lineage_v1(
        spec_path,
        tmp_path / "lineage.jsonl",
        tmp_path / "lineage-report.json",
    )
    assert report["cross_split_group_overlap"] == 0
    assert report["emitted_rows"] == 3

    view = project_stage_i_reasoning_view_v1(
        _formal_reasoning_view(),
        task_type="full_cot",
        modality_profile_id="fpv_instruction_v1",
        instruction_training_role="primary",
    )
    view_path = tmp_path / "views.jsonl"
    view_path.write_text(
        json.dumps(view.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    freeze_report = freeze_openloop_stage_i_views_v1(
        tmp_path / "lineage.jsonl",
        [view_path],
        tmp_path / "frozen-validation.jsonl",
        tmp_path / "frozen-validation-report.json",
        split="validation",
    )
    assert freeze_report["selected_rows"] == 1
    assert freeze_report["selected_group_count"] == 1

    write_source("test.jsonl", "test-row", "clip-1")
    with pytest.raises(ValueError, match="split-group leakage"):
        build_openloop_split_lineage_v1(
            spec_path,
            tmp_path / "invalid-lineage.jsonl",
            tmp_path / "invalid-lineage-report.json",
        )

    spec["overlap_policy"] = "quarantine_evaluation_conflicts_v1"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    quarantined = build_openloop_split_lineage_v1(
        spec_path,
        tmp_path / "safe-lineage.jsonl",
        tmp_path / "safe-lineage-report.json",
        quarantine_jsonl=tmp_path / "safe-lineage-quarantine.jsonl",
    )
    assert quarantined["source_overlap_group_count"] == 1
    assert quarantined["quarantined_rows"] == 2
    assert quarantined["cross_split_group_overlap"] == 0
