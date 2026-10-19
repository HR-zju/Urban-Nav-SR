from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from mcp_server.agent.reasoning_v4_text import render_reasoning_section_v4
from mcp_server.contracts.stage_i_sft import stage_i_projection_digest_v1
from mcp_server.contracts.stage_ii_cfm import (
    StageIIContextCacheV1,
    StageIIContextProducerV1,
    StageIITrajectorySourceRegistryV1,
    stage_ii_digest_v1,
)
from mcp_server.data.stage_i.views import project_stage_i_reasoning_view_v1
from mcp_server.data.openloop.benchmark import build_stage_ii_openloop_reference_from_parent_v1
from mcp_server.data.stage_ii.context import build_stage_ii_context_cache_v1, stage_ii_projected_input_digest_v1
from mcp_server.data.stage_ii.legacy_direct import build_stage_ii_legacy_direct_selection_v1, build_stage_ii_legacy_direct_targets_v1
from mcp_server.data.stage_ii.trajectory import build_stage_ii_trajectory_target_v1
from mcp_server.data.stage_ii.sources import StageIISourceAdapterErrorV1, adapt_stage_ii_trajectory_source_row_v1, load_stage_ii_trajectory_source_registry_v1
from mcp_server.data.stage_ii.sidecars import build_stage_ii_trajectory_sidecars_v1
from mcp_server.data.stage_ii.views import project_stage_ii_cfm_view_v1
from tools.model_processing.build_stage_ii_cfm_views import (
    build_stage_ii_cfm_views,
)
from tools.model_processing.audit_stage_ii_trajectory_targets import (
    audit_stage_ii_trajectory_targets,
)

from test_stage_i_sft import _formal_reasoning_view


ROUTE_ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _source_registry_payload(
    *,
    train_ratio: float = 1.0,
    sidewalk_sources=("calibrated_pose_projection", "stationary_pose_hold"),
):
    split = {
        "policy_id": "stage_i_group_sha256_colon_v1",
        "seed": 20260731,
        "train": train_ratio,
        "validation": 0.0,
        "test": 1.0 - train_ratio,
    }
    common = {
        "sample_identity_fields": ["sidewalk_meta.clip_id", "sidewalk_meta.frame_idx"],
        "split_group_fields": ["sidewalk_meta.clip_id"],
        "trajectory_field": "trajectory_ego_t",
        "source_format": "ego_xy_t_speed_right_front_v1",
        "trajectory_source_fields": [
            "trajectory_policy.trajectory_source",
            "sidewalk_meta.trajectory_source",
        ],
        "yaw_source": "path_tangent",
        "mask_policy": "validated_timed_rows_v1",
        "action_mask_field": None,
        "origin_policy": "require_explicit_origin_v1",
        "required_values": {
            "trajectory_policy.schema_version": "sidewalk_time_conditioned_policy_v1",
            "trajectory_policy.time_unit": "seconds",
        },
        "disabled_reason": None,
    }
    datasets = [
        {
            **common,
            "dataset_source": dataset,
            "status": "enabled",
            "allowed_trajectory_sources": list(sidewalk_sources),
        }
        for dataset in ("sidewalk", "foxglove")
    ]
    datasets.extend([
        {
            "dataset_source": "egowalk",
            "status": "enabled",
            "sample_identity_fields": ["id"],
            "split_group_fields": ["clip_id"],
            "trajectory_field": "trajectory_ego_t",
            "source_format": "ego_xy_t_speed_right_front_v1",
            "trajectory_source_fields": ["trajectory_policy.source"],
            "allowed_trajectory_sources": ["egowalk_odometry"],
            "yaw_source": "path_tangent",
            "mask_policy": "explicit_field_v1",
            "action_mask_field": "action_mask",
            "origin_policy": "prepend_current_origin_v1",
            "required_values": {
                "source_dataset": "egowalk",
                "trajectory_policy.source": "egowalk_odometry",
            },
            "disabled_reason": None,
        },
        {
            **common,
            "dataset_source": "frodobots",
            "status": "disabled",
            "allowed_trajectory_sources": [],
            "required_values": {},
            "disabled_reason": "untrusted_approximate_camera_trajectory",
        },
    ])
    payload = {
        "schema_version": "urbannav_stage_ii_trajectory_source_registry_v1",
        "registry_id": "urbannav_stage_ii_trajectory_sources_v1",
        "registry_version": "1.1",
        "split_policy": split,
        "datasets": datasets,
    }
    payload["registry_digest"] = stage_ii_digest_v1(payload)
    return payload


def _producer(
    source: str = "model_prediction",
    kind: str = "reasoning_grounding",
) -> StageIIContextProducerV1:
    generation_config = {"temperature": 0.0, "max_tokens": 2048}
    parser_contract_id = {
        "reasoning_grounding": "reasoning_v4_6_full_target_strict_v1",
        "decision": "reasoning_v4_6_decision_section_strict_v1",
    }[kind]
    return StageIIContextProducerV1(
        source=source,
        base_model_id="test/qwen",
        checkpoint_id="checkpoint-test",
        checkpoint_digest="1" * 64,
        prompt_profile_id="test_prompt_v1",
        prompt_digest="2" * 64,
        parser_contract_id=parser_contract_id,
        parser_contract_digest="3" * 64,
        generation_config=generation_config,
        generation_config_digest=stage_ii_digest_v1(generation_config),
    )


def _stage_i_full_cot():
    return project_stage_i_reasoning_view_v1(
        _formal_reasoning_view(),
        task_type="full_cot",
        modality_profile_id="fpv_instruction_v1",
        instruction_training_role="primary",
    )


def _stage_i_decision():
    return project_stage_i_reasoning_view_v1(
        _formal_reasoning_view(),
        task_type="decision",
        modality_profile_id="fpv_instruction_v1",
        instruction_training_role="primary",
        allow_fpv_only_decision=True,
    )


def _input_digest(view) -> str:
    return stage_ii_projected_input_digest_v1(
        images=list(view.images),
        image_roles=list(view.image_roles),
        instruction=view.instruction,
        model_visible_context=view.model_visible_context,
        evidence_catalog=[
            item.model_dump(mode="json", exclude_none=True)
            for item in view.evidence_catalog
        ],
        input_modalities=view.input_modalities.model_dump(mode="json"),
    )


def _cache(view, kind: str, raw_generation: str):
    return build_stage_ii_context_cache_v1(
        sample_id=view.sample_id,
        source_group_id=view.source_group_id,
        split_group_id=view.split_group_id,
        dataset_source=view.dataset_source,
        modality_profile_id=view.modality_profile_id,
        parent_source_fingerprint=view.parent_source_fingerprint,
        projected_input_digest=_input_digest(view),
        context_kind=kind,
        producer=_producer(kind=kind),
        raw_generation=raw_generation,
    )


def _trajectory(*, duration: float = 2.5):
    dt = 0.125
    count = int(round(duration / dt)) + 1
    rows = [[0.0, index * dt, index * dt, 1.0] for index in range(count)]
    return build_stage_ii_trajectory_target_v1(
        sample_id="sidewalk:trajectory-test",
        dataset_source="sidewalk",
        raw_trajectory=rows,
        action_mask=[True] * len(rows),
        source_format="ego_xy_t_speed_right_front_v1",
        trajectory_source="dataset_pose_trajectory",
        allowed_trajectory_sources={"dataset_pose_trajectory"},
        yaw_source="path_tangent",
    )


def test_temporal_resample_happens_before_legacy_axis_conversion() -> None:
    result = _trajectory(duration=2.5)

    assert result.accepted
    assert result.target is not None
    assert result.target.valid_step_count == 16
    assert result.target.trajectory_ego_t[-1] == pytest.approx([2.0, 0.0, 0.0])
    assert result.target.policy.horizon_seconds == 2.0
    assert result.target.policy.normalized is False


def test_short_horizon_uses_zero_padded_suffix_and_explicit_mask() -> None:
    result = _trajectory(duration=1.0)

    assert result.accepted
    assert result.target is not None
    assert result.target.valid_step_count == 8
    assert result.target.action_mask == [True] * 8 + [False] * 8
    assert result.target.trajectory_ego_t[8:] == [[0.0, 0.0, 0.0]] * 8


def test_committed_trajectory_source_registry_is_frozen_and_complete() -> None:
    registry = load_stage_ii_trajectory_source_registry_v1(
        ROUTE_ROOT / "config/stage_ii_cfm/trajectory_source_registry_v1.json"
    )

    assert {item.dataset_source for item in registry.datasets} == {
        "sidewalk",
        "foxglove",
        "egowalk",
        "frodobots",
    }
    assert registry.dataset("frodobots").status == "disabled"
    assert registry.dataset("egowalk").origin_policy == "prepend_current_origin_v1"
    assert registry.dataset("sidewalk").sample_identity_fields == (
        "sidewalk_meta.clip_id",
        "sidewalk_meta.frame_name",
    )
    assert registry.dataset("foxglove").sample_identity_fields == ("id",)


def test_full_sidecar_builder_uses_explicit_stage_i_splits(tmp_path) -> None:
    full_cot = _stage_i_full_cot()
    stage_i_path = tmp_path / "stage_i.jsonl"
    stage_i_path.write_text(
        json.dumps(full_cot.model_dump(mode="json", exclude_none=True)) + "\n",
        encoding="utf-8",
    )
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(
        json.dumps({"sample_id": full_cot.sample_id, "task_type": "full_cot"}) + "\n",
        encoding="utf-8",
    )
    validation_path = tmp_path / "validation.jsonl"
    validation_path.write_text("", encoding="utf-8")

    registry_payload = _source_registry_payload()
    sidewalk_spec = next(
        item for item in registry_payload["datasets"] if item["dataset_source"] == "sidewalk"
    )
    sidewalk_spec["sample_identity_fields"] = ["id"]
    registry_payload["registry_digest"] = stage_ii_digest_v1({
        key: value for key, value in registry_payload.items() if key != "registry_digest"
    })
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry_payload), encoding="utf-8")
    source_path = tmp_path / "sidewalk.jsonl"
    source_path.write_text(
        json.dumps({
            "id": "case-straight",
            "trajectory_ego_t": [[0.0, index * 0.125, index * 0.125, 1.0] for index in range(21)],
            "trajectory_policy": {
                "schema_version": "sidewalk_time_conditioned_policy_v1",
                "time_unit": "seconds",
                "trajectory_source": "calibrated_pose_projection",
            },
            "sidewalk_meta": {
                "clip_id": "clip-1",
                "trajectory_source": "calibrated_pose_projection",
            },
        }) + "\n",
        encoding="utf-8",
    )
    duplicate_path = tmp_path / "sidewalk_duplicate.jsonl"
    duplicate_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")

    report = build_stage_ii_trajectory_sidecars_v1(
        stage_i_jsonl=stage_i_path,
        split_manifests={"train": train_path, "validation": validation_path},
        dataset_manifests={"sidewalk": [source_path, duplicate_path]},
        registry_json=registry_path,
        output_dir=tmp_path / "output",
    )

    assert report["complete"] is True
    assert report["accepted_sidecars"] == 1
    assert report["dataset_scan"]["sidewalk"]["counts"]["matched_duplicate_equivalent"] == 1
    assert len(report["dataset_scan"]["sidewalk"]["equivalent_duplicate_examples"]) == 1
    sidecar = json.loads((tmp_path / "output/trajectory_sidecars_train.jsonl").read_text())
    assert sidecar["stage_i_sample_id"] == full_cot.sample_id
    assert sidecar["source_parent_id"] == full_cot.source_group_id
    assert sidecar["action_mask"] == [True] * 21


def test_full_sidecar_builder_rejects_conflicting_duplicate_sources(tmp_path) -> None:
    full_cot = _stage_i_full_cot()
    stage_i_path = tmp_path / "stage_i.jsonl"
    stage_i_path.write_text(
        json.dumps(full_cot.model_dump(mode="json", exclude_none=True)) + "\n",
        encoding="utf-8",
    )
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(
        json.dumps({"sample_id": full_cot.sample_id, "task_type": "full_cot"}) + "\n",
        encoding="utf-8",
    )
    validation_path = tmp_path / "validation.jsonl"
    validation_path.write_text("", encoding="utf-8")
    registry_payload = _source_registry_payload()
    sidewalk_spec = next(
        item for item in registry_payload["datasets"] if item["dataset_source"] == "sidewalk"
    )
    sidewalk_spec["sample_identity_fields"] = ["id"]
    registry_payload["registry_digest"] = stage_ii_digest_v1({
        key: value for key, value in registry_payload.items() if key != "registry_digest"
    })
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry_payload), encoding="utf-8")
    base_row = {
        "id": "case-straight",
        "trajectory_ego_t": [[0.0, index * 0.125, index * 0.125, 1.0] for index in range(21)],
        "trajectory_policy": {
            "schema_version": "sidewalk_time_conditioned_policy_v1",
            "time_unit": "seconds",
            "trajectory_source": "calibrated_pose_projection",
        },
        "sidewalk_meta": {
            "clip_id": "clip-1",
            "trajectory_source": "calibrated_pose_projection",
        },
    }
    first_path = tmp_path / "first.jsonl"
    first_path.write_text(json.dumps(base_row) + "\n", encoding="utf-8")
    conflicting = copy.deepcopy(base_row)
    conflicting["trajectory_ego_t"][-1][1] += 1.0
    second_path = tmp_path / "second.jsonl"
    second_path.write_text(json.dumps(conflicting) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="conflicting duplicate"):
        build_stage_ii_trajectory_sidecars_v1(
            stage_i_jsonl=stage_i_path,
            split_manifests={"train": train_path, "validation": validation_path},
            dataset_manifests={"sidewalk": [first_path, second_path]},
            registry_json=registry_path,
            output_dir=tmp_path / "output",
        )


def test_source_registry_derives_only_authorized_legacy_mask_and_egowalk_origin() -> None:
    registry = StageIITrajectorySourceRegistryV1.model_validate(_source_registry_payload())
    sidewalk_row = {
        "trajectory_ego_t": [[0.0, index * 0.125, index * 0.125, 1.0] for index in range(21)],
        "trajectory_policy": {
            "schema_version": "sidewalk_time_conditioned_policy_v1",
            "time_unit": "seconds",
            "trajectory_source": "calibrated_pose_projection",
        },
        "sidewalk_meta": {
            "clip_id": "clip-sidewalk",
            "frame_idx": 10,
            "trajectory_source": "calibrated_pose_projection",
        },
    }
    sidewalk = adapt_stage_ii_trajectory_source_row_v1(
        sidewalk_row,
        registry.dataset("sidewalk"),
    )
    assert sidewalk.action_mask == (True,) * 21
    assert sidewalk.mask_provenance == "validated_timed_rows_v1"

    egowalk_row = {
        "id": "egowalk-parent",
        "source_dataset": "egowalk",
        "clip_id": "clip-egowalk",
        "trajectory_ego_t": [[0.0, 0.2 * index, 0.2 * index, 1.0] for index in range(1, 6)],
        "action_mask": [True] * 5,
        "trajectory_policy": {"source": "egowalk_odometry"},
    }
    egowalk = adapt_stage_ii_trajectory_source_row_v1(
        egowalk_row,
        registry.dataset("egowalk"),
    )
    assert egowalk.raw_trajectory[0] == (0.0, 0.0, 0.0, 0.0)
    assert egowalk.action_mask == (True,) * 6
    target = build_stage_ii_trajectory_target_v1(
        sample_id=egowalk.sample_id,
        dataset_source=egowalk.dataset_source,
        raw_trajectory=egowalk.raw_trajectory,
        action_mask=egowalk.action_mask,
        source_format=egowalk.source_format,
        trajectory_source=egowalk.trajectory_source,
        allowed_trajectory_sources=registry.dataset("egowalk").allowed_trajectory_sources,
        yaw_source=egowalk.yaw_source,
    )
    assert target.accepted
    assert target.target is not None
    assert target.target.valid_step_count == 8

    with pytest.raises(StageIISourceAdapterErrorV1):
        adapt_stage_ii_trajectory_source_row_v1(
            sidewalk_row,
            registry.dataset("frodobots"),
        )


@pytest.mark.parametrize(
    ("mask", "source", "reason"),
    [
        (None, "dataset_pose_trajectory", "trajectory_mask_missing"),
        ([True, True], "dataset_gps_trajectory", "trajectory_source_not_allowed"),
    ],
)
def test_trajectory_builder_fails_closed(mask, source: str, reason: str) -> None:
    rows = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.25, 0.25, 1.0]]
    result = build_stage_ii_trajectory_target_v1(
        sample_id="case",
        dataset_source="sidewalk",
        raw_trajectory=rows,
        action_mask=mask,
        source_format="ego_xy_t_speed_right_front_v1",
        trajectory_source=source,
        allowed_trajectory_sources={"dataset_pose_trajectory"},
        yaw_source="path_tangent",
    )

    assert not result.accepted
    assert result.target is None
    assert str(result.qc.reason_codes[0]) == reason


def test_trajectory_builder_rejects_forbidden_source_even_if_allowlisted() -> None:
    rows = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.25, 0.25, 1.0]]
    result = build_stage_ii_trajectory_target_v1(
        sample_id="case",
        dataset_source="sidewalk",
        raw_trajectory=rows,
        action_mask=[True, True],
        source_format="ego_xy_t_speed_right_front_v1",
        trajectory_source="dataset_gps_trajectory",
        allowed_trajectory_sources={"dataset_gps_trajectory"},
        yaw_source="path_tangent",
    )

    assert not result.accepted
    assert str(result.qc.reason_codes[0]) == "trajectory_source_forbidden"


def test_trajectory_builder_optional_step_gate_rejects_zero_fill_jump() -> None:
    rows = [
        [0.0, 0.0, 0.0, 1.0],
        [-324.0, 728.0, 0.2, 1.0],
        [-0.1, 0.5, 0.4, 1.0],
    ]
    result = build_stage_ii_trajectory_target_v1(
        sample_id="zero-fill-jump",
        dataset_source="foxglove",
        raw_trajectory=rows,
        action_mask=[True, True, True],
        source_format="ego_xy_t_speed_right_front_v1",
        trajectory_source="calibrated_pose_projection",
        allowed_trajectory_sources={"calibrated_pose_projection"},
        yaw_source="path_tangent",
        max_source_step_m=3.0,
    )

    assert not result.accepted
    assert result.target is None
    assert str(result.qc.reason_codes[0]) == "trajectory_source_step_exceeded"


def test_context_cache_requires_independent_strict_passes() -> None:
    view = _stage_i_full_cot()
    reasoning = _cache(view, "reasoning_grounding", view.assistant_text)
    decision_text = render_reasoning_section_v4(
        "Decision",
        view.reasoning_target.reasoning.decision,
    )
    decision = _cache(view, "decision", decision_text)

    assert reasoning.parse_status == "accepted"
    assert "<Decision>" not in str(reasoning.canonical_text)
    assert decision.parse_status == "accepted"
    assert decision.cache_id != reasoning.cache_id

    mutated = copy.deepcopy(reasoning.model_dump(mode="json"))
    mutated["canonical_text"] = str(mutated["canonical_text"]).replace(
        "urban_sidewalk",
        "changed",
    )
    with pytest.raises(ValidationError, match="cache_digest"):
        StageIIContextCacheV1.model_validate(mutated)


def test_context_cache_materialization_matches_ms_swift() -> None:
    view = _stage_i_full_cot()
    producer = _producer()
    route_cache = _cache(view, "reasoning_grounding", view.assistant_text)
    record = {
        "sample_id": view.sample_id,
        "source_group_id": view.source_group_id,
        "split_group_id": view.split_group_id,
        "dataset_source": view.dataset_source,
        "modality_profile_id": view.modality_profile_id,
        "parent_source_fingerprint": view.parent_source_fingerprint,
        "projected_input_digest": _input_digest(view),
        "context_kind": "reasoning_grounding",
        "producer": producer.model_dump(mode="json", exclude_none=True),
        "raw_generation": view.assistant_text,
    }
    ms_root = Path(__file__).resolve().parents[4] / "ms-swift"
    if not ms_root.is_dir():
        pytest.skip("sibling ms-swift checkout is unavailable")
    sys.path.insert(0, str(ms_root))
    try:
        adapter = importlib.import_module(
            "swift.sidewalk_nav.data.stage_ii.stage_ii_context_cache"
        )
        ms_cache = adapter.build_stage_ii_context_cache_row(record)
    finally:
        sys.path.remove(str(ms_root))

    assert ms_cache.model_dump(mode="json") == route_cache.model_dump(mode="json")


def test_stage_ii_view_builds_cfm_only_ms_swift_row() -> None:
    view = _stage_i_full_cot()
    trajectory = _trajectory()
    assert trajectory.target is not None
    reasoning = _cache(view, "reasoning_grounding", view.assistant_text)
    decision = _cache(
        view,
        "decision",
        render_reasoning_section_v4(
            "Decision",
            view.reasoning_target.reasoning.decision,
        ),
    )
    stage_ii = project_stage_ii_cfm_view_v1(
        view,
        trajectory_target=trajectory.target,
        reasoning_context=reasoning,
        decision_context=decision,
        review_status="not_required",
    )

    serialized = stage_ii.model_dump(mode="json", exclude_none=True)
    assert serialized["loss_mask"]["lm"] is False
    assert serialized["loss_mask"]["system1_action"] is True
    assert "assistant_text" not in serialized
    reference = build_stage_ii_openloop_reference_from_parent_v1(
        view,
        trajectory.target,
    )
    assert reference.trajectory_valid_mask == trajectory.target.action_mask
    assert reference.label_provenance == "deterministic_gt"
    assert reference.subgroup_metadata["yaw_source"] == "path_tangent"

    ms_root = Path(__file__).resolve().parents[4] / "ms-swift"
    if not ms_root.is_dir():
        pytest.skip("sibling ms-swift checkout is unavailable")
    sys.path.insert(0, str(ms_root))
    try:
        adapter = importlib.import_module(
            "swift.sidewalk_nav.data.stage_ii.stage_ii_manifest"
        )
        row = adapter.build_stage_ii_manifest_row(serialized)
        eval_adapter = importlib.import_module(
            "swift.sidewalk_nav.data.stage_ii.stage_ii_eval_manifest"
        )
        case_adapter = importlib.import_module(
            "swift.sidewalk_nav.data.stage_ii.stage_ii_case_manifest"
        )
        benchmark_input = case_adapter.build_stage_ii_case_input_row(
            view,
            _stage_i_decision(),
            split="validation",
            renderer_policy_digest="6" * 64,
        )
        eval_row = eval_adapter.build_stage_ii_eval_manifest_row(
            serialized,
            context_policy="shared_current_checkpoint_v1",
            pass_t_backbone_checkpoint_digest="1" * 64,
            reasoning_renderer_digest="4" * 64,
            decision_renderer_digest="5" * 64,
            benchmark_input=benchmark_input,
        )
    finally:
        sys.path.remove(str(ms_root))
    assert all(message["role"] != "assistant" for message in row["messages"])
    assert row["loss_mask"] == {
        "lm": False,
        "system1_action": True,
        "trajectory": False,
        "segmentation": False,
    }
    assert row["stage_ii_context_mode"] == "predicted"
    prompt = row["messages"][-1]["content"]
    assert prompt.count("[TRAJ_0]") == 1
    assert prompt.count("[TRAJ_15]") == 1
    assert "trajectory_ego_t" not in prompt
    assert "direction_supervision" not in prompt
    eval_serialized = json.dumps(eval_row, ensure_ascii=False)
    assert eval_row["stage_ii_context_policy"] == (
        "shared_current_checkpoint_v1"
    )
    assert eval_row["stage_ii_decision_context_label"] == "straight"
    assert len(eval_row["stage_ii_input_digest"]) == 64
    assert len(eval_row["openloop_input_digest"]) == 64
    assert eval_row["openloop_parent_sample_id"] == view.parent_sample_id
    for forbidden in (
        "trajectory_ego_t",
        "action_mask",
        "direction_label_name",
        "loss_mask",
    ):
        assert forbidden not in eval_serialized


def test_legacy_direct_profile_reuses_visible_prefix_without_pass_rd(
    tmp_path: Path,
) -> None:
    messages = [
        {"role": "system", "content": "legacy navigation system"},
        {"role": "user", "content": "<image><image>\nchoose a direction"},
        {
            "role": "assistant",
            "content": "<think><FPVScene>{}</FPVScene></think><finish>{}</finish>",
        },
    ]
    images = ["/data/fpv/000001.jpg", "/data/bev/000001.jpg"]
    raw = {
        "messages": messages,
        "images": images,
        "direction_name": "straight",
        "direction_choice": "straight",
        "direction_choice_label": 1,
        "trajectory_ego_t": [
            [0.0, float(index) * 0.125, float(index) * 0.125, 1.0]
            for index in range(17)
        ],
        "trajectory_policy": {
            "schema_version": "sidewalk_time_conditioned_policy_v1",
            "time_unit": "seconds",
            "trajectory_source": "calibrated_pose_projection",
        },
        "sidewalk_meta": {
            "clip_id": "clip-a",
            "frame_idx": 1,
            "trajectory_source": "calibrated_pose_projection",
        },
    }
    split_row = {
        "id": "sidewalk_main_multiframe|clip-a|1",
        "messages": messages,
        "images": images,
        "direction_name": "straight",
        "direction_choice": "straight",
        "direction_choice_label": 1,
        "direction_profile_id": "ego_degree10_12way_terminal_centroid_v2",
        "direction_profile_version": "2.0",
        "response_contract": "legacy_think_finish",
        "reasoning_mode": "legacy_full_cot",
        "sidewalk_meta": {
            "experiment_group": {
                "source": "sidewalk_main_multiframe",
                "clip_id": "clip-a",
                "group_key": "sidewalk_main_multiframe|clip-a",
                "split": "train_pool",
            },
            "source_record": {"line_no": 1},
            "image_roles": {
                "roles": [
                    {"modality": "fpv", "temporal_role": "current", "temporal_index": 0},
                    {"modality": "bev", "temporal_role": "current", "temporal_index": 0},
                ],
            },
        },
    }
    paths = {
        "source": tmp_path / "source.jsonl",
        "train": tmp_path / "train.jsonl",
        "validation": tmp_path / "validation.jsonl",
        "registry": tmp_path / "registry.json",
        "output": tmp_path / "targets",
    }
    _write_jsonl(paths["source"], [raw, raw])
    _write_jsonl(paths["train"], [split_row])
    validation_row = copy.deepcopy(split_row)
    validation_row["id"] = "sidewalk_main_multiframe|clip-b|2"
    validation_row["sidewalk_meta"]["source_record"]["line_no"] = 2
    validation_row["sidewalk_meta"]["experiment_group"].update({
        "clip_id": "clip-b",
        "group_key": "sidewalk_main_multiframe|clip-b",
        "split": "val",
    })
    raw_validation = copy.deepcopy(raw)
    raw_validation["sidewalk_meta"]["clip_id"] = "clip-b"
    paths["source"].write_text(
        json.dumps(raw, ensure_ascii=False) + "\n"
        + json.dumps(raw_validation, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(paths["validation"], [validation_row])
    paths["registry"].write_text(
        json.dumps(_source_registry_payload(), ensure_ascii=False),
        encoding="utf-8",
    )

    report = build_stage_ii_legacy_direct_targets_v1(
        source_jsonl=paths["source"],
        split_manifests={
            "train": paths["train"],
            "validation": paths["validation"],
        },
        source_dataset_map={"sidewalk_main_multiframe": "sidewalk"},
        registry_json=paths["registry"],
        output_dir=paths["output"],
    )
    assert report["passed"] is True
    assert report["accepted_rows"] == 2
    assert report["hidden_test_read"] is False

    target = json.loads(
        (paths["output"] / "legacy_direct_targets_train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    ms_root = Path(__file__).resolve().parents[4] / "ms-swift"
    if not ms_root.is_dir():
        pytest.skip("sibling ms-swift checkout is unavailable")
    sys.path.insert(0, str(ms_root))
    try:
        adapter = importlib.import_module(
            "swift.sidewalk_nav.data.legacy.stage_ii_legacy_direct_manifest"
        )
        physical = adapter.build_stage_ii_legacy_direct_manifest_row(
            split_row,
            target,
        )
    finally:
        sys.path.remove(str(ms_root))
    assert physical["stage_ii_profile_id"] == "legacy_full_cot_direct_latent_v1"
    assert physical["stage_ii_context_mode"] == "legacy_direct_latent"
    assert all(message["role"] != "assistant" for message in physical["messages"])
    assert physical["messages"][-1]["content"].count("[TRAJ_0]") == 1
    assert "<finish>" not in physical["messages"][-1]["content"]
    assert physical["loss_mask"]["lm"] is False


def test_legacy_direct_selection_uses_current_stage_i_split_authority(
    tmp_path: Path,
) -> None:
    images = [
        path
        for index in range(5)
        for path in (
            f"/data/fpv/{index:06d}.jpg",
            f"/data/bev/{index:06d}_bev.jpg",
        )
    ]
    messages = [
        {"role": "system", "content": "legacy navigation system"},
        {
            "role": "user",
            "content": "".join("<image>" for _ in images) + "\nchoose a direction",
        },
        {
            "role": "assistant",
            "content": "<think><FPVScene>{}</FPVScene></think><finish>{}</finish>",
        },
    ]

    def raw_row(clip_id: str, frame_idx: int, source_name: str) -> dict:
        return {
            "id": f"{source_name}|{clip_id}|{frame_idx}",
            "messages": copy.deepcopy(messages),
            "images": list(images),
            "direction_name": "straight",
            "direction_choice": "straight",
            "direction_choice_label": 1,
            "direction_bin_mode": "degree10_12way_with_stop",
            "direction_profile_id": "ego_degree10_12way_terminal_centroid_v2",
            "direction_profile_version": "2.0",
            "trajectory_ego_t": [
                [0.0, float(index) * 0.125, float(index) * 0.125, 1.0]
                for index in range(17)
            ],
            "trajectory_policy": {
                "schema_version": "sidewalk_time_conditioned_policy_v1",
                "time_unit": "seconds",
                "trajectory_source": "calibrated_pose_projection",
            },
            "sidewalk_meta": {
                "clip_id": clip_id,
                "frame_idx": frame_idx,
                "trajectory_source": "calibrated_pose_projection",
                "direction_sample_source": source_name,
            },
        }

    def restamped_stage_i_view(
        *,
        sample_id: str,
        source_group_id: str,
        split_group_id: str,
    ) -> dict:
        payload = _stage_i_full_cot().model_dump(
            mode="json",
            exclude_none=True,
        )
        payload.update({
            "sample_id": sample_id,
            "source_group_id": source_group_id,
            "split_group_id": split_group_id,
        })
        payload.pop("projection_digest", None)
        payload["projection_digest"] = stage_i_projection_digest_v1(payload)
        return payload

    train_view = restamped_stage_i_view(
        sample_id="stage-i-full-cot:train-parent",
        source_group_id="sidewalk:clip-train:1",
        split_group_id="sidewalk:clip-train",
    )
    validation_view = restamped_stage_i_view(
        sample_id="stage-i-full-cot:validation-parent",
        source_group_id="sidewalk:clip-validation:2",
        split_group_id="sidewalk:clip-validation",
    )
    train_manifest = tmp_path / "stage_i_train.jsonl"
    validation_manifest = tmp_path / "stage_i_validation.jsonl"
    _write_jsonl(
        train_manifest,
        [train_view],
    )
    _write_jsonl(
        validation_manifest,
        [validation_view],
    )
    source_rows = [
        raw_row("clip-train", 1, "sidewalk_main_multiframe"),
        raw_row("clip-train", 1, "sidewalk_extra_multiframe"),
        raw_row("clip-validation", 2, "sidewalk_main_multiframe"),
    ]
    source_manifest = tmp_path / "combined.jsonl"
    _write_jsonl(source_manifest, source_rows)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(_source_registry_payload(), ensure_ascii=False),
        encoding="utf-8",
    )

    report = build_stage_ii_legacy_direct_selection_v1(
        source_jsonl=source_manifest,
        stage_i_split_manifests={
            "train": train_manifest,
            "validation": validation_manifest,
        },
        source_dataset_map={
            "sidewalk_main_multiframe": "sidewalk",
            "sidewalk_extra_multiframe": "sidewalk",
        },
        registry_json=registry_path,
        output_dir=tmp_path / "selected",
    )

    assert report["complete"] is True
    assert report["selected_rows"] == 2
    assert report["historical_legacy_split_manifests_read"] is False
    assert report["hidden_test_manifest_read"] is False
    assert report["counts"]["matched_duplicate_equivalent"] == 1
    selected_train = json.loads(
        (tmp_path / "selected/legacy_direct_selected_train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert selected_train["id"] == train_view["sample_id"]
    assert selected_train["source_dataset"] == "sidewalk_main_multiframe"
    assert selected_train["sidewalk_meta"]["source_record"]["line_no"] == 1

    conflicting_rows = copy.deepcopy(source_rows)
    conflicting_rows[1]["trajectory_ego_t"][-1][1] += 1.0
    conflicting_source = tmp_path / "combined_conflicting.jsonl"
    _write_jsonl(conflicting_source, conflicting_rows)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        build_stage_ii_legacy_direct_selection_v1(
            source_jsonl=conflicting_source,
            stage_i_split_manifests={
                "train": train_manifest,
                "validation": validation_manifest,
            },
            source_dataset_map={
                "sidewalk_main_multiframe": "sidewalk",
                "sidewalk_extra_multiframe": "sidewalk",
            },
            registry_json=registry_path,
            output_dir=tmp_path / "conflicting",
        )


def test_stage_ii_view_rejects_mixed_oracle_and_predicted_context() -> None:
    view = _stage_i_full_cot()
    trajectory = _trajectory()
    assert trajectory.target is not None
    reasoning = _cache(view, "reasoning_grounding", view.assistant_text)
    decision = build_stage_ii_context_cache_v1(
        sample_id=view.sample_id,
        source_group_id=view.source_group_id,
        split_group_id=view.split_group_id,
        dataset_source=view.dataset_source,
        modality_profile_id=view.modality_profile_id,
        parent_source_fingerprint=view.parent_source_fingerprint,
        projected_input_digest=_input_digest(view),
        context_kind="decision",
        producer=_producer("ground_truth"),
        raw_generation=render_reasoning_section_v4(
            "Decision",
            view.reasoning_target.reasoning.decision,
        ),
    )

    with pytest.raises(ValidationError, match="mixed oracle/predicted"):
        project_stage_ii_cfm_view_v1(
            view,
            trajectory_target=trajectory.target,
            reasoning_context=reasoning,
            decision_context=decision,
            review_status="not_required",
        )


def test_stage_ii_bulk_builder_emits_view_qc_and_report(tmp_path) -> None:
    view = _stage_i_full_cot()
    reasoning = _cache(view, "reasoning_grounding", view.assistant_text)
    decision = _cache(
        view,
        "decision",
        render_reasoning_section_v4(
            "Decision",
            view.reasoning_target.reasoning.decision,
        ),
    )
    raw_target = {
        "stage_i_sample_id": view.sample_id,
        "trajectory_ego_t": [
            [0.0, index * 0.125, index * 0.125, 1.0]
            for index in range(21)
        ],
        "action_mask": [True] * 21,
        "source_format": "ego_xy_t_speed_right_front_v1",
        "trajectory_source": "dataset_pose_trajectory",
        "yaw_source": "path_tangent",
        "review_status": "not_required",
    }
    paths = {
        "stage_i": tmp_path / "stage_i.jsonl",
        "trajectory": tmp_path / "trajectory.jsonl",
        "reasoning": tmp_path / "reasoning.jsonl",
        "decision": tmp_path / "decision.jsonl",
        "registry": tmp_path / "registry.json",
        "views": tmp_path / "views.jsonl",
        "qc": tmp_path / "qc.jsonl",
        "drops": tmp_path / "drops.jsonl",
        "references": tmp_path / "references.jsonl",
        "eligible": tmp_path / "eligible.jsonl",
    }
    paths["stage_i"].write_text(
        view.model_dump_json(exclude_none=True) + "\n",
        encoding="utf-8",
    )
    paths["trajectory"].write_text(
        json.dumps(raw_target) + "\n",
        encoding="utf-8",
    )
    paths["reasoning"].write_text(
        reasoning.model_dump_json(exclude_none=True) + "\n",
        encoding="utf-8",
    )
    paths["decision"].write_text(
        decision.model_dump_json(exclude_none=True) + "\n",
        encoding="utf-8",
    )
    paths["registry"].write_text(
        json.dumps(
            _source_registry_payload(
                sidewalk_sources=("dataset_pose_trajectory",),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_stage_ii_cfm_views(
        stage_i_jsonl=paths["stage_i"],
        trajectory_jsonl=paths["trajectory"],
        reasoning_cache_jsonl=paths["reasoning"],
        decision_cache_jsonl=paths["decision"],
        trajectory_source_registry_json=paths["registry"],
        output_view_jsonl=paths["views"],
        output_qc_jsonl=paths["qc"],
        output_drop_jsonl=paths["drops"],
        output_reference_jsonl=paths["references"],
        output_eligible_jsonl=paths["eligible"],
    )

    assert report["counts"]["input_rows"] == 1
    assert report["counts"]["output_views"] == 1
    assert report["counts"]["eligible_references"] == 1
    assert paths["views"].read_text(encoding="utf-8").count("\n") == 1
    assert paths["references"].read_text(encoding="utf-8").count("\n") == 1
    assert paths["eligible"].read_text(encoding="utf-8").count("\n") == 1
    assert '"status":"accepted"' in paths["qc"].read_text(encoding="utf-8")
    assert paths["drops"].read_text(encoding="utf-8") == ""


def test_stage_ii_bulk_builder_freezes_denominator_without_context_cache(tmp_path) -> None:
    view = _stage_i_full_cot()
    raw_target = {
        "stage_i_sample_id": view.sample_id,
        "trajectory_ego_t": [
            [0.0, index * 0.125, index * 0.125, 1.0]
            for index in range(21)
        ],
        "action_mask": [True] * 21,
        "source_format": "ego_xy_t_speed_right_front_v1",
        "trajectory_source": "dataset_pose_trajectory",
        "yaw_source": "path_tangent",
        "review_status": "not_required",
    }
    paths = {
        name: tmp_path / filename
        for name, filename in {
            "stage_i": "stage_i.jsonl",
            "trajectory": "trajectory.jsonl",
            "registry": "registry.json",
            "views": "views.jsonl",
            "qc": "qc.jsonl",
            "drops": "drops.jsonl",
            "references": "references.jsonl",
            "eligible": "eligible.jsonl",
        }.items()
    }
    paths["stage_i"].write_text(
        view.model_dump_json(exclude_none=True) + "\n", encoding="utf-8"
    )
    paths["trajectory"].write_text(
        json.dumps(raw_target) + "\n", encoding="utf-8"
    )
    paths["registry"].write_text(
        json.dumps(
            _source_registry_payload(
                sidewalk_sources=("dataset_pose_trajectory",),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_stage_ii_cfm_views(
        stage_i_jsonl=paths["stage_i"],
        trajectory_jsonl=paths["trajectory"],
        reasoning_cache_jsonl=None,
        decision_cache_jsonl=None,
        trajectory_source_registry_json=paths["registry"],
        output_view_jsonl=paths["views"],
        output_qc_jsonl=paths["qc"],
        output_drop_jsonl=paths["drops"],
        output_reference_jsonl=paths["references"],
        output_eligible_jsonl=paths["eligible"],
        denominator_only=True,
    )

    assert report["mode"] == "denominator_only"
    assert report["counts"]["eligible_references"] == 1
    assert report["counts"]["denominator_only_rows"] == 1
    assert paths["views"].read_text(encoding="utf-8") == ""
    assert paths["drops"].read_text(encoding="utf-8") == ""
    assert paths["references"].read_text(encoding="utf-8").count("\n") == 1
    assert paths["eligible"].read_text(encoding="utf-8").count("\n") == 1


def test_real_train_target_audit_reports_horizon_mask_and_yaw_distributions(tmp_path) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(_source_registry_payload()) + "\n",
        encoding="utf-8",
    )
    manifests = {
        dataset: tmp_path / f"{dataset}.jsonl"
        for dataset in ("sidewalk", "foxglove", "egowalk")
    }
    for dataset in ("sidewalk", "foxglove"):
        rows = []
        for index in range(34):
            source = "stationary_pose_hold" if index % 7 == 0 else "calibrated_pose_projection"
            rows.append({
                "trajectory_ego_t": [
                    [0.0, 0.0 if source == "stationary_pose_hold" else step * 0.125, step * 0.125, 0.0 if source == "stationary_pose_hold" else 1.0]
                    for step in range(21)
                ],
                "trajectory_policy": {
                    "schema_version": "sidewalk_time_conditioned_policy_v1",
                    "time_unit": "seconds",
                    "trajectory_source": source,
                },
                "sidewalk_meta": {
                    "clip_id": f"clip-{index}",
                    "frame_idx": index,
                    "trajectory_source": source,
                },
            })
        manifests[dataset].write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
    ego_rows = [{
        "id": f"egowalk-{index}",
        "source_dataset": "egowalk",
        "clip_id": f"clip-{index}",
        "trajectory_ego_t": [
            [0.0, 0.2 * step, 0.2 * step, 1.0]
            for step in range(1, 6)
        ],
        "action_mask": [True] * 5,
        "trajectory_policy": {"source": "egowalk_odometry"},
    } for index in range(34)]
    manifests["egowalk"].write_text(
        "".join(json.dumps(row) + "\n" for row in ego_rows),
        encoding="utf-8",
    )

    report = audit_stage_ii_trajectory_targets(
        registry_json=registry_path,
        dataset_manifests=manifests,
        output_dir=tmp_path / "audit",
        parents_per_dataset=34,
        max_parents_per_group=1,
        scan_limit_per_dataset=34,
    )

    assert report["unique_parent_count"] == 102
    assert report["accepted_count"] == 102
    assert report["distributions"]["target_mask_patterns"] == {
        "16/16": 68,
        "8/16": 34,
    }
    assert report["distributions"]["yaw_source"] == {"path_tangent": 102}
    assert (tmp_path / "audit/accepted_targets.jsonl").read_text(encoding="utf-8").count("\n") == 102
