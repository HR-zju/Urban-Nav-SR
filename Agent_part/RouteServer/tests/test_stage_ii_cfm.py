from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from mcp_server.agent.reasoning_v4_text import render_reasoning_section_v4
from mcp_server.contracts.stage_ii_cfm import (
    StageIIContextCacheV1,
    StageIIContextProducerV1,
    StageIITrajectorySourceRegistryV1,
    stage_ii_digest_v1,
)
from mcp_server.data.stage_i_views import project_stage_i_reasoning_view_v1
from mcp_server.data.stage_ii_context import (
    build_stage_ii_context_cache_v1,
    stage_ii_projected_input_digest_v1,
)
from mcp_server.data.stage_ii_trajectory import build_stage_ii_trajectory_target_v1
from mcp_server.data.stage_ii_sources import (
    StageIISourceAdapterErrorV1,
    adapt_stage_ii_trajectory_source_row_v1,
    load_stage_ii_trajectory_source_registry_v1,
)
from mcp_server.data.stage_ii_views import project_stage_ii_cfm_view_v1
from tools.model_processing.build_stage_ii_cfm_views import (
    build_stage_ii_cfm_views,
)
from tools.model_processing.audit_stage_ii_trajectory_targets import (
    audit_stage_ii_trajectory_targets,
)

from test_stage_i_sft import _formal_reasoning_view


ROUTE_ROOT = Path(__file__).resolve().parents[1]


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
        "registry_version": "1.0",
        "split_policy": split,
        "datasets": datasets,
    }
    payload["registry_digest"] = stage_ii_digest_v1(payload)
    return payload


def _producer(source: str = "model_prediction") -> StageIIContextProducerV1:
    generation_config = {"temperature": 0.0, "max_tokens": 2048}
    return StageIIContextProducerV1(
        source=source,
        base_model_id="test/qwen",
        checkpoint_id="checkpoint-test",
        checkpoint_digest="1" * 64,
        prompt_profile_id="test_prompt_v1",
        prompt_digest="2" * 64,
        parser_contract_id="reasoning_v4_6_strict",
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
        producer=_producer(),
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
            "swift.sidewalk_nav.data.stage_ii_context_cache"
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

    ms_root = Path(__file__).resolve().parents[4] / "ms-swift"
    if not ms_root.is_dir():
        pytest.skip("sibling ms-swift checkout is unavailable")
    sys.path.insert(0, str(ms_root))
    try:
        adapter = importlib.import_module(
            "swift.sidewalk_nav.data.stage_ii_manifest"
        )
        row = adapter.build_stage_ii_manifest_row(serialized)
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
    )

    assert report["counts"]["input_rows"] == 1
    assert report["counts"]["output_views"] == 1
    assert paths["views"].read_text(encoding="utf-8").count("\n") == 1
    assert '"status":"accepted"' in paths["qc"].read_text(encoding="utf-8")
    assert paths["drops"].read_text(encoding="utf-8") == ""


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
