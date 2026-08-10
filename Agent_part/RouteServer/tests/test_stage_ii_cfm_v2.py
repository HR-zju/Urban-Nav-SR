from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mcp_server.agent.reasoning_v4_text import render_reasoning_section_v4
from mcp_server.contracts.stage_ii_cfm import (
    StageIICFMDirectLatentViewV1,
    StageIICFMTrainingViewV2,
    StageIIContextProducerV1,
    StageIIProjectionDropV2,
    StageIIReasoningAbsentV2,
    StageIITrajectoryPolicyV2,
    StageIITrajectorySidecarV2,
    StageIITrajectoryTargetV2,
    stage_ii_digest_v1,
)
from mcp_server.data.stage_i_views import (
    project_stage_i_reasoning_dropout_from_full_v2,
    project_stage_i_reasoning_view_v2,
)
from mcp_server.data.stage_ii_context import build_stage_ii_context_cache_v2
from mcp_server.data.stage_ii_route_pool_v2 import (
    build_stage_ii_cfm_route_pool_v2,
)
from mcp_server.data.stage_ii_trajectory import build_stage_ii_trajectory_qc_v2
from mcp_server.data.stage_ii_views import (
    project_stage_ii_cfm_view_v1_from_stage_i_v2,
    project_stage_ii_cfm_view_v2,
    project_stage_ii_cfm_view_with_audit_v2,
    project_stage_ii_direct_latent_view_from_stage_i_v2,
)
from tools.model_processing.build_stage_ii_cfm_views import build_stage_ii_cfm_views

from test_stage_i_sft import _formal_reasoning_view


def _stage_i_full_v2():
    return project_stage_i_reasoning_view_v2(
        _formal_reasoning_view(),
        task_type="full_cot",
        modality_profile_id="fpv_instruction_v1",
        instruction_training_role="primary",
    )


def _producer(kind: str) -> StageIIContextProducerV1:
    generation_config = {"temperature": 0.0, "max_tokens": 1024}
    parser_contract_id = {
        "reasoning_grounding": "reasoning_v4_6_full_target_strict_v1",
        "decision": "reasoning_v4_6_decision_section_strict_v1",
    }[kind]
    return StageIIContextProducerV1(
        source="model_prediction",
        base_model_id="test/qwen",
        checkpoint_id="checkpoint-test",
        checkpoint_digest="1" * 64,
        prompt_profile_id=f"test_{kind}_v1",
        prompt_digest="2" * 64,
        parser_contract_id=parser_contract_id,
        parser_contract_digest="3" * 64,
        generation_config=generation_config,
        generation_config_digest=stage_ii_digest_v1(generation_config),
    )


def _cache(stage_i_view, kind: str, raw_generation: str):
    return build_stage_ii_context_cache_v2(
        stage_i_view,
        context_kind=kind,
        producer=_producer(kind),
        raw_generation=raw_generation,
    )


def _trajectory_v2() -> StageIITrajectoryTargetV2:
    policy = StageIITrajectoryPolicyV2(
        source_adapter_id="sidewalk_raw_pose_v2",
        trajectory_source="odometry",
        raw_timestamp_unit="millisecond",
        yaw_offset_deg=0.0,
    )
    payload = {
        "schema_version": "urbannav_stage_ii_trajectory_target_v2_1",
        "trajectory_ego_t": [
            [0.125 * (index + 1), 0.01 * index, 0.001 * index]
            for index in range(32)
        ],
        "action_mask": [True] * 32,
        "valid_step_count": 32,
        "policy": policy.model_dump(mode="json"),
        "source_target_digest": "4" * 64,
    }
    payload["target_digest"] = stage_ii_digest_v1(payload)
    return StageIITrajectoryTargetV2.model_validate(payload)


def _trajectory_v1():
    """Return the retained clean 16-point target used by the compatibility path."""

    from mcp_server.data.stage_ii_trajectory import build_stage_ii_trajectory_target_v1

    return build_stage_ii_trajectory_target_v1(
        sample_id="sidewalk:trajectory-test",
        dataset_source="sidewalk",
        raw_trajectory=[
            [0.0, index * 0.125, index * 0.125, 1.0]
            for index in range(21)
        ],
        action_mask=[True] * 21,
        source_format="ego_xy_t_speed_right_front_v1",
        trajectory_source="dataset_pose_trajectory",
        allowed_trajectory_sources={"dataset_pose_trajectory"},
        yaw_source="path_tangent",
        max_source_step_m=3.0,
    )


def _decision_text(full) -> str:
    return render_reasoning_section_v4(
        "Decision",
        full.reasoning_target.reasoning.decision,
    )


def _trajectory_sidecar_v2(full) -> StageIITrajectorySidecarV2:
    target = _trajectory_v2()
    qc = build_stage_ii_trajectory_qc_v2(
        sample_id=full.source_group_id,
        dataset_source=full.dataset_source,
        source_adapter_id=target.policy.source_adapter_id,
        trajectory_source=target.policy.trajectory_source,
        clip_id="clip-test",
        frame_idx=10,
        status="accepted",
        reason_codes=("trajectory_build_accepted",),
        raw_pose_count=33,
        source_valid_step_count=33,
        source_horizon_seconds=4.0,
        max_source_gap_sec=0.125,
        max_source_step_m=0.125,
        max_source_speed_mps=1.0,
        max_source_yaw_rate_rad_s=0.1,
        target_valid_step_count=32,
        source_evidence_digest=target.source_target_digest,
    )
    payload = {
        "schema_version": "urbannav_stage_ii_trajectory_sidecar_v2_1",
        "sample_id": full.source_group_id,
        "source_group_id": full.source_group_id,
        "split_group_id": full.split_group_id,
        "dataset_source": full.dataset_source,
        "split": "train",
        "source_manifest_name": "test/source.jsonl",
        "source_manifest_sha256": "5" * 64,
        "source_line": 1,
        "trajectory_target": target.model_dump(mode="json"),
        "trajectory_qc": qc.model_dump(mode="json"),
    }
    payload["sidecar_digest"] = stage_ii_digest_v1(payload)
    return StageIITrajectorySidecarV2.model_validate(payload)


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def test_stage_ii_v2_reasoning_plus_decision_route_is_closed() -> None:
    full = _stage_i_full_v2()
    reasoning = _cache(full, "reasoning_grounding", full.assistant_text)
    decision = _cache(full, "decision", _decision_text(full))

    view = project_stage_ii_cfm_view_v2(
        full,
        trajectory_target=_trajectory_v2(),
        reasoning_context=reasoning,
        decision_context=decision,
        review_status="not_required",
    )

    assert view.training_contract == "stage_ii_cfm_v2"
    assert view.context_mode == "reasoning_plus_decision"
    assert view.reasoning.reasoning_state == "present"
    assert view.stage_i_route_task_type == "full_cot"
    assert view.system1_condition_mode == "traj_query_only"
    assert len(view.trajectory_target.trajectory_ego_t) == 32
    assert len(view.pass_t_input_digest) == 64
    assert "assistant_text" not in view.model_dump(mode="json")


def test_stage_i_v2_full_cot_can_feed_the_frozen_clean_cfm_v1_contract() -> None:
    """Keep V2 prompt lineage separate from the existing 16-point geometry truth."""

    full = _stage_i_full_v2()
    target = _trajectory_v1()
    assert target.accepted and target.target is not None
    reasoning = _cache(full, "reasoning_grounding", full.assistant_text)
    decision = _cache(full, "decision", _decision_text(full))

    view = project_stage_ii_cfm_view_v1_from_stage_i_v2(
        full,
        trajectory_target=target.target,
        reasoning_context=reasoning,
        decision_context=decision,
        review_status="not_required",
    )

    assert view.training_contract == "stage_ii_cfm_v1"
    assert view.parent_stage_i_sample_id == full.sample_id
    assert view.trajectory_target.valid_step_count == 16
    assert view.reasoning_context.projected_input_digest == decision.projected_input_digest


def test_current_v4_direct_latent_view_is_cacheless_and_fails_closed() -> None:
    """Direct latent must not be a V1 view with cache fields blanked out."""

    full = _stage_i_full_v2()
    target = _trajectory_v1()
    assert target.accepted and target.target is not None
    view = project_stage_ii_direct_latent_view_from_stage_i_v2(
        full,
        trajectory_target=target.target,
        review_status="not_required",
    )

    payload = view.model_dump(mode="json", exclude_none=True)
    assert view.sample_id == f"{full.sample_id}:stage_ii_direct_latent"
    assert view.direct_latent_profile_id == "reasoning_v4_6_direct_latent_v1"
    assert "reasoning_context" not in payload
    assert "decision_context" not in payload
    assert len(view.projected_input_digest) == 64

    payload["reasoning_context"] = {}
    with pytest.raises(ValidationError):
        StageIICFMDirectLatentViewV1.model_validate(payload)


def test_bulk_builder_accepts_v2_full_cot_with_the_clean_v1_target(
    tmp_path: Path,
) -> None:
    """The source builder must retain the V2 parent rather than silently skipping it."""

    from test_stage_ii_cfm import _source_registry_payload

    full = _stage_i_full_v2()
    reasoning = _cache(full, "reasoning_grounding", full.assistant_text)
    decision = _cache(full, "decision", _decision_text(full))
    raw_target = {
        "stage_i_sample_id": full.sample_id,
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
    _write_jsonl(paths["stage_i"], [full.model_dump(mode="json", exclude_none=True)])
    _write_jsonl(paths["trajectory"], [raw_target])
    _write_jsonl(paths["reasoning"], [reasoning.model_dump(mode="json", exclude_none=True)])
    _write_jsonl(paths["decision"], [decision.model_dump(mode="json", exclude_none=True)])
    paths["registry"].write_text(
        json.dumps(_source_registry_payload(sidewalk_sources=("dataset_pose_trajectory",))),
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
        max_source_step_m=3.0,
    )

    assert report["counts"]["output_views"] == 1
    assert report["counts"]["eligible_references"] == 1
    assert paths["drops"].read_text(encoding="utf-8") == ""


def test_bulk_builder_direct_latent_requires_no_context_cache_inputs(tmp_path: Path) -> None:
    """The direct CLI route materializes the same parent without Pass R/D."""

    from test_stage_ii_cfm import _source_registry_payload

    full = _stage_i_full_v2()
    raw_target = {
        "stage_i_sample_id": full.sample_id,
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
    stage_i_path = tmp_path / "stage_i.jsonl"
    trajectory_path = tmp_path / "trajectory.jsonl"
    registry_path = tmp_path / "registry.json"
    _write_jsonl(stage_i_path, [full.model_dump(mode="json", exclude_none=True)])
    _write_jsonl(trajectory_path, [raw_target])
    registry_path.write_text(
        json.dumps(_source_registry_payload(sidewalk_sources=("dataset_pose_trajectory",))),
        encoding="utf-8",
    )
    report = build_stage_ii_cfm_views(
        stage_i_jsonl=stage_i_path,
        trajectory_jsonl=trajectory_path,
        reasoning_cache_jsonl=None,
        decision_cache_jsonl=None,
        trajectory_source_registry_json=registry_path,
        output_view_jsonl=tmp_path / "views.jsonl",
        output_qc_jsonl=tmp_path / "qc.jsonl",
        output_drop_jsonl=tmp_path / "drops.jsonl",
        direct_latent=True,
        max_source_step_m=3.0,
    )

    assert report["mode"] == "direct_latent"
    assert report["counts"]["output_views"] == 1
    assert report["counts"]["direct_latent_rows"] == 1
    payload = json.loads((tmp_path / "views.jsonl").read_text(encoding="utf-8"))
    assert payload["sample_id"].endswith(":stage_ii_direct_latent")
    assert "reasoning_context" not in payload
    assert "decision_context" not in payload


def test_stage_ii_v2_decision_only_requires_signed_dropout_absence() -> None:
    full = _stage_i_full_v2()
    dropout = project_stage_i_reasoning_dropout_from_full_v2(full)
    decision = _cache(dropout, "decision", _decision_text(full))

    view = project_stage_ii_cfm_view_v2(
        dropout,
        trajectory_target=_trajectory_v2(),
        reasoning_context=None,
        decision_context=decision,
        review_status="resolved",
    )

    assert view.context_mode == "decision_only"
    assert isinstance(view.reasoning, StageIIReasoningAbsentV2)
    assert view.reasoning.reasoning_state == "absent_by_reasoning_dropout"
    assert view.parent_stage_i_sample_id == full.sample_id
    assert view.stage_i_route_sample_id == dropout.sample_id
    assert view.reasoning.dropout_link == dropout.dropout_link

    with pytest.raises(ValueError, match="requires a Pass-R cache"):
        project_stage_ii_cfm_view_v2(
            full,
            trajectory_target=_trajectory_v2(),
            reasoning_context=None,
            decision_context=_cache(full, "decision", _decision_text(full)),
            review_status="not_required",
        )


def test_stage_ii_v2_rejected_reasoning_cannot_masquerade_as_dropout() -> None:
    full = _stage_i_full_v2()
    dropout = project_stage_i_reasoning_dropout_from_full_v2(full)
    rejected_reasoning = _cache(full, "reasoning_grounding", "not canonical")
    assert rejected_reasoning.parse_status == "rejected"

    with pytest.raises(ValueError, match="forbids both accepted and rejected"):
        project_stage_ii_cfm_view_v2(
            dropout,
            trajectory_target=_trajectory_v2(),
            reasoning_context=rejected_reasoning,
            decision_context=_cache(dropout, "decision", _decision_text(full)),
            review_status="not_required",
        )


def test_stage_ii_v2_rejected_decision_is_a_drop_audit() -> None:
    full = _stage_i_full_v2()
    rejected = _cache(full, "decision", "not a Decision section")
    assert rejected.parse_status == "rejected"

    result = project_stage_ii_cfm_view_with_audit_v2(
        full,
        trajectory_target=_trajectory_v2(),
        reasoning_context=_cache(full, "reasoning_grounding", full.assistant_text),
        decision_context=rejected,
        review_status="not_required",
    )

    assert isinstance(result, StageIIProjectionDropV2)
    assert result.reason_code == "decision_cache_rejected"
    assert result.requested_context_mode == "reasoning_plus_decision"


def test_stage_ii_v2_absence_lineage_and_view_digest_reject_mutation() -> None:
    full = _stage_i_full_v2()
    dropout = project_stage_i_reasoning_dropout_from_full_v2(full)
    view = project_stage_ii_cfm_view_v2(
        dropout,
        trajectory_target=_trajectory_v2(),
        reasoning_context=None,
        decision_context=_cache(dropout, "decision", _decision_text(full)),
        review_status="not_required",
    )
    mutated = copy.deepcopy(view.model_dump(mode="json"))
    mutated["reasoning"]["stage_i_dropout_projection_digest"] = "9" * 64
    with pytest.raises(ValidationError):
        StageIICFMTrainingViewV2.model_validate(mutated)


def test_stage_ii_v2_route_pool_builder_emits_one_signed_pair(tmp_path: Path) -> None:
    full = _stage_i_full_v2()
    reasoning = _cache(full, "reasoning_grounding", full.assistant_text)
    decision = _cache(full, "decision", _decision_text(full))
    sidecar = _trajectory_sidecar_v2(full)
    stage_i_path = tmp_path / "stage_i.jsonl"
    sidecar_path = tmp_path / "trajectory.jsonl"
    reasoning_path = tmp_path / "reasoning.jsonl"
    decision_path = tmp_path / "decision.jsonl"
    route_path = tmp_path / "routes.jsonl"
    qc_path = tmp_path / "qc.jsonl"
    drop_path = tmp_path / "drops.jsonl"
    _write_jsonl(stage_i_path, [full.model_dump(mode="json")])
    _write_jsonl(sidecar_path, [sidecar.model_dump(mode="json")])
    _write_jsonl(reasoning_path, [reasoning.model_dump(mode="json")])
    _write_jsonl(decision_path, [decision.model_dump(mode="json")])

    report = build_stage_ii_cfm_route_pool_v2(
        stage_i_full_jsonl=stage_i_path,
        trajectory_sidecar_jsonl=sidecar_path,
        reasoning_cache_jsonl=reasoning_path,
        decision_cache_jsonl=decision_path,
        output_route_jsonl=route_path,
        output_qc_jsonl=qc_path,
        output_drop_jsonl=drop_path,
    )

    routes = [
        StageIICFMTrainingViewV2.model_validate(json.loads(line))
        for line in route_path.read_text(encoding="utf-8").splitlines()
    ]
    assert {route.context_mode for route in routes} == {
        "reasoning_plus_decision",
        "decision_only",
    }
    assert report["counts"]["parent_routes:paired"] == 1
    assert report["counts"]["output_routes"] == 2
    assert len(qc_path.read_text(encoding="utf-8").splitlines()) == 1
    assert drop_path.read_text(encoding="utf-8") == ""


def test_stage_ii_v2_route_pool_accepts_dataset_qualified_split_identity(
    tmp_path: Path,
) -> None:
    """A legacy native clip ID and its dataset-qualified V2 ID are equivalent."""

    full = _stage_i_full_v2()
    reasoning = _cache(full, "reasoning_grounding", full.assistant_text)
    decision = _cache(full, "decision", _decision_text(full))
    sidecar_payload = _trajectory_sidecar_v2(full).model_dump(mode="json")
    sidecar_payload["split_group_id"] = (
        f"{full.dataset_source}:{full.split_group_id}"
    )
    sidecar_payload.pop("sidecar_digest", None)
    sidecar_payload["sidecar_digest"] = stage_ii_digest_v1(sidecar_payload)
    sidecar = StageIITrajectorySidecarV2.model_validate(sidecar_payload)
    paths = {
        name: tmp_path / f"{name}.jsonl"
        for name in (
            "stage_i",
            "trajectory",
            "reasoning",
            "decision",
            "routes",
            "qc",
            "drops",
        )
    }
    _write_jsonl(paths["stage_i"], [full.model_dump(mode="json")])
    _write_jsonl(paths["trajectory"], [sidecar.model_dump(mode="json")])
    _write_jsonl(paths["reasoning"], [reasoning.model_dump(mode="json")])
    _write_jsonl(paths["decision"], [decision.model_dump(mode="json")])

    report = build_stage_ii_cfm_route_pool_v2(
        stage_i_full_jsonl=paths["stage_i"],
        trajectory_sidecar_jsonl=paths["trajectory"],
        reasoning_cache_jsonl=paths["reasoning"],
        decision_cache_jsonl=paths["decision"],
        output_route_jsonl=paths["routes"],
        output_qc_jsonl=paths["qc"],
        output_drop_jsonl=paths["drops"],
    )

    assert report["counts"]["parent_routes:paired"] == 1
    assert report["join_contract"]["split_group_key"].startswith(
        "dataset_qualified"
    )
    assert paths["drops"].read_text(encoding="utf-8") == ""


def test_stage_ii_v2_route_pool_does_not_turn_rejected_r_into_absence(
    tmp_path: Path,
) -> None:
    full = _stage_i_full_v2()
    rejected_reasoning = _cache(full, "reasoning_grounding", "not canonical")
    decision = _cache(full, "decision", _decision_text(full))
    sidecar = _trajectory_sidecar_v2(full)
    paths = {
        name: tmp_path / f"{name}.jsonl"
        for name in ("stage_i", "trajectory", "reasoning", "decision", "routes", "qc", "drops")
    }
    _write_jsonl(paths["stage_i"], [full.model_dump(mode="json")])
    _write_jsonl(paths["trajectory"], [sidecar.model_dump(mode="json")])
    _write_jsonl(paths["reasoning"], [rejected_reasoning.model_dump(mode="json")])
    _write_jsonl(paths["decision"], [decision.model_dump(mode="json")])

    report = build_stage_ii_cfm_route_pool_v2(
        stage_i_full_jsonl=paths["stage_i"],
        trajectory_sidecar_jsonl=paths["trajectory"],
        reasoning_cache_jsonl=paths["reasoning"],
        decision_cache_jsonl=paths["decision"],
        output_route_jsonl=paths["routes"],
        output_qc_jsonl=paths["qc"],
        output_drop_jsonl=paths["drops"],
    )

    routes = [json.loads(line) for line in paths["routes"].read_text(encoding="utf-8").splitlines()]
    drops = [json.loads(line) for line in paths["drops"].read_text(encoding="utf-8").splitlines()]
    assert [route["context_mode"] for route in routes] == ["decision_only"]
    assert routes[0]["reasoning"]["reasoning_state"] == "absent_by_reasoning_dropout"
    assert drops[0]["requested_context_mode"] == "reasoning_plus_decision"
    assert drops[0]["reason_code"] == "reasoning_cache_rejected"
    assert report["counts"]["parent_routes:decision_only"] == 1
    assert report["counts"].get("parent_routes:paired", 0) == 0
