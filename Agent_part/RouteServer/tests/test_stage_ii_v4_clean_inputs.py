from __future__ import annotations

import json
from pathlib import Path

from mcp_server.contracts.stage_ii_cfm import (
    STAGE_II_LEGACY_DIRECT_PROFILE_ID,
    STAGE_II_LEGACY_DIRECT_PROFILE_VERSION,
    STAGE_II_LEGACY_DIRECT_TARGET_SCHEMA_VERSION,
    stage_ii_digest_v1,
)
from mcp_server.data.stage_i_views import project_stage_i_reasoning_view_v2
from mcp_server.data.stage_ii.trajectory import build_stage_ii_trajectory_target_v1
from tools.model_processing.build_stage_ii_v4_clean_inputs import (
    build_stage_ii_v4_clean_inputs,
)

from test_stage_i_sft import _formal_reasoning_view
from test_stage_ii_cfm import _source_registry_payload


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write a compact JSONL test artifact."""

    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _raw_sidecar(sample_id: str) -> dict:
    """Return one deterministic raw timed source sidecar."""

    return {
        "stage_i_sample_id": sample_id,
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


def _clean_target(
    *,
    parent_id: str,
    source_group_id: str,
    split_group_id: str,
    split: str,
) -> dict:
    """Build an immutable clean-target ledger row for one synthetic parent."""

    target = build_stage_ii_trajectory_target_v1(
        sample_id=parent_id,
        dataset_source="sidewalk",
        raw_trajectory=_raw_sidecar(parent_id)["trajectory_ego_t"],
        action_mask=[True] * 21,
        source_format="ego_xy_t_speed_right_front_v1",
        trajectory_source="dataset_pose_trajectory",
        allowed_trajectory_sources={"dataset_pose_trajectory"},
        yaw_source="path_tangent",
        max_source_step_m=3.0,
    )
    assert target.accepted and target.target is not None
    payload = {
        "schema_version": STAGE_II_LEGACY_DIRECT_TARGET_SCHEMA_VERSION,
        "profile_id": STAGE_II_LEGACY_DIRECT_PROFILE_ID,
        "profile_version": STAGE_II_LEGACY_DIRECT_PROFILE_VERSION,
        "sample_id": f"stage-ii-legacy-direct:{parent_id}",
        "legacy_parent_id": parent_id,
        "split": split,
        "source_group_id": source_group_id,
        "split_group_id": split_group_id,
        "dataset_source": "sidewalk",
        "legacy_source_name": "synthetic",
        "legacy_source_line_no": 1,
        "legacy_split_row_digest": "1" * 64,
        "legacy_source_row_digest": "2" * 64,
        "legacy_visible_input_digest": "3" * 64,
        "trajectory_target": target.target.model_dump(mode="json"),
        "trajectory_qc_digest": target.qc.qc_digest,
    }
    payload["record_digest"] = stage_ii_digest_v1(payload)
    return payload


def test_v4_clean_input_builder_rechecks_targets_and_records_source_coverage(
    tmp_path: Path,
) -> None:
    """Use only current V4 inputs while retaining clean-target coverage drops."""

    view = project_stage_i_reasoning_view_v2(
        _formal_reasoning_view(),
        task_type="full_cot",
        modality_profile_id="fpv_instruction_v1",
        instruction_training_role="primary",
    )
    source = tmp_path / "source.jsonl"
    train_targets = tmp_path / "train_targets.jsonl"
    validation_targets = tmp_path / "validation_targets.jsonl"
    train_sidecars = tmp_path / "train_sidecars.jsonl"
    validation_sidecars = tmp_path / "validation_sidecars.jsonl"
    registry = tmp_path / "registry.json"
    output = tmp_path / "output"

    _write_jsonl(source, [view.model_dump(mode="json", exclude_none=True)])
    _write_jsonl(train_targets, [_clean_target(
        parent_id=view.sample_id,
        source_group_id=view.source_group_id,
        split_group_id=view.split_group_id,
        split="train",
    )])
    _write_jsonl(validation_targets, [_clean_target(
        parent_id="sidewalk:missing:formal",
        source_group_id="sidewalk:missing",
        split_group_id="clip-missing",
        split="validation",
    )])
    _write_jsonl(train_sidecars, [_raw_sidecar(view.sample_id)])
    _write_jsonl(validation_sidecars, [])
    registry.write_text(
        json.dumps(
            _source_registry_payload(
                sidewalk_sources=("dataset_pose_trajectory",),
            )
        ),
        encoding="utf-8",
    )

    report = build_stage_ii_v4_clean_inputs(
        stage_i_jsonl=source,
        clean_train_targets_jsonl=train_targets,
        clean_validation_targets_jsonl=validation_targets,
        prebuilt_train_sidecars_jsonl=train_sidecars,
        prebuilt_validation_sidecars_jsonl=validation_sidecars,
        trajectory_source_registry_json=registry,
        output_dir=output,
    )

    assert report["counts"]["selected:train"] == 1
    assert report["counts"]["drop:stage_i_v4_source_missing:validation"] == 1
    assert report["full_clean_target_coverage_complete"] is False
    selected = json.loads(
        (output / "trajectory_sidecars_train.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert selected["clean_target_max_source_step_m"] == 3.0
    assert selected["clean_target_trajectory_digest"]
