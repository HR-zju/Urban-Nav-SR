from __future__ import annotations

import json
import copy
from pathlib import Path

import pytest

from mcp_server.contracts.direction_profile import (
    EGO_DEGREE10_TERMINAL_CENTROID_V2,
)
from tools.model_processing.project_reasoning_v4_ego_directions import (
    project_reasoning_v4_ego_direction_manifest,
    project_reasoning_v4_ego_direction_row,
)


def _row(trajectory: list[list[float]]) -> dict[str, object]:
    return {
        "id": "egowalk-case",
        "source_dataset": "egowalk",
        "trajectory_ego_t": trajectory,
        "trajectory_policy": {"source": "egowalk_odometry"},
        "direction_choice_label": 19,
        "direction_name": "stop",
        "direction_choice": "T",
        "direction_source": "low_displacement_rule",
        "direction_bin_mode": "degree10_with_stop",
    }


def test_history_sidecar_direction_preserves_gt_and_flu_sign():
    """Left-positive FLU geometry becomes a negative right-positive angle."""
    from urbannav_contracts.stage_ii_cfm import (
        StageIITrajectoryPolicyV2, STAGE_II_TRAJECTORY_TARGET_V2_SCHEMA_VERSION, stage_ii_digest_v1)
    from tools.model_processing.project_reasoning_v4_ego_directions import project_history_sidecar_direction

    policy = StageIITrajectoryPolicyV2(source_adapter_id="sidewalk_npy_pose_4s_v2",
        trajectory_source="sidewalk_frame_pose_odometry", raw_timestamp_unit="nanosecond", yaw_offset_deg=0)
    target = {"schema_version": STAGE_II_TRAJECTORY_TARGET_V2_SCHEMA_VERSION,
              "trajectory_ego_t": [[2.0, 0.35, 0.0] for _ in range(32)],
              "action_mask": [True] * 32, "valid_step_count": 32,
              "policy": policy.model_dump(mode="json"), "source_target_digest": "a" * 64}
    target["target_digest"] = stage_ii_digest_v1(target)
    record = {"source_parent_id": "sidewalk:case", "source_row": {"id": "case"},
              "trajectory_sidecar": {"sample_id": "sidewalk:case", "dataset_source": "sidewalk",
                                     "trajectory_target": target}}
    original = copy.deepcopy(record)
    source = project_history_sidecar_direction(record, dataset_source="sidewalk",
        target_profile_id="ego_degree10_12way_terminal_centroid_v2")
    assert source["direction_name"] == "left_10"
    assert source["direction_raw_angle_deg"] < 0
    assert source["direction_projection"]["target_digest"] == target["target_digest"]
    assert record == original


def test_missing_history_sidecar_has_explicit_exclusion(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"source_parent_id": "sidewalk:missing", "trajectory_sidecar": None}) + "\n")
    report = project_reasoning_v4_ego_direction_manifest(source, tmp_path / "out.jsonl",
        report_json=tmp_path / "report.json", dataset_source="sidewalk", history_package=True,
        direction_from_sidecar=True, target_profile_id="ego_degree10_12way_terminal_centroid_v2")
    assert report["counts"]["missing_linked_sidecar"] == 1
    assert (tmp_path / "out.jsonl").read_text() == ""
    assert json.loads(Path(report["exclusion_jsonl"]).read_text()) == {
        "source_parent_id": "sidewalk:missing", "reason": "missing_linked_sidecar"}


def test_project_ego_direction_uses_metric_terminal_centroid() -> None:
    projected = project_reasoning_v4_ego_direction_row(
        _row(
            [
                [0.1, 0.5, 0.2, 1.0],
                [0.4, 1.0, 0.4, 1.0],
                [0.8, 1.5, 0.6, 1.0],
                [1.2, 2.0, 0.8, 1.0],
                [1.6, 2.5, 1.0, 1.0],
            ]
        )
    )

    supervision = projected["direction_supervision"]
    assert supervision["profile_id"] == EGO_DEGREE10_TERMINAL_CENTROID_V2
    assert supervision["label_name"] == "right_30"
    assert supervision["source"] == "ego_x_right_y_front_terminal_centroid_rule_v2"
    assert projected["direction_reference_start_index"] == 3
    assert projected["direction_reference_point_count"] == 2
    assert projected["direction_projection"]["legacy_direction"]["source"] == "low_displacement_rule"


def test_project_ego_direction_uses_profile_stop_threshold() -> None:
    projected = project_reasoning_v4_ego_direction_row(
        _row([[0.01, 0.02, 0.2, 0.1], [0.02, 0.03, 0.4, 0.1]])
    )

    supervision = projected["direction_supervision"]
    assert supervision["label_name"] == "stop"
    assert supervision["angle_deg"] is None
    assert supervision["source"] == "ego_x_right_y_front_terminal_centroid_rule_v2"


def test_project_ego_direction_rejects_untrusted_trajectory() -> None:
    row = _row([[0.0, 1.0, 0.2, 1.0]])
    row["trajectory_policy"] = {"source": "frodobots_proxy"}

    with pytest.raises(ValueError, match="not approved"):
        project_reasoning_v4_ego_direction_row(row)


def test_project_ego_direction_manifest_reports_changes(
    tmp_path: Path,
) -> None:
    input_jsonl = tmp_path / "input.jsonl"
    output_jsonl = tmp_path / "output.jsonl"
    report_json = tmp_path / "report.json"
    input_jsonl.write_text(
        json.dumps(_row([[0.0, 1.0, 0.2, 1.0], [0.0, 2.0, 0.4, 1.0]]))
        + "\n",
        encoding="utf-8",
    )

    report = project_reasoning_v4_ego_direction_manifest(
        input_jsonl,
        output_jsonl,
        report_json=report_json,
    )

    assert report["passed"] is True
    assert report["trajectory_read"] is True
    assert report["direction_rederived"] is True
    assert report["counts"]["input_rows"] == 1
    assert report["counts"]["output_rows"] == 1
    assert report["counts"]["legacy_label_changed_rows"] == 1
    assert len(output_jsonl.read_text(encoding="utf-8").splitlines()) == 1
