from __future__ import annotations

import json
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
