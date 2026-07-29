from __future__ import annotations

import json

import pytest

from mcp_server.contracts.direction_profile import (
    FPV_DEGREE10_TERMINAL_CENTROID_V2,
)
from tools.model_processing.stamp_reasoning_v4_directions import (
    stamp_reasoning_v4_direction_manifest,
    stamp_reasoning_v4_direction_row,
)


def _legacy_row() -> dict[str, object]:
    return {
        "id": "foxglove-case-1",
        "direction_choice_label": 9,
        "direction_name": "straight",
        "direction_choice": "J",
        "direction_angle_deg": 0.0,
        "direction_source": (
            "pixel_trajectory_terminal_centroid_rule_v2"
        ),
        "direction_bin_mode": "degree10_with_stop",
    }


def test_stamp_adds_only_profile_identity() -> None:
    stamped, added = stamp_reasoning_v4_direction_row(
        _legacy_row(),
        profile_id=FPV_DEGREE10_TERMINAL_CENTROID_V2,
    )

    assert added is True
    assert stamped["direction_supervision"] == {
        "profile_id": "fpv_degree10_terminal_centroid_v2",
        "profile_version": "2.0",
        "label_index": 9,
        "label_name": "straight",
        "choice": "J",
        "angle_deg": 0.0,
        "source": "pixel_trajectory_terminal_centroid_rule_v2",
        "bin_mode": "degree10_with_stop",
        "derivation_version": (
            "pixel_trajectory_terminal_centroid_rule_v2"
        ),
    }


def test_stamp_is_idempotent() -> None:
    first, _ = stamp_reasoning_v4_direction_row(
        _legacy_row(),
        profile_id=FPV_DEGREE10_TERMINAL_CENTROID_V2,
    )
    second, added = stamp_reasoning_v4_direction_row(
        first,
        profile_id=FPV_DEGREE10_TERMINAL_CENTROID_V2,
    )

    assert added is False
    assert second == first


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("direction_name", "left_10"),
        ("direction_choice", "B"),
        ("direction_bin_mode", "degree10_12way_with_stop"),
        ("direction_source", "existing_direction_name"),
        ("direction_source", "route_theta"),
    ],
)
def test_stamp_rejects_inconsistent_or_untrusted_fields(
    field: str,
    value: object,
) -> None:
    row = _legacy_row()
    row[field] = value

    with pytest.raises(ValueError):
        stamp_reasoning_v4_direction_row(
            row,
            profile_id=FPV_DEGREE10_TERMINAL_CENTROID_V2,
        )


def test_manifest_report_states_no_rederivation(tmp_path) -> None:
    input_jsonl = tmp_path / "input.jsonl"
    output_jsonl = tmp_path / "output.jsonl"
    report_json = tmp_path / "report.json"
    input_jsonl.write_text(
        json.dumps(_legacy_row()) + "\n",
        encoding="utf-8",
    )

    report = stamp_reasoning_v4_direction_manifest(
        input_jsonl,
        output_jsonl,
        report_json=report_json,
        profile_id=FPV_DEGREE10_TERMINAL_CENTROID_V2,
    )

    assert report["direction_rederived"] is False
    assert report["trajectory_read"] is False
    assert report["counts"] == {
        "input_rows": 1,
        "output_rows": 1,
        "newly_stamped_rows": 1,
    }
    assert report_json.is_file()
    assert len(output_jsonl.read_text(encoding="utf-8").splitlines()) == 1
