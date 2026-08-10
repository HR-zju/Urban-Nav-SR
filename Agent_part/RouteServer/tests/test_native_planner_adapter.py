"""Focused tests for fail-closed native planner-route adapters V1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mcp_server.contracts.direction_profile import (
    FPV_DEGREE10_AFFORDANCE7_V1,
    FPV_DEGREE10_TERMINAL_CENTROID_V2,
    resolve_affordance_probe_profile,
    resolve_stage_a_direction_profile,
)
from mcp_server.contracts.language_waypoint_benchmark import NativePlannerRouteLogV1
from mcp_server.contracts.reasoning_dataset_v4 import (
    REASONING_SOURCE_V4_SCHEMA_VERSION,
    DirectionContextV4,
    InputModalitiesV4,
    NavObservationV4,
    ReasoningSourceV4,
    reasoning_source_fingerprint_v4,
)
from mcp_server.data.benchmark.native_planner_adapter import (
    build_foxglove_native_gps_anchors_v1,
    build_planner_records_from_native_logs_v1,
)
from mcp_server.data.io.jsonl import iter_jsonl


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write compact JSONL fixture rows.

    Args:
        path: Fixture destination.
        rows: JSON-compatible rows.
    """

    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _foxglove_source(tmp_path: Path) -> ReasoningSourceV4:
    """Build one Foxglove source whose record maps to a raw clip frame.

    Args:
        tmp_path: Temporary fixture directory.

    Returns:
        Valid canonical Reasoning V4 source.
    """

    action = resolve_stage_a_direction_profile(FPV_DEGREE10_TERMINAL_CENTROID_V2)
    probe = resolve_affordance_probe_profile(FPV_DEGREE10_AFFORDANCE7_V1)
    current = tmp_path / "current.jpg"
    current.write_bytes(b"schema-only")
    observation = NavObservationV4(
        source_group_id="foxglove:clip_test_000002",
        dataset_source="foxglove",
        source_record_id="clip_test_000002",
        split_group_id="foxglove:clip_test",
        images=[str(current)],
        image_roles=["fpv.current"],
        input_modalities=InputModalitiesV4(),
        evidence_catalog=[{"ref": "fpv.current", "modality": "fpv"}],
        direction_context=DirectionContextV4(
            action_profile_id=FPV_DEGREE10_TERMINAL_CENTROID_V2,
            action_profile_version=action["profile_version"],
            affordance_probe_profile_id=FPV_DEGREE10_AFFORDANCE7_V1,
            affordance_probe_profile_version=probe["profile_version"],
        ),
    )
    payload = {
        "schema_version": REASONING_SOURCE_V4_SCHEMA_VERSION,
        "observation": observation.model_dump(mode="json"),
        "direction_supervision": None,
        "direction_geometry": None,
        "trajectory_ego_t": None,
        "trajectory_policy": None,
        "action_mask": None,
        "warnings": [],
    }
    payload["source_fingerprint"] = reasoning_source_fingerprint_v4(payload)
    return ReasoningSourceV4.model_validate(payload)


def _planner_log(source: ReasoningSourceV4, *, timestamp_s: float) -> NativePlannerRouteLogV1:
    """Build an independent planner log matching the native current pose.

    Args:
        source: Canonical source providing the explicit join key.
        timestamp_s: Planner response time in seconds.

    Returns:
        Typed planner log with no future trajectory fields.
    """

    return NativePlannerRouteLogV1(
        route_log_id=f"planner-log-{timestamp_s}",
        dataset_source="foxglove",
        source_record_id=source.observation.source_record_id,
        plan_timestamp_s=timestamp_s,
        current_wgs84={"latitude_deg": 30.00001, "longitude_deg": 120.0},
        current_heading_rad=0.0,
        goal_wgs84={"latitude_deg": 30.00020, "longitude_deg": 120.0},
        planner_polyline_wgs84=[
            {"latitude_deg": 30.00001, "longitude_deg": 120.0},
            {"latitude_deg": 30.00010, "longitude_deg": 120.0},
            {"latitude_deg": 30.00020, "longitude_deg": 120.0},
        ],
        planner_id="independent_test_planner",
        route_version="request-v1",
        datum="WGS84",
        source_kind="map_planner",
    )


def test_native_anchor_and_planner_join_are_trajectory_free(tmp_path: Path) -> None:
    """A past-only GPS anchor joins only a timely independent planner route.

    Args:
        tmp_path: Temporary fixture directory.
    """

    source = _foxglove_source(tmp_path)
    source_jsonl = tmp_path / "source.jsonl"
    raw_root = tmp_path / "raw"
    clip_root = raw_root / "clip_test"
    clip_root.mkdir(parents=True)
    np.save(
        clip_root / "frame_gps.npy",
        np.asarray(
            [
                [30.0, 120.0],
                [30.000005, 120.0],
                [30.000010, 120.0],
                [30.000015, 120.0],
            ],
            dtype=np.float64,
        ),
    )
    np.save(clip_root / "frame_times.npy", np.asarray([1e9, 2e9, 3e9, 4e9]))
    _write_jsonl(source_jsonl, [source.model_dump(mode="json")])
    anchors_jsonl = tmp_path / "anchors.jsonl"
    anchor_report = build_foxglove_native_gps_anchors_v1(
        source_jsonl=source_jsonl,
        foxglove_raw_root=raw_root,
        output_jsonl=anchors_jsonl,
        report_json=tmp_path / "anchor_report.json",
        heading_lookback_frames=4,
        min_heading_displacement_m=0.25,
    )
    assert anchor_report["available_count"] == 1
    anchor = next(iter_jsonl(anchors_jsonl))
    assert anchor["localization_source"] == "native_gps_past_bearing"
    assert anchor["current_heading_rad"] == 0.0

    planner_jsonl = tmp_path / "planner.jsonl"
    _write_jsonl(planner_jsonl, [_planner_log(source, timestamp_s=2.5).model_dump(mode="json")])
    records_jsonl = tmp_path / "records.jsonl"
    report = build_planner_records_from_native_logs_v1(
        anchor_jsonl=anchors_jsonl,
        planner_log_jsonl=planner_jsonl,
        output_jsonl=records_jsonl,
        decisions_jsonl=tmp_path / "decisions.jsonl",
        report_json=tmp_path / "route_report.json",
    )
    assert report["accepted_record_count"] == 1
    record = next(iter_jsonl(records_jsonl))
    assert record["parent_sample_id"] == source.observation.source_group_id
    assert record["provenance"]["independent_of_future_target"] is True
    assert "trajectory" not in json.dumps(record)


def test_native_planner_join_rejects_stale_records(tmp_path: Path) -> None:
    """A stale planner response stays rejected instead of becoming guidance.

    Args:
        tmp_path: Temporary fixture directory.
    """

    source = _foxglove_source(tmp_path)
    anchor_jsonl = tmp_path / "anchors.jsonl"
    _write_jsonl(
        anchor_jsonl,
        [
            {
                "schema_version": "urbannav_native_planner_anchor_v1",
                "anchor_id": "anchor-stale",
                "parent_sample_id": source.observation.source_group_id,
                "dataset_source": "foxglove",
                "source_record_id": source.observation.source_record_id,
                "observation_timestamp_s": 30.0,
                "current_wgs84": {"latitude_deg": 30.00001, "longitude_deg": 120.0},
                "current_heading_rad": 0.0,
                "localization_source": "native_gps_past_bearing",
                "available": True,
                "reason": "native_gps_anchor_available",
            }
        ],
    )
    planner_jsonl = tmp_path / "planner.jsonl"
    _write_jsonl(planner_jsonl, [_planner_log(source, timestamp_s=1.0).model_dump(mode="json")])
    records_jsonl = tmp_path / "records.jsonl"
    report = build_planner_records_from_native_logs_v1(
        anchor_jsonl=anchor_jsonl,
        planner_log_jsonl=planner_jsonl,
        output_jsonl=records_jsonl,
        decisions_jsonl=tmp_path / "decisions.jsonl",
        report_json=tmp_path / "report.json",
    )
    assert report["accepted_record_count"] == 0
    decision = next(iter_jsonl(tmp_path / "decisions.jsonl"))
    assert decision["reason"] == "planner_record_stale"
