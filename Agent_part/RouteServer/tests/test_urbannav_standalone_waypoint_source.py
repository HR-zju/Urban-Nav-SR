"""Tests for the standalone UrbanNav waypoint source adapter."""

from __future__ import annotations

import json
from pathlib import Path

from mcp_server.data.io.jsonl import iter_jsonl
from tools.dataset_processing.build_urbannav_standalone_waypoint_source import (
    build_standalone_waypoint_source,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write a small JSONL fixture."""

    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_standalone_route_requires_explicit_derived_heading_policy(tmp_path: Path) -> None:
    """A route without sensor heading stays unavailable by default."""

    source = tmp_path / "source.jsonl"
    _write_jsonl(
        source,
        [{
            "id": "case-1",
            "origin": {"lat": 30.0, "lng": 120.0},
            "gt_route": [{"lat": 30.0, "lng": 120.0}, {"lat": 30.0001, "lng": 120.0}],
        }],
    )
    report = build_standalone_waypoint_source(
        source,
        tmp_path / "public.jsonl",
        rejects_jsonl=tmp_path / "rejects.jsonl",
        report_json=tmp_path / "report.json",
    )
    assert report["accepted_rows"] == 0
    assert report["reject_counts"] == {"missing_current_heading": 1}


def test_standalone_route_emits_public_condition_without_gt_fields(tmp_path: Path) -> None:
    """Explicit route-tangent smoke produces public route input only."""

    source = tmp_path / "source.jsonl"
    _write_jsonl(
        source,
        [{
            "id": "case-1",
            "instruction": "Find the park.",
            "origin": {"lat": 30.0, "lng": 120.0},
            "gt_destination": {"lat": 30.001, "lng": 120.0},
            "gt_route": [{"lat": 30.0, "lng": 120.0}, {"lat": 30.0001, "lng": 120.0}],
        }],
    )
    output = tmp_path / "public.jsonl"
    report = build_standalone_waypoint_source(
        source,
        output,
        rejects_jsonl=tmp_path / "rejects.jsonl",
        report_json=tmp_path / "report.json",
        heading_source="route_start_tangent",
    )
    row = next(iter_jsonl(output))
    assert report["accepted_rows"] == 1
    assert row["id"] == "case-1"
    assert row["source_dataset"] == "urbannav_standalone_benchmark"
    assert row["route_context"]["independent_of_future_target"] is True
    assert row["route_context"]["heading_source"] == "route_start_tangent"
    assert row["observation_timestamp_s"] == 0.0
    assert row["observation_timestamp_source"] == "standalone_static_epoch"
    assert "gt_route" not in row
    assert "gt_destination" not in row
    assert row["current_heading_rad"] == 0.0
