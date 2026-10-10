"""Boundary coverage for the approved 100 ms annotation alignment contract."""

import json
from types import SimpleNamespace

import pytest

from mcp_server.data.history_anchor_mapping import build_anchor_map
from mcp_server.data.history_label_reuse import anchor_label_evidence


@pytest.mark.parametrize("delta,clip,expected", [
    (100_000_000, "clip", True), (100_000_001, "clip", False), (0, "other", False),
])
def test_anchor_tolerance_includes_boundary(delta, clip, expected):
    observation = SimpleNamespace(dataset_source="sidewalk", source_group_id="anchor")
    old = SimpleNamespace(observation=observation, source_fingerprint="old")
    new = SimpleNamespace(observation=observation, source_fingerprint="new")
    records = {
        "old": dict(clip_id="clip", timestamp_ns=1_000_000_000,
                    current_frame_timestamp_ns=1_000_000_000 - delta,
                    current_frame_id="prepared:1", provenance="fixture"),
        "new": dict(clip_id=clip, timestamp_ns=1_000_000_000,
                    current_frame_id="video:2", provenance="fixture"),
    }
    matched, _, evidence = anchor_label_evidence(
        old, new, media_identities={}, anchor_records=records, anchor_tolerance_ns=100_000_000,
    )
    assert matched is expected
    assert evidence["current_frame_delta_ns"] == delta


def test_map_uses_original_clock_and_preserves_split(tmp_path):
    manifest = tmp_path / "observations.jsonl"
    row = dict(source_parent_id="egowalk:sample", split_group_id="egowalk:clip", split="val",
               source_row=dict(clip_id="clip", frame_idx=3, timestamp=1200),
               observation=dict(source_ns=[1_250_000_000], video_ordinals=[4]))
    manifest.write_text(json.dumps(row) + "\n")
    report = build_anchor_map(manifest, "egowalk", tmp_path / "result", old_sources=[])
    result = json.loads((tmp_path / "result/anchor_map.jsonl").read_text())
    assert report["counts"]["mapped"] == 1
    assert result["old_timestamp_ns"] == 1_200_000_000
    assert result["delta_ns"] == 50_000_000
    assert result["split"] == "val"
