"""Check preserved supervision identities and explicit BEV causality limits."""

import json
from pathlib import Path

import numpy as np
import pytest

from mcp_server.data.history_builder import build_clip_observations
from mcp_server.data.history_observation import CausalBevIndex, EgoWalkHistoryProcessor, SampleRejected
from mcp_server.data.stage_ii.sources import load_stage_ii_trajectory_source_registry_v1, stage_ii_assign_split_v1


def test_builder_preserves_original_sidecar_and_excludes_short_history(tmp_path, monkeypatch):
    """Canonical identities retain splits and original sidecar contents."""
    registry = load_stage_ii_trajectory_source_registry_v1(
        Path(__file__).parents[1] / "config/stage_ii_cfm/trajectory_source_registry_v1.json")
    spec = next(item for item in registry.datasets if item.dataset_source == "egowalk")
    processor = EgoWalkHistoryProcessor(0)
    monkeypatch.setattr(processor, "load_timeline", lambda source: (
        np.arange(11, dtype=np.int64) * 500_000_000, np.arange(11)))
    rows = [{"id": f"frame{i}", "clip_id": "clip", "timestamp": i * 500,
             "frame_idx": i, "response": "original label"} for i in (1, 8, 9)]
    split = stage_ii_assign_split_v1("egowalk:clip", registry)
    sidecar = {"sample_id": "egowalk:frame8", "split_group_id": "egowalk:clip",
               "dataset_source": "egowalk", "split": split, "sidecar_digest": "original",
               "trajectory_target": {"preserved": True}}
    accepted, rejected, summary = build_clip_observations(
        rows, source=tmp_path / "clip.parquet", processor=processor, spec=spec,
        registry=registry, sidecars={sidecar["sample_id"]: sidecar}, bev=None)
    assert rejected == [{"source_parent_id": "egowalk:frame1", "reason": "insufficient_history"}]
    assert accepted[0]["trajectory_sidecar"] is sidecar
    assert accepted[0]["source_row"] is rows[1]
    assert accepted[1]["trajectory_sidecar"] is None
    assert accepted[0]["observation"]["bev"] is None
    assert summary["unique_fpv"] == 10
    assert summary["history_with_sidecar_by_split"] == {split: 1}
    sidecar["split"] = "wrong"
    with pytest.raises(ValueError, match="sidecar identity/split mismatch"):
        build_clip_observations(rows, source=tmp_path / "clip.parquet", processor=processor,
                                spec=spec, registry=registry,
                                sidecars={sidecar["sample_id"]: sidecar}, bev=None)


def test_bev_selects_predecessor_and_preserves_sensor_uncertainty(tmp_path):
    """Map-only content and temporal predecessor selection are separate claims."""
    (tmp_path / "map.jpg").write_bytes(b"fixture")
    index = {"clip_id": "clip", "input_profile": "causal_map_v1",
             "bev_route_scope": "none", "draw_trajectory": False,
             "causality_scope": "prepared_frame_gps_rows", "sensor_sync_causality": "unverified",
             "entries": [{"timestamp": t, "frame_idx": i, "bev_image_path": "map.jpg"}
                         for i, t in enumerate((100, 200))]}
    path = tmp_path / "bev_index.json"
    path.write_text(json.dumps(index))
    bev = CausalBevIndex(path, clip_id="clip", source_ns=np.array([100, 200]), max_age_ns=80)
    assert bev.select(150)["timestamp"] == 100
    assert bev.select(200)["age_ns"] == 0
    assert bev.select(150)["bev_tolerance_ns"] == 80
    assert bev.select(150)["sensor_sync_causality"] == "unverified"
    with pytest.raises(SampleRejected, match="causal_bev_missing"):
        bev.select(99)
    with pytest.raises(SampleRejected, match="bev_age_exceeded"):
        bev.select(281)
    index["draw_trajectory"] = True
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="causal_map_v1"):
        CausalBevIndex(path, clip_id="clip", source_ns=np.array([100, 200]), max_age_ns=80)


def test_recorded_bev_relocates_paths_and_retains_route(tmp_path):
    """Recorded route remains visible provenance after latest-causal selection."""
    (tmp_path / "bev_images").mkdir()
    (tmp_path / "bev_images/map.jpg").write_bytes(b"fixture")
    index = {"clip_id": "clip", "draw_trajectory": True, "bev_route_scope": "global_full",
             "bev_sample_fps": 1, "entries": [
                 {"timestamp": 100, "frame_idx": 0, "bev_image_path": "/old/clip/bev_images/map.jpg"}]}
    path = tmp_path / "bev_index.json"
    path.write_text(json.dumps(index))
    item = CausalBevIndex(path, clip_id="clip", source_ns=np.array([100, 200]), max_age_ns=100).select(150)
    assert item["bev_image_path"] == "bev_images/map.jpg"
    assert item["input_profile"] == "recorded_bev_v1"
    assert item["draw_trajectory"] is True
    assert item["bev_route_scope"] == "global_full"
    assert item["age_ns"] == 50
    assert item["bev_tolerance_ns"] == 100
    assert json.loads(path.read_text()) == index
