"""Multi-clip planning aggregates unique media without decoding video."""

import json
from pathlib import Path

import numpy as np
import pytest

from mcp_server.data.history_corpus import plan_history_corpus
from mcp_server.data.history_observation import EgoWalkHistoryProcessor
from mcp_server.data.history_package import index_history_manifests
from mcp_server.data.stage_ii.sources import load_stage_ii_trajectory_source_registry_v1


def test_foxglove_missing_mapping_is_explicit_clip_exclusion(tmp_path):
    registry = load_stage_ii_trajectory_source_registry_v1(
        Path(__file__).parents[1] / "config/stage_ii_cfm/trajectory_source_registry_v1.json")
    spec = next(item for item in registry.datasets if item.dataset_source == "foxglove")
    output = tmp_path / "plan"
    result = plan_history_corpus(jobs=[{"clip_id": "unverified"}], source_index=tmp_path / "index",
        spec=spec, registry=registry, output=output, max_lag_ns=100, max_bev_age_ns=100)
    assert result["excluded_clips"] == 1
    assert json.loads((output / "clips.jsonl").read_text()) == {
        "clip_id": "unverified", "status": "excluded", "reason": "video_mapping_unverified"}


def test_batch_plan_aggregates_and_retains_partial_failure(tmp_path, monkeypatch):
    registry = load_stage_ii_trajectory_source_registry_v1(
        Path(__file__).parents[1] / "config/stage_ii_cfm/trajectory_source_registry_v1.json")
    spec = next(item for item in registry.datasets if item.dataset_source == "egowalk")
    manifest, gt = tmp_path / "source.jsonl", tmp_path / "gt.jsonl"
    manifest.write_text("".join(json.dumps({"id": f"{clip}-{i}", "clip_id": clip,
                                          "timestamp": i * 500, "frame_idx": i}) + "\n"
                                 for clip in ("a", "b") for i in (1, 8, 9)))
    gt.write_text("")
    index = tmp_path / "index.sqlite"
    index_history_manifests(manifests=[manifest], sidecar_paths=[gt], spec=spec, output=index)
    monkeypatch.setattr(EgoWalkHistoryProcessor, "load_timeline", lambda self, source: (
        np.arange(10, dtype=np.int64) * 500_000_000, np.arange(10)))
    jobs = [{"clip_id": clip, "source": str(tmp_path / f"{clip}.parquet"),
             "video": str(tmp_path / f"{clip}.mp4")} for clip in ("a", "b")]
    for job in jobs:
        Path(job["video"]).touch()
    kwargs = dict(jobs=jobs, source_index=index, spec=spec, registry=registry,
                  max_lag_ns=0, max_bev_age_ns=None, fpv_bytes_per_frame=100)
    result = plan_history_corpus(output=tmp_path / "all", **kwargs)
    assert result["clips"] == 2 and result["source_rows"] == 6
    assert result["history_eligible"] == 4 and result["unique_fpv"] == 20
    assert result["rejection_reasons"] == {"insufficient_history": 2}
    assert result["capacity_estimate"]["media_bytes"] == 2000
    assert sum(result["history_eligible_by_split"].values()) == 4
    Path(jobs[1]["video"]).unlink()
    partial = tmp_path / "partial"
    with pytest.raises(FileNotFoundError):
        plan_history_corpus(output=partial, **kwargs)
    assert (partial / "clips/000000/summary.json").is_file()
    assert not (partial / "summary.json").exists()
