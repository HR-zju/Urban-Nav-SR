"""Indexed planning preserves direct-manifest identity and sample order."""

import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from mcp_server.data.history_observation import EgoWalkHistoryProcessor
from mcp_server.data.history_package import index_history_manifests, write_clip_plan
from mcp_server.data.stage_ii.sources import load_stage_ii_trajectory_source_registry_v1, stage_ii_assign_split_v1


def test_indexed_plan_matches_direct_plan(tmp_path, monkeypatch):
    registry = load_stage_ii_trajectory_source_registry_v1(
        Path(__file__).parents[1] / "config/stage_ii_cfm/trajectory_source_registry_v1.json")
    spec = next(item for item in registry.datasets if item.dataset_source == "egowalk")
    rows = [{"id": f"{clip}-{i}", "clip_id": clip, "timestamp": i * 500,
             "frame_idx": i, "response": "original"}
            for i in (1, 8, 9) for clip in ("a", "b")]
    manifest, sidecars = tmp_path / "source.jsonl", tmp_path / "sidecars.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    sidecars.write_text("".join(json.dumps({
        "sample_id": f"egowalk:{clip}-8", "split_group_id": f"egowalk:{clip}",
        "split": stage_ii_assign_split_v1(f"egowalk:{clip}", registry), "dataset_source": "egowalk",
        "trajectory_target": {"original": True},
    }) + "\n" for clip in ("a", "b")))
    index = tmp_path / "source.sqlite"
    summary = index_history_manifests(manifests=[manifest], sidecar_paths=[sidecars], spec=spec, output=index)
    assert summary["clips"] == [
        {"clip_id": clip, "source_rows": 3, "linked_sidecars": 1} for clip in ("a", "b")]
    processor = EgoWalkHistoryProcessor(0)
    monkeypatch.setattr(processor, "load_timeline", lambda source: (
        np.arange(11, dtype=np.int64) * 500_000_000, np.arange(11)))
    kwargs = dict(clip_id="a", source=tmp_path / "a.parquet", video=tmp_path / "a.mp4",
                  processor=processor, spec=spec, registry=registry, bev=None, max_samples=2)
    write_clip_plan(manifests=[manifest], sidecar_paths=[sidecars], output=tmp_path / "direct", **kwargs)
    # The completed index is self-contained: planning reads no original JSONL again.
    manifest.unlink()
    sidecars.unlink()
    write_clip_plan(manifests=[], sidecar_paths=[], source_index=index, output=tmp_path / "indexed", **kwargs)
    for name in ("observation_plan.jsonl", "rejected.jsonl"):
        assert (tmp_path / "direct" / name).read_bytes() == (tmp_path / "indexed" / name).read_bytes()
    with pytest.raises(FileExistsError):
        index_history_manifests(manifests=[], sidecar_paths=[], spec=spec, output=index)
    with sqlite3.connect(index) as db:
        db.execute("DELETE FROM completion")
    with pytest.raises(ValueError, match="incomplete"):
        write_clip_plan(manifests=[], sidecar_paths=[], source_index=index, output=tmp_path / "partial", **kwargs)


def test_annotation_variants_keep_one_observation_and_gt_source(tmp_path, monkeypatch):
    registry = load_stage_ii_trajectory_source_registry_v1(
        Path(__file__).parents[1] / "config/stage_ii_cfm/trajectory_source_registry_v1.json")
    spec = next(item for item in registry.datasets if item.dataset_source == "egowalk")
    main = {"id": "a-8", "clip_id": "a", "timestamp": 4000, "frame_idx": 8,
            "messages": [{"role": "user", "content": "same input"},
                         {"role": "assistant", "content": "main answer"}]}
    extra = {**main, "messages": [main["messages"][0], {"role": "assistant", "content": "extra answer"}]}
    main_path, extra_path, gt_path = (tmp_path / name for name in ("main.jsonl", "extra.jsonl", "gt.jsonl"))
    main_path.write_text(json.dumps(main) + "\n")
    extra_path.write_text(json.dumps(extra) + "\n")
    gt = {"sample_id": "egowalk:a-8", "dataset_source": "egowalk", "split_group_id": "egowalk:a",
          "split": stage_ii_assign_split_v1("egowalk:a", registry), "source_manifest_name": "egowalk/main.jsonl",
          "source_line": 1, "trajectory_target": {"original": True}}
    gt_path.write_text(json.dumps(gt) + "\n")
    index = tmp_path / "variants.sqlite"
    summary = index_history_manifests(manifests=[extra_path, main_path], sidecar_paths=[gt_path],
                                      spec=spec, output=index)
    assert (summary["source_rows"], summary["unique_parents"], summary["multiple_annotation_parents"]) == (2, 1, 1)
    processor = EgoWalkHistoryProcessor(0)
    monkeypatch.setattr(processor, "load_timeline", lambda source: (
        np.arange(9, dtype=np.int64) * 500_000_000, np.arange(9)))
    output = tmp_path / "plan"
    result = write_clip_plan(manifests=[], sidecar_paths=[], source_index=index, clip_id="a",
                            source=tmp_path / "a.parquet", video=tmp_path / "a.mp4", processor=processor,
                            spec=spec, registry=registry, bev=None, output=output, max_samples=None)
    records = [json.loads(line) for line in (output / "observation_plan.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["source_row"] == main
    assert records[0]["trajectory_sidecar"] == gt
    assert [v["source_row"] for v in records[0]["source_variants"]] == [extra, main]
    assert records[0]["semantic_label_selection"] == "pending"
    assert result["multiple_annotation_parents"] == 1
    extra["timestamp"] = 3999
    extra_path.write_text(json.dumps(extra) + "\n")
    with pytest.raises(ValueError, match="conflicting observation"):
        index_history_manifests(manifests=[main_path, extra_path], sidecar_paths=[gt_path],
                                spec=spec, output=tmp_path / "conflict.sqlite")
