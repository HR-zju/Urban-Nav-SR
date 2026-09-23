"""Test causal history and the relocated BEV contract at processor boundaries."""

import json

import numpy as np
import pytest

from mcp_server.data.history_observation import (
    FoxgloveHistoryProcessor, SampleRejected, SidewalkHistoryProcessor,
)


def test_causal_history_uses_actual_timestamps(tmp_path):
    """Selection retains nine unique predecessors and their actual time lag."""
    times = np.arange(61, dtype=np.int64) * 100_000_000
    times[20] += 1
    np.save(tmp_path / "frame_times.npy", times)
    result = SidewalkHistoryProcessor(100_000_000).select_history(tmp_path, 60)
    assert result["video_ordinals"].tolist() == [19, 25, 30, 35, 40, 45, 50, 55, 60]
    assert np.all(result["source_ns"] <= result["target_ns"])


def test_processor_relocates_and_checks_timestamp(tmp_path):
    """Clip-local resolution preserves stamps and surfaces source disagreement."""
    np.save(tmp_path / "frame_times.npy", np.array([123], dtype=np.int64))
    (tmp_path / "bev_images").mkdir()
    (tmp_path / "bev_images/000000_bev.jpg").write_bytes(b"image")
    index = {"clip_id": tmp_path.name, "entries": [
        {"frame_idx": 0, "timestamp": 123, "bev_image_path": "/old/000000_bev.jpg"}
    ]}
    path = tmp_path / "bev_index.json"
    path.write_text(json.dumps(index))
    processor = SidewalkHistoryProcessor(100_000_000)
    assert processor.load_bev_index(tmp_path)[0]["bev_image_path"] == "bev_images/000000_bev.jpg"
    index["entries"][0]["timestamp"] = 124
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="timestamp mismatch"):
        processor.load_bev_index(tmp_path)


def test_foxglove_uses_explicit_mapping(tmp_path):
    """A verified inserted-frame mapping changes ordinals without changing time."""
    times = np.arange(10, dtype=np.int64) * 500_000_000
    np.save(tmp_path / "frame_times.npy", times)
    processor = FoxgloveHistoryProcessor(0, np.arange(10) + 1)
    assert processor.select_history(tmp_path, 8)["video_ordinals"].tolist() == list(range(1, 10))


def test_early_anchor_has_explicit_rejection(tmp_path):
    """Early rows cannot be padded into a complete four-second history."""
    np.save(tmp_path / "frame_times.npy", np.arange(5, dtype=np.int64) * 500_000_000)
    with pytest.raises(SampleRejected, match="insufficient_history") as error:
        SidewalkHistoryProcessor(0).select_history(tmp_path, 4)
    assert error.value.reason == "insufficient_history"


def test_source_corruption_propagates_and_timeline_is_reused(tmp_path, monkeypatch):
    """Expected exclusions are distinct from broken source clocks and mappings."""
    processor = SidewalkHistoryProcessor(0)
    loads = []

    def load(source):
        loads.append(source)
        return np.arange(11) * 500_000_000, np.arange(11)

    monkeypatch.setattr(processor, "load_timeline", load)
    processor.select_history(tmp_path, 8)
    processor.select_history(tmp_path, 9)
    assert loads == [tmp_path]
    with pytest.raises(IndexError, match="anchor row -1"):
        processor.select_history(tmp_path, -1)

    np.save(tmp_path / "frame_times.npy", np.array([1, 1]))
    with pytest.raises(ValueError, match="strictly increase") as error:
        SidewalkHistoryProcessor(0).select_history(tmp_path, 1)
    assert not isinstance(error.value, SampleRejected)


def test_plan_preserves_parent_and_checks_source_anchor(tmp_path):
    """The observation contains source-clock history, while labels remain separate."""
    np.save(tmp_path / "frame_times.npy", np.arange(11, dtype=np.int64) * 500_000_000)
    row = {"sidewalk_meta": {"clip_id": tmp_path.name, "frame_idx": 9,
                             "timestamp": 4_500_000_000}, "label": "original"}
    processor = SidewalkHistoryProcessor(0)
    plan = processor.plan(row, tmp_path, source_parent_id="sidewalk:parent")
    assert plan["source_parent_id"] == "sidewalk:parent"
    assert plan["video_ordinals"] == list(range(1, 10))
    assert plan["observation_profile"] == "history_4s_2hz_9f_v1"
    assert "label" not in plan
    assert row["label"] == "original"
    row["sidewalk_meta"]["timestamp"] += 1
    with pytest.raises(ValueError, match="anchor timestamp/frame mismatch"):
        processor.plan(row, tmp_path, source_parent_id="sidewalk:parent")
