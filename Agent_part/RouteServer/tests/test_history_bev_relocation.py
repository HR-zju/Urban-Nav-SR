"""Check clip-local BEV identity and visible source-index errors."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "tools/dataset_processing/build_sidewalk_phase1_multiframe_archive.py"
SPEC = importlib.util.spec_from_file_location("history_archive", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_relocated_clip_wins_even_when_old_path_exists(tmp_path):
    """Moving a dataset must preserve the index's current clip ownership."""
    old = tmp_path / "old" / "000000_bev.jpg"
    old.parent.mkdir()
    old.write_bytes(b"old")
    clip = tmp_path / "clip"
    image = clip / "bev_images" / old.name
    image.parent.mkdir(parents=True)
    image.write_bytes(b"current")
    (clip / "bev_index.json").write_text(json.dumps({"entries": [
        {"bev_image_path": str(old), "frame_idx": 0, "timestamp": 123}
    ]}))
    entry, = MODULE.load_bev_index_entries(str(clip))
    assert Path(entry["path"]) == image
    assert entry["frame_idx"] == 0
    assert entry["timestamp"] == 123


def test_missing_local_image_raises(tmp_path):
    """A missing image remains a source error rather than a silent fallback."""
    (tmp_path / "bev_index.json").write_text(json.dumps({"entries": [
        {"bev_image_path": "/old/000000_bev.jpg", "frame_idx": 0, "timestamp": 123}
    ]}))
    with pytest.raises(FileNotFoundError, match="image missing"):
        MODULE.load_bev_index_entries(str(tmp_path))


def test_malformed_index_raises(tmp_path):
    """An invalid JSON index must be reported at its source."""
    (tmp_path / "bev_index.json").write_text("{")
    with pytest.raises(json.JSONDecodeError):
        MODULE.load_bev_index_entries(str(tmp_path))
