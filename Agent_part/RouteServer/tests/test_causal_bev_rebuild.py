"""Verify future independence and explicit cache requirements for input BEV."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location("causal_bev", TOOLS / "dataset_processing/rebuild_causal_bev.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
sys.path.insert(0, str(TOOLS / "bev"))
import frodobot_bev_enhanced as renderer


def test_future_gps_changes_preserve_prefix_pose():
    """Changing the future must preserve every prior map center and bearing."""
    gps = np.array([[30., 120.], [30., 120.00001], [30., 120.00002], [30., 120.00003]])
    changed = gps.copy()
    changed[3] = [40., 130.]
    assert MODULE.causal_gps_poses(gps)[:3] == MODULE.causal_gps_poses(changed)[:3]
    assert MODULE.causal_gps_poses(gps)[1][2] == pytest.approx(90.)


def test_stationary_prefix_has_no_invented_heading():
    """A stationary prefix stays in warmup until motion establishes heading."""
    assert MODULE.causal_gps_poses(np.array([[30., 120.], [30., 120.]])) == [None, None]


def test_cached_tile_absence_is_visible(tmp_path):
    """Strict offline input construction must report missing map tiles."""
    with pytest.raises(FileNotFoundError):
        renderer._fetch_tile_image(x=0, y=0, zoom=19, provider_config=renderer.MAP_PROVIDERS["osm"],
                                  cache_dir=tmp_path, require_cached=True)
    assert list(tmp_path.iterdir()) == []


def test_rebuild_uses_causal_pose_and_separate_output(tmp_path, monkeypatch):
    """Verify render arguments and index output; the fixture is not real map evidence."""
    from PIL import Image

    clip = tmp_path / "clip"
    clip.mkdir()
    np.save(clip / "frame_times.npy", np.array([0, 500000000, 1000000000]))
    np.save(clip / "frame_gps.npy", np.array([[30., 120.], [30., 120.00001], [30., 120.00002]]))
    calls = []

    def render(**kwargs):
        calls.append(kwargs)
        return Image.new("RGB", (32, 32), "white")

    monkeypatch.setattr(renderer, "render_bev_map", render)
    output = tmp_path / "rebuilt"
    index = MODULE.rebuild_clip(clip, output, tmp_path / "tiles", [0, 1])
    assert index["rejected"] == [{"frame_idx": 0, "reason": "heading_warming_up"}]
    assert len(index["entries"]) == 1
    assert (output / index["entries"][0]["bev_image_path"]).exists()
    assert calls[0]["navigation_path_gps"] is None
    assert calls[0]["options"].require_cached_tiles
    assert not calls[0]["options"].draw_trajectory
    assert not (clip / "bev_images").exists()


def test_legacy_blank_cache_is_rejected(tmp_path):
    """Old offline fallback images must not qualify as verified map tiles."""
    from PIL import Image
    Image.new("RGB", (256, 256), (242, 239, 233)).save(tmp_path / "OpenStreetMap_19_0_0.png")
    with pytest.raises(ValueError, match="placeholder"):
        renderer._fetch_tile_image(x=0, y=0, zoom=19, provider_config=renderer.MAP_PROVIDERS["osm"],
                                  cache_dir=tmp_path, require_cached=True)


def test_offline_fallback_keeps_cache_empty(tmp_path, monkeypatch):
    """Debug offline rendering may return a placeholder but must not cache it."""
    monkeypatch.setenv("BEV_MAP_TILE_OFFLINE", "1")
    renderer._fetch_tile_image(x=0, y=0, zoom=19, provider_config=renderer.MAP_PROVIDERS["osm"],
                              cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []
