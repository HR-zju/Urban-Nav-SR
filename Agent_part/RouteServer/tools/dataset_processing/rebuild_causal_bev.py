"""Rebuild map-only, source-causal BEV into a separate dataset directory."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def causal_gps_poses(gps: np.ndarray, min_displacement_m: float = 0.5) -> list:
    """Estimate source-causal map centers and motion bearings.

    Args:
        gps: Source-order latitude/longitude array, shape [N, >=2], degrees.
        min_displacement_m: Past displacement required to update motion bearing.

    Returns:
        List of (latitude, longitude, compass_heading) or None before the first
        usable displacement. A stationary prefix has no inferred orientation.

    Raises:
        ValueError: If coordinates or displacement threshold are invalid.
    """
    coordinates = np.asarray(gps, dtype=np.float64)[:, :2]
    if not np.isfinite(coordinates).all() or min_displacement_m <= 0:
        raise ValueError("GPS coordinates must be finite and displacement positive")
    reference = coordinates[0]
    heading = None
    poses = []
    for lat, lng in coordinates:
        # 仅使用当前与过去 GPS：局部 EN 位移，罗盘角 atan2(E,N)。静止时保持已知航向。
        north = math.radians(lat - reference[0]) * 6371000.0
        east = math.radians(lng - reference[1]) * 6371000.0 * math.cos(
            math.radians((lat + reference[0]) / 2)
        )
        if math.hypot(east, north) >= min_displacement_m:
            heading = math.degrees(math.atan2(east, north)) % 360
            reference = np.array([lat, lng])
        poses.append(None if heading is None else (float(lat), float(lng), heading))
    return poses


def rebuild_clip(clip: Path, output: Path, cache: Path, frame_indices: list[int]) -> dict:
    """Render selected source frames using cached map tiles and past/current GPS.

    Args:
        clip: Original NPY clip directory.
        output: New, separate per-clip output directory.
        cache: Existing OSM tile cache compatible with the shared renderer.
        frame_indices: Source row indices requested by the sampling plan.

    Returns:
        BEV index with relative image paths and explicit pose/route provenance.

    Raises:
        ValueError: If timestamps or requested frame indices are invalid.
        FileExistsError: If the destination already exists.
        FileNotFoundError: If input files or required cached tiles are missing.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bev"))
    from frodobot_bev_enhanced import MAP_PROVIDERS, ProcessOptions, render_bev_map

    times = np.load(clip / "frame_times.npy").astype(np.int64)
    gps = np.load(clip / "frame_gps.npy")
    if len(times) != len(gps) or np.any(np.diff(times) <= 0):
        raise ValueError(f"{clip}: GPS/time rows must match and time must increase")
    if any(i < 0 or i >= len(times) for i in frame_indices):
        raise ValueError(f"{clip}: requested frame index outside source timeline")
    poses = causal_gps_poses(gps)
    output.mkdir(parents=True, exist_ok=False)
    media = output / "bev_images"
    media.mkdir()
    options = ProcessOptions(output_size=1024, zoom=19, cache_dir=cache,
                             draw_trajectory=False, require_cached_tiles=True,
                             bev_alignment="ego_aligned")
    entries, rejected = [], []
    for i in sorted(set(frame_indices)):
        pose = poses[i]
        if pose is None:
            rejected.append({"frame_idx": i, "reason": "heading_warming_up"})
            continue
        lat, lng, heading = pose
        # 未来记录路径保留在监督侧；输入地图只渲染当前位置和由过去位移估计的朝向。
        image = render_bev_map(lat=lat, lng=lng, heading_deg=heading,
                               navigation_path_gps=None, options=options,
                               provider_config=MAP_PROVIDERS["osm"])
        filename = f"{i:06d}_bev.jpg"
        image.save(media / filename, quality=90)
        entry = {"frame_idx": i, "frame_name": f"{i:06d}", "timestamp": int(times[i]),
                 "bev_image_path": f"bev_images/{filename}", "draw_trajectory": False,
                 "position_source": "current_raw_gps",
                 "attitude_source": "past_gps_displacement_compass",
                 "used_heading_deg": heading, "bev_alignment": "ego_aligned"}
        (media / (filename + ".bevmeta.json")).write_text(json.dumps(entry), encoding="utf-8")
        entries.append(entry)
    index = {"schema_version": "sidewalk_bev_index_v1", "clip_id": clip.name,
             "input_profile": "causal_map_v1", "bev_route_scope": "none",
             "causality_scope": "prepared_frame_gps_rows",
             "sensor_sync_causality": "unverified",
             "draw_trajectory": False, "entries": entries, "rejected": rejected}
    (output / "bev_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


def main() -> None:
    """Parse one-clip reconstruction arguments and print its result counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-cache", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    args = parser.parse_args()
    result = rebuild_clip(args.clip, args.output, args.tile_cache, args.frames)
    print(json.dumps({"rendered": len(result["entries"]), "rejected": result["rejected"]}))


if __name__ == "__main__":
    main()
