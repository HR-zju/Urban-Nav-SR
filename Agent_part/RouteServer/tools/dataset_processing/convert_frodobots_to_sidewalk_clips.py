#!/usr/bin/env python3
"""Convert Berkeley FrodoBots-7K zarr episodes into SideWalk prepared clips.

The Phase1 SideWalk pipeline expects one clip directory per ride with dense
per-frame pose arrays and a ``front.mp4``.  Berkeley-FrodoBots-7K stores the
same information in a HuggingFace/LeRobot-style zarr cache plus shared video
files.  This script materializes the minimal prepared clip contract consumed by
``process_sidewalk_dataset.py`` and ``sidewalk_phase1_dataset.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_ZARR_KEYS = (
    "timestamp",
    "episode_index",
    "frame_index",
    "observation.images.front.path",
    "observation.images.front.timestamp",
    "observation.latitude",
    "observation.longitude",
    "observation.utm_position",
    "observation.filtered_heading",
)


class MinimalZarrArray:
    """Small zarr v2 reader for row slices in this FrodoBots cache.

    This intentionally covers only the layout used here: directory store,
    one-dimensional or two-dimensional arrays, Blosc-compressed chunks, C order.
    It avoids depending on the full ``zarr`` package, whose legacy dependency
    chain can be awkward in older conda/pip environments.
    """

    def __init__(self, path: Path):
        self.path = path
        self.meta = json.loads((path / ".zarray").read_text(encoding="utf-8"))
        self.shape = tuple(int(v) for v in self.meta["shape"])
        self.chunks = tuple(int(v) for v in self.meta["chunks"])
        self.dtype = np.dtype(self.meta["dtype"])
        self.fill_value = self.meta.get("fill_value")
        compressor = self.meta.get("compressor")
        if compressor:
            try:
                from numcodecs import get_codec  # type: ignore
            except Exception as exc:  # pragma: no cover - local dependency issue.
                raise RuntimeError(
                    "Missing dependency 'numcodecs'. Install conversion deps first, e.g. "
                    "`pip install numcodecs opencv-python`."
                ) from exc
            self.codec = get_codec(compressor)
        else:
            self.codec = None

    def __getitem__(self, item: Any) -> np.ndarray:
        if isinstance(item, slice):
            start, stop, step = item.indices(self.shape[0])
            if step != 1:
                return self[item.start : item.stop][::step]
            return self._read_row_slice(start, stop)
        if isinstance(item, int):
            if item < 0:
                item += self.shape[0]
            return self._read_row_slice(item, item + 1)[0]
        raise TypeError(f"Unsupported zarr index for {self.path}: {item!r}")

    def _read_row_slice(self, start: int, stop: int) -> np.ndarray:
        start = max(0, int(start))
        stop = min(int(stop), self.shape[0])
        if stop <= start:
            return np.empty((0,) + self.shape[1:], dtype=self.dtype)

        row_chunk = self.chunks[0]
        pieces: list[np.ndarray] = []
        chunk_idx = start // row_chunk
        while chunk_idx * row_chunk < stop:
            chunk_start = chunk_idx * row_chunk
            chunk_stop = min(chunk_start + row_chunk, self.shape[0])
            local_start = max(start, chunk_start) - chunk_start
            local_stop = min(stop, chunk_stop) - chunk_start
            chunk = self._read_row_chunk(chunk_idx)
            pieces.append(chunk[local_start:local_stop])
            chunk_idx += 1
        if len(pieces) == 1:
            return pieces[0]
        return np.concatenate(pieces, axis=0)

    def _read_row_chunk(self, row_chunk_idx: int) -> np.ndarray:
        if len(self.shape) == 1:
            return self._read_chunk((int(row_chunk_idx),))
        elif len(self.shape) == 2:
            col_chunks = int(math.ceil(float(self.shape[1]) / float(self.chunks[1])))
            pieces = [self._read_chunk((int(row_chunk_idx), int(col_idx))) for col_idx in range(col_chunks)]
            if len(pieces) == 1:
                return pieces[0]
            return np.concatenate(pieces, axis=1)
        raise RuntimeError(f"Unsupported zarr array rank {len(self.shape)} for {self.path}")

    def _read_chunk(self, chunk_indices: tuple[int, ...]) -> np.ndarray:
        key = ".".join(str(int(idx)) for idx in chunk_indices)
        chunk_shape_items = []
        for axis, chunk_idx in enumerate(chunk_indices):
            axis_start = int(chunk_idx) * self.chunks[axis]
            chunk_shape_items.append(min(self.chunks[axis], self.shape[axis] - axis_start))
        chunk_shape = tuple(chunk_shape_items)
        if len(chunk_shape) != len(self.shape):
            raise RuntimeError(f"Invalid chunk index {chunk_indices} for {self.path}")

        chunk_path = self.path / key
        if not chunk_path.is_file():
            fill = self.fill_value if self.fill_value is not None else 0
            return np.full(chunk_shape, fill, dtype=self.dtype)
        raw = chunk_path.read_bytes()
        decoded = self.codec.decode(raw) if self.codec is not None else raw
        arr = np.frombuffer(decoded, dtype=self.dtype)
        full_chunk_shape = self.chunks[: len(self.shape)]
        arr = arr.reshape(full_chunk_shape, order=str(self.meta.get("order") or "C"))
        slices = tuple(slice(0, size) for size in chunk_shape)
        return arr[slices].copy()


class MinimalZarrGroup:
    def __init__(self, path: Path):
        self.path = path

    def __contains__(self, key: str) -> bool:
        return (self.path / key / ".zarray").is_file()

    def __getitem__(self, key: str) -> MinimalZarrArray:
        array_path = self.path / key
        if not (array_path / ".zarray").is_file():
            raise KeyError(key)
        return MinimalZarrArray(array_path)


def _safe_name(value: str) -> str:
    keep = []
    for ch in str(value):
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "clip"


def _video_stem_to_sequence(path_or_name: str) -> str:
    name = Path(str(path_or_name)).name
    for suffix in ("_front_camera.mp4", "_rear_camera.mp4"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _open_zarr_group(dataset_root: Path):
    zarr_root = dataset_root / "dataset_cache.zarr"
    if not zarr_root.is_dir():
        raise FileNotFoundError(f"Missing zarr cache: {zarr_root}")
    try:
        import zarr  # type: ignore

        group = zarr.open_group(str(zarr_root), mode="r")
    except Exception:
        group = MinimalZarrGroup(zarr_root)
    missing = [key for key in REQUIRED_ZARR_KEYS if key not in group]
    if missing:
        raise RuntimeError(f"FrodoBots zarr cache missing required arrays: {missing}")
    return group


def _load_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta_data" / "info.json"
    if not info_path.is_file():
        return {}
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _scan_sequence_span(
    front_paths: Any,
    *,
    sequence_name: str,
    chunk_rows: int | None = None,
    max_scan_rows: int | None = None,
) -> tuple[int, int, str]:
    """Find the contiguous row span whose front video path matches sequence."""

    target = _video_stem_to_sequence(sequence_name)
    n = int(front_paths.shape[0])
    chunk = int(chunk_rows or getattr(front_paths, "chunks", (50000,))[0] or 50000)
    chunk = max(1, chunk)
    limit = min(n, int(max_scan_rows)) if max_scan_rows else n
    start: int | None = None
    end: int | None = None
    matched_path = ""

    offset = 0
    while offset < limit:
        stop = min(offset + chunk, limit)
        values = front_paths[offset:stop]
        matches = []
        for local_idx, raw in enumerate(values):
            text = str(raw)
            if _video_stem_to_sequence(text) == target or target in text:
                matches.append(local_idx)
        if matches:
            if start is None:
                start = offset + matches[0]
                matched_path = str(values[matches[0]])
            end = offset + matches[-1] + 1
        elif start is not None:
            break
        offset = stop

    if start is None or end is None:
        raise RuntimeError(f"Sequence not found in observation.images.front.path: {sequence_name}")

    return start, end, matched_path


def _scan_sequence_runs(
    front_paths: Any,
    episode_index: Any,
    *,
    chunk_rows: int | None = None,
    max_scan_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Return contiguous sequence spans from the zarr cache."""

    n = int(front_paths.shape[0])
    chunk = int(chunk_rows or getattr(front_paths, "chunks", (50000,))[0] or 50000)
    chunk = max(1, chunk)
    limit = min(n, int(max_scan_rows)) if max_scan_rows else n
    runs: list[dict[str, Any]] = []
    current_path: str | None = None
    current_episode: int | None = None
    current_start: int | None = None
    offset = 0
    while offset < limit:
        stop = min(offset + chunk, limit)
        paths_chunk = np.asarray(front_paths[offset:stop])
        episodes_chunk = np.asarray(episode_index[offset:stop], dtype=np.int64)
        if len(paths_chunk) == 0:
            break

        local_run_starts = [0]
        if current_path is not None:
            if str(paths_chunk[0]) != current_path or int(episodes_chunk[0]) != int(current_episode):
                runs.append(
                    {
                        "sequence_name": _video_stem_to_sequence(str(current_path)),
                        "matched_path": str(current_path),
                        "episode_index": int(current_episode),
                        "start": int(current_start),
                        "end": int(offset),
                        "frame_count": int(offset - int(current_start)),
                    }
                )
                current_path = None
                current_episode = None
                current_start = None
        if len(paths_chunk) > 1:
            path_changes = paths_chunk[1:] != paths_chunk[:-1]
            episode_changes = episodes_chunk[1:] != episodes_chunk[:-1]
            changes = np.nonzero(path_changes | episode_changes)[0] + 1
            local_run_starts.extend(int(v) for v in changes.tolist())

        if current_path is None:
            current_path = str(paths_chunk[0])
            current_episode = int(episodes_chunk[0])
            current_start = int(offset)

        for local_start in local_run_starts[1:]:
            row_idx = int(offset + local_start)
            runs.append(
                {
                    "sequence_name": _video_stem_to_sequence(str(current_path)),
                    "matched_path": str(current_path),
                    "episode_index": int(current_episode),
                    "start": int(current_start),
                    "end": row_idx,
                    "frame_count": int(row_idx - int(current_start)),
                }
            )
            current_path = str(paths_chunk[local_start])
            current_episode = int(episodes_chunk[local_start])
            current_start = row_idx
        offset = stop
    if current_path is not None and current_episode is not None and current_start is not None:
        runs.append(
            {
                "sequence_name": _video_stem_to_sequence(str(current_path)),
                "matched_path": str(current_path),
                "episode_index": int(current_episode),
                "start": int(current_start),
                "end": int(limit),
                "frame_count": int(limit - int(current_start)),
            }
        )
    return runs


def _select_longest_percent(runs: list[dict[str, Any]], percent: float) -> list[dict[str, Any]]:
    if not runs:
        return []
    pct = max(0.0, min(float(percent), 100.0))
    if pct <= 0.0:
        return []
    count = max(1, int(math.ceil(len(runs) * pct / 100.0)))
    return sorted(runs, key=lambda item: int(item["frame_count"]), reverse=True)[:count]


def _bearing_enu_yaw_from_xy(xy: np.ndarray) -> np.ndarray:
    """Return ENU yaw radians from adjacent XY displacements."""

    arr = np.asarray(xy, dtype=np.float64)
    out = np.full((arr.shape[0],), np.nan, dtype=np.float64)
    if arr.shape[0] < 2:
        return out
    deltas = np.diff(arr[:, :2], axis=0)
    step_dist = np.linalg.norm(deltas, axis=1)
    valid = step_dist > 0.05
    yaw = np.full((arr.shape[0] - 1,), np.nan, dtype=np.float64)
    yaw[valid] = np.arctan2(deltas[valid, 1], deltas[valid, 0])
    out[:-1] = yaw
    out[-1] = out[-2] if len(out) > 1 else np.nan
    return _fill_angle_nans(out)


def _fill_angle_nans(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)
    first = int(np.argmax(finite))
    arr[:first] = arr[first]
    last = arr[first]
    for idx in range(first + 1, len(arr)):
        if np.isfinite(arr[idx]):
            last = arr[idx]
        else:
            arr[idx] = last
    next_value = arr[-1]
    for idx in range(len(arr) - 1, -1, -1):
        if np.isfinite(values[idx]):
            next_value = arr[idx]
        else:
            arr[idx] = next_value
    return arr


def _wrap_pi(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) + math.pi) % (2.0 * math.pi) - math.pi


def _circular_mae(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.abs(_wrap_pi(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("inf")
    return float(np.mean(diff))


def _resolve_heading_to_enu_yaw(
    raw_heading: np.ndarray,
    xy: np.ndarray,
    *,
    mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(raw_heading, dtype=np.float64).reshape(-1)
    if raw.size == 0:
        return np.zeros((0,), dtype=np.float64), {"mode": "empty", "unit": None}

    gps_yaw = _bearing_enu_yaw_from_xy(xy)
    finite_abs_max = float(np.nanmax(np.abs(raw[np.isfinite(raw)]))) if np.isfinite(raw).any() else 0.0
    raw_rad = raw if finite_abs_max <= (2.0 * math.pi + 0.5) else np.deg2rad(raw)
    raw_compass_to_enu = np.deg2rad(90.0) - raw_rad

    candidates = {
        "enu_yaw": raw_rad,
        "compass_heading": raw_compass_to_enu,
    }
    if mode in candidates:
        chosen = mode
    elif mode != "auto":
        raise ValueError(f"Unsupported --heading-mode: {mode}")
    else:
        scores = {name: _circular_mae(values, gps_yaw) for name, values in candidates.items()}
        chosen = min(scores, key=scores.get)

    scores = {name: _circular_mae(values, gps_yaw) for name, values in candidates.items()}
    yaw = _fill_angle_nans(candidates[chosen])
    return yaw, {
        "mode": chosen,
        "unit": "radians" if finite_abs_max <= (2.0 * math.pi + 0.5) else "degrees",
        "raw_min": float(np.nanmin(raw)) if np.isfinite(raw).any() else None,
        "raw_max": float(np.nanmax(raw)) if np.isfinite(raw).any() else None,
        "mae_vs_utm_bearing_rad": scores,
    }


def _xyzw_from_enu_yaw(yaw_rad: np.ndarray) -> np.ndarray:
    yaw = np.asarray(yaw_rad, dtype=np.float64).reshape(-1)
    half = yaw * 0.5
    q = np.zeros((yaw.shape[0], 4), dtype=np.float64)
    q[:, 2] = np.sin(half)
    q[:, 3] = np.cos(half)
    return q


def _estimate_speeds(xy: np.ndarray, times_ns: np.ndarray, fps: float) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float64)
    speeds = np.zeros((arr.shape[0],), dtype=np.float32)
    if arr.shape[0] < 2:
        return speeds
    diffs = np.linalg.norm(np.diff(arr[:, :2], axis=0), axis=1)
    dt = np.diff(np.asarray(times_ns, dtype=np.float64)) / 1e9
    fallback_dt = 1.0 / max(float(fps), 1e-6)
    dt[dt <= 1e-6] = fallback_dt
    step_speeds = diffs / dt
    speeds[1:] = step_speeds.astype(np.float32)
    speeds[0] = speeds[1] if len(speeds) > 1 else 0.0
    return speeds


def _timestamp_seconds_to_ns(timestamp: np.ndarray, fps: float) -> np.ndarray:
    arr = np.asarray(timestamp, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if np.nanmax(np.abs(arr)) > 1e12:
        return arr.astype(np.int64)
    if np.nanmax(np.abs(arr)) > 1e6:
        return (arr * 1000.0).astype(np.int64)
    # FrodoBots timestamp is usually episode-relative seconds.
    base = arr[0]
    ns = np.rint((arr - base) * 1e9).astype(np.int64)
    if len(ns) > 1 and np.any(np.diff(ns) <= 0):
        ns = np.rint(np.arange(len(ns), dtype=np.float64) * (1e9 / max(float(fps), 1e-6))).astype(np.int64)
    return ns


def _ffprobe_video(video_path: Path) -> dict[str, Any]:
    if not video_path.is_file():
        return {}
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_frames,r_frame_rate,duration",
                "-of",
                "json",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        data = json.loads(proc.stdout or "{}")
        streams = data.get("streams") or []
        return streams[0] if streams else {}
    except Exception:
        return {}


def _parse_frame_rate(value: Any, default: float) -> float:
    text = str(value or "").strip()
    if "/" in text:
        num, den = text.split("/", 1)
        try:
            den_f = float(den)
            return float(num) / den_f if den_f else float(default)
        except Exception:
            return float(default)
    try:
        parsed = float(text)
        return parsed if parsed > 0 else float(default)
    except Exception:
        return float(default)


def _extract_video_images(
    video_path: Path,
    output_dir: Path,
    *,
    video_frame_indices: np.ndarray,
    row_indices: np.ndarray,
    jpeg_quality: int,
) -> int:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv-python is required for --extract-images") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video for frame extraction: {video_path}")
    written = 0
    try:
        for row_idx in row_indices.astype(np.int64).tolist():
            output_path = output_dir / f"{int(row_idx):06d}.jpg"
            if output_path.is_file():
                continue
            video_idx = int(video_frame_indices[int(row_idx)])
            capture.set(cv2.CAP_PROP_POS_FRAMES, video_idx)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
            if cv2.imwrite(str(output_path), frame, params):
                written += 1
    finally:
        capture.release()
    return written


def _write_video_link(src: Path, dest: Path, mode: str) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        return "exists"
    if mode == "copy":
        import shutil

        shutil.copy2(src, dest)
        return "copy"
    if mode == "symlink":
        dest.symlink_to(src)
        return "symlink"
    try:
        os.link(src, dest)
        return "hardlink"
    except OSError:
        dest.symlink_to(src)
        return "symlink_fallback"


def _write_camera_params(path: Path, *, width: int, height: int, hfov_deg: float, camera_height_m: float) -> dict[str, Any]:
    fx = (float(width) * 0.5) / math.tan(math.radians(float(hfov_deg)) * 0.5)
    fy = fx
    payload = {
        "schema_version": "frodobots_approx_camera_params_v1",
        "image_width": int(width),
        "image_height": int(height),
        "camera_height_meters": float(camera_height_m),
        "K": [[fx, 0.0, float(width) * 0.5], [0.0, fy, float(height) * 0.5], [0.0, 0.0, 1.0]],
        "source": "approximate_from_video_size_and_hfov",
        "hfov_deg": float(hfov_deg),
        "note": "Replace with calibrated FrodoBots intrinsics before using calibrated labels as final supervision.",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def convert_one_sequence(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    group = _open_zarr_group(dataset_root)
    info = _load_info(dataset_root)
    fps = float(args.fps or info.get("fps") or 10.0)

    sequence = _video_stem_to_sequence(args.sequence_name)
    if getattr(args, "sequence_start", None) is not None and getattr(args, "sequence_end", None) is not None:
        start = int(args.sequence_start)
        end = int(args.sequence_end)
        matched_path = str(getattr(args, "matched_path", "") or f"videos/{sequence}_front_camera.mp4")
    else:
        start, end, matched_path = _scan_sequence_span(
            group["observation.images.front.path"],
            sequence_name=sequence,
            chunk_rows=args.scan_chunk_rows,
            max_scan_rows=args.max_scan_rows,
        )
    if args.max_frames and int(args.max_frames) > 0:
        end = min(end, start + int(args.max_frames))
    if end <= start:
        raise RuntimeError(f"Empty sequence span for {sequence}: {start}:{end}")

    clip_id = _safe_name(str(getattr(args, "output_clip_id", "") or sequence))
    clip_dir = output_root / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)

    video_name = Path(matched_path).name or f"{sequence}_front_camera.mp4"
    if not video_name.endswith(".mp4"):
        video_name = f"{sequence}_front_camera.mp4"
    video_path = dataset_root / "videos" / video_name
    if not video_path.is_file():
        candidate = dataset_root / "videos" / f"{sequence}_front_camera.mp4"
        if candidate.is_file():
            video_path = candidate
        else:
            raise FileNotFoundError(f"Cannot resolve front video for {sequence}: {video_path}")

    row_slice = slice(start, end)
    lat = np.asarray(group["observation.latitude"][row_slice], dtype=np.float64)
    lon = np.asarray(group["observation.longitude"][row_slice], dtype=np.float64)
    gps = np.stack([lat, lon], axis=1).astype(np.float64)
    xy = np.asarray(group["observation.utm_position"][row_slice], dtype=np.float64)
    raw_heading = np.asarray(group["observation.filtered_heading"][row_slice], dtype=np.float64)
    timestamp = np.asarray(group["timestamp"][row_slice], dtype=np.float64)
    front_timestamp = np.asarray(group["observation.images.front.timestamp"][row_slice], dtype=np.float64)
    frame_index = np.asarray(group["frame_index"][row_slice], dtype=np.int64)
    episode_index = np.asarray(group["episode_index"][row_slice], dtype=np.int64)

    valid = np.isfinite(gps).all(axis=1) & np.isfinite(xy).all(axis=1)
    valid &= (np.abs(gps[:, 0]) <= 90.0) & (np.abs(gps[:, 1]) <= 180.0)
    valid &= np.isfinite(raw_heading) & np.isfinite(front_timestamp)
    if not valid.all():
        gps = gps[valid]
        xy = xy[valid]
        raw_heading = raw_heading[valid]
        timestamp = timestamp[valid]
        front_timestamp = front_timestamp[valid]
        frame_index = frame_index[valid]
        episode_index = episode_index[valid]
    if len(gps) < int(args.min_frames):
        raise RuntimeError(f"Too few valid frames for {sequence}: {len(gps)} < {args.min_frames}")

    times_ns = _timestamp_seconds_to_ns(timestamp, fps=fps)
    yaw, heading_report = _resolve_heading_to_enu_yaw(raw_heading, xy, mode=args.heading_mode)
    orientations_xyzw = _xyzw_from_enu_yaw(yaw)
    speeds = _estimate_speeds(xy, times_ns, fps=fps)

    link_mode = _write_video_link(video_path, clip_dir / "front.mp4", args.media_mode)
    video_meta = _ffprobe_video(video_path)
    width = int(video_meta.get("width") or args.default_width)
    height = int(video_meta.get("height") or args.default_height)
    video_fps = _parse_frame_rate(video_meta.get("r_frame_rate"), default=fps)
    video_frame_indices = np.rint(front_timestamp * float(video_fps)).astype(np.int64)
    if str(video_meta.get("nb_frames") or "").isdigit():
        max_video_frame = max(int(video_meta["nb_frames"]) - 1, 0)
        video_frame_indices = np.clip(video_frame_indices, 0, max_video_frame)

    np.save(clip_dir / "frame_gps.npy", gps.astype(np.float64))
    np.save(clip_dir / "frame_positions.npy", xy.astype(np.float64))
    np.save(clip_dir / "frame_orientations.npy", orientations_xyzw.astype(np.float64))
    np.save(clip_dir / "frame_times.npy", times_ns.astype(np.int64))
    np.save(clip_dir / "frame_speeds.npy", speeds.astype(np.float32))
    np.save(clip_dir / "frame_video_indices.npy", video_frame_indices.astype(np.int64))

    extracted_images = 0
    image_stride = max(int(args.image_stride), 1)
    if args.extract_images:
        row_indices = np.arange(0, len(video_frame_indices), image_stride, dtype=np.int64)
        extracted_images = _extract_video_images(
            video_path,
            clip_dir / "images",
            video_frame_indices=video_frame_indices,
            row_indices=row_indices,
            jpeg_quality=int(args.jpeg_quality),
        )

    camera_payload = _write_camera_params(
        clip_dir / "camera_params.json",
        width=width,
        height=height,
        hfov_deg=float(args.hfov_deg),
        camera_height_m=float(args.camera_height_m),
    )

    meta = {
        "schema_version": "frodobots_sidewalk_prepared_clip_v1",
        "source_dataset": "Berkeley-FrodoBots-7K",
        "source_dataset_root": str(dataset_root),
        "sequence_name": sequence,
        "zarr_span": {"start": int(start), "end": int(end), "valid_frames": int(len(gps))},
        "episode_index_unique": sorted({int(v) for v in episode_index.tolist()})[:20],
        "front_video_source": str(video_path),
        "front_video_link_mode": link_mode,
        "front_video_fps": float(video_fps),
        "front_video_timestamp_range_s": [float(front_timestamp[0]), float(front_timestamp[-1])],
        "frame_orientations_order": "xyzw",
        "heading_conversion": heading_report,
        "fps": float(fps),
        "camera_params_source": camera_payload["source"],
        "image_extraction": {
            "enabled": bool(args.extract_images),
            "image_stride": int(image_stride),
            "written": int(extracted_images),
            "video_frame_index_range": [int(video_frame_indices[0]), int(video_frame_indices[-1])],
        },
    }
    (clip_dir / "mcap_extract_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        **meta,
        "clip_dir": str(clip_dir),
        "frame_count": int(len(gps)),
        "frame_index_range": [int(frame_index[0]), int(frame_index[-1])],
        "time_ns_range": [int(times_ns[0]), int(times_ns[-1])],
        "duration_s": float((int(times_ns[-1]) - int(times_ns[0])) / 1e9) if len(times_ns) > 1 else 0.0,
        "gps_bounds": {
            "lat_min": float(np.min(gps[:, 0])),
            "lat_max": float(np.max(gps[:, 0])),
            "lon_min": float(np.min(gps[:, 1])),
            "lon_max": float(np.max(gps[:, 1])),
        },
        "xy_bounds": {
            "x_min": float(np.min(xy[:, 0])),
            "x_max": float(np.max(xy[:, 0])),
            "y_min": float(np.min(xy[:, 1])),
            "y_max": float(np.max(xy[:, 1])),
        },
        "speed_mps": {
            "mean": float(np.mean(speeds)),
            "p95": float(np.percentile(speeds, 95)),
            "max": float(np.max(speeds)),
        },
        "video_meta": video_meta,
        "outputs": [
            "front.mp4",
            "frame_gps.npy",
            "frame_positions.npy",
            "frame_orientations.npy",
            "frame_times.npy",
            "frame_speeds.npy",
            "frame_video_indices.npy",
            "camera_params.json",
            "mcap_extract_meta.json",
        ],
    }
    (clip_dir / "conversion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def convert_batch(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    group = _open_zarr_group(dataset_root)

    started = time.time()
    print("[frodobots-convert] scanning sequence spans...")
    runs = _scan_sequence_runs(
        group["observation.images.front.path"],
        group["episode_index"],
        chunk_rows=args.scan_chunk_rows,
        max_scan_rows=args.max_scan_rows,
    )
    selected = _select_longest_percent(runs, float(args.top_percent))
    selected = [run for run in selected if int(run["frame_count"]) >= int(args.min_frames)]
    if args.max_sequences is not None and int(args.max_sequences) > 0:
        selected = selected[: int(args.max_sequences)]

    summary: dict[str, Any] = {
        "schema_version": "frodobots_batch_selection_v1",
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "top_percent": float(args.top_percent),
        "max_scan_rows": args.max_scan_rows,
        "runs_total": len(runs),
        "selected_count": len(selected),
        "selected_total_frames": int(sum(int(item["frame_count"]) for item in selected)),
        "selected_min_frames": int(min((int(item["frame_count"]) for item in selected), default=0)),
        "selected_max_frames": int(max((int(item["frame_count"]) for item in selected), default=0)),
        "extract_images": bool(args.extract_images),
        "image_stride": int(args.image_stride),
        "media_mode": str(args.media_mode),
        "dry_run": bool(args.dry_run),
        "selected": selected,
        "converted": [],
        "errors": [],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.batch_report_json).expanduser() if args.batch_report_json else output_root / f"batch_selection_top{float(args.top_percent):g}.json"

    if args.dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k not in {"selected", "converted", "errors"}}, ensure_ascii=False, indent=2))
        print(f"[frodobots-convert] dry-run report: {report_path}")
        return summary

    for idx, run in enumerate(selected, start=1):
        seq = str(run["sequence_name"])
        clip_dir = output_root / _safe_name(seq)
        if args.skip_existing and (clip_dir / "conversion_report.json").is_file():
            print(f"[frodobots-convert] {idx}/{len(selected)} skip existing {seq}")
            summary["converted"].append({"sequence_name": seq, "status": "skipped_existing", "clip_dir": str(clip_dir)})
            continue
        seq_args = argparse.Namespace(**vars(args))
        seq_args.sequence_name = seq
        seq_args.sequence_start = int(run["start"])
        seq_args.sequence_end = int(run["end"])
        seq_args.matched_path = str(run["matched_path"])
        seq_args.output_clip_id = f"{seq}_ep{int(run['episode_index'])}"
        try:
            print(f"[frodobots-convert] {idx}/{len(selected)} converting {seq} frames={run['frame_count']}")
            report = convert_one_sequence(seq_args)
            summary["converted"].append(
                {
                    "sequence_name": seq,
                    "status": "converted",
                    "clip_dir": report.get("clip_dir"),
                    "frame_count": report.get("frame_count"),
                    "duration_s": report.get("duration_s"),
                }
            )
        except Exception as exc:
            print(f"[frodobots-convert] {idx}/{len(selected)} ERROR {seq}: {type(exc).__name__}: {exc}")
            summary["errors"].append(
                {
                    "sequence_name": seq,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "run": run,
                }
            )
        if idx % max(int(args.report_interval), 1) == 0:
            summary["elapsed_s"] = round(time.time() - started, 3)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary["elapsed_s"] = round(time.time() - started, 3)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"selected", "converted", "errors"}}, ensure_ascii=False, indent=2))
    print(f"[frodobots-convert] batch report: {report_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert FrodoBots-7K sequences into SideWalk prepared clips")
    parser.add_argument("--dataset-root", required=True, help="Path to frodobots_dataset")
    parser.add_argument("--output-root", required=True, help="Prepared SideWalk output root")
    parser.add_argument("--sequence-name", default=None, help="Ride sequence or front video name, e.g. ride_1bb855_20240725023922")
    parser.add_argument("--top-percent", type=float, default=None, help="Batch mode: convert longest N percent of contiguous zarr sequences")
    parser.add_argument("--dry-run", action="store_true", help="Only scan/select batch sequences and write the batch report")
    parser.add_argument("--max-sequences", type=int, default=None, help="Optional cap after top-percent selection")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True, help="Skip clips with conversion_report.json in batch mode")
    parser.add_argument("--report-interval", type=int, default=25, help="Write batch report every N converted sequences")
    parser.add_argument("--batch-report-json", default=None, help="Optional batch selection/conversion report path")
    parser.add_argument("--fps", type=float, default=None, help="Override source fps; default reads meta_data/info.json or 10")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap for smoke conversion")
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--scan-chunk-rows", type=int, default=None, help="Rows per zarr string scan chunk")
    parser.add_argument("--max-scan-rows", type=int, default=None, help="Debug cap for locating sequence")
    parser.add_argument("--heading-mode", default="auto", choices=["auto", "enu_yaw", "compass_heading"])
    parser.add_argument("--media-mode", default="hardlink", choices=["hardlink", "symlink", "copy"])
    parser.add_argument("--extract-images", action="store_true", help="Extract aligned images/{row:06d}.jpg from front video timestamps")
    parser.add_argument("--image-stride", type=int, default=1, help="Extract every Nth row when --extract-images is set")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--hfov-deg", type=float, default=90.0, help="Approximate FPV horizontal FOV for camera_params.json")
    parser.add_argument("--camera-height-m", type=float, default=0.6)
    parser.add_argument("--default-width", type=int, default=640, help="Fallback video width if ffprobe fails")
    parser.add_argument("--default-height", type=int, default=360, help="Fallback video height if ffprobe fails")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_percent is not None:
        convert_batch(args)
        return
    if not args.sequence_name:
        raise SystemExit("--sequence-name is required unless --top-percent is set")
    report = convert_one_sequence(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
