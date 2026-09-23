#!/usr/bin/env python3
"""Build multi-frame SideWalk Phase1 archive manifests from single-frame JSONL.

The script keeps the current sample supervision unchanged and replaces each
single current FPV/BEV pair with temporal FPV/BEV pairs sampled from the raw
SideWalk clip directory.  Selected media are materialized under an archive-local
media root so the output manifest remains portable to training servers.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
BEV_SUFFIXES = ("_bev.jpg", "_bev.png", ".jpg", ".png")
_DEST_LOCKS: dict[str, threading.Lock] = {}
_DEST_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class BevMatch:
    """Metadata for a BEV image matched to one FPV history frame.

    Args:
        path: Resolved BEV image path.
        frame_idx: BEV source frame index when available.
        timestamp: BEV source timestamp in nanoseconds when available.
        time_delta_ms: Absolute time delta from the queried FPV frame in ms.
        mapping_source: Rule used to select the BEV frame.

    Returns:
        Dataclass instances are used as structured script-internal records.
    """

    path: Path
    frame_idx: int | None
    timestamp: int | None
    time_delta_ms: float | None
    mapping_source: str


@dataclass(frozen=True)
class HistoryFrame:
    """Resolved FPV/BEV pair for one requested temporal offset.

    Args:
        requested_offset: Requested offset relative to the supervised current frame.
        requested_offset_sec: Requested time offset in seconds when time sampling is used.
        requested_frame_idx: Raw requested FPV frame index before padding.
        frame_idx: Actual FPV frame index used after padding/fallback.
        timestamp: FPV timestamp in nanoseconds when available.
        target_timestamp: Target FPV timestamp in nanoseconds when time sampling is used.
        fpv_time_delta_ms: Absolute delta between the target and selected FPV timestamp.
        actual_offset_sec: Actual selected FPV timestamp offset from current timestamp.
        fpv_path: Resolved FPV image path.
        bev_match: Resolved BEV image and match metadata.
        padded: Whether fallback/padding changed the requested frame.

    Returns:
        Dataclass instances are used as structured script-internal records.
    """

    requested_offset: int
    requested_offset_sec: float | None
    requested_frame_idx: int
    frame_idx: int
    timestamp: int | None
    target_timestamp: int | None
    fpv_time_delta_ms: float | None
    actual_offset_sec: float | None
    fpv_path: Path
    bev_match: BevMatch
    padded: bool


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(
        description="Create a multi-frame SideWalk Phase1 archive JSONL and copy selected media.",
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Input single-frame archive JSONL.")
    parser.add_argument(
        "--sidewalk-root",
        type=Path,
        default=Path("/media/ubuntu/data2/SR-Nav Dataset/sidewalk"),
        help="Raw SideWalk root containing chunk_*/<clip_id>/images and bev_images.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=None,
        help="Output JSONL. Defaults to input stem plus _multiframe_<N>f.jsonl.",
    )
    parser.add_argument(
        "--output-media-root",
        type=Path,
        default=None,
        help="Archive-local media root. Defaults to <archive_root>/media_multiframe.",
    )
    parser.add_argument(
        "--offsets",
        default="-20,-10,0",
        help="Comma-separated FPV frame offsets relative to current frame for --sampling-mode frame_offsets.",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=("frame_offsets", "time"),
        default="frame_offsets",
        help="Use fixed frame offsets or target-FPS timestamp sampling.",
    )
    parser.add_argument(
        "--num-fpv-frames",
        type=int,
        default=5,
        help="Total FPV frames including the supervised current frame for --sampling-mode time.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=5.0,
        help="Target temporal sampling rate in Hz for --sampling-mode time.",
    )
    parser.add_argument(
        "--layout",
        choices=("fpv_bev_pairs", "fpv_history_current_bev"),
        default="fpv_bev_pairs",
        help="Output image layout. fpv_bev_pairs gives each sampled FPV its matched BEV slot.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for smoke tests.",
    )
    parser.add_argument(
        "--max-output-rows",
        type=int,
        default=None,
        help="Optional output row limit; useful with --require-no-padding.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Optional limit on the number of unique clips written.",
    )
    parser.add_argument(
        "--max-rows-per-clip",
        type=int,
        default=None,
        help="Optional limit on written rows per clip.",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="Zero-based input row offset to start from.",
    )
    parser.add_argument(
        "--max-bev-time-delta-ms",
        type=float,
        default=300.0,
        help="Reject nearest timestamp BEV matches farther than this threshold; <=0 disables.",
    )
    parser.add_argument(
        "--max-fpv-time-delta-ms",
        type=float,
        default=150.0,
        help="Mark time-sampled FPV frames as padded if nearest frame exceeds this target-time delta; <=0 disables.",
    )
    parser.add_argument(
        "--missing-policy",
        choices=("pad_current", "skip"),
        default="pad_current",
        help="How to handle missing history frames.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How to materialize selected media under output-media-root.",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute media paths instead of manifest-relative paths.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output manifest and existing copied media.",
    )
    parser.add_argument(
        "--no-prompt-update",
        action="store_true",
        help="Do not rewrite the user prompt image-token line and temporal legend.",
    )
    parser.add_argument(
        "--require-no-padding",
        action="store_true",
        help="Skip rows where any requested history frame has to be padded/fallbacked.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve samples and print stats without writing output.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Concurrent row workers for resolving history frames and materializing media.",
    )
    return parser.parse_args()


def parse_offsets(text: str) -> list[int]:
    """Parse and normalize temporal offsets.

    Args:
        text: Comma-separated integer offsets.

    Returns:
        Sorted list of unique integer offsets.

    Raises:
        ValueError: If no valid offsets are provided or offset 0 is absent.
    """

    offsets = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    if not offsets:
        raise ValueError("At least one offset is required")
    if 0 not in offsets:
        raise ValueError("Offset 0 must be included so the supervised current frame remains visible")
    return offsets


def infer_archive_root(manifest: Path) -> Path:
    """Infer archive root from a manifest path.

    Args:
        manifest: Input manifest path.

    Returns:
        Archive root path.
    """

    if manifest.parent.name == "manifests":
        return manifest.parent.parent
    return manifest.parent


def default_output_manifest(manifest: Path, frame_count: int) -> Path:
    """Build the default output manifest path.

    Args:
        manifest: Input manifest path.
        frame_count: Number of sampled FPV frames.

    Returns:
        Output manifest path.
    """

    return manifest.with_name(f"{manifest.stem}_multiframe_{frame_count}f.jsonl")


def build_clip_index(sidewalk_root: Path) -> dict[str, Path]:
    """Index raw clip directories by clip id.

    Args:
        sidewalk_root: Raw SideWalk root containing chunk directories, or one chunk directory.

    Returns:
        Mapping from clip id to raw clip directory.
    """

    root = sidewalk_root.expanduser().resolve()
    index: dict[str, Path] = {}

    chunk_dirs: list[Path]
    if _has_child_clip_dirs(root):
        chunk_dirs = [root]
    else:
        chunk_dirs = sorted(path for path in root.glob("chunk_*") if path.is_dir())

    for chunk_dir in chunk_dirs:
        if not chunk_dir.is_dir():
            continue
        for clip_dir in sorted(path for path in chunk_dir.iterdir() if path.is_dir()):
            if (clip_dir / "images").is_dir() or (clip_dir / "front_images").is_dir() or (clip_dir / "front.mp4").is_file():
                index.setdefault(clip_dir.name, clip_dir.resolve())
    return index


def _has_child_clip_dirs(root: Path) -> bool:
    """Return whether a directory directly contains SideWalk clip directories.

    Args:
        root: Directory to inspect.

    Returns:
        True when at least one child looks like a SideWalk clip.
    """

    if not root.is_dir():
        return False
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if (child / "images").is_dir() or (child / "front_images").is_dir() or (child / "front.mp4").is_file():
            return True
    return False


def optional_int(value: Any) -> int | None:
    """Convert a value to int if possible.

    Args:
        value: Input value.

    Returns:
        Integer value or None.
    """

    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def optional_float(value: Any) -> float | None:
    """Convert a value to float if possible.

    Args:
        value: Input value.

    Returns:
        Float value or None.
    """

    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def image_dir_is_one_based(image_dir: Path) -> bool:
    """Detect whether an image directory uses one-based frame names.

    Args:
        image_dir: Directory containing FPV image files.

    Returns:
        True when 000001 exists and 000000 does not.
    """

    if any((image_dir / f"000000{suffix}").is_file() for suffix in IMAGE_SUFFIXES):
        return False
    return any((image_dir / f"000001{suffix}").is_file() for suffix in IMAGE_SUFFIXES)


def bev_dir_is_one_based(bev_dir: Path) -> bool:
    """Detect whether a BEV image directory uses one-based frame names.

    Args:
        bev_dir: Directory containing BEV image files.

    Returns:
        True when 000001 exists and 000000 does not.
    """

    zero_names = ["000000_bev.jpg", "000000_bev.png", "000000.jpg", "000000.png"]
    one_names = ["000001_bev.jpg", "000001_bev.png", "000001.jpg", "000001.png"]
    if any((bev_dir / name).is_file() for name in zero_names):
        return False
    return any((bev_dir / name).is_file() for name in one_names)


def frame_name_candidates(frame_idx: int, *, prefer_one_based: bool = False) -> list[str]:
    """Return likely filename stems for a raw frame index.

    Args:
        frame_idx: Zero-based raw frame index.
        prefer_one_based: Whether one-based filename stems should be tried first.

    Returns:
        Candidate filename stems in priority order.
    """

    zero_based = [f"{int(frame_idx):06d}", str(int(frame_idx))]
    one_based = [f"{int(frame_idx) + 1:06d}", str(int(frame_idx) + 1)]
    return one_based + zero_based if prefer_one_based else zero_based + one_based


def resolve_fpv_image(
    clip_dir: Path,
    frame_idx: int,
    *,
    extraction_root: Path | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Resolve an FPV image for a raw frame index.

    Args:
        clip_dir: Raw clip directory.
        frame_idx: Zero-based raw frame index.
        extraction_root: Optional multi-frame media root used to extract from front.mp4.
        overwrite: Whether an extracted frame may overwrite an existing file.

    Returns:
        Resolved image path or None.
    """

    for image_dir_name in ("images", "front_images"):
        image_dir = clip_dir / image_dir_name
        if not image_dir.is_dir():
            continue
        prefer_one_based = image_dir_name == "front_images" or image_dir_is_one_based(image_dir)
        for stem in frame_name_candidates(frame_idx, prefer_one_based=prefer_one_based):
            for suffix in IMAGE_SUFFIXES:
                candidate = image_dir / f"{stem}{suffix}"
                if candidate.is_file():
                    return candidate.resolve()
    if extraction_root is not None:
        video_path = clip_dir / "front.mp4"
        if video_path.is_file():
            return extract_video_frame(
                video_path.resolve(),
                int(frame_idx),
                extraction_root / "clips" / clip_dir.name / "fpv",
                overwrite=overwrite,
            )
    return None


def _destination_lock(path: Path) -> threading.Lock:
    """Return a process-local lock for one destination path.

    Args:
        path: Destination path.

    Returns:
        A lock shared by threads writing the same destination.
    """

    key = str(path.expanduser().resolve())
    with _DEST_LOCKS_GUARD:
        lock = _DEST_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _DEST_LOCKS[key] = lock
        return lock


def extract_video_frame(video_path: Path, frame_idx: int, output_dir: Path, *, overwrite: bool) -> Path | None:
    """Extract one FPV frame from ``front.mp4`` into a deterministic JPG path.

    Args:
        video_path: Source FPV video path.
        frame_idx: Zero-based frame index to extract.
        output_dir: Directory where extracted JPG frames are cached.
        overwrite: Whether to overwrite an existing extracted frame.

    Returns:
        Extracted JPG path, or None when extraction fails.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{int(frame_idx):06d}.jpg"
    with _destination_lock(output_path):
        if output_path.is_file() and not overwrite:
            return output_path.resolve()
        if output_path.exists() or output_path.is_symlink():
            if overwrite:
                output_path.unlink()
            else:
                return output_path.resolve()

        fps, _source = estimate_clip_fps(video_path.parent)
        seek_seconds = max(0.0, float(frame_idx) / float(fps)) if fps and fps > 0 else None
        try:
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
            if seek_seconds is not None:
                cmd.extend(["-ss", f"{seek_seconds:.6f}"])
            cmd.extend(["-i", str(video_path)])
            if seek_seconds is None:
                cmd.extend(["-vf", f"select=eq(n\\,{int(frame_idx)})", "-vframes", "1"])
            else:
                cmd.extend(["-frames:v", "1"])
            cmd.append(str(output_path))
            subprocess.run(cmd, check=True)
            if output_path.is_file():
                return output_path.resolve()
        except Exception:
            pass

        try:
            import decord  # type: ignore

            reader = decord.VideoReader(str(video_path))
            if 0 <= int(frame_idx) < len(reader):
                from PIL import Image

                image = Image.fromarray(reader[int(frame_idx)].asnumpy())
                image.save(output_path, quality=95)
                return output_path.resolve()
        except Exception:
            pass
    return None


@lru_cache(maxsize=512)
def load_frame_times(clip_dir_text: str) -> tuple[int | None, ...]:
    """Load raw frame timestamps for a clip.

    Args:
        clip_dir_text: Raw clip directory path text.

    Returns:
        Tuple of nanosecond timestamps or an empty tuple when unavailable.
    """

    path = Path(clip_dir_text) / "frame_times.npy"
    if not path.is_file():
        return tuple()
    try:
        import numpy as np

        values = np.load(path)
        return tuple(_time_values_to_ns(values))
    except Exception:
        return tuple()


def _time_values_to_ns(values: Any) -> list[int]:
    """Normalize timestamp arrays with ns/us/ms/s units into nanoseconds.

    Args:
        values: Numeric sequence loaded from a timestamp array.

    Returns:
        Nanosecond timestamp list.
    """

    raw = [float(value) for value in values]
    if not raw:
        return []
    diffs = [abs(raw[idx + 1] - raw[idx]) for idx in range(len(raw) - 1) if raw[idx + 1] != raw[idx]]
    unit = _infer_time_unit(raw, diffs)
    scale = {
        "ns": 1.0,
        "us": 1_000.0,
        "ms": 1_000_000.0,
        "s": 1_000_000_000.0,
    }[unit]
    return [int(round(value * scale)) for value in raw]


def _infer_time_unit(values: list[float], diffs: list[float]) -> str:
    """Infer timestamp unit from magnitudes and adjacent deltas.

    Args:
        values: Raw timestamp values.
        diffs: Non-zero adjacent timestamp differences.

    Returns:
        One of ns, us, ms, or s.
    """

    max_abs = max(abs(value) for value in values)
    if max_abs >= 1e17:
        return "ns"
    if max_abs >= 1e14:
        return "us"
    if max_abs >= 1e11:
        return "ms"
    if diffs:
        median = sorted(diffs)[len(diffs) // 2]
        if median >= 1e6:
            return "ns"
        if median >= 1e3:
            return "us"
        if median >= 1.0:
            return "ms"
    return "s"


def estimate_clip_fps(clip_dir: Path) -> tuple[float | None, str | None]:
    """Estimate FPV FPS using InternNav-style source priority.

    Args:
        clip_dir: Raw SideWalk clip directory.

    Returns:
        Pair of FPS value and source name.
    """

    times = [value for value in load_frame_times(str(clip_dir.resolve())) if value is not None]
    if len(times) >= 2:
        elapsed_seconds = (int(times[-1]) - int(times[0])) / 1e9
        if elapsed_seconds > 0:
            return (len(times) - 1) / elapsed_seconds, "frame_times.npy"

    video_path = clip_dir / "front.mp4"
    if video_path.is_file():
        try:
            import decord  # type: ignore

            fps = float(decord.VideoReader(str(video_path)).get_avg_fps())
            if fps > 0:
                return fps, "front.mp4"
        except Exception:
            pass

    fpv_count = _count_fpv_images(clip_dir)
    bev_count = len(load_bev_index_entries(str(clip_dir.resolve()))) or _count_bev_images(clip_dir)
    if fpv_count > 0 and bev_count > 0:
        return max(float(fpv_count) / float(bev_count), 1.0), "fpv_bev_count_ratio"
    return None, None


def _count_fpv_images(clip_dir: Path) -> int:
    """Count available FPV images in a clip.

    Args:
        clip_dir: Raw clip directory.

    Returns:
        Number of FPV image files.
    """

    total = 0
    for image_dir_name in ("images", "front_images"):
        image_dir = clip_dir / image_dir_name
        if image_dir.is_dir():
            total += sum(1 for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    return total


def _count_bev_images(clip_dir: Path) -> int:
    """Count available BEV images in a clip.

    Args:
        clip_dir: Raw clip directory.

    Returns:
        Number of BEV image files.
    """

    total = 0
    for bev_dir_name in ("bev_images", "bev_images_ego_aligned"):
        bev_dir = clip_dir / bev_dir_name
        if bev_dir.is_dir():
            total += sum(1 for path in bev_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    return total


def timestamp_for_frame(clip_dir: Path, frame_idx: int, fallback: int | None) -> int | None:
    """Return the timestamp for a raw frame index.

    Args:
        clip_dir: Raw clip directory.
        frame_idx: Raw frame index.
        fallback: Timestamp to use when the frame timestamp array is unavailable.

    Returns:
        Timestamp in nanoseconds or None.
    """

    values = load_frame_times(str(clip_dir.resolve()))
    if 0 <= int(frame_idx) < len(values):
        return values[int(frame_idx)]
    return fallback


def read_bev_image_meta(path: Path) -> dict[str, Any]:
    """Read a sidecar BEV metadata file.

    Args:
        path: BEV image path.

    Returns:
        Metadata dictionary, empty when absent or invalid.
    """

    meta_path = path.with_suffix(path.suffix + ".bevmeta.json")
    if not meta_path.is_file():
        return {}
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def normalize_bev_index_entry(item: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Normalize one BEV index item to a path/frame/timestamp dictionary.

    Args:
        item: Raw BEV index item.
        base_dir: Directory used to resolve relative paths.

    Returns:
        Normalized BEV index entry.

    Raises:
        FileNotFoundError: If the indexed image is absent from this clip.
    """

    raw_path = item.get("bev_image_path") or item.get("path")
    path = Path(str(raw_path)).expanduser() if raw_path else base_dir / f"{item.get('frame_name', '')}_bev.jpg"
    # 索引所属 clip 决定媒体身份；旧挂载点即使仍存在，也不能覆盖当前数据根。
    bev_dir = base_dir if base_dir.name.startswith("bev_images") else base_dir / "bev_images"
    path = bev_dir / path.name
    if not path.is_file():
        raise FileNotFoundError(f"BEV index {base_dir}: image missing: {path}")
    frame_idx = optional_int(item.get("frame_idx") if item.get("frame_idx") is not None else item.get("source_frame_idx"))
    timestamp = optional_int(item.get("timestamp") if item.get("timestamp") is not None else item.get("source_timestamp_ns"))
    return {
        "path": str(path),
        "frame_idx": frame_idx,
        "timestamp": timestamp,
    }


@lru_cache(maxsize=512)
def load_bev_index_entries(clip_dir_text: str) -> tuple[dict[str, Any], ...]:
    """Load BEV index entries for a raw clip.

    Args:
        clip_dir_text: Raw clip directory path text.

    Returns:
        Tuple of normalized BEV entries.
    """

    clip_dir = Path(clip_dir_text)
    for index_path in (clip_dir / "bev_index.json", clip_dir / "bev_images" / "bev_index.json"):
        if not index_path.is_file():
            continue
        data = json.loads(index_path.read_text(encoding="utf-8"))
        return tuple(
            normalize_bev_index_entry(item, index_path.parent)
            for item in data["entries"]
        )

    entries: list[dict[str, Any]] = []
    for bev_dir_name in ("bev_images", "bev_images_ego_aligned"):
        bev_dir = clip_dir / bev_dir_name
        if not bev_dir.is_dir():
            continue
        for meta_path in sorted(bev_dir.glob("*.bevmeta.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            image_path = Path(str(meta_path)[: -len(".bevmeta.json")])
            item = dict(meta)
            item["path"] = str(image_path)
            entries.append(normalize_bev_index_entry(item, bev_dir))
    return tuple(entries)


def bev_match_meta(
    *,
    path: Path,
    query_frame_idx: int,
    query_timestamp: int | None,
    raw_meta: dict[str, Any],
    mapping_source: str,
) -> BevMatch:
    """Create BEV match metadata for a selected BEV image.

    Args:
        path: Selected BEV image path.
        query_frame_idx: Queried FPV frame index.
        query_timestamp: Queried FPV timestamp in nanoseconds.
        raw_meta: BEV metadata loaded from sidecar/index.
        mapping_source: Matching strategy name.

    Returns:
        BEV match metadata.
    """

    bev_frame_idx = optional_int(raw_meta.get("frame_idx") if raw_meta.get("frame_idx") is not None else raw_meta.get("source_frame_idx"))
    bev_timestamp = optional_int(raw_meta.get("timestamp") if raw_meta.get("timestamp") is not None else raw_meta.get("source_timestamp_ns"))
    if bev_frame_idx is None:
        stem = path.name.split("_bev", 1)[0]
        bev_frame_idx = int(stem) if stem.isdigit() else None
    delta_ms = None
    if query_timestamp is not None and bev_timestamp is not None:
        delta_ms = abs(float(int(query_timestamp) - int(bev_timestamp))) / 1e6
    return BevMatch(
        path=path.resolve(),
        frame_idx=bev_frame_idx,
        timestamp=bev_timestamp,
        time_delta_ms=delta_ms,
        mapping_source=mapping_source,
    )


def nearest_bev_entry(entries: tuple[dict[str, Any], ...], *, frame_idx: int, timestamp: int | None) -> dict[str, Any] | None:
    """Select the nearest BEV index entry for one FPV frame.

    Args:
        entries: Normalized BEV index entries.
        frame_idx: Queried FPV frame index.
        timestamp: Queried FPV timestamp in nanoseconds.

    Returns:
        Best BEV entry or None.
    """

    if not entries:
        return None
    if timestamp is not None and any(entry.get("timestamp") is not None for entry in entries):
        return min(
            entries,
            key=lambda entry: abs(int(entry.get("timestamp")) - int(timestamp))
            if entry.get("timestamp") is not None
            else 10**30,
        )
    return min(
        entries,
        key=lambda entry: abs(int(entry.get("frame_idx")) - int(frame_idx))
        if entry.get("frame_idx") is not None
        else 10**12,
    )


def resolve_bev_match(
    clip_dir: Path,
    frame_idx: int,
    frame_name: str,
    timestamp: int | None,
    *,
    max_bev_time_delta_ms: float,
) -> BevMatch | None:
    """Resolve a BEV image for one raw FPV frame.

    Args:
        clip_dir: Raw clip directory.
        frame_idx: Queried FPV frame index.
        frame_name: Resolved FPV image stem.
        timestamp: Queried FPV timestamp in nanoseconds.
        max_bev_time_delta_ms: Maximum accepted nearest timestamp delta in ms.

    Returns:
        Matched BEV metadata or None.
    """

    for bev_dir_name in ("bev_images", "bev_images_ego_aligned"):
        bev_dir = clip_dir / bev_dir_name
        if not bev_dir.is_dir():
            continue
        prefer_one_based = bev_dir_is_one_based(bev_dir)
        exact_idx = int(frame_idx) + 1 if prefer_one_based else int(frame_idx)
        stems = [frame_name, f"{exact_idx:06d}", str(exact_idx)]
        for stem in dict.fromkeys(stems):
            for suffix in BEV_SUFFIXES:
                candidate = bev_dir / f"{stem}{suffix}"
                if candidate.is_file():
                    return bev_match_meta(
                        path=candidate,
                        query_frame_idx=frame_idx,
                        query_timestamp=timestamp,
                        raw_meta=read_bev_image_meta(candidate),
                        mapping_source="exact_frame_name",
                    )

    best = nearest_bev_entry(load_bev_index_entries(str(clip_dir.resolve())), frame_idx=frame_idx, timestamp=timestamp)
    if best is None:
        return None
    path = Path(str(best["path"])).expanduser()
    if not path.is_file():
        return None
    match = bev_match_meta(
        path=path,
        query_frame_idx=frame_idx,
        query_timestamp=timestamp,
        raw_meta=best,
        mapping_source="nearest_timestamp" if timestamp is not None and best.get("timestamp") is not None else "nearest_frame_idx",
    )
    if (
        max_bev_time_delta_ms is not None
        and float(max_bev_time_delta_ms) > 0
        and match.time_delta_ms is not None
        and float(match.time_delta_ms) > float(max_bev_time_delta_ms)
    ):
        return None
    return match


def resolve_history_frames(
    *,
    row: dict[str, Any],
    clip_dir: Path,
    fpv_extraction_root: Path | None,
    offsets: list[int],
    sampling_mode: str,
    num_fpv_frames: int,
    target_fps: float,
    missing_policy: str,
    overwrite: bool,
    max_fpv_time_delta_ms: float,
    max_bev_time_delta_ms: float,
) -> list[HistoryFrame] | None:
    """Resolve all requested history frames for one archive row.

    Args:
        row: Input JSONL row.
        clip_dir: Raw clip directory for row clip id.
        fpv_extraction_root: Optional media root used for on-demand front.mp4 frame extraction.
        offsets: Requested frame offsets for frame-offset sampling.
        sampling_mode: frame_offsets or time.
        num_fpv_frames: Total sampled FPV frames for time sampling.
        target_fps: Target sampling frequency for time sampling.
        missing_policy: Either pad_current or skip.
        overwrite: Whether extracted FPV frames may be overwritten.
        max_fpv_time_delta_ms: Maximum target-time delta before time samples are padded.
        max_bev_time_delta_ms: Maximum accepted nearest timestamp delta in ms.

    Returns:
        History records, or None when the row should be skipped.
    """

    meta = row.get("sidewalk_meta") if isinstance(row.get("sidewalk_meta"), dict) else {}
    current_idx = optional_int(meta.get("frame_idx"))
    current_timestamp = optional_int(meta.get("timestamp"))
    if current_idx is None:
        return None

    current_fpv = resolve_fpv_image(
        clip_dir,
        current_idx,
        extraction_root=fpv_extraction_root,
        overwrite=overwrite,
    )
    if current_fpv is None:
        return None
    current_timestamp = timestamp_for_frame(clip_dir, current_idx, current_timestamp)

    resolved: list[HistoryFrame] = []
    frame_plan = build_frame_plan(
        clip_dir=clip_dir,
        current_idx=int(current_idx),
        current_timestamp=current_timestamp,
        offsets=offsets,
        sampling_mode=sampling_mode,
        num_fpv_frames=num_fpv_frames,
        target_fps=target_fps,
        max_fpv_time_delta_ms=max_fpv_time_delta_ms,
    )
    for plan in frame_plan:
        requested_idx = int(plan["requested_frame_idx"])
        candidate_idx = max(0, int(plan["frame_idx"]))
        fpv_path = resolve_fpv_image(
            clip_dir,
            candidate_idx,
            extraction_root=fpv_extraction_root,
            overwrite=overwrite,
        )
        padded = bool(plan["padded"]) or candidate_idx != requested_idx
        if fpv_path is None:
            if missing_policy == "skip":
                return None
            candidate_idx = int(current_idx)
            fpv_path = current_fpv
            padded = True
        timestamp = timestamp_for_frame(
            clip_dir,
            candidate_idx,
            current_timestamp if candidate_idx == int(current_idx) else None,
        )
        bev_match = resolve_bev_match(
            clip_dir,
            candidate_idx,
            fpv_path.stem,
            timestamp,
            max_bev_time_delta_ms=max_bev_time_delta_ms,
        )
        if bev_match is None:
            if missing_policy == "skip":
                return None
            bev_match = resolve_bev_match(
                clip_dir,
                int(current_idx),
                current_fpv.stem,
                current_timestamp,
                max_bev_time_delta_ms=max_bev_time_delta_ms,
            )
            if bev_match is None:
                return None
            candidate_idx = int(current_idx)
            fpv_path = current_fpv
            timestamp = current_timestamp
            padded = True

        resolved.append(
            HistoryFrame(
                requested_offset=int(plan["requested_offset"]),
                requested_offset_sec=optional_float(plan.get("requested_offset_sec")),
                requested_frame_idx=requested_idx,
                frame_idx=candidate_idx,
                timestamp=timestamp,
                target_timestamp=optional_int(plan.get("target_timestamp")),
                fpv_time_delta_ms=optional_float(plan.get("fpv_time_delta_ms")),
                actual_offset_sec=(
                    (float(timestamp) - float(current_timestamp)) / 1e9
                    if timestamp is not None and current_timestamp is not None
                    else None
                ),
                fpv_path=fpv_path,
                bev_match=bev_match,
                padded=padded,
            )
        )
    return resolved


def build_frame_plan(
    *,
    clip_dir: Path,
    current_idx: int,
    current_timestamp: int | None,
    offsets: list[int],
    sampling_mode: str,
    num_fpv_frames: int,
    target_fps: float,
    max_fpv_time_delta_ms: float,
) -> list[dict[str, Any]]:
    """Build the requested FPV frame plan for one supervised frame.

    Args:
        clip_dir: Raw clip directory.
        current_idx: Supervised current FPV frame index.
        current_timestamp: Current FPV timestamp in nanoseconds.
        offsets: Frame-index offsets for frame_offsets mode.
        sampling_mode: frame_offsets or time.
        num_fpv_frames: Total sampled FPV frames for time mode.
        target_fps: Target FPS for time mode.
        max_fpv_time_delta_ms: Maximum target-time matching delta before padding is marked.

    Returns:
        Frame-plan dictionaries ordered oldest to newest.
    """

    if sampling_mode == "frame_offsets":
        return [
            {
                "requested_offset": int(offset),
                "requested_offset_sec": None,
                "requested_frame_idx": int(current_idx) + int(offset),
                "frame_idx": int(current_idx) + int(offset),
                "target_timestamp": None,
                "fpv_time_delta_ms": None,
                "padded": int(current_idx) + int(offset) < 0,
            }
            for offset in offsets
        ]

    frame_count = max(int(num_fpv_frames), 1)
    fps = float(target_fps)
    if fps <= 0:
        raise ValueError("--target-fps must be positive for --sampling-mode time")
    offset_seconds = [-(frame_count - 1 - idx) / fps for idx in range(frame_count)]
    times = load_frame_times(str(clip_dir.resolve()))
    plan: list[dict[str, Any]] = []
    estimated_fps, _fps_source = estimate_clip_fps(clip_dir)
    fallback_fps = estimated_fps if estimated_fps and estimated_fps > 0 else fps
    for offset_sec in offset_seconds:
        if abs(offset_sec) < 1e-9:
            frame_idx = int(current_idx)
            target_timestamp = current_timestamp
            fpv_delta_ms = 0.0
        elif current_timestamp is not None and times:
            target_timestamp = int(round(float(current_timestamp) + offset_sec * 1e9))
            frame_idx, fpv_delta_ms = nearest_frame_index_by_timestamp(
                times,
                target_timestamp,
                max_index=int(current_idx),
            )
        else:
            requested = int(round(float(current_idx) + offset_sec * float(fallback_fps)))
            frame_idx = requested
            target_timestamp = None
            fpv_delta_ms = None
        max_delta = float(max_fpv_time_delta_ms)
        padded = frame_idx < 0
        if max_delta > 0 and fpv_delta_ms is not None and float(fpv_delta_ms) > max_delta:
            padded = True
        requested_frame_idx = frame_idx if target_timestamp is not None else int(round(float(current_idx) + offset_sec * float(fallback_fps)))
        plan.append(
            {
                "requested_offset": int(frame_idx) - int(current_idx),
                "requested_offset_sec": float(offset_sec),
                "requested_frame_idx": int(requested_frame_idx),
                "frame_idx": int(frame_idx),
                "target_timestamp": target_timestamp,
                "fpv_time_delta_ms": fpv_delta_ms,
                "padded": padded,
            }
        )
    return plan


def nearest_frame_index_by_timestamp(
    times: tuple[int | None, ...],
    target_timestamp: int,
    *,
    max_index: int,
) -> tuple[int, float | None]:
    """Find the nearest frame index at or before the current frame.

    Args:
        times: Clip FPV timestamps in nanoseconds.
        target_timestamp: Desired timestamp in nanoseconds.
        max_index: Highest frame index allowed, normally the current frame.

    Returns:
        Pair of selected frame index and target-time delta in milliseconds.
    """

    valid_times = [int(value) for value in times[: max(0, int(max_index)) + 1] if value is not None]
    if not valid_times:
        return max(0, int(max_index)), None
    pos = bisect.bisect_left(valid_times, int(target_timestamp))
    candidates = []
    if 0 <= pos < len(valid_times):
        candidates.append(pos)
    if pos - 1 >= 0:
        candidates.append(pos - 1)
    if pos + 1 < len(valid_times):
        candidates.append(pos + 1)
    if not candidates:
        candidates = [0]
    best = min(candidates, key=lambda idx: abs(valid_times[idx] - int(target_timestamp)))
    delta_ms = abs(float(valid_times[best] - int(target_timestamp))) / 1e6
    return int(best), delta_ms


def materialize_media(
    source: Path,
    *,
    destination: Path,
    copy_mode: str,
    overwrite: bool,
) -> None:
    """Materialize a source media file at a destination path.

    Args:
        source: Source image path.
        destination: Destination image path.
        copy_mode: copy, hardlink, or symlink.
        overwrite: Whether to overwrite an existing destination.

    Returns:
        None.
    """

    with _destination_lock(destination):
        if source.resolve() == destination.resolve() and destination.is_file():
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            if not overwrite:
                return
            destination.unlink()
        if copy_mode == "symlink":
            try:
                destination.symlink_to(source)
            except FileExistsError:
                return
        elif copy_mode == "hardlink":
            try:
                os.link(source, destination)
            except FileExistsError:
                return
            except OSError:
                shutil.copy2(source, destination)
        else:
            shutil.copy2(source, destination)


def media_destination(output_media_root: Path, clip_id: str, kind: str, source: Path) -> Path:
    """Build a deterministic archive-local media destination.

    Args:
        output_media_root: Root for copied multi-frame media.
        clip_id: Clip id.
        kind: Media kind, fpv or bev.
        source: Source image path.

    Returns:
        Destination path.
    """

    return output_media_root / "clips" / clip_id / kind / source.name


def manifest_path_for_image(path: Path, manifest_parent: Path, *, absolute_paths: bool) -> str:
    """Format a media path for the output JSONL row.

    Args:
        path: Materialized media path.
        manifest_parent: Output manifest directory.
        absolute_paths: Whether to write absolute paths.

    Returns:
        Path string for JSONL.
    """

    if absolute_paths:
        return str(path.resolve())
    return os.path.relpath(path.resolve(), manifest_parent.resolve())


def update_user_prompt(row: dict[str, Any], history: list[HistoryFrame], *, layout: str, target_fps: float | None) -> None:
    """Update the user prompt to describe temporal FPV/BEV image order.

    Args:
        row: JSONL row to mutate.
        history: Temporal FPV/BEV matches in output order.
        layout: Output image layout.
        target_fps: Requested target FPS when available.

    Returns:
        None.
    """

    messages = row.get("messages")
    if not isinstance(messages, list):
        return
    image_count = len(history) * 2 if layout == "fpv_bev_pairs" else len(history) + 1
    image_tokens = "<image>" * image_count
    if layout == "fpv_bev_pairs":
        legend = (
            "Temporal image order: FPV/BEV pairs are ordered from oldest to newest; "
            "the final pair is the supervised current frame. "
        )
    else:
        legend = (
            "Temporal image order: FPV frames are ordered from oldest to newest, "
            "followed by the current BEV route context. "
        )
    offsets_s = [item.requested_offset_sec for item in history]
    offsets_idx = [item.requested_offset for item in history]
    if all(value is not None for value in offsets_s):
        legend += f"Target FPS is {target_fps:g}; requested time offsets are {offsets_s} seconds.\n"
    else:
        legend += f"Requested frame offsets are {offsets_idx}.\n"
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if content.startswith("<image>"):
            _, sep, rest = content.partition("\n")
            message["content"] = f"{image_tokens}\n{legend}{rest if sep else ''}"
        else:
            message["content"] = f"{image_tokens}\n{legend}{content}"
        return


def augment_row(
    *,
    row: dict[str, Any],
    history: list[HistoryFrame],
    clip_id: str,
    output_media_root: Path,
    output_manifest_parent: Path,
    copy_mode: str,
    overwrite: bool,
    absolute_paths: bool,
    update_prompt: bool,
    layout: str,
    sampling_mode: str,
    target_fps: float | None,
    estimated_fps: float | None,
    fps_source: str | None,
) -> dict[str, Any]:
    """Build one output multi-frame JSONL row.

    Args:
        row: Input JSONL row.
        history: Resolved temporal frame records.
        clip_id: Clip id.
        output_media_root: Root where selected media are materialized.
        output_manifest_parent: Output manifest directory for relative paths.
        copy_mode: copy, hardlink, or symlink.
        overwrite: Whether to overwrite existing media.
        absolute_paths: Whether manifest image paths should be absolute.
        update_prompt: Whether to rewrite prompt image tokens and temporal legend.
        layout: Output image layout.
        sampling_mode: Sampling mode used to choose FPV frames.
        target_fps: Requested target FPS for time sampling.
        estimated_fps: Estimated raw clip FPS.
        fps_source: Source used to estimate raw clip FPS.

    Returns:
        Augmented output row.
    """

    output = dict(row)
    images: list[str] = []
    frame_meta: list[dict[str, Any]] = []
    fpv_images: list[str] = []
    bev_images: list[str] = []
    for item in history:
        fpv_dst = media_destination(output_media_root, clip_id, "fpv", item.fpv_path)
        bev_dst = media_destination(output_media_root, clip_id, "bev", item.bev_match.path)
        materialize_media(item.fpv_path, destination=fpv_dst, copy_mode=copy_mode, overwrite=overwrite)
        materialize_media(item.bev_match.path, destination=bev_dst, copy_mode=copy_mode, overwrite=overwrite)
        fpv_image = manifest_path_for_image(fpv_dst, output_manifest_parent, absolute_paths=absolute_paths)
        bev_image = manifest_path_for_image(bev_dst, output_manifest_parent, absolute_paths=absolute_paths)
        fpv_images.append(fpv_image)
        bev_images.append(bev_image)
        if layout == "fpv_bev_pairs":
            images.append(fpv_image)
            images.append(bev_image)
        frame_meta.append(
            {
                "requested_offset": item.requested_offset,
                "requested_offset_sec": item.requested_offset_sec,
                "requested_frame_idx": item.requested_frame_idx,
                "frame_idx": item.frame_idx,
                "timestamp": item.timestamp,
                "target_timestamp": item.target_timestamp,
                "fpv_time_delta_ms": item.fpv_time_delta_ms,
                "actual_offset_sec": item.actual_offset_sec,
                "fpv_image": fpv_image,
                "bev_image": bev_image,
                "bev_frame_idx": item.bev_match.frame_idx,
                "bev_timestamp": item.bev_match.timestamp,
                "bev_time_delta_ms": item.bev_match.time_delta_ms,
                "bev_mapping_source": item.bev_match.mapping_source,
                "padded": item.padded,
            }
        )
    if layout == "fpv_history_current_bev":
        images = [*fpv_images, bev_images[-1]]
    output["images"] = images
    meta = dict(output.get("sidewalk_meta") if isinstance(output.get("sidewalk_meta"), dict) else {})
    meta["multiframe_schema_version"] = "sidewalk_phase1_multiframe_archive_v1"
    meta["multiframe_pair_order"] = layout
    meta["multiframe_sampling_mode"] = sampling_mode
    meta["multiframe_target_fps"] = target_fps
    meta["multiframe_estimated_raw_fps"] = estimated_fps
    meta["multiframe_fps_source"] = fps_source
    meta["multiframe_effective_fps"] = target_fps if sampling_mode == "time" else None
    meta["multiframe_offsets"] = [item.requested_offset for item in history]
    meta["multiframe_requested_offsets_sec"] = [item.requested_offset_sec for item in history]
    meta["multiframe_actual_offsets_sec"] = [item.actual_offset_sec for item in history]
    meta["multiframe_num_fpv_frames"] = len(history)
    meta["multiframe_num_images"] = len(images)
    meta["multiframe_bev_unique_count"] = len({item.bev_match.path for item in history})
    meta["multiframe_frames"] = frame_meta
    meta["multiframe_media_mode"] = copy_mode
    output["sidewalk_meta"] = meta
    if update_prompt:
        update_user_prompt(output, history, layout=layout, target_fps=target_fps)
    return output


@dataclass(frozen=True)
class _ManifestTask:
    order: int
    row: dict[str, Any]
    clip_id: str
    clip_dir: Path
    estimated_fps: float | None
    fps_source: str | None


@dataclass(frozen=True)
class _ManifestTaskResult:
    order: int
    clip_id: str
    output_line: str | None
    unresolved_frames: int = 0
    skipped_padded_rows: int = 0
    padded_frames: int = 0
    worker_errors: int = 0
    error: str | None = None


def _process_manifest_task(
    task: _ManifestTask,
    *,
    args: argparse.Namespace,
    offsets: list[int],
    output_media_root: Path,
    output_manifest_parent: Path,
) -> _ManifestTaskResult:
    """Resolve and optionally materialize one multi-frame archive row.

    Args:
        task: Prepared manifest row task.
        args: Parsed CLI arguments.
        offsets: Frame-offset sampling plan.
        output_media_root: Root for archive-local media.
        output_manifest_parent: Directory containing the output manifest.

    Returns:
        Task result with either one JSONL line or skip/error counters.
    """

    try:
        history = resolve_history_frames(
            row=task.row,
            clip_dir=task.clip_dir,
            fpv_extraction_root=None if args.dry_run else output_media_root,
            offsets=offsets,
            sampling_mode=args.sampling_mode,
            num_fpv_frames=int(args.num_fpv_frames),
            target_fps=float(args.target_fps),
            missing_policy=args.missing_policy,
            overwrite=bool(args.overwrite),
            max_fpv_time_delta_ms=float(args.max_fpv_time_delta_ms),
            max_bev_time_delta_ms=args.max_bev_time_delta_ms,
        )
        if history is None:
            return _ManifestTaskResult(order=task.order, clip_id=task.clip_id, output_line=None, unresolved_frames=1)
        if args.require_no_padding and any(item.padded for item in history):
            return _ManifestTaskResult(order=task.order, clip_id=task.clip_id, output_line=None, skipped_padded_rows=1)
        padded_frames = sum(1 for item in history if item.padded)
        output_line = None
        if not args.dry_run:
            output = augment_row(
                row=task.row,
                history=history,
                clip_id=task.clip_id,
                output_media_root=output_media_root,
                output_manifest_parent=output_manifest_parent,
                copy_mode=args.copy_mode,
                overwrite=args.overwrite,
                absolute_paths=args.absolute_paths,
                update_prompt=not args.no_prompt_update,
                layout=args.layout,
                sampling_mode=args.sampling_mode,
                target_fps=float(args.target_fps) if args.sampling_mode == "time" else None,
                estimated_fps=task.estimated_fps,
                fps_source=task.fps_source,
            )
            output_line = json.dumps(output, ensure_ascii=False, separators=(",", ":"))
        return _ManifestTaskResult(
            order=task.order,
            clip_id=task.clip_id,
            output_line=output_line,
            padded_frames=padded_frames,
        )
    except Exception as exc:
        return _ManifestTaskResult(
            order=task.order,
            clip_id=task.clip_id,
            output_line=None,
            worker_errors=1,
            error=f"{type(exc).__name__}:{str(exc)[:300]}",
        )


def process_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Process the input JSONL manifest.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Processing statistics.

    Raises:
        FileExistsError: If the output manifest exists without --overwrite.
    """

    manifest = args.manifest.expanduser().resolve()
    offsets = parse_offsets(args.offsets)
    frame_count = int(args.num_fpv_frames) if args.sampling_mode == "time" else len(offsets)
    archive_root = infer_archive_root(manifest)
    output_manifest = (
        args.output_manifest.expanduser().resolve()
        if args.output_manifest is not None
        else default_output_manifest(manifest, frame_count).resolve()
    )
    output_media_root = (
        args.output_media_root.expanduser().resolve()
        if args.output_media_root is not None
        else (archive_root / "media_multiframe").resolve()
    )
    if output_manifest.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Output manifest exists; pass --overwrite: {output_manifest}")

    clip_index = build_clip_index(args.sidewalk_root)
    stats = {
        "rows_seen": 0,
        "rows_written": 0,
        "rows_skipped_start": 0,
        "missing_clip": 0,
        "unresolved_frames": 0,
        "skipped_padded_rows": 0,
        "skipped_max_clips": 0,
        "skipped_rows_per_clip": 0,
        "padded_frames": 0,
    }
    selected_clip_ids: set[str] = set()
    selected_per_clip: dict[str, int] = {}
    tasks: list[_ManifestTask] = []
    with manifest.open("r", encoding="utf-8") as source:
        input_row_index = -1
        for line in source:
            if not line.strip():
                continue
            input_row_index += 1
            if input_row_index < int(args.start_row):
                stats["rows_skipped_start"] += 1
                continue
            if args.max_rows is not None and stats["rows_seen"] >= int(args.max_rows):
                break
            stats["rows_seen"] += 1
            row = json.loads(line)
            meta = row.get("sidewalk_meta") if isinstance(row.get("sidewalk_meta"), dict) else {}
            clip_id = str(meta.get("clip_id") or meta.get("archive_clip_id") or "").strip()
            clip_dir = clip_index.get(clip_id)
            if clip_dir is None:
                stats["missing_clip"] += 1
                continue
            if args.max_clips is not None and clip_id not in selected_clip_ids and len(selected_clip_ids) >= int(args.max_clips):
                stats["skipped_max_clips"] += 1
                continue
            if args.max_rows_per_clip is not None and selected_per_clip.get(clip_id, 0) >= int(args.max_rows_per_clip):
                stats["skipped_rows_per_clip"] += 1
                continue
            estimated_fps, fps_source = estimate_clip_fps(clip_dir)
            tasks.append(
                _ManifestTask(
                    order=input_row_index,
                    row=row,
                    clip_id=clip_id,
                    clip_dir=clip_dir,
                    estimated_fps=estimated_fps,
                    fps_source=fps_source,
                )
            )
            selected_clip_ids.add(clip_id)
            selected_per_clip[clip_id] = selected_per_clip.get(clip_id, 0) + 1

    results: list[_ManifestTaskResult] = []
    workers = max(1, int(getattr(args, "num_workers", 1)))
    if workers == 1 or len(tasks) <= 1:
        for task in tasks:
            results.append(
                _process_manifest_task(
                    task,
                    args=args,
                    offsets=offsets,
                    output_media_root=output_media_root,
                    output_manifest_parent=output_manifest.parent,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    _process_manifest_task,
                    task,
                    args=args,
                    offsets=offsets,
                    output_media_root=output_media_root,
                    output_manifest_parent=output_manifest.parent,
                ): task
                for task in tasks
            }
            for future in as_completed(future_map):
                results.append(future.result())

    output_rows: list[str] = []
    written_clip_ids: set[str] = set()
    worker_error_examples: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda item: item.order):
        stats["unresolved_frames"] += int(result.unresolved_frames)
        stats["skipped_padded_rows"] += int(result.skipped_padded_rows)
        stats["padded_frames"] += int(result.padded_frames)
        stats["worker_errors"] = int(stats.get("worker_errors", 0)) + int(result.worker_errors)
        if result.error and len(worker_error_examples) < 10:
            worker_error_examples.append({"row": result.order, "clip_id": result.clip_id, "error": result.error})
        if result.unresolved_frames or result.skipped_padded_rows or result.worker_errors:
            continue
        if result.output_line is None and not args.dry_run:
            continue
        if args.max_output_rows is not None and stats["rows_written"] >= int(args.max_output_rows):
            stats["skipped_max_output_rows"] = int(stats.get("skipped_max_output_rows", 0)) + 1
            continue
        if result.output_line is not None:
            output_rows.append(result.output_line)
        written_clip_ids.add(result.clip_id)
        stats["rows_written"] += 1

    if not args.dry_run:
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
        output_manifest.write_text("\n".join(output_rows) + ("\n" if output_rows else ""), encoding="utf-8")
        stats["output_manifest"] = str(output_manifest)  # type: ignore[assignment]
        stats["output_media_root"] = str(output_media_root)  # type: ignore[assignment]
    stats["clips_written"] = len(written_clip_ids)
    stats["clip_index_size"] = len(clip_index)
    stats["num_workers"] = workers
    if worker_error_examples:
        stats["worker_error_examples"] = worker_error_examples
    return stats


def main() -> None:
    """Run the multi-frame archive builder.

    Returns:
        None.
    """

    stats = process_manifest(parse_args())
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
