#!/usr/bin/env python3
"""Build a self-contained multi-frame Frodobots Phase1 source JSONL.

The input is a Phase1 source JSONL produced by ``sidewalk_phase1_dataset.py``.
Each row keeps its current-frame supervision unchanged, while this script
materializes historical FPV frames from the structured clip ``front.mp4`` and
copies the current BEV media into a new output root.  The resulting JSONL can
survive deletion of the original structured-window media directories.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


BEV_RE = re.compile(r"^(\d+)_bev\.(?:jpg|jpeg|png)$", re.IGNORECASE)


@dataclass(frozen=True)
class HistorySpec:
    """Resolved temporal FPV/BEV sample for one row.

    Args:
        order: Position in the oldest-to-newest FPV sequence.
        requested_offset_sec: Requested timestamp offset relative to the current frame.
        requested_timestamp: Target timestamp in nanoseconds.
        frame_idx: Structured-window frame index selected after nearest/padding.
        video_idx: Source video frame index used for decoding.
        timestamp: Selected FPV timestamp in nanoseconds.
        fpv_time_delta_ms: Absolute target-to-selected FPV timestamp error.
        bev_frame_idx: Matched BEV frame index.
        bev_time_delta_ms: Absolute selected-FPV-to-BEV timestamp error.
        padded: Whether padding/fallback was required.

    Returns:
        Immutable record used by the converter.
    """

    order: int
    requested_offset_sec: float
    requested_timestamp: int | None
    frame_idx: int
    video_idx: int
    timestamp: int | None
    fpv_time_delta_ms: float | None
    bev_frame_idx: int | None
    bev_time_delta_ms: float | None
    padded: bool


@dataclass(frozen=True)
class ClipContext:
    """Per-clip media and timing metadata.

    Args:
        clip_id: Phase1 clip identifier.
        clip_dir: Structured-window clip directory.
        front_video: Path to the front camera video.
        frame_times: Structured-window frame timestamps in nanoseconds.
        frame_video_indices: Source video frame indices for each structured frame.
        bev_frames: Sorted BEV frame indices available on disk.
        bev_paths: Mapping from BEV frame index to image path.
        source_fps: Estimated structured-window FPS.

    Returns:
        Immutable record used by per-clip workers.
    """

    clip_id: str
    clip_dir: Path
    front_video: Path
    frame_times: np.ndarray
    frame_video_indices: np.ndarray
    bev_frames: list[int]
    bev_paths: dict[int, Path]
    source_fps: float


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file.

    Args:
        path: JSONL path.

    Returns:
        Parsed rows.

    Raises:
        ValueError: If a row is invalid JSON.
    """

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write rows as JSONL.

    Args:
        path: Output path.
        rows: JSON-serializable rows.

    Returns:
        None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_name(value: Any, fallback: str = "unknown") -> str:
    """Convert an arbitrary value into a filesystem-safe name.

    Args:
        value: Raw value.
        fallback: Name to use when the normalized value is empty.

    Returns:
        Safe basename.
    """

    text = str(value or fallback).strip()
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
    return cleaned.strip("._") or fallback


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(description="Create self-contained 5Hz multi-frame Phase1 source data.")
    parser.add_argument("--source-jsonl", type=Path, required=True, help="Input Phase1 source JSONL.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output dataset root.")
    parser.add_argument("--split-name", default=None, help="Output split name. Defaults to input stem plus multiframe suffix.")
    parser.add_argument("--num-fpv-frames", type=int, default=5, help="Total FPV frames including current frame.")
    parser.add_argument("--target-fps", type=float, default=5.0, help="Historical FPV sampling rate in Hz.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of clip workers.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for smoke tests.")
    parser.add_argument("--max-clips", type=int, default=None, help="Optional clip limit for smoke tests.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="Output FPV JPEG quality.")
    parser.add_argument("--absolute-paths", action="store_true", default=True, help="Write absolute paths in the output JSONL.")
    parser.add_argument("--relative-paths", dest="absolute_paths", action="store_false", help="Write paths relative to manifest dir.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing manifest and media files.")
    return parser.parse_args()


def _clip_dir_from_row(row: dict[str, Any]) -> Path:
    bev_path = Path(str(row.get("bev_image_path") or "")).expanduser()
    if not bev_path.is_file():
        raise FileNotFoundError(f"Missing BEV image for row {row.get('id')}: {bev_path}")
    if bev_path.parent.name != "bev_images":
        raise ValueError(f"Expected BEV image under bev_images/: {bev_path}")
    return bev_path.parent.parent.resolve()


def _estimate_fps(frame_times: np.ndarray) -> float:
    if frame_times.size < 2:
        return 10.0
    diffs = np.diff(frame_times.astype(np.float64))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 10.0
    median_ns = float(np.median(diffs))
    if median_ns <= 0:
        return 10.0
    return 1e9 / median_ns


def _load_clip_context(clip_id: str, clip_dir: Path) -> ClipContext:
    frame_times_path = clip_dir / "frame_times.npy"
    frame_video_indices_path = clip_dir / "frame_video_indices.npy"
    front_video = clip_dir / "front.mp4"
    if not frame_times_path.is_file():
        raise FileNotFoundError(f"Missing frame_times.npy: {frame_times_path}")
    if not frame_video_indices_path.is_file():
        raise FileNotFoundError(f"Missing frame_video_indices.npy: {frame_video_indices_path}")
    if not front_video.exists():
        raise FileNotFoundError(f"Missing front.mp4: {front_video}")

    frame_times = np.load(frame_times_path).astype(np.int64)
    frame_video_indices = np.load(frame_video_indices_path).astype(np.int64)
    if frame_times.shape[0] != frame_video_indices.shape[0]:
        raise ValueError(f"frame_times/frame_video_indices length mismatch for {clip_id}")

    bev_dir = clip_dir / "bev_images"
    bev_paths: dict[int, Path] = {}
    for image_path in bev_dir.iterdir():
        if not image_path.is_file():
            continue
        match = BEV_RE.match(image_path.name)
        if match:
            bev_paths[int(match.group(1))] = image_path.resolve()
    if not bev_paths:
        raise FileNotFoundError(f"No BEV images found for {clip_id}: {bev_dir}")

    return ClipContext(
        clip_id=clip_id,
        clip_dir=clip_dir,
        front_video=front_video.resolve(),
        frame_times=frame_times,
        frame_video_indices=frame_video_indices,
        bev_frames=sorted(bev_paths),
        bev_paths=bev_paths,
        source_fps=_estimate_fps(frame_times),
    )


def _nearest_index(values: np.ndarray, target: int) -> int:
    pos = int(np.searchsorted(values, target))
    if pos <= 0:
        return 0
    if pos >= values.shape[0]:
        return values.shape[0] - 1
    before = pos - 1
    after = pos
    if abs(int(values[after]) - target) < abs(target - int(values[before])):
        return after
    return before


def _nearest_bev_frame(bev_frames: list[int], frame_idx: int) -> int:
    pos = bisect.bisect_left(bev_frames, frame_idx)
    if pos <= 0:
        return bev_frames[0]
    if pos >= len(bev_frames):
        return bev_frames[-1]
    before = bev_frames[pos - 1]
    after = bev_frames[pos]
    return after if abs(after - frame_idx) < abs(frame_idx - before) else before


def _resolve_history(row: dict[str, Any], ctx: ClipContext, *, num_fpv_frames: int, target_fps: float) -> list[HistorySpec]:
    current_idx = int(row.get("frame_idx") if row.get("frame_idx") is not None else (row.get("sidewalk_meta") or {}).get("frame_idx"))
    if current_idx < 0 or current_idx >= ctx.frame_times.shape[0]:
        raise ValueError(f"Current frame_idx out of range for {row.get('id')}: {current_idx}")
    current_ts = int(ctx.frame_times[current_idx])
    dt_sec = 1.0 / float(target_fps)
    history: list[HistorySpec] = []
    for order in range(num_fpv_frames):
        offset_steps = num_fpv_frames - 1 - order
        requested_offset_sec = -offset_steps * dt_sec
        target_ts = int(round(current_ts + requested_offset_sec * 1e9))
        selected_idx = _nearest_index(ctx.frame_times, target_ts)
        if order == num_fpv_frames - 1:
            selected_idx = current_idx
        selected_ts = int(ctx.frame_times[selected_idx])
        fpv_delta_ms = abs(selected_ts - target_ts) / 1e6
        padded = target_ts < int(ctx.frame_times[0]) or target_ts > int(ctx.frame_times[-1])
        bev_idx = _nearest_bev_frame(ctx.bev_frames, selected_idx)
        bev_delta_ms = None
        if 0 <= bev_idx < ctx.frame_times.shape[0]:
            bev_delta_ms = abs(int(ctx.frame_times[bev_idx]) - selected_ts) / 1e6
        history.append(
            HistorySpec(
                order=order,
                requested_offset_sec=requested_offset_sec,
                requested_timestamp=target_ts,
                frame_idx=int(selected_idx),
                video_idx=int(ctx.frame_video_indices[selected_idx]),
                timestamp=selected_ts,
                fpv_time_delta_ms=round(float(fpv_delta_ms), 3),
                bev_frame_idx=int(bev_idx),
                bev_time_delta_ms=round(float(bev_delta_ms), 3) if bev_delta_ms is not None else None,
                padded=bool(padded),
            )
        )
    return history


def _manifest_path(path: Path, *, manifest_dir: Path, absolute_paths: bool) -> str:
    resolved = path.resolve()
    if absolute_paths:
        return str(resolved)
    return os.path.relpath(resolved, manifest_dir.resolve())


def _copy_file(source: Path, dest: Path, *, overwrite: bool) -> None:
    if dest.is_file() and not overwrite:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def _extract_video_frames(
    ctx: ClipContext,
    frame_specs: Iterable[HistorySpec],
    output_paths: dict[int, Path],
    *,
    image_width: int,
    image_height: int,
    jpeg_quality: int,
    overwrite: bool,
) -> int:
    pending = [spec for spec in sorted({spec.frame_idx: spec for spec in frame_specs}.values(), key=lambda item: item.video_idx) if overwrite or not output_paths[spec.frame_idx].is_file()]
    if not pending:
        return 0
    cap = cv2.VideoCapture(str(ctx.front_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for {ctx.clip_id}: {ctx.front_video}")
    written = 0
    try:
        for spec in pending:
            dest = output_paths[spec.frame_idx]
            dest.parent.mkdir(parents=True, exist_ok=True)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(spec.video_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Cannot decode video frame {spec.video_idx} for {ctx.clip_id}")
            if image_width > 0 and image_height > 0 and (frame.shape[1] != image_width or frame.shape[0] != image_height):
                frame = cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_AREA)
            ok = cv2.imwrite(str(dest), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            if not ok:
                raise RuntimeError(f"Cannot write FPV frame: {dest}")
            written += 1
    finally:
        cap.release()
    return written


def _copy_sidecars(row: dict[str, Any], *, output_root: Path, manifest_dir: Path, absolute_paths: bool, overwrite: bool) -> dict[str, str]:
    clip_id = safe_name(row.get("clip_id") or (row.get("sidewalk_meta") or {}).get("clip_id"), "unknown_clip")
    frame_name = safe_name(row.get("frame_name") or (row.get("sidewalk_meta") or {}).get("frame_name"), "unknown_frame")
    sidecar_map = {
        "trajectory_label_path": ("trajectory_labels", f"{frame_name}.json"),
        "bev_patch_meta_path": ("bev_patch_meta", f"{frame_name}.json"),
        "qa_bundle_path": ("qa", f"{frame_name}_qa.json"),
    }
    out: dict[str, str] = {}
    for key, (subdir, filename) in sidecar_map.items():
        source = Path(str(row.get(key) or (row.get("sidewalk_meta") or {}).get(key) or "")).expanduser()
        if not source.is_file():
            continue
        dest = output_root / "sidecars" / "clips" / clip_id / subdir / filename
        _copy_file(source.resolve(), dest, overwrite=overwrite)
        out[key] = _manifest_path(dest, manifest_dir=manifest_dir, absolute_paths=absolute_paths)
    return out


def _process_clip(
    clip_key: tuple[str, str],
    indexed_rows: list[tuple[int, dict[str, Any]]],
    *,
    output_root: Path,
    manifest_dir: Path,
    num_fpv_frames: int,
    target_fps: float,
    jpeg_quality: int,
    absolute_paths: bool,
    overwrite: bool,
) -> tuple[list[tuple[int, dict[str, Any]]], dict[str, Any]]:
    clip_id, clip_dir_text = clip_key
    ctx = _load_clip_context(clip_id, Path(clip_dir_text))
    media_root = output_root / "media" / "clips" / safe_name(clip_id)
    histories: dict[int, list[HistorySpec]] = {}
    unique_specs: dict[int, HistorySpec] = {}
    for index, row in indexed_rows:
        history = _resolve_history(row, ctx, num_fpv_frames=num_fpv_frames, target_fps=target_fps)
        histories[index] = history
        for spec in history:
            unique_specs.setdefault(spec.frame_idx, spec)

    fpv_output_paths = {
        frame_idx: media_root / "fpv" / f"{frame_idx:06d}.jpg"
        for frame_idx in unique_specs
    }
    fpv_written = _extract_video_frames(
        ctx,
        unique_specs.values(),
        fpv_output_paths,
        image_width=int(indexed_rows[0][1].get("image_width") or 0),
        image_height=int(indexed_rows[0][1].get("image_height") or 0),
        jpeg_quality=jpeg_quality,
        overwrite=overwrite,
    )

    unique_bev_frames = sorted({spec.bev_frame_idx for history in histories.values() for spec in history if spec.bev_frame_idx is not None})
    bev_output_paths: dict[int, Path] = {}
    bev_copied = 0
    for bev_idx in unique_bev_frames:
        source = ctx.bev_paths[int(bev_idx)]
        dest = media_root / "bev" / source.name
        if overwrite or not dest.is_file():
            _copy_file(source, dest, overwrite=overwrite)
            bev_copied += 1
        bev_output_paths[int(bev_idx)] = dest

    processed: list[tuple[int, dict[str, Any]]] = []
    padded_rows = 0
    padded_frames = 0
    for index, row in indexed_rows:
        history = histories[index]
        out = json.loads(json.dumps(row, ensure_ascii=False))
        fpv_paths = [_manifest_path(fpv_output_paths[spec.frame_idx], manifest_dir=manifest_dir, absolute_paths=absolute_paths) for spec in history]
        bev_paths = [
            _manifest_path(bev_output_paths[int(spec.bev_frame_idx)], manifest_dir=manifest_dir, absolute_paths=absolute_paths)
            if spec.bev_frame_idx is not None
            else ""
            for spec in history
        ]
        current_fpv = fpv_paths[-1]
        current_bev = bev_paths[-1] if bev_paths else ""
        history_fpv_paths = fpv_paths[:-1]
        out["fpv_image_path"] = current_fpv
        out["bev_image_path"] = current_bev
        out["history_fpv_image_paths"] = history_fpv_paths
        out["history_size"] = len(history_fpv_paths)
        out["history_stride"] = max(1, int(round(ctx.source_fps / target_fps))) if target_fps > 0 else 1
        out["history_frame_indices"] = [spec.frame_idx for spec in history[:-1]]
        out["history_padding"] = [spec.padded for spec in history[:-1]]
        out["fpv_image_paths"] = fpv_paths
        out["bev_image_paths"] = bev_paths
        out["images"] = [*history_fpv_paths, current_fpv, current_bev] if current_bev else [*history_fpv_paths, current_fpv]
        out["fps"] = float(target_fps)
        out["frame_dt_sec"] = round(1.0 / float(target_fps), 6)
        out["coordinate_frame"] = "current_fpv"

        sidecar_paths = _copy_sidecars(out, output_root=output_root, manifest_dir=manifest_dir, absolute_paths=absolute_paths, overwrite=overwrite)
        for key, value in sidecar_paths.items():
            out[key] = value

        meta = dict(out.get("sidewalk_meta") or {})
        for key, value in sidecar_paths.items():
            meta[key] = value
        frame_meta = [
            {
                "order": spec.order,
                "requested_offset_sec": spec.requested_offset_sec,
                "requested_timestamp": spec.requested_timestamp,
                "frame_idx": spec.frame_idx,
                "video_idx": spec.video_idx,
                "timestamp": spec.timestamp,
                "fpv_time_delta_ms": spec.fpv_time_delta_ms,
                "bev_frame_idx": spec.bev_frame_idx,
                "bev_time_delta_ms": spec.bev_time_delta_ms,
                "padded": spec.padded,
            }
            for spec in history
        ]
        row_padded_frames = sum(1 for spec in history if spec.padded)
        padded_rows += int(row_padded_frames > 0)
        padded_frames += row_padded_frames
        meta.update(
            {
                "multiframe_schema_version": "frodobots_phase1_source_multiframe_v1",
                "multiframe_source_clip_id": ctx.clip_id,
                "multiframe_num_fpv_frames": len(history),
                "multiframe_num_history_fpv_frames": len(history_fpv_paths),
                "multiframe_num_images": len(out["images"]),
                "multiframe_target_fps": float(target_fps),
                "multiframe_source_fps": round(float(ctx.source_fps), 6),
                "multiframe_frame_dt_sec": round(1.0 / float(target_fps), 6),
                "multiframe_order": "history_fpv_oldest_to_newest,current_fpv,current_bev",
                "multiframe_bev_policy": "nearest_available_bev_for_metadata_current_bev_for_images",
                "multiframe_frames": frame_meta,
                "multiframe_padded_frame_count": row_padded_frames,
                "selfcontained_root": str(output_root),
            }
        )
        out["sidewalk_meta"] = meta
        processed.append((index, out))

    report = {
        "clip_id": clip_id,
        "rows": len(indexed_rows),
        "unique_fpv_frames": len(unique_specs),
        "fpv_frames_written": fpv_written,
        "unique_bev_frames": len(unique_bev_frames),
        "bev_frames_copied": bev_copied,
        "padded_rows": padded_rows,
        "padded_frames": padded_frames,
    }
    return processed, report


def build_multiframe_source(args: argparse.Namespace) -> dict[str, Any]:
    """Build the self-contained multi-frame source dataset.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Summary report.

    Raises:
        FileExistsError: If the output manifest exists and ``--overwrite`` is not set.
    """

    source_jsonl = args.source_jsonl.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    split = args.split_name or f"{source_jsonl.stem}_multiframe_{args.num_fpv_frames}f_{str(args.target_fps).replace('.', 'p')}hz"
    manifest_dir = output_root / "manifests"
    output_jsonl = manifest_dir / f"{safe_name(split)}.jsonl"
    report_path = manifest_dir / f"{safe_name(split)}_report.json"
    if output_jsonl.exists() and not args.overwrite:
        raise FileExistsError(f"Output manifest exists: {output_jsonl}; pass --overwrite to replace it")

    rows = read_jsonl(source_jsonl)
    if args.max_rows is not None:
        rows = rows[: max(int(args.max_rows), 0)]
    grouped: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(rows):
        clip_id = str(row.get("clip_id") or (row.get("sidewalk_meta") or {}).get("clip_id") or "")
        clip_dir = _clip_dir_from_row(row)
        key = (clip_id, str(clip_dir))
        if args.max_clips is not None and key not in grouped and len(grouped) >= int(args.max_clips):
            continue
        grouped.setdefault(key, []).append((index, row))

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    processed: list[tuple[int, dict[str, Any]]] = []
    clip_reports: list[dict[str, Any]] = []
    workers = max(int(args.num_workers), 1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _process_clip,
                key,
                indexed_rows,
                output_root=output_root,
                manifest_dir=manifest_dir,
                num_fpv_frames=int(args.num_fpv_frames),
                target_fps=float(args.target_fps),
                jpeg_quality=int(args.jpeg_quality),
                absolute_paths=bool(args.absolute_paths),
                overwrite=bool(args.overwrite),
            )
            for key, indexed_rows in grouped.items()
        ]
        for future in as_completed(futures):
            clip_rows, clip_report = future.result()
            processed.extend(clip_rows)
            clip_reports.append(clip_report)
            if len(clip_reports) % 50 == 0:
                print(f"[multiframe] processed clips: {len(clip_reports)}/{len(futures)}", flush=True)

    ordered_rows = [row for _, row in sorted(processed, key=lambda item: item[0])]
    write_jsonl(output_jsonl, ordered_rows)

    report = {
        "schema_version": "frodobots_phase1_source_multiframe_report_v1",
        "source_jsonl": str(source_jsonl),
        "output_root": str(output_root),
        "output_jsonl": str(output_jsonl),
        "rows_in": len(rows),
        "rows_out": len(ordered_rows),
        "clips": len(grouped),
        "num_fpv_frames": int(args.num_fpv_frames),
        "history_fpv_frames": int(args.num_fpv_frames) - 1,
        "target_fps": float(args.target_fps),
        "path_mode": "absolute" if args.absolute_paths else "relative_to_manifest_dir",
        "unique_fpv_frames": sum(int(item["unique_fpv_frames"]) for item in clip_reports),
        "fpv_frames_written": sum(int(item["fpv_frames_written"]) for item in clip_reports),
        "unique_bev_frames": sum(int(item["unique_bev_frames"]) for item in clip_reports),
        "bev_frames_copied": sum(int(item["bev_frames_copied"]) for item in clip_reports),
        "padded_rows": sum(int(item["padded_rows"]) for item in clip_reports),
        "padded_frames": sum(int(item["padded_frames"]) for item in clip_reports),
        "clip_reports": sorted(clip_reports, key=lambda item: str(item["clip_id"])),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    """Run the CLI converter.

    Returns:
        None.
    """

    args = parse_args()
    if int(args.num_fpv_frames) < 1:
        raise SystemExit("--num-fpv-frames must be >= 1")
    if float(args.target_fps) <= 0:
        raise SystemExit("--target-fps must be > 0")
    if not (1 <= int(args.jpeg_quality) <= 100):
        raise SystemExit("--jpeg-quality must be in [1, 100]")
    report = build_multiframe_source(args)
    printable = {k: v for k, v in report.items() if k != "clip_reports"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
