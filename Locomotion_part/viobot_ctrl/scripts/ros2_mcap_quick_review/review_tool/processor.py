from __future__ import annotations
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from .extractors import (
    choose_topic, extract_gps, extract_pose, header_stamp_ns,
    image_to_bgr, latlon_to_local_xy,
)
from .io_utils import atomic_write_json, read_json
from .rosbag_reader import RosbagReader
from .video_renderer import TimedFrame, TimedPoint, render_review_video

@dataclass
class RawFrame:
    header_ns: int | None
    record_ns: int
    path: str

@dataclass
class RawPoint:
    header_ns: int | None
    record_ns: int
    values: dict[str, Any]

def _allow_sample(last_ns: int | None, current_ns: int, max_fps: float) -> bool:
    if max_fps <= 0 or last_ns is None:
        return True
    return current_ns - last_ns >= int(1_000_000_000 / max_fps)

def _header_ratio(samples) -> float:
    if not samples:
        return 1.0
    return sum(item.header_ns is not None for item in samples) / len(samples)

def _header_interval(samples):
    values = sorted(item.header_ns for item in samples if item.header_ns is not None)
    return (int(values[0]), int(values[-1])) if len(values) >= 2 else None

def choose_time_source(requested: str, streams: list, cfg: dict):
    warnings = []
    available = [stream for stream in streams if stream]
    if requested == "record":
        return "record", warnings
    if requested == "header":
        return "header", warnings
    if not available:
        return "record", ["No samples available; using record time"]
    min_valid = float(cfg.get("minimum_header_valid_ratio", 0.8))
    ratios = [_header_ratio(stream) for stream in available]
    if any(ratio < min_valid for ratio in ratios):
        return "record", [
            "Auto timestamp selection fell back to record time because "
            f"header validity ratios were {ratios}"
        ]
    intervals = [x for stream in available if (x := _header_interval(stream)) is not None]
    if len(intervals) >= 2:
        common_start = max(start for start, _ in intervals)
        common_end = min(end for _, end in intervals)
        common = max(0, common_end - common_start)
        shortest = min(max(1, end - start) for start, end in intervals)
        overlap_ratio = common / shortest
        min_overlap = float(cfg.get("minimum_header_overlap_ratio", 0.5))
        if overlap_ratio < min_overlap:
            return "record", [
                "Auto timestamp selection fell back to record time because "
                f"header interval overlap ratio was {overlap_ratio:.3f}"
            ]
    return "header", warnings

def _stamp_ns(sample, source: str):
    return sample.header_ns if source == "header" else sample.record_ns

def _prepare_frames(samples, source, start_ns, end_ns):
    result = []
    for item in samples:
        stamp = _stamp_ns(item, source)
        if stamp is not None and start_ns <= stamp <= end_ns:
            result.append(TimedFrame((stamp - start_ns) / 1e9, item.path))
    return sorted(result, key=lambda item: item.t)

def _prepare_points(samples, source, start_ns, end_ns, role):
    valid = []
    for item in samples:
        stamp = _stamp_ns(item, source)
        if stamp is not None and start_ns <= stamp <= end_ns:
            valid.append((stamp, item.values))
    valid.sort(key=lambda x: x[0])
    if not valid:
        return []
    if role == "gps" and valid[0][1].get("mode") == "latlon":
        lons = np.asarray([float(v["x"]) for _, v in valid], dtype=float)
        lats = np.asarray([float(v["y"]) for _, v in valid], dtype=float)
        xs, ys = latlon_to_local_xy(lons, lats)
    else:
        xs = np.asarray([float(v["x"]) for _, v in valid], dtype=float)
        ys = np.asarray([float(v["y"]) for _, v in valid], dtype=float)
    result = []
    for idx, (stamp, values) in enumerate(valid):
        result.append(TimedPoint(
            t=(stamp - start_ns) / 1e9,
            x=float(xs[idx]), y=float(ys[idx]), z=float(values.get("z", 0.0)),
            qx=float(values.get("qx", 0.0)), qy=float(values.get("qy", 0.0)),
            qz=float(values.get("qz", 0.0)), qw=float(values.get("qw", 1.0)),
        ))
    return result

def _selected_interval(source, frames, gps, vio, max_duration_sec):
    warnings = []
    def stamps(samples):
        return sorted(
            stamp for item in samples
            if (stamp := _stamp_ns(item, source)) is not None
        )
    camera_stamps = stamps(frames)
    all_stamps = camera_stamps or (stamps(gps) + stamps(vio))
    if not all_stamps:
        now_ns = int(time.time() * 1e9)
        return now_ns, now_ns + int(5e9), ["No valid samples; placeholder duration used"]
    all_stamps.sort()
    start_ns, end_ns = (
        (camera_stamps[0], camera_stamps[-1])
        if camera_stamps
        else (all_stamps[0], all_stamps[-1])
    )
    if end_ns <= start_ns:
        end_ns = start_ns + int(5e9)
        warnings.append("Invalid/zero duration; placeholder duration used")
    max_ns = int(max_duration_sec * 1e9)
    if end_ns - start_ns > max_ns:
        end_ns = start_ns + max_ns
        warnings.append(f"Review interval clipped to {max_duration_sec:.1f} seconds")
    return start_ns, end_ns, warnings

def process_bag(row: dict[str, str], cfg: dict, force: bool = False) -> dict[str, Any]:
    bag_id = row["bag_id"]
    bag_path = Path(row["bag_path"])
    storage_id = row["storage_id"]
    cache_dir = Path(row["cache_dir"])
    output_video, info_path = cache_dir / "review.mp4", cache_dir / "info.json"

    if output_video.exists() and info_path.exists() and not force:
        info = read_json(info_path, {}) or {}
        selected = info.get("selected_topics", {})
        return {
            "bag_id": bag_id, "status": "ready", "error": "",
            "camera_topic": selected.get("camera") or "",
            "gps_topic": selected.get("gps") or "",
            "vio_topic": selected.get("vio") or "",
            "time_source": info.get("time_source", ""),
            "duration_sec": str(info.get("duration_sec", "")),
        }

    cache_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = cache_dir / "_frames"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    reader = RosbagReader(bag_path, storage_id, cfg.get("reader", {}))
    raw_frames, raw_gps, raw_vio = [], [], []
    warnings = []
    decode_errors = {"camera": 0, "gps": 0, "vio": 0}

    try:
        reader.open()
        topic_types = reader.topic_types
        camera_topic = choose_topic(topic_types, cfg["topics"]["camera"])
        gps_topic = choose_topic(topic_types, cfg["topics"]["gps"])
        vio_topic = choose_topic(topic_types, cfg["topics"]["vio"])
        if camera_topic is None:
            warnings.append("Camera topic not found")
        if gps_topic is None:
            warnings.append("GPS topic not found")
        if vio_topic is None:
            warnings.append("VIO topic not found")
        selected = {x for x in (camera_topic, gps_topic, vio_topic) if x is not None}

        pp = cfg["preprocess"]
        last_camera_record = last_gps_record = last_vio_record = None
        frame_index = 0

        for topic, msg, record_ns, type_name in reader.iter_messages(selected):
            h_ns = header_stamp_ns(msg)
            if topic == camera_topic:
                if not _allow_sample(last_camera_record, record_ns, float(pp.get("max_source_camera_fps", 15))):
                    continue
                try:
                    image = image_to_bgr(msg, type_name)
                    frame_path = temp_dir / f"{frame_index:06d}.jpg"
                    ok = cv2.imwrite(
                        str(frame_path), image,
                        [cv2.IMWRITE_JPEG_QUALITY, int(pp.get("jpeg_quality", 85))],
                    )
                    if not ok:
                        raise RuntimeError("cv2.imwrite returned false")
                    raw_frames.append(RawFrame(h_ns, record_ns, str(frame_path)))
                    frame_index += 1
                    last_camera_record = record_ns
                except Exception:
                    decode_errors["camera"] += 1
            elif topic == gps_topic:
                if not _allow_sample(last_gps_record, record_ns, float(pp.get("max_gps_fps", 10))):
                    continue
                try:
                    raw_gps.append(RawPoint(h_ns, record_ns, extract_gps(msg, cfg["topics"]["gps"])))
                    last_gps_record = record_ns
                except Exception:
                    decode_errors["gps"] += 1
            elif topic == vio_topic:
                if not _allow_sample(last_vio_record, record_ns, float(pp.get("max_vio_fps", 20))):
                    continue
                try:
                    raw_vio.append(RawPoint(h_ns, record_ns, extract_pose(msg, cfg["topics"]["vio"])))
                    last_vio_record = record_ns
                except Exception:
                    decode_errors["vio"] += 1

        time_source, time_warnings = choose_time_source(
            cfg["timestamps"]["source"], [raw_frames, raw_gps, raw_vio], cfg["timestamps"]
        )
        warnings.extend(time_warnings)
        start_ns, end_ns, interval_warnings = _selected_interval(
            time_source, raw_frames, raw_gps, raw_vio,
            float(pp.get("max_review_duration_sec", 180)),
        )
        warnings.extend(interval_warnings)
        duration_sec = (end_ns - start_ns) / 1e9
        frames = _prepare_frames(raw_frames, time_source, start_ns, end_ns)
        gps = _prepare_points(raw_gps, time_source, start_ns, end_ns, "gps")
        vio = _prepare_points(raw_vio, time_source, start_ns, end_ns, "vio")
        if not frames:
            warnings.append("No valid camera frames in selected interval")
        if not gps:
            warnings.append("No valid GPS points in selected interval")
        if not vio:
            warnings.append("No valid VIO points in selected interval")

        render_cfg = dict(pp)
        render_cfg["vio_heading_yaw_offset_deg"] = float(
            cfg["topics"]["vio"].get("heading_yaw_offset_deg", 0.0)
        )
        render_stats = render_review_video(
            output_video, frames, gps, vio, duration_sec, render_cfg,
            [
                f"Time source: {time_source} | Camera: {camera_topic or 'NONE'}",
                f"GPS: {gps_topic or 'NONE'} | VIO: {vio_topic or 'NONE'}",
            ],
        )
        info = {
            "bag_id": bag_id, "bag_path": str(bag_path), "storage_id": storage_id,
            "available_topics": topic_types,
            "selected_topics": {"camera": camera_topic, "gps": gps_topic, "vio": vio_topic},
            "time_source": time_source, "timeline_start_ns": start_ns,
            "timeline_end_ns": end_ns, "duration_sec": duration_sec,
            "sample_counts": {"camera": len(frames), "gps": len(gps), "vio": len(vio)},
            "decode_errors": decode_errors,
            "header_valid_ratio": {
                "camera": _header_ratio(raw_frames),
                "gps": _header_ratio(raw_gps),
                "vio": _header_ratio(raw_vio),
            },
            "vio_heading_yaw_offset_deg": render_cfg["vio_heading_yaw_offset_deg"],
            "warnings": warnings, "render": render_stats,
        }
        atomic_write_json(info_path, info)
        return {
            "bag_id": bag_id, "status": "ready", "error": "",
            "camera_topic": camera_topic or "", "gps_topic": gps_topic or "",
            "vio_topic": vio_topic or "", "time_source": time_source,
            "duration_sec": f"{duration_sec:.3f}",
        }
    except Exception as exc:
        return {
            "bag_id": bag_id, "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "camera_topic": "", "gps_topic": "", "vio_topic": "",
            "time_source": "", "duration_sec": "",
        }
    finally:
        reader.close()
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
