#!/usr/bin/env python3
"""Build vis_gt-style sidewalker inputs and BEV/FPV GT overlays from MCAPs.

This entry point intentionally follows the data alignment and GT coordinate
logic from the COCO formatter and vis_gt_multi_gt/nav_dataset references:

- use foxglove.CompressedVideo's embedded frame timestamp, never MCAP log_time;
- skip H264 samples before the first keyframe;
- align frames to ``timestamp - 150ms``;
- interpolate odometry by ``/odom.header.stamp``;
- convert quaternion order ``[w, x, y, z]`` with nav_dataset's formulas;
- rotate future positions with ``rotate_points_along_z(..., -yaw)``.

The generated output is split into:

    output_root/
      dataset_manifest.json
      output/<sequence>/
        bev/frame_000000_bev.jpg
        fpv/frame_000000_fpv.jpg
        videos/{bev,fpv}.mp4       # only with --make-videos
      input/<sequence>/             # only with --save-input
        front.mp4
        frame_times.npy
        frame_positions.npy
        frame_orientations.npy
        frame_gps.npy
        ...

For chunked MCAP data, omitting ``--output-root`` derives an ext4-friendly
layout under ``data_chunk``. For example, ``chunk_26/mcap`` writes rendered
frames to ``data_chunk/chunk26/output26``. The old ``input26`` directory is not
written unless ``--save-input`` is set.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    from .mcap_to_sidewalker_inputs import (
        CdrReader,
        GpsSample,
        ImageSample,
        Ros1Reader,
        _choose_optional_channel,
        _decode_chunk,
        _extract_jpeg_bytes,
        _iter_chunk_records,
        _iter_mcap_records,
        _parse_channel,
        _parse_message_record,
        _parse_schema,
        _read_protobuf_varint,
    )
    from .overlays import BEVMapOptions, GpsPoint, project_gps_route_to_bev_pixels, render_bev_map
except ImportError:  # Support direct execution from tools/sidewalker.
    from mcap_to_sidewalker_inputs import (  # type: ignore
        CdrReader,
        GpsSample,
        ImageSample,
        Ros1Reader,
        _choose_optional_channel,
        _decode_chunk,
        _extract_jpeg_bytes,
        _iter_chunk_records,
        _iter_mcap_records,
        _parse_channel,
        _parse_message_record,
        _parse_schema,
        _read_protobuf_varint,
    )
    from overlays import BEVMapOptions, GpsPoint, project_gps_route_to_bev_pixels, render_bev_map  # type: ignore


INTRINSIC_F = 790.0
INTRINSIC_WIDTH = 1920.0
INTRINSIC_HEIGHT = 1080.0
FISHEYE_K = 0.0
CAMERA_FORWARD_OFFSET_M = -0.1
HORIZONS = (200, 120, 60, 20)
HORIZON_COLORS_BGR = (
    (100, 230, 100),
    (100, 100, 255),
    (255, 100, 100),
    (100, 255, 255),
)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNK_OUTPUT_BASE = REPO_ROOT / "data_chunk"


@dataclass(frozen=True)
class BuildConfig:
    output_root: str
    input_dir_name: str
    output_dir_name: str
    image_topic: str
    odom_topic: str
    gps_topic: str
    frame_time_offset_ms: float
    odom_yaw_offset_deg: float
    bake_odom_yaw_offset_into_input: bool
    output_fps: float
    sample_stride: int
    target_output_fps: float
    save_input: bool
    frame_limit: int
    future_horizon: int
    bev_future_horizon: int
    camera_height_m: float
    camera_pitch_deg: float
    bev_output_size: int
    bev_zoom: int
    bev_mosaic_tiles: int
    bev_view_scale: float
    osm_cache_dir: str
    bev_arrow_every_px: float
    bev_arrow_size_px: float
    input_front_width: int
    input_front_height: int
    fpv_output_width: int
    fpv_output_height: int
    fpv_line_width: int
    fpv_point_radius: int
    fpv_arrow_size_px: float
    fpv_arrow_thickness: int
    fpv_arrow_offset_px: float
    fpv_waypoint_step: int
    make_videos: bool
    skip_existing: bool
    jpeg_quality: int
    progress_every: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcap-path", action="append", default=[], help="Specific MCAP path. Can be repeated.")
    parser.add_argument("--mcap-dir", default=None, help="Directory containing MCAP files for batch processing.")
    parser.add_argument("--sequence-name", action="append", default=[], help="MCAP stem under --mcap-dir. Can be repeated.")
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Root that receives input/output directories. If omitted for chunk data "
            "(for example an MCAP path under chunk_26), defaults to "
            f"{DEFAULT_CHUNK_OUTPUT_BASE}/chunk26."
        ),
    )
    parser.add_argument(
        "--chunk-output-base",
        default=str(DEFAULT_CHUNK_OUTPUT_BASE),
        help="Base directory for auto-derived chunk outputs when --output-root is omitted.",
    )
    parser.add_argument(
        "--chunk-name",
        default=None,
        help="Override the auto-derived chunk folder name, for example chunk26.",
    )
    parser.add_argument(
        "--input-dir-name",
        default=None,
        help="Name of the input directory under output root. Defaults to input26 for chunk26 auto layout, otherwise input.",
    )
    parser.add_argument(
        "--output-dir-name",
        default=None,
        help="Name of the output directory under output root. Defaults to output26 for chunk26 auto layout, otherwise output.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep processing remaining MCAPs when one sequence fails; failed items are recorded in dataset_manifest.json.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of MCAPs to process after filtering. 0 means no limit.")

    parser.add_argument("--image-topic", default="robot.h264_video.front")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--gps-topic", default="/acu_driver/gps_nav_topic")
    parser.add_argument("--frame-time-offset-ms", type=float, default=150.0)
    parser.add_argument(
        "--odom-yaw-offset-deg",
        type=float,
        default=0.0,
        help="Yaw offset applied when rendering FPV/BEV route prompts. Use 90 for go_ZJU if its odom frame is rotated.",
    )
    parser.add_argument(
        "--bake-odom-yaw-offset-into-input",
        action="store_true",
        help=(
            "Also write frame_orientations.npy with --odom-yaw-offset-deg baked into yaw. "
            "When enabled, FPV/BEV prompt rendering uses the baked orientation directly "
            "and does not apply the yaw offset a second time."
        ),
    )
    parser.add_argument("--output-fps", type=float, default=0.0, help="0 estimates source FPS from aligned frame timestamps.")
    parser.add_argument(
        "--target-output-fps",
        type=float,
        default=5.0,
        help="Target BEV/FPV render FPS when --sample-stride is 0. Default renders 5Hz outputs.",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=0,
        help="Render every Nth aligned frame. 0 auto-selects a stride from --target-output-fps.",
    )
    parser.add_argument(
        "--save-input",
        action="store_true",
        help="Also write the old input directory with front.mp4 and aligned numpy arrays. Default off.",
    )
    parser.add_argument("--frame-limit", type=int, default=0, help="Maximum rendered frames per sequence. 0 renders all available.")
    parser.add_argument("--future-horizon", type=int, default=200)
    parser.add_argument(
        "--bev-future-horizon",
        type=int,
        default=200,
        help="Future frames to draw on BEV. Use 0 to draw until the end of the clip.",
    )
    parser.add_argument("--camera-height-m", type=float, default=0.41)
    parser.add_argument(
        "--camera-pitch-deg",
        type=float,
        default=0.0,
        help="Camera pitch angle in degrees for FPV projection. Positive follows tools.sidewalker.overlays convention.",
    )

    parser.add_argument("--bev-output-size", type=int, default=900)
    parser.add_argument("--bev-zoom", type=int, default=19)
    parser.add_argument("--bev-mosaic-tiles", type=int, default=5)
    parser.add_argument("--bev-view-scale", type=float, default=1.0, help="Additional BEV projection scale after OSM zoom; >1 renders a smaller real-world area into the same BEV image.")
    parser.add_argument("--osm-cache-dir", default="/tmp/urban_nav_osm_tiles")
    parser.add_argument("--bev-arrow-every-px", type=float, default=60.0)
    parser.add_argument("--bev-arrow-size-px", type=float, default=14.0)
    parser.add_argument(
        "--input-front-width",
        type=int,
        default=0,
        help="Resize --save-input front.mp4 to this width. 0 keeps source width.",
    )
    parser.add_argument(
        "--input-front-height",
        type=int,
        default=0,
        help="Resize --save-input front.mp4 to this height. 0 keeps source height.",
    )

    parser.add_argument(
        "--fpv-output-width",
        type=int,
        default=0,
        help="Resize FPV prompt frames to this width before drawing. 0 keeps source width.",
    )
    parser.add_argument(
        "--fpv-output-height",
        type=int,
        default=0,
        help="Resize FPV prompt frames to this height before drawing. 0 keeps source height.",
    )
    parser.add_argument("--fpv-line-width", type=int, default=6)
    parser.add_argument("--fpv-point-radius", type=int, default=5)
    parser.add_argument("--fpv-arrow-size-px", type=float, default=22.0)
    parser.add_argument("--fpv-arrow-thickness", type=int, default=3)
    parser.add_argument("--fpv-arrow-offset-px", type=float, default=0.0)
    parser.add_argument("--fpv-waypoint-step", type=int, default=10)
    parser.add_argument("--jpeg-quality", type=int, default=92)

    parser.add_argument("--make-videos", action="store_true", help="Also write output/<sequence>/videos/bev.mp4 and fpv.mp4.")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def quaternions_to_euler_angles(quaternions: np.ndarray) -> np.ndarray:
    """nav_dataset.py conversion, expecting quaternion order [w, x, y, z]."""
    q = np.asarray(quaternions, dtype=np.float64).reshape(-1, 4)
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1.0, np.copysign(np.pi / 2.0, sinp), np.arcsin(sinp))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.column_stack([roll, pitch, yaw])


def euler_angles_to_quaternions(euler_angles: np.ndarray) -> np.ndarray:
    """Convert roll, pitch, yaw to quaternion order [w, x, y, z]."""
    angles = np.asarray(euler_angles, dtype=np.float64).reshape(-1, 3)
    roll = angles[:, 0]
    pitch = angles[:, 1]
    yaw = angles[:, 2]

    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.column_stack([w, x, y, z]).astype(np.float32)


def _apply_yaw_offset_to_orientations(orientations: np.ndarray, offset_deg: float) -> np.ndarray:
    if abs(float(offset_deg)) <= 1e-9:
        return np.asarray(orientations, dtype=np.float32)
    angles = quaternions_to_euler_angles(orientations)
    angles[:, 2] += np.deg2rad(float(offset_deg))
    return euler_angles_to_quaternions(angles)


def rotate_points_along_z(points: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """nav_dataset.py row-vector rotation."""
    pts = np.asarray(points, dtype=np.float32)
    ang = np.asarray(angle, dtype=np.float32).reshape(-1)
    if pts.ndim != 3:
        raise ValueError("points must have shape (B, N, C)")
    if pts.shape[0] != len(ang):
        raise ValueError("angle must have one value per batch")
    out = pts.copy()
    cosa = np.cos(ang)
    sina = np.sin(ang)
    for batch_idx in range(pts.shape[0]):
        rot2 = np.asarray(
            [[cosa[batch_idx], sina[batch_idx]], [-sina[batch_idx], cosa[batch_idx]]],
            dtype=np.float32,
        )
        out[batch_idx, :, :2] = pts[batch_idx, :, :2] @ rot2
    return out


def _skip_protobuf_value(payload: bytes, offset: int, wire_type: int) -> int:
    if wire_type == 0:
        _, offset = _read_protobuf_varint(payload, offset)
        return offset
    if wire_type == 1:
        return offset + 8
    if wire_type == 2:
        size, offset = _read_protobuf_varint(payload, offset)
        return offset + size
    if wire_type == 5:
        return offset + 4
    raise RuntimeError(f"unsupported protobuf wire type: {wire_type}")


def _parse_protobuf_timestamp(raw: bytes) -> int:
    offset = 0
    seconds = 0
    nanos = 0
    while offset < len(raw):
        key, offset = _read_protobuf_varint(raw, offset)
        field = key >> 3
        wire_type = key & 0x7
        if wire_type == 0:
            value, offset = _read_protobuf_varint(raw, offset)
            if field == 1:
                seconds = int(value)
            elif field == 2:
                nanos = int(value)
            continue
        offset = _skip_protobuf_value(raw, offset, wire_type)
    return int(seconds) * 1_000_000_000 + int(nanos)


def _parse_foxglove_compressed_video(payload: bytes) -> tuple[int, bytes, str]:
    offset = 0
    timestamp_ns: int | None = None
    data: bytes | None = None
    fmt = ""
    while offset < len(payload):
        key, offset = _read_protobuf_varint(payload, offset)
        field = key >> 3
        wire_type = key & 0x7
        if wire_type == 2:
            size, offset = _read_protobuf_varint(payload, offset)
            value = payload[offset : offset + size]
            offset += size
            if field == 1:
                timestamp_ns = _parse_protobuf_timestamp(value)
            elif field == 3:
                data = value
            elif field == 6:
                fmt = value.decode("utf-8", "replace")
            continue
        offset = _skip_protobuf_value(payload, offset, wire_type)
    if data is None:
        raise RuntimeError("foxglove.CompressedVideo payload has no data field")
    if timestamp_ns is None:
        raise RuntimeError("foxglove.CompressedVideo payload has no embedded timestamp")
    return int(timestamp_ns), data, fmt


def _parse_compressed_image_timestamp(payload: bytes, *, message_encoding: str) -> int:
    if message_encoding == "ros1":
        reader = Ros1Reader(payload)
        _ = reader.uint32()  # header.seq
        sec = reader.uint32()
        nsec = reader.uint32()
        return int(sec) * 1_000_000_000 + int(nsec)

    reader = CdrReader(payload)
    sec = reader.int32()
    nsec = reader.uint32()
    return int(sec) * 1_000_000_000 + int(nsec)


def _h264_nal_types(data: bytes) -> list[int]:
    nal_types: list[int] = []
    idx = 0
    size = len(data)
    while idx < size - 4:
        start: int | None = None
        if data[idx : idx + 3] == b"\x00\x00\x01":
            start = idx + 3
        elif data[idx : idx + 4] == b"\x00\x00\x00\x01":
            start = idx + 4
        if start is None:
            idx += 1
            continue
        nal_types.append(int(data[start]) & 0x1F)
        idx = start + 1
    return nal_types


def _is_h264_keyframe(data: bytes) -> bool:
    return any(nal_type in {5, 7, 8} for nal_type in _h264_nal_types(data))


def _parse_ros1_odom(payload: bytes) -> tuple[int, list[float]]:
    reader = Ros1Reader(payload)
    _ = reader.uint32()
    sec = reader.uint32()
    nsec = reader.uint32()
    _ = reader.string()
    _ = reader.string()
    x_m = reader.float64()
    y_m = reader.float64()
    z_m = reader.float64()
    qx = reader.float64()
    qy = reader.float64()
    qz = reader.float64()
    qw = reader.float64()
    reader.skip_float64(36)
    linear_x = reader.float64()
    linear_y = reader.float64()
    linear_z = reader.float64()
    angular_x = reader.float64()
    angular_y = reader.float64()
    angular_z = reader.float64()
    timestamp_ns = int(sec) * 1_000_000_000 + int(nsec)
    return timestamp_ns, [
        x_m,
        y_m,
        z_m,
        qw,
        qx,
        qy,
        qz,
        linear_x,
        linear_y,
        linear_z,
        angular_x,
        angular_y,
        angular_z,
    ]


def _parse_cdr_odom(payload: bytes) -> tuple[int, list[float]]:
    reader = CdrReader(payload)
    sec = reader.int32()
    nsec = reader.uint32()
    _ = reader.string()
    _ = reader.string()
    x_m = reader.float64()
    y_m = reader.float64()
    z_m = reader.float64()
    qx = reader.float64()
    qy = reader.float64()
    qz = reader.float64()
    qw = reader.float64()
    reader.offset += 8 * 36
    linear_x = reader.float64()
    linear_y = reader.float64()
    linear_z = reader.float64()
    angular_x = reader.float64()
    angular_y = reader.float64()
    angular_z = reader.float64()
    timestamp_ns = int(sec) * 1_000_000_000 + int(nsec)
    return timestamp_ns, [
        x_m,
        y_m,
        z_m,
        qw,
        qx,
        qy,
        qz,
        linear_x,
        linear_y,
        linear_z,
        angular_x,
        angular_y,
        angular_z,
    ]


def _parse_ros1_navsatfix(payload: bytes, *, source_topic: str) -> GpsSample:
    reader = Ros1Reader(payload)
    _ = reader.uint32()
    sec = reader.uint32()
    nsec = reader.uint32()
    _ = reader.string()
    status = reader.int8()
    _ = reader.uint16()
    lat = reader.float64()
    lng = reader.float64()
    altitude = reader.float64()
    return GpsSample(
        timestamp_ns=int(sec) * 1_000_000_000 + int(nsec),
        lat=float(lat),
        lng=float(lng),
        altitude_m=float(altitude),
        status=int(status),
        source_topic=source_topic,
    )


def _parse_cdr_navsatfix(payload: bytes, *, source_topic: str) -> GpsSample:
    reader = CdrReader(payload)
    sec = reader.int32()
    nsec = reader.uint32()
    _ = reader.string()
    status = reader.int8()
    _ = reader.uint16()
    lat = reader.float64()
    lng = reader.float64()
    altitude = reader.float64()
    return GpsSample(
        timestamp_ns=int(sec) * 1_000_000_000 + int(nsec),
        lat=float(lat),
        lng=float(lng),
        altitude_m=float(altitude),
        status=int(status),
        source_topic=source_topic,
    )


def _decode_h264_frames(encoded_samples: list[ImageSample]) -> list[ImageSample]:
    if not encoded_samples:
        return []
    if cv2 is None:
        raise RuntimeError("OpenCV is required to decode H264")
    with tempfile.NamedTemporaryFile(suffix=".h264") as stream:
        for sample in encoded_samples:
            stream.write(sample.data)
        stream.flush()
        capture = cv2.VideoCapture(stream.name)
        decoded: list[ImageSample] = []
        for sample in encoded_samples:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            ok, encoded = cv2.imencode(".jpg", frame_bgr)
            if ok:
                decoded.append(
                    ImageSample(
                        timestamp_ns=int(sample.timestamp_ns),
                        data=encoded.tobytes(),
                        source_topic=sample.source_topic,
                        encoding="jpeg",
                    )
                )
        capture.release()
    if not decoded:
        raise RuntimeError("No H264 frames decoded")
    return decoded


def _estimate_fps(timestamps_ns: Sequence[int | float]) -> float:
    if len(timestamps_ns) < 2:
        return 20.0
    deltas = np.diff(np.asarray(timestamps_ns, dtype=np.float64)) / 1e9
    deltas = deltas[np.isfinite(deltas) & (deltas > 1e-6)]
    return float(1.0 / np.median(deltas)) if len(deltas) else 20.0


def _interp_columns(target_times: np.ndarray, sample_times: np.ndarray, values: np.ndarray) -> np.ndarray:
    order = np.argsort(sample_times)
    sample_times = np.asarray(sample_times, dtype=np.float64)[order]
    values = np.asarray(values, dtype=np.float64)[order]
    return np.stack([np.interp(target_times, sample_times, values[:, col]) for col in range(values.shape[1])], axis=1)


def _get_aligned_data(mcap_path: Path, config: BuildConfig) -> dict[str, Any]:
    schemas: dict[int, Any] = {}
    channels: dict[int, Any] = {}
    for opcode, payload in _iter_mcap_records(mcap_path):
        if opcode == 3:
            schema = _parse_schema(payload)
            schemas[schema.schema_id] = schema
        elif opcode == 4:
            channel = _parse_channel(payload, schemas)
            channels[channel.channel_id] = channel

    image_channel = _choose_optional_channel(
        channels,
        requested_topic=config.image_topic,
        schema_substrings=["CompressedImage", "CompressedVideo"],
        role="image",
    )
    odom_channel = _choose_optional_channel(
        channels,
        requested_topic=config.odom_topic,
        schema_substrings=["Odometry"],
        role="odometry",
    )
    gps_channel = _choose_optional_channel(
        channels,
        requested_topic=config.gps_topic,
        schema_substrings=["NavSatFix"],
        role="gps",
    )
    if image_channel is None or odom_channel is None or gps_channel is None:
        raise RuntimeError(f"Missing required image/odom/gps channel in {mcap_path}")

    encoded_frames: list[ImageSample] = []
    pose_times: list[int] = []
    poses: list[list[float]] = []
    gps_times: list[int] = []
    gps_values: list[list[float]] = []
    raw_counts: dict[str, int] = {}
    found_keyframe = False
    skipped_until_keyframe = 0

    for opcode, payload in _iter_mcap_records(mcap_path):
        if opcode != 6:
            continue
        chunk = _decode_chunk(payload)
        for inner_opcode, inner_payload in _iter_chunk_records(chunk):
            if inner_opcode != 5:
                continue
            channel_id, log_timestamp_ns, message_payload = _parse_message_record(inner_payload)
            channel = channels.get(channel_id)
            if channel is None:
                continue
            raw_counts[channel.topic] = raw_counts.get(channel.topic, 0) + 1
            if channel_id == image_channel.channel_id:
                try:
                    if "CompressedVideo" in channel.schema_name:
                        image_ts, data, _ = _parse_foxglove_compressed_video(message_payload)
                        if not found_keyframe:
                            if not _is_h264_keyframe(data):
                                skipped_until_keyframe += 1
                                continue
                            found_keyframe = True
                        encoded_frames.append(
                            ImageSample(timestamp_ns=image_ts, data=data, source_topic=channel.topic, encoding="h264")
                        )
                    else:
                        image_ts = _parse_compressed_image_timestamp(
                            message_payload,
                            message_encoding=channel.message_encoding,
                        )
                        encoded_frames.append(
                            ImageSample(
                                timestamp_ns=image_ts,
                                data=_extract_jpeg_bytes(message_payload),
                                source_topic=channel.topic,
                                encoding="jpeg",
                            )
                        )
                except Exception:
                    continue
            elif channel_id == odom_channel.channel_id:
                try:
                    pose_ts, pose = _parse_ros1_odom(message_payload) if channel.message_encoding == "ros1" else _parse_cdr_odom(message_payload)
                    pose_times.append(int(pose_ts))
                    poses.append(pose)
                except Exception:
                    continue
            elif channel_id == gps_channel.channel_id:
                try:
                    gps = (
                        _parse_ros1_navsatfix(message_payload, source_topic=channel.topic)
                        if channel.message_encoding == "ros1"
                        else _parse_cdr_navsatfix(message_payload, source_topic=channel.topic)
                    )
                    if math.isfinite(gps.lat) and math.isfinite(gps.lng):
                        gps_times.append(int(gps.timestamp_ns))
                        gps_values.append([gps.lat, gps.lng, gps.altitude_m if gps.altitude_m is not None else 0.0])
                except Exception:
                    continue

    encoded_frames = sorted(encoded_frames, key=lambda sample: sample.timestamp_ns)
    if not encoded_frames or not poses or not gps_values:
        raise RuntimeError(
            "No usable image, odometry, or GPS samples decoded from "
            f"{mcap_path}; raw_counts={raw_counts}; "
            f"usable_image_frames={len(encoded_frames)}; "
            f"usable_odom_samples={len(poses)}; "
            f"usable_gps_samples={len(gps_values)}; "
            f"skipped_until_keyframe={skipped_until_keyframe}"
        )

    decoded_images = _decode_h264_frames(encoded_frames) if encoded_frames[0].encoding == "h264" else encoded_frames
    usable = min(len(encoded_frames), len(decoded_images))
    encoded_frames = encoded_frames[:usable]
    decoded_images = decoded_images[:usable]

    raw_frame_times = np.asarray([sample.timestamp_ns for sample in encoded_frames], dtype=np.float64)
    aligned_frame_times = raw_frame_times - float(config.frame_time_offset_ms) * 1e6
    pose_interp = _interp_columns(
        aligned_frame_times,
        np.asarray(pose_times, dtype=np.float64),
        np.asarray(poses, dtype=np.float64),
    )
    gps_interp = _interp_columns(
        aligned_frame_times,
        np.asarray(gps_times, dtype=np.float64),
        np.asarray(gps_values, dtype=np.float64),
    )

    frame_speeds = np.linalg.norm(pose_interp[:, 7:10], axis=1).astype(np.float32)
    frame_orientations = pose_interp[:, 3:7].astype(np.float32)
    baked_yaw_offset_deg = 0.0
    if bool(config.bake_odom_yaw_offset_into_input):
        baked_yaw_offset_deg = float(config.odom_yaw_offset_deg)
        frame_orientations = _apply_yaw_offset_to_orientations(frame_orientations, baked_yaw_offset_deg)

    meta: dict[str, Any] = {
        "mcap_path": str(mcap_path),
        "sequence": mcap_path.stem,
        "image_topic": image_channel.topic,
        "image_schema": image_channel.schema_name,
        "odom_topic": odom_channel.topic,
        "odom_schema": odom_channel.schema_name,
        "gps_topic": gps_channel.topic,
        "gps_schema": gps_channel.schema_name,
        "raw_counts": raw_counts,
        "encoded_frames": int(len(encoded_frames)),
        "decoded_frames": int(len(decoded_images)),
        "odom_samples": int(len(poses)),
        "gps_samples": int(len(gps_values)),
        "skipped_until_keyframe": int(skipped_until_keyframe),
        "frame_time_offset_ms": float(config.frame_time_offset_ms),
        "frame_orientation_yaw_offset_baked_deg": float(baked_yaw_offset_deg),
        "alignment": "image message timestamp - frame_time_offset_ms, odom/GPS header.stamp; MCAP log_time is not used",
        "quaternion_order": "[w, x, y, z]",
    }
    return {
        "encoded_frames": encoded_frames,
        "decoded_images": decoded_images,
        "raw_frame_times": raw_frame_times.astype(np.float64),
        "frame_times": aligned_frame_times.astype(np.float64),
        "frame_speeds": frame_speeds,
        "frame_positions": pose_interp[:, :3].astype(np.float32),
        "frame_orientations": frame_orientations,
        "frame_linear_velocities": pose_interp[:, 7:10].astype(np.float32),
        "frame_angular_velocities": pose_interp[:, 10:13].astype(np.float32),
        "frame_gps": gps_interp.astype(np.float64),
        "meta": meta,
    }


def _write_input_clip(data: dict[str, Any], input_dir: Path, fps: float, config: BuildConfig) -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is required")
    input_dir.mkdir(parents=True, exist_ok=True)
    encoded_frames: list[ImageSample] = data["encoded_frames"]
    decoded_images: list[ImageSample] = data["decoded_images"]

    if encoded_frames and encoded_frames[0].encoding == "h264":
        with (input_dir / "front.h264").open("wb") as stream:
            for sample in encoded_frames:
                stream.write(sample.data)

    first_rgb = np.asarray(Image.open(io.BytesIO(decoded_images[0].data)).convert("RGB"))
    if int(config.input_front_width) > 0 and int(config.input_front_height) > 0:
        first_rgb = cv2.resize(
            first_rgb,
            (int(config.input_front_width), int(config.input_front_height)),
            interpolation=cv2.INTER_AREA,
        )
    height, width = first_rgb.shape[:2]
    writer = cv2.VideoWriter(str(input_dir / "front.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (int(width), int(height)))
    for sample in decoded_images:
        frame_rgb = np.asarray(Image.open(io.BytesIO(sample.data)).convert("RGB"))
        if int(config.input_front_width) > 0 and int(config.input_front_height) > 0:
            frame_rgb = cv2.resize(
                frame_rgb,
                (int(config.input_front_width), int(config.input_front_height)),
                interpolation=cv2.INTER_AREA,
            )
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    writer.release()

    keys = [
        "raw_frame_times",
        "frame_times",
        "frame_speeds",
        "frame_positions",
        "frame_orientations",
        "frame_linear_velocities",
        "frame_angular_velocities",
        "frame_gps",
    ]
    for key in keys:
        np.save(input_dir / f"{key}.npy", np.asarray(data[key]))

    manifest = dict(data["meta"])
    manifest.update(
        {
            "front_mp4": str(input_dir / "front.mp4"),
            "front_h264": str(input_dir / "front.h264") if (input_dir / "front.h264").exists() else None,
            "front_width": int(width),
            "front_height": int(height),
            "source_fps": float(fps),
            "render_stride": int(config.sample_stride),
            "target_output_fps": float(config.target_output_fps),
        }
    )
    (input_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")


def _heading_deg_from_yaw_rad(yaw_rad: float) -> float:
    return (90.0 - math.degrees(float(yaw_rad))) % 360.0


def _draw_route_arrows_rgb(
    image_rgb: Image.Image,
    route_px: np.ndarray,
    *,
    every_px: float,
    size_px: float,
    color_rgb: tuple[int, int, int],
) -> Image.Image:
    if cv2 is None:
        return image_rgb
    points = np.asarray(route_px, dtype=np.float32).reshape(-1, 2)
    if len(points) < 2:
        return image_rgb
    arr = np.asarray(image_rgb.convert("RGB")).copy()
    height, width = arr.shape[:2]
    next_at = max(float(every_px), 10.0)
    travelled = 0.0
    for p0, p1 in zip(points[:-1], points[1:]):
        if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
            continue
        segment = p1 - p0
        seg_len = float(np.linalg.norm(segment))
        if seg_len < 1e-6:
            continue
        while travelled + seg_len >= next_at:
            ratio = (next_at - travelled) / seg_len
            center = p0 + ratio * segment
            if 0 <= center[0] < width and 0 <= center[1] < height:
                direction = segment / seg_len
                _draw_chevron(
                    arr,
                    center,
                    direction,
                    size_px=float(size_px),
                    color=color_rgb,
                    thickness=2,
                    outline_color=(255, 255, 255),
                    outline_thickness=5,
                )
            next_at += max(float(every_px), 10.0)
        travelled += seg_len
    return Image.fromarray(arr)


def _draw_chevron(
    image: np.ndarray,
    center: np.ndarray,
    direction: np.ndarray,
    *,
    size_px: float,
    color: tuple[int, int, int],
    thickness: int,
    outline_color: tuple[int, int, int] | None = None,
    outline_thickness: int = 0,
) -> None:
    direction = np.asarray(direction, dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return
    direction = direction / norm
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    size = max(float(size_px), 2.0)
    tip = center + direction * size * 0.45
    left = center - direction * size * 0.32 + normal * size * 0.58
    right = center - direction * size * 0.32 - normal * size * 0.58
    tip_i = tuple(np.round(tip).astype(int))
    left_i = tuple(np.round(left).astype(int))
    right_i = tuple(np.round(right).astype(int))
    if outline_color is not None and int(outline_thickness) > 0:
        cv2.line(image, left_i, tip_i, outline_color, max(int(outline_thickness), 1), cv2.LINE_AA)
        cv2.line(image, right_i, tip_i, outline_color, max(int(outline_thickness), 1), cv2.LINE_AA)
    cv2.line(image, left_i, tip_i, color, max(int(thickness), 1), cv2.LINE_AA)
    cv2.line(image, right_i, tip_i, color, max(int(thickness), 1), cv2.LINE_AA)


def _project_waypoints_center_fisheye(
    waypoints_xy: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    camera_height_m: float,
    camera_pitch_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(waypoints_xy, dtype=np.float64).reshape(-1, 2)
    projected = np.full((len(points), 2), np.nan, dtype=np.float32)
    in_front = np.zeros((len(points),), dtype=bool)
    if len(points) == 0:
        return projected, in_front

    ego_points = np.zeros((len(points), 3), dtype=np.float64)
    ego_points[:, :2] = points[:, :2]
    cam_translation = np.asarray([CAMERA_FORWARD_OFFSET_M, 0.0, float(camera_height_m)], dtype=np.float64)
    rotation = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    cam_points = (rotation @ (ego_points - cam_translation).T).T
    if abs(float(camera_pitch_deg)) > 1e-6:
        pitch = -math.radians(float(camera_pitch_deg))
        cp, sp = math.cos(pitch), math.sin(pitch)
        y_cam = cam_points[:, 1].copy()
        z_cam = cam_points[:, 2].copy()
        cam_points[:, 1] = cp * y_cam - sp * z_cam
        cam_points[:, 2] = sp * y_cam + cp * z_cam
    in_front = cam_points[:, 2] > 1e-6
    if not np.any(in_front):
        return projected, in_front

    fx = INTRINSIC_F * (float(image_width) / INTRINSIC_WIDTH)
    fy = INTRINSIC_F * (float(image_height) / INTRINSIC_HEIGHT)
    cx = float(image_width) / 2.0
    cy = float(image_height) / 2.0
    valid_points = cam_points[in_front]
    x = valid_points[:, 0] / valid_points[:, 2]
    y = valid_points[:, 1] / valid_points[:, 2]
    r = np.sqrt(x * x + y * y)
    theta = np.arctan(r)
    theta_d = theta * (1.0 + FISHEYE_K * theta * theta)
    scaling = np.ones_like(r)
    mask = r > 1e-8
    scaling[mask] = theta_d[mask] / r[mask]
    u = fx * x * scaling + cx
    v = fy * y * scaling + cy
    projected[in_front] = np.column_stack((u, v)).astype(np.float32)
    return projected, in_front


def _point_inside(point: np.ndarray, width: int, height: int) -> bool:
    return bool(0.0 <= float(point[0]) < float(width) and 0.0 <= float(point[1]) < float(height))


def _valid_inside_mask(projected: np.ndarray, in_front: np.ndarray, width: int, height: int) -> np.ndarray:
    points = np.asarray(projected, dtype=np.float32)
    finite = np.isfinite(points).all(axis=1)
    inside = (
        (points[:, 0] >= 0.0)
        & (points[:, 0] < float(width))
        & (points[:, 1] >= 0.0)
        & (points[:, 1] < float(height))
    )
    return np.asarray(in_front, dtype=bool) & finite & inside


def _iter_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(np.asarray(mask, dtype=bool)):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            if idx - start >= 2:
                runs.append((start, idx))
            start = None
    if start is not None and len(mask) - start >= 2:
        runs.append((start, len(mask)))
    return runs


def _draw_polyline_runs(
    img_bgr: np.ndarray,
    points: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    width: int,
) -> None:
    if cv2 is None:
        return
    points = np.asarray(points, dtype=np.float32)
    for start, end in _iter_true_runs(mask):
        polyline = np.round(points[start:end]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img_bgr, [polyline], False, color_bgr, max(int(width), 1), cv2.LINE_AA)


def _draw_fpv_path_lines(
    img_bgr: np.ndarray,
    projected: np.ndarray,
    in_front: np.ndarray,
    *,
    max_horizon: int,
    line_width: int,
) -> np.ndarray:
    height, width = img_bgr.shape[:2]
    points = np.asarray(projected, dtype=np.float32)
    usable = min(len(points), int(max_horizon) + 1)
    if usable < 2:
        return np.zeros((len(points),), dtype=bool)
    valid_mask = _valid_inside_mask(points, in_front, width, height)
    segment_start = 0
    for horizon, color in zip(reversed(HORIZONS), reversed(HORIZON_COLORS_BGR)):
        horizon_end = min(usable, int(horizon) + 1)
        if horizon_end - segment_start < 2:
            continue
        _draw_polyline_runs(
            img_bgr,
            points[segment_start:horizon_end],
            valid_mask[segment_start:horizon_end],
            color,
            int(line_width),
        )
        segment_start = min(int(horizon), usable - 1)
    return valid_mask


def _horizon_color_for_index(index: int) -> tuple[int, int, int]:
    for horizon, color in zip(reversed(HORIZONS), reversed(HORIZON_COLORS_BGR)):
        if int(index) <= int(horizon):
            return color
    return HORIZON_COLORS_BGR[0]


def _path_anchor_between_indices(
    points: np.ndarray,
    valid_mask: np.ndarray,
    *,
    start_idx: int,
    end_idx: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    points = np.asarray(points, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    start_idx = max(0, min(int(start_idx), len(points) - 1))
    end_idx = max(start_idx + 1, min(int(end_idx), len(points) - 1))
    segments: list[tuple[np.ndarray, np.ndarray, float]] = []
    total = 0.0
    for idx in range(start_idx, end_idx):
        if not (bool(valid_mask[idx]) and bool(valid_mask[idx + 1])):
            continue
        p0 = points[idx]
        p1 = points[idx + 1]
        if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
            continue
        seg = p1 - p0
        length = float(np.linalg.norm(seg))
        if length < 1e-6:
            continue
        segments.append((p0, p1, length))
        total += length
    if not segments or total < 1e-6:
        return None
    target = total * 0.5
    walked = 0.0
    for p0, p1, length in segments:
        if walked + length >= target:
            ratio = (target - walked) / length
            center = p0 + (p1 - p0) * float(ratio)
            direction = (p1 - p0) / length
            return center.astype(np.float32), direction.astype(np.float32)
        walked += length
    p0, p1, length = segments[-1]
    return ((p0 + p1) * 0.5).astype(np.float32), ((p1 - p0) / length).astype(np.float32)


def _draw_fpv_waypoint_markers(
    img_bgr: np.ndarray,
    projected: np.ndarray,
    in_front: np.ndarray,
    *,
    step: int,
    point_radius: int,
    arrow_size_px: float,
    arrow_thickness: int,
    arrow_offset_px: float,
) -> None:
    if cv2 is None:
        return
    height, width = img_bgr.shape[:2]
    points = np.asarray(projected, dtype=np.float32)
    step = max(int(step), 1)
    valid_mask = _valid_inside_mask(points, in_front, width, height)
    selected = [idx for idx in range(1, len(points)) if bool(valid_mask[idx]) and idx % step == 0]
    valid_inside = [idx for idx in range(1, len(points)) if bool(valid_mask[idx])]
    if valid_inside and (not selected or selected[-1] != valid_inside[-1]):
        selected.append(valid_inside[-1])

    arrow_ranges = (
        (0, 20, HORIZON_COLORS_BGR[3]),
        (20, 60, HORIZON_COLORS_BGR[2]),
        (60, 120, HORIZON_COLORS_BGR[1]),
        (120, 200, HORIZON_COLORS_BGR[0]),
    )
    for start_idx, end_idx, color in arrow_ranges:
        if start_idx >= len(points) - 1:
            continue
        anchor_direction = _path_anchor_between_indices(points, valid_mask, start_idx=start_idx, end_idx=end_idx)
        if anchor_direction is None:
            continue
        anchor, direction = anchor_direction
        if abs(float(arrow_offset_px)) > 1e-6:
            normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
            anchor = anchor + normal * float(arrow_offset_px)
            alt_anchor = anchor - normal * float(arrow_offset_px) * 2.0
        else:
            alt_anchor = anchor
        if not _point_inside(anchor, width, height) and _point_inside(alt_anchor, width, height):
            anchor = alt_anchor
        _draw_chevron(
            img_bgr,
            anchor,
            direction,
            size_px=float(arrow_size_px),
            color=color,
            thickness=max(int(arrow_thickness), 1),
        )

    if not selected:
        return
    for idx in selected:
        p = points[idx]
        color = _horizon_color_for_index(idx)
        center = tuple(np.round(p).astype(int))
        cv2.circle(img_bgr, center, max(int(point_radius) + 2, 3), (245, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(img_bgr, center, max(int(point_radius), 1), color, -1, cv2.LINE_AA)


def _draw_fpv_frame(
    frame_rgb: np.ndarray,
    *,
    positions: np.ndarray,
    angles_all: np.ndarray,
    frame_index: int,
    config: BuildConfig,
) -> np.ndarray:
    img_bgr = cv2.cvtColor(np.asarray(frame_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    height, width = img_bgr.shape[:2]
    max_horizon = int(config.future_horizon)
    full_end = min(len(positions), frame_index + max_horizon + 1)
    if full_end - frame_index < 2:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    full_waypoints = positions[frame_index:full_end, :2].copy()
    full_waypoints = full_waypoints - full_waypoints[0:1]
    full_local = rotate_points_along_z(full_waypoints.reshape(1, -1, 2), -angles_all[frame_index : frame_index + 1, 2])[0]
    full_projected, full_in_front = _project_waypoints_center_fisheye(
        full_local,
        image_width=width,
        image_height=height,
        camera_height_m=float(config.camera_height_m),
        camera_pitch_deg=float(config.camera_pitch_deg),
    )

    _draw_fpv_path_lines(
        img_bgr,
        full_projected,
        full_in_front,
        max_horizon=max_horizon,
        line_width=int(config.fpv_line_width),
    )

    _draw_fpv_waypoint_markers(
        img_bgr,
        full_projected,
        full_in_front,
        step=int(config.fpv_waypoint_step),
        point_radius=int(config.fpv_point_radius),
        arrow_size_px=float(config.fpv_arrow_size_px),
        arrow_thickness=int(config.fpv_arrow_thickness),
        arrow_offset_px=float(config.fpv_arrow_offset_px),
    )
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _draw_bev_frame(
    *,
    frame_index: int,
    frame_gps: np.ndarray,
    angles_all: np.ndarray,
    config: BuildConfig,
) -> Image.Image:
    current_lat, current_lng = float(frame_gps[frame_index, 0]), float(frame_gps[frame_index, 1])
    heading_deg = _heading_deg_from_yaw_rad(float(angles_all[frame_index, 2]))
    if int(config.bev_future_horizon) <= 0:
        end = len(frame_gps)
    else:
        end = min(len(frame_gps), frame_index + int(config.bev_future_horizon) + 1)
    route = [GpsPoint(lat=float(lat), lng=float(lng)) for lat, lng in frame_gps[frame_index:end, :2]]
    options = BEVMapOptions(
        output_size=int(config.bev_output_size),
        zoom=int(config.bev_zoom),
        mosaic_tiles=int(config.bev_mosaic_tiles),
        view_scale=float(config.bev_view_scale),
        cache_dir=Path(config.osm_cache_dir).expanduser(),
        heading_convention="compass",
        draw_trajectory=True,
    )
    image = render_bev_map(
        current_gps=GpsPoint(lat=current_lat, lng=current_lng),
        heading_deg=heading_deg,
        route_gps=route,
        options=options,
    )
    try:
        route_px, _, _ = project_gps_route_to_bev_pixels(
            current_gps=GpsPoint(lat=current_lat, lng=current_lng),
            heading_deg=heading_deg,
            route_gps=route,
            options=options,
        )
        image = _draw_route_arrows_rgb(
            image,
            route_px,
            every_px=float(config.bev_arrow_every_px),
            size_px=float(config.bev_arrow_size_px),
            color_rgb=(14, 165, 233),
        )
    except Exception:
        pass
    return image


def _save_jpeg_rgb(path: Path, image_rgb: np.ndarray | Image.Image, *, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image_rgb, Image.Image):
        image_rgb.save(path, quality=int(quality))
    else:
        Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).save(path, quality=int(quality))


def _open_video_writer(path: Path, frame_rgb: np.ndarray | Image.Image, fps: float) -> Any:
    if cv2 is None:
        raise RuntimeError("OpenCV is required")
    arr = np.asarray(frame_rgb.convert("RGB") if isinstance(frame_rgb, Image.Image) else frame_rgb, dtype=np.uint8)
    height, width = arr.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (int(width), int(height)))


def _write_video_frame(writer: Any, frame_rgb: np.ndarray | Image.Image) -> None:
    arr = np.asarray(frame_rgb.convert("RGB") if isinstance(frame_rgb, Image.Image) else frame_rgb, dtype=np.uint8)
    writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def _render_outputs(data: dict[str, Any], output_dir: Path, fps: float, config: BuildConfig) -> dict[str, Any]:
    decoded_images: list[ImageSample] = data["decoded_images"]
    raw_frame_times = np.asarray(data["raw_frame_times"], dtype=np.float64)
    frame_times = np.asarray(data["frame_times"], dtype=np.float64)
    positions = np.asarray(data["frame_positions"], dtype=np.float32)
    orientations = np.asarray(data["frame_orientations"], dtype=np.float32)
    frame_gps = np.asarray(data["frame_gps"], dtype=np.float64)
    angles_all = quaternions_to_euler_angles(orientations)
    if not bool(config.bake_odom_yaw_offset_into_input) and abs(float(config.odom_yaw_offset_deg)) > 1e-9:
        angles_all = angles_all.copy()
        angles_all[:, 2] += np.deg2rad(float(config.odom_yaw_offset_deg))

    bev_dir = output_dir / "bev"
    fpv_dir = output_dir / "fpv"
    videos_dir = output_dir / "videos"
    bev_dir.mkdir(parents=True, exist_ok=True)
    fpv_dir.mkdir(parents=True, exist_ok=True)

    indices = list(range(0, len(decoded_images), max(int(config.sample_stride), 1)))
    if int(config.frame_limit) > 0:
        indices = indices[: int(config.frame_limit)]
    rendered_indices = np.asarray(indices, dtype=np.int64)
    np.save(output_dir / "rendered_frame_indices.npy", rendered_indices)
    np.save(output_dir / "rendered_frame_times.npy", frame_times[rendered_indices])
    np.save(output_dir / "rendered_raw_frame_times.npy", raw_frame_times[rendered_indices])

    bev_writer = None
    fpv_writer = None
    rendered = 0
    for out_idx, frame_index in enumerate(indices):
        image_sample = decoded_images[frame_index]
        frame_rgb = np.asarray(Image.open(io.BytesIO(image_sample.data)).convert("RGB"))
        if int(config.fpv_output_width) > 0 and int(config.fpv_output_height) > 0:
            frame_rgb = cv2.resize(
                frame_rgb,
                (int(config.fpv_output_width), int(config.fpv_output_height)),
                interpolation=cv2.INTER_AREA,
            )
        fpv_rgb = _draw_fpv_frame(frame_rgb, positions=positions, angles_all=angles_all, frame_index=frame_index, config=config)
        bev_rgb = _draw_bev_frame(frame_index=frame_index, frame_gps=frame_gps, angles_all=angles_all, config=config)

        _save_jpeg_rgb(fpv_dir / f"frame_{out_idx:06d}_fpv.jpg", fpv_rgb, quality=int(config.jpeg_quality))
        _save_jpeg_rgb(bev_dir / f"frame_{out_idx:06d}_bev.jpg", bev_rgb, quality=int(config.jpeg_quality))

        if config.make_videos:
            if fpv_writer is None:
                fpv_writer = _open_video_writer(videos_dir / "fpv.mp4", fpv_rgb, fps)
            if bev_writer is None:
                bev_writer = _open_video_writer(videos_dir / "bev.mp4", bev_rgb, fps)
            _write_video_frame(fpv_writer, fpv_rgb)
            _write_video_frame(bev_writer, bev_rgb)

        rendered += 1
        if config.progress_every > 0 and rendered % int(config.progress_every) == 0:
            print(f"{data['meta']['sequence']}: rendered {rendered} frames", flush=True)

    if fpv_writer is not None:
        fpv_writer.release()
    if bev_writer is not None:
        bev_writer.release()

    manifest = {
        "sequence": data["meta"]["sequence"],
        "frames_rendered": int(rendered),
        "sample_stride": int(config.sample_stride),
        "frame_limit": int(config.frame_limit),
        "target_output_fps": float(config.target_output_fps),
        "future_horizon": int(config.future_horizon),
        "bev_future_horizon": int(config.bev_future_horizon),
        "odom_yaw_offset_deg": float(config.odom_yaw_offset_deg),
        "bake_odom_yaw_offset_into_input": bool(config.bake_odom_yaw_offset_into_input),
        "frame_orientation_yaw_offset_baked_deg": float(data["meta"].get("frame_orientation_yaw_offset_baked_deg", 0.0)),
        "input_front_width": int(config.input_front_width),
        "input_front_height": int(config.input_front_height),
        "fpv_output_width": int(config.fpv_output_width),
        "fpv_output_height": int(config.fpv_output_height),
        "fpv_line_width": int(config.fpv_line_width),
        "fpv_point_radius": int(config.fpv_point_radius),
        "fpv_arrow_size_px": float(config.fpv_arrow_size_px),
        "fpv_arrow_thickness": int(config.fpv_arrow_thickness),
        "fpv_arrow_offset_px": float(config.fpv_arrow_offset_px),
        "fpv_waypoint_step": int(config.fpv_waypoint_step),
        "camera_height_m": float(config.camera_height_m),
        "camera_pitch_deg": float(config.camera_pitch_deg),
        "bev_output_size": int(config.bev_output_size),
        "bev_zoom": int(config.bev_zoom),
        "bev_mosaic_tiles": int(config.bev_mosaic_tiles),
        "bev_view_scale": float(config.bev_view_scale),
        "jpeg_quality": int(config.jpeg_quality),
        "fps": float(fps),
        "alignment": data["meta"].get("alignment"),
        "raw_frame_time_source": "image message/header timestamp",
        "mcap_log_time_used": False,
        "bev_dir": str(bev_dir),
        "fpv_dir": str(fpv_dir),
        "videos_dir": str(videos_dir) if config.make_videos else None,
        "rendered_frame_indices": str(output_dir / "rendered_frame_indices.npy"),
        "rendered_frame_times": str(output_dir / "rendered_frame_times.npy"),
        "rendered_raw_frame_times": str(output_dir / "rendered_raw_frame_times.npy"),
        "visualization": {
            "bev": "OSM ego-aligned route with direction arrows",
            "fpv": "vis_gt fisheye/extrinsic projection, center guide line only, waypoint dots and small arrows",
            "horizons": list(HORIZONS),
            "horizon_colors_bgr": [list(color) for color in HORIZON_COLORS_BGR],
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return manifest


def _resolve_render_stride(source_fps: float, config: BuildConfig) -> int:
    if int(config.sample_stride) > 0:
        return max(int(config.sample_stride), 1)
    target_fps = float(config.target_output_fps)
    if target_fps <= 0.0:
        return 1
    if source_fps <= 0.0 or not math.isfinite(source_fps):
        return 1
    return max(1, int(round(float(source_fps) / target_fps)))


def _existing_output_matches(output_manifest_path: Path, config: BuildConfig) -> bool:
    if not output_manifest_path.exists():
        return False
    try:
        manifest = json.loads(output_manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if int(manifest.get("frames_rendered") or 0) <= 0:
        return False
    expected_ints = {
        "future_horizon": int(config.future_horizon),
        "bev_future_horizon": int(config.bev_future_horizon),
        "input_front_width": int(config.input_front_width),
        "input_front_height": int(config.input_front_height),
        "fpv_output_width": int(config.fpv_output_width),
        "fpv_output_height": int(config.fpv_output_height),
        "fpv_line_width": int(config.fpv_line_width),
        "fpv_point_radius": int(config.fpv_point_radius),
        "fpv_arrow_thickness": int(config.fpv_arrow_thickness),
        "fpv_waypoint_step": int(config.fpv_waypoint_step),
        "bev_output_size": int(config.bev_output_size),
        "bev_zoom": int(config.bev_zoom),
        "bev_mosaic_tiles": int(config.bev_mosaic_tiles),
        "jpeg_quality": int(config.jpeg_quality),
        "bake_odom_yaw_offset_into_input": int(bool(config.bake_odom_yaw_offset_into_input)),
    }
    for key, expected in expected_ints.items():
        try:
            existing = int(manifest[key])
        except (KeyError, TypeError, ValueError):
            return False
        if existing != expected:
            return False
    expected_floats = {
        "odom_yaw_offset_deg": float(config.odom_yaw_offset_deg),
        "frame_orientation_yaw_offset_baked_deg": (
            float(config.odom_yaw_offset_deg)
            if bool(config.bake_odom_yaw_offset_into_input)
            else 0.0
        ),
        "fpv_arrow_size_px": float(config.fpv_arrow_size_px),
        "fpv_arrow_offset_px": float(config.fpv_arrow_offset_px),
        "camera_height_m": float(config.camera_height_m),
        "camera_pitch_deg": float(config.camera_pitch_deg),
        "bev_view_scale": float(config.bev_view_scale),
    }
    for key, expected in expected_floats.items():
        try:
            existing = float(manifest[key])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isclose(existing, expected, rel_tol=0.0, abs_tol=1e-6):
            return False
    if int(config.sample_stride) > 0:
        return int(manifest.get("sample_stride") or 0) == int(config.sample_stride)
    expected_target = float(config.target_output_fps)
    if expected_target <= 0.0:
        return True
    try:
        existing_target = float(manifest.get("target_output_fps"))
    except (TypeError, ValueError):
        return False
    return math.isclose(existing_target, expected_target, rel_tol=0.0, abs_tol=1e-6)


def _process_one(mcap_path_str: str, config: BuildConfig) -> dict[str, Any]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required")
    mcap_path = Path(mcap_path_str).expanduser().resolve()
    output_root = Path(config.output_root).expanduser().resolve()
    input_dir = output_root / config.input_dir_name / mcap_path.stem
    output_dir = output_root / config.output_dir_name / mcap_path.stem
    input_ready = (input_dir / "manifest.json").exists() if config.save_input else True
    output_ready = _existing_output_matches(output_dir / "manifest.json", config)
    if config.skip_existing and output_ready and input_ready:
        return {
            "sequence": mcap_path.stem,
            "status": "skipped",
            "input_dir": str(input_dir) if config.save_input else None,
            "output_dir": str(output_dir),
        }

    data = _get_aligned_data(mcap_path, config)
    source_fps = float(config.output_fps)
    if source_fps <= 0.0:
        source_fps = _estimate_fps(data["frame_times"])
    render_stride = _resolve_render_stride(source_fps, config)
    render_fps = source_fps / float(render_stride)
    render_config = replace(config, sample_stride=render_stride)
    if config.save_input:
        _write_input_clip(data, input_dir, source_fps, render_config)
    output_manifest = _render_outputs(data, output_dir, render_fps, render_config)
    return {
        "sequence": mcap_path.stem,
        "status": "ok",
        "input_dir": str(input_dir) if config.save_input else None,
        "output_dir": str(output_dir),
        "frames": int(output_manifest["frames_rendered"]),
        "source_fps": float(source_fps),
        "fps": float(render_fps),
        "sample_stride": int(render_stride),
        "target_output_fps": float(config.target_output_fps),
        "input_manifest": str(input_dir / "manifest.json") if config.save_input else None,
        "output_manifest": str(output_dir / "manifest.json"),
    }


def _error_result(mcap_path: Path, exc: BaseException) -> dict[str, Any]:
    return {
        "sequence": mcap_path.stem,
        "status": "error",
        "mcap_path": str(mcap_path),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _resolve_mcap_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(path).expanduser().resolve() for path in args.mcap_path]
    if args.sequence_name:
        if not args.mcap_dir:
            raise RuntimeError("--sequence-name requires --mcap-dir")
        mcap_dir = Path(args.mcap_dir).expanduser().resolve()
        for name in args.sequence_name:
            candidate = Path(name)
            if candidate.suffix == ".mcap":
                paths.append((mcap_dir / candidate.name).resolve())
            else:
                paths.append((mcap_dir / f"{candidate.stem}.mcap").resolve())
    elif args.mcap_dir:
        paths.extend(sorted(Path(args.mcap_dir).expanduser().resolve().glob("*.mcap")))
    if not paths:
        raise RuntimeError("No MCAPs selected. Use --mcap-path or --mcap-dir.")
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(resolved)
    missing = [str(path) for path in deduped if not path.exists()]
    if missing:
        raise RuntimeError(f"MCAP path(s) not found: {missing[:5]}")
    if int(args.limit) > 0:
        deduped = deduped[: int(args.limit)]
    return deduped


def _find_chunk_number(paths: Sequence[Path]) -> str | None:
    pattern = re.compile(r"^chunk[_-]?(\d+)$", re.IGNORECASE)
    for path in paths:
        for part in path.parts:
            match = pattern.match(part)
            if match:
                return match.group(1).lstrip("0") or "0"
    return None


def _resolve_output_layout(args: argparse.Namespace, mcap_paths: Sequence[Path]) -> tuple[Path, str, str]:
    output_root_arg = getattr(args, "output_root", None)
    chunk_name_arg = getattr(args, "chunk_name", None)

    chunk_number = _find_chunk_number([Path(path) for path in mcap_paths])
    if chunk_name_arg:
        chunk_name = str(chunk_name_arg)
        chunk_name_match = re.match(r"^chunk[_-]?(\d+)$", chunk_name, flags=re.IGNORECASE)
        if chunk_name_match:
            chunk_number = chunk_name_match.group(1).lstrip("0") or "0"
            chunk_name = f"chunk{chunk_number}"
    elif chunk_number is not None:
        chunk_name = f"chunk{chunk_number}"
    else:
        chunk_name = "chunk"

    if output_root_arg:
        output_root = Path(output_root_arg).expanduser().resolve()
        root_match = re.match(r"^chunk[_-]?(\d+)$", output_root.name, flags=re.IGNORECASE)
        if root_match and chunk_number is None:
            chunk_number = root_match.group(1).lstrip("0") or "0"
    else:
        output_root = (Path(args.chunk_output_base).expanduser().resolve() / chunk_name)

    input_dir_name = args.input_dir_name
    output_dir_name = args.output_dir_name
    if input_dir_name is None:
        input_dir_name = f"input{chunk_number}" if chunk_number is not None else "input"
    if output_dir_name is None:
        output_dir_name = f"output{chunk_number}" if chunk_number is not None else "output"
    return output_root, str(input_dir_name), str(output_dir_name)


def _config_from_args(args: argparse.Namespace, output_root: Path, input_dir_name: str, output_dir_name: str) -> BuildConfig:
    return BuildConfig(
        output_root=str(output_root),
        input_dir_name=str(input_dir_name),
        output_dir_name=str(output_dir_name),
        image_topic=str(args.image_topic),
        odom_topic=str(args.odom_topic),
        gps_topic=str(args.gps_topic),
        frame_time_offset_ms=float(args.frame_time_offset_ms),
        odom_yaw_offset_deg=float(args.odom_yaw_offset_deg),
        bake_odom_yaw_offset_into_input=bool(args.bake_odom_yaw_offset_into_input),
        output_fps=float(args.output_fps),
        sample_stride=max(int(args.sample_stride), 0),
        target_output_fps=max(float(args.target_output_fps), 0.0),
        save_input=bool(args.save_input),
        frame_limit=max(int(args.frame_limit), 0),
        future_horizon=max(int(args.future_horizon), 2),
        bev_future_horizon=int(args.bev_future_horizon),
        camera_height_m=float(args.camera_height_m),
        camera_pitch_deg=float(args.camera_pitch_deg),
        bev_output_size=int(args.bev_output_size),
        bev_zoom=int(args.bev_zoom),
        bev_mosaic_tiles=int(args.bev_mosaic_tiles),
        bev_view_scale=max(float(args.bev_view_scale), 1.0),
        osm_cache_dir=str(args.osm_cache_dir),
        bev_arrow_every_px=float(args.bev_arrow_every_px),
        bev_arrow_size_px=float(args.bev_arrow_size_px),
        input_front_width=max(int(args.input_front_width), 0),
        input_front_height=max(int(args.input_front_height), 0),
        fpv_output_width=max(int(args.fpv_output_width), 0),
        fpv_output_height=max(int(args.fpv_output_height), 0),
        fpv_line_width=max(int(args.fpv_line_width), 1),
        fpv_point_radius=max(int(args.fpv_point_radius), 1),
        fpv_arrow_size_px=float(args.fpv_arrow_size_px),
        fpv_arrow_thickness=max(int(args.fpv_arrow_thickness), 1),
        fpv_arrow_offset_px=float(args.fpv_arrow_offset_px),
        fpv_waypoint_step=max(int(args.fpv_waypoint_step), 1),
        make_videos=bool(args.make_videos),
        skip_existing=bool(args.skip_existing),
        jpeg_quality=max(1, min(int(args.jpeg_quality), 100)),
        progress_every=int(args.progress_every),
    )


def main() -> None:
    args = parse_args()
    mcap_paths = _resolve_mcap_paths(args)
    output_root, input_dir_name, output_dir_name = _resolve_output_layout(args, mcap_paths)
    config = _config_from_args(args, output_root, input_dir_name, output_dir_name)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    workers = max(int(args.workers), 1)
    continue_on_error = bool(args.continue_on_error)
    if workers == 1 or len(mcap_paths) == 1:
        for path in mcap_paths:
            try:
                results.append(_process_one(str(path), config))
            except Exception as exc:
                if not continue_on_error:
                    raise
                results.append(_error_result(path, exc))
                print(f"ERROR: failed processing {path}: {exc}", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_process_one, str(path), config): path for path in mcap_paths}
            for future in as_completed(future_map):
                path = future_map[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    if not continue_on_error:
                        raise
                    results.append(_error_result(path, exc))
                    print(f"ERROR: failed processing {path}: {exc}", file=sys.stderr)

    manifest = {
        "script": "tools.sidewalker.build_visgt_sidewalker_dataset",
        "output_root": str(output_root),
        "input_root": str(output_root / input_dir_name) if config.save_input else None,
        "output_root_sequences": str(output_root / output_dir_name),
        "input_dir_name": input_dir_name,
        "output_dir_name": output_dir_name,
        "save_input": bool(config.save_input),
        "target_output_fps": float(config.target_output_fps),
        "sample_stride": int(config.sample_stride),
        "odom_yaw_offset_deg": float(config.odom_yaw_offset_deg),
        "bake_odom_yaw_offset_into_input": bool(config.bake_odom_yaw_offset_into_input),
        "mcap_count": len(mcap_paths),
        "ok_count": sum(1 for item in results if item.get("status") == "ok"),
        "skipped_count": sum(1 for item in results if item.get("status") == "skipped"),
        "error_count": sum(1 for item in results if item.get("status") == "error"),
        "results": results,
    }
    (output_root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
