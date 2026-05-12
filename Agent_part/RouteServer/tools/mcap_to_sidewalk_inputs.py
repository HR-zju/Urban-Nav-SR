#!/usr/bin/env python3
"""Extract RouteServer sidewalk-pipeline inputs from a single ROS2 MCAP file.

The sidewalk visual-prompt batch runner consumes a deliberately small contract:

    output_dir/
      images/000001.jpg
      trajectory.json

where each trajectory row contains ``timestamp``, ``lat``, ``lng``,
``heading_deg``, and ``image_path``. This tool builds that contract directly
from a single MCAP file without requiring a ROS2 installation.

Supported ROS2 message schemas:

    sensor_msgs/msg/CompressedImage
    sensor_msgs/msg/NavSatFix
    std_msgs/msg/Float64

Core synchronization:

    i_k = sampled image at time t_k
    gps_k     = argmin_g |t_g - t_k|
    heading_k = argmin_h |t_h - t_k|

Frames whose nearest GPS or heading sample is farther than
``--max-sync-delta-s`` are skipped. The resulting ``trajectory.json`` can be
passed directly to ``tools/run_sidewalk_images_visual_prompt_pipeline.py``.
"""

from __future__ import annotations

import argparse
import bisect
import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import lz4.frame as lz4_frame
except ModuleNotFoundError:
    lz4_frame = None


MCAP_MAGIC = b"\x89MCAP0\r\n"


@dataclass(frozen=True)
class SchemaInfo:
    schema_id: int
    name: str
    encoding: str


@dataclass(frozen=True)
class ChannelInfo:
    channel_id: int
    schema_id: int
    topic: str
    message_encoding: str
    schema_name: str


@dataclass(frozen=True)
class ImageSample:
    timestamp_ns: int
    data: bytes
    source_topic: str


@dataclass(frozen=True)
class GpsSample:
    timestamp_ns: int
    lat: float
    lng: float
    altitude_m: float | None
    status: int | None
    source_topic: str


@dataclass(frozen=True)
class HeadingSample:
    timestamp_ns: int
    heading_deg: float
    source_topic: str


def _read_u16(buffer: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<H", buffer, offset)[0], offset + 2


def _read_u32(buffer: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<I", buffer, offset)[0], offset + 4


def _read_u64(buffer: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<Q", buffer, offset)[0], offset + 8


def _read_mcap_string(buffer: bytes, offset: int) -> tuple[str, int]:
    size, offset = _read_u32(buffer, offset)
    raw = buffer[offset : offset + size]
    return raw.decode("utf-8", "replace"), offset + size


def _read_mcap_bytes_u32(buffer: bytes, offset: int) -> tuple[bytes, int]:
    size, offset = _read_u32(buffer, offset)
    return buffer[offset : offset + size], offset + size


def _read_mcap_bytes_u64(buffer: bytes, offset: int) -> tuple[bytes, int]:
    size, offset = _read_u64(buffer, offset)
    return buffer[offset : offset + size], offset + size


def _read_mcap_map(buffer: bytes, offset: int) -> tuple[dict[str, str], int]:
    total_size, offset = _read_u32(buffer, offset)
    end = offset + total_size
    values: dict[str, str] = {}
    while offset < end:
        key, offset = _read_mcap_string(buffer, offset)
        value, offset = _read_mcap_string(buffer, offset)
        values[key] = value
    return values, offset


def _iter_mcap_records(mcap_path: Path) -> Iterable[tuple[int, bytes]]:
    with mcap_path.open("rb") as stream:
        magic = stream.read(len(MCAP_MAGIC))
        if magic != MCAP_MAGIC:
            raise RuntimeError(f"Not an MCAP file or unsupported magic: {mcap_path}")
        while True:
            header = stream.read(9)
            if not header:
                return
            if len(header) != 9:
                if header == MCAP_MAGIC:
                    return
                raise RuntimeError("Truncated MCAP record header")
            if header[: len(MCAP_MAGIC)] == MCAP_MAGIC:
                return
            opcode = int(header[0])
            record_length = struct.unpack("<Q", header[1:])[0]
            payload = stream.read(record_length)
            if len(payload) != record_length:
                raise RuntimeError("Truncated MCAP record payload")
            yield opcode, payload


def _iter_chunk_records(chunk_payload: bytes) -> Iterable[tuple[int, bytes]]:
    offset = 0
    while offset + 9 <= len(chunk_payload):
        opcode = int(chunk_payload[offset])
        record_length = struct.unpack_from("<Q", chunk_payload, offset + 1)[0]
        offset += 9
        end = offset + record_length
        if end > len(chunk_payload):
            raise RuntimeError("Truncated record inside MCAP chunk")
        yield opcode, chunk_payload[offset:end]
        offset = end


def _decode_chunk(record_payload: bytes) -> bytes:
    offset = 0
    _, offset = _read_u64(record_payload, offset)  # message_start_time
    _, offset = _read_u64(record_payload, offset)  # message_end_time
    _, offset = _read_u64(record_payload, offset)  # uncompressed_size
    _, offset = _read_u32(record_payload, offset)  # uncompressed_crc
    compression, offset = _read_mcap_string(record_payload, offset)
    compressed_data, offset = _read_mcap_bytes_u64(record_payload, offset)
    if compression in {"", "none"}:
        return compressed_data
    if compression == "lz4":
        if lz4_frame is None:
            raise RuntimeError(
                "This MCAP uses lz4-compressed chunks, but the Python package 'lz4' is not installed. "
                "Install it in the active environment with `python -m pip install lz4` "
                "or `conda install -c conda-forge lz4`."
            )
        return lz4_frame.decompress(compressed_data)
    raise RuntimeError(f"Unsupported MCAP chunk compression: {compression!r}")


class CdrReader:
    """Minimal little-endian CDR reader for the ROS2 messages used here."""

    def __init__(self, data: bytes) -> None:
        if len(data) < 4:
            raise RuntimeError("CDR payload is too short")
        self.data = data
        self.base_offset = 4
        self.offset = 4

    def align(self, size: int) -> None:
        relative = self.offset - self.base_offset
        self.offset = self.base_offset + ((relative + size - 1) & ~(size - 1))

    def int8(self) -> int:
        value = struct.unpack_from("<b", self.data, self.offset)[0]
        self.offset += 1
        return int(value)

    def uint16(self) -> int:
        self.align(2)
        value = struct.unpack_from("<H", self.data, self.offset)[0]
        self.offset += 2
        return int(value)

    def int32(self) -> int:
        self.align(4)
        value = struct.unpack_from("<i", self.data, self.offset)[0]
        self.offset += 4
        return int(value)

    def uint32(self) -> int:
        self.align(4)
        value = struct.unpack_from("<I", self.data, self.offset)[0]
        self.offset += 4
        return int(value)

    def float64(self) -> float:
        self.align(8)
        value = struct.unpack_from("<d", self.data, self.offset)[0]
        self.offset += 8
        return float(value)

    def string(self) -> str:
        self.align(4)
        size = self.uint32()
        raw = self.data[self.offset : self.offset + size]
        self.offset += size
        if raw.endswith(b"\x00"):
            raw = raw[:-1]
        return raw.decode("utf-8", "replace")

    def bytes(self) -> bytes:
        self.align(4)
        size = self.uint32()
        raw = self.data[self.offset : self.offset + size]
        self.offset += size
        return raw


def _parse_schema(record_payload: bytes) -> SchemaInfo:
    offset = 0
    schema_id, offset = _read_u16(record_payload, offset)
    name, offset = _read_mcap_string(record_payload, offset)
    encoding, offset = _read_mcap_string(record_payload, offset)
    return SchemaInfo(schema_id=int(schema_id), name=name, encoding=encoding)


def _parse_channel(record_payload: bytes, schemas: dict[int, SchemaInfo]) -> ChannelInfo:
    offset = 0
    channel_id, offset = _read_u16(record_payload, offset)
    schema_id, offset = _read_u16(record_payload, offset)
    topic, offset = _read_mcap_string(record_payload, offset)
    message_encoding, offset = _read_mcap_string(record_payload, offset)
    _, offset = _read_mcap_map(record_payload, offset)
    schema = schemas.get(int(schema_id))
    return ChannelInfo(
        channel_id=int(channel_id),
        schema_id=int(schema_id),
        topic=topic,
        message_encoding=message_encoding,
        schema_name=schema.name if schema is not None else "",
    )


def _parse_message_record(record_payload: bytes) -> tuple[int, int, bytes]:
    offset = 0
    channel_id, offset = _read_u16(record_payload, offset)
    _, offset = _read_u32(record_payload, offset)  # sequence
    log_time, offset = _read_u64(record_payload, offset)
    _, offset = _read_u64(record_payload, offset)  # publish_time
    return int(channel_id), int(log_time), record_payload[offset:]


def _parse_navsatfix(payload: bytes, *, timestamp_ns: int, source_topic: str) -> GpsSample:
    reader = CdrReader(payload)
    _ = reader.int32()  # header.stamp.sec
    _ = reader.uint32()  # header.stamp.nanosec
    _ = reader.string()  # header.frame_id
    status = reader.int8()
    _ = reader.uint16()  # service
    lat = reader.float64()
    lng = reader.float64()
    altitude = reader.float64()
    return GpsSample(
        timestamp_ns=int(timestamp_ns),
        lat=float(lat),
        lng=float(lng),
        altitude_m=float(altitude),
        status=int(status),
        source_topic=source_topic,
    )


def _parse_float64(payload: bytes, *, timestamp_ns: int, source_topic: str) -> HeadingSample:
    heading = CdrReader(payload).float64()
    return HeadingSample(timestamp_ns=int(timestamp_ns), heading_deg=float(heading) % 360.0, source_topic=source_topic)


def _extract_jpeg_bytes(payload: bytes) -> bytes:
    # sensor_msgs/CompressedImage stores format + uint8[] data, but direct JPEG
    # marker extraction is more tolerant of custom encoders and still validates
    # against SOI/EOI markers.
    start = payload.find(b"\xff\xd8")
    end = payload.rfind(b"\xff\xd9")
    if start < 0 or end < start:
        raise RuntimeError("CompressedImage payload does not contain JPEG SOI/EOI markers")
    return payload[start : end + 2]


def _nearest_sample(samples: list[Any], timestamps: list[int], target_ns: int) -> tuple[Any | None, int | None]:
    if not samples:
        return None, None
    right = bisect.bisect_left(timestamps, int(target_ns))
    if right <= 0:
        sample = samples[0]
    elif right >= len(samples):
        sample = samples[-1]
    else:
        left_sample = samples[right - 1]
        right_sample = samples[right]
        if abs(int(left_sample.timestamp_ns) - int(target_ns)) <= abs(int(right_sample.timestamp_ns) - int(target_ns)):
            sample = left_sample
        else:
            sample = right_sample
    delta_ns = abs(int(sample.timestamp_ns) - int(target_ns))
    return sample, delta_ns


def _choose_channel(
    channels: dict[int, ChannelInfo],
    *,
    requested_topic: str,
    schema_substring: str,
    role: str,
) -> ChannelInfo:
    requested = str(requested_topic or "").strip()
    if requested and requested.lower() != "auto":
        for channel in channels.values():
            if channel.topic == requested:
                return channel
        raise RuntimeError(f"Requested {role} topic not found: {requested}")
    candidates = [
        channel
        for channel in channels.values()
        if schema_substring.lower() in channel.schema_name.lower()
    ]
    if not candidates:
        raise RuntimeError(f"No {role} channel found for schema containing {schema_substring!r}")
    return sorted(candidates, key=lambda item: item.topic)[0]


def extract_mcap_inputs(
    *,
    mcap_path: Path,
    output_dir: Path,
    image_topic: str,
    gps_topic: str,
    heading_topic: str,
    sample_fps: float,
    max_sync_delta_s: float,
    image_start_index: int,
) -> tuple[Path, Path]:
    schemas: dict[int, SchemaInfo] = {}
    channels: dict[int, ChannelInfo] = {}
    images: list[ImageSample] = []
    gps_samples: list[GpsSample] = []
    heading_samples: list[HeadingSample] = []
    message_counts: dict[str, int] = {}

    # MCAP files commonly store schemas/channels in the summary section after
    # all chunks. Keep discovery and message extraction separate so single-file
    # MCAPs written by mcap-rs do not depend on record ordering.
    for opcode, payload in _iter_mcap_records(mcap_path):
        if opcode == 3:  # schema
            schema = _parse_schema(payload)
            schemas[schema.schema_id] = schema
            continue
        if opcode == 4:  # channel
            channel = _parse_channel(payload, schemas)
            channels[channel.channel_id] = channel

    image_channel = _choose_channel(channels, requested_topic=image_topic, schema_substring="CompressedImage", role="image")
    gps_channel = _choose_channel(channels, requested_topic=gps_topic, schema_substring="NavSatFix", role="gps")
    heading_channel = _choose_channel(channels, requested_topic=heading_topic, schema_substring="Float64", role="heading")

    for opcode, payload in _iter_mcap_records(mcap_path):
        if opcode != 6:  # chunk
            continue
        chunk = _decode_chunk(payload)
        for inner_opcode, inner_payload in _iter_chunk_records(chunk):
            if inner_opcode != 5:
                continue
            channel_id, timestamp_ns, message_payload = _parse_message_record(inner_payload)
            channel = channels.get(channel_id)
            if channel is None:
                continue
            message_counts[channel.topic] = message_counts.get(channel.topic, 0) + 1
            if channel_id == image_channel.channel_id:
                images.append(
                    ImageSample(
                        timestamp_ns=timestamp_ns,
                        data=_extract_jpeg_bytes(message_payload),
                        source_topic=channel.topic,
                    )
                )
            elif channel_id == gps_channel.channel_id:
                try:
                    gps_samples.append(_parse_navsatfix(message_payload, timestamp_ns=timestamp_ns, source_topic=channel.topic))
                except Exception:
                    continue
            elif channel_id == heading_channel.channel_id:
                try:
                    heading_samples.append(_parse_float64(message_payload, timestamp_ns=timestamp_ns, source_topic=channel.topic))
                except Exception:
                    continue

    if not images:
        raise RuntimeError(f"No images extracted from topic: {image_channel.topic}")
    if not gps_samples:
        raise RuntimeError(f"No GPS samples extracted from topic: {gps_channel.topic}")
    if not heading_samples:
        raise RuntimeError(f"No heading samples extracted from topic: {heading_channel.topic}")

    images = sorted(images, key=lambda item: item.timestamp_ns)
    gps_samples = sorted(gps_samples, key=lambda item: item.timestamp_ns)
    heading_samples = sorted(heading_samples, key=lambda item: item.timestamp_ns)
    gps_timestamps = [sample.timestamp_ns for sample in gps_samples]
    heading_timestamps = [sample.timestamp_ns for sample in heading_samples]

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    trajectory: list[dict[str, Any]] = []
    frame_records: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []
    period_ns = 0 if float(sample_fps) <= 0.0 else int(round(1e9 / float(sample_fps)))
    next_emit_ns: int | None = None
    frame_index = int(image_start_index)
    max_delta_ns = int(round(float(max_sync_delta_s) * 1e9))

    for image in images:
        if next_emit_ns is None:
            next_emit_ns = int(image.timestamp_ns)
        if period_ns > 0 and int(image.timestamp_ns) < int(next_emit_ns):
            continue

        gps, gps_delta_ns = _nearest_sample(gps_samples, gps_timestamps, image.timestamp_ns)
        heading, heading_delta_ns = _nearest_sample(heading_samples, heading_timestamps, image.timestamp_ns)
        if gps is None or heading is None or gps_delta_ns is None or heading_delta_ns is None:
            skipped_records.append({"timestamp": image.timestamp_ns, "reason": "missing_nearest_pose"})
            continue
        if gps_delta_ns > max_delta_ns or heading_delta_ns > max_delta_ns:
            skipped_records.append(
                {
                    "timestamp": image.timestamp_ns,
                    "reason": "sync_delta_too_large",
                    "gps_delta_s": round(gps_delta_ns / 1e9, 6),
                    "heading_delta_s": round(heading_delta_ns / 1e9, 6),
                }
            )
            continue

        image_path = images_dir / f"{frame_index:06d}.jpg"
        image_path.write_bytes(image.data)
        trajectory.append(
            {
                "timestamp": int(image.timestamp_ns),
                "lat": float(gps.lat),
                "lng": float(gps.lng),
                "heading_deg": float(heading.heading_deg),
                "image_path": str(image_path),
            }
        )
        frame_records.append(
            {
                "frame_index": int(frame_index),
                "timestamp": int(image.timestamp_ns),
                "image_path": str(image_path),
                "gps": {
                    "timestamp": int(gps.timestamp_ns),
                    "lat": float(gps.lat),
                    "lng": float(gps.lng),
                    "altitude_m": gps.altitude_m,
                    "status": gps.status,
                    "sync_delta_s": round(float(gps_delta_ns) / 1e9, 6),
                },
                "heading": {
                    "timestamp": int(heading.timestamp_ns),
                    "heading_deg": float(heading.heading_deg),
                    "sync_delta_s": round(float(heading_delta_ns) / 1e9, 6),
                },
            }
        )
        frame_index += 1
        if period_ns > 0:
            next_emit_ns = int(image.timestamp_ns) + period_ns

    if not trajectory:
        raise RuntimeError("No synchronized frames were written; check topics or --max-sync-delta-s")

    trajectory_path = output_dir / "trajectory.json"
    frames_manifest_path = output_dir / "frames_manifest.json"
    meta_path = output_dir / "mcap_extract_meta.json"
    trajectory_path.write_text(json.dumps(trajectory, indent=2, ensure_ascii=True), encoding="utf-8")
    frames_manifest_path.write_text(json.dumps(frame_records, indent=2, ensure_ascii=True), encoding="utf-8")
    meta = {
        "mcap_path": str(mcap_path),
        "images_dir": str(images_dir),
        "trajectory_json": str(trajectory_path),
        "frames_manifest_path": str(frames_manifest_path),
        "image_topic": image_channel.topic,
        "gps_topic": gps_channel.topic,
        "heading_topic": heading_channel.topic,
        "sample_fps": float(sample_fps),
        "max_sync_delta_s": float(max_sync_delta_s),
        "image_start_index": int(image_start_index),
        "raw_counts": {
            "images": len(images),
            "gps_samples": len(gps_samples),
            "heading_samples": len(heading_samples),
        },
        "written_frames": len(trajectory),
        "skipped_frames": len(skipped_records),
        "skipped_records_preview": skipped_records[:20],
        "channels": [asdict(channel) for channel in sorted(channels.values(), key=lambda item: item.channel_id)],
        "message_counts_by_topic": message_counts,
        "pipeline_command_hint": {
            "images_dir": str(images_dir),
            "trajectory_json": str(trajectory_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
    return trajectory_path, meta_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract numbered JPEG frames and RouteServer trajectory.json from a single ROS2 MCAP file."
    )
    parser.add_argument("--mcap-path", required=True, help="Path to a single .mcap file")
    parser.add_argument("--output-dir", required=True, help="Output directory containing images/ and trajectory.json")
    parser.add_argument("--image-topic", default="/camera_x001/image_raw/compressed", help="CompressedImage topic, or auto")
    parser.add_argument("--gps-topic", default="/baton/gnss/spp_fix", help="NavSatFix topic, or auto")
    parser.add_argument("--heading-topic", default="/baton/gnss/heading", help="Float64 heading topic, or auto")
    parser.add_argument("--sample-fps", type=float, default=1.0, help="Target image sampling rate; <=0 keeps all images")
    parser.add_argument("--max-sync-delta-s", type=float, default=0.5, help="Max nearest GPS/heading sync delta per frame")
    parser.add_argument("--image-start-index", type=int, default=1, help="First output frame index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectory_path, meta_path = extract_mcap_inputs(
        mcap_path=Path(args.mcap_path).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        image_topic=str(args.image_topic),
        gps_topic=str(args.gps_topic),
        heading_topic=str(args.heading_topic),
        sample_fps=float(args.sample_fps),
        max_sync_delta_s=float(args.max_sync_delta_s),
        image_start_index=int(args.image_start_index),
    )
    print("Done.")
    print(f"trajectory_json={trajectory_path}")
    print(f"images_dir={trajectory_path.parent / 'images'}")
    print(f"meta_json={meta_path}")


if __name__ == "__main__":
    main()
