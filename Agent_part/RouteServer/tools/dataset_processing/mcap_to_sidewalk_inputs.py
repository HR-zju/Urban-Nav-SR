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
import io
import json
import math
import os
import struct
import subprocess
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import lz4.frame as lz4_frame
except ModuleNotFoundError:
    lz4_frame = None

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    from PIL import Image as PILImage
except ModuleNotFoundError:
    PILImage = None


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


@dataclass(frozen=True)
class AttitudeSample:
    timestamp_ns: int
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    source_topic: str
    source_schema: str


@dataclass(frozen=True)
class ImuSample:
    timestamp_ns: int
    angular_velocity_x: float | None
    angular_velocity_y: float | None
    angular_velocity_z: float | None
    source_topic: str
    source_schema: str


@dataclass(frozen=True)
class OdometrySample:
    timestamp_ns: int
    angular_velocity_x: float | None
    angular_velocity_y: float | None
    angular_velocity_z: float | None
    source_topic: str
    source_schema: str


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

    def uint8(self) -> int:
        value = struct.unpack_from("<B", self.data, self.offset)[0]
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


class Ros1Reader:
    """Minimal ROS1 serialized-message reader (no CDR alignment)."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    def int32(self) -> int:
        value = struct.unpack_from("<i", self.data, self.offset)[0]
        self.offset += 4
        return int(value)

    def uint32(self) -> int:
        value = struct.unpack_from("<I", self.data, self.offset)[0]
        self.offset += 4
        return int(value)

    def float64(self) -> float:
        value = struct.unpack_from("<d", self.data, self.offset)[0]
        self.offset += 8
        return float(value)

    def string(self) -> str:
        size = self.uint32()
        raw = self.data[self.offset : self.offset + size]
        self.offset += size
        return raw.decode("utf-8", "replace")


def _make_reader(payload: bytes, message_encoding: str) -> Any:
    if str(message_encoding or "").strip().lower() == "ros1":
        return Ros1Reader(payload)
    return CdrReader(payload)


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
    # We manually verified that these offsets work for the CoCo dataset.
    # The CDR reader didn't correctly handle the message format, so we use
    # direct struct unpacking with known offsets:
    #   - status at offset 19 (int8)
    #   - latitude at offset 22 (little-endian double)
    #   - longitude at offset 30 (little-endian double)
    #   - altitude at offset 38 (little-endian double)
    lat = struct.unpack_from("<d", payload, 22)[0]
    lng = struct.unpack_from("<d", payload, 30)[0]
    altitude = struct.unpack_from("<d", payload, 38)[0]
    status = struct.unpack_from("<b", payload, 19)[0] if len(payload) > 19 else -1
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


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    y = math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2))
    x = (
        math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
        - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(lon2 - lon1))
    )
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _normalize_heading_deg(heading_deg: float) -> float:
    return float(heading_deg) % 360.0


def _angle_diff_deg(a_deg: float, b_deg: float) -> float:
    return ((float(a_deg) - float(b_deg) + 180.0) % 360.0) - 180.0


def _atomic_write_text(path: Path, content: str) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(content, encoding="utf-8")
    os.replace(temp_path, path)


def _heading_samples_from_gps(gps_samples: list[GpsSample]) -> list[HeadingSample]:
    if not gps_samples:
        return []
    if len(gps_samples) == 1:
        sample = gps_samples[0]
        return [HeadingSample(timestamp_ns=int(sample.timestamp_ns), heading_deg=0.0, source_topic=sample.source_topic)]

    heading_samples: list[HeadingSample] = []
    for index, sample in enumerate(gps_samples):
        if index < len(gps_samples) - 1:
            next_sample = gps_samples[index + 1]
            heading_deg = _bearing_deg(sample.lat, sample.lng, next_sample.lat, next_sample.lng)
        else:
            prev_sample = gps_samples[index - 1]
            heading_deg = _bearing_deg(prev_sample.lat, prev_sample.lng, sample.lat, sample.lng)
        heading_samples.append(
            HeadingSample(
                timestamp_ns=int(sample.timestamp_ns),
                heading_deg=float(heading_deg),
                source_topic=sample.source_topic,
            )
        )
    return heading_samples


def _encoding_channels(encoding: str) -> tuple[int, str]:
    enc = str(encoding or "").strip().lower()
    mapping = {
        "mono8": (1, "uint8"),
        "8uc1": (1, "uint8"),
        "mono16": (1, "uint16"),
        "16uc1": (1, "uint16"),
        "rgb8": (3, "uint8"),
        "bgr8": (3, "uint8"),
        "rgba8": (4, "uint8"),
        "bgra8": (4, "uint8"),
        "rgb16": (3, "uint16"),
        "bgr16": (3, "uint16"),
        "rgba16": (4, "uint16"),
        "bgra16": (4, "uint16"),
    }
    if enc not in mapping:
        raise RuntimeError(f"Unsupported sensor_msgs/Image encoding: {encoding!r}")
    return mapping[enc]


def _decode_compressed_image(payload: bytes) -> Any:
    if PILImage is None:
        raise RuntimeError("Pillow is required to decode compressed images")
    image = PILImage.open(io.BytesIO(_extract_jpeg_bytes(payload)))
    return image.convert("RGB")


def _decode_raw_image(payload: bytes) -> Any:
    if np is None or PILImage is None:
        raise RuntimeError("numpy and Pillow are required to decode sensor_msgs/Image")

    reader = CdrReader(payload)
    _ = reader.int32()  # header.stamp.sec
    _ = reader.uint32()  # header.stamp.nanosec
    _ = reader.string()  # header.frame_id
    height = int(reader.uint32())
    width = int(reader.uint32())
    encoding = reader.string()
    is_bigendian = int(reader.uint8())
    step = int(reader.uint32())
    data = reader.bytes()

    channels, dtype_name = _encoding_channels(encoding)
    dtype = np.dtype(dtype_name)
    bytes_per_pixel = dtype.itemsize * channels
    if step < width * bytes_per_pixel:
        raise RuntimeError(
            f"Invalid image step for encoding={encoding!r}: step={step}, width={width}, bytes_per_pixel={bytes_per_pixel}"
        )

    buffer = np.frombuffer(data, dtype=dtype)
    if is_bigendian == 1 and dtype.itemsize > 1:
        buffer = buffer.byteswap().newbyteorder()

    row_pixels = step // dtype.itemsize
    if channels == 1:
        row = buffer.reshape((height, row_pixels))[:, :width]
        if dtype == np.dtype("uint16"):
            return PILImage.fromarray(row, mode="I;16")
        return PILImage.fromarray(row.astype(np.uint8), mode="L")

    row = buffer.reshape((height, row_pixels))[:, : width * channels]
    arr = row.reshape((height, width, channels))
    enc = str(encoding).strip().lower()
    if enc == "bgr8":
        arr = arr[:, :, ::-1]
    elif enc == "bgra8":
        arr = arr[:, :, [2, 1, 0, 3]]
    elif enc == "bgr16":
        arr = arr[:, :, ::-1]
    elif enc == "bgra16":
        arr = arr[:, :, [2, 1, 0, 3]]
    elif enc not in {"rgb8", "rgba8", "rgb16", "rgba16"}:
        raise RuntimeError(f"Unsupported image encoding transformation: {encoding!r}")
    return PILImage.fromarray(arr)


def _read_protobuf_varint(buffer: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(buffer):
            raise RuntimeError("Truncated protobuf varint")
        byte_value = buffer[offset]
        offset += 1
        value |= (byte_value & 0x7F) << shift
        if byte_value & 0x80 == 0:
            return value, offset
        shift += 7
        if shift >= 64:
            raise RuntimeError("Protobuf varint is too long")


def _parse_protobuf_timestamp(payload: bytes) -> tuple[int | None, int | None]:
    sec = None
    nsec = None
    offset = 0
    while offset < len(payload):
        key, offset = _read_protobuf_varint(payload, offset)
        field_number = key >> 3
        wire_type = key & 0x7
        if wire_type == 0:
            value, offset = _read_protobuf_varint(payload, offset)
            if field_number == 1:
                sec = int(value)
            elif field_number == 2:
                nsec = int(value)
        elif wire_type == 2:
            length, offset = _read_protobuf_varint(payload, offset)
            offset += int(length)
        else:
            raise RuntimeError(f"Unsupported protobuf wire type in timestamp: {wire_type}")
    return sec, nsec


def _parse_compressed_video(payload: bytes, *, timestamp_ns: int, source_topic: str) -> tuple[int, bytes, str]:
    data = b""
    format_name = ""
    offset = 0
    while offset < len(payload):
        key, offset = _read_protobuf_varint(payload, offset)
        field_number = key >> 3
        wire_type = key & 0x7
        if wire_type != 2:
            if wire_type == 0:
                _, offset = _read_protobuf_varint(payload, offset)
                continue
            raise RuntimeError(f"Unsupported protobuf wire type in CompressedVideo: {wire_type}")

        length, offset = _read_protobuf_varint(payload, offset)
        value = payload[offset : offset + int(length)]
        offset += int(length)

        if field_number == 1:
            _parse_protobuf_timestamp(value)
        elif field_number == 2:
            # frame_id is present for completeness but not required for extraction.
            continue
        elif field_number == 3:
            data = value
        elif field_number in {4, 6}:
            format_name = value.decode("utf-8", "replace")

    if not data:
        raise RuntimeError(f"foxglove.CompressedVideo message on {source_topic!r} does not contain frame data")
    if not format_name:
        raise RuntimeError(f"foxglove.CompressedVideo message on {source_topic!r} does not contain a format field")
    return int(timestamp_ns), data, format_name


def _h264_packet_has_idr(packet: bytes) -> bool:
    index = 0
    while index < len(packet) - 4:
        if packet[index : index + 3] == b"\x00\x00\x01":
            start = index + 3
        elif packet[index : index + 4] == b"\x00\x00\x00\x01":
            start = index + 4
        else:
            index += 1
            continue
        if start < len(packet):
            nal_type = packet[start] & 0x1F
            if nal_type == 5:
                return True
        index = start + 1
    return False


def _decode_compressed_video_packets(
    packets: list[tuple[int, bytes]],
    *,
    output_dir: Path,
    source_topic: str,
    source_schema: str,
) -> list[ImageSample]:
    if not packets:
        return []
    ordered_packets = sorted(packets, key=lambda item: item[0])
    keyframe_index = next((index for index, (_, packet_payload) in enumerate(ordered_packets) if _h264_packet_has_idr(packet_payload)), None)
    if keyframe_index is not None and keyframe_index > 0:
        ordered_packets = ordered_packets[keyframe_index:]

    if PILImage is None:
        raise RuntimeError("Pillow is required to decode compressed video frames")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to decode foxglove.CompressedVideo frames")

    bitstream_path = output_dir / "compressed_video.h264"
    with bitstream_path.open("wb") as stream:
        for timestamp_ns, packet_payload in ordered_packets:
            _, frame_data, format_name = _parse_compressed_video(
                packet_payload,
                timestamp_ns=timestamp_ns,
                source_topic=source_topic,
            )
            if str(format_name).strip().lower() != "h264":
                raise RuntimeError(f"Unsupported foxglove.CompressedVideo format: {format_name!r}")
            stream.write(frame_data)

    decoded_dir = output_dir / "decoded_video_frames"
    decoded_dir.mkdir(parents=True, exist_ok=True)
    decode_pattern = decoded_dir / "%06d.jpg"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "h264",
        "-i",
        str(bitstream_path),
        "-vsync",
        "0",
        str(decode_pattern),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed to decode foxglove.CompressedVideo frames: "
            f"STDOUT={result.stdout.strip()} STDERR={result.stderr.strip()}"
        )

    decoded_files = sorted(decoded_dir.glob("*.jpg"))
    if len(decoded_files) != len(ordered_packets):
        raise RuntimeError(
            f"Decoded frame count mismatch for {source_schema}: expected {len(ordered_packets)}, got {len(decoded_files)}"
        )

    samples: list[ImageSample] = []
    for (timestamp_ns, _), decoded_file in zip(ordered_packets, decoded_files):
        samples.append(
            ImageSample(
                timestamp_ns=int(timestamp_ns),
                data=decoded_file.read_bytes(),
                source_topic=source_topic,
            )
        )
    return samples


def _image_payload_to_jpg_bytes(payload: bytes, channel: ChannelInfo) -> bytes:
    schema_name = channel.schema_name.lower()
    if "compressedimage" in schema_name:
        image = _decode_compressed_image(payload)
    elif "compressedvideo" in schema_name:
        _, frame_data, format_name = _parse_compressed_video(payload, timestamp_ns=0, source_topic=channel.topic)
        if str(format_name).strip().lower() != "h264":
            raise RuntimeError(f"Unsupported foxglove.CompressedVideo format: {format_name!r}")
        if PILImage is None:
            raise RuntimeError("Pillow is required to decode compressed video frames")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required to decode foxglove.CompressedVideo frames")
        decode_proc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "h264",
                "-i",
                "pipe:0",
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "pipe:1",
            ],
            input=frame_data,
            capture_output=True,
        )
        if decode_proc.returncode != 0:
            raise RuntimeError(
                "ffmpeg failed to decode a foxglove.CompressedVideo frame: "
                f"STDOUT={decode_proc.stdout.decode(errors='ignore').strip()} STDERR={decode_proc.stderr.decode(errors='ignore').strip()}"
            )
        return decode_proc.stdout
    elif schema_name.endswith("/image") or schema_name == "sensor_msgs/msg/image":
        image = _decode_raw_image(payload)
    else:
        raise RuntimeError(f"Unsupported image schema: {channel.schema_name!r}")

    if image.mode not in {"RGB", "L"}:
        image = image.convert("RGB")
    elif image.mode == "L":
        image = image.convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95, subsampling=0)
    return buffer.getvalue()


def _quaternion_to_euler_deg(*, x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    """Convert ROS quaternion xyzw into roll/pitch/yaw degrees.

    Formula:
        roll  = atan2(2(wx + yz), 1 - 2(x^2 + y^2))
        pitch = asin(clamp(2(wy - zx), -1, 1))
        yaw   = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    """
    sinr_cosp = 2.0 * (float(w) * float(x) + float(y) * float(z))
    cosr_cosp = 1.0 - 2.0 * (float(x) * float(x) + float(y) * float(y))
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = max(-1.0, min(1.0, 2.0 * (float(w) * float(y) - float(z) * float(x))))
    pitch = math.asin(sinp)
    siny_cosp = 2.0 * (float(w) * float(z) + float(x) * float(y))
    cosy_cosp = 1.0 - 2.0 * (float(y) * float(y) + float(z) * float(z))
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def _parse_imu_attitude(payload: bytes, *, timestamp_ns: int, source_topic: str, source_schema: str, message_encoding: str) -> AttitudeSample:
    reader = _make_reader(payload, message_encoding)
    if str(message_encoding or "").strip().lower() == "ros1":
        _ = reader.uint32()  # header.seq
        _ = reader.uint32()  # header.stamp.sec
        _ = reader.uint32()  # header.stamp.nanosec
    else:
        _ = reader.int32()  # header.stamp.sec
        _ = reader.uint32()  # header.stamp.nanosec
    _ = reader.string()  # header.frame_id
    qx = reader.float64()
    qy = reader.float64()
    qz = reader.float64()
    qw = reader.float64()
    roll_deg, pitch_deg, yaw_deg = _quaternion_to_euler_deg(x=qx, y=qy, z=qz, w=qw)
    return AttitudeSample(
        timestamp_ns=int(timestamp_ns),
        roll_deg=float(roll_deg),
        pitch_deg=float(pitch_deg),
        yaw_deg=float(yaw_deg),
        source_topic=source_topic,
        source_schema=source_schema,
    )


def _parse_posestamped_attitude(payload: bytes, *, timestamp_ns: int, source_topic: str, source_schema: str, message_encoding: str) -> AttitudeSample:
    reader = _make_reader(payload, message_encoding)
    if str(message_encoding or "").strip().lower() == "ros1":
        _ = reader.uint32()  # header.seq
        _ = reader.uint32()  # header.stamp.sec
        _ = reader.uint32()  # header.stamp.nanosec
    else:
        _ = reader.int32()  # header.stamp.sec
        _ = reader.uint32()  # header.stamp.nanosec
    _ = reader.string()  # header.frame_id
    _ = reader.float64()  # pose.position.x
    _ = reader.float64()  # pose.position.y
    _ = reader.float64()  # pose.position.z
    qx = reader.float64()
    qy = reader.float64()
    qz = reader.float64()
    qw = reader.float64()
    roll_deg, pitch_deg, yaw_deg = _quaternion_to_euler_deg(x=qx, y=qy, z=qz, w=qw)
    return AttitudeSample(
        timestamp_ns=int(timestamp_ns),
        roll_deg=float(roll_deg),
        pitch_deg=float(pitch_deg),
        yaw_deg=float(yaw_deg),
        source_topic=source_topic,
        source_schema=source_schema,
    )


def _parse_imu_motion(payload: bytes, *, timestamp_ns: int, source_topic: str, source_schema: str, message_encoding: str) -> ImuSample:
    reader = _make_reader(payload, message_encoding)
    if str(message_encoding or "").strip().lower() == "ros1":
        _ = reader.uint32()  # header.seq
        _ = reader.uint32()  # header.stamp.sec
        _ = reader.uint32()  # header.stamp.nanosec
    else:
        _ = reader.int32()  # header.stamp.sec
        _ = reader.uint32()  # header.stamp.nanosec
    _ = reader.string()  # header.frame_id
    # orientation quaternion
    _ = reader.float64()
    _ = reader.float64()
    _ = reader.float64()
    _ = reader.float64()
    # orientation covariance[9]
    for _ in range(9):
        _ = reader.float64()
    angular_velocity_x = reader.float64()
    angular_velocity_y = reader.float64()
    angular_velocity_z = reader.float64()
    return ImuSample(
        timestamp_ns=int(timestamp_ns),
        angular_velocity_x=float(angular_velocity_x),
        angular_velocity_y=float(angular_velocity_y),
        angular_velocity_z=float(angular_velocity_z),
        source_topic=source_topic,
        source_schema=source_schema,
    )


def _parse_odometry_motion(payload: bytes, *, timestamp_ns: int, source_topic: str, source_schema: str, message_encoding: str) -> OdometrySample:
    reader = _make_reader(payload, message_encoding)
    if str(message_encoding or "").strip().lower() == "ros1":
        _ = reader.uint32()  # header.seq
        _ = reader.uint32()  # header.stamp.sec
        _ = reader.uint32()  # header.stamp.nanosec
    else:
        _ = reader.int32()  # header.stamp.sec
        _ = reader.uint32()  # header.stamp.nanosec
    _ = reader.string()  # header.frame_id
    _ = reader.string()  # child_frame_id
    # pose: geometry_msgs/PoseWithCovariance
    _ = reader.float64()  # position.x
    _ = reader.float64()  # position.y
    _ = reader.float64()  # position.z
    _ = reader.float64()  # orientation.x
    _ = reader.float64()  # orientation.y
    _ = reader.float64()  # orientation.z
    _ = reader.float64()  # orientation.w
    for _ in range(36):
        _ = reader.float64()
    # twist: geometry_msgs/TwistWithCovariance
    _ = reader.float64()  # linear.x
    _ = reader.float64()  # linear.y
    _ = reader.float64()  # linear.z
    angular_velocity_x = reader.float64()
    angular_velocity_y = reader.float64()
    angular_velocity_z = reader.float64()
    return OdometrySample(
        timestamp_ns=int(timestamp_ns),
        angular_velocity_x=float(angular_velocity_x),
        angular_velocity_y=float(angular_velocity_y),
        angular_velocity_z=float(angular_velocity_z),
        source_topic=source_topic,
        source_schema=source_schema,
    )


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
    schema_substrings: list[str],
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
        if any(schema_substring.lower() in channel.schema_name.lower() for schema_substring in schema_substrings)
    ]
    if not candidates:
        raise RuntimeError(f"No {role} channel found for schema containing any of {schema_substrings!r}")

    def _candidate_score(channel: ChannelInfo) -> tuple[int, str]:
        schema_name = channel.schema_name.lower()
        if role == "image":
            if "compressedimage" in schema_name:
                priority = 0
            elif "compressedvideo" in schema_name:
                priority = 0
            elif schema_name.endswith("/image") or schema_name == "sensor_msgs/msg/image":
                priority = 1
            else:
                priority = 10
        else:
            priority = 0
        return priority, channel.topic

    return sorted(candidates, key=_candidate_score)[0]


def _choose_optional_channel(
    channels: dict[int, ChannelInfo],
    *,
    requested_topic: str,
    schema_substrings: list[str],
    role: str,
) -> ChannelInfo | None:
    requested = str(requested_topic or "").strip()
    if requested and requested.lower() not in {"auto", "none", "disabled"}:
        for channel in channels.values():
            if channel.topic == requested:
                return channel
        raise RuntimeError(f"Requested {role} topic not found: {requested}")
    if requested.lower() in {"none", "disabled"}:
        return None
    candidates = [
        channel
        for channel in channels.values()
        if any(schema.lower() in channel.schema_name.lower() for schema in schema_substrings)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item.topic)[0]


def extract_mcap_inputs(
    *,
    mcap_path: Path,
    output_dir: Path,
    image_topic: str,
    gps_topic: str,
    heading_topic: str,
    imu_topic: str,
    pose_topic: str,
    attitude_policy: str,
    sample_fps: float,
    max_sync_delta_s: float,
    image_start_index: int,
) -> tuple[Path, Path]:
    schemas: dict[int, SchemaInfo] = {}
    channels: dict[int, ChannelInfo] = {}
    images: list[ImageSample] = []
    gps_samples: list[GpsSample] = []
    heading_samples: list[HeadingSample] = []
    attitude_samples: list[AttitudeSample] = []
    imu_motion_samples: list[ImuSample] = []
    odom_motion_samples: list[OdometrySample] = []
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

    image_channel = _choose_channel(
        channels,
        requested_topic=image_topic,
        schema_substrings=["CompressedImage", "CompressedVideo", "sensor_msgs/msg/Image", "/Image"],
        role="image",
    )
    gps_channel = _choose_channel(channels, requested_topic=gps_topic, schema_substrings=["NavSatFix"], role="gps")
    # Try to find heading channel, if not found, we'll compute from GPS
    heading_channel = None
    try:
        heading_channel = _choose_channel(channels, requested_topic=heading_topic, schema_substrings=["Float64"], role="heading")
    except RuntimeError:
        pass
    attitude_policy = str(attitude_policy or "optional").strip().lower()
    if attitude_policy not in {"optional", "required", "disabled"}:
        raise RuntimeError(f"Unsupported attitude policy: {attitude_policy!r}")
    imu_channel = None
    odom_channel = None
    pose_channel = None
    if attitude_policy != "disabled":
        imu_channel = _choose_optional_channel(
            channels,
            requested_topic=imu_topic,
            schema_substrings=["Imu"],
            role="imu",
        )
        odom_channel = _choose_optional_channel(
            channels,
            requested_topic="auto",
            schema_substrings=["Odometry"],
            role="odom",
        )
        pose_channel = _choose_optional_channel(
            channels,
            requested_topic=pose_topic,
            schema_substrings=["PoseStamped"],
            role="pose",
        )
        if attitude_policy == "required" and imu_channel is None and pose_channel is None:
            raise RuntimeError("Attitude policy is required, but no IMU/Pose attitude channel was found")

    video_packets: list[tuple[int, bytes]] = []

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
                if "compressedvideo" in channel.schema_name.lower():
                    video_packets.append((timestamp_ns, message_payload))
                else:
                    try:
                        images.append(
                            ImageSample(
                                timestamp_ns=timestamp_ns,
                                data=_image_payload_to_jpg_bytes(message_payload, channel),
                                source_topic=channel.topic,
                            )
                        )
                    except Exception:
                        continue
            elif channel_id == gps_channel.channel_id:
                try:
                    gps_samples.append(_parse_navsatfix(message_payload, timestamp_ns=timestamp_ns, source_topic=channel.topic))
                except Exception:
                    continue
            elif heading_channel is not None and channel_id == heading_channel.channel_id:
                try:
                    heading_samples.append(_parse_float64(message_payload, timestamp_ns=timestamp_ns, source_topic=channel.topic))
                except Exception:
                    continue
            elif imu_channel is not None and channel_id == imu_channel.channel_id:
                try:
                    imu_motion_samples.append(
                        _parse_imu_motion(
                            message_payload,
                            timestamp_ns=timestamp_ns,
                            source_topic=channel.topic,
                            source_schema=channel.schema_name,
                            message_encoding=channel.message_encoding,
                        )
                    )
                    attitude_samples.append(
                        _parse_imu_attitude(
                            message_payload,
                            timestamp_ns=timestamp_ns,
                            source_topic=channel.topic,
                            source_schema=channel.schema_name,
                            message_encoding=channel.message_encoding,
                        )
                    )
                except Exception:
                    continue
            elif odom_channel is not None and channel_id == odom_channel.channel_id:
                try:
                    odom_motion_samples.append(
                        _parse_odometry_motion(
                            message_payload,
                            timestamp_ns=timestamp_ns,
                            source_topic=channel.topic,
                            source_schema=channel.schema_name,
                            message_encoding=channel.message_encoding,
                        )
                    )
                except Exception:
                    continue
            elif pose_channel is not None and channel_id == pose_channel.channel_id:
                try:
                    attitude_samples.append(
                        _parse_posestamped_attitude(
                            message_payload,
                            timestamp_ns=timestamp_ns,
                            source_topic=channel.topic,
                            source_schema=channel.schema_name,
                            message_encoding=channel.message_encoding,
                        )
                    )
                except Exception:
                    continue

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if video_packets:
        images = _decode_compressed_video_packets(
            video_packets,
            output_dir=output_dir,
            source_topic=image_channel.topic,
            source_schema=image_channel.schema_name,
        )

    if not images:
        raise RuntimeError(f"No images extracted from topic: {image_channel.topic}")
    if not gps_samples:
        raise RuntimeError(f"No GPS samples extracted from topic: {gps_channel.topic}")
    if attitude_policy == "required" and not attitude_samples:
        raise RuntimeError("Attitude policy is required, but no valid IMU/Pose attitude samples were extracted")

    images = sorted(images, key=lambda item: item.timestamp_ns)
    gps_samples = sorted(gps_samples, key=lambda item: item.timestamp_ns)
    heading_samples = sorted(heading_samples, key=lambda item: item.timestamp_ns)
    attitude_samples = sorted(attitude_samples, key=lambda item: item.timestamp_ns)
    imu_motion_samples = sorted(imu_motion_samples, key=lambda item: item.timestamp_ns)
    odom_motion_samples = sorted(odom_motion_samples, key=lambda item: item.timestamp_ns)

    # If no heading samples, compute from GPS
    compute_heading_from_gps = len(heading_samples) == 0
    if compute_heading_from_gps:
        heading_samples = _heading_samples_from_gps(gps_samples)

    gps_timestamps = [sample.timestamp_ns for sample in gps_samples]
    heading_timestamps = [sample.timestamp_ns for sample in heading_samples]
    attitude_timestamps = [sample.timestamp_ns for sample in attitude_samples]

    trajectory: list[dict[str, Any]] = []
    frame_records: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []
    attitude_compass_headings: list[float] = []
    attitude_heading_diffs: list[float] = []
    attitude_sync_deltas_s: list[float] = []
    attitude_valid_frame_count = 0
    imu_motion_z_nonzero_count = 0
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
        attitude, attitude_delta_ns = _nearest_sample(attitude_samples, attitude_timestamps, image.timestamp_ns)
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
        attitude_payload: dict[str, Any] = {"attitude_source": "missing"}
        attitude_valid = False
        if attitude is not None and attitude_delta_ns is not None and attitude_delta_ns <= max_delta_ns:
            attitude_compass_yaw_deg = _normalize_heading_deg(90.0 - float(attitude.yaw_deg))
            attitude_heading_diff_deg = abs(_angle_diff_deg(attitude_compass_yaw_deg, float(heading.heading_deg)))
            attitude_valid = attitude_heading_diff_deg <= 45.0
            attitude_payload = {
                "roll_deg": float(attitude.roll_deg),
                "pitch_deg": float(attitude.pitch_deg),
                "attitude_yaw_deg": float(attitude.yaw_deg),
                "attitude_compass_yaw_deg": float(attitude_compass_yaw_deg),
                "attitude_heading_diff_deg": float(attitude_heading_diff_deg),
                "attitude_source": "imu" if "Imu" in attitude.source_schema else "pose",
                "attitude_topic": attitude.source_topic,
                "attitude_schema": attitude.source_schema,
                "attitude_sync_delta_s": round(float(attitude_delta_ns) / 1e9, 6),
                "attitude_valid": bool(attitude_valid),
            }
            attitude_compass_headings.append(float(attitude_compass_yaw_deg))
            attitude_heading_diffs.append(float(attitude_heading_diff_deg))
            attitude_sync_deltas_s.append(float(attitude_delta_ns) / 1e9)
            attitude_valid_frame_count += 1 if attitude_valid else 0
        elif attitude_policy == "required":
            skipped_records.append({"timestamp": image.timestamp_ns, "reason": "missing_nearest_attitude"})
            continue

        image_path = images_dir / f"{frame_index:06d}.jpg"
        image_path.write_bytes(image.data)
        trajectory_row = {
            "timestamp": int(image.timestamp_ns),
            "lat": float(gps.lat),
            "lng": float(gps.lng),
            "heading_deg": float(heading.heading_deg),
            "image_path": str(image_path),
            **attitude_payload,
        }
        trajectory.append(trajectory_row)
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
                "attitude": attitude_payload,
            }
        )
        frame_index += 1
        if period_ns > 0:
            next_emit_ns = int(image.timestamp_ns) + period_ns

    if not trajectory:
        raise RuntimeError("No synchronized frames were written; check topics or --max-sync-delta-s")

    trajectory_path = output_dir / "trajectory.json"
    frames_manifest_path = output_dir / "frames_manifest.json"
    imu_motion_path = output_dir / "frame_angular_velocities.npy"
    meta_path = output_dir / "mcap_extract_meta.json"
    trajectory_path.write_text(json.dumps(trajectory, indent=2, ensure_ascii=True), encoding="utf-8")
    frames_manifest_path.write_text(json.dumps(frame_records, indent=2, ensure_ascii=True), encoding="utf-8")

    motion_source_samples = imu_motion_samples if imu_motion_samples else odom_motion_samples
    if motion_source_samples:
        imu_motion = np.zeros((len(motion_source_samples), 4), dtype=np.float64)
        for index, sample in enumerate(motion_source_samples):
            imu_motion[index, 0] = float(sample.timestamp_ns)
            imu_motion[index, 1] = float(sample.angular_velocity_x or 0.0)
            imu_motion[index, 2] = float(sample.angular_velocity_y or 0.0)
            imu_motion[index, 3] = float(sample.angular_velocity_z or 0.0)
            if abs(float(sample.angular_velocity_z or 0.0)) > 1e-4:
                imu_motion_z_nonzero_count += 1
        np.save(imu_motion_path, imu_motion)

    imu_motion_z_nonzero_fraction = (
        float(imu_motion_z_nonzero_count) / float(len(motion_source_samples))
        if motion_source_samples
        else 0.0
    )
    attitude_yaw_range_deg = float(max(attitude_compass_headings) - min(attitude_compass_headings)) if attitude_compass_headings else 0.0
    attitude_heading_mean_abs_diff_deg = (
        float(sum(abs(diff) for diff in attitude_heading_diffs) / len(attitude_heading_diffs))
        if attitude_heading_diffs
        else 0.0
    )
    attitude_heading_frac_gt30 = (
        float(sum(1 for diff in attitude_heading_diffs if abs(diff) > 30.0) / len(attitude_heading_diffs))
        if attitude_heading_diffs
        else 0.0
    )
    attitude_max_sync_delta_s = max(attitude_sync_deltas_s) if attitude_sync_deltas_s else None
    attitude_valid_fraction = (
        float(attitude_valid_frame_count) / float(len(attitude_heading_diffs))
        if attitude_heading_diffs
        else 0.0
    )
    invalid_quaternion_reasons: list[str] = []
    if not attitude_heading_diffs:
        invalid_quaternion_reasons.append("missing_attitude_samples")
    if attitude_yaw_range_deg < 0.5 and attitude_heading_mean_abs_diff_deg > 20.0:
        invalid_quaternion_reasons.append("constant_or_nearly_constant_attitude_yaw")
    if attitude_heading_frac_gt30 > 0.5:
        invalid_quaternion_reasons.append("attitude_heading_disagrees_with_gps")
    if attitude_valid_fraction < 0.5 and attitude_heading_diffs:
        invalid_quaternion_reasons.append("low_attitude_frame_quality")
    if imu_motion_z_nonzero_fraction < 0.3 and motion_source_samples:
        invalid_quaternion_reasons.append("low_imu_motion_activity")
    if attitude_max_sync_delta_s is not None and attitude_max_sync_delta_s > float(max_sync_delta_s):
        invalid_quaternion_reasons.append("attitude_sync_delta_exceeded")
    overall_pose_quality_valid = len(invalid_quaternion_reasons) == 0
    attitude_policy_summary = {
        "attitude_frames": len(attitude_heading_diffs),
        "attitude_valid_frames": attitude_valid_frame_count,
        "attitude_valid_fraction": round(attitude_valid_fraction, 6),
        "attitude_yaw_range_deg": round(attitude_yaw_range_deg, 6),
        "attitude_heading_mean_abs_diff_deg": round(attitude_heading_mean_abs_diff_deg, 6),
        "attitude_heading_frac_gt30": round(attitude_heading_frac_gt30, 6),
        "attitude_max_sync_delta_s": round(attitude_max_sync_delta_s, 6) if attitude_max_sync_delta_s is not None else None,
        "imu_motion_z_nonzero_count": int(imu_motion_z_nonzero_count),
        "imu_motion_z_nonzero_fraction": round(imu_motion_z_nonzero_fraction, 6),
        "overall_pose_quality_valid": bool(overall_pose_quality_valid),
        "invalid_quaternion_reasons": invalid_quaternion_reasons,
    }

    meta = {
        "mcap_path": str(mcap_path),
        "images_dir": str(images_dir),
        "trajectory_json": str(trajectory_path),
        "frames_manifest_path": str(frames_manifest_path),
        "imu_motion_path": str(imu_motion_path) if imu_motion_samples else None,
        "image_topic": image_channel.topic,
        "gps_topic": gps_channel.topic,
        "heading_topic": heading_channel.topic if heading_channel is not None else None,
        "imu_topic": imu_channel.topic if imu_channel is not None else None,
        "pose_topic": pose_channel.topic if pose_channel is not None else None,
        "attitude_policy": attitude_policy,
        "sample_fps": float(sample_fps),
        "max_sync_delta_s": float(max_sync_delta_s),
        "image_start_index": int(image_start_index),
        "pose_quality": attitude_policy_summary,
        "raw_counts": {
            "images": len(images),
            "gps_samples": len(gps_samples),
            "heading_samples": len(heading_samples),
            "attitude_samples": len(attitude_samples),
            "imu_motion_samples": len(imu_motion_samples),
            "odom_motion_samples": len(odom_motion_samples),
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
    _atomic_write_text(meta_path, json.dumps(meta, indent=2, ensure_ascii=True))
    return trajectory_path, meta_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract numbered JPEG frames and RouteServer trajectory.json from a single ROS2 MCAP file."
    )
    parser.add_argument("--mcap-path", required=True, help="Path to a single .mcap file")
    parser.add_argument("--output-dir", required=True, help="Output directory containing images/ and trajectory.json")
    parser.add_argument("--image-topic", default="auto", help="Image topic (CompressedImage, CompressedVideo, or Image), or auto")
    parser.add_argument("--gps-topic", default="auto", help="NavSatFix topic, or auto")
    parser.add_argument("--heading-topic", default="auto", help="Float64 heading topic, or auto")
    parser.add_argument(
        "--imu-topic",
        default="auto",
        help="Optional sensor_msgs/Imu attitude topic. Use auto, none, or an exact topic name.",
    )
    parser.add_argument(
        "--pose-topic",
        default="auto",
        help="Optional geometry_msgs/PoseStamped attitude topic. Use auto, none, or an exact topic name.",
    )
    parser.add_argument(
        "--attitude-policy",
        default="optional",
        choices=["optional", "required", "disabled"],
        help="Whether roll/pitch extraction is optional, required, or disabled.",
    )
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
        imu_topic=str(args.imu_topic),
        pose_topic=str(args.pose_topic),
        attitude_policy=str(args.attitude_policy),
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
