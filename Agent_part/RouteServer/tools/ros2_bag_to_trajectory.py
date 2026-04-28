#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import bisect
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

_ROS2_IMPORT_ERROR: Exception | None = None
try:
    import numpy as np
    from PIL import Image as PILImage
    from nav_msgs.msg import Odometry  # type: ignore[import-not-found]
    from rclpy.serialization import deserialize_message  # type: ignore[import-not-found]
    from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions  # type: ignore[import-not-found]
    from sensor_msgs.msg import CompressedImage, Image  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover - handled at runtime entry.
    _ROS2_IMPORT_ERROR = exc
    np = None  # type: ignore[assignment]
    PILImage = None  # type: ignore[assignment]
    Odometry = None  # type: ignore[assignment]
    deserialize_message = None  # type: ignore[assignment]
    ConverterOptions = None  # type: ignore[assignment]
    SequentialReader = None  # type: ignore[assignment]
    StorageOptions = None  # type: ignore[assignment]
    CompressedImage = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]


EARTH_METERS_PER_DEG_LAT = 110540.0
EARTH_METERS_PER_DEG_LON_AT_EQUATOR = 111320.0


@dataclass(frozen=True)
class OdomSample:
    timestamp_ns: int
    x: float
    y: float
    qx: float
    qy: float
    qz: float
    qw: float


@dataclass(frozen=True)
class ImageSample:
    timestamp_ns: int
    image_path: str


@dataclass(frozen=True)
class BagTopicInfo:
    name: str
    type: str
    message_count: int | None = None


def _require_ros2_dependencies() -> None:
    if _ROS2_IMPORT_ERROR is not None:
        raise RuntimeError(
            "ROS2 Python dependencies are not available. "
            "This script needs rclpy + rosbag2_py from a ROS2 environment. "
            "Try sourcing your ROS2 setup before running, e.g.:\n"
            "  source /opt/ros/humble/setup.bash\n"
            "  python3 tools/ros2_bag_to_trajectory.py ...\n"
            f"Original import error: {_ROS2_IMPORT_ERROR!r}"
        )


def _load_yaml_text(path: Path) -> Any:
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _parse_topics_from_metadata_text(text: str) -> list[BagTopicInfo]:
    topics: list[BagTopicInfo] = []
    current: dict[str, Any] | None = None
    in_topics = False
    indent_level = None

    lines = text.splitlines()
    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped == "topics_with_message_count:":
            in_topics = True
            indent_level = None
            continue
        if not in_topics:
            continue
        if stripped.startswith("relative_file_paths:") or stripped.startswith("files:"):
            break
        if stripped.startswith("- topic_metadata:"):
            if current and current.get("name") and current.get("type"):
                topics.append(
                    BagTopicInfo(
                        name=str(current["name"]),
                        type=str(current["type"]),
                        message_count=(int(current["message_count"]) if current.get("message_count") is not None else None),
                    )
                )
            current = {}
            indent_level = len(raw_line) - len(raw_line.lstrip())
            continue
        if current is None:
            continue
        if indent_level is None:
            indent_level = len(raw_line) - len(raw_line.lstrip())
        if len(raw_line) - len(raw_line.lstrip()) < indent_level:
            continue
        if "name:" in stripped and stripped.startswith("name:"):
            current["name"] = stripped.split("name:", 1)[1].strip()
        elif "type:" in stripped and stripped.startswith("type:"):
            current["type"] = stripped.split("type:", 1)[1].strip()
        elif "message_count:" in stripped and stripped.startswith("message_count:"):
            try:
                current["message_count"] = int(stripped.split("message_count:", 1)[1].strip())
            except Exception:
                current["message_count"] = None

    if current and current.get("name") and current.get("type"):
        topics.append(
            BagTopicInfo(
                name=str(current["name"]),
                type=str(current["type"]),
                message_count=(int(current["message_count"]) if current.get("message_count") is not None else None),
            )
        )
    return topics


def _load_bag_topics(bag_path: Path) -> list[BagTopicInfo]:
    metadata_path = bag_path / "metadata.yaml"
    if not metadata_path.exists():
        raise RuntimeError(f"metadata.yaml not found under bag path: {bag_path}")

    parsed = _load_yaml_text(metadata_path)
    topics: list[BagTopicInfo] = []
    if isinstance(parsed, dict):
        bag_info = parsed.get("rosbag2_bagfile_information") if isinstance(parsed.get("rosbag2_bagfile_information"), dict) else parsed
        topic_entries = bag_info.get("topics_with_message_count") if isinstance(bag_info, dict) else None
        if isinstance(topic_entries, list):
            for item in topic_entries:
                if not isinstance(item, dict):
                    continue
                topic_meta = item.get("topic_metadata") if isinstance(item.get("topic_metadata"), dict) else item
                if not isinstance(topic_meta, dict):
                    continue
                name = topic_meta.get("name")
                type_name = topic_meta.get("type")
                if not name or not type_name:
                    continue
                message_count = item.get("message_count")
                if message_count is None:
                    message_count = topic_meta.get("message_count")
                try:
                    message_count_int = int(message_count) if message_count is not None else None
                except Exception:
                    message_count_int = None
                topics.append(BagTopicInfo(name=str(name), type=str(type_name), message_count=message_count_int))
            if topics:
                return topics

    return _parse_topics_from_metadata_text(metadata_path.read_text(encoding="utf-8"))


def _score_topic_candidate(topic: BagTopicInfo, *, kind: str) -> tuple[int, int, str]:
    type_name = topic.type.lower()
    count = int(topic.message_count or 0)
    if kind == "image":
        preference = 0
        if "compressedimage" in type_name:
            preference = 0
        elif "sensor_msgs/msg/image" in type_name:
            preference = 1
        else:
            preference = 10
        return preference, -count, topic.name
    if kind == "odom":
        preference = 0 if "nav_msgs/msg/odometry" in type_name else 10
        return preference, -count, topic.name
    return 100, -count, topic.name


def _find_topic_candidates(topics: list[BagTopicInfo], kind: str) -> list[BagTopicInfo]:
    if kind == "image":
        return [topic for topic in topics if any(key in topic.type.lower() for key in ["sensor_msgs/msg/compressedimage", "sensor_msgs/msg/image"])]
    if kind == "odom":
        return [topic for topic in topics if "nav_msgs/msg/odometry" in topic.type.lower()]
    return []


def _print_topics(topics: list[BagTopicInfo]) -> None:
    print("Available topics in bag:")
    for index, topic in enumerate(sorted(topics, key=lambda item: (item.type, item.name))):
        count_text = f" ({topic.message_count} msgs)" if topic.message_count is not None else ""
        print(f"  [{index}] {topic.name} :: {topic.type}{count_text}")


def _choose_topic(topics: list[BagTopicInfo], *, kind: str, requested: str | None) -> BagTopicInfo:
    normalized = str(requested or "").strip()
    if normalized and normalized.lower() not in {"auto", "ask"}:
        for topic in topics:
            if topic.name == normalized:
                return topic
        raise RuntimeError(f"Requested {kind} topic not found in bag: {normalized}")

    candidates = _find_topic_candidates(topics, kind)
    if not candidates:
        raise RuntimeError(f"No {kind} topic candidates found in bag topics")

    candidates = sorted(candidates, key=lambda item: _score_topic_candidate(item, kind=kind))
    if len(candidates) == 1:
        chosen = candidates[0]
        print(f"Auto-selected {kind} topic: {chosen.name} ({chosen.type})")
        return chosen

    _print_topics(candidates)
    if sys.stdin.isatty():
        while True:
            raw = input(f"Select {kind} topic index [0-{len(candidates) - 1}] (default 0): ").strip()
            if raw == "":
                return candidates[0]
            try:
                selection = int(raw)
            except Exception:
                print("Please enter a numeric index.")
                continue
            if 0 <= selection < len(candidates):
                return candidates[selection]
            print("Index out of range.")

    chosen = candidates[0]
    print(f"Non-interactive session; defaulting to {kind} topic: {chosen.name} ({chosen.type})")
    return chosen


def _load_message_type(topic_type: str):
    normalized = topic_type.lower()
    if "compressedimage" in normalized:
        return CompressedImage
    if normalized.endswith("/image") or normalized == "sensor_msgs/msg/image":
        return Image
    if "odometry" in normalized:
        return Odometry
    raise RuntimeError(f"Unsupported ROS message type: {topic_type}")


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


def _decode_compressed_image(msg: Any) -> Any:
    if PILImage is None:
        raise RuntimeError("Pillow is required to decode compressed images")
    image = PILImage.open(io.BytesIO(bytes(msg.data)))
    return image.convert("RGB")


def _decode_raw_image(msg: Any) -> Any:
    if np is None or PILImage is None:
        raise RuntimeError("numpy and Pillow are required to decode sensor_msgs/Image")
    channels, dtype_name = _encoding_channels(getattr(msg, "encoding", ""))
    dtype = np.dtype(dtype_name)
    bytes_per_pixel = dtype.itemsize * channels
    width = int(msg.width)
    height = int(msg.height)
    step = int(msg.step)
    row_pixels = step // dtype.itemsize
    buffer = np.frombuffer(msg.data, dtype=dtype)
    if int(getattr(msg, "is_bigendian", 0)) == 1 and dtype.itemsize > 1:
        buffer = buffer.byteswap().newbyteorder()
    if step < width * bytes_per_pixel:
        raise RuntimeError(
            f"Invalid image step for encoding={msg.encoding!r}: step={step}, width={width}, bytes_per_pixel={bytes_per_pixel}"
        )
    if channels == 1:
        row = buffer.reshape((height, row_pixels))[:, :width]
        if dtype == np.dtype("uint16"):
            return PILImage.fromarray(row, mode="I;16")
        return PILImage.fromarray(row.astype(np.uint8), mode="L")

    row = buffer.reshape((height, row_pixels))[:, : width * channels]
    arr = row.reshape((height, width, channels))
    enc = str(msg.encoding).strip().lower()
    if enc == "bgr8":
        arr = arr[:, :, ::-1]
    elif enc == "bgra8":
        arr = arr[:, :, [2, 1, 0, 3]]
    elif enc == "bgr16":
        arr = arr[:, :, ::-1]
    elif enc == "bgra16":
        arr = arr[:, :, [2, 1, 0, 3]]
    elif enc not in {"rgb8", "rgba8", "rgb16", "rgba16"}:
        raise RuntimeError(f"Unsupported image encoding transformation: {msg.encoding!r}")
    return PILImage.fromarray(arr)


def _save_ros_image_to_jpg(msg: Any, output_path: Path) -> dict[str, Any]:
    if CompressedImage is not None and isinstance(msg, CompressedImage):
        image = _decode_compressed_image(msg)
        image.save(output_path, format="JPEG", quality=95, subsampling=0)
        return {
            "topic_type": "sensor_msgs/msg/CompressedImage",
            "encoding": getattr(msg, "format", None),
            "image_width": int(image.size[0]),
            "image_height": int(image.size[1]),
        }

    if Image is not None and isinstance(msg, Image):
        image = _decode_raw_image(msg)
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        elif image.mode == "L":
            image = image.convert("RGB")
        image.save(output_path, format="JPEG", quality=95, subsampling=0)
        return {
            "topic_type": "sensor_msgs/msg/Image",
            "encoding": getattr(msg, "encoding", None),
            "image_width": int(image.size[0]),
            "image_height": int(image.size[1]),
        }

    raise RuntimeError(f"Unsupported image message type: {type(msg)!r}")


def _quaternion_to_yaw_deg(qx: float, qy: float, qz: float, qw: float) -> float:
    # Yaw from quaternion (Z-axis rotation), returned in degrees.
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def _bearing_from_yaw_deg(yaw_deg: float) -> float:
    # Convert ENU yaw (0=East, CCW positive) to compass bearing (0=North, CW positive).
    return (90.0 - float(yaw_deg)) % 360.0


def _nearest_odom(
    odom_samples: list[OdomSample],
    odom_timestamps: list[int],
    target_ts_ns: int,
) -> OdomSample:
    right = bisect.bisect_left(odom_timestamps, int(target_ts_ns))
    if right <= 0:
        return odom_samples[0]
    if right >= len(odom_samples):
        return odom_samples[-1]
    left = right - 1
    if abs(odom_timestamps[left] - target_ts_ns) <= abs(odom_timestamps[right] - target_ts_ns):
        return odom_samples[left]
    return odom_samples[right]


def _build_reader(bag_path: Path, storage_id: str) -> Any:
    reader = SequentialReader()
    storage_options = StorageOptions(uri=str(bag_path), storage_id=str(storage_id))
    converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader.open(storage_options, converter_options)
    return reader


def _collect_samples(
    *,
    bag_path: Path,
    storage_id: str,
    image_topic: BagTopicInfo,
    odom_topic: BagTopicInfo,
    output_images_dir: Path,
    sample_fps: float,
    image_start_index: int,
) -> tuple[list[ImageSample], list[OdomSample], list[dict[str, Any]]]:
    reader = _build_reader(bag_path, storage_id)
    output_images_dir.mkdir(parents=True, exist_ok=True)

    image_samples: list[ImageSample] = []
    odom_samples: list[OdomSample] = []
    frame_records: list[dict[str, Any]] = []

    image_index = int(image_start_index)
    period_ns = 0 if sample_fps <= 0.0 else int(round(1e9 / float(sample_fps)))
    next_emit_ns: int | None = None
    image_msg_type = _load_message_type(image_topic.type)
    odom_msg_type = _load_message_type(odom_topic.type)

    while reader.has_next():
        topic, serialized_data, timestamp_ns = reader.read_next()
        if topic == odom_topic.name:
            msg = deserialize_message(serialized_data, odom_msg_type)
            pose = msg.pose.pose
            odom_samples.append(
                OdomSample(
                    timestamp_ns=int(timestamp_ns),
                    x=float(pose.position.x),
                    y=float(pose.position.y),
                    qx=float(pose.orientation.x),
                    qy=float(pose.orientation.y),
                    qz=float(pose.orientation.z),
                    qw=float(pose.orientation.w),
                )
            )
            continue

        if topic != image_topic.name:
            continue

        ts_ns = int(timestamp_ns)
        if next_emit_ns is None:
            next_emit_ns = ts_ns
        if period_ns > 0 and ts_ns < next_emit_ns:
            continue

        msg = deserialize_message(serialized_data, image_msg_type)
        image_file = output_images_dir / f"{image_index:06d}.jpg"
        image_meta = _save_ros_image_to_jpg(msg, image_file)
        image_samples.append(ImageSample(timestamp_ns=ts_ns, image_path=str(image_file)))
        frame_records.append(
            {
                "frame_index": image_index,
                "timestamp": ts_ns,
                "image_path": str(image_file),
                "image_topic": image_topic.name,
                "image_topic_type": image_topic.type,
                "odom_topic": odom_topic.name,
                **image_meta,
            }
        )
        image_index += 1

        if period_ns > 0:
            next_emit_ns = ts_ns + period_ns

    return image_samples, odom_samples, frame_records


def _trajectory_from_samples(
    *,
    image_samples: list[ImageSample],
    odom_samples: list[OdomSample],
    start_lat: float,
    start_lng: float,
    odom_to_enu_yaw_deg: float,
    heading_offset_deg: float,
) -> list[dict[str, Any]]:
    if not image_samples:
        raise RuntimeError("No sampled image messages were collected from the bag.")
    if not odom_samples:
        raise RuntimeError("No odometry messages were collected from the bag.")

    origin = odom_samples[0]
    odom_timestamps = [item.timestamp_ns for item in odom_samples]
    theta = math.radians(float(odom_to_enu_yaw_deg))

    lat0 = float(start_lat)
    lng0 = float(start_lng)
    cos_lat0 = math.cos(math.radians(lat0))
    if abs(cos_lat0) < 1e-6:
        raise RuntimeError("start_lat is too close to poles for local equirectangular conversion")

    trajectory: list[dict[str, Any]] = []
    for item in image_samples:
        odom = _nearest_odom(odom_samples, odom_timestamps, item.timestamp_ns)

        dx = float(odom.x - origin.x)
        dy = float(odom.y - origin.y)

        # Rotate odom XY into East/North axes.
        east_m = dx * math.cos(theta) - dy * math.sin(theta)
        north_m = dx * math.sin(theta) + dy * math.cos(theta)

        lat = lat0 + (north_m / EARTH_METERS_PER_DEG_LAT)
        lng = lng0 + (east_m / (EARTH_METERS_PER_DEG_LON_AT_EQUATOR * cos_lat0))

        yaw_odom_deg = _quaternion_to_yaw_deg(odom.qx, odom.qy, odom.qz, odom.qw)
        yaw_enu_deg = yaw_odom_deg + float(odom_to_enu_yaw_deg)
        heading_deg = (_bearing_from_yaw_deg(yaw_enu_deg) + float(heading_offset_deg)) % 360.0

        trajectory.append(
            {
                "timestamp": int(item.timestamp_ns),
                "lat": float(lat),
                "lng": float(lng),
                "heading_deg": float(heading_deg),
                "image_path": str(item.image_path),
            }
        )

    return trajectory


def _write_outputs(
    *,
    output_dir: Path,
    trajectory: list[dict[str, Any]],
    frame_records: list[dict[str, Any]],
    image_topic: str,
    odom_topic: str,
    bag_path: Path,
    sample_fps: float,
    start_lat: float,
    start_lng: float,
    odom_to_enu_yaw_deg: float,
    heading_offset_deg: float,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)

    trajectory_path = output_dir / "trajectory.json"
    meta_path = output_dir / "adapter_meta.json"
    frames_manifest_path = output_dir / "frames_manifest.json"
    route_context_capture_input_path = output_dir / "route_context_capture_input.json"

    trajectory_path.write_text(json.dumps(trajectory, indent=2, ensure_ascii=True), encoding="utf-8")
    frames_manifest_path.write_text(json.dumps(frame_records, indent=2, ensure_ascii=True), encoding="utf-8")

    meta = {
        "bag_path": str(bag_path),
        "image_topic": str(image_topic),
        "odom_topic": str(odom_topic),
        "sample_fps": float(sample_fps),
        "start_lat": float(start_lat),
        "start_lng": float(start_lng),
        "odom_to_enu_yaw_deg": float(odom_to_enu_yaw_deg),
        "heading_offset_deg": float(heading_offset_deg),
        "frames": len(trajectory),
        "frames_manifest_path": str(frames_manifest_path),
        "fields": ["timestamp", "lat", "lng", "heading_deg", "image_path"],
        "route_context_capture_ready": {
            "trajectory_path": str(trajectory_path),
            "images_dir": str(output_dir / "images"),
            "notes": "Pass trajectory.json directly to tools/route_context_capture.py",
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
    route_context_capture_input_path.write_text(
        json.dumps(
            {
                "trajectory": str(trajectory_path),
                "images_dir": str(output_dir / "images"),
                "meta": str(meta_path),
                "frames_manifest": str(frames_manifest_path),
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    return trajectory_path, meta_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a ROS2 bag (sqlite3 rosbag2) into sampled frame images and a "
            "RouteServer-compatible trajectory.json for route_context_capture."
        )
    )
    parser.add_argument("--bag-path", required=True, help="Path to ROS2 bag directory containing metadata.yaml")
    parser.add_argument("--start-lat", type=float, required=True, help="Anchor latitude at odom origin")
    parser.add_argument("--start-lng", type=float, required=True, help="Anchor longitude at odom origin")
    parser.add_argument("--output-dir", required=True, help="Output directory for images and trajectory.json")
    parser.add_argument("--sample-fps", type=float, default=1.0, help="Target image sampling fps (<=0 keeps all images)")
    parser.add_argument("--image-topic", default="auto", help="Image topic name in bag, or 'auto' to detect/prompt")
    parser.add_argument("--odom-topic", default="auto", help="Odometry topic name in bag, or 'auto' to detect/prompt")
    parser.add_argument("--storage-id", default="sqlite3", help="rosbag2 storage id (default: sqlite3)")
    parser.add_argument(
        "--odom-to-enu-yaw-deg",
        type=float,
        default=0.0,
        help=(
            "Rotation (deg) from odom XY to ENU (East/North), CCW positive. "
            "0 means odom +X=East, +Y=North."
        ),
    )
    parser.add_argument(
        "--heading-offset-deg",
        type=float,
        default=0.0,
        help="Additional offset added to computed heading (deg)",
    )
    parser.add_argument(
        "--image-start-index",
        type=int,
        default=1,
        help="Start index for output image filenames (default: 1 -> 000001.jpg)",
    )
    return parser.parse_args()


def main() -> None:
    if _ROS2_IMPORT_ERROR is not None:
        raise RuntimeError(
            "ROS2 Python dependencies are not available. "
            "This script needs rclpy + rosbag2_py from a ROS2 environment. "
            "Try sourcing your ROS2 setup before running, e.g. '\n"
            "  source /opt/ros/humble/setup.bash\n"
            "  python3 tools/ros2_bag_to_trajectory.py ...'\n"
            f"Original import error: {_ROS2_IMPORT_ERROR!r}"
        )

    args = parse_args()
    _require_ros2_dependencies()

    bag_path = Path(args.bag_path).expanduser().resolve()
    if not bag_path.exists() or not bag_path.is_dir():
        raise RuntimeError(f"Invalid --bag-path: {bag_path}")
    if not (bag_path / "metadata.yaml").exists():
        raise RuntimeError(f"metadata.yaml not found under bag path: {bag_path}")

    topics = _load_bag_topics(bag_path)
    if not topics:
        raise RuntimeError(f"No topics found in bag metadata: {bag_path}")
    image_topic = _choose_topic(topics, kind="image", requested=str(args.image_topic))
    odom_topic = _choose_topic(topics, kind="odom", requested=str(args.odom_topic))
    print(f"Selected image topic: {image_topic.name} :: {image_topic.type}")
    print(f"Selected odom topic: {odom_topic.name} :: {odom_topic.type}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    images_dir = output_dir / "images"

    image_samples, odom_samples, frame_records = _collect_samples(
        bag_path=bag_path,
        storage_id=str(args.storage_id),
        image_topic=image_topic,
        odom_topic=odom_topic,
        output_images_dir=images_dir,
        sample_fps=float(args.sample_fps),
        image_start_index=int(args.image_start_index),
    )

    trajectory = _trajectory_from_samples(
        image_samples=image_samples,
        odom_samples=odom_samples,
        start_lat=float(args.start_lat),
        start_lng=float(args.start_lng),
        odom_to_enu_yaw_deg=float(args.odom_to_enu_yaw_deg),
        heading_offset_deg=float(args.heading_offset_deg),
    )

    trajectory_path, meta_path = _write_outputs(
        output_dir=output_dir,
        trajectory=trajectory,
        frame_records=frame_records,
        image_topic=image_topic.name,
        odom_topic=odom_topic.name,
        bag_path=bag_path,
        sample_fps=float(args.sample_fps),
        start_lat=float(args.start_lat),
        start_lng=float(args.start_lng),
        odom_to_enu_yaw_deg=float(args.odom_to_enu_yaw_deg),
        heading_offset_deg=float(args.heading_offset_deg),
    )

    print(f"Done. sampled_frames={len(trajectory)}")
    print(f"images_dir={images_dir}")
    print(f"trajectory_json={trajectory_path}")
    print(f"meta_json={meta_path}")


if __name__ == "__main__":
    main()
