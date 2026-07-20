from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

from mcap.reader import make_reader
from mcap_ros2.reader import read_ros2_messages


def _resolve_mcap_files(path: Path) -> List[Path]:
    """Resolve a standalone MCAP file or a rosbag2 directory containing MCAP files."""
    path = path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".mcap":
            raise ValueError(
                "Python 3.8 compatibility build supports MCAP only; "
                "the selected file is not .mcap: {}".format(path)
            )
        return [path]

    if not path.is_dir():
        raise FileNotFoundError("Bag path does not exist: {}".format(path))

    files = sorted(p.resolve() for p in path.glob("*.mcap") if p.is_file())
    if not files:
        # Some datasets place split files in one nested directory.
        files = sorted(p.resolve() for p in path.rglob("*.mcap") if p.is_file())
    if not files:
        raise FileNotFoundError("No .mcap files found under: {}".format(path))
    return files


def _topics_from_file(path: Path):
    """Read topic/type metadata without decoding message payloads."""
    result = {}
    with path.open("rb") as stream:
        reader = make_reader(stream)
        summary = reader.get_summary()
        if summary is not None:
            for channel in summary.channels.values():
                schema = summary.schemas.get(channel.schema_id)
                type_name = schema.name if schema is not None else ""
                result.setdefault(channel.topic, type_name)
            return result

        # Unindexed MCAP fallback. This may scan the file once, but still avoids decoding.
        for schema, channel, _message in reader.iter_messages():
            type_name = schema.name if schema is not None else ""
            result.setdefault(channel.topic, type_name)
    return result


class RosbagReader:
    """Pure-Python ROS2 MCAP reader.

    This backend uses ``mcap`` and ``mcap-ros2-support`` only. It does not
    import ROS, rclpy, rosbag2_py, cv_bridge, or rosbag2_storage_mcap.

    Supported inputs:
      * standalone ``.mcap`` files;
      * rosbag2 directories containing one or more ``.mcap`` files.

    The MCAP must contain ``ros2msg`` schemas, which is the normal output of
    rosbag2_storage_mcap. Custom message types are decoded from the embedded
    schema and therefore do not need ROS packages installed on the NAS.
    """

    def __init__(self, bag_path, storage_id="", reader_config=None):
        del storage_id
        del reader_config
        self.bag_path = Path(bag_path).expanduser().resolve()
        self.mcap_files = []
        self.topic_types = {}
        self._opened = False

    def open(self) -> None:
        self.mcap_files = _resolve_mcap_files(self.bag_path)
        topic_types = {}
        for mcap_path in self.mcap_files:
            for topic, type_name in _topics_from_file(mcap_path).items():
                old = topic_types.get(topic)
                if old and type_name and old != type_name:
                    raise RuntimeError(
                        "Topic {!r} has inconsistent types across split MCAP files: "
                        "{!r} and {!r}".format(topic, old, type_name)
                    )
                topic_types.setdefault(topic, type_name)
        self.topic_types = topic_types
        self._opened = True

    def iter_messages(self, selected_topics) -> Iterator:
        if not self._opened:
            raise RuntimeError("Reader is not open")
        topics = sorted(selected_topics)
        if not topics:
            return

        for mcap_path in self.mcap_files:
            try:
                iterator = read_ros2_messages(str(mcap_path), topics=topics)
                for record in iterator:
                    schema = record.schema
                    type_name = schema.name if schema is not None else ""
                    yield (
                        record.channel.topic,
                        record.ros_msg,
                        int(record.log_time_ns),
                        type_name,
                    )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to decode ROS2 messages from {}. The MCAP must embed "
                    "ros2msg schemas. Original error: {}".format(mcap_path, exc)
                ) from exc

    def close(self) -> None:
        self._opened = False
        self.mcap_files = []
