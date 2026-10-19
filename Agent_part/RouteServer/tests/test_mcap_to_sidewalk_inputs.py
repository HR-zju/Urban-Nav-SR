from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "dataset_processing" / "mcap_to_sidewalk_inputs.py"
_SPEC = importlib.util.spec_from_file_location("mcap_to_sidewalk_inputs_test", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

ChannelInfo = _MODULE.ChannelInfo
_choose_channel = _MODULE._choose_channel
_encoding_channels = _MODULE._encoding_channels


class McapToSidewalkInputsTest(unittest.TestCase):
    def test_choose_channel_prefers_compressed_image_but_accepts_raw_image(self) -> None:
        channels = {
            1: ChannelInfo(
                channel_id=1,
                schema_id=1,
                topic="/camera/image_raw",
                message_encoding="cdr",
                schema_name="sensor_msgs/msg/Image",
            ),
            2: ChannelInfo(
                channel_id=2,
                schema_id=2,
                topic="/camera/image_raw/compressed",
                message_encoding="cdr",
                schema_name="sensor_msgs/msg/CompressedImage",
            ),
        }

        chosen = _choose_channel(
            channels,
            requested_topic="auto",
            schema_substrings=["CompressedImage", "sensor_msgs/msg/Image", "/Image"],
            role="image",
        )

        self.assertEqual(chosen.topic, "/camera/image_raw/compressed")

    def test_choose_channel_accepts_raw_image_when_compressed_is_missing(self) -> None:
        channels = {
            1: ChannelInfo(
                channel_id=1,
                schema_id=1,
                topic="/camera/image_raw",
                message_encoding="cdr",
                schema_name="sensor_msgs/msg/Image",
            )
        }

        chosen = _choose_channel(
            channels,
            requested_topic="auto",
            schema_substrings=["CompressedImage", "sensor_msgs/msg/Image", "/Image"],
            role="image",
        )

        self.assertEqual(chosen.topic, "/camera/image_raw")

    def test_choose_channel_accepts_compressed_video_when_no_image_schema_exists(self) -> None:
        channels = {
            1: ChannelInfo(
                channel_id=1,
                schema_id=1,
                topic="robot.h264_video.front",
                message_encoding="cdr",
                schema_name="foxglove.CompressedVideo",
            )
        }

        chosen = _choose_channel(
            channels,
            requested_topic="auto",
            schema_substrings=["CompressedImage", "CompressedVideo", "sensor_msgs/msg/Image", "/Image"],
            role="image",
        )

        self.assertEqual(chosen.topic, "robot.h264_video.front")

    def test_encoding_channels_rejects_unknown_encoding(self) -> None:
        with self.assertRaises(RuntimeError):
            _encoding_channels("weird12")


if __name__ == "__main__":
    unittest.main()
