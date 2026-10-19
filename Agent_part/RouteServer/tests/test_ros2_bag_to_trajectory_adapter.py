from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "dataset_processing" / "ros2_bag_to_trajectory.py"
_SPEC = importlib.util.spec_from_file_location("ros2_bag_to_trajectory", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

BagTopicInfo = _MODULE.BagTopicInfo
_choose_topic = _MODULE._choose_topic
_load_bag_topics = _MODULE._load_bag_topics
_write_outputs = _MODULE._write_outputs


class Ros2BagToTrajectoryAdapterTest(unittest.TestCase):
    def test_load_bag_topics_parses_metadata_yaml(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            bag_path = Path(tmp)
            (bag_path / "metadata.yaml").write_text(
                """
rosbag2_bagfile_information:
  topics_with_message_count:
    - topic_metadata:
        name: /camera/camera/color/image_raw
        type: sensor_msgs/msg/Image
      message_count: 10
    - topic_metadata:
        name: /camera/camera/color/image_raw/compressed
        type: sensor_msgs/msg/CompressedImage
      message_count: 20
    - topic_metadata:
        name: /globalEstimator/global_odometry
        type: nav_msgs/msg/Odometry
      message_count: 30
""".strip(),
                encoding="utf-8",
            )

            topics = _load_bag_topics(bag_path)
            self.assertEqual(len(topics), 3)
            self.assertEqual(topics[0].name, "/camera/camera/color/image_raw")
            self.assertEqual(topics[2].type, "nav_msgs/msg/Odometry")

    def test_choose_topic_prefers_compressed_image(self) -> None:
        topics = [
            BagTopicInfo(name="/camera/raw", type="sensor_msgs/msg/Image", message_count=1),
            BagTopicInfo(name="/camera/compressed", type="sensor_msgs/msg/CompressedImage", message_count=2),
            BagTopicInfo(name="/odom", type="nav_msgs/msg/Odometry", message_count=3),
        ]
        chosen = _choose_topic(topics, kind="image", requested="auto")
        self.assertEqual(chosen.name, "/camera/compressed")

    def test_write_outputs_creates_route_context_capture_ready_files(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            trajectory = [
                {
                    "timestamp": 1,
                    "lat": 30.0,
                    "lng": 120.0,
                    "heading_deg": 90.0,
                    "image_path": str(output_dir / "images" / "000001.jpg"),
                }
            ]
            frame_records = [
                {
                    "frame_index": 1,
                    "timestamp": 1,
                    "image_path": str(output_dir / "images" / "000001.jpg"),
                    "image_topic": "/camera/camera/color/image_raw",
                    "image_topic_type": "sensor_msgs/msg/Image",
                    "odom_topic": "/globalEstimator/global_odometry",
                    "topic_type": "sensor_msgs/msg/Image",
                    "encoding": "rgb8",
                    "image_width": 640,
                    "image_height": 360,
                }
            ]

            trajectory_path, meta_path = _write_outputs(
                output_dir=output_dir,
                trajectory=trajectory,
                frame_records=frame_records,
                image_topic="/camera/camera/color/image_raw",
                odom_topic="/globalEstimator/global_odometry",
                bag_path=Path("/tmp/sidewalk_he4"),
                sample_fps=1.0,
                start_lat=30.0,
                start_lng=120.0,
                odom_to_enu_yaw_deg=0.0,
                heading_offset_deg=0.0,
            )

            self.assertTrue(trajectory_path.exists())
            self.assertTrue(meta_path.exists())
            self.assertTrue((output_dir / "frames_manifest.json").exists())
            self.assertTrue((output_dir / "route_context_capture_input.json").exists())
            self.assertTrue((output_dir / "images").exists())

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["frames"], 1)
            self.assertIn("route_context_capture_ready", meta)
            capture_input = json.loads((output_dir / "route_context_capture_input.json").read_text(encoding="utf-8"))
            self.assertEqual(capture_input["trajectory"], str(trajectory_path))
            self.assertEqual(capture_input["images_dir"], str(output_dir / "images"))


if __name__ == "__main__":
    unittest.main()
