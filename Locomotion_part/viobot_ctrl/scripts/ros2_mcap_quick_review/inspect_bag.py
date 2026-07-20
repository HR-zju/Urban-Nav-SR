from __future__ import annotations

import argparse
from pathlib import Path

from review_tool.config import load_config
from review_tool.rosbag_reader import RosbagReader


def main() -> None:
    parser = argparse.ArgumentParser(description="List topics and types without ROS.")
    parser.add_argument("bag", help="ROS2 MCAP bag directory or standalone .mcap")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    reader = RosbagReader(Path(args.bag), reader_config=cfg.get("reader", {}))
    try:
        reader.open()
        width = max((len(x) for x in reader.topic_types), default=20)
        for topic, msgtype in sorted(reader.topic_types.items()):
            print(f"{topic:<{width}}  {msgtype}")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
