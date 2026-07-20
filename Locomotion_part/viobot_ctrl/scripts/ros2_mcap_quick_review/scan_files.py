from __future__ import annotations
import argparse
from pathlib import Path
from review_tool.config import load_config
from review_tool.dataset import scan_to_dataset

def main():
    parser = argparse.ArgumentParser(description="Scan ROS2 bag directories and standalone MCAP files.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    df = scan_to_dataset(cfg)
    print(f"Found {len(df)} bags")
    print(f"Dataset index: {Path(cfg['paths']['work_dir']) / 'dataset.csv'}")

if __name__ == "__main__":
    main()
