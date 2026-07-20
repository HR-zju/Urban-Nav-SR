from __future__ import annotations
import argparse
from pathlib import Path
from review_tool.annotations import build_manifest, load_annotations
from review_tool.config import load_config

def main():
    parser = argparse.ArgumentParser(description="Export Good/Bad manifests.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    work_dir = Path(cfg["paths"]["work_dir"])
    annotations = load_annotations(work_dir / "annotations.csv")
    good_path, bad_path = work_dir / "good_files.txt", work_dir / "bad_files.txt"
    good_path.write_text(build_manifest(annotations, "Good"), encoding="utf-8")
    bad_path.write_text(build_manifest(annotations, "Bad"), encoding="utf-8")
    print(f"Good: {(annotations['label'] == 'Good').sum()} -> {good_path}")
    print(f"Bad:  {(annotations['label'] == 'Bad').sum()} -> {bad_path}")

if __name__ == "__main__":
    main()
