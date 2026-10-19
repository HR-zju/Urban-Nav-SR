#!/usr/bin/env python3
"""Build full train/validation Stage II trajectory sidecars."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from mcp_server.data.stage_ii.sidecars import build_stage_ii_trajectory_sidecars_v1


def _key_value(value: str) -> tuple[str, Path]:
    """Parse one ``name=path`` CLI value."""

    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected name=/absolute/path.jsonl")
    return name, Path(raw_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse full sidecar build arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-i-jsonl", required=True, type=Path)
    parser.add_argument(
        "--split-manifest",
        action="append",
        required=True,
        type=_key_value,
        help="Repeat train=/path and validation=/path exactly once.",
    )
    parser.add_argument(
        "--dataset-manifest",
        action="append",
        required=True,
        type=_key_value,
        help="Repeat dataset=/path; a dataset may have multiple manifests.",
    )
    parser.add_argument("--registry-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Build sidecars and fail closed when coverage is incomplete."""

    args = parse_args(argv)
    split_manifests = dict(args.split_manifest)
    if len(split_manifests) != len(args.split_manifest):
        raise ValueError("duplicate --split-manifest name")
    dataset_manifests: dict[str, list[Path]] = defaultdict(list)
    for dataset, path in args.dataset_manifest:
        dataset_manifests[dataset].append(path)
    report = build_stage_ii_trajectory_sidecars_v1(
        stage_i_jsonl=args.stage_i_jsonl,
        split_manifests=split_manifests,
        dataset_manifests=dataset_manifests,
        registry_json=args.registry_json,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
