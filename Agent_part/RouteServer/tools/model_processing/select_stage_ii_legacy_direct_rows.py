#!/usr/bin/env python3
"""Recover legacy prompts for the exact current Stage I train/validation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.stage_ii.legacy_direct import build_stage_ii_legacy_direct_selection_v1


def _mapping(value: str) -> tuple[str, str]:
    """Parse one required ``legacy_name=canonical_dataset`` argument."""

    name, separator, target = value.partition("=")
    if not separator or not name.strip() or not target.strip():
        raise argparse.ArgumentTypeError("expected name=value")
    return name.strip(), target.strip()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse exact Stage I allowlist selection arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--stage-i-train-manifest", required=True, type=Path)
    parser.add_argument(
        "--stage-i-validation-manifest",
        required=True,
        type=Path,
    )
    parser.add_argument("--registry-json", required=True, type=Path)
    parser.add_argument(
        "--source-dataset-map",
        action="append",
        required=True,
        type=_mapping,
        help="Repeat legacy_source=canonical_dataset.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Write selected rows and fail unless coverage is exact."""

    args = parse_args(argv)
    source_dataset_map = dict(args.source_dataset_map)
    if len(source_dataset_map) != len(args.source_dataset_map):
        raise ValueError("duplicate --source-dataset-map source")
    report = build_stage_ii_legacy_direct_selection_v1(
        source_jsonl=args.source_jsonl,
        stage_i_split_manifests={
            "train": args.stage_i_train_manifest,
            "validation": args.stage_i_validation_manifest,
        },
        source_dataset_map=source_dataset_map,
        registry_json=args.registry_json,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
