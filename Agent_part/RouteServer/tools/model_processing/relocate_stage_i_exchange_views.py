#!/usr/bin/env python3
"""Materialize canonical Stage I exchange views as a portable artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.stage_i_portable import relocate_stage_i_training_views_v1


def _prefix_map(value: str) -> tuple[str, str]:
    """Parse one ``OLD=NEW`` anchored source-prefix mapping."""

    old, separator, new = value.partition("=")
    if not separator or not old or not new:
        raise argparse.ArgumentTypeError("prefix map must be OLD=NEW")
    return old, new


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse portable-relocation command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output-name")
    parser.add_argument("--root-image-dir", type=Path)
    parser.add_argument(
        "--source-prefix-map",
        action="append",
        default=[],
        type=_prefix_map,
        metavar="OLD=NEW",
        help="Explicit anchored repair for stale absolute image paths.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run relocation and print the immutable audit report."""

    args = parse_args(argv)
    report = relocate_stage_i_training_views_v1(
        args.input_jsonl,
        args.output_root,
        output_name=args.output_name,
        root_image_dir=args.root_image_dir,
        source_prefix_maps=args.source_prefix_map,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
