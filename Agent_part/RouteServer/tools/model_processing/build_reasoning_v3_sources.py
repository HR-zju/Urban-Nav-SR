#!/usr/bin/env python3
"""Build minimal Reasoning V3.4 source rows from existing navigation manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.reasoning_source import ReasoningSourceBuildConfig
from mcp_server.data.reasoning_v3_source import build_reasoning_source_manifest_v3


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse source-builder CLI arguments.

    Args:
        argv: Optional argument vector for tests or programmatic use.

    Returns:
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--dataset-source", required=True)
    parser.add_argument("--default-instruction", default="Continue safe pedestrian navigation.")
    parser.add_argument("--travel-mode", default="pedestrian_assist")
    parser.add_argument("--max-fpv-frames", type=int, default=5)
    parser.add_argument("--no-current-bev", action="store_true")
    parser.add_argument("--disable-system1", action="store_true")
    parser.add_argument("--allow-pseudo-goal-condition", action="store_true")
    parser.add_argument("--allow-ambiguous-single-image", action="store_true")
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--disabled-system1-datasets", default="frodobots,frodobots-2k,frodobots-7k")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Build a frozen V3.4 source manifest and optional report.

    Args:
        argv: Optional argument vector for tests or programmatic use.
    """

    args = parse_args(argv)
    disabled = frozenset(
        item.strip()
        for item in args.disabled_system1_datasets.split(",")
        if item.strip()
    )
    config = ReasoningSourceBuildConfig(
        dataset_source=args.dataset_source,
        default_instruction=args.default_instruction,
        travel_mode=args.travel_mode,
        max_fpv_frames=max(1, args.max_fpv_frames),
        include_current_bev=not args.no_current_bev,
        enable_system1=not args.disable_system1,
        disabled_system1_datasets=disabled,
        allow_pseudo_goal_condition=args.allow_pseudo_goal_condition,
        strict_image_roles=not args.allow_ambiguous_single_image,
        media_root=args.media_root,
    )
    report = build_reasoning_source_manifest_v3(
        args.input_jsonl,
        args.output_jsonl,
        config=config,
        limit=args.limit,
        skip_errors=args.skip_errors,
    )
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
