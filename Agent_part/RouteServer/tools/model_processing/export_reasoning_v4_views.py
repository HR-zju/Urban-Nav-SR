#!/usr/bin/env python3
"""Export V4 annotations into language-supervised training views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.reasoning_v4_views import (
    export_reasoning_training_views_v4,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse V4 view-exporter arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--annotation-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument(
        "--release-track",
        choices=("review_preview", "formal_training"),
        default="review_preview",
    )
    parser.add_argument("--resolution-jsonl", type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Export V4 views and write an optional report."""

    args = parse_args(argv)
    report = export_reasoning_training_views_v4(
        args.source_jsonl,
        args.annotation_jsonl,
        args.output_jsonl,
        release_track=args.release_track,
        resolution_jsonl=args.resolution_jsonl,
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
