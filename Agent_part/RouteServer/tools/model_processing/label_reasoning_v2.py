#!/usr/bin/env python3
"""Label frozen ReasoningSourceV2 rows with a validated multimodal teacher."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Sequence

from mcp_server.teacher.reasoning_v2_teacher import ReasoningTeacherConfig, label_reasoning_source_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse offline teacher arguments.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--annotation-jsonl", required=True, type=Path)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--image-detail", choices=("auto", "low", "high"), default="auto")
    parser.add_argument("--retry-failed-on-resume", action="store_true")
    parser.add_argument(
        "--api-keys-file",
        type=Path,
        default=None,
        help="Optional one-key-per-line worker rotation. Environment key is preferred for a single worker.",
    )
    return parser.parse_args(argv)


def read_api_keys(path: Path | None) -> tuple[str, ...]:
    """Read non-empty API keys without logging them.

    Args:
        path: Optional key file.

    Returns:
        In-memory key tuple.
    """

    if path is None:
        return ()
    return tuple(
        line.strip()
        for line in path.expanduser().read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run recoverable V2 teacher annotation.

    Args:
        argv: Optional command-line arguments.

    Returns:
        ``None``.
    """

    args = parse_args(argv)
    config = ReasoningTeacherConfig(
        max_retries=max(0, args.max_retries),
        image_detail=args.image_detail,
        concurrency=max(1, args.concurrency),
        retry_failed_on_resume=args.retry_failed_on_resume,
    )
    report = asyncio.run(
        label_reasoning_source_manifest(
            args.source_jsonl,
            args.annotation_jsonl,
            report_json=args.report_json,
            config=config,
            limit=args.limit,
            api_keys=read_api_keys(args.api_keys_file),
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
