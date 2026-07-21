#!/usr/bin/env python3
"""Label frozen ReasoningSourceV3 rows through an OpenAI-compatible LVLM."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Sequence

from mcp_server.teacher.reasoning_v3_batch import ReasoningV3BatchConfig, label_reasoning_source_manifest_v3


def _read_api_keys(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return ()
    return tuple(
        line.strip()
        for line in path.expanduser().read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse observation-labeling CLI arguments.

    Args:
        argv: Optional argument vector for tests or programmatic use.

    Returns:
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--annotation-jsonl", required=True, type=Path)
    parser.add_argument("--rejected-jsonl", type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--image-detail", choices=("auto", "low", "high"), default="low")
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--retry-failed-on-resume", action="store_true")
    parser.add_argument("--api-keys-file", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run resumable observation labeling and deterministic composition.

    Args:
        argv: Optional argument vector for tests or programmatic use.
    """

    args = parse_args(argv)
    report = asyncio.run(
        label_reasoning_source_manifest_v3(
            args.source_jsonl,
            args.annotation_jsonl,
            rejected_jsonl=args.rejected_jsonl,
            report_json=args.report_json,
            config=ReasoningV3BatchConfig(
                max_retries=max(0, args.max_retries),
                image_detail=args.image_detail,
                concurrency=max(1, args.concurrency),
                retry_failed_on_resume=args.retry_failed_on_resume,
                progress_interval=max(0, args.progress_interval),
            ),
            limit=args.limit,
            api_keys=_read_api_keys(args.api_keys_file),
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
