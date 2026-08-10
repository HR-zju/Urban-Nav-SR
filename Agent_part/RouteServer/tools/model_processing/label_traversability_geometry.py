#!/usr/bin/env python3
"""Annotate local FPV traversability quadrilaterals into a V1 sidecar ledger."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Sequence

from mcp_server.teacher.navigation_quality_batch import (
    NavigationQualityBatchConfigV1,
    label_traversability_geometry_sources_v1,
)


def _read_api_keys(path: Path | None) -> tuple[str, ...]:
    """Read non-comment provider keys without serializing them to artifacts."""

    if path is None:
        return ()
    return tuple(
        line.strip()
        for line in path.expanduser().read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the bounded direct-LVLM geometry annotation contract."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--geometry-jsonl", required=True, type=Path)
    parser.add_argument("--decisions-jsonl", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--image-detail", choices=("auto", "low", "high"), default="high")
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--retry-nonaccepted-on-resume", action="store_true")
    parser.add_argument("--api-keys-file", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run append-only local-traversability geometry annotation."""

    args = parse_args(argv)
    report = asyncio.run(
        label_traversability_geometry_sources_v1(
            source_jsonl=args.source_jsonl,
            geometry_jsonl=args.geometry_jsonl,
            decisions_jsonl=args.decisions_jsonl,
            report_json=args.report_json,
            limit=args.limit,
            api_keys=_read_api_keys(args.api_keys_file),
            config=NavigationQualityBatchConfigV1(
                max_retries=max(0, args.max_retries),
                image_detail=args.image_detail,
                concurrency=max(1, args.concurrency),
                retry_nonaccepted_on_resume=args.retry_nonaccepted_on_resume,
                progress_interval=max(0, args.progress_interval),
            ),
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
