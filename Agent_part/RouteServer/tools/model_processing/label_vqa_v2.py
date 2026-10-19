#!/usr/bin/env python3
"""Run resumable active-task annotation for Navigation VQA V2."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Sequence

from mcp_server.data.vqa.eligibility import load_vqa_capabilities_v2
from mcp_server.teacher.vqa_v2_batch import (
    VQABatchConfigV2,
    label_vqa_source_manifest_v2,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse annotation runner arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--annotation-jsonl", required=True, type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--capability-json", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--image-detail", default="low")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run annotation without logging provider credentials."""

    args = parse_args(argv)
    capabilities = load_vqa_capabilities_v2(
        args.capability_json
    )
    keys = [
        value.strip()
        for value in (
            os.environ.get("ARK_API_KEYS")
            or os.environ.get("ARK_API_KEY")
            or os.environ.get("LLM_API_KEY")
            or ""
        ).split(",")
        if value.strip()
    ]
    report = asyncio.run(
        label_vqa_source_manifest_v2(
            args.source_jsonl,
            args.annotation_jsonl,
            capabilities=capabilities,
            report_json=args.report_json,
            config=VQABatchConfigV2(
                max_retries=args.max_retries,
                concurrency=args.concurrency,
                image_detail=args.image_detail,
            ),
            limit=args.limit,
            api_keys=keys,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
