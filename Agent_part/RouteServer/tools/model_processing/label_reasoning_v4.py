#!/usr/bin/env python3
"""Label frozen Reasoning V4 sources with independent LVLM section calls."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Sequence

from mcp_server.teacher.reasoning_v4_batch import (
    ReasoningV4BatchConfig,
    label_reasoning_source_manifest_v4,
)
from mcp_server.teacher.reasoning_v4_image_policy import (
    REASONING_V4_ALL_VISIBLE_IMAGES_POLICY_ID,
    REASONING_V4_SECTION_IMAGE_POLICIES,
)


def _read_api_keys(path: Path | None) -> tuple[str, ...]:
    """Read non-comment provider keys without serializing them."""

    if path is None:
        return ()
    return tuple(
        line.strip()
        for line in path.expanduser().read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse V4 annotation-runner arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--annotation-jsonl", required=True, type=Path)
    parser.add_argument("--rejected-jsonl", type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--image-detail",
        choices=("auto", "low", "high"),
        default="low",
    )
    parser.add_argument(
        "--navigation-context-image-detail",
        choices=("auto", "low", "high"),
        default="high",
        help=(
            "Provider detail used only for NavigationContextGrounding; "
            "defaults to high so thin BEV route overlays are preserved."
        ),
    )
    parser.add_argument(
        "--section-image-policy-id",
        choices=tuple(REASONING_V4_SECTION_IMAGE_POLICIES),
        default=REASONING_V4_ALL_VISIBLE_IMAGES_POLICY_ID,
        help=(
            "Versioned section-specific image routing policy. The routed "
            "visual policy exposes temporal FPV only where needed and keeps "
            "Decision visual with current FPV plus optional current BEV."
        ),
    )
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--retry-failed-on-resume", action="store_true")
    parser.add_argument("--api-keys-file", type=Path)
    parser.add_argument("--input-cost-per-million-tokens", type=float)
    parser.add_argument("--output-cost-per-million-tokens", type=float)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run resumable V4 section annotation."""

    args = parse_args(argv)
    report = asyncio.run(
        label_reasoning_source_manifest_v4(
            args.source_jsonl,
            args.annotation_jsonl,
            rejected_jsonl=args.rejected_jsonl,
            report_json=args.report_json,
            config=ReasoningV4BatchConfig(
                max_retries=max(0, args.max_retries),
                image_detail=args.image_detail,
                navigation_context_image_detail=(
                    args.navigation_context_image_detail
                ),
                section_image_policy_id=(
                    args.section_image_policy_id
                ),
                concurrency=max(1, args.concurrency),
                retry_failed_on_resume=(
                    args.retry_failed_on_resume
                ),
                progress_interval=max(0, args.progress_interval),
                input_cost_per_million_tokens=(
                    args.input_cost_per_million_tokens
                ),
                output_cost_per_million_tokens=(
                    args.output_cost_per_million_tokens
                ),
            ),
            limit=args.limit,
            api_keys=_read_api_keys(args.api_keys_file),
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
