#!/usr/bin/env python3
"""Run dataset suitability filtering and project read-only training views."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.quality.views import (
    NavigationTrainabilityViewConfigV1,
    write_navigation_trainability_views_v1,
)
from mcp_server.teacher.navigation_quality_batch import (
    NavigationQualityBatchConfigV1,
    filter_navigation_trainability_sources_v1,
)


def _read_api_keys(path: Path | None) -> tuple[str, ...]:
    """Read non-comment provider keys without persisting credentials."""

    if path is None:
        return ()
    return tuple(
        line.strip()
        for line in path.expanduser().read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse bounded direct-LVLM suitability and view-projection arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--filter-jsonl", required=True, type=Path)
    parser.add_argument("--decisions-jsonl", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--views-root", type=Path)
    parser.add_argument("--views-report-json", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--trajectory-sidecar-jsonl",
        type=Path,
        help="Optional accepted Stage-II V2 sidecar ledger for teacher-only motion QC.",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-temporal-fpv-frames", type=int, default=5)
    parser.add_argument("--image-detail", choices=("auto", "low", "high"), default="high")
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--retry-pending-on-resume", action="store_true")
    parser.add_argument("--api-keys-file", type=Path)
    parser.add_argument("--no-eligibility-views", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run append-only suitability filtering then derive source-only views."""

    args = parse_args(argv)
    filter_report = asyncio.run(
        filter_navigation_trainability_sources_v1(
            source_jsonl=args.source_jsonl,
            filter_jsonl=args.filter_jsonl,
            decisions_jsonl=args.decisions_jsonl,
            report_json=args.report_json,
            limit=args.limit,
            trajectory_sidecar_jsonl=args.trajectory_sidecar_jsonl,
            api_keys=_read_api_keys(args.api_keys_file),
            config=NavigationQualityBatchConfigV1(
                max_retries=max(0, args.max_retries),
                image_detail=args.image_detail,
                concurrency=max(1, args.concurrency),
                max_temporal_fpv_frames=max(1, args.max_temporal_fpv_frames),
                retry_pending_on_resume=args.retry_pending_on_resume,
                progress_interval=max(0, args.progress_interval),
            ),
        )
    )
    report: dict[str, object] = {"filter": filter_report}
    if args.views_root is not None:
        report["views"] = write_navigation_trainability_views_v1(
            source_jsonl=args.source_jsonl,
            filter_jsonl=args.filter_jsonl,
            output_root=args.views_root,
            report_json=args.views_report_json,
            config=NavigationTrainabilityViewConfigV1(
                write_eligibility_views=not args.no_eligibility_views,
            ),
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
