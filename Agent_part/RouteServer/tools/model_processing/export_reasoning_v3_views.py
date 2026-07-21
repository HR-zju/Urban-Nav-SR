#!/usr/bin/env python3
"""Export accepted Reasoning V3.3 annotations into Full-CoT views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.reasoning_v3_views import ReasoningViewBuildConfigV3, export_reasoning_training_views_v3


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse Full-CoT exporter CLI arguments.

    Args:
        argv: Optional argument vector for tests or programmatic use.

    Returns:
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--annotation-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--require-complete-full-cot-supervision", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Export Full-CoT views and an optional JSON report.

    Args:
        argv: Optional argument vector for tests or programmatic use.
    """

    args = parse_args(argv)
    report = export_reasoning_training_views_v3(
        args.source_jsonl,
        args.annotation_jsonl,
        args.output_jsonl,
        config=ReasoningViewBuildConfigV3(
            require_complete_full_cot_supervision=args.require_complete_full_cot_supervision,
        ),
        limit=args.limit,
        skip_errors=args.skip_errors,
    )
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
