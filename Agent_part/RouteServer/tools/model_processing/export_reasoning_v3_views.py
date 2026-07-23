#!/usr/bin/env python3
"""Export accepted Reasoning V3.4 annotations into provenance-gated views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.reasoning_v3_views import (
    DEFAULT_REASONING_V3_TRAINING_PROFILE,
    REASONING_V3_TRAINING_PROFILES,
    ReasoningViewBuildConfigV3,
    export_reasoning_training_views_v3,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse minimal V3.4 view-exporter CLI arguments.

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
    parser.add_argument(
        "--training-profile",
        choices=REASONING_V3_TRAINING_PROFILES,
        default=DEFAULT_REASONING_V3_TRAINING_PROFILE,
        help=(
            "V3.4 loss policy: audit_only keeps the trace with zero loss; "
            "strong_action_only enables only independently supervised action sidecars."
        ),
    )
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Export provenance-gated V3.4 views and an optional JSON report.

    Args:
        argv: Optional argument vector for tests or programmatic use.
    """

    args = parse_args(argv)
    report = export_reasoning_training_views_v3(
        args.source_jsonl,
        args.annotation_jsonl,
        args.output_jsonl,
        config=ReasoningViewBuildConfigV3(
            training_profile=args.training_profile,
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
