#!/usr/bin/env python3
"""Merge resumable language-goal annotation lanes in canonical candidate order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.benchmark.language_waypoint_annotation_shards import (
    merge_language_goal_annotation_shards_v1,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse lane-ledger merge arguments.

    Args:
        argv: Optional test-time argument sequence.

    Returns:
        Parsed command-line namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-jsonl", required=True, type=Path)
    parser.add_argument("--annotation-jsonl", action="append", required=True, type=Path)
    parser.add_argument("--decisions-jsonl", action="append", required=True, type=Path)
    parser.add_argument("--output-annotations-jsonl", required=True, type=Path)
    parser.add_argument("--output-decisions-jsonl", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Write one deterministic merged annotation/audit pair.

    Args:
        argv: Optional test-time argument sequence.

    Returns:
        Process exit status.
    """

    args = parse_args(argv)
    report = merge_language_goal_annotation_shards_v1(
        candidates_jsonl=args.candidates_jsonl,
        annotation_jsonls=args.annotation_jsonl,
        decisions_jsonls=args.decisions_jsonl,
        output_annotations_jsonl=args.output_annotations_jsonl,
        output_decisions_jsonl=args.output_decisions_jsonl,
        report_json=args.report_json,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
