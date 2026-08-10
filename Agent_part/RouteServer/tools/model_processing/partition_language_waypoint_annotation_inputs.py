#!/usr/bin/env python3
"""Partition language-goal candidates into deterministic provider lanes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.benchmark.language_waypoint_annotation_shards import (
    partition_language_goal_annotation_inputs_v1,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse immutable source/candidate sharding arguments.

    Args:
        argv: Optional test-time argument sequence.

    Returns:
        Parsed command-line namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--candidates-jsonl", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--lane-count", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Write parent-disjoint source/candidate shard pairs.

    Args:
        argv: Optional test-time argument sequence.

    Returns:
        Process exit status.
    """

    args = parse_args(argv)
    report = partition_language_goal_annotation_inputs_v1(
        source_jsonl=args.source_jsonl,
        candidates_jsonl=args.candidates_jsonl,
        output_root=args.output_root,
        report_json=args.report_json,
        lane_count=args.lane_count,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
