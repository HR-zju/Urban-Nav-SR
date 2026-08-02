#!/usr/bin/env python3
"""Freeze canonical Stage I views against audited split lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcp_server.data.openloop_lineage import freeze_openloop_stage_i_views_v1


def main() -> None:
    """Parse arguments and write one immutable Stage I split."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage-jsonl", required=True, type=Path)
    parser.add_argument(
        "--view-jsonl",
        required=True,
        action="append",
        type=Path,
        help="Repeat for every canonical Stage I view artifact.",
    )
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument(
        "--task-type",
        action="append",
        choices=("full_cot", "vqa", "decision"),
    )
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    args = parser.parse_args()
    report = freeze_openloop_stage_i_views_v1(
        args.lineage_jsonl,
        args.view_jsonl,
        args.output_jsonl,
        args.report_json,
        split=args.split,
        task_types=set(args.task_type) if args.task_type else None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
