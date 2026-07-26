#!/usr/bin/env python3
"""Audit multi-dataset Reasoning V4 sources before teacher annotation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.reasoning_v4_audit import (
    audit_reasoning_v4_source_coverage,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse source-coverage audit arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-jsonl",
        action="append",
        required=True,
        type=Path,
        help="Repeat for every V4 source manifest in the coverage set.",
    )
    parser.add_argument(
        "--require-dataset",
        action="append",
        default=[],
    )
    parser.add_argument("--min-source-rows", type=int, default=0)
    parser.add_argument("--check-media", action="store_true")
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the deterministic source-coverage audit."""

    args = parse_args(argv)
    report = audit_reasoning_v4_source_coverage(
        args.source_jsonl,
        required_dataset_families=tuple(args.require_dataset),
        min_source_rows=max(0, args.min_source_rows),
        check_media=args.check_media,
    )
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
