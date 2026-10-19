#!/usr/bin/env python3
"""Audit complete V4 annotation coverage and designed stop quarantines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.canonical.reasoning_v4_audit import audit_reasoning_v4_annotation_completeness


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse annotation-completeness audit arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--annotation-jsonl", required=True, type=Path)
    parser.add_argument("--expected-rows", required=True, type=int)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the completeness audit and persist its report."""

    args = parse_args(argv)
    report = audit_reasoning_v4_annotation_completeness(
        args.source_jsonl,
        args.annotation_jsonl,
        expected_rows=args.expected_rows,
    )
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
