#!/usr/bin/env python3
"""Audit a natural-random Reasoning V4.6 pilot before full annotation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.canonical.reasoning_v4_audit import audit_reasoning_v4_pilot


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse pilot-audit arguments.

    Args:
        argv: Optional explicit CLI arguments.

    Returns:
        Parsed command-line namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--annotation-jsonl", required=True, type=Path)
    parser.add_argument("--formal-view-jsonl", required=True, type=Path)
    parser.add_argument("--expected-rows", required=True, type=int)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pilot audit and persist its machine-readable report.

    Args:
        argv: Optional explicit CLI arguments.

    Returns:
        Zero when the audit passes or strict mode is disabled, otherwise two.
    """

    args = parse_args(argv)
    report = audit_reasoning_v4_pilot(
        args.source_jsonl,
        args.annotation_jsonl,
        args.formal_view_jsonl,
        expected_rows=args.expected_rows,
    )
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
