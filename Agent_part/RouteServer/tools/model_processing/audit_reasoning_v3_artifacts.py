#!/usr/bin/env python3
"""Audit Reasoning V3 artifacts against the documented acceptance gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.reasoning_v3_audit import audit_reasoning_v3_artifacts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse acceptance-audit CLI arguments.

    Args:
        argv: Optional argument vector for tests or programmatic use.

    Returns:
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--annotation-jsonl", required=True, type=Path)
    parser.add_argument("--view-jsonl", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--max-insufficient-fraction", type=float, default=0.10)
    parser.add_argument("--max-unsupported-rationale-fraction", type=float, default=0.20)
    parser.add_argument("--require-dataset", action="append", default=[])
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the offline audit and write its JSON report.

    Args:
        argv: Optional argument vector for tests or programmatic use.
    """

    args = parse_args(argv)
    report = audit_reasoning_v3_artifacts(
        args.source_jsonl,
        args.annotation_jsonl,
        args.view_jsonl,
        max_insufficient_fraction=args.max_insufficient_fraction,
        max_unsupported_rationale_fraction=args.max_unsupported_rationale_fraction,
        required_dataset_families=args.require_dataset,
    )
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
