#!/usr/bin/env python3
"""Audit Reasoning V4 source, annotation, and view manifests offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.reasoning_v4_audit import (
    audit_reasoning_v4_artifacts,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse V4 artifact-audit arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--annotation-jsonl", required=True, type=Path)
    parser.add_argument("--resolution-jsonl", type=Path)
    parser.add_argument("--view-jsonl", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--require-dataset", action="append", default=[])
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the offline V4 audit and write its report."""

    args = parse_args(argv)
    report = audit_reasoning_v4_artifacts(
        args.source_jsonl,
        args.annotation_jsonl,
        args.view_jsonl,
        resolution_jsonl=args.resolution_jsonl,
        required_dataset_families=tuple(args.require_dataset),
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
