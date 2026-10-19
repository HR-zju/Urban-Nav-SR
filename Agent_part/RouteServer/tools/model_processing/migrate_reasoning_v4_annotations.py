#!/usr/bin/env python3
"""Migrate V4.6 vocabulary aliases into a new canonical JSONL artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.canonical.reasoning_v4_migration import migrate_reasoning_annotation_manifest_v4


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit immutable inputs and new output destinations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--annotation-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run deterministic migration without modifying either input artifact."""

    args = parse_args(argv)
    report = migrate_reasoning_annotation_manifest_v4(
        args.source_jsonl,
        args.annotation_jsonl,
        args.output_jsonl,
        report_json=args.report_json,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
