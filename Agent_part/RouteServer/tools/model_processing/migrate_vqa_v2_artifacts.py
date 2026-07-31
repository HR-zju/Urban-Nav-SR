#!/usr/bin/env python3
"""Migrate legacy VQA V2 source identity without relabeling Q/A pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.vqa_v2_migration import migrate_vqa_artifacts_v2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit source, annotation, and destination artifacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--annotation-jsonl", required=True, type=Path)
    parser.add_argument("--output-source-jsonl", required=True, type=Path)
    parser.add_argument("--output-annotation-jsonl", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the append-new-artifact migration."""

    args = parse_args(argv)
    report = migrate_vqa_artifacts_v2(
        args.source_jsonl,
        args.annotation_jsonl,
        args.output_source_jsonl,
        args.output_annotation_jsonl,
        report_json=args.report_json,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
