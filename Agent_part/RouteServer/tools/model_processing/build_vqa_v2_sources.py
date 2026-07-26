#!/usr/bin/env python3
"""Build frozen Navigation VQA V2 sources from Reasoning V4 sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.vqa_v2_source import (
    build_vqa_source_manifest_v2,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse source-builder arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Build VQA sources and print the conversion report."""

    args = parse_args(argv)
    report = build_vqa_source_manifest_v2(
        args.input_jsonl,
        args.output_jsonl,
        limit=args.limit,
        skip_errors=args.skip_errors,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
