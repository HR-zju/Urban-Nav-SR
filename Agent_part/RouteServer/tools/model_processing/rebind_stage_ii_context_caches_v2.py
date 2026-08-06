#!/usr/bin/env python3
"""Rebind Stage II context caches after a transport-only Stage I migration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.stage_ii_cache_migration_v2 import (
    rebind_stage_ii_context_cache_manifest_v2, )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse immutable cache-rebind artifact paths.

    Args:
        argv: Optional explicit command-line arguments.

    Returns:
        Parsed command-line namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-i-v1-jsonl", required=True, type=Path)
    parser.add_argument("--stage-i-v2-jsonl", required=True, type=Path)
    parser.add_argument("--migration-report-json", required=True, type=Path)
    parser.add_argument("--input-cache-jsonl", required=True, type=Path)
    parser.add_argument("--output-cache-jsonl", required=True, type=Path)
    parser.add_argument("--cache-input-case-jsonl", type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the fail-closed transport-only cache rebind.

    Args:
        argv: Optional explicit command-line arguments.

    Returns:
        None.
    """

    args = parse_args(argv)
    report = rebind_stage_ii_context_cache_manifest_v2(
        stage_i_v1_jsonl=args.stage_i_v1_jsonl,
        stage_i_v2_jsonl=args.stage_i_v2_jsonl,
        migration_report_json=args.migration_report_json,
        input_cache_jsonl=args.input_cache_jsonl,
        output_cache_jsonl=args.output_cache_jsonl,
        cache_input_case_jsonl=args.cache_input_case_jsonl,
        report_json=args.report_json,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
