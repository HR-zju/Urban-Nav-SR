#!/usr/bin/env python3
"""Extract validated V1 FullCoT rows from a mixed Stage I manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.stage_i_migration_v2 import (
    extract_stage_i_full_cot_manifest_v1,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse immutable mixed-source selection paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the fail-closed V1 FullCoT selector."""

    args = parse_args(argv)
    report = extract_stage_i_full_cot_manifest_v1(
        args.input_jsonl,
        args.output_jsonl,
        report_json=args.report_json,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
