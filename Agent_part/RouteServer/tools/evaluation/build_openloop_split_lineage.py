#!/usr/bin/env python3
"""Build canonical train/validation/test lineage from explicit mappings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcp_server.data.openloop.lineage import build_openloop_split_lineage_v1


def main() -> None:
    """Parse arguments and write immutable split lineage artifacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--quarantine-jsonl", type=Path)
    args = parser.parse_args()
    report = build_openloop_split_lineage_v1(
        args.spec,
        args.output_jsonl,
        args.report_json,
        quarantine_jsonl=args.quarantine_jsonl,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
