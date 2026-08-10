#!/usr/bin/env python3
"""Build a source-only retry manifest from failed geometry decisions."""

from __future__ import annotations

import argparse
import json

from mcp_server.data.quality.retry import build_geometry_retry_source_manifest_v1


def main() -> int:
    """Parse CLI arguments and write one immutable retry-source receipt."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--geometry-decisions-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--retry-status", action="append", default=["error", "rejected"])
    args = parser.parse_args()
    report = build_geometry_retry_source_manifest_v1(
        source_jsonl=args.source_jsonl,
        geometry_decisions_jsonl=args.geometry_decisions_jsonl,
        output_jsonl=args.output_jsonl,
        report_json=args.report_json,
        retry_statuses=args.retry_status,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
