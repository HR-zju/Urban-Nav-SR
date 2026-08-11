#!/usr/bin/env python3
"""Build the Student-only DF-OPD Decision-to-Trajectory baseline manifest."""

from __future__ import annotations

import argparse
import json

from mcp_server.data.df_opd.export import build_df_opd_dtot_manifest_v1


def main() -> int:
    """Parse local paths, export Student-only rows, and print the receipt."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-jsonl", required=True)
    parser.add_argument("--messages-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    args = parser.parse_args()
    report = build_df_opd_dtot_manifest_v1(
        canonical_jsonl=args.canonical_jsonl,
        messages_jsonl=args.messages_jsonl,
        output_jsonl=args.output_jsonl,
        report_json=args.report_json,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
