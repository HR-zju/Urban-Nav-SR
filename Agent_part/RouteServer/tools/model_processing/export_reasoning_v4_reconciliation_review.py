#!/usr/bin/env python3
"""Export Reasoning V4 Action Reconciliation review artifacts."""

from __future__ import annotations

import argparse
import json

from mcp_server.data.canonical.reasoning_v4_reconciliation_review import export_action_reconciliation_review_v4


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--annotation-jsonl", required=True)
    parser.add_argument("--audit-jsonl", required=True)
    parser.add_argument("--queue-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--diagnosis-jsonl")
    parser.add_argument("--adjudication-jsonl")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit status when the report contains errors.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the deterministic review export."""

    args = parse_args()
    report = export_action_reconciliation_review_v4(
        args.source_jsonl,
        args.annotation_jsonl,
        audit_jsonl=args.audit_jsonl,
        queue_jsonl=args.queue_jsonl,
        report_json=args.report_json,
        diagnosis_jsonl=args.diagnosis_jsonl,
        adjudication_jsonl=args.adjudication_jsonl,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
