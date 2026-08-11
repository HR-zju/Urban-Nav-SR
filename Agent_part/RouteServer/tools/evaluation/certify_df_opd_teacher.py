#!/usr/bin/env python3
"""Certify frozen DF-OPD Teacher scores with paired bootstrap evidence."""

from __future__ import annotations

import argparse
import json

from mcp_server.eval.df_opd_teacher import certify_df_opd_teacher_jsonl_v1


def main() -> int:
    """Parse local score artifacts, persist certification outputs, and print a receipt."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--high-conflict-jsonl")
    parser.add_argument("--bootstrap-seed", required=True, type=int)
    parser.add_argument("--bootstrap-replicates", default=1000, type=int)
    parser.add_argument("--high-conflict-limit", default=50, type=int)
    args = parser.parse_args()
    report = certify_df_opd_teacher_jsonl_v1(
        score_jsonl=args.score_jsonl,
        report_json=args.report_json,
        high_conflict_jsonl=args.high_conflict_jsonl,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
        high_conflict_limit=args.high_conflict_limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
