#!/usr/bin/env python3
"""Evaluate exported visible-language human review decisions before full labeling."""

from __future__ import annotations

import argparse
import json

from mcp_server.data.benchmark.language_waypoint_human_review import (
    LanguageGoalHumanReviewGateConfigV1,
    evaluate_language_goal_human_review_v1,
)


def main() -> int:
    """Parse human-review policy, write a gate report, and print it.

    Returns:
        Process exit code zero when the report was written.  A failed gate is
        represented by `passed=false`; it is not a malformed execution.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-jsonl", required=True)
    parser.add_argument("--adjudications-json", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--min-accept-rate", type=float, default=0.90)
    parser.add_argument("--max-uncertain-rate", type=float, default=0.05)
    parser.add_argument("--min-reviewed-per-instruction-type", type=int, default=1)
    args = parser.parse_args()
    report = evaluate_language_goal_human_review_v1(
        annotation_jsonl=args.annotation_jsonl,
        adjudications_json=args.adjudications_json,
        report_json=args.report_json,
        config=LanguageGoalHumanReviewGateConfigV1(
            require_all_eligible_reviewed=not args.allow_unreviewed,
            min_accept_rate=args.min_accept_rate,
            max_uncertain_rate=args.max_uncertain_rate,
            min_reviewed_per_instruction_type=args.min_reviewed_per_instruction_type,
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
