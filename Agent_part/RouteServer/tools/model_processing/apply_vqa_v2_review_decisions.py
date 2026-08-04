#!/usr/bin/env python3
"""Apply exported human VQA V2 review decisions to training artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.vqa_v2_review import (
    filter_vqa_artifacts_by_review_v2,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse VQA review-filter arguments.

    Args:
        argv: Optional explicit CLI arguments.

    Returns:
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-jsonl",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--decision-jsonl",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-annotation-jsonl",
        required=True,
        type=Path,
    )
    parser.add_argument("--rejected-annotation-jsonl", type=Path)
    parser.add_argument("--needs-fix-annotation-jsonl", type=Path)
    parser.add_argument("--training-view-jsonl", type=Path)
    parser.add_argument("--output-training-view-jsonl", type=Path)
    parser.add_argument(
        "--unreviewed-policy",
        choices=("error", "keep", "drop"),
        default="error",
        help=(
            "Explicitly keep, drop, or reject execution when rows have no "
            "human decision. Sampled review patches normally use 'keep'."
        ),
    )
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Filter VQA artifacts and print the exact operation report.

    Args:
        argv: Optional explicit CLI arguments.

    Returns:
        ``None``.
    """

    args = parse_args(argv)
    report = filter_vqa_artifacts_by_review_v2(
        annotation_jsonl=args.annotation_jsonl,
        decision_jsonl=args.decision_jsonl,
        output_annotation_jsonl=args.output_annotation_jsonl,
        rejected_annotation_jsonl=(
            args.rejected_annotation_jsonl
        ),
        needs_fix_annotation_jsonl=(
            args.needs_fix_annotation_jsonl
        ),
        training_view_jsonl=args.training_view_jsonl,
        output_training_view_jsonl=(
            args.output_training_view_jsonl
        ),
        unreviewed_policy=args.unreviewed_policy,
        report_json=args.report_json,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
