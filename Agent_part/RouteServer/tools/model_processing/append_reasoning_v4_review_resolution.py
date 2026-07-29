#!/usr/bin/env python3
"""Append one validated Reasoning V4.6 human-review resolution."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

from mcp_server.contracts.reasoning_dataset_v4 import (
    ActionReviewResolutionV4,
)
from mcp_server.data.reasoning_v4_state import (
    append_action_review_resolution_v4,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse resolution arguments.

    Args:
        argv: Optional explicit CLI arguments.

    Returns:
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--audit-id", required=True)
    parser.add_argument(
        "--review-status",
        required=True,
        choices=("resolved", "rejected"),
    )
    parser.add_argument("--resolved-action-label")
    parser.add_argument("--resolution-code", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument(
        "--resolved-at",
        required=True,
        help="Timezone-aware ISO-8601 timestamp.",
    )
    parser.add_argument("--note")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Validate and append one immutable resolution.

    Args:
        argv: Optional explicit CLI arguments.
    """

    args = parse_args(argv)
    resolution = ActionReviewResolutionV4(
        audit_id=args.audit_id,
        review_status=args.review_status,
        resolved_action_label=args.resolved_action_label,
        resolution_code=args.resolution_code,
        reviewer=args.reviewer,
        resolved_at=datetime.fromisoformat(args.resolved_at),
        note=args.note,
    )
    append_action_review_resolution_v4(
        args.output_jsonl,
        resolution,
    )
    print(
        json.dumps(
            resolution.model_dump(
                mode="json",
                exclude_none=True,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
