#!/usr/bin/env python3
"""Export deterministic V4.6 repair requests from human adjudications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from mcp_server.data.jsonl_io import write_jsonl_atomic
from mcp_server.data.reasoning_v4_review import (
    build_reasoning_repair_request_v4,
)
from mcp_server.data.reasoning_v4_state import (
    load_latest_action_adjudications_v4,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse repair-request export arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjudication-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args(argv)


def export_reasoning_v4_repair_requests(
    *,
    adjudication_jsonl: str | Path,
    output_jsonl: str | Path,
) -> dict[str, Any]:
    """Project latest adjudications into immutable upstream repair requests.

    Args:
        adjudication_jsonl: Append-only canonical adjudication sidecar.
        output_jsonl: Atomic repair-request manifest.

    Returns:
        Export counts and output identity.
    """

    adjudications = load_latest_action_adjudications_v4(
        adjudication_jsonl
    )
    requests = []
    skipped = 0
    for audit_id in sorted(adjudications):
        request = build_reasoning_repair_request_v4(
            adjudications[audit_id]
        )
        if request is None:
            skipped += 1
            continue
        requests.append(
            request.model_dump(mode="json", exclude_none=True)
        )
    write_jsonl_atomic(output_jsonl, requests)
    return {
        "schema_version": (
            "urbannav_reasoning_repair_request_export_report_v1"
        ),
        "adjudication_jsonl": str(adjudication_jsonl),
        "output_jsonl": str(output_jsonl),
        "latest_adjudications": len(adjudications),
        "repair_requests": len(requests),
        "kept_or_quarantined": skipped,
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Export repair requests and write an optional report."""

    args = parse_args(argv)
    report = export_reasoning_v4_repair_requests(
        adjudication_jsonl=args.adjudication_jsonl,
        output_jsonl=args.output_jsonl,
    )
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
