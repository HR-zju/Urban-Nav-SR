#!/usr/bin/env python3
"""Build or apply an authoritative SideWalk BEV route-contract registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from mcp_server.data.adapters.sidewalk_bev_contract_recovery import build_sidewalk_bev_contract_registry, build_sidewalk_navigation_repair_subset


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse registry build/apply arguments.

    Args:
        argv: Optional explicit argument vector.

    Returns:
        Parsed command namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-registry")
    build.add_argument("--failure-rows-jsonl", required=True, type=Path)
    build.add_argument("--producer-root", required=True, type=Path)
    build.add_argument("--output-jsonl", required=True, type=Path)
    build.add_argument("--report-json", required=True, type=Path)
    build.add_argument("--media-repair-root", type=Path)

    apply = subparsers.add_parser("build-repair-subset")
    apply.add_argument("--input-jsonl", required=True, type=Path)
    apply.add_argument("--registry-jsonl", required=True, type=Path)
    apply.add_argument("--output-jsonl", required=True, type=Path)
    apply.add_argument("--report-json", required=True, type=Path)
    apply.add_argument("--media-root", type=Path)
    apply.add_argument("--expected-rows", required=True, type=int)
    return parser.parse_args(argv)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    """Write one formatted machine-readable report atomically enough for CLI use."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Execute one recovery phase and enforce its exact row gate.

    Args:
        argv: Optional explicit argument vector.

    Returns:
        ``None``.
    """

    args = parse_args(argv)
    if args.command == "build-registry":
        report = build_sidewalk_bev_contract_registry(
            args.failure_rows_jsonl,
            args.producer_root,
            args.output_jsonl,
            media_repair_root=args.media_repair_root,
        )
    else:
        report = build_sidewalk_navigation_repair_subset(
            args.input_jsonl,
            args.registry_jsonl,
            args.output_jsonl,
            media_root=args.media_root,
        )
        if report["counts"].get("output_rows") != args.expected_rows:
            report["passed"] = False
            report["expected_rows"] = args.expected_rows
    _write_report(args.report_json, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
