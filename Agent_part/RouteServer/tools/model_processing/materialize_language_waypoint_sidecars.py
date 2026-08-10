#!/usr/bin/env python3
"""Materialize dataset-local accepted Stage-II V2 sidecar ledgers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.benchmark.language_waypoint_annotation_shards import (
    materialize_language_goal_sidecars_by_dataset_v1,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse immutable V2-sidecar materialization arguments.

    Args:
        argv: Optional test-time argument sequence.

    Returns:
        Parsed command-line namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-sidecar-jsonl", action="append", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Write deduplicated dataset-local V2 sidecars.

    Args:
        argv: Optional test-time argument sequence.

    Returns:
        Process exit status.
    """

    args = parse_args(argv)
    report = materialize_language_goal_sidecars_by_dataset_v1(
        trajectory_sidecar_jsonls=args.trajectory_sidecar_jsonl,
        output_root=args.output_root,
        report_json=args.report_json,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
