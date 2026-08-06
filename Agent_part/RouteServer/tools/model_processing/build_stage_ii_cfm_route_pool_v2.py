#!/usr/bin/env python3
"""Build the canonical Stage II CFM V2 reasoning-route pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.stage_ii_route_pool_v2 import (
    build_stage_ii_cfm_route_pool_v2,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the Stage II V2 route-pool command line.

    Args:
        argv: Optional explicit argument sequence.

    Returns:
        Parsed command-line namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-i-full-jsonl", required=True, type=Path)
    parser.add_argument("--trajectory-sidecar-jsonl", required=True, type=Path)
    parser.add_argument("--reasoning-cache-jsonl", required=True, type=Path)
    parser.add_argument("--decision-cache-jsonl", required=True, type=Path)
    parser.add_argument("--output-route-jsonl", required=True, type=Path)
    parser.add_argument("--output-qc-jsonl", required=True, type=Path)
    parser.add_argument("--output-drop-jsonl", required=True, type=Path)
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Build the route pool and persist its deterministic report.

    Args:
        argv: Optional explicit argument sequence.

    Returns:
        None.
    """

    args = parse_args(argv)
    report = build_stage_ii_cfm_route_pool_v2(
        stage_i_full_jsonl=args.stage_i_full_jsonl,
        trajectory_sidecar_jsonl=args.trajectory_sidecar_jsonl,
        reasoning_cache_jsonl=args.reasoning_cache_jsonl,
        decision_cache_jsonl=args.decision_cache_jsonl,
        output_route_jsonl=args.output_route_jsonl,
        output_qc_jsonl=args.output_qc_jsonl,
        output_drop_jsonl=args.output_drop_jsonl,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
