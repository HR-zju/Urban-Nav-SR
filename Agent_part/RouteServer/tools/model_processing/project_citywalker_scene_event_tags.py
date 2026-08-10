#!/usr/bin/env python3
"""Project CityWalker-compatible multi-label scene-event sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.quality.scene_events import write_citywalker_scene_event_tags_v1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse immutable source/sidecar projection inputs.

    Args:
        argv: Optional command-line values used by focused tests.

    Returns:
        Parsed projection arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--filter-jsonl", required=True, type=Path)
    parser.add_argument("--geometry-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Write deterministic event sidecars and print their receipt.

    Args:
        argv: Optional command-line values used by focused tests.

    Returns:
        Process success status.
    """

    args = parse_args(argv)
    report = write_citywalker_scene_event_tags_v1(
        source_jsonl=args.source_jsonl,
        filter_jsonl=args.filter_jsonl,
        geometry_jsonl=args.geometry_jsonl,
        output_jsonl=args.output_jsonl,
        report_json=args.report_json,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
