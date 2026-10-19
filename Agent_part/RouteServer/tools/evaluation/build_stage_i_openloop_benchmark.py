#!/usr/bin/env python3
"""Build a validated Stage I open-loop package manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from mcp_server.eval.openloop.openloop_package import build_openloop_package_manifest_v1


def main() -> None:
    """Parse inputs and build one Stage I package manifest."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build_openloop_package_manifest_v1(
        benchmark_id=args.benchmark_id,
        split=args.split,
        stage="stage_i",
        input_jsonl=args.inputs,
        reference_jsonl=args.references,
        config_json=args.config,
        output_json=args.output,
    )


if __name__ == "__main__":
    main()
