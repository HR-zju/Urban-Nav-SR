#!/usr/bin/env python3
"""Project canonical Stage I views to hidden benchmark references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcp_server.data.openloop.benchmark import build_openloop_reference_jsonl_v1


def main() -> None:
    """Parse arguments and build hidden reference JSONL."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("stage_i",))
    args = parser.parse_args()
    print(
        json.dumps(
            build_openloop_reference_jsonl_v1(
                args.input,
                args.output,
                stage=args.stage,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
