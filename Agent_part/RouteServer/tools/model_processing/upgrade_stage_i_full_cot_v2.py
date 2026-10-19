#!/usr/bin/env python3
"""Upgrade a validated Stage I V1 FullCoT manifest to V2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.stage_i.migration_v2 import upgrade_stage_i_full_cot_manifest_v2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse immutable Stage I FullCoT migration paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument(
        "--image-prefix-map",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Rewrite a missing absolute image prefix before V2 migration",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the fail-closed V1-to-V2 FullCoT migration."""

    args = parse_args(argv)
    prefix_maps = []
    for value in args.image_prefix_map:
        old, separator, new = value.partition("=")
        if not separator or not old or not new:
            raise ValueError("--image-prefix-map must be OLD=NEW")
        prefix_maps.append((old, new))
    report = upgrade_stage_i_full_cot_manifest_v2(
        args.input_jsonl,
        args.output_jsonl,
        report_json=args.report_json,
        image_prefix_maps=prefix_maps,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
