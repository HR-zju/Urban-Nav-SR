#!/usr/bin/env python3
"""Export accepted Navigation VQA V2 annotations as training views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.vqa_v2_eligibility import (
    load_vqa_capabilities_v2,
)
from mcp_server.data.vqa_v2_views import (
    export_vqa_training_views_v2,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse training-view export arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--annotation-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--capability-json", type=Path)
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Export VQA views and print the report."""

    args = parse_args(argv)
    report = export_vqa_training_views_v2(
        args.source_jsonl,
        args.annotation_jsonl,
        args.output_jsonl,
        capabilities=load_vqa_capabilities_v2(
            args.capability_json
        ),
        skip_errors=args.skip_errors,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
