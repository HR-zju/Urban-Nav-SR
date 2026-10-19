#!/usr/bin/env python3
"""Audit Navigation VQA V2 source, annotation, and training artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.vqa.audit import audit_vqa_artifacts_v2
from mcp_server.data.vqa.eligibility import load_vqa_capabilities_v2


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse artifact-audit arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path)
    parser.add_argument("--annotation-jsonl", type=Path)
    parser.add_argument("--training-view-jsonl", type=Path)
    parser.add_argument("--capability-json", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the audit and optionally persist its report."""

    args = parse_args(argv)
    report = audit_vqa_artifacts_v2(
        capabilities=load_vqa_capabilities_v2(
            args.capability_json
        ),
        source_jsonl=args.source_jsonl,
        annotation_jsonl=args.annotation_jsonl,
        training_view_jsonl=args.training_view_jsonl,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
