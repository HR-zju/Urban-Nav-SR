#!/usr/bin/env python3
"""Evaluate frozen Navigation VQA V2 predictions offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.vqa.eligibility import load_vqa_capabilities_v2
from mcp_server.eval.openloop.vqa_v2 import evaluate_vqa_predictions_v2


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse evaluator arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-jsonl", required=True, type=Path)
    parser.add_argument("--prediction-jsonl", required=True, type=Path)
    parser.add_argument("--capability-json", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Evaluate predictions and optionally persist the report."""

    args = parse_args(argv)
    report = evaluate_vqa_predictions_v2(
        args.annotation_jsonl,
        args.prediction_jsonl,
        capabilities=load_vqa_capabilities_v2(
            args.capability_json
        ),
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
