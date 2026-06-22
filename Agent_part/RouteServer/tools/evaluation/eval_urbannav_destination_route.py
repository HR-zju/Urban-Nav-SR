#!/usr/bin/env python3
"""Evaluate UrbanNav instruction-to-destination/route predictions.

This evaluator is fully offline. It reads frozen UrbanNav GT and saved
predictions, then computes the five main metrics:

destination_success, hit@5, route_success, median_route_ratio, spl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcp_server.eval.urbannav_io import load_urbannav_cases, load_urbannav_predictions
from mcp_server.eval.urbannav_metrics import UrbanNavEvalConfig, evaluate_predictions, summarize_by


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed ``argparse.Namespace`` with GT path, prediction path,
        tolerance settings, and optional output paths.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, type=Path, help="UrbanNav task_gt JSON/JSONL")
    parser.add_argument("--predictions", required=True, type=Path, help="UrbanNav prediction JSON/JSONL")
    parser.add_argument("--destination-radius-m", type=float, default=30.0, help="Coordinate fallback tolerance")
    parser.add_argument("--endpoint-radius-m", type=float, default=30.0, help="Route endpoint tolerance")
    parser.add_argument("--output-json", type=Path, default=None, help="Write summary JSON")
    parser.add_argument("--output-jsonl", type=Path, default=None, help="Write per-case rows JSONL")
    return parser.parse_args()


def main() -> None:
    r"""Run the offline UrbanNav evaluator.

    The command loads frozen GT and saved predictions, computes per-case rows
    and aggregate metrics, then writes:

    \[
    destination\_success,\ hit@5,\ route\_success,\ median\_route\_ratio,\ SPL
    \]

    Returns:
        None. Results are printed to stdout or written to ``--output-json`` and
        ``--output-jsonl`` when those paths are supplied.
    """

    args = parse_args()
    cases = load_urbannav_cases(args.ground_truth)
    predictions = load_urbannav_predictions(args.predictions)
    config = UrbanNavEvalConfig(
        destination_tolerance_m=args.destination_radius_m,
        endpoint_tolerance_m=args.endpoint_radius_m,
    )
    rows, summary = evaluate_predictions(cases, predictions, config=config)
    report = {
        "summary": summary,
        "by_category": summarize_by(rows, "category"),
        "by_subcategory": summarize_by(rows, "subcategory"),
        "by_city": summarize_by(rows, "city"),
        "by_scene_id": summarize_by(rows, "scene_id"),
        "by_difficulty": summarize_by(rows, "difficulty"),
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output_jsonl:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
