#!/usr/bin/env python3
"""Re-evaluate saved UrbanNav predictions with offline v2 metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcp_server.eval.urbannav.urbannav_io import load_urbannav_cases, load_urbannav_predictions
from mcp_server.eval.urbannav.urbannav_metrics_v2 import UrbanNavEvalV2Config, evaluate_predictions_v2, write_eval_v2_artifacts


def parse_args() -> argparse.Namespace:
    """Parse v2 evaluator command-line arguments.

    Returns:
        Parsed command-line namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--destination-tolerance-m", type=float, default=30.0)
    parser.add_argument("--endpoint-tolerance-m", type=float, default=10.0)
    parser.add_argument("--continuity-tolerance-m", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    """Run v2 evaluation and print compact artifact and metric metadata."""

    args = parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"dataset not found: {args.dataset}")
    if not args.predictions.exists():
        raise SystemExit(f"predictions not found: {args.predictions}")
    config = UrbanNavEvalV2Config(
        destination_tolerance_m=float(args.destination_tolerance_m),
        endpoint_tolerance_m=float(args.endpoint_tolerance_m),
        multi_stop_continuity_tolerance_m=float(args.continuity_tolerance_m),
    )
    rows, summary, sensitivity = evaluate_predictions_v2(
        load_urbannav_cases(args.dataset),
        load_urbannav_predictions(args.predictions),
        config=config,
    )
    run_id = str(args.run_id or args.predictions.parent.name)
    paths = write_eval_v2_artifacts(
        rows=rows,
        summary=summary,
        sensitivity=sensitivity,
        output_dir=args.output_dir,
        run_id=run_id,
        dataset_path=args.dataset,
        predictions_path=args.predictions,
        config=config,
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "artifacts": paths,
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
