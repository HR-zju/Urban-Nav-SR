#!/usr/bin/env python3
"""Run a small commercial-provider UrbanNav smoke collection and HTML review."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

ROUTE_SERVER_ROOT = Path(__file__).resolve().parents[2]
if str(ROUTE_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(ROUTE_SERVER_ROOT))

from tools.dataset_processing.collect_urbannav_commercial_review_gt import collect_commercial_review
from tools.evaluation.build_urbannav_combined_overview_html import build_combined_overview_html


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path, help="Commercial review package output directory.")
    parser.add_argument("--review-dir", required=True, type=Path, help="HTML review output directory.")
    parser.add_argument("--config", default=ROUTE_SERVER_ROOT / "config" / "urbannav_global_gt_collection.json", type=Path)
    parser.add_argument("--provider", choices=["tomtom"], default="tomtom")
    parser.add_argument("--target-cases", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--origins-per-region", type=int, default=8)
    parser.add_argument("--candidate-limit", type=int, default=5)
    parser.add_argument("--search-radius-m", type=int, default=1500)
    parser.add_argument("--acceptable-route-ratio", type=float, default=None)
    parser.add_argument("--min-route-distance-m", type=float, default=None)
    parser.add_argument("--max-route-distance-m", type=float, default=None)
    parser.add_argument("--throttle-s", type=float, default=0.25)
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument("--case-concurrency", type=int, default=2)
    parser.add_argument("--route-concurrency", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tomtom-key", default=None)
    parser.add_argument("--tomtom-search-url", default=None)
    parser.add_argument("--tomtom-routing-url", default=None)
    parser.add_argument("--max-samples", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    """Run commercial smoke collection and HTML review.

    Returns:
        None.
    """

    asyncio.run(run_smoke(parse_args()))


async def run_smoke(args: argparse.Namespace) -> dict[str, str]:
    """Run commercial collection and build a review HTML.

    Args:
        args: Parsed arguments.

    Returns:
        Output path dictionary.
    """

    await collect_commercial_review(args)
    overview = build_combined_overview_html(
        output_dir=args.review_dir,
        plan_path=ROUTE_SERVER_ROOT / "config" / "urbannav_10k_distribution_plan.json",
        packages=[(f"{args.provider} commercial", args.output_root / "task_gt.jsonl")],
        max_samples=args.max_samples,
        title="UrbanNav Commercial Provider Smoke Review",
    )
    return {"index_html": overview["index_html"], "manifest": overview["manifest"]}


if __name__ == "__main__":
    main()
