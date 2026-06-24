#!/usr/bin/env python3
"""Run a global UrbanNav pilot collection and build Leaflet review HTML."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

ROUTE_SERVER_ROOT = Path(__file__).resolve().parents[2]
if str(ROUTE_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(ROUTE_SERVER_ROOT))

from tools.dataset_processing.collect_urbannav_global_gt import DEFAULT_CONFIG, collect_global_gt
from tools.evaluation.build_urbannav_leaflet_review_html import build_leaflet_review_html


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path, help="Benchmark package output directory.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Collection config JSON.")
    parser.add_argument("--target-cases", type=int, default=1000, help="Target valid GT cases.")
    parser.add_argument("--review-cases", type=int, default=160, help="Cases rendered in review HTML.")
    parser.add_argument("--review-dir", type=Path, default=None, help="Review HTML output directory.")
    parser.add_argument("--provider", choices=["osm", "tomtom"], default="osm", help="Live provider backend.")
    parser.add_argument("--seed", type=int, default=20260623, help="Sampling seed.")
    parser.add_argument("--origins-per-region", type=int, default=45, help="Origins sampled per region.")
    parser.add_argument("--candidate-limit", type=int, default=3, help="Routed candidates per case.")
    parser.add_argument("--search-radius-m", type=int, default=1200, help="POI search radius.")
    parser.add_argument("--acceptable-route-ratio", type=float, default=None, help="Acceptable near-tie ratio.")
    parser.add_argument("--min-route-distance-m", type=float, default=None, help="Minimum route distance.")
    parser.add_argument("--max-route-distance-m", type=float, default=None, help="Maximum route distance.")
    parser.add_argument("--throttle-s", type=float, default=0.8, help="Sleep seconds per attempted case.")
    parser.add_argument("--retry", type=int, default=2, help="Retry count.")
    parser.add_argument("--case-concurrency", type=int, default=2, help="Concurrent cases.")
    parser.add_argument("--route-concurrency", type=int, default=3, help="Concurrent route calls per case.")
    parser.add_argument("--max-attempts", type=int, default=0, help="Optional attempted pair cap.")
    parser.add_argument("--resume", action="store_true", help="Resume existing output package.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output package.")
    parser.add_argument("--dry-run", action="store_true", help="Only write collection plan and manifest.")
    parser.add_argument("--overpass-url", default=None, help="Override Overpass endpoint.")
    parser.add_argument("--osrm-url", default=None, help="Override OSRM endpoint prefix.")
    parser.add_argument("--osrm-profile", default="foot", help="OSRM profile.")
    parser.add_argument("--tomtom-key", default=None, help="TomTom API key.")
    parser.add_argument("--tomtom-search-url", default=None, help="Override TomTom Search API prefix.")
    parser.add_argument("--tomtom-routing-url", default=None, help="Override TomTom Routing API prefix.")
    return parser.parse_args()


def main() -> None:
    """Run pilot collection and review generation.

    Returns:
        None.
    """

    asyncio.run(run_pilot_review(parse_args()))


async def run_pilot_review(args: argparse.Namespace) -> dict[str, Any]:
    """Run collection and create review HTML.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Summary containing manifest and review paths.
    """

    collector_args = SimpleNamespace(**vars(args))
    manifest = await collect_global_gt(collector_args)
    review_dir = args.review_dir or (args.output_root / "review_leaflet")
    review = {"skipped": True, "reason": "dry_run"} if args.dry_run else build_leaflet_review_html(
        gt_path=args.output_root / "task_gt.jsonl",
        output_dir=review_dir,
        title=f"UrbanNav Global Pilot Review ({args.provider})",
        max_cases=args.review_cases,
    )
    return {"manifest": manifest, "review": review}


if __name__ == "__main__":
    main()
