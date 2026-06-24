#!/usr/bin/env python3
"""Collect a commercial-provider UrbanNav review split."""

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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path, help="Output benchmark package directory.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Global collection config JSON.")
    parser.add_argument("--provider", choices=["tomtom"], default="tomtom", help="Commercial provider.")
    parser.add_argument("--target-cases", type=int, default=1000, help="Target valid commercial cases.")
    parser.add_argument("--seed", type=int, default=20260623, help="Sampling seed.")
    parser.add_argument("--origins-per-region", type=int, default=20, help="Origins sampled per region.")
    parser.add_argument("--candidate-limit", type=int, default=5, help="Routed candidates per case.")
    parser.add_argument("--search-radius-m", type=int, default=1500, help="POI search radius.")
    parser.add_argument("--acceptable-route-ratio", type=float, default=None, help="Acceptable near-tie ratio.")
    parser.add_argument("--min-route-distance-m", type=float, default=None, help="Minimum route distance.")
    parser.add_argument("--max-route-distance-m", type=float, default=None, help="Maximum route distance.")
    parser.add_argument("--throttle-s", type=float, default=0.25, help="Sleep seconds per attempted case.")
    parser.add_argument("--retry", type=int, default=2, help="Retry count.")
    parser.add_argument("--case-concurrency", type=int, default=2, help="Concurrent cases.")
    parser.add_argument("--route-concurrency", type=int, default=4, help="Concurrent route calls per case.")
    parser.add_argument("--max-attempts", type=int, default=0, help="Optional attempted pair cap.")
    parser.add_argument("--resume", action="store_true", help="Resume existing output package.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output package.")
    parser.add_argument("--dry-run", action="store_true", help="Only write collection plan and manifest.")
    parser.add_argument("--tomtom-key", default=None, help="TomTom API key; TOMTOM_API_KEY is used if omitted.")
    parser.add_argument("--tomtom-search-url", default=None, help="Override TomTom Search API prefix.")
    parser.add_argument("--tomtom-routing-url", default=None, help="Override TomTom Routing API prefix.")
    return parser.parse_args()


def main() -> None:
    """Run commercial review collection.

    Returns:
        None.
    """

    asyncio.run(collect_commercial_review(parse_args()))


async def collect_commercial_review(args: argparse.Namespace) -> dict[str, Any]:
    """Collect the commercial review split.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Collection manifest.
    """

    collector_args = SimpleNamespace(
        output_root=args.output_root,
        config=args.config,
        target_cases=args.target_cases,
        seed=args.seed,
        origins_per_region=args.origins_per_region,
        candidate_limit=args.candidate_limit,
        search_radius_m=args.search_radius_m,
        acceptable_route_ratio=args.acceptable_route_ratio,
        min_route_distance_m=args.min_route_distance_m,
        max_route_distance_m=args.max_route_distance_m,
        throttle_s=args.throttle_s,
        retry=args.retry,
        case_concurrency=args.case_concurrency,
        route_concurrency=args.route_concurrency,
        resume=args.resume,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        max_attempts=args.max_attempts,
        provider=args.provider,
        overpass_url=None,
        osrm_url=None,
        osrm_profile="foot",
        tomtom_key=args.tomtom_key,
        tomtom_search_url=args.tomtom_search_url,
        tomtom_routing_url=args.tomtom_routing_url,
    )
    return await collect_global_gt(collector_args)


if __name__ == "__main__":
    main()
