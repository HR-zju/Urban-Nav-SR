#!/usr/bin/env python3
"""Validate a frozen UrbanNav benchmark package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcp_server.eval.urbannav_validator import UrbanNavValidationConfig, validate_urbannav_package


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments with GT/cache paths and validation tolerances.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, type=Path, help="Path to task_gt.jsonl")
    parser.add_argument("--poi-cache", type=Path, default=None, help="Path to poi_cache.jsonl")
    parser.add_argument("--route-cache", type=Path, default=None, help="Path to route_cache.jsonl")
    parser.add_argument("--rejects", type=Path, default=None, help="Path to task_rejects.jsonl")
    parser.add_argument("--origin-tolerance-m", type=float, default=30.0)
    parser.add_argument("--destination-tolerance-m", type=float, default=30.0)
    parser.add_argument("--multi-stop-continuity-tolerance-m", type=float, default=5.0)
    parser.add_argument("--min-inter-stop-distance-m", type=float, default=0.0)
    parser.add_argument("--require-disjoint-multistop-target-types", action="store_true")
    parser.add_argument("--require-full-multistop-route-matrix", action="store_true")
    parser.add_argument("--allow-missing-cache-keys", action="store_true")
    parser.add_argument("--allow-missing-ranked-destinations", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None, help="Output validation_report.json path")
    return parser.parse_args()


def main() -> None:
    """Run package validation and write a JSON report.

    Returns:
        None. Prints the report to stdout unless ``--output-json`` is supplied.
    """

    args = parse_args()
    config = UrbanNavValidationConfig(
        origin_tolerance_m=args.origin_tolerance_m,
        destination_tolerance_m=args.destination_tolerance_m,
        require_cache_keys=not args.allow_missing_cache_keys,
        require_ranked_destinations=not args.allow_missing_ranked_destinations,
        multi_stop_continuity_tolerance_m=args.multi_stop_continuity_tolerance_m,
        min_inter_stop_distance_m=args.min_inter_stop_distance_m,
        require_disjoint_multistop_target_types=args.require_disjoint_multistop_target_types,
        require_full_multistop_route_matrix=args.require_full_multistop_route_matrix,
    )
    report = validate_urbannav_package(
        gt_path=str(args.ground_truth),
        poi_cache_path=str(args.poi_cache) if args.poi_cache else None,
        route_cache_path=str(args.route_cache) if args.route_cache else None,
        rejects_path=str(args.rejects) if args.rejects else None,
        config=config,
    ).to_dict()
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
