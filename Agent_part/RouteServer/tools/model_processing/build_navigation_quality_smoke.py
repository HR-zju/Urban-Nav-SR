#!/usr/bin/env python3
"""Build a deterministic stratified source smoke manifest for quality reviews."""

from __future__ import annotations

import argparse
import json

from mcp_server.data.quality.smoke import (
    NavigationQualitySmokeConfigV1,
    build_navigation_quality_smoke_manifest_v1,
)


def main() -> int:
    """Parse source sampling arguments and emit source-only smoke views."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--dataset-source")
    parser.add_argument("--target-per-stratum", type=int, default=24)
    parser.add_argument("--max-rows", type=int, default=160)
    parser.add_argument("--high-speed-mps", type=float, default=3.0)
    parser.add_argument("--seed", default="urbannav-navigation-quality-smoke-v1")
    args = parser.parse_args()
    report = build_navigation_quality_smoke_manifest_v1(
        source_jsonl=args.source_jsonl,
        output_jsonl=args.output_jsonl,
        report_json=args.report_json,
        dataset_source=args.dataset_source,
        config=NavigationQualitySmokeConfigV1(
            target_per_stratum=args.target_per_stratum,
            max_rows=args.max_rows,
            high_speed_mps=args.high_speed_mps,
            seed=args.seed,
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
