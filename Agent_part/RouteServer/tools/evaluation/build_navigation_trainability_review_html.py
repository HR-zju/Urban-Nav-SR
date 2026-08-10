#!/usr/bin/env python3
"""Build a self-contained V1 navigation dataset-suitability review page."""

from __future__ import annotations

import argparse
import json

from mcp_server.data.quality.review import (
    NavigationQualityReviewConfigV1,
    write_navigation_trainability_review_html_v1,
)


def main() -> int:
    """Parse arguments and write the portable suitability review artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--filter-jsonl", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--output-manifest-json")
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help="Zero-based offset into deterministic sidecar order for paginated review.",
    )
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--thumbnail-max-width-px", type=int, default=960)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--max-temporal-frames", type=int, default=5)
    parser.add_argument("--title", default="UrbanNav Navigation Dataset Suitability Review")
    args = parser.parse_args()
    report = write_navigation_trainability_review_html_v1(
        source_jsonl=args.source_jsonl,
        filter_jsonl=args.filter_jsonl,
        output_html=args.output_html,
        output_manifest_json=args.output_manifest_json,
        config=NavigationQualityReviewConfigV1(
            sample_offset=args.sample_offset,
            max_samples=args.max_samples,
            thumbnail_max_width_px=args.thumbnail_max_width_px,
            jpeg_quality=args.jpeg_quality,
            max_temporal_frames=args.max_temporal_frames,
        ),
        title=args.title,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
