#!/usr/bin/env python3
"""Build a self-contained human review page for language-goal annotations."""

from __future__ import annotations

import argparse
import json

from mcp_server.data.benchmark.language_waypoint_review import (
    LanguageWaypointReviewConfigV1,
    write_language_waypoint_review_html_v1,
)


def main() -> int:
    """Parse review arguments and write deterministic HTML artifacts.

    Returns:
        Process exit code.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--annotation-jsonl", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--output-manifest-json")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--thumbnail-max-width-px", type=int, default=960)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--title", default="UrbanNav Visible Language Goal Review")
    args = parser.parse_args()
    report = write_language_waypoint_review_html_v1(
        source_jsonl=args.source_jsonl,
        annotation_jsonl=args.annotation_jsonl,
        output_html=args.output_html,
        output_manifest_json=args.output_manifest_json,
        config=LanguageWaypointReviewConfigV1(
            max_samples=args.max_samples,
            thumbnail_max_width_px=args.thumbnail_max_width_px,
            jpeg_quality=args.jpeg_quality,
        ),
        title=args.title,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
