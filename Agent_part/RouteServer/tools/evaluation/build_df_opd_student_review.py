#!/usr/bin/env python3
"""Build a local-only private Student-versus-GT DF-OPD review bundle."""

from __future__ import annotations

import argparse
import json

from mcp_server.data.df_opd.student_review import build_df_opd_student_review_bundle_v1


def main() -> int:
    """Parse local paths, build the review, and print its receipt."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-jsonl", required=True)
    parser.add_argument("--prediction-jsonl", action="append", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--output-manifest-json", required=True)
    parser.add_argument("--review-selection-jsonl")
    parser.add_argument("--media-root")
    parser.add_argument("--max-samples", default=50, type=int)
    parser.add_argument("--visualization-index")
    parser.add_argument("--scorecard-json")
    parser.add_argument("--run-id")
    parser.add_argument("--run-root")
    args = parser.parse_args()
    receipt = build_df_opd_student_review_bundle_v1(
        canonical_jsonl=args.canonical_jsonl,
        prediction_jsonls=args.prediction_jsonl,
        output_html=args.output_html,
        output_manifest_json=args.output_manifest_json,
        review_selection_jsonl=args.review_selection_jsonl,
        media_root=args.media_root,
        max_samples=args.max_samples,
        visualization_index_json=args.visualization_index,
        scorecard_json=args.scorecard_json,
        run_id=args.run_id,
        run_root=args.run_root,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
