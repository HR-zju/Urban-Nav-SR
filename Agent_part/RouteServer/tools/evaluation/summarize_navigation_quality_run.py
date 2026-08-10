#!/usr/bin/env python3
"""Write one read-only aggregate QC receipt for a navigation-quality run."""

from __future__ import annotations

import argparse
import json

from mcp_server.data.quality.reports import summarize_navigation_quality_run_v1


def main() -> int:
    """Parse CLI arguments and serialize the run-level receipt."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = summarize_navigation_quality_run_v1(
        run_root=args.run_root,
        output_json=args.output_json,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
