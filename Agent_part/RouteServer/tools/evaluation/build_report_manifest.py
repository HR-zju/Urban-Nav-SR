#!/usr/bin/env python3
"""Build a ClearML report_manifest.json from standard media directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mcp_server.data.report_manifest import build_conventional_report_manifest, write_report_manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Build and write one deterministic report manifest."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--default-section",
        help="Explicit fallback section for media without success/failure/review path components.",
    )
    args = parser.parse_args(argv)
    output = args.out or args.experiment_root / "report_manifest.json"
    manifest = build_conventional_report_manifest(
        args.experiment_root, default_section=args.default_section
    )
    write_report_manifest(output, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
