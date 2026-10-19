#!/usr/bin/env python3
"""Build a fail-closed checkpoint training-exposure registry."""

from __future__ import annotations

import argparse
from pathlib import Path

from mcp_server.eval.openloop.openloop_registry import build_checkpoint_exposure_registry_v1


def main() -> None:
    """Parse arguments and write one exposure registry."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--checkpoint-digest", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build_checkpoint_exposure_registry_v1(
        args.spec,
        args.output,
        checkpoint_digest=args.checkpoint_digest,
    )


if __name__ == "__main__":
    main()
