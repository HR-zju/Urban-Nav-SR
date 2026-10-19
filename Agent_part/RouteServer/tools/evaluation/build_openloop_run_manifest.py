#!/usr/bin/env python3
"""Validate and canonically render an UrbanNav open-loop run manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from mcp_server.eval.openloop.openloop_registry import build_openloop_run_manifest_v1, write_validated_openloop_run_manifest_v1


def main() -> None:
    """Parse arguments and write one canonical run manifest."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--stage", choices=("stage_i", "stage_ii"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--repositories", type=Path)
    parser.add_argument("--context-identity", type=Path)
    parser.add_argument("--generation-parameters", type=Path)
    parser.add_argument("--hardware", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--precision")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.input is not None:
        if any(
            value is not None
            for value in (
                args.package,
                args.run_id,
                args.stage,
                args.checkpoint,
                args.repositories,
            )
        ):
            parser.error("--input cannot be combined with assembly arguments")
        write_validated_openloop_run_manifest_v1(args.input, args.output)
        return
    required = {
        "--package": args.package,
        "--run-id": args.run_id,
        "--stage": args.stage,
        "--checkpoint": args.checkpoint,
        "--repositories": args.repositories,
        "--batch-size": args.batch_size,
        "--precision": args.precision,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("missing assembly arguments: " + ", ".join(missing))
    build_openloop_run_manifest_v1(
        run_id=args.run_id,
        stage=args.stage,
        package_json=args.package,
        checkpoint_json=args.checkpoint,
        repositories_json=args.repositories,
        output_json=args.output,
        context_identity_json=args.context_identity,
        generation_parameters_json=args.generation_parameters,
        hardware_json=args.hardware,
        batch_size=args.batch_size,
        precision=args.precision,
    )


if __name__ == "__main__":
    main()
