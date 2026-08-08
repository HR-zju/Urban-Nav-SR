#!/usr/bin/env python3
"""Generate the ms-swift mirror of the canonical Stage I SFT contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence


_GENERATED_HEADER = (
    "# Generated from RouteServer canonical UrbanNav Stage I SFT contract. "
    "Do not edit by hand.\n"
)


def render_stage_i_contract_mirror(source_text: str) -> str:
    """Rewrite canonical imports for the ms-swift package.

    Args:
        source_text: Canonical RouteServer contract source.

    Returns:
        Generated ms-swift package-local source.

    Raises:
        ValueError: If an expected canonical import is missing or ambiguous.
    """

    replacements = {
        "from mcp_server.contracts.reasoning import ReasonCode": (
            "from .contracts.reasoning import ReasonCode"
        ),
        "from mcp_server.contracts.reasoning_dataset_v4 import (": (
            "from ._generated_reasoning_dataset_v4 import ("
        ),
        "from mcp_server.contracts.reasoning_v4 import DecisionV4, ReasoningTargetV4": (
            "from ._generated_reasoning_v4_contract import DecisionV4, ReasoningTargetV4"
        ),
        "from mcp_server.contracts.vqa_v2 import VQAQuestionV2": (
            "from ._generated_vqa_v2_contract import VQAQuestionV2"
        ),
    }
    rendered = source_text
    for expected, replacement in replacements.items():
        if rendered.count(expected) != 1:
            raise ValueError(
                "canonical Stage I import is missing or ambiguous: "
                f"{expected}"
            )
        rendered = rendered.replace(expected, replacement)
    return _GENERATED_HEADER + rendered


def sync_stage_i_sft_cross_repo(
    ms_swift_root: str | Path,
    *,
    check: bool = False,
) -> dict[str, object]:
    """Write or verify the generated Python and JSON contract mirrors.

    Args:
        ms_swift_root: Sibling ms-swift repository root.
        check: Compare without writing when true.

    Returns:
        Machine-readable mirror paths, hashes, and parity status.

    Raises:
        ValueError: If generated mirrors are stale in check mode.
    """

    route_root = Path(__file__).resolve().parents[2]
    ms_root = Path(ms_swift_root).expanduser().resolve()
    source_path = route_root / "src/mcp_server/contracts/stage_i_sft.py"
    destination = ms_root / "swift/sidewalk_nav/_generated_stage_i_sft.py"
    rendered = render_stage_i_contract_mirror(source_path.read_text(encoding="utf-8"))
    written: list[str] = []
    stale: list[str] = []
    if not destination.is_file() or destination.read_text(encoding="utf-8") != rendered:
        stale.append(str(destination))
        if not check:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(rendered, encoding="utf-8")
            written.append(str(destination))

    artifact_hashes: dict[str, str] = {}
    for filename in (
        "stage_i_training_view_v1.schema.json",
        "stage_i_training_view_v2.schema.json",
        "stage_i_projection_drop_v1.schema.json",
        "stage_i_instruction_provenance_v1.schema.json",
        "contract_digest.json",
    ):
        source = route_root / "schemas/stage_i_sft" / filename
        target = ms_root / "swift/sidewalk_nav/schemas/stage_i_sft" / filename
        content = source.read_text(encoding="utf-8")
        artifact_hashes[filename] = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if not target.is_file() or target.read_text(encoding="utf-8") != content:
            stale.append(str(target))
            if not check:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                written.append(str(target))

    if check and stale:
        raise ValueError("stale Stage I generated mirrors: " + ", ".join(stale))
    return {
        "schema_version": "urbannav_stage_i_cross_repo_sync_v1",
        "ms_swift_root": str(ms_root),
        "checked": check,
        "written": written,
        "stale": stale,
        "python_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "artifact_sha256": artifact_hashes,
        "passed": not stale if check else True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ms-swift-root", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Generate or check the cross-repository mirrors."""

    args = parse_args(argv)
    report = sync_stage_i_sft_cross_repo(args.ms_swift_root, check=args.check)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
