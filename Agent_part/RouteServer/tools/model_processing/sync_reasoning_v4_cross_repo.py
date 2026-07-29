#!/usr/bin/env python3
"""Generate the ms-swift Reasoning V4 contract and prompt-policy mirrors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence


_GENERATED_HEADER = (
    "# Generated from RouteServer canonical Reasoning V4. "
    "Do not edit by hand.\n"
)
_DIRECTION_GENERATED_HEADER = (
    "# Generated from RouteServer canonical Stage-A direction profiles. "
    "Do not edit by hand.\n"
)


def render_direction_profile_mirror(source_text: str) -> str:
    """Rewrite the package-local ReasonCode import for ms-swift."""

    expected = "from mcp_server.contracts.reasoning import ReasonCode"
    if source_text.count(expected) != 1:
        raise ValueError(
            "canonical direction ReasonCode import is missing or ambiguous"
        )
    return _DIRECTION_GENERATED_HEADER + source_text.replace(
        expected,
        "from .reasoning import ReasonCode",
    )


def render_reasoning_v4_contract_mirror(source_text: str) -> str:
    """Rewrite only the package-local ReasonCode import for ms-swift."""

    expected = (
        "from mcp_server.contracts.reasoning import ReasonCode"
    )
    if source_text.count(expected) != 1:
        raise ValueError("canonical ReasonCode import is missing or ambiguous")
    return _GENERATED_HEADER + source_text.replace(
        expected,
        "from .reasoning import ReasonCode",
    )


def render_reasoning_v4_dataset_contract_mirror(
    source_text: str,
) -> str:
    """Rewrite dataset-contract imports for the ms-swift package."""

    replacements = {
        "from mcp_server.contracts.direction_profile import (": (
            "from ._generated_direction_profile import ("
        ),
        "from mcp_server.contracts.reasoning import ReasonCode": (
            "from .reasoning import ReasonCode"
        ),
        "from mcp_server.contracts.reasoning_v4 import (": (
            "from ._generated_reasoning_v4_contract import ("
        ),
    }
    rendered = source_text
    for expected, replacement in replacements.items():
        if rendered.count(expected) != 1:
            raise ValueError(
                "canonical dataset-contract import is missing or ambiguous: "
                f"{expected}"
            )
        rendered = rendered.replace(expected, replacement)
    return _GENERATED_HEADER + rendered


def render_reasoning_v4_prompt_policy_mirror(source_text: str) -> str:
    """Rewrite canonical prompt-policy imports for the ms-swift package."""

    replacements = {
        "from mcp_server.contracts.direction_profile import (": (
            "from ._generated_direction_profile import ("
        ),
        "from mcp_server.contracts.reasoning_v4 import (": (
            "from ._generated_reasoning_v4_contract import ("
        ),
    }
    rendered = source_text
    for expected, replacement in replacements.items():
        if rendered.count(expected) != 1:
            raise ValueError(
                "canonical prompt-policy import is missing or ambiguous: "
                f"{expected}"
            )
        rendered = rendered.replace(expected, replacement)
    return _GENERATED_HEADER + rendered


def render_reasoning_v4_compact_mirror(source_text: str) -> str:
    """Rewrite compact-projector imports for the ms-swift package."""

    replacements = {
        "from mcp_server.agent.reasoning_v4_text import (": (
            "from .reasoning_v4_codec import ("
        ),
        "from mcp_server.contracts.direction_profile import (": (
            "from ._generated_direction_profile import ("
        ),
        "from mcp_server.contracts.reasoning_v4 import (": (
            "from ._generated_reasoning_v4_contract import ("
        ),
    }
    rendered = source_text
    for expected, replacement in replacements.items():
        if rendered.count(expected) != 1:
            raise ValueError(
                "canonical compact-projector import is missing or ambiguous: "
                f"{expected}"
            )
        rendered = rendered.replace(expected, replacement)
    return _GENERATED_HEADER + rendered


def sync_reasoning_v4_cross_repo(
    ms_swift_root: str | Path,
    *,
    check: bool = False,
) -> dict[str, object]:
    """Write or verify generated Python and JSON contract mirrors."""

    route_root = Path(__file__).resolve().parents[2]
    ms_root = Path(ms_swift_root).expanduser().resolve()
    pairs = (
        (
            route_root
            / "src/mcp_server/contracts/direction_profile.py",
            ms_root
            / "swift/sidewalk_nav/_generated_direction_profile.py",
            render_direction_profile_mirror,
        ),
        (
            route_root
            / "src/mcp_server/contracts/reasoning_v4.py",
            ms_root
            / "swift/sidewalk_nav/_generated_reasoning_v4_contract.py",
            render_reasoning_v4_contract_mirror,
        ),
        (
            route_root
            / "src/mcp_server/contracts/reasoning_dataset_v4.py",
            ms_root
            / "swift/sidewalk_nav/_generated_reasoning_dataset_v4.py",
            render_reasoning_v4_dataset_contract_mirror,
        ),
        (
            route_root
            / "src/mcp_server/agent/reasoning_v4_prompt_policy.py",
            ms_root
            / "swift/sidewalk_nav/_generated_reasoning_v4_prompt_policy.py",
            render_reasoning_v4_prompt_policy_mirror,
        ),
        (
            route_root
            / "src/mcp_server/data/reasoning_v4_compact.py",
            ms_root
            / "swift/sidewalk_nav/_generated_reasoning_v4_compact.py",
            render_reasoning_v4_compact_mirror,
        ),
    )
    written: list[str] = []
    stale: list[str] = []
    hashes: dict[str, str] = {}
    for source_path, destination, renderer in pairs:
        rendered = renderer(source_path.read_text(encoding="utf-8"))
        hashes[destination.name] = hashlib.sha256(
            rendered.encode("utf-8")
        ).hexdigest()
        current = (
            destination.read_text(encoding="utf-8")
            if destination.is_file()
            else None
        )
        if current == rendered:
            continue
        stale.append(str(destination))
        if not check:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(rendered, encoding="utf-8")
            written.append(str(destination))
    artifact_hashes: dict[str, str] = {}
    for filename in (
        "reasoning_v4.schema.json",
        "reasoning_training_view_v4.schema.json",
        "reasoning_v4_golden_valid.json",
        "reasoning_v4_golden_invalid.json",
        "contract_digest.json",
    ):
        source_path = route_root / "schemas/reasoning_v4" / filename
        destination = (
            ms_root
            / "swift/sidewalk_nav/schemas/reasoning_v4"
            / filename
        )
        rendered = source_path.read_text(encoding="utf-8")
        artifact_hashes[filename] = hashlib.sha256(
            rendered.encode("utf-8")
        ).hexdigest()
        current = (
            destination.read_text(encoding="utf-8")
            if destination.is_file()
            else None
        )
        if current == rendered:
            continue
        stale.append(str(destination))
        if not check:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(rendered, encoding="utf-8")
            written.append(str(destination))
    report: dict[str, object] = {
        "schema_version": "reasoning_v4_cross_repo_python_sync_v1",
        "ms_swift_root": str(ms_root),
        "checked": bool(check),
        "written": written,
        "stale": stale,
        "sha256": hashes,
        "artifact_sha256": artifact_hashes,
        "passed": not stale if check else True,
    }
    if check and stale:
        raise ValueError(
            "stale Reasoning V4 generated Python mirrors: "
            + ", ".join(stale)
        )
    return report


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse cross-repository sync arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ms-swift-root",
        required=True,
        type=Path,
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Generate or verify mirrors and print the sync report."""

    args = parse_args(argv)
    report = sync_reasoning_v4_cross_repo(
        args.ms_swift_root,
        check=args.check,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
