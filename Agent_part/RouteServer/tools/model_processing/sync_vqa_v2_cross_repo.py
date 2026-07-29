#!/usr/bin/env python3
"""Generate ms-swift Navigation VQA V2 Python mirrors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

from tools.model_processing.export_vqa_v2_contract_artifacts import (
    ARTIFACT_FILENAMES,
)

_GENERATED_HEADER = (
    "# Generated from RouteServer canonical Navigation VQA V2. "
    "Do not edit by hand.\n"
)
_DIRECTION_GENERATED_HEADER = (
    "# Generated from RouteServer canonical Stage-A direction profiles. "
    "Do not edit by hand.\n"
)


def _replace_once(
    text: str,
    expected: str,
    replacement: str,
) -> str:
    """Replace one required import exactly once."""

    if text.count(expected) != 1:
        raise ValueError(
            "canonical import is missing or ambiguous: "
            f"{expected}"
        )
    return text.replace(expected, replacement)


def render_direction_profile_mirror(source_text: str) -> str:
    """Rewrite the package-local ReasonCode import."""

    rendered = _replace_once(
        source_text,
        "from mcp_server.contracts.reasoning import ReasonCode",
        "from .reasoning import ReasonCode",
    )
    return _DIRECTION_GENERATED_HEADER + rendered


def render_vqa_contract_mirror(source_text: str) -> str:
    """Rewrite VQA contract imports for ms-swift."""

    rendered = _replace_once(
        source_text,
        "from mcp_server.contracts.reasoning import ReasonCode",
        "from .reasoning import ReasonCode",
    )
    return _GENERATED_HEADER + rendered


def render_vqa_dataset_mirror(source_text: str) -> str:
    """Rewrite dataset-contract imports for ms-swift."""

    rendered = source_text
    replacements = {
        "from mcp_server.contracts.direction_profile import "
        "DirectionLabelStamp": (
            "from ._generated_direction_profile import "
            "DirectionLabelStamp"
        ),
        "from mcp_server.contracts.reasoning import ReasonCode": (
            "from .reasoning import ReasonCode"
        ),
        "from mcp_server.contracts.reasoning_dataset_v4 import (": (
            "from ._generated_reasoning_dataset_v4 import ("
        ),
        "from mcp_server.contracts.vqa_v2 import (": (
            "from ._generated_vqa_v2_contract import ("
        ),
    }
    for expected, replacement in replacements.items():
        rendered = _replace_once(
            rendered,
            expected,
            replacement,
        )
    return _GENERATED_HEADER + rendered


def render_vqa_codec_mirror(source_text: str) -> str:
    """Rewrite the VQA contract import for training-side codec."""

    rendered = _replace_once(
        source_text,
        "from mcp_server.contracts.vqa_v2 import (",
        "from ._generated_vqa_v2_contract import (",
    )
    return _GENERATED_HEADER + rendered


def render_vqa_prompt_policy_mirror(source_text: str) -> str:
    """Rewrite canonical prompt-policy imports for ms-swift."""

    rendered = source_text
    replacements = {
        "from mcp_server.agent.vqa_v2_text import (": (
            "from .vqa_v2_codec import ("
        ),
        "from mcp_server.contracts.reasoning_dataset_v4 import "
        "NavObservationV4": (
            "from ._generated_reasoning_dataset_v4 import "
            "NavObservationV4"
        ),
        "from mcp_server.contracts.vqa_v2 import (": (
            "from ._generated_vqa_v2_contract import ("
        ),
    }
    for expected, replacement in replacements.items():
        rendered = _replace_once(
            rendered,
            expected,
            replacement,
        )
    return _GENERATED_HEADER + rendered


def sync_vqa_v2_cross_repo(
    ms_swift_root: str | Path,
    *,
    check: bool = False,
) -> dict[str, object]:
    """Write or verify generated VQA V2 Python and JSON mirrors.

    Args:
        ms_swift_root: Sibling ms-swift repository.
        check: Compare without writing when true.

    Returns:
        Written/stale paths and generated hashes.

    Raises:
        ValueError: If check mode finds stale mirrors.
    """

    route_root = Path(__file__).resolve().parents[2]
    ms_root = Path(ms_swift_root).expanduser().resolve()
    pairs: tuple[
        tuple[Path, Path, Callable[[str], str]],
        ...,
    ] = (
        (
            route_root
            / "src/mcp_server/contracts/direction_profile.py",
            ms_root
            / "swift/sidewalk_nav/_generated_direction_profile.py",
            render_direction_profile_mirror,
        ),
        (
            route_root / "src/mcp_server/contracts/vqa_v2.py",
            ms_root
            / "swift/sidewalk_nav/_generated_vqa_v2_contract.py",
            render_vqa_contract_mirror,
        ),
        (
            route_root
            / "src/mcp_server/contracts/vqa_dataset_v2.py",
            ms_root
            / "swift/sidewalk_nav/_generated_vqa_dataset_v2.py",
            render_vqa_dataset_mirror,
        ),
        (
            route_root / "src/mcp_server/agent/vqa_v2_text.py",
            ms_root / "swift/sidewalk_nav/vqa_v2_codec.py",
            render_vqa_codec_mirror,
        ),
        (
            route_root
            / "src/mcp_server/agent/vqa_v2_prompt_policy.py",
            ms_root
            / "swift/sidewalk_nav/_generated_vqa_v2_prompt_policy.py",
            render_vqa_prompt_policy_mirror,
        ),
    )
    written: list[str] = []
    stale: list[str] = []
    hashes: dict[str, str] = {}
    for source, destination, renderer in pairs:
        text = renderer(source.read_text(encoding="utf-8"))
        hashes[destination.name] = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()
        current = (
            destination.read_text(encoding="utf-8")
            if destination.is_file()
            else None
        )
        if current == text:
            continue
        stale.append(str(destination))
        if not check:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
            written.append(str(destination))
    artifact_hashes: dict[str, str] = {}
    for filename in ARTIFACT_FILENAMES:
        source = route_root / "schemas/vqa_v2" / filename
        destination = (
            ms_root
            / "swift/sidewalk_nav/schemas/vqa_v2"
            / filename
        )
        text = source.read_text(encoding="utf-8")
        artifact_hashes[filename] = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()
        current = (
            destination.read_text(encoding="utf-8")
            if destination.is_file()
            else None
        )
        if current == text:
            continue
        stale.append(str(destination))
        if not check:
            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            destination.write_text(text, encoding="utf-8")
            written.append(str(destination))
    report: dict[str, object] = {
        "schema_version": "vqa_v2_cross_repo_python_sync_v1",
        "ms_swift_root": str(ms_root),
        "checked": check,
        "written": written,
        "stale": stale,
        "sha256": hashes,
        "artifact_sha256": artifact_hashes,
        "passed": not stale if check else True,
    }
    if check and stale:
        raise ValueError(
            "stale VQA V2 generated Python/JSON mirrors: "
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
    """Generate or verify VQA mirrors and print the report."""

    args = parse_args(argv)
    report = sync_vqa_v2_cross_repo(
        args.ms_swift_root,
        check=args.check,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
