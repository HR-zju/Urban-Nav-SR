#!/usr/bin/env python3
"""Generate ms-swift mirrors of the canonical open-loop contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from tools.model_processing.export_openloop_benchmark_contract_artifacts import (
    ARTIFACT_FILENAMES,
)


_HEADER = (
    "# Generated from RouteServer canonical UrbanNav open-loop contract. "
    "Do not edit by hand.\n"
)


def render_openloop_contract_mirror(source_text: str) -> str:
    """Rewrite the canonical direction-profile import for ms-swift."""

    expected = (
        "from mcp_server.contracts.direction_profile import "
        "DirectionLabelStamp"
    )
    if source_text.count(expected) != 1:
        raise ValueError("canonical open-loop direction import is ambiguous")
    return _HEADER + source_text.replace(
        expected,
        "from ._generated_direction_profile import DirectionLabelStamp",
    )


def sync_openloop_contract_cross_repo(
    ms_swift_root: str | Path,
    *,
    check: bool = False,
) -> dict[str, object]:
    """Write or verify generated Python and JSON contract mirrors."""

    route_root = Path(__file__).resolve().parents[2]
    ms_root = Path(ms_swift_root).expanduser().resolve()
    source = route_root / "src/mcp_server/contracts/openloop_benchmark.py"
    destination = ms_root / "swift/sidewalk_nav/_generated_openloop_benchmark.py"
    rendered = render_openloop_contract_mirror(source.read_text(encoding="utf-8"))
    stale: list[str] = []
    written: list[str] = []
    if not destination.is_file() or destination.read_text(encoding="utf-8") != rendered:
        stale.append(str(destination))
        if not check:
            destination.write_text(rendered, encoding="utf-8")
            written.append(str(destination))
    for filename in ARTIFACT_FILENAMES:
        source_artifact = route_root / "schemas/openloop_benchmark" / filename
        target = ms_root / "swift/sidewalk_nav/schemas/openloop_benchmark" / filename
        content = source_artifact.read_text(encoding="utf-8")
        if not target.is_file() or target.read_text(encoding="utf-8") != content:
            stale.append(str(target))
            if not check:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                written.append(str(target))
    if check and stale:
        raise ValueError("stale open-loop generated mirrors: " + ", ".join(stale))
    return {
        "checked": check,
        "written": written,
        "stale": stale,
        "passed": not stale if check else True,
        "python_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Generate or check the sibling ms-swift mirrors."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ms-swift-root", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            sync_openloop_contract_cross_repo(
                args.ms_swift_root,
                check=args.check,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
