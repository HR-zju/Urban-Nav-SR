#!/usr/bin/env python3
"""Generate ms-swift mirrors of the canonical pluggable benchmark contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from tools.model_processing.export_benchmark_plugin_contract_artifacts import ARTIFACT_FILENAMES


_HEADER = "# Generated from RouteServer canonical benchmark plugin contract. Do not edit by hand.\n"
_PYTHON_MIRRORS = (
    (
        "src/mcp_server/contracts/benchmark_plugin.py",
        "swift/sidewalk_nav/_generated_benchmark_plugin.py",
    ),
    (
        "src/mcp_server/contracts/citywalker_to_urbannav_adapter.py",
        "swift/sidewalk_nav/_generated_citywalker_to_urbannav_adapter.py",
    ),
)


def sync_benchmark_plugin_cross_repo_v1(ms_swift_root: str | Path, *, check: bool = False) -> dict[str, object]:
    """Write or verify generated Python and JSON mirrors in ms-swift.

    Args:
        ms_swift_root: Sibling ms-swift checkout root.
        check: Verify bytes only when true.

    Returns:
        Sync report including generated Python digest.

    Raises:
        ValueError: If check mode sees a stale mirror.
    """

    route_root = Path(__file__).resolve().parents[2]
    ms_root = Path(ms_swift_root).expanduser().resolve()
    stale: list[str] = []
    written: list[str] = []
    python_digests: dict[str, str] = {}
    for source_relative, target_relative in _PYTHON_MIRRORS:
        source = route_root / source_relative
        destination = ms_root / target_relative
        rendered = _HEADER + source.read_text(encoding="utf-8")
        python_digests[target_relative] = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        if not destination.is_file() or destination.read_text(encoding="utf-8") != rendered:
            stale.append(str(destination))
            if not check:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(rendered, encoding="utf-8")
                written.append(str(destination))
    for filename in ARTIFACT_FILENAMES:
        source_artifact = route_root / "schemas/benchmark_plugin" / filename
        target = ms_root / "swift/sidewalk_nav/schemas/benchmark_plugin" / filename
        content = source_artifact.read_text(encoding="utf-8")
        if not target.is_file() or target.read_text(encoding="utf-8") != content:
            stale.append(str(target))
            if not check:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                written.append(str(target))
    if check and stale:
        raise ValueError("stale benchmark plugin generated mirrors: " + ", ".join(stale))
    return {
        "checked": check,
        "written": written,
        "stale": stale,
        "passed": not stale if check else True,
        "python_sha256": python_digests,
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI sync arguments and print a deterministic sync report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ms-swift-root", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    report = sync_benchmark_plugin_cross_repo_v1(
        args.ms_swift_root,
        check=args.check,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
