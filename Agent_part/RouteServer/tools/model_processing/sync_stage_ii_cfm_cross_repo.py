#!/usr/bin/env python3
"""Generate ms-swift mirrors of the canonical Stage II CFM contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence


_GENERATED_HEADER = (
    "# Generated from RouteServer canonical UrbanNav Stage II CFM contract. "
    "Do not edit by hand.\n"
)


def render_stage_ii_contract_mirror(source_text: str) -> str:
    """Rewrite canonical imports for the ms-swift package."""

    replacements = {
        "from mcp_server.contracts.reasoning import ReasonCode": (
            "from .reasoning import ReasonCode"
        ),
        "from mcp_server.contracts.reasoning_dataset_v4 import (": (
            "from ._generated_reasoning_dataset_v4 import ("
        ),
        "from mcp_server.contracts.stage_i_sft import (": (
            "from ._generated_stage_i_sft import ("
        ),
    }
    rendered = source_text
    for expected, replacement in replacements.items():
        if rendered.count(expected) != 1:
            raise ValueError(
                "canonical Stage II import is missing or ambiguous: "
                f"{expected}"
            )
        rendered = rendered.replace(expected, replacement)
    return _GENERATED_HEADER + rendered


def sync_stage_ii_cfm_cross_repo(
    ms_swift_root: str | Path,
    *,
    check: bool = False,
) -> dict[str, object]:
    """Write or verify generated Python and JSON mirrors."""

    route_root = Path(__file__).resolve().parents[2]
    ms_root = Path(ms_swift_root).expanduser().resolve()
    source_path = route_root / "src/mcp_server/contracts/stage_ii_cfm.py"
    destination = ms_root / "swift/sidewalk_nav/_generated_stage_ii_cfm.py"
    rendered = render_stage_ii_contract_mirror(source_path.read_text(encoding="utf-8"))
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
        "stage_ii_context_cache_v1.schema.json",
        "stage_ii_reasoning_context_v2.schema.json",
        "stage_ii_trajectory_target_v1.schema.json",
        "stage_ii_trajectory_qc_v1.schema.json",
        "stage_ii_trajectory_target_v2.schema.json",
        "stage_ii_trajectory_qc_v2.schema.json",
        "stage_ii_trajectory_sidecar_v2.schema.json",
        "stage_ii_trajectory_source_registry_v1.schema.json",
        "stage_ii_legacy_direct_target_v1.schema.json",
        "stage_ii_cfm_training_view_v1.schema.json",
        "stage_ii_cfm_training_view_v2.schema.json",
        "stage_ii_projection_drop_v1.schema.json",
        "stage_ii_projection_drop_v2.schema.json",
        "contract_digest.json",
    ):
        source = route_root / "schemas/stage_ii_cfm" / filename
        target = ms_root / "swift/sidewalk_nav/schemas/stage_ii_cfm" / filename
        content = source.read_text(encoding="utf-8")
        artifact_hashes[filename] = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if not target.is_file() or target.read_text(encoding="utf-8") != content:
            stale.append(str(target))
            if not check:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                written.append(str(target))
    if check and stale:
        raise ValueError("stale Stage II generated mirrors: " + ", ".join(stale))
    return {
        "schema_version": "urbannav_stage_ii_cfm_cross_repo_sync_v1",
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
    report = sync_stage_ii_cfm_cross_repo(args.ms_swift_root, check=args.check)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
