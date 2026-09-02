#!/usr/bin/env python3
"""Export the standalone Python and JSON Reasoning V4 contract bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

if __package__:
    from tools.model_processing.export_reasoning_v4_contract_artifacts import (
        export_reasoning_v4_contract_artifacts,
    )
else:  # Direct ``python tools/.../script.py`` execution.
    from export_reasoning_v4_contract_artifacts import (  # type: ignore[no-redef]
        export_reasoning_v4_contract_artifacts,
    )


_GENERATED_HEADER = (
    "# Generated from RouteServer canonical Reasoning V4. "
    "Do not edit by hand.\n"
)
_DIRECTION_GENERATED_HEADER = (
    "# Generated from RouteServer canonical Stage-A direction profiles. "
    "Do not edit by hand.\n"
)


def render_direction_profile_mirror(source_text: str) -> str:
    """Rewrite the ReasonCode import for the standalone package."""

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
    """Rewrite the ReasonCode import for the standalone package."""

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
    """Rewrite dataset-contract imports for the standalone package."""

    replacements = {
        "from mcp_server.contracts.direction_profile import (": (
            "from .direction_profile import ("
        ),
        "from mcp_server.contracts.reasoning import ReasonCode": (
            "from .reasoning import ReasonCode"
        ),
        "from mcp_server.contracts.reasoning_v4 import (": (
            "from .reasoning_v4 import ("
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
    """Rewrite canonical prompt-policy imports for the standalone package."""

    replacements = {
        "from mcp_server.contracts.direction_profile import (": (
            "from .direction_profile import ("
        ),
        "from mcp_server.contracts.reasoning_v4 import (": (
            "from .reasoning_v4 import ("
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
    """Rewrite compact-projector imports for the standalone package."""

    replacements = {
        "from mcp_server.agent.reasoning_v4_text import (": (
            "from .reasoning_v4_text import ("
        ),
        "from mcp_server.contracts.direction_profile import (": (
            "from .direction_profile import ("
        ),
        "from mcp_server.contracts.reasoning_v4 import (": (
            "from .reasoning_v4 import ("
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


def render_reasoning_v4_text_mirror(source_text: str) -> str:
    """Rewrite the Reasoning V4 text codec import for the package."""

    expected = "from mcp_server.contracts.reasoning_v4 import ("
    if source_text.count(expected) != 1:
        raise ValueError("canonical Reasoning V4 text import is ambiguous")
    return _GENERATED_HEADER + source_text.replace(
        expected,
        "from .reasoning_v4 import (",
    )


def render_vqa_v2_contract_mirror(source_text: str) -> str:
    """Rewrite the VQA contract's shared ReasonCode import."""

    expected = "from mcp_server.contracts.reasoning import ReasonCode"
    if source_text.count(expected) != 1:
        raise ValueError("canonical VQA ReasonCode import is ambiguous")
    return _GENERATED_HEADER + source_text.replace(
        expected,
        "from .reasoning import ReasonCode",
    )


def sync_reasoning_v4_cross_repo(
    output_dir: str | Path,
    *,
    check: bool = False,
) -> dict[str, object]:
    """Write or verify the independent Reasoning V4 contract bundle."""

    route_root = Path(__file__).resolve().parents[2]
    output_root = Path(output_dir).expanduser().resolve()
    package_root = output_root / "src/urbannav_contracts"
    pairs: tuple[
        tuple[Path, Path, Callable[[str], str]],
        ...,
    ] = (
        (
            route_root / "src/mcp_server/contracts/reasoning.py",
            package_root / "reasoning.py",
            lambda source_text: _GENERATED_HEADER + source_text,
        ),
        (
            route_root
            / "src/mcp_server/contracts/direction_profile.py",
            package_root / "direction_profile.py",
            render_direction_profile_mirror,
        ),
        (
            route_root
            / "src/mcp_server/contracts/reasoning_v4.py",
            package_root / "reasoning_v4.py",
            render_reasoning_v4_contract_mirror,
        ),
        (
            route_root
            / "src/mcp_server/contracts/reasoning_dataset_v4.py",
            package_root / "reasoning_dataset_v4.py",
            render_reasoning_v4_dataset_contract_mirror,
        ),
        (
            route_root
            / "src/mcp_server/agent/reasoning_v4_prompt_policy.py",
            package_root / "reasoning_v4_prompt_policy.py",
            render_reasoning_v4_prompt_policy_mirror,
        ),
        (
            route_root
            / "src/mcp_server/data/canonical/reasoning_v4_compact.py",
            package_root / "reasoning_v4_compact.py",
            render_reasoning_v4_compact_mirror,
        ),
        (
            route_root / "src/mcp_server/agent/reasoning_v4_text.py",
            package_root / "reasoning_v4_text.py",
            render_reasoning_v4_text_mirror,
        ),
        (
            route_root / "src/mcp_server/contracts/vqa_v2.py",
            package_root / "vqa_v2.py",
            render_vqa_v2_contract_mirror,
        ),
    )
    written: list[str] = []
    stale: list[str] = []
    hashes: dict[str, str] = {}
    for source_path, destination, renderer in pairs:
        rendered = renderer(source_path.read_text(encoding="utf-8"))
        import_lines = (
            line.strip()
            for line in rendered.splitlines()
            if line.strip().startswith(("from ", "import "))
        )
        if any(
            line.startswith(("from mcp_server", "import mcp_server"))
            for line in import_lines
        ):
            raise ValueError(
                f"standalone contract still imports RouteServer: {source_path}"
            )
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
    artifact_report = export_reasoning_v4_contract_artifacts(
        package_root / "schemas/reasoning_v4",
        check=check,
    )
    if check:
        stale.extend(
            str(package_root / "schemas/reasoning_v4" / filename)
            for filename in artifact_report["stale"]
        )
    else:
        written.extend(
            str(package_root / "schemas/reasoning_v4" / filename)
            for filename in artifact_report["written"]
        )
    report: dict[str, object] = {
        "schema_version": "reasoning_v4_cross_repo_python_sync_v1",
        "output_dir": str(output_root),
        "checked": bool(check),
        "written": written,
        "stale": stale,
        "sha256": hashes,
        "python_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "artifact_sha256": artifact_report["artifact_sha256"],
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
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Generate or verify mirrors and print the sync report."""

    args = parse_args(argv)
    report = sync_reasoning_v4_cross_repo(
        args.output_dir,
        check=args.check,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
