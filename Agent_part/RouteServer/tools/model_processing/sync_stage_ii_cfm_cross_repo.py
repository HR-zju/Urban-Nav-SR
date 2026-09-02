#!/usr/bin/env python3
"""Export the standalone Python and JSON Stage II CFM contract bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

if __package__:
    from tools.model_processing.export_stage_ii_cfm_contract_artifacts import (
        ARTIFACT_FILENAMES,
        export_stage_ii_cfm_contract_artifacts,
    )
else:  # Direct ``python tools/.../script.py`` execution.
    from export_stage_ii_cfm_contract_artifacts import (  # type: ignore[no-redef]
        ARTIFACT_FILENAMES,
        export_stage_ii_cfm_contract_artifacts,
    )


_GENERATED_HEADER = (
    "# Generated from RouteServer canonical UrbanNav Stage II CFM contract. "
    "Do not edit by hand.\n"
)


def render_stage_ii_contract_mirror(source_text: str) -> str:
    """Rewrite canonical imports for the standalone contract package."""

    replacements = {
        "from mcp_server.contracts.reasoning import ReasonCode": (
            "from .reasoning import ReasonCode"
        ),
        "from mcp_server.contracts.reasoning_dataset_v4 import (": (
            "from .reasoning_dataset_v4 import ("
        ),
        "from mcp_server.contracts.stage_i_sft import (": (
            "from .stage_i_sft import ("
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
    import_lines = (
        line.strip()
        for line in rendered.splitlines()
        if line.strip().startswith(("from ", "import "))
    )
    if any(
        line.startswith(("from mcp_server", "import mcp_server"))
        for line in import_lines
    ):
        raise ValueError("standalone Stage II contract imports RouteServer")
    return _GENERATED_HEADER + rendered


def sync_stage_ii_cfm_cross_repo(
    output_dir: str | Path,
    *,
    check: bool = False,
) -> dict[str, object]:
    """Write or verify the independent Stage II CFM contract bundle."""

    route_root = Path(__file__).resolve().parents[2]
    output_root = Path(output_dir).expanduser().resolve()
    package_root = output_root / "src/urbannav_contracts"
    source_path = route_root / "src/mcp_server/contracts/stage_ii_cfm.py"
    destination = package_root / "stage_ii_cfm.py"
    rendered = render_stage_ii_contract_mirror(source_path.read_text(encoding="utf-8"))
    written: list[str] = []
    stale: list[str] = []
    if not destination.is_file() or destination.read_text(encoding="utf-8") != rendered:
        stale.append(str(destination))
        if not check:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(rendered, encoding="utf-8")
            written.append(str(destination))

    artifact_report = export_stage_ii_cfm_contract_artifacts(
        package_root / "schemas/stage_ii_cfm",
        check=check,
    )
    if check:
        stale.extend(
            str(package_root / "schemas/stage_ii_cfm" / filename)
            for filename in artifact_report["stale"]
        )
    else:
        written.extend(
            str(package_root / "schemas/stage_ii_cfm" / filename)
            for filename in artifact_report["written"]
        )
    artifact_hashes = {
        filename: hashlib.sha256(
            (package_root / "schemas/stage_ii_cfm" / filename).read_bytes()
        ).hexdigest()
        for filename in ARTIFACT_FILENAMES
    }
    if check and stale:
        raise ValueError("stale Stage II generated mirrors: " + ", ".join(stale))
    return {
        "schema_version": "urbannav_stage_ii_cfm_cross_repo_sync_v1",
        "output_dir": str(output_root),
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
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Generate or check the cross-repository mirrors."""

    args = parse_args(argv)
    report = sync_stage_ii_cfm_cross_repo(args.output_dir, check=args.check)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
