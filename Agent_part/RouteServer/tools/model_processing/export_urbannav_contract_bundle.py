#!/usr/bin/env python3
"""Export the repository-neutral UrbanNav contract bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

if __package__:
    from tools.model_processing.sync_df_opd_contract_cross_repo import (
        sync_df_opd_contract_cross_repo,
    )
    from tools.model_processing.sync_reasoning_v4_cross_repo import (
        sync_reasoning_v4_cross_repo,
    )
    from tools.model_processing.sync_stage_i_sft_cross_repo import (
        sync_stage_i_sft_cross_repo,
    )
    from tools.model_processing.sync_stage_ii_cfm_cross_repo import (
        sync_stage_ii_cfm_cross_repo,
    )
else:  # Direct ``python tools/.../script.py`` execution.
    from sync_df_opd_contract_cross_repo import (  # type: ignore[no-redef]
        sync_df_opd_contract_cross_repo,
    )
    from sync_reasoning_v4_cross_repo import (  # type: ignore[no-redef]
        sync_reasoning_v4_cross_repo,
    )
    from sync_stage_i_sft_cross_repo import (  # type: ignore[no-redef]
        sync_stage_i_sft_cross_repo,
    )
    from sync_stage_ii_cfm_cross_repo import (  # type: ignore[no-redef]
        sync_stage_ii_cfm_cross_repo,
    )


BUNDLE_SCHEMA_VERSION = "urbannav_contract_bundle_v1"
MANIFEST_FILENAME = "contract_bundle_manifest.json"


def export_urbannav_contract_bundle(
    output_dir: str | Path,
    *,
    check: bool = False,
) -> dict[str, Any]:
    """Export or verify the complete standalone contract bundle.

    Args:
        output_dir: Root of the ``UrbanNav-Contracts`` checkout.
        check: Verify generated files and manifest without writing.

    Returns:
        Deterministic component reports and file digests.

    Raises:
        ValueError: If a generated component or manifest is stale in check mode.
    """

    output_root = Path(output_dir).expanduser().resolve()
    component_reports = {
        "df_opd": sync_df_opd_contract_cross_repo(output_root, check=check),
        "reasoning_v4": sync_reasoning_v4_cross_repo(output_root, check=check),
        "stage_i_sft": sync_stage_i_sft_cross_repo(output_root, check=check),
        "stage_ii_cfm": sync_stage_ii_cfm_cross_repo(output_root, check=check),
    }
    manifest = _build_manifest(output_root, component_reports)
    manifest_path = output_root / MANIFEST_FILENAME
    manifest_text = _json_text(manifest)
    current = (
        manifest_path.read_text(encoding="utf-8")
        if manifest_path.is_file()
        else None
    )
    if check:
        if current != manifest_text:
            raise ValueError(f"stale UrbanNav contract bundle: {manifest_path}")
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(manifest_text, encoding="utf-8")
    return manifest


def _build_manifest(
    output_root: Path,
    component_reports: dict[str, dict[str, object]],
) -> dict[str, Any]:
    """Build a stable manifest from generated package files."""

    package_root = output_root / "src/urbannav_contracts"
    files = []
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        files.append({
            "path": path.relative_to(output_root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "components": {
            name: {
                "schema_version": report["schema_version"],
                "python_sha256": report.get("python_sha256"),
                "artifact_sha256": report.get("artifact_sha256", {}),
            }
            for name, report in sorted(component_reports.items())
        },
        "files": files,
    }


def _json_text(payload: Any) -> str:
    """Render deterministic manifest JSON."""

    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse bundle export arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Export or verify the standalone UrbanNav contract bundle."""

    args = parse_args(argv)
    report = export_urbannav_contract_bundle(args.output_dir, check=args.check)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
