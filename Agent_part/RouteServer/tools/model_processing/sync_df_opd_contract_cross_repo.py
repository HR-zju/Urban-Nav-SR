#!/usr/bin/env python3
"""Generate the ms-swift mirror of the canonical DF-OPD contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

try:
    from tools.model_processing.export_df_opd_contract_artifacts import (
        ARTIFACT_FILENAMES,
        export_df_opd_contract_artifacts,
    )
except ModuleNotFoundError:  # Direct ``python tools/.../script.py`` execution.
    from export_df_opd_contract_artifacts import (  # type: ignore[no-redef]
        ARTIFACT_FILENAMES,
        export_df_opd_contract_artifacts,
    )


_GENERATED_HEADER = (
    '# Generated from RouteServer canonical UrbanNav DF-OPD contract. '
    'Do not edit by hand.\n'
)


def render_df_opd_contract_mirror(source_text: str) -> str:
    """Render a package-independent byte mirror of the canonical contract.

    Args:
        source_text: Contents of RouteServer's canonical contract module.

    Returns:
        Generated ms-swift Python source with its ownership header.

    Raises:
        ValueError: If the source has unexpectedly gained RouteServer imports.
    """

    if 'from mcp_server' in source_text or 'import mcp_server' in source_text:
        raise ValueError('DF-OPD canonical contract must remain package-independent')
    return _GENERATED_HEADER + source_text


def sync_df_opd_contract_cross_repo(
    ms_swift_root: str | Path,
    *,
    check: bool = False,
) -> dict[str, object]:
    """Write or verify generated DF-OPD Python and JSON mirrors.

    Args:
        ms_swift_root: Sibling ms-swift checkout root.
        check: Verify freshness without modifying the target when true.

    Returns:
        Mirror paths, deterministic SHA-256 values, and freshness status.

    Raises:
        ValueError: If canonical artifacts or ms-swift mirrors are stale.
    """

    route_root = Path(__file__).resolve().parents[2]
    ms_root = Path(ms_swift_root).expanduser().resolve()
    artifacts_root = route_root / 'schemas/df_opd'
    export_df_opd_contract_artifacts(artifacts_root, check=True)

    source = route_root / 'src/mcp_server/contracts/df_opd.py'
    destination = ms_root / 'swift/sidewalk_nav/_generated_df_opd_contract.py'
    rendered = render_df_opd_contract_mirror(source.read_text(encoding='utf-8'))
    stale: list[str] = []
    written: list[str] = []
    if not destination.is_file() or destination.read_text(encoding='utf-8') != rendered:
        stale.append(str(destination))
        if not check:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(rendered, encoding='utf-8')
            written.append(str(destination))

    artifact_hashes: dict[str, str] = {}
    for filename in ARTIFACT_FILENAMES:
        source_path = artifacts_root / filename
        target = ms_root / 'swift/sidewalk_nav/schemas/df_opd' / filename
        content = source_path.read_text(encoding='utf-8')
        artifact_hashes[filename] = hashlib.sha256(
            content.encode('utf-8')
        ).hexdigest()
        if not target.is_file() or target.read_text(encoding='utf-8') != content:
            stale.append(str(target))
            if not check:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding='utf-8')
                written.append(str(target))
    if check and stale:
        raise ValueError('stale DF-OPD generated mirrors: ' + ', '.join(stale))
    return {
        'schema_version': 'urbannav_df_opd_cross_repo_sync_v1',
        'ms_swift_root': str(ms_root),
        'checked': bool(check),
        'written': written,
        'stale': stale,
        'python_sha256': hashlib.sha256(rendered.encode('utf-8')).hexdigest(),
        'artifact_sha256': artifact_hashes,
        'passed': not stale if check else True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse cross-repository synchronization arguments.

    Args:
        argv: Optional command-line tokens excluding the executable name.

    Returns:
        Parsed sibling checkout and verification arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ms-swift-root', required=True, type=Path)
    parser.add_argument('--check', action='store_true')
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Generate or check mirrors and print a machine-readable report.

    Args:
        argv: Optional command-line tokens excluding the executable name.
    """

    args = parse_args(argv)
    report = sync_df_opd_contract_cross_repo(
        args.ms_swift_root,
        check=args.check,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
