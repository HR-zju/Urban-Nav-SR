#!/usr/bin/env python3
"""Export the standalone Python and JSON DF-OPD contract bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

if __package__:
    from tools.model_processing.export_df_opd_contract_artifacts import (
        export_df_opd_contract_artifacts,
    )
else:  # Direct ``python tools/.../script.py`` execution.
    from export_df_opd_contract_artifacts import (  # type: ignore[no-redef]
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
        Generated standalone Python source with its ownership header.

    Raises:
        ValueError: If the source has unexpectedly gained RouteServer imports.
    """

    import_lines = (
        line.strip()
        for line in source_text.splitlines()
        if line.strip().startswith(('from ', 'import '))
    )
    if any(
        line.startswith(('from mcp_server', 'import mcp_server'))
        for line in import_lines
    ):
        raise ValueError('DF-OPD canonical contract must remain package-independent')
    return _GENERATED_HEADER + source_text


def sync_df_opd_contract_cross_repo(
    output_dir: str | Path,
    *,
    check: bool = False,
) -> dict[str, object]:
    """Write or verify the independent DF-OPD contract bundle.

    Args:
        output_dir: Root of the ``UrbanNav-Contracts`` checkout.
        check: Verify freshness without modifying the target when true.

    Returns:
        Bundle paths, deterministic SHA-256 values, and freshness status.

    Raises:
        ValueError: If canonical artifacts or bundle files are stale.
    """

    route_root = Path(__file__).resolve().parents[2]
    output_root = Path(output_dir).expanduser().resolve()
    package_root = output_root / 'src/urbannav_contracts'
    target_artifacts_root = package_root / 'schemas/df_opd'
    artifact_report = export_df_opd_contract_artifacts(
        target_artifacts_root,
        check=check,
    )

    source = route_root / 'src/mcp_server/contracts/df_opd.py'
    destination = package_root / 'df_opd.py'
    rendered = render_df_opd_contract_mirror(source.read_text(encoding='utf-8'))
    stale: list[str] = []
    written: list[str] = []
    if not destination.is_file() or destination.read_text(encoding='utf-8') != rendered:
        stale.append(str(destination))
        if not check:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(rendered, encoding='utf-8')
            written.append(str(destination))

    if check and artifact_report['stale']:
        stale.extend(
            str(target_artifacts_root / filename)
            for filename in artifact_report['stale']
        )
    elif not check:
        written.extend(
            str(target_artifacts_root / filename)
            for filename in artifact_report['written']
        )

    if check and stale:
        raise ValueError('stale DF-OPD generated mirrors: ' + ', '.join(stale))
    return {
        'schema_version': 'urbannav_df_opd_cross_repo_sync_v1',
        'output_dir': str(output_root),
        'checked': bool(check),
        'written': written,
        'stale': stale,
        'python_sha256': hashlib.sha256(rendered.encode('utf-8')).hexdigest(),
        'artifact_sha256': artifact_report['artifact_sha256'],
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
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--check', action='store_true')
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Generate or check mirrors and print a machine-readable report.

    Args:
        argv: Optional command-line tokens excluding the executable name.
    """

    args = parse_args(argv)
    report = sync_df_opd_contract_cross_repo(
        args.output_dir,
        check=args.check,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
