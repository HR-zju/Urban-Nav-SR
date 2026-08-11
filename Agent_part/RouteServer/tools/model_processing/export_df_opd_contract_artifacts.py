#!/usr/bin/env python3
"""Export deterministic DF-OPD canonical schemas and profile bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from mcp_server.contracts.df_opd import (
    DFOPDCanonicalSampleV1,
    DFOPDProfileV1,
    DF_OPD_CANONICAL_SAMPLE_SCHEMA_VERSION,
    DF_OPD_PROFILE_SCHEMA_VERSION,
    DF_OPD_PROFILES_V1,
)


ARTIFACT_SCHEMA_VERSION = 'urbannav_df_opd_contract_artifacts_v1'
ARTIFACT_FILENAMES = (
    'df_opd_profile_v1.schema.json',
    'df_opd_canonical_sample_v1.schema.json',
    'contract_digest.json',
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse DF-OPD artifact export arguments.

    Args:
        argv: Optional command-line tokens excluding the executable name.

    Returns:
        Parsed output/check arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument(
        '--check',
        action='store_true',
        help='Fail if committed files differ instead of replacing them.',
    )
    return parser.parse_args(argv)


def build_df_opd_contract_artifacts() -> dict[str, Any]:
    """Build canonical DF-OPD schemas and profile/digest bindings in memory.

    Returns:
        Mapping from stable artifact filename to JSON-compatible payload.
    """

    artifacts: dict[str, Any] = {
        'df_opd_profile_v1.schema.json': DFOPDProfileV1.model_json_schema(),
        'df_opd_canonical_sample_v1.schema.json': (
            DFOPDCanonicalSampleV1.model_json_schema()
        ),
    }
    artifacts['contract_digest.json'] = {
        'schema_version': ARTIFACT_SCHEMA_VERSION,
        'canonical_sample_schema_version': (
            DF_OPD_CANONICAL_SAMPLE_SCHEMA_VERSION
        ),
        'profile_schema_version': DF_OPD_PROFILE_SCHEMA_VERSION,
        'canonical_source_sha256': _canonical_source_hash(),
        'profiles': {
            profile_id: {
                'profile': profile.model_dump(mode='json'),
                'profile_digest': profile.profile_digest,
            }
            for profile_id, profile in sorted(DF_OPD_PROFILES_V1.items())
        },
        'artifact_sha256': {
            filename: _payload_sha256(payload)
            for filename, payload in artifacts.items()
        },
    }
    return artifacts


def export_df_opd_contract_artifacts(
    output_dir: str | Path,
    *,
    check: bool = False,
) -> dict[str, Any]:
    """Write or verify deterministic DF-OPD contract artifacts.

    Args:
        output_dir: Directory containing the stable artifact filenames.
        check: Compare against committed files without modifying them.

    Returns:
        Machine-readable file digests and check status.

    Raises:
        ValueError: If ``check`` finds a missing or stale artifact.
    """

    destination = Path(output_dir).expanduser()
    artifacts = build_df_opd_contract_artifacts()
    stale: list[str] = []
    written: list[str] = []
    for filename in ARTIFACT_FILENAMES:
        text = _json_text(artifacts[filename])
        path = destination / filename
        if check:
            if not path.is_file() or path.read_text(encoding='utf-8') != text:
                stale.append(filename)
            continue
        destination.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        written.append(filename)
    if stale:
        raise ValueError(
            'DF-OPD contract artifacts are stale or missing: '
            f'{stale}'
        )
    return {
        'schema_version': ARTIFACT_SCHEMA_VERSION,
        'output_dir': str(destination),
        'checked': bool(check),
        'written': written,
        'stale': stale,
        'artifact_sha256': {
            filename: _payload_sha256(artifacts[filename])
            for filename in ARTIFACT_FILENAMES
        },
    }


def _canonical_source_hash() -> str:
    """Return the SHA-256 for the canonical DF-OPD semantic source."""

    route_root = Path(__file__).resolve().parents[2]
    source = route_root / 'src/mcp_server/contracts/df_opd.py'
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _json_text(payload: Any) -> str:
    """Render one artifact deterministically with a trailing newline."""

    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n'


def _payload_sha256(payload: Any) -> str:
    """Return the stable SHA-256 digest for one JSON-compatible payload."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    """Export or verify artifacts and print a machine-readable report.

    Args:
        argv: Optional command-line tokens excluding the executable name.
    """

    args = parse_args(argv)
    report = export_df_opd_contract_artifacts(
        args.output_dir,
        check=args.check,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
