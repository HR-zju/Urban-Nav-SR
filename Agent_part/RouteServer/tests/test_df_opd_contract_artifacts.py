from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from mcp_server.contracts.df_opd import DF_OPD_PROFILES_V1
from tools.model_processing.export_df_opd_contract_artifacts import (
    ARTIFACT_FILENAMES,
    export_df_opd_contract_artifacts,
)
from tools.model_processing.sync_df_opd_contract_cross_repo import (
    sync_df_opd_contract_cross_repo,
)


ROUTE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parents[4]
ROUTE_ARTIFACTS = ROUTE_ROOT / 'schemas' / 'df_opd'
MS_ROOT = CODE_ROOT / 'ms-swift'
MS_ARTIFACTS = MS_ROOT / 'swift' / 'sidewalk_nav' / 'schemas' / 'df_opd'


def test_committed_df_opd_contract_artifacts_are_current(tmp_path: Path) -> None:
    """Keep RouteServer's committed DF-OPD schemas deterministic."""

    generated = tmp_path / 'df_opd'
    export_df_opd_contract_artifacts(generated)
    for filename in ARTIFACT_FILENAMES:
        assert (ROUTE_ARTIFACTS / filename).read_bytes() == (
            generated / filename
        ).read_bytes()
    export_df_opd_contract_artifacts(ROUTE_ARTIFACTS, check=True)


def test_df_opd_ms_swift_mirror_is_current_and_profile_equivalent() -> None:
    """Require generated source/schema parity and preserve frozen profiles."""

    if not MS_ROOT.is_dir():
        pytest.skip('sibling ms-swift checkout is unavailable')
    for filename in ARTIFACT_FILENAMES:
        assert (MS_ARTIFACTS / filename).read_bytes() == (
            ROUTE_ARTIFACTS / filename
        ).read_bytes()
    report = sync_df_opd_contract_cross_repo(MS_ROOT, check=True)
    assert report['passed'] is True
    assert report['stale'] == []

    sys.path.insert(0, str(MS_ROOT))
    try:
        module = importlib.import_module(
            'swift.sidewalk_nav._generated_df_opd_contract'
        )
        assert {
            profile_id: profile.model_dump(mode='json')
            for profile_id, profile in module.DF_OPD_PROFILES_V1.items()
        } == {
            profile_id: profile.model_dump(mode='json')
            for profile_id, profile in DF_OPD_PROFILES_V1.items()
        }
        digest_artifact = json.loads(
            (MS_ARTIFACTS / 'contract_digest.json').read_text(encoding='utf-8')
        )
        assert digest_artifact['profiles'] == {
            profile_id: {
                'profile': profile.model_dump(mode='json'),
                'profile_digest': profile.profile_digest,
            }
            for profile_id, profile in sorted(DF_OPD_PROFILES_V1.items())
        }
    finally:
        sys.path.remove(str(MS_ROOT))
