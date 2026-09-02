from __future__ import annotations

from pathlib import Path

import pytest

from tools.model_processing.export_stage_i_sft_contract_artifacts import (
    ARTIFACT_FILENAMES,
    export_stage_i_sft_contract_artifacts,
)
from tools.model_processing.sync_stage_i_sft_cross_repo import (
    sync_stage_i_sft_cross_repo,
)


ROUTE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parents[4]
ROUTE_ARTIFACTS = ROUTE_ROOT / "schemas" / "stage_i_sft"
CONTRACT_ROOT = CODE_ROOT / "UrbanNav-Contracts"
CONTRACT_ARTIFACTS = (
    CONTRACT_ROOT / "src" / "urbannav_contracts" / "schemas" / "stage_i_sft"
)


def test_committed_stage_i_contract_artifacts_are_current(tmp_path: Path) -> None:
    generated = tmp_path / "stage_i_sft"
    export_stage_i_sft_contract_artifacts(generated)
    for filename in ARTIFACT_FILENAMES:
        assert (ROUTE_ARTIFACTS / filename).read_bytes() == (generated / filename).read_bytes()
    export_stage_i_sft_contract_artifacts(ROUTE_ARTIFACTS, check=True)


def test_stage_i_contract_bundle_is_current() -> None:
    if not CONTRACT_ROOT.is_dir():
        pytest.skip("sibling UrbanNav-Contracts checkout is unavailable")
    for filename in ARTIFACT_FILENAMES:
        assert (CONTRACT_ARTIFACTS / filename).read_bytes() == (
            ROUTE_ARTIFACTS / filename
        ).read_bytes()
    report = sync_stage_i_sft_cross_repo(CONTRACT_ROOT, check=True)
    assert report["passed"] is True
    assert report["stale"] == []
