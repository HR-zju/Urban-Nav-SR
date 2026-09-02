from __future__ import annotations

from pathlib import Path

from tools.model_processing.export_stage_ii_cfm_contract_artifacts import (
    ARTIFACT_FILENAMES,
    export_stage_ii_cfm_contract_artifacts,
)
from tools.model_processing.sync_stage_ii_cfm_cross_repo import (
    sync_stage_ii_cfm_cross_repo,
)


CODE_ROOT = Path(__file__).resolve().parents[4]
CONTRACT_ROOT = CODE_ROOT / "UrbanNav-Contracts"
CONTRACT_ARTIFACTS = (
    CONTRACT_ROOT
    / "src"
    / "urbannav_contracts"
    / "schemas"
    / "stage_ii_cfm"
)


def test_stage_ii_contract_export_is_current(tmp_path: Path) -> None:
    generated = tmp_path / "stage_ii_cfm"
    export_stage_ii_cfm_contract_artifacts(generated)
    for filename in ARTIFACT_FILENAMES:
        assert (CONTRACT_ARTIFACTS / filename).read_bytes() == (generated / filename).read_bytes()
    export_stage_ii_cfm_contract_artifacts(CONTRACT_ARTIFACTS, check=True)


def test_stage_ii_contract_bundle_is_current() -> None:
    assert CONTRACT_ROOT.is_dir()
    for filename in ARTIFACT_FILENAMES:
        assert (CONTRACT_ARTIFACTS / filename).is_file()
    report = sync_stage_ii_cfm_cross_repo(CONTRACT_ROOT, check=True)
    assert report["passed"] is True
    assert report["stale"] == []
