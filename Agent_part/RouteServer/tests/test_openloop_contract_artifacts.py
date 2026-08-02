from __future__ import annotations

from pathlib import Path

import pytest

from tools.model_processing.export_openloop_benchmark_contract_artifacts import (
    ARTIFACT_FILENAMES,
    export_openloop_contract_artifacts,
)
from tools.model_processing.sync_openloop_benchmark_cross_repo import (
    sync_openloop_contract_cross_repo,
)


ROUTE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parents[4]
ROUTE_ARTIFACTS = ROUTE_ROOT / "schemas" / "openloop_benchmark"
MS_ROOT = CODE_ROOT / "ms-swift"
MS_ARTIFACTS = MS_ROOT / "swift/sidewalk_nav/schemas/openloop_benchmark"


def test_committed_openloop_contract_artifacts_are_current(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "openloop_benchmark"
    export_openloop_contract_artifacts(generated)
    for filename in ARTIFACT_FILENAMES:
        assert (ROUTE_ARTIFACTS / filename).read_bytes() == (
            generated / filename
        ).read_bytes()
    export_openloop_contract_artifacts(ROUTE_ARTIFACTS, check=True)


def test_openloop_ms_swift_mirror_is_current() -> None:
    if not MS_ROOT.is_dir():
        pytest.skip("sibling ms-swift checkout is unavailable")
    for filename in ARTIFACT_FILENAMES:
        assert (MS_ARTIFACTS / filename).read_bytes() == (
            ROUTE_ARTIFACTS / filename
        ).read_bytes()
    report = sync_openloop_contract_cross_repo(MS_ROOT, check=True)
    assert report["passed"] is True
    assert report["stale"] == []
