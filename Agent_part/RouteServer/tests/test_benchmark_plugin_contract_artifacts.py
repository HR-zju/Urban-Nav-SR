"""Generated artifact and cross-repository parity tests for benchmark-plugin V1."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.model_processing.export_benchmark_plugin_contract_artifacts import (
    ARTIFACT_FILENAMES,
    export_benchmark_plugin_contract_artifacts_v1,
)
from tools.model_processing.sync_benchmark_plugin_cross_repo import sync_benchmark_plugin_cross_repo_v1


ROUTE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parents[4]
ROUTE_ARTIFACTS = ROUTE_ROOT / 'schemas' / 'benchmark_plugin'
MS_ROOT = CODE_ROOT / 'ms-swift'
MS_ARTIFACTS = MS_ROOT / 'swift' / 'sidewalk_nav' / 'schemas' / 'benchmark_plugin'


def test_committed_benchmark_plugin_contract_artifacts_are_current(tmp_path: Path) -> None:
    """Canonical JSON schemas must equal a fresh deterministic export."""

    generated = tmp_path / 'benchmark_plugin'
    export_benchmark_plugin_contract_artifacts_v1(generated)
    for filename in ARTIFACT_FILENAMES:
        assert (ROUTE_ARTIFACTS / filename).read_bytes() == (generated / filename).read_bytes()
    export_benchmark_plugin_contract_artifacts_v1(ROUTE_ARTIFACTS, check=True)


def test_benchmark_plugin_ms_swift_mirror_is_current() -> None:
    """The model-side generated contract mirror must match RouteServer bytes."""

    if not MS_ROOT.is_dir():
        pytest.skip('sibling ms-swift checkout is unavailable')
    for filename in ARTIFACT_FILENAMES:
        assert (MS_ARTIFACTS / filename).read_bytes() == (ROUTE_ARTIFACTS / filename).read_bytes()
    report = sync_benchmark_plugin_cross_repo_v1(MS_ROOT, check=True)
    assert report['passed'] is True
    assert report['stale'] == []
