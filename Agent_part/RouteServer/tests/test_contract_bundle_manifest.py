from __future__ import annotations

from pathlib import Path

from tools.model_processing.export_urbannav_contract_bundle import _build_manifest


def test_contract_bundle_manifest_excludes_python_cache(tmp_path: Path) -> None:
    package_root = tmp_path / "src" / "urbannav_contracts"
    (package_root / "__pycache__").mkdir(parents=True)
    (package_root / "contract.py").write_text("value = 1\n", encoding="utf-8")
    (package_root / "__pycache__" / "contract.cpython-312.pyc").write_bytes(b"cache")

    manifest = _build_manifest(tmp_path, {})

    assert [item["path"] for item in manifest["files"]] == [
        "src/urbannav_contracts/contract.py"
    ]
