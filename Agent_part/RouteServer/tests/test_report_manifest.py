from __future__ import annotations

import json

import pytest

from mcp_server.data.report_manifest import (
    REPORT_MANIFEST_SCHEMA_V1,
    build_conventional_report_manifest,
    write_report_manifest,
)


def test_report_manifest_classifies_standard_media_paths(tmp_path):
    (tmp_path / "charts").mkdir()
    (tmp_path / "images" / "success").mkdir(parents=True)
    (tmp_path / "videos" / "failure").mkdir(parents=True)
    (tmp_path / "charts" / "loss.png").write_bytes(b"png")
    (tmp_path / "images" / "success" / "case_001.jpg").write_bytes(b"jpg")
    (tmp_path / "videos" / "failure" / "case_002.mp4").write_bytes(b"mp4")

    manifest = build_conventional_report_manifest(tmp_path)

    assert manifest["schema_version"] == REPORT_MANIFEST_SCHEMA_V1
    assert set(manifest["sections"]) == {"主要指标", "成功案例", "失败案例与人工校阅"}
    output = tmp_path / "report_manifest.json"
    write_report_manifest(output, manifest)
    assert json.loads(output.read_text(encoding="utf-8")) == manifest


def test_report_manifest_requires_explicit_fallback_for_unclassified_media(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "case.jpg").write_bytes(b"jpg")

    with pytest.raises(ValueError, match="Unclassified report media"):
        build_conventional_report_manifest(tmp_path)

    manifest = build_conventional_report_manifest(tmp_path, default_section="成功案例")
    assert manifest["sections"]["成功案例"][0]["path"] == "images/case.jpg"
