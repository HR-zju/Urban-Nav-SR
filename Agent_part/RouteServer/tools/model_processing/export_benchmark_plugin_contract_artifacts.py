#!/usr/bin/env python3
"""Export deterministic schemas for the pluggable benchmark contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from mcp_server.contracts.benchmark_plugin import (
    BENCHMARK_PLUGIN_CONTRACT_DIGEST,
    BENCHMARK_PLUGIN_SCHEMA_VERSION,
    BenchmarkFailureV1,
    BenchmarkInputV1,
    BenchmarkMediaManifestEntryV1,
    BenchmarkModelCapabilitiesV1,
    BenchmarkPackageManifestV1,
    BenchmarkPredictionV1,
    BenchmarkProfileV1,
    BenchmarkReferenceV1,
)
from mcp_server.contracts.citywalker_to_urbannav_adapter import (
    CITYWALKER_TO_URBANNAV_ADAPTER_CONTRACT_VERSION_V1,
    CityWalkerToUrbanNavAdapterTraceV1,
)

ARTIFACT_SCHEMA_VERSION = "urbannav_benchmark_plugin_contract_artifacts_v1"
_MODELS = {
    "benchmark_input_v1.schema.json":
    BenchmarkInputV1,
    "benchmark_reference_v1.schema.json":
    BenchmarkReferenceV1,
    "benchmark_prediction_v1.schema.json":
    BenchmarkPredictionV1,
    "benchmark_failure_v1.schema.json":
    BenchmarkFailureV1,
    "benchmark_media_manifest_entry_v1.schema.json":
    BenchmarkMediaManifestEntryV1,
    "benchmark_profile_v1.schema.json":
    BenchmarkProfileV1,
    "benchmark_package_v1.schema.json":
    BenchmarkPackageManifestV1,
    "benchmark_capabilities_v1.schema.json":
    BenchmarkModelCapabilitiesV1,
    "citywalker_to_urbannav_adapter_trace_v1.schema.json":
    CityWalkerToUrbanNavAdapterTraceV1,
}
ARTIFACT_FILENAMES = (*_MODELS, "contract_digest.json")


def build_benchmark_plugin_contract_artifacts_v1() -> dict[str, Any]:
    """Build all strict JSON schemas and one freshness manifest.

    Returns:
        Filename to canonical JSON-compatible artifact mapping.
    """

    artifacts = {
        filename: model.model_json_schema()
        for filename, model in _MODELS.items()
    }
    contract_root = Path(
        __file__).resolve().parents[2] / "src/mcp_server/contracts"
    source_path = contract_root / "benchmark_plugin.py"
    native_adapter_source_path = contract_root / "citywalker_to_urbannav_adapter.py"
    artifacts["contract_digest.json"] = {
        "schema_version":
        ARTIFACT_SCHEMA_VERSION,
        "benchmark_plugin_schema_version":
        BENCHMARK_PLUGIN_SCHEMA_VERSION,
        "semantic_contract_digest":
        BENCHMARK_PLUGIN_CONTRACT_DIGEST,
        "canonical_source_sha256":
        hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "citywalker_to_urbannav_adapter_contract_version":
        CITYWALKER_TO_URBANNAV_ADAPTER_CONTRACT_VERSION_V1,
        "citywalker_to_urbannav_adapter_source_sha256":
        hashlib.sha256(native_adapter_source_path.read_bytes()).hexdigest(),
        "artifact_sha256": {
            filename: _payload_sha256(value)
            for filename, value in artifacts.items()
        },
    }
    return artifacts


def export_benchmark_plugin_contract_artifacts_v1(output_dir: str | Path,
                                                  *,
                                                  check: bool = False
                                                  ) -> dict[str, Any]:
    """Write or verify canonical schema artifacts.

    Args:
        output_dir: Contract artifact directory.
        check: Verify committed bytes without writing when true.

    Returns:
        Written/stale artifact report.

    Raises:
        ValueError: If check mode finds stale artifacts.
    """

    destination = Path(output_dir).expanduser()
    artifacts = build_benchmark_plugin_contract_artifacts_v1()
    stale: list[str] = []
    written: list[str] = []
    for filename in ARTIFACT_FILENAMES:
        rendered = _json_text(artifacts[filename])
        path = destination / filename
        if check:
            if not path.is_file() or path.read_text(
                    encoding="utf-8") != rendered:
                stale.append(filename)
        else:
            destination.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered, encoding="utf-8")
            written.append(filename)
    if stale:
        raise ValueError(
            f"benchmark plugin contract artifacts are stale: {stale}")
    return {"written": written, "stale": stale, "checked": check}


def _json_text(value: Any) -> str:
    """Render deterministic UTF-8 JSON with a terminating newline."""

    return json.dumps(value, ensure_ascii=False, indent=2,
                      sort_keys=True) + "\n"


def _payload_sha256(value: Any) -> str:
    """Hash one canonical JSON-compatible schema payload."""

    text = json.dumps(value,
                      ensure_ascii=False,
                      sort_keys=True,
                      separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI arguments and export/check one artifact directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    report = export_benchmark_plugin_contract_artifacts_v1(
        args.output_dir,
        check=args.check,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
