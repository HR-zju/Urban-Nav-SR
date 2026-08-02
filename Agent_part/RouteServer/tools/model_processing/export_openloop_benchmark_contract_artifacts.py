#!/usr/bin/env python3
"""Export deterministic UrbanNav open-loop benchmark schemas."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from mcp_server.contracts.openloop_benchmark import (
    OPENLOOP_BENCHMARK_SCHEMA_DIGEST,
    OPENLOOP_BENCHMARK_SCHEMA_VERSION,
    CheckpointExposureRegistryV1,
    OpenLoopFailureV1,
    OpenLoopInputV1,
    OpenLoopPackageManifestV1,
    OpenLoopPredictionV1,
    OpenLoopReferenceV1,
    OpenLoopRunManifestV1,
    OpenLoopSplitLineageEntryV1,
)


ARTIFACT_SCHEMA_VERSION = "urbannav_openloop_contract_artifacts_v1"
_MODELS = {
    "openloop_input_v2.schema.json": OpenLoopInputV1,
    "openloop_reference_v2.schema.json": OpenLoopReferenceV1,
    "openloop_prediction_v2.schema.json": OpenLoopPredictionV1,
    "openloop_failure_v2.schema.json": OpenLoopFailureV1,
    "openloop_run_v2.schema.json": OpenLoopRunManifestV1,
    "openloop_package_v2.schema.json": OpenLoopPackageManifestV1,
    "checkpoint_exposure_registry_v2.schema.json": (
        CheckpointExposureRegistryV1
    ),
    "openloop_split_lineage_entry_v2.schema.json": (
        OpenLoopSplitLineageEntryV1
    ),
}
ARTIFACT_FILENAMES = (*_MODELS, "contract_digest.json")


def build_openloop_contract_artifacts() -> dict[str, Any]:
    """Build all JSON Schemas and their freshness manifest."""

    artifacts = {
        filename: model.model_json_schema()
        for filename, model in _MODELS.items()
    }
    source_path = (
        Path(__file__).resolve().parents[2]
        / "src/mcp_server/contracts/openloop_benchmark.py"
    )
    artifacts["contract_digest.json"] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "benchmark_schema_version": OPENLOOP_BENCHMARK_SCHEMA_VERSION,
        "semantic_schema_digest": OPENLOOP_BENCHMARK_SCHEMA_DIGEST,
        "canonical_source_sha256": hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
        "artifact_sha256": {
            filename: _payload_sha256(payload)
            for filename, payload in artifacts.items()
        },
    }
    return artifacts


def export_openloop_contract_artifacts(
    output_dir: str | Path,
    *,
    check: bool = False,
) -> dict[str, Any]:
    """Write or verify deterministic open-loop artifacts.

    Args:
        output_dir: Destination schema directory.
        check: Verify committed bytes without writing when true.

    Returns:
        Written/stale artifact report.

    Raises:
        ValueError: If check mode finds stale artifacts.
    """

    destination = Path(output_dir).expanduser()
    artifacts = build_openloop_contract_artifacts()
    stale: list[str] = []
    written: list[str] = []
    for filename in ARTIFACT_FILENAMES:
        rendered = _json_text(artifacts[filename])
        path = destination / filename
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != rendered:
                stale.append(filename)
        else:
            destination.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered, encoding="utf-8")
            written.append(filename)
    if stale:
        raise ValueError(f"open-loop contract artifacts are stale: {stale}")
    return {"written": written, "stale": stale, "checked": check}


def _json_text(value: Any) -> str:
    """Render stable repository JSON."""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _payload_sha256(value: Any) -> str:
    """Hash one canonical JSON payload."""

    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    """Export or check the requested artifact directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            export_openloop_contract_artifacts(
                args.output_dir,
                check=args.check,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
