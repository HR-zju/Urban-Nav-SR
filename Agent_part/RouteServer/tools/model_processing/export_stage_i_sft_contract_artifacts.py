#!/usr/bin/env python3
"""Export deterministic UrbanNav Stage I SFT contract artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from mcp_server.contracts.stage_i_sft import (
    STAGE_I_LOSS_SPAN_POLICY_ID,
    STAGE_I_INSTRUCTION_PROVENANCE_SCHEMA_VERSION,
    STAGE_I_PROJECTION_POLICY_ID,
    STAGE_I_PROJECTION_POLICY_VERSION,
    STAGE_I_TRAINING_VIEW_ADAPTER_V1,
    STAGE_I_TRAINING_VIEW_SCHEMA_VERSION,
    STAGE_I_TRAINING_VIEW_ADAPTER_V2,
    STAGE_I_TRAINING_VIEW_V2_SCHEMA_VERSION,
    StageIProjectionDropV1,
    StageIInstructionProvenanceV1,
)


ARTIFACT_SCHEMA_VERSION = "urbannav_stage_i_contract_artifacts_v1"
ARTIFACT_FILENAMES = (
    "stage_i_training_view_v1.schema.json",
    "stage_i_training_view_v2.schema.json",
    "stage_i_projection_drop_v1.schema.json",
    "stage_i_instruction_provenance_v1.schema.json",
    "contract_digest.json",
)


def build_stage_i_sft_contract_artifacts() -> dict[str, Any]:
    """Build the Stage I JSON Schema and freshness manifest.

    Returns:
        Mapping from stable artifact filename to JSON-compatible payload.
    """

    artifacts: dict[str, Any] = {
        "stage_i_training_view_v1.schema.json": (
            STAGE_I_TRAINING_VIEW_ADAPTER_V1.json_schema()
        ),
        "stage_i_training_view_v2.schema.json": (
            STAGE_I_TRAINING_VIEW_ADAPTER_V2.json_schema()
        ),
        "stage_i_projection_drop_v1.schema.json": (
            StageIProjectionDropV1.model_json_schema()
        ),
        "stage_i_instruction_provenance_v1.schema.json": (
            StageIInstructionProvenanceV1.model_json_schema()
        ),
    }
    artifacts["contract_digest.json"] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "training_view_schema_version": STAGE_I_TRAINING_VIEW_SCHEMA_VERSION,
        "training_view_v2_schema_version": STAGE_I_TRAINING_VIEW_V2_SCHEMA_VERSION,
        "projection_policy_id": STAGE_I_PROJECTION_POLICY_ID,
        "projection_policy_version": STAGE_I_PROJECTION_POLICY_VERSION,
        "loss_span_policy_id": STAGE_I_LOSS_SPAN_POLICY_ID,
        "instruction_provenance_schema_version": (
            STAGE_I_INSTRUCTION_PROVENANCE_SCHEMA_VERSION
        ),
        "canonical_source_sha256": _canonical_source_hashes(),
        "artifact_sha256": {
            filename: _payload_sha256(payload)
            for filename, payload in artifacts.items()
        },
    }
    return artifacts


def export_stage_i_sft_contract_artifacts(
    output_dir: str | Path,
    *,
    check: bool = False,
) -> dict[str, Any]:
    """Write or verify deterministic Stage I contract artifacts.

    Args:
        output_dir: Artifact destination directory.
        check: Compare without writing when true.

    Returns:
        Machine-readable write/check report.

    Raises:
        ValueError: If committed artifacts are missing or stale in check mode.
    """

    destination = Path(output_dir).expanduser()
    artifacts = build_stage_i_sft_contract_artifacts()
    stale: list[str] = []
    written: list[str] = []
    for filename in ARTIFACT_FILENAMES:
        rendered = _json_text(artifacts[filename])
        path = destination / filename
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != rendered:
                stale.append(filename)
            continue
        destination.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        written.append(filename)
    if stale:
        raise ValueError(f"Stage I contract artifacts are stale or missing: {stale}")
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "output_dir": str(destination),
        "checked": check,
        "written": written,
        "stale": stale,
    }


def _canonical_source_hashes() -> dict[str, str]:
    """Hash behavior-bearing Stage I canonical sources."""

    route_root = Path(__file__).resolve().parents[2]
    relatives = (
        "src/mcp_server/contracts/stage_i_sft.py",
        "src/mcp_server/data/stage_i_views.py",
    )
    return {
        relative: hashlib.sha256((route_root / relative).read_bytes()).hexdigest()
        for relative in relatives
    }


def _json_text(payload: Any) -> str:
    """Render stable repository JSON."""

    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _payload_sha256(payload: Any) -> str:
    """Hash one canonical JSON payload."""

    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Export or check the requested artifact directory."""

    args = parse_args(argv)
    report = export_stage_i_sft_contract_artifacts(args.output_dir, check=args.check)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
