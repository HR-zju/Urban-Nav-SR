#!/usr/bin/env python3
"""Export deterministic UrbanNav Stage II CFM contract artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from mcp_server.contracts.stage_ii_cfm import (
    STAGE_II_CFM_TRAINING_VIEW_SCHEMA_VERSION,
    STAGE_II_CONTEXT_CACHE_SCHEMA_VERSION,
    STAGE_II_PROJECTION_POLICY_ID,
    STAGE_II_PROJECTION_POLICY_VERSION,
    STAGE_II_TRAINING_CONTRACT_ID,
    STAGE_II_TRAJECTORY_POLICY_ID,
    STAGE_II_TRAJECTORY_POLICY_VERSION,
    STAGE_II_TRAJECTORY_QC_SCHEMA_VERSION,
    STAGE_II_TRAJECTORY_SOURCE_REGISTRY_SCHEMA_VERSION,
    STAGE_II_TRAJECTORY_TARGET_SCHEMA_VERSION,
    StageIICFMTrainingViewV1,
    StageIIContextCacheV1,
    StageIIProjectionDropV1,
    StageIITrajectoryQCReportV1,
    StageIITrajectorySourceRegistryV1,
    StageIITrajectoryTargetV1,
)


ARTIFACT_SCHEMA_VERSION = "urbannav_stage_ii_cfm_contract_artifacts_v1"
ARTIFACT_FILENAMES = (
    "stage_ii_context_cache_v1.schema.json",
    "stage_ii_trajectory_target_v1.schema.json",
    "stage_ii_trajectory_qc_v1.schema.json",
    "stage_ii_trajectory_source_registry_v1.schema.json",
    "stage_ii_cfm_training_view_v1.schema.json",
    "stage_ii_projection_drop_v1.schema.json",
    "contract_digest.json",
)


def build_stage_ii_cfm_contract_artifacts() -> dict[str, Any]:
    """Build JSON Schemas and one freshness manifest."""

    artifacts: dict[str, Any] = {
        "stage_ii_context_cache_v1.schema.json": StageIIContextCacheV1.model_json_schema(),
        "stage_ii_trajectory_target_v1.schema.json": StageIITrajectoryTargetV1.model_json_schema(),
        "stage_ii_trajectory_qc_v1.schema.json": StageIITrajectoryQCReportV1.model_json_schema(),
        "stage_ii_trajectory_source_registry_v1.schema.json": (
            StageIITrajectorySourceRegistryV1.model_json_schema()
        ),
        "stage_ii_cfm_training_view_v1.schema.json": StageIICFMTrainingViewV1.model_json_schema(),
        "stage_ii_projection_drop_v1.schema.json": StageIIProjectionDropV1.model_json_schema(),
    }
    artifacts["contract_digest.json"] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "training_contract": STAGE_II_TRAINING_CONTRACT_ID,
        "context_cache_schema_version": STAGE_II_CONTEXT_CACHE_SCHEMA_VERSION,
        "trajectory_target_schema_version": STAGE_II_TRAJECTORY_TARGET_SCHEMA_VERSION,
        "trajectory_qc_schema_version": STAGE_II_TRAJECTORY_QC_SCHEMA_VERSION,
        "trajectory_source_registry_schema_version": (
            STAGE_II_TRAJECTORY_SOURCE_REGISTRY_SCHEMA_VERSION
        ),
        "training_view_schema_version": STAGE_II_CFM_TRAINING_VIEW_SCHEMA_VERSION,
        "projection_policy_id": STAGE_II_PROJECTION_POLICY_ID,
        "projection_policy_version": STAGE_II_PROJECTION_POLICY_VERSION,
        "trajectory_policy_id": STAGE_II_TRAJECTORY_POLICY_ID,
        "trajectory_policy_version": STAGE_II_TRAJECTORY_POLICY_VERSION,
        "canonical_source_sha256": _canonical_source_hashes(),
        "artifact_sha256": {
            filename: _payload_sha256(payload)
            for filename, payload in artifacts.items()
        },
    }
    return artifacts


def export_stage_ii_cfm_contract_artifacts(
    output_dir: str | Path,
    *,
    check: bool = False,
) -> dict[str, Any]:
    """Write or verify deterministic contract artifacts."""

    destination = Path(output_dir).expanduser()
    artifacts = build_stage_ii_cfm_contract_artifacts()
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
        raise ValueError(f"Stage II contract artifacts are stale or missing: {stale}")
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "output_dir": str(destination),
        "checked": check,
        "written": written,
        "stale": stale,
    }


def _canonical_source_hashes() -> dict[str, str]:
    """Hash behavior-bearing canonical Stage II sources."""

    route_root = Path(__file__).resolve().parents[2]
    relatives = (
        "src/mcp_server/contracts/stage_ii_cfm.py",
        "src/mcp_server/data/stage_ii_context.py",
        "src/mcp_server/data/stage_ii_trajectory.py",
        "src/mcp_server/data/stage_ii_sources.py",
        "src/mcp_server/data/stage_ii_views.py",
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
    report = export_stage_ii_cfm_contract_artifacts(args.output_dir, check=args.check)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
