#!/usr/bin/env python3
"""Export deterministic UrbanNav Stage II CFM contract artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from mcp_server.contracts.stage_ii_cfm import (
    STAGE_II_ACTION_MASK_SCHEMA_VERSION,
    STAGE_II_CFM_TRAINING_VIEW_V2_SCHEMA_VERSION,
    STAGE_II_CACHED_TRAJ_QUERY_OBSERVATION_SCHEMA_VERSION,
    STAGE_II_CACHED_TRAJ_QUERY_BINDING_V2_SCHEMA_VERSION,
    STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE_V2,
    STAGE_II_CACHED_TRAJ_QUERY_TRAINING_VIEW_SCHEMA_VERSION,
    STAGE_II_CACHED_TRAJ_QUERY_TRAINING_VIEW_V2_SCHEMA_VERSION,
    STAGE_II_CACHED_TRAJ_QUERY_SOURCE_VIEW_V2_SCHEMA_VERSION,
    STAGE_II_CFM_TRAINING_VIEW_SCHEMA_VERSION,
    STAGE_II_CONTEXT_CACHE_SCHEMA_VERSION,
    STAGE_II_DIRECT_LATENT_VIEW_SCHEMA_VERSION,
    STAGE_II_LEGACY_DIRECT_TARGET_SCHEMA_VERSION,
    STAGE_II_PROJECTION_DROP_V2_SCHEMA_VERSION,
    STAGE_II_PROJECTION_POLICY_ID,
    STAGE_II_PROJECTION_POLICY_VERSION,
    STAGE_II_PROJECTION_POLICY_V2_ID,
    STAGE_II_PROJECTION_POLICY_V2_VERSION,
    STAGE_II_REASONING_CONTEXT_ADAPTER_V2,
    STAGE_II_REASONING_CONTEXT_V2_SCHEMA_VERSION,
    STAGE_II_TRAINING_CONTRACT_ID,
    STAGE_II_V2_TRAINING_CONTRACT_ID,
    STAGE_II_TRAJECTORY_POLICY_ID,
    STAGE_II_TRAJECTORY_POLICY_VERSION,
    STAGE_II_TRAJECTORY_POLICY_V2_ID,
    STAGE_II_TRAJECTORY_POLICY_V2_VERSION,
    STAGE_II_TRAJECTORY_QC_SCHEMA_VERSION,
    STAGE_II_TRAJECTORY_QC_V2_SCHEMA_VERSION,
    STAGE_II_TRAJECTORY_SIDECAR_V2_SCHEMA_VERSION,
    STAGE_II_TRAJECTORY_SOURCE_REGISTRY_SCHEMA_VERSION,
    STAGE_II_TRAJECTORY_TARGET_SCHEMA_VERSION,
    STAGE_II_TRAJECTORY_TARGET_V2_SCHEMA_VERSION,
    StageIICFMTrainingViewV1,
    StageIICFMTrainingViewV2,
    StageIICachedTrajQueryObservationV1,
    StageIICachedTrajQueryTrainingViewV1,
    StageIICachedTrajQueryBindingV2,
    StageIICachedTrajQueryTrainingViewV2,
    StageIICachedTrajQuerySourceViewV2,
    StageIICFMDirectLatentViewV1,
    StageIIContextCacheV1,
    StageIILegacyDirectTargetV1,
    StageIIProjectionDropV1,
    StageIIProjectionDropV2,
    StageIITrajectoryQCReportV1,
    StageIITrajectoryQCReportV2,
    StageIITrajectorySidecarV2,
    StageIITrajectorySourceRegistryV1,
    StageIITrajectoryTargetV1,
    StageIITrajectoryTargetV2,
)


ARTIFACT_SCHEMA_VERSION = "urbannav_stage_ii_cfm_contract_artifacts_v4"
ARTIFACT_FILENAMES = (
    "stage_ii_context_cache_v1.schema.json",
    "stage_ii_reasoning_context_v2.schema.json",
    "stage_ii_trajectory_target_v1.schema.json",
    "stage_ii_trajectory_qc_v1.schema.json",
    "stage_ii_trajectory_target_v2.schema.json",
    "stage_ii_trajectory_qc_v2.schema.json",
    "stage_ii_trajectory_sidecar_v2.schema.json",
    "stage_ii_trajectory_source_registry_v1.schema.json",
    "stage_ii_legacy_direct_target_v1.schema.json",
    "stage_ii_cfm_training_view_v1.schema.json",
    "stage_ii_cfm_training_view_v2.schema.json",
    "stage_ii_cached_traj_query_observation_v1.schema.json",
    "stage_ii_cached_traj_query_training_view_v1.schema.json",
    "stage_ii_cached_traj_query_binding_v2.schema.json",
    "stage_ii_cached_traj_query_training_view_v2.schema.json",
    "stage_ii_cached_traj_query_source_view_v2.schema.json",
    "stage_ii_direct_latent_view_v1.schema.json",
    "stage_ii_projection_drop_v1.schema.json",
    "stage_ii_projection_drop_v2.schema.json",
    "contract_digest.json",
)


def build_stage_ii_cfm_contract_artifacts() -> dict[str, Any]:
    """Build JSON Schemas and one freshness manifest."""

    artifacts: dict[str, Any] = {
        "stage_ii_context_cache_v1.schema.json": StageIIContextCacheV1.model_json_schema(),
        "stage_ii_reasoning_context_v2.schema.json": (
            STAGE_II_REASONING_CONTEXT_ADAPTER_V2.json_schema()
        ),
        "stage_ii_trajectory_target_v1.schema.json": StageIITrajectoryTargetV1.model_json_schema(),
        "stage_ii_trajectory_qc_v1.schema.json": StageIITrajectoryQCReportV1.model_json_schema(),
        "stage_ii_trajectory_target_v2.schema.json": StageIITrajectoryTargetV2.model_json_schema(),
        "stage_ii_trajectory_qc_v2.schema.json": StageIITrajectoryQCReportV2.model_json_schema(),
        "stage_ii_trajectory_sidecar_v2.schema.json": StageIITrajectorySidecarV2.model_json_schema(),
        "stage_ii_trajectory_source_registry_v1.schema.json": (
            StageIITrajectorySourceRegistryV1.model_json_schema()
        ),
        "stage_ii_legacy_direct_target_v1.schema.json": (
            StageIILegacyDirectTargetV1.model_json_schema()
        ),
        "stage_ii_cfm_training_view_v1.schema.json": StageIICFMTrainingViewV1.model_json_schema(),
        "stage_ii_cfm_training_view_v2.schema.json": StageIICFMTrainingViewV2.model_json_schema(),
        "stage_ii_cached_traj_query_observation_v1.schema.json": (
            StageIICachedTrajQueryObservationV1.model_json_schema()
        ),
        "stage_ii_cached_traj_query_training_view_v1.schema.json": (
            StageIICachedTrajQueryTrainingViewV1.model_json_schema()
        ),
        "stage_ii_cached_traj_query_binding_v2.schema.json": (
            StageIICachedTrajQueryBindingV2.model_json_schema()
        ),
        "stage_ii_cached_traj_query_training_view_v2.schema.json": (
            StageIICachedTrajQueryTrainingViewV2.model_json_schema()
        ),
        "stage_ii_cached_traj_query_source_view_v2.schema.json": (
            StageIICachedTrajQuerySourceViewV2.model_json_schema()
        ),
        "stage_ii_direct_latent_view_v1.schema.json": (
            StageIICFMDirectLatentViewV1.model_json_schema()
        ),
        "stage_ii_projection_drop_v1.schema.json": StageIIProjectionDropV1.model_json_schema(),
        "stage_ii_projection_drop_v2.schema.json": StageIIProjectionDropV2.model_json_schema(),
    }
    artifacts["contract_digest.json"] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "training_contract": STAGE_II_TRAINING_CONTRACT_ID,
        "training_contract_v2": STAGE_II_V2_TRAINING_CONTRACT_ID,
        "context_cache_schema_version": STAGE_II_CONTEXT_CACHE_SCHEMA_VERSION,
        "reasoning_context_v2_schema_version": (
            STAGE_II_REASONING_CONTEXT_V2_SCHEMA_VERSION
        ),
        "trajectory_target_schema_version": STAGE_II_TRAJECTORY_TARGET_SCHEMA_VERSION,
        "trajectory_qc_schema_version": STAGE_II_TRAJECTORY_QC_SCHEMA_VERSION,
        "trajectory_target_v2_schema_version": STAGE_II_TRAJECTORY_TARGET_V2_SCHEMA_VERSION,
        "action_mask_schema_version": STAGE_II_ACTION_MASK_SCHEMA_VERSION,
        "trajectory_qc_v2_schema_version": STAGE_II_TRAJECTORY_QC_V2_SCHEMA_VERSION,
        "trajectory_sidecar_v2_schema_version": STAGE_II_TRAJECTORY_SIDECAR_V2_SCHEMA_VERSION,
        "trajectory_source_registry_schema_version": (
            STAGE_II_TRAJECTORY_SOURCE_REGISTRY_SCHEMA_VERSION
        ),
        "legacy_direct_target_schema_version": (
            STAGE_II_LEGACY_DIRECT_TARGET_SCHEMA_VERSION
        ),
        "training_view_schema_version": STAGE_II_CFM_TRAINING_VIEW_SCHEMA_VERSION,
        "training_view_v2_schema_version": (
            STAGE_II_CFM_TRAINING_VIEW_V2_SCHEMA_VERSION
        ),
        "cached_traj_query_observation_schema_version": (
            STAGE_II_CACHED_TRAJ_QUERY_OBSERVATION_SCHEMA_VERSION
        ),
        "cached_traj_query_training_view_schema_version": (
            STAGE_II_CACHED_TRAJ_QUERY_TRAINING_VIEW_SCHEMA_VERSION
        ),
        "cached_traj_query_condition_mode_v2": STAGE_II_CACHED_TRAJ_QUERY_CONDITION_MODE_V2,
        "cached_traj_query_binding_v2_schema_version": (
            STAGE_II_CACHED_TRAJ_QUERY_BINDING_V2_SCHEMA_VERSION
        ),
        "cached_traj_query_training_view_v2_schema_version": (
            STAGE_II_CACHED_TRAJ_QUERY_TRAINING_VIEW_V2_SCHEMA_VERSION
        ),
        "direct_latent_view_schema_version": STAGE_II_DIRECT_LATENT_VIEW_SCHEMA_VERSION,
        "projection_drop_v2_schema_version": STAGE_II_PROJECTION_DROP_V2_SCHEMA_VERSION,
        "projection_policy_id": STAGE_II_PROJECTION_POLICY_ID,
        "projection_policy_version": STAGE_II_PROJECTION_POLICY_VERSION,
        "projection_policy_v2_id": STAGE_II_PROJECTION_POLICY_V2_ID,
        "projection_policy_v2_version": STAGE_II_PROJECTION_POLICY_V2_VERSION,
        "trajectory_policy_id": STAGE_II_TRAJECTORY_POLICY_ID,
        "trajectory_policy_version": STAGE_II_TRAJECTORY_POLICY_VERSION,
        "trajectory_policy_v2_id": STAGE_II_TRAJECTORY_POLICY_V2_ID,
        "trajectory_policy_v2_version": STAGE_II_TRAJECTORY_POLICY_V2_VERSION,
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
        "src/mcp_server/data/stage_ii/context.py",
        "src/mcp_server/data/stage_ii/trajectory.py",
        "src/mcp_server/data/stage_ii/raw_trajectory_v2.py",
        "src/mcp_server/data/stage_ii/trajectory_v2_corpus.py",
        "src/mcp_server/data/stage_ii/sources.py",
        "src/mcp_server/data/stage_ii/legacy_direct.py",
        "src/mcp_server/data/stage_ii/views.py",
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
