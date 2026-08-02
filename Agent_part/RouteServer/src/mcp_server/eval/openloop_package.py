"""Build immutable UrbanNav open-loop package manifests."""

from __future__ import annotations

import json
from pathlib import Path

from mcp_server.contracts.openloop_benchmark import (
    OPENLOOP_BENCHMARK_SCHEMA_VERSION,
    OpenLoopInputV1,
    OpenLoopMetricPolicyV1,
    OpenLoopPackageManifestV1,
    OpenLoopReferenceV1,
    OpenLoopRendererPolicyV1,
)
from mcp_server.eval.openloop_common import (
    artifact_sha256_v1,
    audit_input_reference_identity_v1,
    canonical_row_digest_v1,
    load_unique_jsonl_models_v1,
)


def build_openloop_package_manifest_v1(
    *,
    benchmark_id: str,
    split: str,
    stage: str,
    input_jsonl: str | Path,
    reference_jsonl: str | Path,
    config_json: str | Path,
    output_json: str | Path,
) -> OpenLoopPackageManifestV1:
    """Validate inputs/references and write one immutable package manifest.

    Args:
        benchmark_id: Human-readable versioned package identity.
        split: Frozen validation or test split.
        stage: Expected ``stage_i`` or ``stage_ii`` row stage.
        input_jsonl: Model-visible input artifact.
        reference_jsonl: Hidden reference artifact.
        config_json: Stage-specific canonical evaluation config.
        output_json: Package manifest destination.

    Returns:
        Validated package manifest.

    Raises:
        ValueError: If row identity, split, stage, renderer, or config drifts.
    """

    if split not in {"validation", "test"}:
        raise ValueError("package split must be validation or test")
    if stage not in {"stage_i", "stage_ii"}:
        raise ValueError("package stage must be stage_i or stage_ii")
    input_path = Path(input_jsonl).expanduser().resolve()
    reference_path = Path(reference_jsonl).expanduser().resolve()
    output_path = Path(output_json).expanduser().resolve()
    config = json.loads(
        Path(config_json).expanduser().read_text(encoding="utf-8")
    )
    metric_policy = OpenLoopMetricPolicyV1.model_validate(
        config["metric_policy"]
    )
    renderer_policy = OpenLoopRendererPolicyV1.model_validate(
        config["renderer_policy"]
    )
    renderer_digest = canonical_row_digest_v1(renderer_policy)
    inputs = load_unique_jsonl_models_v1(input_path, OpenLoopInputV1)
    references = load_unique_jsonl_models_v1(
        reference_path,
        OpenLoopReferenceV1,
    )
    audit_input_reference_identity_v1(inputs, references)
    for row in inputs.values():
        if row.split != split:
            raise ValueError(f"input split drift: {row.sample_id}")
        if row.stage != stage:
            raise ValueError(f"input stage drift: {row.sample_id}")
        if row.renderer_policy_digest != renderer_digest:
            raise ValueError(
                f"input renderer-policy drift: {row.sample_id}"
            )
    manifest = OpenLoopPackageManifestV1(
        schema_version=OPENLOOP_BENCHMARK_SCHEMA_VERSION,
        benchmark_id=benchmark_id,
        split=split,
        input_path=_portable_path(input_path, output_path.parent),
        input_digest=artifact_sha256_v1(input_path),
        reference_path=_portable_path(reference_path, output_path.parent),
        reference_digest=artifact_sha256_v1(reference_path),
        metric_policy=metric_policy,
        renderer_policy=renderer_policy,
        reference_count=len(references),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _portable_path(path: Path, root: Path) -> str:
    """Prefer a package-relative path when both artifacts share a root."""

    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


__all__ = ["build_openloop_package_manifest_v1"]
