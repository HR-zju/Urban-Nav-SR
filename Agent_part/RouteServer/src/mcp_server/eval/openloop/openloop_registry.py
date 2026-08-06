"""Build checkpoint exposure and validated run-lineage artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mcp_server.contracts.openloop_benchmark import (
    OPENLOOP_EXPOSURE_REGISTRY_VERSION,
    CheckpointExposureEntryV1,
    CheckpointExposureRegistryV1,
    OpenLoopCheckpointIdentityV1,
    OpenLoopContextIdentityV1,
    OpenLoopPackageManifestV1,
    OpenLoopRepositoryStateV1,
    OpenLoopRunManifestV1,
)
from mcp_server.data.io.jsonl import iter_jsonl
from mcp_server.data.openloop.lineage import resolve_explicit_field_v1
from mcp_server.eval.openloop.openloop_common import artifact_sha256_v1


def build_checkpoint_exposure_registry_v1(
    spec_json: str | Path,
    output_json: str | Path,
    *,
    checkpoint_digest: str,
) -> CheckpointExposureRegistryV1:
    """Build an exposure registry from explicit train-manifest mappings.

    The specification is a JSON object with a ``manifests`` list.  Every item
    must declare ``path``, ``stage``, ``parent_field``, and
    ``split_group_field``. Selectors may be explicit dotted paths. An optional
    ``where`` object applies exact equality filters before projection.
    Optional field-name selectors are
    ``source_fingerprint_field``, ``annotation_version_field``, and
    ``cache_version_field``. Literal version values may instead be supplied as
    ``annotation_version`` or ``cache_version``.

    Args:
        spec_json: Explicit train-manifest mapping specification.
        output_json: Registry destination.
        checkpoint_digest: Immutable evaluated checkpoint lineage digest.

    Returns:
        Validated exposure registry.

    Raises:
        ValueError: If mappings are absent, identities are missing, or one
            identity maps to conflicting exposure metadata.
    """

    spec_path = Path(spec_json).expanduser().resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    manifests = spec.get("manifests")
    if not isinstance(manifests, list) or not manifests:
        raise ValueError("exposure spec requires a non-empty manifests list")
    entries: dict[tuple[str, str], CheckpointExposureEntryV1] = {}
    for manifest_spec in manifests:
        if not isinstance(manifest_spec, dict):
            raise ValueError("each exposure manifest spec must be an object")
        manifest_path = _resolve_spec_path(spec_path.parent, manifest_spec["path"])
        manifest_digest = artifact_sha256_v1(manifest_path)
        for row in iter_jsonl(manifest_path):
            if not _matches_explicit_where(row, manifest_spec.get("where")):
                continue
            entry = _exposure_entry(
                row,
                manifest_spec,
                manifest_digest=manifest_digest,
            )
            key = (entry.parent_sample_id, entry.split_group_id)
            prior = entries.get(key)
            if prior is not None and prior != entry:
                raise ValueError(
                    "conflicting exposure metadata for parent/group: "
                    f"{key[0]}/{key[1]}"
                )
            entries[key] = entry
    registry = CheckpointExposureRegistryV1(
        schema_version=OPENLOOP_EXPOSURE_REGISTRY_VERSION,
        checkpoint_digest=checkpoint_digest,
        entries=[entries[key] for key in sorted(entries)],
    )
    _write_model(output_json, registry)
    return registry


def write_validated_openloop_run_manifest_v1(
    input_json: str | Path,
    output_json: str | Path,
) -> OpenLoopRunManifestV1:
    """Validate and canonically render one run manifest specification.

    Args:
        input_json: Run-manifest payload.
        output_json: Validated canonical destination.

    Returns:
        Validated run manifest.
    """

    payload = json.loads(Path(input_json).expanduser().read_text(encoding="utf-8"))
    run = OpenLoopRunManifestV1.model_validate(payload)
    _write_model(output_json, run)
    return run


def build_openloop_run_manifest_v1(
    *,
    run_id: str,
    stage: str,
    package_json: str | Path,
    checkpoint_json: str | Path,
    repositories_json: str | Path,
    output_json: str | Path,
    context_identity_json: str | Path | None = None,
    generation_parameters_json: str | Path | None = None,
    hardware_json: str | Path | None = None,
    batch_size: int,
    precision: str,
) -> OpenLoopRunManifestV1:
    """Assemble a run manifest from frozen typed component artifacts.

    Args:
        run_id: Stable run identity.
        stage: Exact Stage I or Stage II selector.
        package_json: Frozen benchmark package manifest.
        checkpoint_json: Checkpoint identity JSON object.
        repositories_json: JSON list of repository states.
        output_json: Run manifest destination.
        context_identity_json: Required Stage II context-identity JSON.
        generation_parameters_json: Optional decoding/sampling settings.
        hardware_json: Optional non-secret hardware metadata.
        batch_size: Physical inference batch size.
        precision: Inference precision identity.

    Returns:
        Validated canonical run manifest.

    Raises:
        ValueError: If stage or Stage II context identity is incomplete.
    """

    if stage not in {"stage_i", "stage_ii"}:
        raise ValueError("run stage must be stage_i or stage_ii")
    package_path = Path(package_json).expanduser().resolve()
    package = OpenLoopPackageManifestV1.model_validate_json(
        package_path.read_text(encoding="utf-8")
    )
    checkpoint = OpenLoopCheckpointIdentityV1.model_validate(
        _read_json(checkpoint_json)
    )
    repository_payload = _read_json(repositories_json)
    if not isinstance(repository_payload, list):
        raise ValueError("repositories JSON must contain a list")
    repositories = [
        OpenLoopRepositoryStateV1.model_validate(row)
        for row in repository_payload
    ]
    if stage == "stage_i":
        if context_identity_json is not None:
            raise ValueError("Stage I run does not accept context_identity_json")
        context = OpenLoopContextIdentityV1(
            policy="not_applicable",
            reasoning_context_source="not_applicable",
            decision_context_source="not_applicable",
            input_manifest_digest=package.input_digest,
            seed=0,
        )
    else:
        if context_identity_json is None:
            raise ValueError("Stage II run requires context_identity_json")
        context = OpenLoopContextIdentityV1.model_validate(
            _read_json(context_identity_json)
        )
        if context.input_manifest_digest != package.input_digest:
            raise ValueError(
                "Stage II context input manifest digest differs from package"
            )
    run = OpenLoopRunManifestV1(
        run_id=run_id,
        benchmark_id=package.benchmark_id,
        benchmark_digest=artifact_sha256_v1(package_path),
        stage=stage,
        repositories=repositories,
        checkpoint=checkpoint,
        renderer_policy=package.renderer_policy,
        metric_policy_digest=_canonical_model_digest(package.metric_policy),
        context_identity=context,
        generation_parameters=_read_optional_object(generation_parameters_json),
        hardware=_read_optional_object(hardware_json),
        batch_size=batch_size,
        precision=precision,
    )
    _write_model(output_json, run)
    return run


def _exposure_entry(
    row: dict[str, Any],
    spec: dict[str, Any],
    *,
    manifest_digest: str,
) -> CheckpointExposureEntryV1:
    """Project one explicitly mapped training row to exposure identity."""

    parent = _required_row_value(row, spec, "parent_field")
    split_group = _required_row_value(row, spec, "split_group_field")
    return CheckpointExposureEntryV1(
        parent_sample_id=parent,
        split_group_id=split_group,
        source_fingerprint=_optional_row_value(
            row,
            spec.get("source_fingerprint_field"),
        ),
        manifest_digest=manifest_digest,
        stage=str(spec["stage"]),
        annotation_version=_version_value(row, spec, "annotation_version"),
        cache_version=_version_value(row, spec, "cache_version"),
    )


def _required_row_value(
    row: dict[str, Any],
    spec: dict[str, Any],
    selector: str,
) -> str:
    """Resolve one required explicitly named exposure identity field."""

    field = spec.get(selector)
    if not isinstance(field, str) or not field:
        raise ValueError(f"exposure spec requires {selector}")
    return resolve_explicit_field_v1(row, field)


def _optional_row_value(
    row: dict[str, Any],
    field: Any,
) -> str | None:
    """Resolve an optional explicitly named row field."""

    if field is None:
        return None
    if not isinstance(field, str) or not field:
        raise ValueError("optional exposure field selector must be a string")
    try:
        return resolve_explicit_field_v1(row, field)
    except ValueError as exc:
        if "is missing explicit field" in str(exc):
            return None
        raise


def _matches_explicit_where(
    row: dict[str, Any],
    where: Any,
) -> bool:
    """Apply optional exact-equality filters with explicit dotted selectors."""

    if where is None:
        return True
    if not isinstance(where, dict) or not where:
        raise ValueError("exposure manifest where filter must be an object")
    for selector, expected in where.items():
        if not isinstance(selector, str) or not selector:
            raise ValueError("exposure where selector must be a string")
        if isinstance(expected, (dict, list)) or expected is None:
            raise ValueError("exposure where values must be non-null scalars")
        if resolve_explicit_field_v1(row, selector) != str(expected):
            return False
    return True


def _version_value(
    row: dict[str, Any],
    spec: dict[str, Any],
    name: str,
) -> str | None:
    """Resolve a literal or explicit row-field version without guessing."""

    field = spec.get(f"{name}_field")
    literal = spec.get(name)
    if field is not None and literal is not None:
        raise ValueError(f"choose either {name} or {name}_field")
    if field is not None:
        return _optional_row_value(row, field)
    return None if literal is None else str(literal)


def _resolve_spec_path(root: Path, value: Any) -> Path:
    """Resolve a manifest path relative to the exposure specification."""

    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _read_json(path: str | Path) -> Any:
    """Read one JSON artifact."""

    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _read_optional_object(path: str | Path | None) -> dict[str, Any]:
    """Read an optional JSON object and reject non-object values."""

    if path is None:
        return {}
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ValueError("optional run metadata JSON must contain an object")
    return value


def _canonical_model_digest(model: Any) -> str:
    """Return stable SHA256 for one Pydantic model payload."""

    encoded = json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_model(path: str | Path, model: Any) -> None:
    """Write one Pydantic artifact using stable pretty JSON."""

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


__all__ = [
    "build_openloop_run_manifest_v1",
    "build_checkpoint_exposure_registry_v1",
    "write_validated_openloop_run_manifest_v1",
]
