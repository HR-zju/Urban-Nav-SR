"""Build public-only UrbanNav subsets admissible by a native CityWalker model.

This is a dataset-side admission adapter, not an evaluator and not the native
model runner.  It reads a frozen generic UrbanNav package, checks each visible
row without opening its hidden reference, and writes a smaller immutable
package only for compatible public inputs.  The corresponding references are
copied *after* admission solely to preserve the score denominator.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from mcp_server.contracts.benchmark_plugin import (
    BenchmarkInputV1,
    BenchmarkProfileV1,
    BenchmarkReferenceV1,
    BenchmarkTimePolicyV1,
    benchmark_plugin_digest_v1,
)
from mcp_server.contracts.citywalker_to_urbannav_adapter import (
    CITYWALKER_TO_URBANNAV_ADAPTER_CONTRACT_VERSION_V1,
    CityWalkerToUrbanNavInputErrorV1,
    URBANNAV_CITYWALKER_NATIVE_INPUT_PROFILE_ID_V1,
    URBANNAV_CURRENT_EGO_FLU_PREDICTION_GEOMETRY_ID_V1,
    prepare_citywalker_to_urbannav_public_input_v1,
)
from mcp_server.data.benchmark.urbannav_dataset_plugin import URBANNAV_DATASET_PLUGIN_ID_V1
from mcp_server.eval.openloop.benchmark_plugin_package import (
    artifact_sha256_v1,
    build_benchmark_package_manifest_v1,
    load_benchmark_package_v1,
    write_benchmark_rows_v1,
)

URBANNAV_CITYWALKER_COMPATIBILITY_AUDIT_SCHEMA_VERSION_V1 = "urbannav_citywalker_compatibility_audit_v1"
URBANNAV_CITYWALKER_DATASET_PLUGIN_ID_V1 = "urbannav_citywalker_native_projection_v1"


def urbannav_citywalker_native_default_profile_v1(
    *,
    profile_id: str = "urbannav_citywalker_native_openloop_v1",
) -> BenchmarkProfileV1:
    """Return the reverse-baseline profile with only native time support.

    CityWalker emits five explicit one-Hz waypoints.  This initial profile
    evaluates exactly the reliably covered 1 s and 2 s horizons.  Its UrbanNav
    direction suite remains explicitly ``N/A`` because the native model emits
    no direction readout.

    Args:
        profile_id: Stable derived-package profile identity.

    Returns:
        Strict native-CityWalker-on-UrbanNav benchmark profile.
    """

    return BenchmarkProfileV1(
        profile_id=profile_id,
        dataset_plugin_id=URBANNAV_CITYWALKER_DATASET_PLUGIN_ID_V1,
        model_input_profile_id=URBANNAV_CITYWALKER_NATIVE_INPUT_PROFILE_ID_V1,
        prediction_geometry_id=
        URBANNAV_CURRENT_EGO_FLU_PREDICTION_GEOMETRY_ID_V1,
        metric_suite_ids=("trajectory_geometry_core_v1",
                          "urbannav_openloop_metrics_v1"),
        time_policy=BenchmarkTimePolicyV1(
            policy_id="citywalker_native_five_waypoint_1hz_v1",
            horizons_s=(1.0, 2.0),
            min_angle_reference_distance_m=1e-3,
        ),
    )


def audit_urbannav_citywalker_compatibility_v1(
    inputs: dict[str, BenchmarkInputV1], ) -> dict[str, Any]:
    """Audit a frozen UrbanNav visible denominator without touching references.

    Args:
        inputs: Frozen generic model-visible inputs keyed by stable sample id.

    Returns:
        Deterministic coverage funnel and per-sample public admission rows.
    """

    entries: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for sample_id, input_row in inputs.items():
        entry: dict[str, Any] = {
            "sample_id": sample_id,
            "benchmark_input_digest": benchmark_plugin_digest_v1(input_row),
            "eligibility_status": "input_ineligible",
            "reject_reason": None,
            "selected_image_indices": [],
            "selected_image_roles": [],
            "public_scale_value": None,
        }
        try:
            prepared = prepare_citywalker_to_urbannav_public_input_v1(
                images=input_row.images,
                image_roles=input_row.image_roles,
                public_context=input_row.public_context,
            )
        except CityWalkerToUrbanNavInputErrorV1 as exc:
            entry["reject_reason"] = exc.reason
            reason_counts[exc.reason] += 1
        else:
            entry.update({
                "eligibility_status":
                "eligible",
                "selected_image_indices":
                list(prepared.selected_image_indices),
                "selected_image_roles":
                list(prepared.selected_image_roles),
                "public_scale_value":
                prepared.public_scale_value,
            })
        entries.append(entry)
    eligible_count = sum(entry["eligibility_status"] == "eligible"
                         for entry in entries)
    return {
        "schema_version":
        URBANNAV_CITYWALKER_COMPATIBILITY_AUDIT_SCHEMA_VERSION_V1,
        "adapter_contract_version":
        CITYWALKER_TO_URBANNAV_ADAPTER_CONTRACT_VERSION_V1,
        "denominator_count": len(entries),
        "eligible_count": eligible_count,
        "eligible_fraction": eligible_count / len(entries) if entries else 0.0,
        "reject_reason_counts": dict(sorted(reason_counts.items())),
        "entries": entries,
    }


def build_urbannav_citywalker_compatible_package_v1(
    *,
    source_package_json: str | Path,
    output_root: str | Path,
    package_id: str,
    profile: BenchmarkProfileV1,
    audit_only: bool = False,
) -> dict[str, Any]:
    """Audit and, when admitted rows exist, write a reverse-baseline package.

    Args:
        source_package_json: Frozen generic UrbanNav package manifest.
        output_root: Destination for audit and optional derived package.
        package_id: Immutable destination package identity.
        profile: Derived native CityWalker profile.
        audit_only: Write only the public compatibility report when true.

    Returns:
        Paths, full-denominator coverage, and optional derived package details.

    Raises:
        ValueError: If source/profile ownership drifts or no row is admissible
            for a requested non-audit package.
    """

    _validate_profile_v1(profile)
    source_manifest, source_inputs, source_references = load_benchmark_package_v1(
        source_package_json)
    if source_manifest.benchmark_profile.dataset_plugin_id != URBANNAV_DATASET_PLUGIN_ID_V1:
        raise ValueError(
            "source package must be an UrbanNav Stage-II generic package")
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit = audit_urbannav_citywalker_compatibility_v1(source_inputs)
    audit_path = output / "compatibility_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    report: dict[str, Any] = {
        "source_package_json":
        str(Path(source_package_json).expanduser().resolve()),
        "compatibility_audit_path":
        str(audit_path),
        "denominator_count":
        audit["denominator_count"],
        "eligible_count":
        audit["eligible_count"],
        "eligible_fraction":
        audit["eligible_fraction"],
        "reject_reason_counts":
        audit["reject_reason_counts"],
        "audit_only":
        audit_only,
    }
    if audit_only:
        return report
    eligible_ids = {
        str(entry["sample_id"])
        for entry in audit["entries"]
        if entry["eligibility_status"] == "eligible"
    }
    if not eligible_ids:
        raise ValueError(
            "no public UrbanNav rows are compatible with the native CityWalker contract"
        )
    inputs = [
        _derive_input_v1(source_inputs[sample_id], profile)
        for sample_id in source_inputs if sample_id in eligible_ids
    ]
    references = [
        _derive_reference_v1(source_references[sample_id], profile)
        for sample_id in source_references if sample_id in eligible_ids
    ]
    input_path = output / "inputs.jsonl"
    reference_path = output / "references.jsonl"
    manifest_path = output / "package.json"
    write_benchmark_rows_v1(input_path, inputs)
    write_benchmark_rows_v1(reference_path, references)
    manifest = build_benchmark_package_manifest_v1(
        package_id=package_id,
        profile=profile,
        split=source_manifest.split,
        input_jsonl=input_path,
        reference_jsonl=reference_path,
        output_json=manifest_path,
        source_fingerprints={
            "source_package_json": artifact_sha256_v1(source_package_json),
            "source_input_digest": source_manifest.input_digest,
            "source_reference_digest": source_manifest.reference_digest,
            "compatibility_audit_sha256": artifact_sha256_v1(audit_path),
        },
    )
    report.update({
        "package_path": str(manifest_path),
        "input_path": str(input_path),
        "reference_path": str(reference_path),
        "input_count": manifest.input_count,
        "reference_count": manifest.reference_count,
    })
    return report


def _derive_input_v1(source: BenchmarkInputV1,
                     profile: BenchmarkProfileV1) -> BenchmarkInputV1:
    """Copy one admitted public input while reducing media to the native five."""

    prepared = prepare_citywalker_to_urbannav_public_input_v1(
        images=source.images,
        image_roles=source.image_roles,
        public_context=source.public_context,
    )
    context = dict(source.public_context)
    # The derived input contains only five selected frames.  Its public history
    # association must therefore use this derived image index space, while the
    # audit retains the source indices separately before package construction.
    context["history_image_indices"] = list(
        range(len(prepared.selected_images)))
    context["citywalker_to_urbannav_adapter_contract"] = {
        "version": CITYWALKER_TO_URBANNAV_ADAPTER_CONTRACT_VERSION_V1,
        "source_selected_image_indices": list(prepared.selected_image_indices),
        "public_scale_estimator_id": "public_history_median_step_v1",
    }
    return source.model_copy(
        update={
            "dataset_plugin_id": URBANNAV_CITYWALKER_DATASET_PLUGIN_ID_V1,
            "benchmark_profile_id": profile.profile_id,
            "images": list(prepared.selected_images),
            "image_roles": list(prepared.selected_image_roles),
            "public_context": context,
            "modality_profile_id":
            URBANNAV_CITYWALKER_NATIVE_INPUT_PROFILE_ID_V1,
        })


def _derive_reference_v1(source: BenchmarkReferenceV1,
                         profile: BenchmarkProfileV1) -> BenchmarkReferenceV1:
    """Copy hidden references only after their public input passed admission."""

    return source.model_copy(
        update={
            "dataset_plugin_id": URBANNAV_CITYWALKER_DATASET_PLUGIN_ID_V1,
            "benchmark_profile_id": profile.profile_id,
        })


def _validate_profile_v1(profile: BenchmarkProfileV1) -> None:
    """Require the exact declared reverse-baseline ownership and geometry."""

    if profile.dataset_plugin_id != URBANNAV_CITYWALKER_DATASET_PLUGIN_ID_V1:
        raise ValueError("derived profile has a different dataset plugin")
    if profile.model_input_profile_id != URBANNAV_CITYWALKER_NATIVE_INPUT_PROFILE_ID_V1:
        raise ValueError("derived profile has a different input profile")
    if profile.prediction_geometry_id != URBANNAV_CURRENT_EGO_FLU_PREDICTION_GEOMETRY_ID_V1:
        raise ValueError("derived profile has a different prediction geometry")
    if profile.time_policy.horizons_s != (1.0, 2.0):
        raise ValueError(
            "native CityWalker profile must freeze only 1s and 2s horizons")


__all__ = [
    "URBANNAV_CITYWALKER_COMPATIBILITY_AUDIT_SCHEMA_VERSION_V1",
    "URBANNAV_CITYWALKER_DATASET_PLUGIN_ID_V1",
    "audit_urbannav_citywalker_compatibility_v1",
    "build_urbannav_citywalker_compatible_package_v1",
    "urbannav_citywalker_native_default_profile_v1",
]
