"""Build immutable Stage II Pass-R and Pass-D context-cache records."""

from __future__ import annotations

import re
from typing import Any, Mapping

from mcp_server.agent.reasoning_v4_text import (
    parse_reasoning_section_v4,
    parse_reasoning_target_v4,
    render_reasoning_section_v4,
)
from mcp_server.contracts.stage_ii_cfm import (
    STAGE_II_CONTEXT_CACHE_SCHEMA_VERSION,
    STAGE_II_GROUNDING_SECTION_NAMES,
    STAGE_II_REASONING_CONTEXT_V2_SCHEMA_VERSION,
    StageIIContextCacheV1,
    StageIIContextProducerV1,
    StageIIReasoningAbsentV2,
    StageIIReasoningPresentV2,
    stage_ii_digest_v1,
)
from mcp_server.contracts.stage_i_sft import (
    StageIFullCoTViewV2,
    StageIReasoningDropoutViewV2,
    StageITrainingViewV2,
    parse_stage_i_training_view_v2,
)


def stage_ii_projected_input_digest_v1(
    *,
    images: list[str],
    image_roles: list[str],
    instruction: str | None,
    model_visible_context: Mapping[str, Any],
    evidence_catalog: list[Mapping[str, Any]],
    input_modalities: Mapping[str, Any],
) -> str:
    """Fingerprint the exact student-visible input before Pass R/D/T.

    Args:
        images: Ordered projected media references.
        image_roles: Semantic role aligned with every image.
        instruction: Optional visible instruction.
        model_visible_context: Additional safe visible context.
        evidence_catalog: Prompt-visible evidence inventory.
        input_modalities: Canonical projected modality flags.

    Returns:
        Canonical SHA-256 input digest shared by all three passes.
    """

    return stage_ii_digest_v1(
        {
            "images": images,
            "image_roles": image_roles,
            "instruction": instruction,
            "model_visible_context": dict(model_visible_context),
            "evidence_catalog": evidence_catalog,
            "input_modalities": dict(input_modalities),
        }
    )


def build_stage_ii_context_cache_v1(
    *,
    sample_id: str,
    source_group_id: str,
    split_group_id: str,
    dataset_source: str,
    modality_profile_id: str,
    parent_source_fingerprint: str,
    projected_input_digest: str,
    context_kind: str,
    producer: StageIIContextProducerV1 | Mapping[str, Any],
    raw_generation: str,
) -> StageIIContextCacheV1:
    """Strictly parse one independent context pass and seal its cache identity.

    Pass R accepts a complete V4.6 FullCoT but stores only the first five
    grounding sections.  Pass D accepts exactly one standalone Decision tag.
    The Decision embedded in Pass R is intentionally never reused here.

    Args:
        sample_id: Parent Stage I FullCoT sample identity.
        source_group_id: Parent source-group identity.
        split_group_id: Leakage-safe split-group identity.
        dataset_source: Dataset family.
        modality_profile_id: Exact projected modality profile.
        parent_source_fingerprint: Immutable parent source digest.
        projected_input_digest: Exact student-visible input digest.
        context_kind: ``reasoning_grounding`` or ``decision``.
        producer: Frozen producer identity.
        raw_generation: Unmodified LVLM output or canonical oracle text.

    Returns:
        Accepted or rejected immutable context-cache record.

    Raises:
        ValueError: If the context kind or producer contract is invalid.
    """

    typed_producer = (
        producer
        if isinstance(producer, StageIIContextProducerV1)
        else StageIIContextProducerV1.model_validate(producer)
    )
    canonical_text: str | None = None
    errors: tuple[str, ...] = ()
    if context_kind == "reasoning_grounding":
        parsed = parse_reasoning_target_v4(raw_generation)
        section_names = STAGE_II_GROUNDING_SECTION_NAMES
        if parsed.valid and parsed.target is not None:
            trace = parsed.target.reasoning
            values = (
                trace.task_grounding,
                trace.scene_grounding,
                trace.local_traversability,
                trace.navigation_context_grounding,
                trace.interaction_constraint_grounding,
            )
            canonical_text = "\n".join(
                render_reasoning_section_v4(name, value)
                for name, value in zip(section_names, values)
            )
        else:
            errors = tuple(_bounded_error_code(item) for item in parsed.errors)
    elif context_kind == "decision":
        parsed = parse_reasoning_section_v4(raw_generation, "Decision")
        section_names = ("Decision",)
        if parsed.valid and parsed.section is not None:
            canonical_text = render_reasoning_section_v4(
                "Decision",
                parsed.section,
            )
        else:
            errors = tuple(_bounded_error_code(item) for item in parsed.errors)
    else:
        raise ValueError(f"unsupported Stage II context_kind: {context_kind}")
    parse_status = "accepted" if canonical_text is not None else "rejected"
    cache_identity = stage_ii_digest_v1(
        {
            "sample_id": sample_id,
            "modality_profile_id": modality_profile_id,
            "projected_input_digest": projected_input_digest,
            "context_kind": context_kind,
            "producer": typed_producer.model_dump(mode="json", exclude_none=True),
        }
    )
    payload = {
        "schema_version": STAGE_II_CONTEXT_CACHE_SCHEMA_VERSION,
        "cache_id": f"stage_ii:{context_kind}:{cache_identity}",
        "sample_id": sample_id,
        "source_group_id": source_group_id,
        "split_group_id": split_group_id,
        "dataset_source": dataset_source,
        "modality_profile_id": modality_profile_id,
        "parent_source_fingerprint": parent_source_fingerprint,
        "projected_input_digest": projected_input_digest,
        "context_kind": context_kind,
        "section_names": section_names,
        "producer": typed_producer.model_dump(mode="json", exclude_none=True),
        "raw_generation": raw_generation,
        "canonical_text": canonical_text,
        "parse_status": parse_status,
        "errors": errors,
    }
    payload["cache_digest"] = stage_ii_digest_v1(payload)
    return StageIIContextCacheV1.model_validate(payload)


def stage_ii_projected_input_digest_from_stage_i_v2(
    stage_i_view: StageITrainingViewV2 | Mapping[str, Any],
) -> str:
    """Fingerprint the exact observation shared by both Stage II V2 routes."""

    typed = (
        parse_stage_i_training_view_v2(dict(stage_i_view))
        if isinstance(stage_i_view, Mapping)
        else stage_i_view
    )
    if not isinstance(typed, (StageIFullCoTViewV2, StageIReasoningDropoutViewV2)):
        raise ValueError("Stage II V2 context requires a Reasoning Stage I parent")
    return stage_ii_projected_input_digest_v1(
        images=list(typed.images),
        image_roles=list(typed.image_roles),
        instruction=typed.instruction,
        model_visible_context=typed.model_visible_context,
        evidence_catalog=[
            item.model_dump(mode="json", exclude_none=True)
            for item in typed.evidence_catalog
        ],
        input_modalities=typed.input_modalities.model_dump(mode="json"),
    )


def stage_ii_parent_sample_id_v2(
    stage_i_view: StageITrainingViewV2 | Mapping[str, Any],
) -> str:
    """Resolve the stable FullCoT parent identity behind either V2 route."""

    typed = (
        parse_stage_i_training_view_v2(dict(stage_i_view))
        if isinstance(stage_i_view, Mapping)
        else stage_i_view
    )
    if isinstance(typed, StageIFullCoTViewV2):
        return typed.sample_id
    if isinstance(typed, StageIReasoningDropoutViewV2):
        return typed.dropout_link.parent_full_cot_sample_id
    raise ValueError("Stage II V2 parent identity requires a Reasoning Stage I view")


def build_stage_ii_context_cache_v2(
    stage_i_view: StageITrainingViewV2 | Mapping[str, Any],
    *,
    context_kind: str,
    producer: StageIIContextProducerV1 | Mapping[str, Any],
    raw_generation: str,
) -> StageIIContextCacheV1:
    """Build an accepted-or-rejected cache over one V2 route observation.

    A reasoning-dropout parent may only create Pass D.  Pass R absence is
    materialized separately from the signed dropout lineage, never from a
    failed parser result.
    """

    typed = (
        parse_stage_i_training_view_v2(dict(stage_i_view))
        if isinstance(stage_i_view, Mapping)
        else stage_i_view
    )
    if not isinstance(typed, (StageIFullCoTViewV2, StageIReasoningDropoutViewV2)):
        raise ValueError("Stage II V2 cache requires a Reasoning Stage I view")
    if isinstance(typed, StageIReasoningDropoutViewV2) and context_kind != "decision":
        raise ValueError("reasoning dropout cannot create a Pass-R cache")
    return build_stage_ii_context_cache_v1(
        sample_id=stage_ii_parent_sample_id_v2(typed),
        source_group_id=typed.source_group_id,
        split_group_id=typed.split_group_id,
        dataset_source=typed.dataset_source,
        modality_profile_id=typed.modality_profile_id,
        parent_source_fingerprint=typed.parent_source_fingerprint,
        projected_input_digest=stage_ii_projected_input_digest_from_stage_i_v2(typed),
        context_kind=context_kind,
        producer=producer,
        raw_generation=raw_generation,
    )


def build_stage_ii_reasoning_present_v2(
    reasoning_context: StageIIContextCacheV1 | Mapping[str, Any],
) -> StageIIReasoningPresentV2:
    """Seal one accepted Pass-R cache as a V2 present state."""

    cache = (
        reasoning_context
        if isinstance(reasoning_context, StageIIContextCacheV1)
        else StageIIContextCacheV1.model_validate(reasoning_context)
    )
    payload = {
        "schema_version": STAGE_II_REASONING_CONTEXT_V2_SCHEMA_VERSION,
        "reasoning_state": "present",
        "reasoning_context": cache.model_dump(mode="json", exclude_none=True),
    }
    payload["state_digest"] = stage_ii_digest_v1(payload)
    return StageIIReasoningPresentV2.model_validate(payload)


def build_stage_ii_reasoning_absent_v2(
    stage_i_view: StageIReasoningDropoutViewV2 | Mapping[str, Any],
) -> StageIIReasoningAbsentV2:
    """Seal explicit no-CoT lineage without constructing a fake R cache."""

    typed = (
        parse_stage_i_training_view_v2(dict(stage_i_view))
        if isinstance(stage_i_view, Mapping)
        else stage_i_view
    )
    if not isinstance(typed, StageIReasoningDropoutViewV2):
        raise ValueError("Stage II V2 absence requires a reasoning-dropout view")
    payload = {
        "schema_version": STAGE_II_REASONING_CONTEXT_V2_SCHEMA_VERSION,
        "reasoning_state": "absent_by_reasoning_dropout",
        "parent_stage_i_sample_id": typed.dropout_link.parent_full_cot_sample_id,
        "stage_i_dropout_sample_id": typed.sample_id,
        "stage_i_dropout_projection_digest": typed.projection_digest,
        "source_group_id": typed.source_group_id,
        "split_group_id": typed.split_group_id,
        "dataset_source": typed.dataset_source,
        "modality_profile_id": typed.modality_profile_id,
        "parent_source_fingerprint": typed.parent_source_fingerprint,
        "projected_input_digest": stage_ii_projected_input_digest_from_stage_i_v2(
            typed
        ),
        "dropout_link": typed.dropout_link.model_dump(mode="json"),
    }
    payload["state_digest"] = stage_ii_digest_v1(payload)
    return StageIIReasoningAbsentV2.model_validate(payload)


def _bounded_error_code(value: str) -> str:
    """Map detailed parser diagnostics to the open bounded ReasonCode wire."""

    prefix = str(value).split(":", 1)[0].strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", prefix).strip("_")
    return (normalized or "context_parse_failed")[:96]


__all__ = [
    "build_stage_ii_context_cache_v1",
    "build_stage_ii_context_cache_v2",
    "build_stage_ii_reasoning_absent_v2",
    "build_stage_ii_reasoning_present_v2",
    "stage_ii_parent_sample_id_v2",
    "stage_ii_projected_input_digest_v1",
    "stage_ii_projected_input_digest_from_stage_i_v2",
]
