"""Deterministic and dormant external-label producer adapters for VQA V2."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol

from mcp_server.contracts.vqa_dataset_v2 import (
    ExternalLabelInterfaceBaseV2,
    VQAAnnotationV2,
    VQACapabilityConfigV2,
    VQASourceV2,
    annotation_context_digest_v2,
)
from mcp_server.contracts.vqa_v2 import (
    VQAQuestionV2,
    build_vqa_answer_v2,
    get_vqa_task_spec_v2,
)
from mcp_server.data.vqa_v2_eligibility import (
    vqa_task_eligibility_v2,
)


CANONICAL_DIRECTION_PRODUCER_ID = (
    "canonical_demonstrated_direction"
)
CANONICAL_DIRECTION_PRODUCER_VERSION = (
    "canonical_demonstrated_direction_v2"
)


@dataclass(frozen=True)
class VQAProducerResultV2:
    """Represent producer output without placeholder annotations.

    Args:
        status: Produced, unavailable, or rejected.
        annotation: Accepted annotation only for ``produced``.
        reason: Stable unavailable/rejected reason.
        errors: Bounded validation diagnostics.
    """

    status: str
    annotation: VQAAnnotationV2 | None = None
    reason: str | None = None
    errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Keep producer result and annotation presence consistent."""

        if self.status not in {
            "produced",
            "unavailable",
            "rejected",
        }:
            raise ValueError(
                f"unsupported producer status: {self.status}"
            )
        if (self.status == "produced") != (
            self.annotation is not None
        ):
            raise ValueError(
                "only produced results may carry an annotation"
            )


class VQALabelProducer(Protocol):
    """Protocol for deterministic or frozen external label producers."""

    producer_id: str
    producer_version: str
    supported_task_ids: tuple[str, ...]

    def produce(
        self,
        source: VQASourceV2,
        question: VQAQuestionV2,
        capabilities: VQACapabilityConfigV2,
    ) -> VQAProducerResultV2:
        """Produce one accepted annotation or explicit unavailable result."""


class AsyncVQALabelProducer(Protocol):
    """Protocol for provider-backed asynchronous label producers."""

    producer_id: str
    producer_version: str
    supported_task_ids: tuple[str, ...]

    async def produce(
        self,
        source: VQASourceV2,
        question: VQAQuestionV2,
        capabilities: VQACapabilityConfigV2,
    ) -> VQAProducerResultV2:
        """Produce one accepted annotation or explicit failure result."""


class CanonicalDemonstratedDirectionProducerV2:
    """Project an approved private direction stamp into the E0 target."""

    producer_id = CANONICAL_DIRECTION_PRODUCER_ID
    producer_version = CANONICAL_DIRECTION_PRODUCER_VERSION
    supported_task_ids = (
        "e0_demonstrated_direction_selection",
    )

    def produce(
        self,
        source: VQASourceV2,
        question: VQAQuestionV2,
        capabilities: VQACapabilityConfigV2,
    ) -> VQAProducerResultV2:
        """Build the only currently active deterministic VQA annotation.

        Args:
            source: Frozen VQA source.
            question: E0 answer-independent question.
            capabilities: Versioned task activation policy.

        Returns:
            Produced annotation or explicit unavailability.
        """

        if question.task_id not in self.supported_task_ids:
            return VQAProducerResultV2(
                status="unavailable",
                reason="producer_does_not_support_task",
            )
        eligibility = vqa_task_eligibility_v2(
            source,
            question.task_id,
            capabilities,
        )
        if not eligibility.eligible:
            return VQAProducerResultV2(
                status="unavailable",
                reason=eligibility.reason,
                errors=list(eligibility.missing_requirements),
            )
        direction = source.direction_supervision
        if direction is None:
            return VQAProducerResultV2(
                status="unavailable",
                reason="canonical_direction_unavailable",
            )
        try:
            answer = build_vqa_answer_v2(
                question.task_id,
                {"direction_label": direction.label_name},
            )
            context_digest = annotation_context_digest_v2(
                source=source,
                question=question,
                producer_id=self.producer_id,
                producer_version=self.producer_version,
            )
            annotation = VQAAnnotationV2(
                annotation_id=_annotation_id(
                    source,
                    question,
                    self.producer_version,
                ),
                source_group_id=source.observation.source_group_id,
                source_fingerprint=source.source_fingerprint,
                question=question,
                answer=answer,
                label_source="derived_gt",
                source_kind="expert_trajectory_direction",
                evidence_refs=list(question.evidence_refs),
                source_field_refs=[str(direction.source)],
                producer_id=self.producer_id,
                producer_version=self.producer_version,
                input_fingerprint=source.source_fingerprint,
                annotation_context_digest=context_digest,
                metadata={
                    "direction_stamp": direction.model_dump(
                        mode="json"
                    )
                },
            )
        except (TypeError, ValueError) as exc:
            return VQAProducerResultV2(
                status="rejected",
                reason="canonical_direction_validation_failed",
                errors=[str(exc)],
            )
        return VQAProducerResultV2(
            status="produced",
            annotation=annotation,
        )


def produce_external_interface_annotation_v2(
    source: VQASourceV2,
    question: VQAQuestionV2,
    capabilities: VQACapabilityConfigV2,
) -> VQAProducerResultV2:
    """Adapt a future accepted external interface when activated.

    Args:
        source: Frozen source that may carry an external interface.
        question: Exact dormant-task question.
        capabilities: Activation registry.

    Returns:
        External-GT annotation only after producer and export gates open.
    """

    eligibility = vqa_task_eligibility_v2(
        source,
        question.task_id,
        capabilities,
    )
    if not eligibility.eligible:
        return VQAProducerResultV2(
            status="unavailable",
            reason=eligibility.reason,
            errors=list(eligibility.missing_requirements),
        )
    interface = _interface_for_task(source, question.task_id)
    if interface is None:
        return VQAProducerResultV2(
            status="unavailable",
            reason="source_interface_unavailable",
        )
    try:
        answer = interface.answer
        context_digest = annotation_context_digest_v2(
            source=source,
            question=question,
            producer_id=interface.producer_id,
            producer_version=interface.producer_version,
        )
        annotation = VQAAnnotationV2(
            annotation_id=_annotation_id(
                source,
                question,
                interface.producer_version,
            ),
            source_group_id=source.observation.source_group_id,
            source_fingerprint=source.source_fingerprint,
            question=question,
            answer=answer,
            label_source="external_gt",
            source_kind=str(interface.source_kind),
            evidence_refs=list(interface.evidence_refs),
            source_field_refs=list(interface.source_field_refs),
            producer_id=interface.producer_id,
            producer_version=interface.producer_version,
            input_fingerprint=interface.input_fingerprint,
            annotation_context_digest=context_digest,
            metadata={"qc_status": interface.qc_status},
        )
    except (TypeError, ValueError) as exc:
        return VQAProducerResultV2(
            status="rejected",
            reason="external_interface_validation_failed",
            errors=[str(exc)],
        )
    return VQAProducerResultV2(
        status="produced",
        annotation=annotation,
    )


def _interface_for_task(
    source: VQASourceV2,
    task_id: str,
) -> ExternalLabelInterfaceBaseV2 | None:
    """Resolve one task's optional external source interface."""

    field_name = get_vqa_task_spec_v2(
        task_id
    ).source_interface_field
    if field_name is None:
        return None
    return getattr(source.interfaces, field_name)


def _annotation_id(
    source: VQASourceV2,
    question: VQAQuestionV2,
    producer_version: str,
) -> str:
    """Return one stable annotation id."""

    digest = hashlib.sha256(
        (
            source.source_fingerprint
            + ":"
            + question.task_id
            + ":"
            + question.template_id
            + ":"
            + producer_version
        ).encode("utf-8")
    ).hexdigest()[:20]
    return (
        f"{source.observation.source_group_id}:vqa_v2:"
        f"{question.task_id}:{digest}"
    )


__all__ = [
    "AsyncVQALabelProducer",
    "CANONICAL_DIRECTION_PRODUCER_ID",
    "CANONICAL_DIRECTION_PRODUCER_VERSION",
    "CanonicalDemonstratedDirectionProducerV2",
    "VQALabelProducer",
    "VQAProducerResultV2",
    "produce_external_interface_annotation_v2",
]
