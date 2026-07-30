"""Prompt and parse helpers for two-stage Reasoning V4.6 review."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from pydantic import BaseModel, ValidationError

from mcp_server.contracts.reasoning_dataset_v4 import (
    ActionBlindReviewV4,
    NavObservationV4,
    ReasoningAnnotationV4,
    ReasoningSourceV4,
)
from mcp_server.teacher.reasoning_v4_review_profiles import (
    ReasoningV4ReviewProfile,
)


@dataclass(frozen=True)
class ReasoningV4ReviewParseResult:
    """Store one parsed review stage and deterministic diagnostics.

    Args:
        value: Typed stage output when parsing succeeds.
        errors: JSON or schema failures.
    """

    value: BaseModel | None
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Return whether a typed review output was parsed."""

        return self.value is not None and not self.errors


def build_reasoning_v4_blind_review_question(
    observation: NavObservationV4,
    profile: ReasoningV4ReviewProfile,
    *,
    previous_answer: str | None = None,
    previous_errors: Iterable[str] = (),
) -> str:
    """Build the strong-label-blind evidence-review request.

    Args:
        observation: Complete prompt-visible observation only.
        profile: Exact blind-review profile.
        previous_answer: Optional invalid response for bounded repair.
        previous_errors: Deterministic parser errors from that response.

    Returns:
        JSON-only reviewer question without private supervision.

    Raises:
        ValueError: If the supplied profile is not the blind stage.
    """

    if profile.stage_name != "blind_visual":
        raise ValueError("blind question requires blind_visual profile")
    payload = {
        "review_stage": profile.stage_name,
        "reviewer_visible_input": _visible_observation_payload(observation),
        "output_json_schema": profile.output_json_schema,
    }
    return _question_with_repair(
        "Assess only the attached visible evidence.",
        payload,
        previous_answer=previous_answer,
        previous_errors=previous_errors,
    )


def build_reasoning_v4_comparative_review_question(
    source: ReasoningSourceV4,
    annotation: ReasoningAnnotationV4,
    blind_assessment: ActionBlindReviewV4,
    profile: ReasoningV4ReviewProfile,
    *,
    previous_answer: str | None = None,
    previous_errors: Iterable[str] = (),
) -> str:
    """Build the four-way comparative diagnosis request.

    Args:
        source: Frozen source containing visible input and strong direction.
        annotation: Exact annotation containing groundings, proposal, and audit.
        blind_assessment: Accepted evidence-only blind pass.
        profile: Exact comparative-review profile.
        previous_answer: Optional invalid response for bounded repair.
        previous_errors: Deterministic parser errors from that response.

    Returns:
        JSON-only comparison request without candidate or trajectory payloads.

    Raises:
        ValueError: If identities, stage, or required audit inputs disagree.
    """

    if profile.stage_name != "comparative_diagnosis":
        raise ValueError(
            "comparative question requires comparative_diagnosis profile"
        )
    if (
        source.observation.source_group_id != annotation.source_group_id
        or source.source_fingerprint != annotation.source_fingerprint
    ):
        raise ValueError("review source and annotation identity mismatch")
    if (
        source.direction_supervision is None
        or annotation.decision_proposal is None
        or annotation.action_reconciliation is None
        or not annotation.grounding_outputs.complete
    ):
        raise ValueError(
            "comparative review requires strong direction, complete "
            "groundings, proposal, and action audit"
        )
    stage_provenance = {
        name: {
            "status": record.status,
            "warnings": list(record.warnings),
            "producer_kind": record.producer_kind,
            "producer_profile_id": record.producer_profile_id,
            "derivation_version": record.derivation_version,
            "prompt_policy_digest": record.prompt_policy_digest,
        }
        for name, record in annotation.stage_records.items()
    }
    payload = {
        "review_stage": profile.stage_name,
        "reviewer_visible_input": _visible_observation_payload(
            source.observation
        ),
        "blind_visual_assessment": blind_assessment.model_dump(
            mode="json",
            exclude_none=True,
        ),
        "comparison_inputs": {
            "normalized_grounding_sections": (
                annotation.grounding_outputs.model_dump(
                    mode="json",
                    exclude_none=True,
                )
            ),
            "teacher_decision_proposal": (
                annotation.decision_proposal.model_dump(
                    mode="json",
                    exclude_none=True,
                )
            ),
            "strong_direction_stamp": (
                source.direction_supervision.model_dump(
                    mode="json",
                    exclude_none=True,
                )
            ),
            "deterministic_action_audit": (
                annotation.action_reconciliation.model_dump(
                    mode="json",
                    exclude_none=True,
                )
            ),
            "stage_provenance": stage_provenance,
        },
        "output_json_schema": profile.output_json_schema,
    }
    return _question_with_repair(
        "Diagnose the suspected error source without outputting corrections.",
        payload,
        previous_answer=previous_answer,
        previous_errors=previous_errors,
    )


def parse_reasoning_v4_review_answer(
    text: str,
    profile: ReasoningV4ReviewProfile,
) -> ReasoningV4ReviewParseResult:
    """Parse one strict JSON review-stage answer.

    Args:
        text: Raw provider response.
        profile: Stage profile determining the target model.

    Returns:
        Typed output or deterministic parse diagnostics.
    """

    value = str(text or "").strip()
    if not value:
        return ReasoningV4ReviewParseResult(
            value=None,
            errors=["empty_review_answer"],
        )
    match = re.fullmatch(
        r"```(?:json)?\s*(?P<body>.*?)\s*```",
        value,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        value = match.group("body").strip()
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        return ReasoningV4ReviewParseResult(
            value=None,
            errors=[f"invalid_json:{exc}"],
        )
    if not isinstance(payload, dict):
        return ReasoningV4ReviewParseResult(
            value=None,
            errors=["review_payload_must_be_json_object"],
        )
    forbidden = _corrective_paths(payload)
    if forbidden:
        return ReasoningV4ReviewParseResult(
            value=None,
            errors=[
                "corrected_or_suggested_values_forbidden:"
                f"{forbidden[:16]}"
            ],
        )
    try:
        parsed = profile.model_type.model_validate(payload)
    except (TypeError, ValueError, ValidationError) as exc:
        return ReasoningV4ReviewParseResult(
            value=None,
            errors=[f"{profile.stage_name}_schema:{exc}"],
        )
    return ReasoningV4ReviewParseResult(value=parsed)


def _visible_observation_payload(
    observation: NavObservationV4,
) -> dict[str, Any]:
    """Return only the frozen prompt-visible observation fields."""

    return {
        "source_group_id": observation.source_group_id,
        "travel_mode": observation.travel_mode,
        "instruction": observation.instruction,
        "image_order": [
            {"ordinal": index + 1, "role": role}
            for index, role in enumerate(observation.image_roles)
        ],
        "input_modalities": observation.input_modalities.model_dump(
            mode="json"
        ),
        "direction_context": observation.direction_context.model_dump(
            mode="json"
        ),
        "model_visible_context": observation.model_visible_context,
        "visible_input_catalog": [
            item.model_dump(mode="json", exclude_none=True)
            for item in observation.evidence_catalog
        ],
    }


def _question_with_repair(
    instruction: str,
    payload: dict[str, Any],
    *,
    previous_answer: str | None,
    previous_errors: Iterable[str],
) -> str:
    """Render one stage request and optional bounded repair context."""

    repair = ""
    errors = [str(value)[:500] for value in previous_errors if str(value)]
    if previous_answer is not None or errors:
        repair = (
            "\n\nThe previous response was invalid. Repair only the requested "
            "review stage using these diagnostics:\n"
            + json.dumps(
                {
                    "previous_answer": str(previous_answer or "")[:12000],
                    "validator_errors": errors[:16],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return (
        f"{instruction} The ordered images are attached in image_order. "
        "Return exactly one JSON object matching output_json_schema.\n\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + repair
    )


def _corrective_paths(
    value: Any,
    *,
    prefix: str = "",
) -> list[str]:
    """Find keys that attempt to carry corrected or suggested values."""

    findings: list[str] = []
    if isinstance(value, dict):
        for raw_key, item in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            normalized = key.lower()
            if any(
                part in normalized
                for part in (
                    "suggested_value",
                    "corrected_value",
                    "replacement_value",
                    "new_value",
                    "corrected_label",
                    "suggested_label",
                )
            ):
                findings.append(path)
            findings.extend(_corrective_paths(item, prefix=path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(
                _corrective_paths(
                    item,
                    prefix=f"{prefix}[{index}]",
                )
            )
    return findings


__all__ = [
    "ReasoningV4ReviewParseResult",
    "build_reasoning_v4_blind_review_question",
    "build_reasoning_v4_comparative_review_question",
    "parse_reasoning_v4_review_answer",
]
