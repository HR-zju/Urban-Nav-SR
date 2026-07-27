"""Prompt construction and strict parsing for Reasoning V4 section teachers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from pydantic import BaseModel, ValidationError

from mcp_server.contracts.reasoning_dataset_v4 import (
    GroundingOutputsV4,
    NavObservationV4,
    TeacherActionConstraintV4,
)
from mcp_server.teacher.reasoning_v4_profiles import (
    ReasoningV4TeacherProfile,
)


@dataclass(frozen=True)
class ReasoningV4SectionParseResult:
    """Return one parsed section with bounded diagnostics.

    Args:
        section: Typed section value when parsing succeeds.
        errors: JSON or schema failures.
        warnings: Reserved non-fatal parser findings.
    """

    section: BaseModel | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Return whether a typed section was parsed without errors."""

        return self.section is not None and not self.errors


def build_reasoning_v4_section_question(
    observation: NavObservationV4,
    profile: ReasoningV4TeacherProfile,
    *,
    groundings: GroundingOutputsV4 | None = None,
    teacher_action_constraint: TeacherActionConstraintV4 | None = None,
    previous_answer: str | None = None,
    previous_errors: Iterable[str] = (),
) -> str:
    """Build a section-specific question from the complete visible input.

    Args:
        observation: Complete prompt-visible V4 input. Private source
            sidecars cannot cross this function boundary.
        profile: Requested section profile.
        groundings: Five validated groundings required only for Decision.
        teacher_action_constraint: Optional private demonstrated direction,
            legal only in the Decision teacher prompt.
        previous_answer: Optional invalid provider response for repair.
        previous_errors: Deterministic errors from the previous attempt.

    Returns:
        JSON-only provider question with the exact output schema.

    Raises:
        ValueError: If Decision is requested without complete groundings.
    """

    visible_input: dict[str, Any] = {
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
        "direction_output_space": (
            observation.direction_output_space.model_dump(mode="json")
        ),
        "temporal_evidence_eligible": observation.temporal_evidence_eligible,
        "model_visible_context": observation.model_visible_context,
        "evidence_catalog": [
            item.model_dump(mode="json", exclude_none=True)
            for item in observation.evidence_catalog
        ],
    }
    payload: dict[str, Any] = {
        "requested_section": profile.section_name,
        "complete_visible_input": visible_input,
        "output_json_schema": profile.output_json_schema,
    }
    if profile.section_name == "Decision":
        if groundings is None or not groundings.complete:
            raise ValueError(
                "Decision teacher prompt requires five validated groundings"
            )
        payload["validated_grounding_sections"] = groundings.model_dump(
            mode="json",
            exclude_none=True,
        )
        if teacher_action_constraint is not None:
            demonstrated = str(
                teacher_action_constraint.demonstrated_direction_label
            )
            if demonstrated not in {
                str(value)
                for value in observation.direction_output_space.allowed_labels
            }:
                raise ValueError(
                    "teacher demonstrated direction is outside the public "
                    "direction output space"
                )
            payload["teacher_only_action_constraint"] = (
                teacher_action_constraint.model_dump(mode="json")
            )
    elif groundings is not None:
        raise ValueError(
            "grounding section prompts must not consume other section outputs"
        )
    elif teacher_action_constraint is not None:
        raise ValueError(
            "grounding prompts cannot consume teacher action constraint"
        )

    repair = ""
    errors = [str(item)[:500] for item in previous_errors if str(item)]
    if previous_answer is not None or errors:
        repair_payload = {
            "previous_answer": str(previous_answer or "")[:12000],
            "validator_errors": errors[:16],
        }
        repair = (
            "\n\nThe previous response was invalid. Repair only the requested "
            "section using these diagnostics:\n"
            + json.dumps(
                repair_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return (
        "Annotate the requested Reasoning V4 section. The ordered images are "
        "attached to this request in image_order. Return exactly one JSON "
        "object matching output_json_schema.\n\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + repair
    )


def parse_reasoning_v4_section_answer(
    text: str,
    profile: ReasoningV4TeacherProfile,
) -> ReasoningV4SectionParseResult:
    """Parse one teacher response into the profile's section type.

    Args:
        text: Raw provider response.
        profile: Section profile that determines the target model.

    Returns:
        Typed section or deterministic parse errors.
    """

    value = str(text or "").strip()
    if not value:
        return ReasoningV4SectionParseResult(
            section=None,
            errors=["empty_teacher_answer"],
        )
    value = _strip_single_json_fence(value)
    if value.startswith(f"<{profile.section_name}>") and value.endswith(
        f"</{profile.section_name}>"
    ):
        prefix = f"<{profile.section_name}>"
        suffix = f"</{profile.section_name}>"
        value = value[len(prefix) : -len(suffix)].strip()
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        return ReasoningV4SectionParseResult(
            section=None,
            errors=[f"invalid_json:{exc}"],
        )
    if not isinstance(payload, dict):
        return ReasoningV4SectionParseResult(
            section=None,
            errors=["section_payload_must_be_json_object"],
        )
    forbidden_numeric = _forbidden_numeric_uncertainty_fields(payload)
    if forbidden_numeric:
        return ReasoningV4SectionParseResult(
            section=None,
            errors=[
                "numeric_confidence_or_cost_forbidden:"
                f"{forbidden_numeric}"
            ],
        )
    try:
        section = profile.model_type.model_validate(payload)
    except (TypeError, ValueError, ValidationError) as exc:
        return ReasoningV4SectionParseResult(
            section=None,
            errors=[f"{profile.section_name}_schema:{exc}"],
        )
    return ReasoningV4SectionParseResult(section=section)


def _strip_single_json_fence(value: str) -> str:
    """Strip one whole-response JSON markdown fence, if present."""

    match = re.fullmatch(
        r"```(?:json)?\s*(?P<body>.*?)\s*```",
        value,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return match.group("body").strip() if match else value


def _forbidden_numeric_uncertainty_fields(
    value: Any,
    *,
    prefix: str = "",
) -> list[str]:
    """Find numeric confidence/cost fields in a teacher payload."""

    findings: list[str] = []
    if isinstance(value, dict):
        for raw_key, item in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            normalized = key.lower()
            if (
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and (
                    "confidence" in normalized
                    or normalized == "cost"
                    or normalized.endswith("_cost")
                )
            ):
                findings.append(path)
            findings.extend(
                _forbidden_numeric_uncertainty_fields(
                    item,
                    prefix=path,
                )
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(
                _forbidden_numeric_uncertainty_fields(
                    item,
                    prefix=f"{prefix}[{index}]",
                )
            )
    return findings
