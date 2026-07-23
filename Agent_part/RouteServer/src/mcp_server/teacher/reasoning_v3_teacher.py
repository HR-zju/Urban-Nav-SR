"""Minimal, evidence-bounded LVLM teachers for Reasoning V3.4.

Pass A labels only student-visible task, scene, traversability, and context
semantics.  Pass B receives a frozen, profile-stamped direction label and may
add only visible interaction/rule weak labels plus an evidence-bounded
rationale.  Neither pass receives raw trajectory, GPS, route geometry,
projection metadata, runtime task phase, or calibrated confidence targets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from pydantic import ValidationError

from mcp_server.contracts.reasoning_dataset_v3 import ReasoningSourceV3
from mcp_server.contracts.reasoning_v3_observation import (
    DECISION_RATIONALE_OBSERVATION_V3_SCHEMA_VERSION,
    REASONING_OBSERVATION_V3_SCHEMA_VERSION,
    DecisionRationaleObservationV3,
    ReasoningObservationV3,
)
from mcp_server.data.reasoning_v3_conditions import build_navigation_context_sources_v3


@dataclass(frozen=True)
class ReasoningObservationParseResultV3:
    """Structured parse result for the first LVLM stage."""

    observation: ReasoningObservationV3 | None
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Return whether one valid observation was parsed."""

        return self.observation is not None and not self.errors


@dataclass(frozen=True)
class DecisionRationaleParseResultV3:
    """Structured parse result for the direction-conditioned LVLM stage."""

    rationale: DecisionRationaleObservationV3 | None
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Return whether one valid interaction/rationale object was parsed."""

        return self.rationale is not None and not self.errors


def build_reasoning_v3_teacher_question(
    source: ReasoningSourceV3,
    *,
    previous_errors: Iterable[str] = (),
    previous_answer: str | None = None,
) -> str:
    """Build the model-visible V3.4 weak-observation prompt."""

    catalog = [
        item.model_dump(mode="json", exclude_none=True)
        for item in source.model_input.evidence_catalog
    ]
    catalog_refs = {str(item["ref"]) for item in catalog}
    fpv_refs = [
        str(item["ref"])
        for item in catalog
        if str(item.get("modality") or "") == "fpv"
    ]
    current_fpv_ref = (
        "fpv.current"
        if "fpv.current" in fpv_refs
        else (fpv_refs[-1] if fpv_refs else None)
    )
    context_specs = build_navigation_context_sources_v3(source)
    skeleton = {
        "schema_version": REASONING_OBSERVATION_V3_SCHEMA_VERSION,
        "task_semantic": {
            "applicable": "instruction.active" in catalog_refs,
            "requirements": [],
            "referents": [],
            "ambiguities": [],
        },
        "scene_grounding": {
            "applicable": any(ref.startswith("fpv.") for ref in catalog_refs),
            "entities": [],
            "traffic_controls": [],
        },
        "local_traversability": {
            "applicable": current_fpv_ref is not None,
            "regions": (
                [
                    {
                        "region_ref": "region.visible_path",
                        "surface": "unknown",
                        "state": "unknown",
                        "blocker_refs": [],
                        "reason": "insufficient_visible_evidence",
                        "evidence_refs": [current_fpv_ref],
                    }
                ]
                if current_fpv_ref is not None
                else []
            ),
        },
        "navigation_context_grounding": {
            "applicable": bool(context_specs),
            "source_groundings": [
                {
                    "source_type": item.source_type,
                    "source_ref": item.source_ref,
                    "usage": "ignored",
                    "relation": None,
                    "ignored_reason": "insufficient_visible_evidence",
                    "evidence_refs": [item.source_ref],
                }
                for item in context_specs
            ],
        },
    }
    visible_input = {
        "source_group_id": source.model_input.source_group_id,
        "instruction": source.model_input.instruction,
        "image_roles": source.model_input.image_roles,
        "available_modalities": source.model_input.available_modalities,
        "model_visible_context": source.model_input.model_visible_context,
        "evidence_catalog": catalog,
    }
    blocks = [
        "Annotate one Reasoning V3.4 minimal observation.",
        "MODEL_VISIBLE_INPUT:\n" + _json(visible_input),
        "OUTPUT_SKELETON:\n" + _json(skeleton),
        "OUTPUT_JSON_SCHEMA:\n" + _json(_semantic_schema(ReasoningObservationV3.model_json_schema())),
        _observation_rules(),
    ]
    repair = _repair_block(previous_errors=previous_errors, previous_answer=previous_answer)
    if repair:
        blocks.append(repair)
    blocks.append("Return exactly one JSON object and no markdown or prose.")
    return "\n\n".join(blocks)


def parse_reasoning_v3_teacher_observation(answer: str) -> ReasoningObservationParseResultV3:
    """Parse one strict V3.4 observation response."""

    payload, errors = _parse_json_object(answer)
    if payload is None:
        return ReasoningObservationParseResultV3(observation=None, errors=errors)
    try:
        observation = ReasoningObservationV3.model_validate(payload)
    except (ValidationError, ValueError, TypeError) as exc:
        return ReasoningObservationParseResultV3(
            observation=None,
            errors=[f"observation_schema:{_validation_error_text(exc)}"],
        )
    return ReasoningObservationParseResultV3(observation=observation)


def build_reasoning_v3_decision_rationale_question(
    source: ReasoningSourceV3,
    *,
    observation: ReasoningObservationV3,
    expected_outcome: dict[str, Any],
    previous_errors: Iterable[str] = (),
    previous_answer: str | None = None,
) -> str:
    """Build the direction-conditioned interaction/rationale prompt.

    ``expected_outcome`` contains only an approved profile-stamped direction
    record.  It never contains a trajectory, projected corridor, speed,
    social action, runtime objective, or route geometry.
    """

    identity = {
        key: expected_outcome.get(key)
        for key in (
            "profile_id",
            "profile_version",
            "label_index",
            "label_name",
            "choice",
            "angle_deg",
            "source",
        )
    }
    skeleton = {
        "schema_version": DECISION_RATIONALE_OBSERVATION_V3_SCHEMA_VERSION,
        "direction_profile_id": identity["profile_id"],
        "direction_profile_version": identity["profile_version"],
        "direction_label_index": identity["label_index"],
        "direction_label_name": identity["label_name"],
        "direction_choice": identity["choice"],
        "direction_angle_deg": identity["angle_deg"],
        "direction_source": identity["source"],
        "interaction_and_rule_grounding": {
            "applicable": False,
            "interactions": [],
            "traffic_rules": [],
        },
        "rationale_status": "unsupported",
        "supporting_reasons": [],
    }
    catalog = [
        item.model_dump(mode="json", exclude_none=True)
        for item in source.model_input.evidence_catalog
    ]
    payload = {
        "source_group_id": source.model_input.source_group_id,
        "instruction": source.model_input.instruction,
        "image_roles": source.model_input.image_roles,
        "evidence_catalog": catalog,
        "validated_observation": observation.model_dump(mode="json", exclude_none=True),
        "frozen_canonical_direction": expected_outcome,
    }
    blocks = [
        "Annotate visible interactions/rules and support for the frozen canonical direction.",
        "VISIBLE_INPUT_AND_FROZEN_DIRECTION:\n" + _json(payload),
        "OUTPUT_SKELETON:\n" + _json(skeleton),
        "OUTPUT_JSON_SCHEMA:\n"
        + _json(_semantic_schema(DecisionRationaleObservationV3.model_json_schema())),
        _rationale_rules(),
    ]
    repair = _repair_block(previous_errors=previous_errors, previous_answer=previous_answer)
    if repair:
        blocks.append(repair)
    blocks.append("Return exactly one JSON object and no markdown or prose.")
    return "\n\n".join(blocks)


def parse_reasoning_v3_decision_rationale(answer: str) -> DecisionRationaleParseResultV3:
    """Parse one strict V3.4 interaction/rationale response."""

    payload, errors = _parse_json_object(answer)
    if payload is None:
        return DecisionRationaleParseResultV3(rationale=None, errors=errors)
    try:
        rationale = DecisionRationaleObservationV3.model_validate(payload)
    except (ValidationError, ValueError, TypeError) as exc:
        return DecisionRationaleParseResultV3(
            rationale=None,
            errors=[f"decision_rationale_schema:{_validation_error_text(exc)}"],
        )
    return DecisionRationaleParseResultV3(rationale=rationale)


def _observation_rules() -> str:
    return "\n".join(
        [
            "RULES:",
            "1. Task fields contain only explicit requirement spans, referents, and material ambiguity.",
            "2. Do not emit instruction_type, local_objective, task phase, goal distance, or confidence.",
            "3. Entity/control/region refs are sample-local relation keys, not instance IDs, masks, boxes, or tracks.",
            "4. Every scene and traversability claim must cite at least one real fpv.* evidence ref. A local region_ref is an identifier and is not repeated as evidence.",
            "5. Traversability is a coarse LVLM weak label. Use state=unknown when evidence is insufficient.",
            "6. Navigation context only records source_type/source_ref and used/ignored semantic relation.",
            "7. Do not infer direction, route-to-corridor geometry, GPS, pose, projection, speed, or action.",
            "8. The top-level keys are exactly schema_version, task_semantic, scene_grounding, local_traversability, and navigation_context_grounding.",
            "9. entities and traffic_controls belong only inside scene_grounding; never emit either as a top-level key.",
            "10. Every code field, including surface, state reason, relation, category, and ambiguity type, is one lower_snake_case token without spaces or prose.",
            "11. Preserve exact evidence refs from the catalog.",
        ]
    )


def _rationale_rules() -> str:
    return "\n".join(
        [
            "RULES:",
            "1. Copy direction profile, index, label name, and source exactly; never change or re-bin them.",
            "2. Interaction and rule fields are LVLM weak annotations based on visible FPV plus the frozen direction.",
            "3. Do not emit a no-conflict interaction record. Use empty lists and applicable=false when none is visible.",
            "4. single_frame supports only static/occupancy relations; apparent dynamic relations require two fpv.* refs and multi_frame.",
            "5. Never claim a persistent track, TTC, overlap metric, mask, box, instance GT, calibrated confidence, speed, social action, or fallback.",
            "6. supporting_reasons may cite only validated observation/evidence refs.",
            "7. Use rationale_status=unsupported when the visible facts do not explain the frozen direction; then supporting_reasons MUST be an empty list.",
        ]
    )


def _repair_block(
    *,
    previous_errors: Iterable[str],
    previous_answer: str | None,
) -> str:
    errors = [str(item) for item in previous_errors if str(item)]
    if not errors and not previous_answer:
        return ""
    payload: dict[str, Any] = {"validation_errors": errors}
    if previous_answer:
        payload["previous_answer"] = str(previous_answer)[:20000]
    return "REPAIR_CONTEXT:\n" + _json(payload)


def _parse_json_object(value: str) -> tuple[dict[str, Any] | None, list[str]]:
    text = _strip_code_fence(str(value or "").strip())
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, [f"invalid_json:{exc.msg}"]
    if not isinstance(payload, dict):
        return None, ["not_object"]
    return payload, []


def _strip_code_fence(value: str) -> str:
    if not value.startswith("```"):
        return value
    lines = value.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _semantic_schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _semantic_schema(item)
            for key, item in value.items()
            if key not in {"title", "description"}
        }
    if isinstance(value, list):
        return [_semantic_schema(item) for item in value]
    return value


def _validation_error_text(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return ";".join(
            f"{'.'.join(str(part) for part in item['loc'])}:{item['msg']}"
            for item in exc.errors()
        )
    return str(exc)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "DecisionRationaleParseResultV3",
    "ReasoningObservationParseResultV3",
    "build_reasoning_v3_decision_rationale_question",
    "build_reasoning_v3_teacher_question",
    "parse_reasoning_v3_decision_rationale",
    "parse_reasoning_v3_teacher_observation",
]
