"""Two-stage LVLM teacher prompts for canonical Reasoning V3.3."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, get_args

from mcp_server.agent.reasoning_v3_prompt_policy import (
    TEACHER_DECISION_RATIONALE_PROFILE_V3,
    TEACHER_OBSERVATION_PROFILE_V3,
)
from mcp_server.contracts.reasoning import ConfidenceLevel
from mcp_server.contracts.reasoning_dataset_v3 import ReasoningSourceV3
from mcp_server.contracts.reasoning_v3 import (
    BlockSeverityV3,
    AmbiguityTypeV3,
    CaptionRoleV3,
    ContextRoleV3,
    ContextStatusV3,
    ContextUsageV3,
    ControlStateV3,
    CrossModalRelationTypeV3,
    CrossModalStatusV3,
    DepthPositionV3,
    DesiredDirectionV3,
    DecisionReasonV3,
    DecisionRationaleStatusV3,
    InstructionTypeV3,
    InteractionRelationV3,
    LateralPositionV3,
    MotionStateV3,
    RequiredBehaviorV3,
    ReferentResolutionV3,
    RequirementCodeV3,
    RequirementPolarityV3,
    RiskLevelV3,
    SceneEntityTypeV3,
    SurfaceTypeV3,
    TemporalBasisV3,
    TrafficControlTypeV3,
    TrafficRuleV3,
    TraversabilityStateV3,
    VisibilityV3,
    LocalObjectiveV3,
)
from mcp_server.contracts.reasoning_v3_observation import (
    DECISION_RATIONALE_OBSERVATION_V3_SCHEMA_VERSION,
    REASONING_OBSERVATION_V3_SCHEMA_VERSION,
    DecisionRationaleObservationV3,
    ReasoningObservationV3,
)
from mcp_server.data.reasoning_v3_conditions import build_navigation_context_sources_v3
from mcp_server.teacher.reasoning_v3_fewshot import (
    REASONING_V3_FEWSHOT_VERSION,
    select_reasoning_v3_few_shots,
)


REASONING_V3_TEACHER_SYSTEM_PROMPT = TEACHER_OBSERVATION_PROFILE_V3.system_prompt
REASONING_V3_DECISION_RATIONALE_SYSTEM_PROMPT = TEACHER_DECISION_RATIONALE_PROFILE_V3.system_prompt


@dataclass(frozen=True)
class ReasoningObservationParseResultV3:
    """Represent parsing of one teacher observation.

    Args:
        observation: Parsed typed observation, if valid.
        errors: Parse or schema errors.
    """

    observation: ReasoningObservationV3 | None
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Return whether parsing produced a valid observation.

        Returns:
            ``True`` when an observation exists and no errors were recorded.
        """

        return self.observation is not None and not self.errors


@dataclass(frozen=True)
class DecisionRationaleParseResultV3:
    """Represent parsing of one hindsight decision rationale.

    Args:
        rationale: Parsed typed rationale, if valid.
        errors: Parse or schema errors.
    """

    rationale: DecisionRationaleObservationV3 | None
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Return whether parsing produced a valid rationale.

        Returns:
            ``True`` when a rationale exists and no error was recorded.
        """

        return self.rationale is not None and not self.errors


def build_reasoning_v3_teacher_question(
    source: ReasoningSourceV3,
    *,
    previous_errors: Iterable[str] = (),
    previous_answer: str | None = None,
) -> str:
    """Build a V3.3 semantic-observation prompt without GT leakage.

    Args:
        source: Frozen source with separated model input and private labels.
        previous_errors: Validation errors from the prior repair attempt.
        previous_answer: Prior provider response for bounded repair.

    Returns:
        Strict multimodal teacher question containing only model-visible data.
    """

    model_input = source.model_input
    catalog = [item.model_dump(mode="json", exclude_none=True) for item in model_input.evidence_catalog]
    catalog_refs = {item["ref"] for item in catalog}
    fpv_ref = "fpv.current" if "fpv.current" in catalog_refs else None
    context_sources = build_navigation_context_sources_v3(source)
    source_by_ref = {item.source_ref: item for item in context_sources}
    optional_context = [
        item for item in catalog
        if item.get("modality") in {"bev", "goal", "goal_image", "waypoint", "route", "tool", "tool_route"}
    ]

    # 每个非 FPV 上下文都预留一个账本项，模型不能静默忽略 BEV 或 route。
    context_observations = []
    for entry in optional_context:
        spec = source_by_ref.get(str(entry["ref"]))
        context_observations.append(
            {
                "source_ref": entry["ref"],
                "role": spec.role if spec is not None else "context_only",
                "usage": "ignored",
                "facts": [],
                "ignored_reason": "unresolved_context_role",
                "evidence_refs": [entry["ref"]],
            }
        )
    source_groundings = [
        {
            "source_type": item.source_type,
            "source_ref": item.source_ref,
            "directive": "unknown",
            "target_corridor_ref": None,
            "status": "insufficient_evidence",
            "reliability": "unknown",
            "ignored_reason": None,
            "evidence_refs": [item.source_ref, *([fpv_ref] if fpv_ref else [])],
        }
        for item in context_sources
    ]
    instruction_ref = "instruction.active" if "instruction.active" in catalog_refs else None
    task_semantic = {
        "applicable": instruction_ref is not None,
        "instruction_type": "unknown",
        "caption_role": "unknown",
        "inferred_objective": "unknown",
        "objective_evidence_refs": [],
        "requirements": [],
        "referents": [],
        "ambiguities": (
            [{"type": "instruction_scope", "text_span": None, "evidence_refs": [instruction_ref]}]
            if instruction_ref is not None
            else []
        ),
        "confidence": "low" if instruction_ref is not None else "unknown",
    }
    skeleton = {
        "schema_version": REASONING_OBSERVATION_V3_SCHEMA_VERSION,
        "task_semantic": task_semantic,
        "scene_grounding": {
            "applicable": fpv_ref is not None,
            "scene_facts": [],
            "entities": [],
            "traffic_controls": [],
            "context_observations": context_observations,
            "cross_modal_relations": [],
            "visibility": "unknown",
            "confidence": "unknown",
        },
        "local_traversability": {
            "applicable": fpv_ref is not None,
            "regions": [],
            "preferred_clear_corridor_ref": None,
            "visibility": "unknown",
            "local_risk": "unknown",
            "confidence": "unknown",
        },
        "navigation_context_grounding": {
            "applicable": bool(source_groundings),
            "source_groundings": source_groundings,
        },
        "interaction_and_rule_grounding": {
            "applicable": False,
            "interactions": [],
            "traffic_rules": [],
        },
    }
    visible = {
        "travel_mode": model_input.travel_mode,
        "instruction": model_input.instruction,
        "image_order": [
            {"ordinal": index + 1, "role": role}
            for index, role in enumerate(model_input.image_roles)
        ],
        "available_modalities": model_input.available_modalities,
        "model_visible_context": model_input.model_visible_context,
        "evidence_catalog": catalog,
        "required_context_refs": [item["ref"] for item in optional_context],
        "navigation_context_sources": [
            {
                "source_type": item.source_type,
                "source_ref": item.source_ref,
                "role": item.role,
            }
            for item in context_sources
        ],
    }
    parts = [
        "Annotate one UrbanNav task-and-observation record. Return exactly one JSON object matching OUTPUT_OBSERVATION_SKELETON.",
        "PROMPT_PROFILE_ID=" + TEACHER_OBSERVATION_PROFILE_V3.profile_id,
        "FEWSHOT_REGISTRY_VERSION=" + REASONING_V3_FEWSHOT_VERSION,
        "SELECTED_FEWSHOTS=" + json.dumps(
            _selected_few_shots_v3(source, stage="semantic_observation"),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "CONTROLLED_ENUMS=" + json.dumps(_teacher_enum_contract_v3(), ensure_ascii=False, separators=(",", ":")),
        "ARRAY_ITEM_SCHEMAS=" + json.dumps(_teacher_array_item_schemas_v3(), ensure_ascii=False, separators=(",", ":")),
        (
            "Use exactly the object keys shown in ARRAY_ITEM_SCHEMAS. Input evidence_refs must come from "
            "evidence_catalog. Generated entity.* and control.* ids are local object ids, not input evidence refs."
        ),
        (
            "TaskSemantic interprets the complete instruction or caption. Extract exact text spans, polarity, "
            "referents, ambiguity, caption role, and a semantic local objective. Do not derive task semantics "
            "from trajectory, action, dataset name, or demonstration stop. If the text is only a scene caption, "
            "set applicable=false and caption_role=scene_description. Few-shots teach boundaries only; never copy "
            "their entities, corridors, spans, or answers into this sample."
        ),
        (
            "SceneGrounding extracts visible entities, controls, temporal facts, and optional-context use. "
            "Every FROZEN_MODEL_INPUT.required_context_refs entry must occur exactly once in context_observations "
            "as used or ignored with a concrete ignored_reason. Never silently omit BEV, goal, route, or tool context."
        ),
        (
            "For every used BEV or other visual navigation context, emit a cross_modal_relation when a relation "
            "to FPV is resolved. BEV supplies route topology, not local clearance; FPV remains authoritative for "
            "direct, conditional, blocked, and forbidden region states."
        ),
        (
            "LocalTraversability must classify each cited corridor exactly once. direct is immediately executable; "
            "conditional requires an explicit unsatisfied condition; blocked is temporarily occupied; forbidden is "
            "terrain- or rule-invalid. preferred_clear_corridor_ref may cite only a direct region."
        ),
        (
            "Ground every FROZEN_MODEL_INPUT.navigation_context_sources entry exactly once and independently. "
            "Use aligned or locally_blocked only with a known directive and cited target corridor. Use context_only, "
            "not_visible, insufficient_evidence, or ignored instead of inventing a route relation."
        ),
        (
            "An interaction requires an entity.*, affected corridor, actor-path relation, required behavior, and "
            "supporting FPV evidence. crossing_path, approaching_path, or ahead_same_direction requires multi-frame "
            "or tracked_metadata evidence. A traffic rule requires a control.*, affected corridor, visible state, "
            "and required behavior. Do not emit generic maintain_clearance or slow labels without those relations."
        ),
        (
            "Output task_semantic but not the final composed TaskGrounding. Do not output final Decision, finish, "
            "selected execution corridor, trajectory, provenance, dataset policy, or any hidden teacher label."
        ),
        "FROZEN_MODEL_INPUT=" + json.dumps(visible, ensure_ascii=False, separators=(",", ":")),
        "OUTPUT_OBSERVATION_SKELETON=" + json.dumps(skeleton, ensure_ascii=False, separators=(",", ":")),
    ]
    errors = [str(item) for item in previous_errors if str(item)]
    if previous_answer is not None:
        parts.extend(
            [
                "The previous observation failed validation. Correct only invalid observational fields.",
                "VALIDATION_ERRORS=" + json.dumps(errors, ensure_ascii=False, separators=(",", ":")),
                "PREVIOUS_RESPONSE=" + str(previous_answer)[:30000],
            ]
        )
    return "\n".join(parts)


def parse_reasoning_v3_teacher_observation(answer: str) -> ReasoningObservationParseResultV3:
    """Parse one provider JSON response as a typed V3.3 observation.

    Args:
        answer: Raw provider text, optionally wrapped in a JSON code fence.

    Returns:
        Parsed observation and schema errors.
    """

    text = str(answer).strip()
    try:
        payload = json.loads(_strip_code_fence(text))
    except json.JSONDecodeError as exc:
        return ReasoningObservationParseResultV3(
            observation=None,
            errors=[f"teacher_invalid_json:{exc.msg}"],
        )
    if not isinstance(payload, dict):
        return ReasoningObservationParseResultV3(observation=None, errors=["teacher_output_not_object"])
    if isinstance(payload.get("observation"), dict):
        payload = payload["observation"]
    try:
        observation = ReasoningObservationV3.model_validate(payload)
    except (TypeError, ValueError) as exc:
        return ReasoningObservationParseResultV3(observation=None, errors=[f"teacher_schema:{exc}"])
    return ReasoningObservationParseResultV3(observation=observation)


def build_reasoning_v3_decision_rationale_question(
    source: ReasoningSourceV3,
    *,
    observation: ReasoningObservationV3,
    expected_outcome: Mapping[str, Any],
    previous_errors: Iterable[str] = (),
    previous_answer: str | None = None,
) -> str:
    """Build the GT-constrained hindsight-rationale prompt.

    The prompt intentionally omits raw trajectory, pixel trajectory, and
    dataset trust policy. The frozen outcome is a conclusion constraint, while
    every reason must cite the already validated observation.

    Args:
        source: Frozen source used for evidence references and few-shot retrieval.
        observation: Validated task and visual facts from the first teacher call.
        expected_outcome: Deterministically composed symbolic decision fields.
        previous_errors: Semantic errors from a prior bounded repair.
        previous_answer: Prior provider response for correction.

    Returns:
        Strict JSON-only rationale question.
    """

    fixed = source.teacher_supervision_context.get("fixed_labels") or {}
    runtime_facts = {
        "goal_reached": bool(fixed.get("goal_reached")),
        "authoritative_local_objective": str(fixed.get("authoritative_local_objective") or "unknown"),
    }
    outcome = {str(key): value for key, value in expected_outcome.items()}
    skeleton = {
        "schema_version": DECISION_RATIONALE_OBSERVATION_V3_SCHEMA_VERSION,
        "outcome_motion": str(outcome.get("motion") or "stop"),
        "status": "unsupported",
        "supporting_reasons": [],
        "rejected_alternatives": [],
        "confidence": "low",
    }
    parts = [
        "Explain one frozen UrbanNav symbolic outcome. Return exactly one JSON object matching OUTPUT_RATIONALE_SKELETON.",
        "PROMPT_PROFILE_ID=" + TEACHER_DECISION_RATIONALE_PROFILE_V3.profile_id,
        "FEWSHOT_REGISTRY_VERSION=" + REASONING_V3_FEWSHOT_VERSION,
        "SELECTED_FEWSHOTS=" + json.dumps(
            _selected_few_shots_v3(source, stage="decision_rationale"),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "CONTROLLED_REASON_ENUM=" + json.dumps(list(get_args(DecisionReasonV3)), separators=(",", ":")),
        "CONTROLLED_STATUS_ENUM=" + json.dumps(list(get_args(DecisionRationaleStatusV3)), separators=(",", ":")),
        (
            "FROZEN_OUTCOME is teacher-only conclusion supervision, not evidence. Every supporting reason and "
            "rejected alternative must cite refs already present in VALIDATED_OBSERVATION or the evidence catalog. "
            "Do not create entities, controls, corridors, requirements, interactions, traffic rules, or route relations."
        ),
        (
            "Use supported only when all material outcome claims have visible or authoritative support; partial when "
            "some but not all claims are supported; unsupported when no defensible reason exists. A demonstrated stop "
            "without goal arrival, explicit stop instruction, blockage, interaction, or rule support must remain unsupported."
        ),
        "EVIDENCE_CATALOG=" + json.dumps(
            [item.model_dump(mode="json", exclude_none=True) for item in source.model_input.evidence_catalog],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "AUTHORITATIVE_RUNTIME_FACTS=" + json.dumps(runtime_facts, ensure_ascii=False, separators=(",", ":")),
        "VALIDATED_OBSERVATION=" + observation.model_dump_json(exclude_none=True),
        "FROZEN_OUTCOME=" + json.dumps(outcome, ensure_ascii=False, separators=(",", ":")),
        "OUTPUT_RATIONALE_SKELETON=" + json.dumps(skeleton, ensure_ascii=False, separators=(",", ":")),
    ]
    errors = [str(item) for item in previous_errors if str(item)]
    if previous_answer is not None:
        parts.extend(
            [
                "The previous rationale failed validation. Correct it without changing the frozen outcome or facts.",
                "VALIDATION_ERRORS=" + json.dumps(errors, ensure_ascii=False, separators=(",", ":")),
                "PREVIOUS_RESPONSE=" + str(previous_answer)[:30000],
            ]
        )
    return "\n".join(parts)


def parse_reasoning_v3_decision_rationale(answer: str) -> DecisionRationaleParseResultV3:
    """Parse one provider JSON response as a typed V3.3 rationale.

    Args:
        answer: Raw provider text, optionally wrapped in a JSON code fence.

    Returns:
        Parsed rationale and schema errors.
    """

    text = str(answer).strip()
    try:
        payload = json.loads(_strip_code_fence(text))
    except json.JSONDecodeError as exc:
        return DecisionRationaleParseResultV3(
            rationale=None,
            errors=[f"rationale_invalid_json:{exc.msg}"],
        )
    if not isinstance(payload, dict):
        return DecisionRationaleParseResultV3(rationale=None, errors=["rationale_output_not_object"])
    if isinstance(payload.get("decision_rationale"), dict):
        payload = payload["decision_rationale"]
    try:
        rationale = DecisionRationaleObservationV3.model_validate(payload)
    except (TypeError, ValueError) as exc:
        return DecisionRationaleParseResultV3(rationale=None, errors=[f"rationale_schema:{exc}"])
    return DecisionRationaleParseResultV3(rationale=rationale)


def _teacher_enum_contract_v3() -> dict[str, list[str]]:
    """Return controlled vocabularies for LVLM-authored observation fields."""

    return {
        "instruction_type": list(get_args(InstructionTypeV3)),
        "caption_role": list(get_args(CaptionRoleV3)),
        "local_objective": list(get_args(LocalObjectiveV3)),
        "requirement_code": list(get_args(RequirementCodeV3)),
        "requirement_polarity": list(get_args(RequirementPolarityV3)),
        "referent_resolution": list(get_args(ReferentResolutionV3)),
        "ambiguity_type": list(get_args(AmbiguityTypeV3)),
        "desired_direction": list(get_args(DesiredDirectionV3)),
        "context_role": list(get_args(ContextRoleV3)),
        "context_usage": list(get_args(ContextUsageV3)),
        "context_status": list(get_args(ContextStatusV3)),
        "cross_modal_relation": list(get_args(CrossModalRelationTypeV3)),
        "cross_modal_status": list(get_args(CrossModalStatusV3)),
        "surface": list(get_args(SurfaceTypeV3)),
        "traversability_state": list(get_args(TraversabilityStateV3)),
        "block_severity": list(get_args(BlockSeverityV3)),
        "entity_type": list(get_args(SceneEntityTypeV3)),
        "lateral_position": list(get_args(LateralPositionV3)),
        "depth_position": list(get_args(DepthPositionV3)),
        "motion_state": list(get_args(MotionStateV3)),
        "control_type": list(get_args(TrafficControlTypeV3)),
        "control_state": list(get_args(ControlStateV3)),
        "interaction_relation": list(get_args(InteractionRelationV3)),
        "temporal_basis": list(get_args(TemporalBasisV3)),
        "required_behavior": list(get_args(RequiredBehaviorV3)),
        "traffic_rule": list(get_args(TrafficRuleV3)),
        "visibility": list(get_args(VisibilityV3)),
        "local_risk": list(get_args(RiskLevelV3)),
        "confidence": list(get_args(ConfidenceLevel)),
    }


def _teacher_array_item_schemas_v3() -> dict[str, dict[str, Any]]:
    """Show exact object shapes for every generated observation array."""

    return {
        "task_semantic.requirements[]": {
            "code": "avoid_stairs",
            "polarity": "avoid",
            "text_span": "avoid the stairs",
            "target_ref": None,
            "evidence_refs": ["instruction.active", "fpv.current"],
            "confidence": "high",
        },
        "task_semantic.referents[]": {
            "phrase": "the entrance on the right",
            "resolution": "resolved",
            "resolved_ref": "corridor.right",
            "evidence_refs": ["instruction.active", "fpv.current", "corridor.right"],
            "confidence": "high",
        },
        "task_semantic.ambiguities[]": {
            "type": "referent",
            "text_span": "it",
            "evidence_refs": ["instruction.active", "fpv.current"],
        },
        "scene_grounding.scene_facts[]": {
            "fact_code": "crosswalk_visible",
            "evidence_refs": ["fpv.current"],
        },
        "scene_grounding.entities[]": {
            "entity_ref": "entity.person_01",
            "type": "pedestrian",
            "lateral_position": "center",
            "depth_position": "near",
            "motion_state": "unknown",
            "evidence_refs": ["fpv.current"],
            "confidence": "high",
        },
        "scene_grounding.traffic_controls[]": {
            "control_ref": "control.crosswalk_01",
            "type": "crosswalk",
            "state": "uncontrolled",
            "lateral_position": "center",
            "evidence_refs": ["fpv.current"],
            "confidence": "high",
        },
        "scene_grounding.context_observations[]": {
            "source_ref": "bev.current",
            "role": "route_topology",
            "usage": "used",
            "facts": ["route_overlay_visible"],
            "ignored_reason": None,
            "evidence_refs": ["bev.current"],
        },
        "scene_grounding.cross_modal_relations[]": {
            "source_ref": "bev.current",
            "relation": "route_to_corridor",
            "target_ref": "corridor.center",
            "status": "aligned",
            "evidence_refs": ["bev.current", "fpv.current", "corridor.center"],
            "confidence": "high",
        },
        "local_traversability.regions[]": {
            "region_ref": "corridor.center",
            "surface": "sidewalk",
            "state": "direct",
            "required_condition": None,
            "blocker_refs": [],
            "reason": None,
            "severity": "unknown",
            "evidence_refs": ["corridor.center", "fpv.current"],
        },
        "navigation_context_grounding.source_groundings[]": {
            "source_type": "bev",
            "source_ref": "bev.current",
            "directive": "straight",
            "target_corridor_ref": "corridor.center",
            "status": "aligned",
            "reliability": "high",
            "ignored_reason": None,
            "evidence_refs": ["bev.current", "fpv.current", "corridor.center"],
        },
        "interaction_and_rule_grounding.interactions[]": {
            "actor_ref": "entity.person_01",
            "relation": "crossing_path",
            "affected_corridor_ref": "corridor.center",
            "temporal_basis": "multi_frame",
            "required_behavior": "yield",
            "rule": "pedestrian_priority",
            "evidence_refs": ["fpv.history_03", "fpv.current", "corridor.center"],
            "confidence": "high",
        },
        "interaction_and_rule_grounding.traffic_rules[]": {
            "control_ref": "control.crosswalk_01",
            "rule": "use_designated_crossing",
            "affected_corridor_ref": "corridor.center",
            "required_behavior": "slow",
            "evidence_refs": ["fpv.current", "corridor.center"],
            "confidence": "high",
        },
    }


def _selected_few_shots_v3(source: ReasoningSourceV3, *, stage: str) -> list[dict[str, Any]]:
    """Serialize deterministically selected few-shot examples for one stage."""

    examples = select_reasoning_v3_few_shots(source, stage=stage)  # type: ignore[arg-type]
    return [
        {
            "example_id": item.example_id,
            "input_summary": item.input_summary,
            "output": item.output,
        }
        for item in examples
    ]


def _strip_code_fence(value: str) -> str:
    """Remove one optional Markdown code fence from provider output."""

    if not value.startswith("```"):
        return value
    lines = value.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


__all__ = [
    "REASONING_V3_DECISION_RATIONALE_SYSTEM_PROMPT",
    "REASONING_V3_TEACHER_SYSTEM_PROMPT",
    "DecisionRationaleParseResultV3",
    "ReasoningObservationParseResultV3",
    "build_reasoning_v3_decision_rationale_question",
    "build_reasoning_v3_teacher_question",
    "parse_reasoning_v3_decision_rationale",
    "parse_reasoning_v3_teacher_observation",
]
