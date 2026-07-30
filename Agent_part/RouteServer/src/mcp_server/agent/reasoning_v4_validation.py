"""Semantic validation for compact shared-input Reasoning V4.6."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from pydantic import BaseModel

from mcp_server.agent.reasoning_v4_text import (
    parse_reasoning_target_v4,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    DirectionSupervisionV4,
    NavObservationV4,
)
from mcp_server.contracts.reasoning_v4 import (
    DecisionProposalV4,
    DecisionV4,
    InteractionConstraintGroundingV4,
    InteractionConstraintV4,
    LocalTraversabilityV4,
    NavigationContextGroundingV4,
    ReasoningTargetV4,
    SceneGroundingV4,
    TaskGroundingV4,
)


_SCENE_ENTITY_ALIASES_V4 = {
    "barrier": "barrier",
    "barriers": "barrier",
    "bollard": "bollard",
    "bollards": "bollard",
    "building": "building",
    "buildings": "building",
    "bush": "bush",
    "bushes": "bush",
    "crosswalk": "crosswalk",
    "crosswalks": "crosswalk",
    "fence": "fence",
    "fences": "fence",
    "hedge": "hedge",
    "hedges": "hedge",
    "pedestrian": "pedestrian",
    "pedestrians": "pedestrian",
    "planter": "planter",
    "planters": "planter",
    "road": "road",
    "roads": "road",
    "sign": "sign",
    "signs": "sign",
    "signage": "sign",
    "street_sign": "sign",
    "street_signs": "sign",
    "street_lamp": "street_light",
    "street_lamps": "street_light",
    "street_light": "street_light",
    "street_lights": "street_light",
    "streetlamp": "street_light",
    "streetlamps": "street_light",
    "traffic_cone": "traffic_cone",
    "traffic_cones": "traffic_cone",
    "traffic_light": "traffic_light",
    "traffic_lights": "traffic_light",
    "traffic_sign": "traffic_sign",
    "traffic_signs": "traffic_sign",
    "trash_can": "trash_can",
    "trash_cans": "trash_can",
    "tree": "tree",
    "trees": "tree",
    "turnstile": "turnstile",
    "turnstiles": "turnstile",
    "car": "vehicle",
    "cars": "vehicle",
    "sedan": "vehicle",
    "sedans": "vehicle",
    "vehicle": "vehicle",
    "vehicles": "vehicle",
    "parked_car": "vehicle",
    "parked_cars": "vehicle",
    "parked_vehicle": "vehicle",
    "parked_vehicles": "vehicle",
    "scooter": "micromobility",
    "scooters": "micromobility",
    "parked_scooter": "micromobility",
    "parked_scooters": "micromobility",
    "delivery_scooter": "micromobility",
    "motorcycle": "micromobility",
    "motorcycles": "micromobility",
    "micromobility": "micromobility",
    "parked_bicycle": "bicycle",
    "parked_bicycles": "bicycle",
    "bicycles": "bicycle",
    "bicycle": "bicycle",
}
_TRAFFIC_CONTEXT_ALIASES_V4 = {
    "active_control_visible": "traffic_control_present",
    "active_pedestrian_signal": "traffic_control_present",
    "traffic_light_visible": "traffic_control_present",
    "signalized_intersection": "traffic_control_present",
    "traffic_control_present": "traffic_control_present",
    "uncontrolled_intersection": "no_active_control_visible",
    "no_active_control_visible": "no_active_control_visible",
    "none_visible": "no_active_control_visible",
    "none": "no_active_control_visible",
    "unknown": "unknown",
    "uncertain": "unknown",
}
_CANONICAL_TRAFFIC_CONTEXTS_V4 = {
    "traffic_control_present",
    "no_active_control_visible",
    "unknown",
}
_CONSTRAINT_KIND_ALIASES_V4 = {
    "crossing_interaction": "crossing_interaction",
    "pedestrian_crossing": "crossing_interaction",
    "crosswalk_interaction": "crossing_interaction",
    "pedestrian_interaction": "social_interaction",
    "pedestrian_proximity": "social_interaction",
    "cyclist_interaction": "social_interaction",
    "social_interaction": "social_interaction",
    "static_obstacle": "local_obstacle",
    "sidewalk_obstacle": "local_obstacle",
    "local_obstacle": "local_obstacle",
    "traffic_signal": "traffic_control",
    "pedestrian_signal": "traffic_control",
    "traffic_light": "traffic_control",
    "traffic_control": "traffic_control",
}


@dataclass(frozen=True)
class ReasoningValidationResultV4:
    """Return hard failures separately from non-fatal findings.

    Args:
        errors: Hard semantic or wire failures.
        warnings: Non-fatal audit findings.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Return whether no hard validation error occurred.

        Returns:
            True when ``errors`` is empty.
        """

        return not self.errors


def validate_reasoning_section_v4(
    section_name: str,
    section: BaseModel,
    *,
    observation: NavObservationV4,
) -> ReasoningValidationResultV4:
    """Validate one independent section against public vocabularies.

    Args:
        section_name: Canonical six-section name.
        section: Parsed typed section.
        observation: Complete prompt-visible input envelope.

    Returns:
        Bounded errors and warnings.

    Raises:
        ValueError: If section name and model type disagree.
    """

    expected_types: dict[str, tuple[type[BaseModel], ...]] = {
        "TaskGrounding": (TaskGroundingV4,),
        "SceneGrounding": (SceneGroundingV4,),
        "LocalTraversability": (LocalTraversabilityV4,),
        "NavigationContextGrounding": (
            NavigationContextGroundingV4,
        ),
        "InteractionConstraintGrounding": (
            InteractionConstraintGroundingV4,
        ),
        "Decision": (DecisionProposalV4, DecisionV4),
    }
    expected = expected_types.get(section_name)
    if expected is None or not isinstance(section, expected):
        raise ValueError(
            "section/type mismatch: "
            f"{section_name}/{type(section).__name__}"
        )

    errors: list[str] = []
    warnings: list[str] = []
    action_labels = set(observation.direction_context.action_labels)
    probe_labels = set(
        observation.direction_context.affordance_probe_labels
    )

    if isinstance(section, LocalTraversabilityV4):
        for item in section.direction_affordances:
            label = str(item.direction_label)
            if label not in probe_labels:
                errors.append(
                    f"unknown_affordance_probe_label:{label}"
                )
            if label == "stop":
                errors.append(
                    "local_traversability_cannot_probe_stop"
                )

    if isinstance(section, SceneGroundingV4):
        entity_aliases = [
            str(value)
            for value in section.key_entities
            if _SCENE_ENTITY_ALIASES_V4.get(str(value), str(value))
            != str(value)
        ]
        if entity_aliases:
            errors.append(
                "noncanonical_scene_entity_labels:"
                f"{sorted(entity_aliases)}"
            )
        if str(section.traffic_context) not in (
            _CANONICAL_TRAFFIC_CONTEXTS_V4
        ):
            errors.append(
                "noncanonical_traffic_context:"
                f"{section.traffic_context}"
            )

    if isinstance(section, NavigationContextGroundingV4):
        task_direction = section.task_direction
        if (
            task_direction.direction is not None
            and str(task_direction.direction) not in action_labels
        ):
            errors.append(
                "task_direction_outside_action_profile:"
                f"{task_direction.direction}"
            )
        if (
            task_direction.status == "aligned"
            and task_direction.source == "bev_route"
            and not _has_visible_bev_direction_authority(observation)
        ):
            errors.append(
                "bev_route_task_direction_without_visible_authority"
            )
        if (
            task_direction.status == "aligned"
            and task_direction.source == "instruction"
            and not _instruction_has_explicit_direction(
                observation.instruction
            )
        ):
            errors.append(
                "instruction_task_direction_without_explicit_direction"
            )
        if (
            task_direction.status == "aligned"
            and task_direction.source not in {
                "bev_route",
                "instruction",
            }
        ):
            errors.append(
                "aligned_task_direction_without_visible_source"
            )

    if isinstance(section, InteractionConstraintGroundingV4):
        for constraint in section.constraints:
            kind = str(constraint.kind)
            if _CONSTRAINT_KIND_ALIASES_V4.get(kind, kind) != kind:
                errors.append(
                    f"noncanonical_constraint_kind:{kind}"
                )
            unknown = sorted(
                set(
                    str(label)
                    for label in constraint.affected_direction_labels
                )
                - action_labels
            )
            if unknown:
                errors.append(
                    "constraint_direction_outside_action_profile:"
                    f"{unknown}"
                )
            if constraint.status == "uncertain":
                warnings.append(
                    "uncertain_constraint_requires_review_if_action_relevant"
                )

    if isinstance(section, (DecisionProposalV4, DecisionV4)):
        label = str(section.selected_action_label)
        if label not in action_labels:
            errors.append(
                f"decision_action_outside_profile:{label}"
            )

    return ReasoningValidationResultV4(
        errors=_unique(errors),
        warnings=_unique(warnings),
    )


def sanitize_reasoning_section_evidence_v4(
    section_name: str,
    section: BaseModel,
    *,
    observation: NavObservationV4,
) -> tuple[BaseModel, list[str]]:
    """Normalize bounded vocabularies without adding semantic fields.

    V4.6 section models contain no evidence refs or temporal claims.  This
    boundary only folds a small frozen set of Scene and Constraint synonyms so
    teacher wording cannot fragment downstream labels.

    Args:
        section_name: Canonical section name.
        section: Typed section.
        observation: Prompt-visible observation.

    Returns:
        Canonical section and bounded normalization warnings.

    Raises:
        ValueError: If section name and model type disagree.
    """

    normalized = section
    warnings: list[str] = []
    if isinstance(section, SceneGroundingV4):
        entities = _unique(
            _SCENE_ENTITY_ALIASES_V4.get(str(value), str(value))
            for value in section.key_entities
        )
        traffic_context = _TRAFFIC_CONTEXT_ALIASES_V4.get(
            str(section.traffic_context),
            "unknown",
        )
        normalized = SceneGroundingV4(
            scene_type=section.scene_type,
            walkable_support=section.walkable_support,
            key_entities=entities,
            traffic_context=traffic_context,
        )
        if normalized != section:
            warnings.append("scene_vocabulary_normalized")
    elif isinstance(section, InteractionConstraintGroundingV4):
        normalized = InteractionConstraintGroundingV4(
            constraints=[
                InteractionConstraintV4(
                    kind=_CONSTRAINT_KIND_ALIASES_V4.get(
                        str(item.kind),
                        str(item.kind),
                    ),
                    status=item.status,
                    effect=item.effect,
                    affected_direction_labels=(
                        item.affected_direction_labels
                    ),
                )
                for item in section.constraints
            ]
        )
        if normalized != section:
            warnings.append("constraint_vocabulary_normalized")
    validate_reasoning_section_v4(
        section_name,
        normalized,
        observation=observation,
    )
    return normalized, warnings


def validate_reasoning_target_v4(
    target: ReasoningTargetV4,
    *,
    observation: NavObservationV4,
    direction_supervision: DirectionSupervisionV4 | None = None,
) -> ReasoningValidationResultV4:
    """Validate all six sections and optional strong-direction ownership.

    Args:
        target: Typed V4.6 target.
        observation: Complete prompt-visible input.
        direction_supervision: Optional upstream deterministic direction.

    Returns:
        Aggregated bounded findings.
    """

    trace = target.reasoning
    sections: tuple[tuple[str, BaseModel], ...] = (
        ("TaskGrounding", trace.task_grounding),
        ("SceneGrounding", trace.scene_grounding),
        ("LocalTraversability", trace.local_traversability),
        (
            "NavigationContextGrounding",
            trace.navigation_context_grounding,
        ),
        (
            "InteractionConstraintGrounding",
            trace.interaction_constraint_grounding,
        ),
        ("Decision", trace.decision),
    )
    errors: list[str] = []
    warnings: list[str] = []
    for section_name, section in sections:
        result = validate_reasoning_section_v4(
            section_name,
            section,
            observation=observation,
        )
        errors.extend(result.errors)
        warnings.extend(result.warnings)
    if (
        direction_supervision is not None
        and str(trace.decision.selected_action_label)
        != str(direction_supervision.label_name)
    ):
        errors.append(
            "decision_selected_action_must_equal_strong_direction"
        )
    errors.extend(
        decision_constraint_errors_v4(
            trace.decision,
            trace.interaction_constraint_grounding,
        )
    )
    return ReasoningValidationResultV4(
        errors=_unique(errors),
        warnings=_unique(warnings),
    )


def validate_reasoning_training_text_v4(
    assistant_text: str,
    *,
    expected_target: ReasoningTargetV4,
) -> ReasoningValidationResultV4:
    """Validate canonical text round-trip equality.

    Args:
        assistant_text: Rendered assistant target.
        expected_target: Exact typed target expected after parsing.

    Returns:
        Wire/schema mismatch findings.
    """

    parsed = parse_reasoning_target_v4(assistant_text)
    if not parsed.valid or parsed.target is None:
        return ReasoningValidationResultV4(errors=parsed.errors)
    if parsed.target != expected_target:
        return ReasoningValidationResultV4(
            errors=["assistant_text_target_mismatch"],
        )
    return ReasoningValidationResultV4()


def decision_constraint_errors_v4(
    decision: DecisionProposalV4 | DecisionV4,
    constraints: InteractionConstraintGroundingV4,
) -> list[str]:
    """Return deterministic action/constraint consistency failures.

    Only ``applicable`` constraints affect the tuple.  A strong moving action
    under a global stop is a review conflict and must not be silently rewritten.
    An applicable avoid subset is not a forbidden-direction set: choosing a
    direction in that subset is valid exactly with ``slow/detour``.  Choosing
    stop safely satisfies every avoid constraint.

    Args:
        decision: Action tuple to audit.
        constraints: Independently annotated behavior constraints.

    Returns:
        Stable error codes.
    """

    label = str(decision.selected_action_label)
    relevant = [
        constraint
        for constraint in constraints.constraints
        if (
            constraint.status == "applicable"
            and (
                not constraint.affected_direction_labels
                or label
                in {
                    str(value)
                    for value
                    in constraint.affected_direction_labels
                }
            )
        )
    ]
    effects = {
        constraint.effect for constraint in relevant
    }
    if label == "stop":
        return (
            []
            if decision.speed == "stop"
            and decision.response == "stop"
            else ["stop_action_tuple_conflict"]
        )
    if "stop" in effects:
        return [
            "strong_moving_action_conflicts_with_stop_constraint"
        ]
    expected = (
        ("slow", "detour")
        if "avoid" in effects
        else ("slow", "yield")
        if "yield" in effects
        else ("slow", "slow")
        if "slow" in effects
        else ("normal", "proceed")
    )
    if (decision.speed, decision.response) != expected:
        return [
            "constraint_effect_tuple_conflict:"
            f"expected={expected[0]}/{expected[1]}"
        ]
    return []


def applicable_constraint_effects_v4(
    selected_action_label: str,
    constraints: InteractionConstraintGroundingV4,
) -> tuple[str, ...]:
    """Return applicable effects for one action in deterministic order.

    Args:
        selected_action_label: Public action label.
        constraints: Independently annotated constraints.

    Returns:
        Unique effects ordered by ``stop, avoid, yield, slow`` priority.
    """

    effects = {
        constraint.effect
        for constraint in constraints.constraints
        if (
            constraint.status == "applicable"
            and (
                not constraint.affected_direction_labels
                or selected_action_label
                in {
                    str(value)
                    for value
                    in constraint.affected_direction_labels
                }
            )
        )
    }
    return tuple(
        effect
        for effect in ("stop", "avoid", "yield", "slow")
        if effect in effects
    )


def _has_visible_bev_direction_authority(
    observation: NavObservationV4,
) -> bool:
    """Return whether one visible BEV may support route direction."""

    return any(
        entry.modality == "bev"
        and entry.bev_route_evidence is not None
        and entry.bev_route_evidence.route_overlay_visible
        and entry.bev_route_evidence.route_direction_authority
        == "visual_conditional"
        and entry.bev_route_evidence.frame != "unknown"
        and entry.bev_route_evidence.ego_forward != "unknown"
        for entry in observation.evidence_catalog
    )


def _instruction_has_explicit_direction(
    instruction: str | None,
) -> bool:
    """Return a conservative lexical explicit-direction signal."""

    if instruction is None:
        return False
    normalized = instruction.casefold()
    markers = (
        "left",
        "right",
        "straight",
        "forward",
        "turn",
        "左",
        "右",
        "直行",
        "前方",
        "转",
    )
    return any(marker in normalized for marker in markers)


def _unique(values: Iterable[str]) -> list[str]:
    """Return stable duplicate-free strings."""

    return list(dict.fromkeys(str(value) for value in values))


__all__ = [
    "ReasoningValidationResultV4",
    "applicable_constraint_effects_v4",
    "decision_constraint_errors_v4",
    "sanitize_reasoning_section_evidence_v4",
    "validate_reasoning_section_v4",
    "validate_reasoning_target_v4",
    "validate_reasoning_training_text_v4",
]
