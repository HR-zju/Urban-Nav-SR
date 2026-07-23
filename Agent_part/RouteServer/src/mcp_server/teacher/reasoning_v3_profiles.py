"""Executable prompt-profile registry for minimal Reasoning V3.4 annotation."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

from mcp_server.agent.reasoning_v3_observation_validation import (
    ObservationValidationResultV3,
    validate_decision_rationale_v3,
    validate_reasoning_observation_v3,
)
from mcp_server.agent.reasoning_v3_prompt_policy import (
    PromptProfileV3,
    TEACHER_DECISION_RATIONALE_PROFILE_V3,
    TEACHER_OBSERVATION_PROFILE_V3,
)
from mcp_server.contracts.reasoning_v3_observation import (
    DecisionRationaleObservationV3,
    ReasoningObservationV3,
)
from mcp_server.teacher.reasoning_v3_teacher import (
    DecisionRationaleParseResultV3,
    ReasoningObservationParseResultV3,
    build_reasoning_v3_decision_rationale_question,
    build_reasoning_v3_teacher_question,
    parse_reasoning_v3_decision_rationale,
    parse_reasoning_v3_teacher_observation,
)


@dataclass(frozen=True)
class ObservationPromptRuntimeV3:
    """Bind one descriptor to executable observation prompt components.

    Args:
        profile: Versioned text and component identifiers.
        user_prompt_builder: Builds a model-visible teacher question.
        parser: Parses provider text into a typed observation.
        validator: Validates observation semantics against frozen evidence.
    """

    profile: PromptProfileV3
    user_prompt_builder: Callable[..., str]
    parser: Callable[[str], ReasoningObservationParseResultV3]
    validator: Callable[..., ObservationValidationResultV3]


@dataclass(frozen=True)
class DecisionRationalePromptRuntimeV3:
    """Bind the hindsight-rationale profile to executable components.

    Args:
        profile: Versioned rationale policy descriptor.
        user_prompt_builder: Builds a prompt from frozen observations and outcome.
        parser: Parses provider text into a typed rationale.
        validator: Checks that reasons only cite frozen facts.
    """

    profile: PromptProfileV3
    user_prompt_builder: Callable[..., str]
    parser: Callable[[str], DecisionRationaleParseResultV3]
    validator: Callable[..., ObservationValidationResultV3]


TEACHER_OBSERVATION_RUNTIME_PROFILE_V3 = ObservationPromptRuntimeV3(
    profile=TEACHER_OBSERVATION_PROFILE_V3,
    user_prompt_builder=build_reasoning_v3_teacher_question,
    parser=parse_reasoning_v3_teacher_observation,
    validator=validate_reasoning_observation_v3,
)

TEACHER_DECISION_RATIONALE_RUNTIME_PROFILE_V3 = DecisionRationalePromptRuntimeV3(
    profile=TEACHER_DECISION_RATIONALE_PROFILE_V3,
    user_prompt_builder=build_reasoning_v3_decision_rationale_question,
    parser=parse_reasoning_v3_decision_rationale,
    validator=validate_decision_rationale_v3,
)

OBSERVATION_PROMPT_RUNTIME_REGISTRY_V3: Mapping[str, ObservationPromptRuntimeV3] = MappingProxyType(
    {
        TEACHER_OBSERVATION_RUNTIME_PROFILE_V3.profile.profile_id: TEACHER_OBSERVATION_RUNTIME_PROFILE_V3,
    }
)

DECISION_RATIONALE_RUNTIME_REGISTRY_V3: Mapping[str, DecisionRationalePromptRuntimeV3] = MappingProxyType(
    {
        TEACHER_DECISION_RATIONALE_RUNTIME_PROFILE_V3.profile.profile_id:
            TEACHER_DECISION_RATIONALE_RUNTIME_PROFILE_V3,
    }
)


def get_observation_prompt_runtime_v3(profile_id: str) -> ObservationPromptRuntimeV3:
    """Resolve one executable observation prompt profile.

    Args:
        profile_id: Stable profile identifier.

    Returns:
        Registered executable prompt profile.

    Raises:
        ValueError: If the profile identifier is missing or unsupported.
    """

    try:
        return OBSERVATION_PROMPT_RUNTIME_REGISTRY_V3[str(profile_id)]
    except KeyError as exc:
        raise ValueError(f"unsupported observation prompt runtime: {profile_id or '<missing>'}") from exc


def validate_observation_with_runtime_v3(
    runtime: ObservationPromptRuntimeV3,
    observation: ReasoningObservationV3,
    **kwargs: object,
) -> ObservationValidationResultV3:
    """Invoke a runtime validator with an explicit typed boundary.

    Args:
        runtime: Resolved executable observation profile.
        observation: Parsed LVLM observation.
        **kwargs: Frozen evidence and condition-source arguments.

    Returns:
        Structured observation validation result.
    """

    return runtime.validator(observation, **kwargs)


def validate_rationale_with_runtime_v3(
    runtime: DecisionRationalePromptRuntimeV3,
    rationale: DecisionRationaleObservationV3,
    **kwargs: object,
) -> ObservationValidationResultV3:
    """Invoke the rationale validator through its typed runtime profile.

    Args:
        runtime: Resolved executable rationale profile.
        rationale: Parsed hindsight rationale.
        **kwargs: Frozen direction outcome, observations, and visible evidence.

    Returns:
        Structured rationale validation result.
    """

    return runtime.validator(rationale, **kwargs)


__all__ = [
    "DECISION_RATIONALE_RUNTIME_REGISTRY_V3",
    "DecisionRationalePromptRuntimeV3",
    "OBSERVATION_PROMPT_RUNTIME_REGISTRY_V3",
    "ObservationPromptRuntimeV3",
    "TEACHER_OBSERVATION_RUNTIME_PROFILE_V3",
    "TEACHER_DECISION_RATIONALE_RUNTIME_PROFILE_V3",
    "get_observation_prompt_runtime_v3",
    "validate_observation_with_runtime_v3",
    "validate_rationale_with_runtime_v3",
]
