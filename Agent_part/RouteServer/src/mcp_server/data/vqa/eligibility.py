"""Capability and per-source eligibility gates for Navigation VQA V2."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from mcp_server.contracts.vqa_dataset_v2 import (
    VQACapabilityConfigV2,
    VQASourceV2,
)
from mcp_server.contracts.vqa_v2 import get_vqa_task_spec_v2


DEFAULT_VQA_V2_CAPABILITY_CONFIG = (
    # vqa 子包多了一层目录，配置仍归属 RouteServer 根目录的 config。
    Path(__file__).resolve().parents[4]
    / "config"
    / "vqa_v2_stage_a_capabilities.json"
)

@dataclass(frozen=True)
class VQAEligibilityResultV2:
    """Represent deterministic task eligibility.

    Args:
        eligible: Whether a producer may create an annotation.
        reason: Stable unavailable reason when ineligible.
        missing_requirements: Bounded missing input names.
    """

    eligible: bool
    reason: str | None = None
    missing_requirements: list[str] = field(default_factory=list)


def load_vqa_capabilities_v2(
    path: str | Path | None = None,
) -> VQACapabilityConfigV2:
    """Load and validate the versioned task capability policy.

    Args:
        path: Optional explicit configuration path.

    Returns:
        Frozen capability registry.
    """

    config_path = Path(
        path or DEFAULT_VQA_V2_CAPABILITY_CONFIG
    ).expanduser()
    return VQACapabilityConfigV2.model_validate(
        json.loads(config_path.read_text(encoding="utf-8"))
    )


def vqa_task_eligibility_v2(
    source: VQASourceV2,
    task_id: str,
    capabilities: VQACapabilityConfigV2,
    *,
    require_training_export: bool = True,
) -> VQAEligibilityResultV2:
    """Evaluate activation, dependency, and input availability.

    Args:
        source: Frozen VQA source.
        task_id: Exact atomic task id.
        capabilities: Versioned activation policy.
        require_training_export: Require the training export gate when true.

    Returns:
        Eligibility result without creating placeholder annotations.
    """

    capability = capabilities.capability(task_id)
    task_spec = get_vqa_task_spec_v2(task_id)
    if not capability.label_producer_ready:
        return VQAEligibilityResultV2(
            eligible=False,
            reason="label_producer_unavailable",
            missing_requirements=[
                str(capability.disabled_reason)
            ],
        )
    if (
        require_training_export
        and not capability.training_export_enabled
    ):
        return VQAEligibilityResultV2(
            eligible=False,
            reason="training_export_disabled",
            missing_requirements=[
                str(capability.disabled_reason)
            ],
        )
    observation = source.observation
    modality_flags = observation.input_modalities.model_dump(
        mode="python"
    )
    missing_modalities = [
        name
        for name in task_spec.required_modalities
        if not bool(modality_flags.get(name))
    ]
    if missing_modalities:
        return VQAEligibilityResultV2(
            eligible=False,
            reason="required_modality_unavailable",
            missing_requirements=missing_modalities,
        )
    interface_name = task_spec.source_interface_field
    if interface_name is not None and getattr(
        source.interfaces,
        interface_name,
    ) is None:
        return VQAEligibilityResultV2(
            eligible=False,
            reason="source_interface_unavailable",
            missing_requirements=[interface_name],
        )
    required_context = set(task_spec.required_context)
    if "visible_task_context" in required_context and (
        observation.instruction is None
        and not observation.model_visible_context.get("goal")
    ):
        return VQAEligibilityResultV2(
            eligible=False,
            reason="visible_task_context_unavailable",
            missing_requirements=["instruction_or_goal"],
        )
    if "canonical_direction" in required_context:
        if source.direction_supervision is None:
            return VQAEligibilityResultV2(
                eligible=False,
                reason="canonical_direction_unavailable",
                missing_requirements=["direction_supervision"],
            )
    if (
        "non_stop_direction_path" in required_context
        and source.direction_supervision is not None
        and source.direction_supervision.label_name == "stop"
    ):
        return VQAEligibilityResultV2(
            eligible=False,
            reason="stop_has_no_directional_path",
            missing_requirements=[],
        )
    if "visible_goal" in required_context:
        goal_visible = (
            observation.input_modalities.goal_image
            or observation.model_visible_context.get("goal") is not None
        )
        if not goal_visible:
            return VQAEligibilityResultV2(
                eligible=False,
                reason="visible_goal_unavailable",
                missing_requirements=["goal_or_goal_image"],
            )
    if "verified_history_fpv" in required_context:
        if not observation.temporal_evidence_eligible:
            return VQAEligibilityResultV2(
                eligible=False,
                reason="distinct_temporal_evidence_unavailable",
                missing_requirements=["verified_history_fpv"],
            )
    return VQAEligibilityResultV2(eligible=True)


__all__ = [
    "DEFAULT_VQA_V2_CAPABILITY_CONFIG",
    "VQAEligibilityResultV2",
    "load_vqa_capabilities_v2",
    "vqa_task_eligibility_v2",
]
