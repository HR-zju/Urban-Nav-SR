"""Section-specific image routing for Reasoning V4 annotation teachers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from mcp_server.contracts.reasoning_dataset_v4 import (
    InputModalitiesV4,
    NavObservationV4,
)


REASONING_V4_ALL_VISIBLE_IMAGES_POLICY_ID = (
    "reasoning_v4_section_images_all_visible_v1"
)
REASONING_V4_SECTION_ROUTED_VISUAL_POLICY_ID = (
    "reasoning_v4_section_images_routed_visual_v1"
)
_SECTION_NAMES = (
    "TaskGrounding",
    "SceneGrounding",
    "LocalTraversability",
    "NavigationContextGrounding",
    "InteractionConstraintGrounding",
    "Decision",
)
_IMAGE_MODALITIES = frozenset({"fpv", "bev", "goal_image"})
_VALID_SELECTORS = frozenset(
    {
        "all_visible",
        "temporal_fpv",
        "current_fpv",
        "current_bev_if_available",
    }
)


@dataclass(frozen=True)
class SectionImageSelection:
    """Store the exact ordered images exposed to one annotation section.

    Args:
        section_name: Canonical Reasoning V4 section.
        images: Selected image paths in provider order.
        image_roles: Selected evidence roles in the same order.
    """

    section_name: str
    images: tuple[str, ...]
    image_roles: tuple[str, ...]


@dataclass(frozen=True)
class SectionImagePolicy:
    """Route prompt-visible images according to section responsibility.

    Args:
        policy_id: Stable annotation-policy identity.
        selectors: Ordered role selectors for every canonical section.
        policy_digest: SHA-256 over the complete routing contract.
    """

    policy_id: str
    selectors: Mapping[str, tuple[str, ...]]
    policy_digest: str

    def select(
        self,
        observation: NavObservationV4,
        section_name: str,
    ) -> SectionImageSelection:
        """Select exact images for one section.

        Args:
            observation: Full frozen source observation.
            section_name: Canonical Reasoning V4 section.

        Returns:
            Ordered section image selection.

        Raises:
            ValueError: If the section is unsupported or routing removes the
                required current FPV image.
        """

        try:
            selectors = self.selectors[str(section_name)]
        except KeyError as exc:
            raise ValueError(
                "unsupported Reasoning V4 image-policy section: "
                f"{section_name!r}"
            ) from exc
        pairs = list(zip(observation.images, observation.image_roles))
        selected: list[tuple[str, str]] = []
        for selector in selectors:
            if selector == "all_visible":
                candidates = pairs
            elif selector == "temporal_fpv":
                candidates = [
                    pair
                    for pair in pairs
                    if pair[1].startswith("fpv.")
                ]
            elif selector == "current_fpv":
                candidates = [
                    pair for pair in pairs if pair[1] == "fpv.current"
                ]
            elif selector == "current_bev_if_available":
                candidates = [
                    pair for pair in pairs if pair[1] == "bev.current"
                ]
            else:
                raise ValueError(
                    "unsupported Reasoning V4 image selector: "
                    f"{selector!r}"
                )
            for pair in candidates:
                if pair not in selected:
                    selected.append(pair)
        roles = tuple(role for _, role in selected)
        if roles.count("fpv.current") != 1:
            raise ValueError(
                "section image policy must expose exactly one fpv.current: "
                f"policy={self.policy_id} section={section_name} roles={roles}"
            )
        return SectionImageSelection(
            section_name=str(section_name),
            images=tuple(image for image, _ in selected),
            image_roles=roles,
        )

    def project_observation(
        self,
        observation: NavObservationV4,
        section_name: str,
    ) -> NavObservationV4:
        """Build a validated section-visible observation.

        Args:
            observation: Full frozen source observation.
            section_name: Canonical Reasoning V4 section.

        Returns:
            Observation containing only selected image evidence while
            preserving non-image instruction, goal, and waypoint evidence.
        """

        selection = self.select(observation, section_name)
        role_set = set(selection.image_roles)
        evidence_catalog = [
            entry
            for entry in observation.evidence_catalog
            if entry.modality not in _IMAGE_MODALITIES
            or str(entry.ref) in role_set
        ]
        modalities = InputModalitiesV4(
            history_fpv=any(
                role.startswith("fpv.history_")
                for role in selection.image_roles
            ),
            instruction=observation.input_modalities.instruction,
            bev=any(
                role.startswith("bev.")
                for role in selection.image_roles
            ),
            goal_image=any(
                role.startswith("goal.image_")
                for role in selection.image_roles
            ),
            pixel_goal=observation.input_modalities.pixel_goal,
            waypoint=observation.input_modalities.waypoint,
        )
        payload = observation.model_dump(mode="python")
        payload.update(
            {
                "images": list(selection.images),
                "image_roles": list(selection.image_roles),
                "evidence_catalog": [
                    entry.model_dump(mode="python")
                    for entry in evidence_catalog
                ],
                "input_modalities": modalities.model_dump(mode="python"),
                # section 投影移除历史帧时，不能继续声明时序证据可用。
                "temporal_evidence_eligible": bool(
                    observation.temporal_evidence_eligible
                    and modalities.history_fpv
                ),
            }
        )
        return NavObservationV4.model_validate(payload)


def _build_policy(
    policy_id: str,
    selectors: Mapping[str, tuple[str, ...]],
) -> SectionImagePolicy:
    """Build and validate one immutable section image policy."""

    normalized = {
        str(section): tuple(str(value) for value in values)
        for section, values in selectors.items()
    }
    if set(normalized) != set(_SECTION_NAMES):
        raise ValueError(
            "section image policy must define exactly the canonical sections"
        )
    unsupported = {
        value
        for values in normalized.values()
        for value in values
        if value not in _VALID_SELECTORS
    }
    if unsupported:
        raise ValueError(
            "unsupported section image selectors: "
            f"{sorted(unsupported)}"
        )
    payload = {
        "policy_id": policy_id,
        "selection_contract_version": "reasoning_v4_image_selection_v1",
        "selectors": normalized,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return SectionImagePolicy(
        policy_id=policy_id,
        selectors=MappingProxyType(normalized),
        policy_digest=digest,
    )


REASONING_V4_SECTION_IMAGE_POLICIES: Mapping[
    str,
    SectionImagePolicy,
] = MappingProxyType(
    {
        REASONING_V4_ALL_VISIBLE_IMAGES_POLICY_ID: _build_policy(
            REASONING_V4_ALL_VISIBLE_IMAGES_POLICY_ID,
            {
                section_name: ("all_visible",)
                for section_name in _SECTION_NAMES
            },
        ),
        REASONING_V4_SECTION_ROUTED_VISUAL_POLICY_ID: _build_policy(
            REASONING_V4_SECTION_ROUTED_VISUAL_POLICY_ID,
            {
                "TaskGrounding": (
                    "current_fpv",
                    "current_bev_if_available",
                ),
                "SceneGrounding": ("temporal_fpv",),
                "LocalTraversability": ("current_fpv",),
                "NavigationContextGrounding": (
                    "current_fpv",
                    "current_bev_if_available",
                ),
                "InteractionConstraintGrounding": ("temporal_fpv",),
                "Decision": (
                    "current_fpv",
                    "current_bev_if_available",
                ),
            },
        ),
    }
)


def get_reasoning_v4_section_image_policy(
    policy_id: str,
) -> SectionImagePolicy:
    """Resolve one registered Reasoning V4 image policy.

    Args:
        policy_id: Exact registered policy identity.

    Returns:
        Immutable section image policy.

    Raises:
        ValueError: If the policy is unsupported.
    """

    try:
        return REASONING_V4_SECTION_IMAGE_POLICIES[str(policy_id)]
    except KeyError as exc:
        raise ValueError(
            "unsupported Reasoning V4 section image policy: "
            f"{policy_id!r}"
        ) from exc


__all__ = [
    "REASONING_V4_ALL_VISIBLE_IMAGES_POLICY_ID",
    "REASONING_V4_SECTION_IMAGE_POLICIES",
    "REASONING_V4_SECTION_ROUTED_VISUAL_POLICY_ID",
    "SectionImagePolicy",
    "SectionImageSelection",
    "get_reasoning_v4_section_image_policy",
]
