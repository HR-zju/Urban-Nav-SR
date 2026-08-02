"""Build deterministic, annotation-only Reasoning V4.6 action audits."""

from __future__ import annotations

from typing import Any

from mcp_server.agent.reasoning_v4_validation import (
    applicable_constraint_effects_v4,
)
from mcp_server.contracts.direction_profile import (
    affordance_probe_candidates_for_action,
    approved_affordance_probe_profiles,
    approved_stage_a_direction_profiles,
    direction_comparison_facts,
    resolve_stage_a_direction_profile,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    ActionComparisonFactsV4,
    ActionProbeProjectionV4,
    ActionReconciliationAuditV4,
    ActionReconciliationChecksV4,
    GroundingOutputsV4,
    ReasoningSourceV4,
    reasoning_payload_digest_v4,
)
from mcp_server.contracts.reasoning_v4 import DecisionProposalV4


ACTION_RECONCILIATION_POLICY_VERSION_V4 = (
    "urbannav_action_reconciliation_policy_v4"
)
ACTION_RECONCILIATION_ANGLE_REVIEW_THRESHOLD_DEG_V4 = 20.0
ACTION_RECONCILIATION_REVIEW_ADMISSION_POLICY_VERSION_V4 = (
    "urbannav_action_reconciliation_review_admission_v2"
)

_TRIGGER_POLICY = {
    "proposal_delta": (
        "proposal_direction_delta_exceeds_profile_tolerance"
    ),
    "proposal_sign": "proposal_direction_sign_conflict",
    "proposal_stop_move": "proposal_stop_move_mismatch",
    "probe_unsupported": "strong_action_probe_unsupported",
    "probe_mixed": "strong_action_probe_mixed",
    "probe_unknown": "strong_action_probe_unknown",
    "constraint": "strong_action_constraint_conflict",
    "tuple": "proposal_action_tuple_conflict",
}
_ADVISORY_TRIGGER_CODES = frozenset(
    {
        _TRIGGER_POLICY["probe_unsupported"],
        _TRIGGER_POLICY["probe_mixed"],
        _TRIGGER_POLICY["probe_unknown"],
    }
)


def action_reconciliation_policy_digest_v4() -> str:
    """Return the deterministic reconciliation-policy digest.

    Returns:
        SHA-256 over policy and direction registries.
    """

    return _canonical_digest(
        {
            "policy_version": (
                ACTION_RECONCILIATION_POLICY_VERSION_V4
            ),
            "human_review_angle_threshold_deg": (
                ACTION_RECONCILIATION_ANGLE_REVIEW_THRESHOLD_DEG_V4
            ),
            "review_admission_policy_version": (
                ACTION_RECONCILIATION_REVIEW_ADMISSION_POLICY_VERSION_V4
            ),
            "advisory_trigger_codes": sorted(_ADVISORY_TRIGGER_CODES),
            "trigger_policy": _TRIGGER_POLICY,
            "action_profiles": (
                approved_stage_a_direction_profiles()
            ),
            "affordance_profiles": (
                approved_affordance_probe_profiles()
            ),
        }
    )


def build_action_reconciliation_audit_v4(
    source: ReasoningSourceV4,
    groundings: GroundingOutputsV4,
    decision_proposal: DecisionProposalV4,
) -> ActionReconciliationAuditV4 | None:
    """Build one normalized proposal-versus-strong audit.

    Args:
        source: Frozen observation and optional strong supervision.
        groundings: Five independent grounding sections.
        decision_proposal: Blind LVLM Decision proposal.

    Returns:
        Immutable audit when strong supervision exists, else ``None``.

    Raises:
        ValueError: If grounding outputs are incomplete.
    """

    if not groundings.complete:
        raise ValueError(
            "action reconciliation requires complete groundings"
        )
    direction = source.direction_supervision
    if direction is None:
        return None
    profile_id = source.observation.direction_context.action_profile_id
    profile = resolve_stage_a_direction_profile(profile_id)
    proposal_label = str(
        decision_proposal.selected_action_label
    )
    strong_label = str(direction.label_name)
    comparison_raw = direction_comparison_facts(
        profile_id,
        proposal_label,
        strong_label,
    )
    comparison = ActionComparisonFactsV4(
        angular_delta_deg=comparison_raw.angular_delta_deg,
        stop_move_mismatch=comparison_raw.stop_move_mismatch,
        sign_conflict=comparison_raw.sign_conflict,
    )
    projections = _probe_projections_v4(
        profile_id,
        strong_label,
        groundings,
    )
    proposal_relation = _proposal_relation_v4(
        comparison,
        tolerance_deg=float(
            profile["intent_compatibility_tolerance_deg"]
        ),
    )
    constraint_consistency = _constraint_check_v4(
        groundings,
        strong_action=strong_label,
    )
    checks = ActionReconciliationChecksV4(
        proposal_strong_relation=proposal_relation,
        probe_support=_probe_support_check_v4(projections),
        constraint_effect_consistency=constraint_consistency,
        action_tuple_consistency=(
            "passed"
            if _proposal_tuple_is_consistent_v4(decision_proposal)
            else "conflict"
        ),
    )
    trigger_codes = _review_trigger_codes_v4(
        checks,
        comparison=comparison,
        proposal_action=proposal_label,
        strong_action=strong_label,
    )
    proposal_digest = _canonical_digest(
        decision_proposal.model_dump(
            mode="json",
            exclude_none=True,
        )
    )
    groundings_digest = _canonical_digest(
        groundings.model_dump(
            mode="json",
            exclude_none=True,
        )
    )
    audit_id = _canonical_digest(
        {
            "schema": ACTION_RECONCILIATION_POLICY_VERSION_V4,
            "source_fingerprint": source.source_fingerprint,
            "proposal_digest": proposal_digest,
            "groundings_digest": groundings_digest,
            "comparison": comparison.model_dump(mode="json"),
            "probe_projections": [
                projection.model_dump(
                    mode="json",
                    exclude_none=True,
                )
                for projection in projections
            ],
            "checks": checks.model_dump(mode="json"),
            "review_trigger_codes": trigger_codes,
        }
    )
    return ActionReconciliationAuditV4(
        audit_id=audit_id,
        source_fingerprint=source.source_fingerprint,
        proposal_digest=proposal_digest,
        groundings_digest=groundings_digest,
        comparison=comparison,
        probe_projections=projections,
        checks=checks,
        review_trigger_codes=trigger_codes,
    )


def validate_action_reconciliation_audit_v4(
    source: ReasoningSourceV4,
    groundings: GroundingOutputsV4,
    decision_proposal: DecisionProposalV4,
    audit: ActionReconciliationAuditV4,
) -> None:
    """Require a stored audit to equal deterministic reconstruction.

    Args:
        source: Frozen source joined to annotation.
        groundings: Parent grounding outputs.
        decision_proposal: Parent visual grounding-conditioned proposal.
        audit: Stored immutable audit.

    Raises:
        ValueError: If reconstruction is absent or differs.
    """

    expected = build_action_reconciliation_audit_v4(
        source,
        groundings,
        decision_proposal,
    )
    if expected is None:
        raise ValueError(
            "audit requires strong direction supervision"
        )
    if audit != expected:
        raise ValueError(
            "stored action audit does not match inputs"
        )


def action_reconciliation_review_required_v4(
    audit: ActionReconciliationAuditV4,
) -> bool:
    """Return whether deterministic triggers require human review.

    Args:
        audit: Canonical action audit.

    Returns:
        True when at least one blocking action-conflict trigger exists. Sparse
        traversability probes remain auditable but cannot, by themselves,
        override profile tolerance or force human action adjudication.
    """

    return bool(action_reconciliation_blocking_trigger_codes_v4(audit))


def action_reconciliation_blocking_trigger_codes_v4(
    audit: ActionReconciliationAuditV4,
) -> tuple[str, ...]:
    """Return action-conflict codes that block automatic admission.

    Args:
        audit: Canonical action audit.

    Returns:
        Sorted trigger codes excluding sparse grounding-probe advisories.
    """

    return tuple(
        sorted(
            str(value)
            for value in audit.review_trigger_codes
            if str(value) not in _ADVISORY_TRIGGER_CODES
        )
    )


def action_reconciliation_advisory_trigger_codes_v4(
    audit: ActionReconciliationAuditV4,
) -> tuple[str, ...]:
    """Return non-blocking sparse grounding-probe QC codes.

    Args:
        audit: Canonical action audit.

    Returns:
        Sorted advisory codes retained for reporting or automatic relabeling.
    """

    return tuple(
        sorted(
            str(value)
            for value in audit.review_trigger_codes
            if str(value) in _ADVISORY_TRIGGER_CODES
        )
    )


def action_reconciliation_review_priority_v4(
    audit: ActionReconciliationAuditV4,
) -> str:
    """Derive review priority without persisting mutable state.

    Args:
        audit: Canonical action audit.

    Returns:
        ``critical``, ``high``, ``normal``, or ``not_required``.
    """

    codes = set(action_reconciliation_blocking_trigger_codes_v4(audit))
    if not codes:
        return "not_required"
    if codes & {
        _TRIGGER_POLICY["proposal_stop_move"],
        _TRIGGER_POLICY["proposal_sign"],
        _TRIGGER_POLICY["constraint"],
    }:
        return "critical"
    if _TRIGGER_POLICY["proposal_delta"] in codes:
        return "high"
    return "normal"


def _probe_projections_v4(
    profile_id: str,
    strong_action: str,
    groundings: GroundingOutputsV4,
) -> tuple[ActionProbeProjectionV4, ...]:
    """Project the strong action to every equally nearest public probe."""

    if strong_action == "stop":
        return ()
    candidates = affordance_probe_candidates_for_action(
        profile_id,
        strong_action,
    )
    traversability = groundings.local_traversability
    assert traversability is not None
    affordance_by_label = {
        str(item.direction_label): item
        for item in traversability.direction_affordances
    }
    return tuple(
        ActionProbeProjectionV4(
            direction_label=candidate.label,
            angle_gap_deg=candidate.angle_gap_deg,
            affordance=(
                affordance_by_label[candidate.label].affordance
                if candidate.label in affordance_by_label
                else None
            ),
            clearance=(
                affordance_by_label[candidate.label].clearance
                if candidate.label in affordance_by_label
                else None
            ),
            limiting_factors=(
                tuple(
                    affordance_by_label[
                        candidate.label
                    ].limiting_factors
                )
                if candidate.label in affordance_by_label
                else ()
            ),
        )
        for candidate in sorted(
            candidates,
            key=lambda value: value.label,
        )
    )


def _probe_support_check_v4(
    projections: tuple[ActionProbeProjectionV4, ...],
) -> str:
    """Aggregate physical support without discarding nearest-probe ties."""

    if not projections:
        return "not_applicable"
    per_probe = {
        _single_probe_support_v4(projection)
        for projection in projections
    }
    if per_probe == {"supported"}:
        return "all_supported"
    if per_probe == {"unsupported"}:
        return "all_unsupported"
    if {"supported", "unsupported"}.issubset(per_probe):
        return "mixed"
    return "unknown"


def _single_probe_support_v4(
    projection: ActionProbeProjectionV4,
) -> str:
    """Classify one nearest physical probe."""

    if projection.clearance == "insufficient":
        return "unsupported"
    if projection.affordance in {"direct", "conditional"}:
        return "supported"
    if projection.affordance in {"blocked", "forbidden"}:
        return "unsupported"
    return "unknown"


def _constraint_check_v4(
    groundings: GroundingOutputsV4,
    *,
    strong_action: str,
) -> str:
    """Check only hard strong-action/constraint contradictions."""

    interaction = groundings.interaction_constraint_grounding
    assert interaction is not None
    effects = applicable_constraint_effects_v4(
        strong_action,
        interaction,
    )
    if not effects:
        return "not_applicable"
    if strong_action != "stop" and "stop" in effects:
        return "conflict"
    return "passed"


def _proposal_relation_v4(
    comparison: ActionComparisonFactsV4,
    *,
    tolerance_deg: float,
) -> str:
    """Map low-level direction facts to one relation code."""

    if comparison.stop_move_mismatch:
        return "stop_move_mismatch"
    if comparison.sign_conflict:
        return "sign_conflict"
    if comparison.angular_delta_deg is None:
        return "exact"
    if comparison.angular_delta_deg == 0.0:
        return "exact"
    if (
        comparison.angular_delta_deg is not None
        and comparison.angular_delta_deg <= tolerance_deg
    ):
        return "within_tolerance"
    return "large_delta"


def _proposal_tuple_is_consistent_v4(
    proposal: DecisionProposalV4,
) -> bool:
    """Return whether Pydantic-validated basic tuple rules hold."""

    label = str(proposal.selected_action_label)
    if label == "stop":
        return (
            proposal.speed == "stop"
            and proposal.response == "stop"
        )
    return (
        proposal.speed != "stop"
        and proposal.response != "stop"
        and (
            proposal.response == "proceed"
            or proposal.speed == "slow"
        )
    )


def _review_trigger_codes_v4(
    checks: ActionReconciliationChecksV4,
    *,
    comparison: ActionComparisonFactsV4,
    proposal_action: str,
    strong_action: str,
) -> list[str]:
    """Derive review triggers without conflating two angle policies.

    The action profile keeps its 10-degree compatibility tolerance for the
    descriptive ``proposal_strong_relation`` check.  Human review uses an
    independent strict ``>20`` degree threshold. Sign conflicts, stop/move
    mismatches, constraint conflicts, and invalid action tuples remain hard
    review triggers at every angle. Sparse probe conflicts stay in the audit,
    but the admission policy treats probe-only findings as Grounding QC
    advisories rather than action-label failures.
    """

    values: set[str] = set()
    relation = checks.proposal_strong_relation
    if relation == "stop_move_mismatch":
        values.add(_TRIGGER_POLICY["proposal_stop_move"])
    elif relation == "sign_conflict":
        values.add(_TRIGGER_POLICY["proposal_sign"])
    elif (
        relation == "large_delta"
        and proposal_action != "stop"
        and strong_action != "stop"
        and not comparison.sign_conflict
        and comparison.angular_delta_deg is not None
        and comparison.angular_delta_deg
        > ACTION_RECONCILIATION_ANGLE_REVIEW_THRESHOLD_DEG_V4
    ):
        values.add(_TRIGGER_POLICY["proposal_delta"])
    if checks.probe_support == "all_unsupported":
        values.add(_TRIGGER_POLICY["probe_unsupported"])
    elif checks.probe_support == "mixed":
        values.add(_TRIGGER_POLICY["probe_mixed"])
    elif checks.probe_support == "unknown":
        values.add(_TRIGGER_POLICY["probe_unknown"])
    if checks.constraint_effect_consistency == "conflict":
        values.add(_TRIGGER_POLICY["constraint"])
    if checks.action_tuple_consistency == "conflict":
        values.add(_TRIGGER_POLICY["tuple"])
    return sorted(values)


def _canonical_digest(value: Any) -> str:
    """Return SHA-256 over deterministic JSON serialization."""

    return reasoning_payload_digest_v4(value)


__all__ = [
    "ACTION_RECONCILIATION_ANGLE_REVIEW_THRESHOLD_DEG_V4",
    "ACTION_RECONCILIATION_POLICY_VERSION_V4",
    "ACTION_RECONCILIATION_REVIEW_ADMISSION_POLICY_VERSION_V4",
    "action_reconciliation_advisory_trigger_codes_v4",
    "action_reconciliation_blocking_trigger_codes_v4",
    "action_reconciliation_policy_digest_v4",
    "action_reconciliation_review_priority_v4",
    "action_reconciliation_review_required_v4",
    "build_action_reconciliation_audit_v4",
    "validate_action_reconciliation_audit_v4",
]
