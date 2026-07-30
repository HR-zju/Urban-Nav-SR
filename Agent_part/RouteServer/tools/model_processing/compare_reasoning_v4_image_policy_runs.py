#!/usr/bin/env python3
"""Compare paired Reasoning V4 section-image-policy annotation runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from mcp_server.teacher.reasoning_v4_image_policy import (
    REASONING_V4_ALL_VISIBLE_IMAGES_POLICY_ID,
    REASONING_V4_SECTION_ROUTED_VISUAL_POLICY_ID,
)


COMPARISON_SCHEMA_VERSION = (
    "urbannav_reasoning_v4_image_policy_paired_comparison_v1"
)
_SECTIONS = (
    "TaskGrounding",
    "SceneGrounding",
    "LocalTraversability",
    "NavigationContextGrounding",
    "InteractionConstraintGrounding",
    "Decision",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse paired-run comparison arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument(
        "--baseline-annotation-jsonl",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--routed-annotation-jsonl",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--baseline-report-json",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--routed-report-json",
        required=True,
        type=Path,
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int, default=256)
    parser.add_argument(
        "--minimum-throughput-gain",
        type=float,
        default=0.10,
    )
    return parser.parse_args(argv)


def compare_reasoning_v4_image_policy_runs(
    *,
    source_jsonl: str | Path,
    baseline_annotation_jsonl: str | Path,
    routed_annotation_jsonl: str | Path,
    baseline_report_json: str | Path,
    routed_report_json: str | Path,
    expected_rows: int = 256,
    minimum_throughput_gain: float = 0.10,
) -> dict[str, Any]:
    """Build one machine-readable paired image-policy comparison.

    Args:
        source_jsonl: Shared frozen Reasoning V4 source manifest.
        baseline_annotation_jsonl: Latest all-visible annotation log.
        routed_annotation_jsonl: Latest section-routed annotation log.
        baseline_report_json: Final baseline runner report.
        routed_report_json: Final routed runner report.
        expected_rows: Exact paired source count.
        minimum_throughput_gain: Relative speed gain required to recommend
            replacing the baseline policy.

    Returns:
        Comparison metrics, quality gates, and bounded changed-row examples.

    Raises:
        ValueError: If source identities are duplicated or the two arms do not
            cover the exact same source IDs.
    """

    sources = _load_unique_sources(source_jsonl)
    baseline, baseline_versions = _load_latest_annotations(
        baseline_annotation_jsonl
    )
    routed, routed_versions = _load_latest_annotations(
        routed_annotation_jsonl
    )
    source_ids = set(sources)
    if len(source_ids) != int(expected_rows):
        raise ValueError(
            "paired comparison source count mismatch: "
            f"expected={expected_rows} actual={len(source_ids)}"
        )
    for arm_name, rows in (
        ("baseline", baseline),
        ("routed", routed),
    ):
        missing = sorted(source_ids - set(rows))
        extra = sorted(set(rows) - source_ids)
        if missing or extra:
            raise ValueError(
                f"{arm_name} source IDs do not match shared source: "
                f"missing={missing[:8]} extra={extra[:8]}"
            )

    baseline_report = _load_json(baseline_report_json)
    routed_report = _load_json(routed_report_json)
    baseline_metrics = _arm_metrics(
        sources,
        baseline,
        baseline_report,
        annotation_versions=baseline_versions,
    )
    routed_metrics = _arm_metrics(
        sources,
        routed,
        routed_report,
        annotation_versions=routed_versions,
    )
    paired = _paired_metrics(source_ids, baseline, routed)
    deltas = {
        key: _fraction_delta(
            baseline_metrics.get(key),
            routed_metrics.get(key),
        )
        for key in (
            "all_sections_accepted_fraction",
            "candidate_valid_fraction",
            "gt_action_exact_agreement_fraction",
            "bev_route_grounding_fraction",
            "retry_row_fraction",
            "pending_human_fraction",
        )
    }
    baseline_rate = _number(
        baseline_report.get("rate_rows_per_hour")
    )
    routed_rate = _number(routed_report.get("rate_rows_per_hour"))
    throughput_gain = (
        routed_rate / baseline_rate - 1.0
        if baseline_rate > 0.0
        else None
    )
    gates = {
        "same_source_ids": True,
        "baseline_policy_identity": (
            baseline_metrics["policy_id_counts"]
            == {REASONING_V4_ALL_VISIBLE_IMAGES_POLICY_ID: len(source_ids)}
        ),
        "routed_policy_identity": (
            routed_metrics["policy_id_counts"]
            == {
                REASONING_V4_SECTION_ROUTED_VISUAL_POLICY_ID: len(source_ids)
            }
        ),
        "baseline_policy_digest_complete": (
            len(baseline_metrics["policy_digest_counts"]) == 1
            and "missing" not in baseline_metrics["policy_digest_counts"]
        ),
        "routed_policy_digest_complete": (
            len(routed_metrics["policy_digest_counts"]) == 1
            and "missing" not in routed_metrics["policy_digest_counts"]
        ),
        "routed_all_sections_accepted_not_worse_2pp": (
            deltas["all_sections_accepted_fraction"] >= -0.02
        ),
        "routed_candidate_valid_not_worse_2pp": (
            deltas["candidate_valid_fraction"] >= -0.02
        ),
        "routed_gt_action_agreement_not_worse_3pp": (
            deltas["gt_action_exact_agreement_fraction"] >= -0.03
        ),
        "routed_bev_grounding_not_worse_3pp": (
            deltas["bev_route_grounding_fraction"] >= -0.03
        ),
        "routed_retry_not_worse_3pp": (
            deltas["retry_row_fraction"] <= 0.03
        ),
        "routed_pending_human_not_worse_5pp": (
            deltas["pending_human_fraction"] <= 0.05
        ),
        "throughput_gain_reaches_target": (
            throughput_gain is not None
            and throughput_gain >= float(minimum_throughput_gain)
        ),
    }
    quality_gate_names = [
        name
        for name in gates
        if name != "throughput_gain_reaches_target"
    ]
    quality_passed = all(gates[name] for name in quality_gate_names)
    replacement_recommended = bool(
        quality_passed
        and gates["throughput_gain_reaches_target"]
    )
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "source_jsonl": str(Path(source_jsonl).expanduser()),
        "expected_rows": int(expected_rows),
        "paired_source_rows": len(source_ids),
        "baseline": baseline_metrics,
        "routed": routed_metrics,
        "paired": paired,
        "deltas_routed_minus_baseline": deltas,
        "throughput": {
            "baseline_rows_per_hour": baseline_rate,
            "routed_rows_per_hour": routed_rate,
            "relative_gain": throughput_gain,
            "minimum_required_gain": float(minimum_throughput_gain),
        },
        "gates": gates,
        "quality_gates_passed": quality_passed,
        "replacement_recommended": replacement_recommended,
        "decision_note": (
            "Replace only after manual review of changed rows."
            if replacement_recommended
            else "Keep the active full run unchanged."
        ),
    }


def _arm_metrics(
    sources: dict[str, dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    report: dict[str, Any],
    *,
    annotation_versions: Counter[str],
) -> dict[str, Any]:
    """Aggregate one arm while preserving quality-axis separation."""

    total = len(sources)
    complete = 0
    accepted = 0
    candidate_valid = 0
    gt_comparable = 0
    gt_exact = 0
    bev_rows = 0
    bev_aligned = 0
    interaction_nonempty = 0
    retry_rows = 0
    pending_human = 0
    policy_ids: Counter[str] = Counter()
    policy_digests: Counter[str] = Counter()
    interaction_effects: Counter[str] = Counter()
    stage_statuses: dict[str, Counter[str]] = {
        section: Counter() for section in _SECTIONS
    }
    stage_profile_ids: dict[str, Counter[str]] = {
        section: Counter() for section in _SECTIONS
    }
    selected_role_patterns: dict[str, Counter[str]] = {
        section: Counter() for section in _SECTIONS
    }

    for source_id, source in sources.items():
        annotation = annotations[source_id]
        complete += int(annotation.get("record_status") == "complete")
        stage_records = annotation.get("stage_records") or {}
        all_accepted = (
            set(stage_records) == set(_SECTIONS)
            and all(
                (stage_records.get(name) or {}).get("status") == "accepted"
                for name in _SECTIONS
            )
        )
        accepted += int(all_accepted)
        candidate_valid += int(
            _nested(
                annotation,
                "candidate_structure_record",
                "status",
            )
            == "valid"
        )
        policy_id = str(
            _nested(
                annotation,
                "metadata",
                "section_image_policy_id",
            )
            or "missing"
        )
        policy_ids[policy_id] += 1
        policy_digest = str(
            _nested(
                annotation,
                "metadata",
                "section_image_policy_digest",
            )
            or "missing"
        )
        policy_digests[policy_digest] += 1

        for section_name in _SECTIONS:
            record = stage_records.get(section_name) or {}
            stage_statuses[section_name][
                str(record.get("status") or "missing")
            ] += 1
            stage_profile_ids[section_name][
                str(record.get("producer_profile_id") or "missing")
            ] += 1
            roles = _nested(record, "telemetry", "selected_image_roles")
            if isinstance(roles, list):
                selected_role_patterns[section_name][
                    "+".join(str(role) for role in roles)
                ] += 1
        retry_rows += int(
            any(
                int((record or {}).get("attempts") or 0) > 1
                for record in stage_records.values()
            )
        )
        pending_human += int(
            bool(
                _nested(
                    annotation,
                    "action_reconciliation",
                    "review_trigger_codes",
                )
                or []
            )
        )

        proposal = _nested(
            annotation,
            "decision_proposal",
            "selected_action_label",
        )
        strong = _nested(
            source,
            "direction_supervision",
            "label_name",
        )
        if proposal is not None and strong is not None:
            gt_comparable += 1
            gt_exact += int(str(proposal) == str(strong))

        has_bev = bool(
            _nested(source, "observation", "input_modalities", "bev")
        )
        if has_bev:
            bev_rows += 1
            task_direction = (
                _nested(
                    annotation,
                    "grounding_outputs",
                    "navigation_context_grounding",
                    "task_direction",
                )
                or {}
            )
            bev_aligned += int(
                task_direction.get("status") == "aligned"
                and task_direction.get("source") == "bev_route"
                and task_direction.get("direction") is not None
            )

        constraints = (
            _nested(
                annotation,
                "grounding_outputs",
                "interaction_constraint_grounding",
                "constraints",
            )
            or []
        )
        interaction_nonempty += int(bool(constraints))
        for constraint in constraints:
            if isinstance(constraint, dict):
                interaction_effects[
                    str(constraint.get("effect") or "missing")
                ] += 1

    return {
        "rows": total,
        "annotation_version_counts": dict(
            sorted(
                Counter(annotation_versions.values()).items()
            )
        ),
        "record_complete_fraction": _fraction(complete, total),
        "all_sections_accepted_fraction": _fraction(accepted, total),
        "candidate_valid_fraction": _fraction(candidate_valid, total),
        "gt_action_comparable_rows": gt_comparable,
        "gt_action_exact_agreement_fraction": _fraction(
            gt_exact,
            gt_comparable,
        ),
        "bev_rows": bev_rows,
        "bev_route_grounding_fraction": _fraction(
            bev_aligned,
            bev_rows,
        ),
        "interaction_nonempty_fraction": _fraction(
            interaction_nonempty,
            total,
        ),
        "interaction_effect_counts": dict(
            sorted(interaction_effects.items())
        ),
        "retry_row_fraction": _fraction(retry_rows, total),
        "pending_human_fraction": _fraction(pending_human, total),
        "policy_id_counts": dict(sorted(policy_ids.items())),
        "policy_digest_counts": dict(sorted(policy_digests.items())),
        "stage_status_counts": {
            name: dict(sorted(values.items()))
            for name, values in stage_statuses.items()
        },
        "stage_profile_id_counts": {
            name: dict(sorted(values.items()))
            for name, values in stage_profile_ids.items()
        },
        "selected_image_role_patterns": {
            name: dict(sorted(values.items()))
            for name, values in selected_role_patterns.items()
        },
        "runner_report": {
            "status": report.get("status"),
            "rate_rows_per_hour": report.get("rate_rows_per_hour"),
            "telemetry_totals": report.get("telemetry_totals") or {},
            "stage_metrics": report.get("stage_metrics") or {},
        },
    }


def _paired_metrics(
    source_ids: Iterable[str],
    baseline: dict[str, dict[str, Any]],
    routed: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare output semantics on exact source identities."""

    ids = sorted(source_ids)
    decision_equal = 0
    navigation_equal = 0
    interaction_equal = 0
    changed_count = 0
    changed: list[dict[str, Any]] = []
    for source_id in ids:
        baseline_row = baseline[source_id]
        routed_row = routed[source_id]
        baseline_decision = _nested(
            baseline_row,
            "decision_proposal",
            "selected_action_label",
        )
        routed_decision = _nested(
            routed_row,
            "decision_proposal",
            "selected_action_label",
        )
        baseline_navigation = _nested(
            baseline_row,
            "grounding_outputs",
            "navigation_context_grounding",
            "task_direction",
        )
        routed_navigation = _nested(
            routed_row,
            "grounding_outputs",
            "navigation_context_grounding",
            "task_direction",
        )
        baseline_interaction = _nested(
            baseline_row,
            "grounding_outputs",
            "interaction_constraint_grounding",
            "constraints",
        )
        routed_interaction = _nested(
            routed_row,
            "grounding_outputs",
            "interaction_constraint_grounding",
            "constraints",
        )
        baseline_interaction = _canonical_constraints(
            baseline_interaction
        )
        routed_interaction = _canonical_constraints(
            routed_interaction
        )
        decision_equal += int(baseline_decision == routed_decision)
        navigation_equal += int(
            baseline_navigation == routed_navigation
        )
        interaction_equal += int(
            baseline_interaction == routed_interaction
        )
        is_changed = (
            baseline_decision != routed_decision
            or baseline_navigation != routed_navigation
            or baseline_interaction != routed_interaction
        )
        changed_count += int(is_changed)
        if is_changed and len(changed) < 64:
            changed.append(
                {
                    "source_group_id": source_id,
                    "decision": {
                        "baseline": baseline_decision,
                        "routed": routed_decision,
                    },
                    "navigation_context": {
                        "baseline": baseline_navigation,
                        "routed": routed_navigation,
                    },
                    "interaction_constraints": {
                        "baseline": baseline_interaction,
                        "routed": routed_interaction,
                    },
                }
            )
    total = len(ids)
    return {
        "decision_exact_agreement_fraction": _fraction(
            decision_equal,
            total,
        ),
        "navigation_context_exact_agreement_fraction": _fraction(
            navigation_equal,
            total,
        ),
        "interaction_constraint_exact_agreement_fraction": _fraction(
            interaction_equal,
            total,
        ),
        "changed_rows": changed_count,
        "changed_row_examples": changed,
        "changed_row_examples_truncated": changed_count > len(changed),
    }


def _load_unique_sources(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load one source manifest and reject duplicate source identities."""

    result: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        source_id = str(
            _nested(row, "observation", "source_group_id") or ""
        )
        if not source_id:
            raise ValueError("source row is missing source_group_id")
        if source_id in result:
            raise ValueError(f"duplicate source_group_id: {source_id}")
        result[source_id] = row
    return result


def _load_latest_annotations(
    path: str | Path,
) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    """Load the latest append-only annotation for each source ID."""

    result: dict[str, dict[str, Any]] = {}
    versions: Counter[str] = Counter()
    for row in _iter_jsonl(path):
        source_id = str(row.get("source_group_id") or "")
        if not source_id:
            raise ValueError("annotation row is missing source_group_id")
        result[source_id] = row
        versions[source_id] += 1
    return result, versions


def _iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    """Yield JSON objects from a non-empty JSONL file."""

    value = Path(path).expanduser()
    with value.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(
                    f"JSONL row must be an object: {value}:{line_number}"
                )
            yield row


def _load_json(path: str | Path) -> dict[str, Any]:
    """Load one JSON object."""

    payload = json.loads(
        Path(path).expanduser().read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return payload


def _nested(payload: Any, *keys: str) -> Any:
    """Read a nested mapping path without inventing defaults."""

    value = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _canonical_constraints(value: Any) -> list[dict[str, Any]]:
    """Normalize unordered interaction constraints for paired comparison."""

    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        payload = dict(item)
        affected = payload.get("affected_direction_labels")
        if isinstance(affected, list):
            payload["affected_direction_labels"] = sorted(
                str(label) for label in affected
            )
        normalized.append(payload)
    return sorted(
        normalized,
        key=lambda item: json.dumps(
            item,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _fraction(numerator: int, denominator: int) -> float:
    """Return a stable zero-safe fraction."""

    return float(numerator) / denominator if denominator else 0.0


def _fraction_delta(baseline: Any, routed: Any) -> float:
    """Return routed minus baseline for numeric fractions."""

    return _number(routed) - _number(baseline)


def _number(value: Any) -> float:
    """Return a finite numeric value or zero."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if result != result or result in {float("inf"), float("-inf")}:
        return 0.0
    return result


def main(argv: Sequence[str] | None = None) -> None:
    """Compare paired runs and write the requested report."""

    args = parse_args(argv)
    report = compare_reasoning_v4_image_policy_runs(
        source_jsonl=args.source_jsonl,
        baseline_annotation_jsonl=args.baseline_annotation_jsonl,
        routed_annotation_jsonl=args.routed_annotation_jsonl,
        baseline_report_json=args.baseline_report_json,
        routed_report_json=args.routed_report_json,
        expected_rows=args.expected_rows,
        minimum_throughput_gain=args.minimum_throughput_gain,
    )
    args.output_json.expanduser().parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output_json.expanduser().write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
