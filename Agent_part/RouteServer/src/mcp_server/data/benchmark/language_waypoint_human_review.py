"""Deterministic acceptance gates for visible language-goal human review.

The browser review page exports optional adjudications.  This module turns
those selections into a versioned, dataset-level gate before any expensive
full language annotation is allowed to start.  It never edits teacher labels
or calls a model.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp_server.contracts.language_waypoint_benchmark import LanguageGoalAnnotationV1
from mcp_server.data.benchmark.language_waypoint_review import language_goal_review_id_v1
from mcp_server.data.io.jsonl import iter_jsonl


LANGUAGE_GOAL_HUMAN_REVIEW_GATE_SCHEMA_VERSION_V1 = "urbannav_language_goal_human_review_gate_v1"

_ACCEPT = "accept"
_REJECT_DECISIONS = frozenset(
    {
        "reject_not_visible",
        "reject_wrong_relation",
        "reject_generic",
        "needs_rewrite",
    }
)
_UNCERTAIN = "uncertain"
_VALID_DECISIONS = _REJECT_DECISIONS | {_ACCEPT, _UNCERTAIN}


@dataclass(frozen=True)
class LanguageGoalHumanReviewGateConfigV1:
    """Specify transparent promotion criteria for a language annotation smoke.

    The gate is intentionally parameterized rather than encoding a hidden
    research decision.  Defaults require every teacher-accepted smoke label to
    receive one human selection, at least 90 percent accepts, at most 5 percent
    uncertain labels, and representation for every observed instruction type.

    Args:
        require_all_eligible_reviewed: Reject a smoke with any unreviewed
            teacher-accepted annotation.
        min_accept_rate: Minimum accepted fraction among reviewed labels.
        max_uncertain_rate: Largest allowed uncertain fraction among reviewed labels.
        min_reviewed_per_instruction_type: Required review coverage for each
            teacher-accepted type present at or above this population size.
    """

    require_all_eligible_reviewed: bool = True
    min_accept_rate: float = 0.90
    max_uncertain_rate: float = 0.05
    min_reviewed_per_instruction_type: int = 1

    def __post_init__(self) -> None:
        """Validate bounded promotion thresholds.

        Raises:
            ValueError: If a fraction or required per-type count is invalid.
        """

        if not 0.0 <= self.min_accept_rate <= 1.0:
            raise ValueError("min_accept_rate must be within [0, 1]")
        if not 0.0 <= self.max_uncertain_rate <= 1.0:
            raise ValueError("max_uncertain_rate must be within [0, 1]")
        if self.min_reviewed_per_instruction_type < 1:
            raise ValueError("min_reviewed_per_instruction_type must be positive")


def evaluate_language_goal_human_review_v1(
    *,
    annotation_jsonl: str | Path,
    adjudications_json: str | Path,
    report_json: str | Path,
    config: LanguageGoalHumanReviewGateConfigV1 | None = None,
) -> dict[str, Any]:
    r"""Evaluate browser-exported human decisions without modifying annotations.

    Let $A$ be teacher-accepted annotations and $R\subseteq A$ the reviewed
    subset.  The gate reports

    \[
    r_{accept}=\frac{|\{a\in R:d(a)=accept\}|}{|R|},\qquad
    r_{uncertain}=\frac{|\{a\in R:d(a)=uncertain\}|}{|R|}.
    \]

    Promotion succeeds only if the configured coverage and rate constraints
    hold, and every adjudication resolves to one known review card.

    Args:
        annotation_jsonl: Typed language-goal annotation ledger reviewed by humans.
        adjudications_json: JSON array exported by the portable review HTML.
        report_json: Deterministic gate report destination.
        config: Explicit review coverage and acceptance policy.

    Returns:
        Versioned report including decision slices and `passed`.

    Raises:
        ValueError: If annotations or adjudications have duplicate/invalid identities.
    """

    cfg = config or LanguageGoalHumanReviewGateConfigV1()
    annotations = _load_unique_annotations_v1(annotation_jsonl)
    eligible = [item for item in annotations.values() if item.validator_status == "accepted"]
    decisions = _load_adjudications_v1(adjudications_json)
    review_to_annotation = {
        language_goal_review_id_v1(annotation): annotation for annotation in eligible
    }
    unknown_review_ids = sorted(set(decisions) - set(review_to_annotation))
    decision_counts: Counter[str] = Counter()
    reviewed_by_type: Counter[str] = Counter()
    population_by_type: Counter[str] = Counter(
        str(annotation.instruction_type) for annotation in eligible
    )
    accepted_by_type: Counter[str] = Counter()
    missing_review_ids: list[str] = []
    for review_id, annotation in sorted(review_to_annotation.items()):
        decision = decisions.get(review_id)
        if decision is None:
            missing_review_ids.append(review_id)
            continue
        instruction_type = str(annotation.instruction_type)
        decision_counts[decision] += 1
        reviewed_by_type[instruction_type] += 1
        if decision == _ACCEPT:
            accepted_by_type[instruction_type] += 1

    reviewed_count = sum(decision_counts.values())
    accepted_count = decision_counts[_ACCEPT]
    uncertain_count = decision_counts[_UNCERTAIN]
    reject_count = sum(decision_counts[item] for item in _REJECT_DECISIONS)
    accept_rate = accepted_count / reviewed_count if reviewed_count else 0.0
    uncertain_rate = uncertain_count / reviewed_count if reviewed_count else 0.0
    coverage_failures = {
        instruction_type: {
            "population": population,
            "reviewed": reviewed_by_type[instruction_type],
            "required": min(population, cfg.min_reviewed_per_instruction_type),
        }
        for instruction_type, population in sorted(population_by_type.items())
        if reviewed_by_type[instruction_type] < min(population, cfg.min_reviewed_per_instruction_type)
    }
    failures: list[str] = []
    if unknown_review_ids:
        failures.append("adjudication_unknown_review_id")
    if cfg.require_all_eligible_reviewed and missing_review_ids:
        failures.append("eligible_annotation_unreviewed")
    if accept_rate < cfg.min_accept_rate:
        failures.append("accept_rate_below_threshold")
    if uncertain_rate > cfg.max_uncertain_rate:
        failures.append("uncertain_rate_above_threshold")
    if coverage_failures:
        failures.append("instruction_type_review_coverage_insufficient")
    report = {
        "schema_version": LANGUAGE_GOAL_HUMAN_REVIEW_GATE_SCHEMA_VERSION_V1,
        "annotation_jsonl": str(Path(annotation_jsonl).expanduser().resolve()),
        "adjudications_json": str(Path(adjudications_json).expanduser().resolve()),
        "annotation_jsonl_sha256": _artifact_sha256_v1(annotation_jsonl),
        "adjudications_json_sha256": _artifact_sha256_v1(adjudications_json),
        "config": {
            "require_all_eligible_reviewed": cfg.require_all_eligible_reviewed,
            "min_accept_rate": cfg.min_accept_rate,
            "max_uncertain_rate": cfg.max_uncertain_rate,
            "min_reviewed_per_instruction_type": cfg.min_reviewed_per_instruction_type,
        },
        "eligible_annotation_count": len(eligible),
        "teacher_rejected_annotation_count": len(annotations) - len(eligible),
        "reviewed_count": reviewed_count,
        "missing_review_count": len(missing_review_ids),
        "unknown_review_id_count": len(unknown_review_ids),
        "decision_counts": dict(sorted(decision_counts.items())),
        "accept_rate": accept_rate,
        "uncertain_rate": uncertain_rate,
        "reject_count": reject_count,
        "instruction_type_population": dict(sorted(population_by_type.items())),
        "instruction_type_reviewed": dict(sorted(reviewed_by_type.items())),
        "instruction_type_accepted": dict(sorted(accepted_by_type.items())),
        "instruction_type_coverage_failures": coverage_failures,
        "missing_review_ids": missing_review_ids,
        "unknown_review_ids": unknown_review_ids,
        "gate_failures": failures,
        "passed": not failures,
    }
    _write_json_v1(report_json, report)
    return report


def _load_unique_annotations_v1(path: str | Path) -> dict[str, LanguageGoalAnnotationV1]:
    """Load language annotations keyed by immutable annotation identity."""

    rows: dict[str, LanguageGoalAnnotationV1] = {}
    for payload in iter_jsonl(path):
        annotation = LanguageGoalAnnotationV1.model_validate(payload)
        if annotation.annotation_id in rows:
            raise ValueError(f"duplicate annotation_id: {annotation.annotation_id}")
        rows[annotation.annotation_id] = annotation
    return rows


def _load_adjudications_v1(path: str | Path) -> dict[str, str]:
    """Load one browser-exported decision per stable review card."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("adjudications_json must contain a JSON array")
    rows: dict[str, str] = {}
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"adjudication {index} is not an object")
        review_id = item.get("review_id")
        decision = item.get("decision")
        if not isinstance(review_id, str) or not review_id:
            raise ValueError(f"adjudication {index} lacks review_id")
        if decision not in _VALID_DECISIONS:
            raise ValueError(f"adjudication {index} has unsupported decision: {decision!r}")
        if review_id in rows:
            raise ValueError(f"duplicate adjudication review_id: {review_id}")
        rows[review_id] = str(decision)
    return rows


def _artifact_sha256_v1(path: str | Path) -> str:
    """Hash one immutable input artifact without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_v1(path: str | Path, payload: dict[str, Any]) -> None:
    """Write one deterministic small human-review gate report."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "LANGUAGE_GOAL_HUMAN_REVIEW_GATE_SCHEMA_VERSION_V1",
    "LanguageGoalHumanReviewGateConfigV1",
    "evaluate_language_goal_human_review_v1",
]
