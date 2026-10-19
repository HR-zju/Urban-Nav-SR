#!/usr/bin/env python3
"""Build a digest-bound Stage I instruction registry from explicit decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from mcp_server.contracts.reasoning_dataset_v4 import ReasoningTrainingViewV4
from mcp_server.contracts.vqa_dataset_v2 import VQASourceV2
from mcp_server.data.io.jsonl import iter_jsonl
from mcp_server.data.stage_i.provenance import build_stage_i_instruction_provenance_v1


def build_registry(
    assignment_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    report_json: str | Path | None = None,
) -> dict[str, Any]:
    """Validate explicit assignments and write canonical registry entries."""

    assignment_path = Path(assignment_jsonl).expanduser()
    output_path = Path(output_jsonl).expanduser()
    entries = []
    keys: set[tuple[str, str]] = set()
    for raw in iter_jsonl(assignment_path):
        required = {
            "source_group_id",
            "source_fingerprint",
            "instruction",
            "training_role",
            "provenance_source",
            "policy_id",
            "policy_version",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(
                f"instruction provenance assignment is missing: {missing}"
            )
        entry = build_stage_i_instruction_provenance_v1(
            source_group_id=str(raw["source_group_id"]),
            source_fingerprint=str(raw["source_fingerprint"]),
            instruction=(
                None
                if raw["instruction"] is None
                else str(raw["instruction"])
            ),
            training_role=str(raw["training_role"]),  # type: ignore[arg-type]
            provenance_source=str(raw["provenance_source"]),
            policy_id=str(raw["policy_id"]),
            policy_version=str(raw["policy_version"]),
        )
        key = (entry.source_group_id, entry.source_fingerprint)
        if key in keys:
            raise ValueError(f"duplicate instruction assignment: {key}")
        keys.add(key)
        entries.append(entry.model_dump(mode="json", exclude_none=True))
    content = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in entries
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    report = {
        "schema_version": "urbannav_stage_i_instruction_registry_build_v1",
        "assignment_artifact": {
            "path": str(assignment_path),
            "sha256": hashlib.sha256(
                assignment_path.read_bytes()
            ).hexdigest(),
        },
        "output_artifact": {
            "path": str(output_path),
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "rows": len(entries),
        },
    }
    if report_json is not None:
        path = Path(report_json).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    return report


def build_conservative_registry(
    output_jsonl: str | Path,
    *,
    reasoning_view_jsonls: Sequence[str | Path] = (),
    vqa_source_jsonls: Sequence[str | Path] = (),
    report_json: str | Path | None = None,
) -> dict[str, Any]:
    """Seed exact entries as unknown/not-present without granting primary use."""

    decisions: dict[tuple[str, str], dict[str, Any]] = {}
    inputs: list[Path] = []
    for value in reasoning_view_jsonls:
        path = Path(value).expanduser()
        inputs.append(path)
        for raw in iter_jsonl(path):
            view = ReasoningTrainingViewV4.model_validate(raw)
            fingerprint = str(view.metadata.get("source_fingerprint") or "")
            _add_conservative_decision(
                decisions,
                source_group_id=view.source_group_id,
                source_fingerprint=fingerprint,
                instruction=view.instruction,
            )
    for value in vqa_source_jsonls:
        path = Path(value).expanduser()
        inputs.append(path)
        for raw in iter_jsonl(path):
            source = VQASourceV2.model_validate(raw)
            _add_conservative_decision(
                decisions,
                source_group_id=source.observation.source_group_id,
                source_fingerprint=source.source_fingerprint,
                instruction=source.observation.instruction,
            )
    if not decisions:
        raise ValueError("conservative provenance seed inputs are empty")
    entries = [
        build_stage_i_instruction_provenance_v1(
            **decision,
            training_role=(
                "unknown"
                if decision["instruction"] is not None
                else "not_present"
            ),
            provenance_source="unreviewed_source_instruction",
            policy_id="stage_i_conservative_instruction_seed",
            policy_version="1.0",
        ).model_dump(mode="json", exclude_none=True)
        for _key, decision in sorted(decisions.items())
    ]
    output_path = Path(output_jsonl).expanduser()
    content = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in entries
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    report = {
        "schema_version": "urbannav_stage_i_instruction_registry_build_v1",
        "mode": "conservative_seed",
        "input_artifacts": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in inputs
        ],
        "output_artifact": {
            "path": str(output_path),
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "rows": len(entries),
        },
        "present_instruction_role": "unknown",
        "formal_instruction_use_granted": False,
    }
    if report_json is not None:
        path = Path(report_json).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    return report


def build_policy_registry(
    policy_json: str | Path,
    output_jsonl: str | Path,
    *,
    reasoning_view_jsonls: Sequence[str | Path] = (),
    vqa_source_jsonls: Sequence[str | Path] = (),
    report_json: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve exact visible instruction strings through a frozen policy.

    The policy is keyed by the SHA256 of the exact instruction text.  It must
    fail closed on every unlisted value, so this path cannot infer provenance
    from a dataset name or from a caller-supplied default.

    Args:
        policy_json: Frozen exact-text policy artifact.
        output_jsonl: Destination for canonical per-source registry entries.
        reasoning_view_jsonls: Formal Reasoning V4 training-view artifacts.
        vqa_source_jsonls: Canonical VQA V2 source artifacts.
        report_json: Optional deterministic build-report destination.

    Returns:
        Deterministic registry build report.

    Raises:
        ValueError: If an input is empty, malformed, or contains an instruction
            without an exact policy decision.
    """

    policy_path = Path(policy_json).expanduser()
    policy = _load_instruction_policy(policy_path)
    decisions: dict[tuple[str, str], dict[str, Any]] = {}
    inputs: list[Path] = []
    for value in reasoning_view_jsonls:
        path = Path(value).expanduser()
        inputs.append(path)
        for raw in iter_jsonl(path):
            view = ReasoningTrainingViewV4.model_validate(raw)
            _add_conservative_decision(
                decisions,
                source_group_id=view.source_group_id,
                source_fingerprint=str(
                    view.metadata.get("source_fingerprint") or ""
                ),
                instruction=view.instruction,
            )
    for value in vqa_source_jsonls:
        path = Path(value).expanduser()
        inputs.append(path)
        for raw in iter_jsonl(path):
            source = VQASourceV2.model_validate(raw)
            _add_conservative_decision(
                decisions,
                source_group_id=source.observation.source_group_id,
                source_fingerprint=source.source_fingerprint,
                instruction=source.observation.instruction,
            )
    if not decisions:
        raise ValueError("policy provenance inputs are empty")

    rules = {
        str(item["instruction_sha256"]): item
        for item in policy["decisions"]
    }
    entries = []
    role_counts: Counter[str] = Counter()
    instruction_counts: Counter[str] = Counter()
    for _key, decision in sorted(decisions.items()):
        instruction = decision["instruction"]
        if instruction is None:
            training_role = "not_present"
            provenance_source = "no_source_instruction"
            instruction_key = "<NULL>"
        else:
            instruction_digest = hashlib.sha256(
                instruction.encode("utf-8")
            ).hexdigest()
            rule = rules.get(instruction_digest)
            if rule is None or rule["instruction"] != instruction:
                raise ValueError(
                    "instruction policy has no exact decision for: "
                    f"{instruction_digest}"
                )
            training_role = str(rule["training_role"])
            provenance_source = str(rule["provenance_source"])
            instruction_key = instruction_digest
        entry = build_stage_i_instruction_provenance_v1(
            **decision,
            training_role=training_role,  # type: ignore[arg-type]
            provenance_source=provenance_source,
            policy_id=str(policy["policy_id"]),
            policy_version=str(policy["policy_version"]),
        )
        entries.append(entry.model_dump(mode="json", exclude_none=True))
        role_counts[training_role] += 1
        instruction_counts[instruction_key] += 1

    output_path = Path(output_jsonl).expanduser()
    content = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in entries
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    report = {
        "schema_version": "urbannav_stage_i_instruction_registry_build_v1",
        "mode": "exact_policy",
        "policy_artifact": {
            "path": str(policy_path),
            "sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
            "policy_digest": policy["policy_digest"],
            "decision_authority": policy["decision_authority"],
        },
        "input_artifacts": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in inputs
        ],
        "output_artifact": {
            "path": str(output_path),
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "rows": len(entries),
        },
        "training_role_counts": dict(sorted(role_counts.items())),
        "instruction_sha256_counts": dict(sorted(instruction_counts.items())),
        "formal_instruction_use_granted": bool(role_counts["primary"]),
    }
    if report_json is not None:
        path = Path(report_json).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    return report


def _load_instruction_policy(path: Path) -> dict[str, Any]:
    """Validate an exact-string provenance policy and its content digest.

    Args:
        path: Policy JSON to validate.

    Returns:
        Validated raw policy payload.

    Raises:
        ValueError: If the schema, decisions, or content digest is invalid.
    """

    policy = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "policy_id",
        "policy_version",
        "decision_authority",
        "unmatched_instruction_policy",
        "decisions",
        "policy_digest",
    }
    if set(policy) != required:
        raise ValueError(
            "instruction policy fields differ: "
            f"missing={sorted(required - set(policy))}, "
            f"extra={sorted(set(policy) - required)}"
        )
    if policy["schema_version"] != "urbannav_stage_i_instruction_policy_v1":
        raise ValueError("unsupported instruction policy schema")
    if policy["unmatched_instruction_policy"] != "fail_closed":
        raise ValueError("instruction policy must fail closed on unmatched text")
    raw_decisions = policy["decisions"]
    if not isinstance(raw_decisions, list) or not raw_decisions:
        raise ValueError("instruction policy decisions are empty")
    seen: set[str] = set()
    for item in raw_decisions:
        if not isinstance(item, dict) or set(item) != {
            "instruction",
            "instruction_sha256",
            "training_role",
            "provenance_source",
            "rationale",
        }:
            raise ValueError("instruction policy decision fields differ")
        instruction = str(item["instruction"])
        expected = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        if item["instruction_sha256"] != expected:
            raise ValueError("instruction policy text digest mismatch")
        if expected in seen:
            raise ValueError("duplicate instruction policy decision")
        seen.add(expected)
        if item["training_role"] not in {"primary", "audit_only"}:
            raise ValueError("policy decision role must be primary or audit_only")
        if not str(item["provenance_source"]):
            raise ValueError("policy decision provenance_source is empty")
        if not str(item["rationale"]):
            raise ValueError("policy decision rationale is empty")
    digest_payload = {
        key: value
        for key, value in policy.items()
        if key != "policy_digest"
    }
    rendered = json.dumps(
        digest_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    expected_digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    if policy["policy_digest"] != expected_digest:
        raise ValueError("instruction policy digest mismatch")
    return policy


def _add_conservative_decision(
    decisions: dict[tuple[str, str], dict[str, Any]],
    *,
    source_group_id: str,
    source_fingerprint: str,
    instruction: str | None,
) -> None:
    """Collect one exact source revision and reject conflicting duplicates."""

    if len(source_fingerprint) != 64:
        raise ValueError("source fingerprint is missing from provenance seed")
    key = (source_group_id, source_fingerprint)
    decision = {
        "source_group_id": source_group_id,
        "source_fingerprint": source_fingerprint,
        "instruction": instruction,
    }
    if key in decisions and decisions[key] != decision:
        raise ValueError(f"conflicting provenance seed source: {key}")
    decisions[key] = decision


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit decisions and immutable output paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment-jsonl", type=Path)
    parser.add_argument("--policy-json", type=Path)
    parser.add_argument(
        "--reasoning-view-jsonl",
        action="append",
        default=[],
        type=Path,
    )
    parser.add_argument(
        "--vqa-source-jsonl",
        action="append",
        default=[],
        type=Path,
    )
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Build the registry without inferring roles from dataset names."""

    args = parse_args(argv)
    if args.assignment_jsonl is not None:
        if (
            args.policy_json is not None
            or args.reasoning_view_jsonl
            or args.vqa_source_jsonl
        ):
            raise ValueError(
                "assignment, policy, and conservative modes are mutually exclusive"
            )
        report = build_registry(
            args.assignment_jsonl,
            args.output_jsonl,
            report_json=args.report_json,
        )
    elif args.policy_json is not None:
        report = build_policy_registry(
            args.policy_json,
            args.output_jsonl,
            reasoning_view_jsonls=args.reasoning_view_jsonl,
            vqa_source_jsonls=args.vqa_source_jsonl,
            report_json=args.report_json,
        )
    else:
        report = build_conservative_registry(
            args.output_jsonl,
            reasoning_view_jsonls=args.reasoning_view_jsonl,
            vqa_source_jsonls=args.vqa_source_jsonl,
            report_json=args.report_json,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
