#!/usr/bin/env python3
"""Build a digest-bound Stage I instruction registry from explicit decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from mcp_server.contracts.reasoning_dataset_v4 import ReasoningTrainingViewV4
from mcp_server.contracts.vqa_dataset_v2 import VQASourceV2
from mcp_server.data.jsonl_io import iter_jsonl
from mcp_server.data.stage_i_provenance import (
    build_stage_i_instruction_provenance_v1,
)


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
        if args.reasoning_view_jsonl or args.vqa_source_jsonl:
            raise ValueError(
                "assignment and conservative seed modes are mutually exclusive"
            )
        report = build_registry(
            args.assignment_jsonl,
            args.output_jsonl,
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
