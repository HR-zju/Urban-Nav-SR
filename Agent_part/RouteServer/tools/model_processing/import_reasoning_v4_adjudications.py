#!/usr/bin/env python3
"""Validate browser-exported V4.6 adjudications and append bound records."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from mcp_server.contracts.reasoning_dataset_v4 import (
    ActionAdjudicationPayloadV4,
    ReasoningAnnotationV4,
    ReasoningSourceV4,
)
from mcp_server.data.jsonl_io import iter_jsonl
from mcp_server.data.reasoning_v4_review import (
    bind_action_adjudication_v4,
    load_latest_action_review_diagnoses_v4,
)
from mcp_server.data.reasoning_v4_state import (
    append_action_adjudication_v4,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse human-adjudication import arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--annotation-jsonl", required=True, type=Path)
    parser.add_argument("--diagnosis-jsonl", required=True, type=Path)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args(argv)


def import_reasoning_v4_adjudications(
    *,
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    diagnosis_jsonl: str | Path,
    input_jsonl: str | Path,
    output_jsonl: str | Path,
) -> dict[str, Any]:
    """Bind and append review-UI payloads against current immutable lineage.

    Args:
        source_jsonl: Frozen V4.6 source manifest.
        annotation_jsonl: Append-only V4.6 annotation manifest.
        diagnosis_jsonl: Completed two-stage diagnosis records.
        input_jsonl: Unbound JSONL downloaded by the review HTML.
        output_jsonl: Append-only canonical adjudication sidecar.

    Returns:
        Import report with exact appended identities.

    Raises:
        ValueError: If any UI identity, semantic payload, or parent lineage is
            invalid.
    """

    sources = _source_index_v4(source_jsonl)
    annotations = _annotation_index_v4(annotation_jsonl)
    diagnoses = load_latest_action_review_diagnoses_v4(
        diagnosis_jsonl
    )
    appended: list[str] = []
    for row in iter_jsonl(input_jsonl):
        source_id = str(row.get("source_group_id") or "")
        audit_id = str(row.get("audit_id") or "")
        diagnosis_id = str(row.get("diagnosis_id") or "")
        reviewer = str(row.get("reviewer") or "")
        reviewed_at_raw = str(row.get("reviewed_at") or "")
        if not all(
            (
                source_id,
                audit_id,
                diagnosis_id,
                reviewer,
                reviewed_at_raw,
            )
        ):
            raise ValueError(
                "adjudication UI row is missing required identity fields"
            )
        source = sources.get(source_id)
        annotation = annotations.get(source_id)
        diagnosis = diagnoses.get(audit_id)
        if source is None or annotation is None or diagnosis is None:
            raise ValueError(
                f"adjudication lineage is unavailable: {source_id}/{audit_id}"
            )
        if (
            diagnosis.diagnosis is None
            or diagnosis.diagnosis.diagnosis_id != diagnosis_id
        ):
            raise ValueError(
                f"adjudication diagnosis_id is stale: {source_id}"
            )
        semantic = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "source_group_id",
                "audit_id",
                "diagnosis_id",
                "reviewer",
                "reviewed_at",
            }
        }
        payload = ActionAdjudicationPayloadV4.model_validate(
            semantic
        )
        adjudication = bind_action_adjudication_v4(
            source,
            annotation,
            diagnosis,
            payload,
            reviewer=reviewer,
            reviewed_at=datetime.fromisoformat(reviewed_at_raw),
        )
        append_action_adjudication_v4(
            output_jsonl,
            adjudication,
        )
        appended.append(adjudication.adjudication_id)
    return {
        "schema_version": (
            "urbannav_action_adjudication_import_report_v1"
        ),
        "source_jsonl": str(source_jsonl),
        "annotation_jsonl": str(annotation_jsonl),
        "diagnosis_jsonl": str(diagnosis_jsonl),
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "appended_rows": len(appended),
        "adjudication_ids": appended,
    }


def _source_index_v4(
    path: str | Path,
) -> dict[str, ReasoningSourceV4]:
    """Load unique frozen sources by source-group identity."""

    result: dict[str, ReasoningSourceV4] = {}
    for row in iter_jsonl(path):
        source = ReasoningSourceV4.model_validate(row)
        source_id = source.observation.source_group_id
        if source_id in result:
            raise ValueError(
                f"duplicate source_group_id: {source_id}"
            )
        result[source_id] = source
    return result


def _annotation_index_v4(
    path: str | Path,
) -> dict[str, ReasoningAnnotationV4]:
    """Load the latest append-only annotation per source group."""

    result: dict[str, ReasoningAnnotationV4] = {}
    for row in iter_jsonl(path):
        annotation = ReasoningAnnotationV4.model_validate(row)
        result[annotation.source_group_id] = annotation
    return result


def main(argv: Sequence[str] | None = None) -> None:
    """Run the adjudication importer and emit its report."""

    args = parse_args(argv)
    report = import_reasoning_v4_adjudications(
        source_jsonl=args.source_jsonl,
        annotation_jsonl=args.annotation_jsonl,
        diagnosis_jsonl=args.diagnosis_jsonl,
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
    )
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
