#!/usr/bin/env python3
"""Apply authorized V4.6 teacher/grounding repairs append-only."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Sequence

from mcp_server.contracts.reasoning_dataset_v4 import (
    ReasoningAnnotationV4,
    ReasoningRepairRequestV4,
    ReasoningSourceV4,
)
from mcp_server.data.io.jsonl import append_jsonl_row, iter_jsonl
from mcp_server.teacher.reasoning_v4_batch import (
    ReasoningV4BatchConfig,
    VisionReasoningV4TeacherBackend,
    repair_reasoning_annotation_v4,
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse append-only repair-runner arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--annotation-jsonl", required=True, type=Path)
    parser.add_argument("--repair-request-jsonl", required=True, type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--image-detail",
        choices=("auto", "low", "high"),
        default="low",
    )
    parser.add_argument(
        "--navigation-context-image-detail",
        choices=("auto", "low", "high"),
        default="high",
    )
    parser.add_argument("--section-image-policy-id")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


async def repair_reasoning_v4_manifest(
    *,
    source_jsonl: str | Path,
    annotation_jsonl: str | Path,
    repair_request_jsonl: str | Path,
    api_key: str | None = None,
    config: ReasoningV4BatchConfig | None = None,
) -> dict[str, Any]:
    """Apply valid partial repairs and append each new annotation revision.

    Args:
        source_jsonl: Frozen V4.6 source manifest.
        annotation_jsonl: Existing append-only annotations and destination.
        repair_request_jsonl: Human-authorized immutable repair requests.
        api_key: Optional provider key for the reviewer backend.
        config: Current annotation prompt and image policy.

    Returns:
        Repair report. Strong-sidecar requests are reported as upstream-blocked
        and never mutate the annotation manifest.
    """

    sources = _source_index_v4(source_jsonl)
    annotations = _annotation_index_v4(annotation_jsonl)
    requests = [
        ReasoningRepairRequestV4.model_validate(row)
        for row in iter_jsonl(repair_request_jsonl)
    ]
    backend = VisionReasoningV4TeacherBackend(api_key=api_key)
    cfg = config or ReasoningV4BatchConfig()
    output_path = Path(annotation_jsonl).expanduser()
    appended: list[dict[str, str]] = []
    blocked: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(
        "a",
        encoding="utf-8",
        buffering=1,
    ) as handle:
        for request in requests:
            source = sources.get(request.source_group_id)
            annotation = annotations.get(request.source_group_id)
            if source is None or annotation is None:
                errors.append(
                    {
                        "repair_request_id": request.repair_request_id,
                        "error": "source_or_annotation_missing",
                    }
                )
                continue
            if request.repair_action == "rebuild_strong_sidecar":
                blocked.append(
                    {
                        "repair_request_id": request.repair_request_id,
                        "reason": (
                            "new_source_fingerprint_and_full_reannotation_required"
                        ),
                    }
                )
                continue
            try:
                repaired = await repair_reasoning_annotation_v4(
                    source,
                    annotation,
                    request,
                    backend=backend,
                    config=cfg,
                )
            except Exception as exc:
                errors.append(
                    {
                        "repair_request_id": request.repair_request_id,
                        "error": (
                            f"{type(exc).__name__}:{str(exc)[:1000]}"
                        ),
                    }
                )
                continue
            append_jsonl_row(
                handle,
                repaired.model_dump(mode="json", exclude_none=True),
            )
            annotations[request.source_group_id] = repaired
            appended.append(
                {
                    "repair_request_id": request.repair_request_id,
                    "source_group_id": request.source_group_id,
                    "new_audit_id": (
                        repaired.action_reconciliation.audit_id
                        if repaired.action_reconciliation is not None
                        else ""
                    ),
                }
            )
    return {
        "schema_version": (
            "urbannav_reasoning_v4_6_repair_run_report_v1"
        ),
        "source_jsonl": str(source_jsonl),
        "annotation_jsonl": str(annotation_jsonl),
        "repair_request_jsonl": str(repair_request_jsonl),
        "request_rows": len(requests),
        "appended_rows": len(appended),
        "upstream_blocked_rows": len(blocked),
        "error_rows": len(errors),
        "appended": appended,
        "upstream_blocked": blocked,
        "errors": errors,
        "passed": not errors and not blocked,
    }


def _source_index_v4(
    path: str | Path,
) -> dict[str, ReasoningSourceV4]:
    """Load unique source rows by group."""

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
    """Load latest append-only annotation revisions by group."""

    result: dict[str, ReasoningAnnotationV4] = {}
    for row in iter_jsonl(path):
        annotation = ReasoningAnnotationV4.model_validate(row)
        result[annotation.source_group_id] = annotation
    return result


def _read_api_key(path: Path | None) -> str | None:
    """Read the first non-comment key without serializing it."""

    if path is None:
        return None
    for line in path.expanduser().read_text(
        encoding="utf-8"
    ).splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            return value
    raise ValueError("api key file contains no usable key")


def main(argv: Sequence[str] | None = None) -> None:
    """Run append-only repairs and write the machine-readable report."""

    args = parse_args(argv)
    defaults = ReasoningV4BatchConfig()
    config = ReasoningV4BatchConfig(
        max_retries=max(0, args.max_retries),
        image_detail=args.image_detail,
        navigation_context_image_detail=(
            args.navigation_context_image_detail
        ),
        section_image_policy_id=(
            args.section_image_policy_id
            or defaults.section_image_policy_id
        ),
    )
    report = asyncio.run(
        repair_reasoning_v4_manifest(
            source_jsonl=args.source_jsonl,
            annotation_jsonl=args.annotation_jsonl,
            repair_request_jsonl=args.repair_request_jsonl,
            api_key=_read_api_key(args.api_key_file),
            config=config,
        )
    )
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
