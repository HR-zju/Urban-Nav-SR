#!/usr/bin/env python3
"""Refine soft natural-language fields in repaired SideWalk Phase1 CoT rows.

This script is intentionally narrower than ``sidewalk_phase1_cot_teacher.py``.
It assumes geometry fields have already been repaired from fixed odom/GNSS
source rows, then asks a VLM teacher to improve only human-readable reasoning
fields.  Frozen fields such as intent, target direction, trajectory, trajectory
class, route direction bin, and motion labels are copied from the repaired CoT
row and cannot be changed by the VLM output.

Typical flow:

1. Generate/fix Phase1 source JSONL from corrected odometry.
2. Deterministically repair old CoT rows against that fixed source.
3. Run this script to refine only soft fields using FPV + BEV images.

The output is a normal ms-swift CoT manifest row.  On teacher failure the default
policy keeps the deterministic repaired row, so the dataset stays complete.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

# The companion teacher module owns the strict SideWalk CoT contract.  Reusing
# its parser and validator keeps this refiner from drifting into a second schema.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sidewalk_phase1_cot_teacher as cot

REFINE_SCHEMA_VERSION = "sidewalk_phase1_cot_soft_refine_v1"
SOFT_FIELDS: dict[str, set[str]] = {
    "BEVContext": {"route_summary"},
    "Plan": {"primary_route", "safety_policy", "contingency"},
}


class SoftRefineError(RuntimeError):
    """Raised when a repaired row cannot be parsed or refined."""


def sample_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("sample_id") or "")


def extract_answer(row: dict[str, Any]) -> str:
    for key in ("response", "answer", "teacher_answer", "text"):
        if row.get(key) is not None:
            return str(row[key])
    messages = row.get("messages") or []
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return str(message.get("content") or "")
    return ""


def load_rows_by_id(path: Path) -> tuple[dict[str, dict[str, Any]], int, int]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    duplicates = 0
    missing_ids = 0
    for row in cot.read_jsonl(path):
        sid = sample_id(row)
        if not sid:
            missing_ids += 1
            continue
        if sid in rows_by_id:
            duplicates += 1
            continue
        rows_by_id[sid] = row
    return rows_by_id, duplicates, missing_ids


def parse_answer_sections(answer: str) -> dict[str, dict[str, Any]]:
    think = cot.extract_tag(answer, "think")
    if think is None:
        raise SoftRefineError("missing_think")
    sections: dict[str, dict[str, Any]] = {}
    for tag in cot.THINK_SECTION_TAGS:
        body = cot.extract_tag(think, tag)
        if body is None:
            raise SoftRefineError(f"missing_tag:{tag}")
        parsed = cot.parse_json_object(body)
        if parsed is None:
            raise SoftRefineError(f"invalid_json:{tag}")
        sections[tag] = parsed
    finish_body = cot.extract_tag(answer, "finish")
    if finish_body is None:
        raise SoftRefineError("missing_tag:finish")
    finish = cot.parse_json_object(finish_body)
    if finish is None:
        raise SoftRefineError("invalid_json:finish")
    sections["finish"] = finish
    return sections


def render_answer_sections(sections: dict[str, dict[str, Any]]) -> str:
    lines = ["<think>"]
    for tag in cot.THINK_SECTION_TAGS:
        payload = sections.get(tag)
        if not isinstance(payload, dict):
            raise SoftRefineError(f"missing_section:{tag}")
        lines.append(f"<{tag}>{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}</{tag}>")
    finish = sections.get("finish")
    if not isinstance(finish, dict):
        raise SoftRefineError("missing_section:finish")
    lines.extend([
        "</think>",
        f"<finish>{json.dumps(finish, ensure_ascii=False, separators=(',', ':'))}</finish>",
    ])
    return "\n".join(lines)


def soft_field_snapshot(sections: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        tag: {field: sections.get(tag, {}).get(field) for field in sorted(fields)}
        for tag, fields in SOFT_FIELDS.items()
    }


def frozen_field_snapshot(sections: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    frozen: dict[str, dict[str, Any]] = {}
    for tag, payload in sections.items():
        if not isinstance(payload, dict):
            continue
        allowed = SOFT_FIELDS.get(tag, set())
        frozen[tag] = {key: value for key, value in payload.items() if key not in allowed}
    return frozen


def merge_soft_fields(
    base_sections: dict[str, dict[str, Any]],
    candidate_sections: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    """Copy only whitelisted soft fields from candidate into base sections."""

    merged = deepcopy(base_sections)
    changed: list[str] = []
    attempted_frozen_changes: list[str] = []
    for tag, fields in SOFT_FIELDS.items():
        candidate = candidate_sections.get(tag)
        if not isinstance(candidate, dict):
            continue
        base = base_sections.get(tag, {})
        for key, value in candidate.items():
            if key in fields:
                if merged.setdefault(tag, {}).get(key) != value:
                    changed.append(f"{tag}.{key}")
                merged[tag][key] = value
            elif key in base and value != base.get(key):
                attempted_frozen_changes.append(f"{tag}.{key}")
    return merged, changed, attempted_frozen_changes


def refine_validation_is_acceptable(validation: cot.ValidationResult) -> bool:
    """Allow preserved non-trajectory CoT fields to remain unchanged.

    The refiner is intentionally limited to route_summary and Plan prose.  The
    base repaired CoT may preserve old VLM route bins or anchors because those
    fields are visual-semantic annotations, not direct odom targets.  Strict
    teacher validation can flag those fields, but they should not force the
    refined row to fall back when the trajectory and allowed soft fields are
    otherwise valid.
    """

    if validation.valid:
        return True
    ignorable_prefixes = (
        "route_direction_bin_mismatch:",
    )
    return bool(validation.errors) and all(
        any(error.startswith(prefix) for prefix in ignorable_prefixes)
        for error in validation.errors
    )


def build_refine_system_prompt() -> str:
    return (
        "You refine only human-readable reasoning fields for a sidewalk navigation CoT dataset. "
        "The provided frozen fields are ground truth or validated geometry-derived labels. "
        "Do not change frozen fields, trajectory, intent, route direction, CrossView grounding, motion labels, or finish JSON. "
        "Use the FPV image and BEV image to make the allowed soft fields more specific, concise, and visually grounded. "
        "Return the complete required tag structure and valid JSON only."
    )


def build_refine_question(source_row: dict[str, Any], base_sections: dict[str, dict[str, Any]], *, previous_answer: str | None = None, errors: list[str] | None = None) -> str:
    repair_block = ""
    if previous_answer is not None:
        repair_block = (
            "\nPrevious answer failed validation.\n"
            f"Errors: {json.dumps(errors or [], ensure_ascii=False)}\n"
            "Repair only the allowed soft fields. Frozen fields must remain unchanged.\n"
        )
    payload = {
        "sample_id": sample_id(source_row),
        "instruction": source_row.get("instruction"),
        "image_size": {"width": source_row.get("image_width"), "height": source_row.get("image_height")},
        "allowed_soft_fields": {tag: sorted(fields) for tag, fields in SOFT_FIELDS.items()},
        "frozen_fields": frozen_field_snapshot(base_sections),
        "current_soft_fields": soft_field_snapshot(base_sections),
        "phase1_limitations": {
            "has_segmentation": False,
            "forbidden": "Do not invent mask ids, region ids, mIoU, support_ratio, or precise route angles.",
        },
    }
    return (
        "Refine this repaired structured CoT sample.\n"
        "You may rewrite ONLY the fields listed in allowed_soft_fields.\n"
        "All frozen_fields must be copied exactly.\n"
        "Return the full nested structure in this exact order:\n"
        "<think>\n"
        "<Scene>{...}</Scene>\n"
        "<FPVScene>{...}</FPVScene>\n"
        "<BEVContext>{...}</BEVContext>\n"
        "<CrossView>{...}</CrossView>\n"
        "<Plan>{...}</Plan>\n"
        "</think>\n"
        "<finish>{...}</finish>\n\n"
        "Do not add markdown or prose outside the tags.\n\n"
        f"{cot.COT_BEV_VISUAL_CONVENTION}\n"
        "Evidence JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        f"{repair_block}"
    )


async def call_llm_refiner(
    source_row: dict[str, Any],
    base_sections: dict[str, dict[str, Any]],
    *,
    image_detail: str,
    previous_answer: str | None = None,
    errors: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    VisionAnalyzer, local_image_path_to_data_url = cot.load_agent_llm_helpers()
    analyzer = VisionAnalyzer()
    if not analyzer.is_available():
        raise RuntimeError("LLM/VLM refiner is not configured. Set ARK_API_KEY or LLM_API_KEY.")

    fpv_url = local_image_path_to_data_url(source_row["fpv_image_path"])
    extra_urls: list[str] = []
    if source_row.get("bev_image_path"):
        extra_urls.append(local_image_path_to_data_url(source_row["bev_image_path"]))
    result = await analyzer.analyze(
        image_url=fpv_url,
        extra_image_urls=extra_urls,
        question=build_refine_question(source_row, base_sections, previous_answer=previous_answer, errors=errors),
        system_prompt=build_refine_system_prompt(),
        detail=image_detail,
    )
    return str(result.get("answer") or "").strip(), result


def inc(counter: dict[str, int], key: str) -> None:
    counter[key] = int(counter.get(key) or 0) + 1


async def refine_soft_fields(
    source_jsonl: str,
    repaired_cot_jsonl: str,
    output_jsonl: str,
    *,
    report_json: str | None = None,
    limit: int | None = None,
    expected_points: int = 16,
    trajectory_tolerance: float = 0.025,
    max_retries: int = 1,
    image_detail: str = "auto",
    progress_interval: int = 25,
    fallback_policy: str = "keep",
) -> dict[str, Any]:
    source_rows = list(cot.read_jsonl(Path(source_jsonl)))
    if limit is not None:
        source_rows = source_rows[: int(limit)]
    repaired_by_id, duplicate_repaired, missing_repaired_id = load_rows_by_id(Path(repaired_cot_jsonl))

    output_path = Path(output_jsonl)
    existing_rows = cot.count_jsonl_rows(output_path)
    if existing_rows > len(source_rows):
        raise RuntimeError(f"Existing output has more rows than source: {output_jsonl}")

    cfg = cot.TeacherConfig(expected_points=int(expected_points), trajectory_tolerance=float(trajectory_tolerance))
    report: dict[str, Any] = {
        "schema_version": f"{REFINE_SCHEMA_VERSION}_report",
        "source_jsonl": str(source_jsonl),
        "repaired_cot_jsonl": str(repaired_cot_jsonl),
        "output_jsonl": str(output_jsonl),
        "input_rows": len(source_rows),
        "repaired_rows_available": len(repaired_by_id),
        "duplicate_repaired_rows": duplicate_repaired,
        "missing_repaired_id_rows": missing_repaired_id,
        "processed_rows": existing_rows,
        "resumed_rows": existing_rows,
        "written_rows": existing_rows,
        "refined_rows": 0,
        "fallback_rows": 0,
        "dropped_rows": 0,
        "missing_repaired_rows": 0,
        "base_invalid_rows": 0,
        "attempted_frozen_change_rows": 0,
        "changed_field_counts": {},
        "error_counts": {},
        "status": "running",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    print(
        f"[cot-soft-refine] starting source={source_jsonl} rows={len(source_rows)} "
        f"resume={existing_rows} output={output_jsonl}",
        flush=True,
    )

    with output_path.open("a" if existing_rows else "w", encoding="utf-8", buffering=1) as out_f:
        for index, source_row in enumerate(source_rows[existing_rows:], start=existing_rows):
            sid = sample_id(source_row)
            repaired_row = repaired_by_id.get(sid)
            if repaired_row is None:
                report["missing_repaired_rows"] += 1
                report["processed_rows"] = index + 1
                if fallback_policy == "error":
                    raise RuntimeError(f"Missing repaired CoT row for {sid}")
                continue

            base_answer = extract_answer(repaired_row)
            try:
                base_sections = parse_answer_sections(base_answer)
            except SoftRefineError as exc:
                inc(report["error_counts"], str(exc).split(":", 1)[0])
                report["base_invalid_rows"] += 1
                report["processed_rows"] = index + 1
                if fallback_policy == "error":
                    raise
                if fallback_policy == "keep":
                    out_f.write(json.dumps(repaired_row, ensure_ascii=False) + "\n")
                    report["written_rows"] += 1
                    report["fallback_rows"] += 1
                continue

            evidence = cot.load_evidence(source_row)
            base_validation = cot.validate_teacher_answer(base_answer, source_row, evidence, cfg)
            if not base_validation.valid:
                report["base_invalid_rows"] += 1

            candidate_answer = ""
            telemetry: dict[str, Any] | None = None
            validation = cot.ValidationResult(False, ["not_started"], {})
            changed_fields: list[str] = []
            attempted_frozen_changes: list[str] = []
            teacher_model: str | None = None
            for attempt in range(int(max_retries) + 1):
                try:
                    candidate_answer, telemetry = await call_llm_refiner(
                        source_row,
                        base_sections,
                        image_detail=image_detail,
                        previous_answer=candidate_answer if attempt else None,
                        errors=validation.errors if attempt else None,
                    )
                    cot.add_usage_to_report(report, telemetry)
                    candidate_sections = parse_answer_sections(candidate_answer)
                    merged_sections, changed_fields, attempted_frozen_changes = merge_soft_fields(base_sections, candidate_sections)
                    merged_answer = render_answer_sections(merged_sections)
                    validation = cot.validate_teacher_answer(merged_answer, source_row, evidence, cfg)
                    if not validation.valid and refine_validation_is_acceptable(validation):
                        validation = cot.ValidationResult(True, [], validation.sections)
                    if validation.valid:
                        candidate_answer = merged_answer
                        break
                except RuntimeError as exc:
                    if "not configured" in str(exc).lower():
                        raise
                    validation = cot.ValidationResult(False, [f"refine_failed:{type(exc).__name__}:{str(exc)[:200]}"], {})
                except Exception as exc:
                    validation = cot.ValidationResult(False, [f"refine_failed:{type(exc).__name__}:{str(exc)[:200]}"], {})

            if isinstance(telemetry, dict):
                llm_telemetry = cot.telemetry_llm_metadata(telemetry)
                teacher_model = str(llm_telemetry.get("model") or telemetry.get("model") or "") or None

            if validation.valid:
                if attempted_frozen_changes:
                    report["attempted_frozen_change_rows"] += 1
                    for field in attempted_frozen_changes:
                        inc(report.setdefault("attempted_frozen_change_counts", {}), field)
                for field in changed_fields:
                    inc(report["changed_field_counts"], field)
                out_row = cot.build_cot_manifest_row(
                    source_row,
                    answer=candidate_answer,
                    validation=validation,
                    teacher_source="llm_soft_refine",
                    teacher_model=teacher_model,
                    teacher_usage=cot.telemetry_usage(telemetry),
                )
                out_row.setdefault("cot_teacher", {})["soft_refine"] = {
                    "schema_version": REFINE_SCHEMA_VERSION,
                    "changed_fields": changed_fields,
                    "attempted_frozen_changes": attempted_frozen_changes,
                    "base_teacher_source": (repaired_row.get("cot_teacher") or {}).get("teacher_source"),
                }
                out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                report["written_rows"] += 1
                report["refined_rows"] += 1
            else:
                for error in validation.errors:
                    inc(report["error_counts"], error.split(":", 1)[0])
                if fallback_policy == "error":
                    raise RuntimeError(f"Soft refine failed for {sid}: {validation.errors}")
                if fallback_policy == "keep":
                    # Rebuild on the fixed source row so prompt/images/meta match the new source,
                    # while keeping the deterministic repaired answer unchanged.
                    fallback_row = cot.build_cot_manifest_row(
                        source_row,
                        answer=base_answer,
                        validation=base_validation,
                        teacher_source="soft_refine_fallback_keep",
                    )
                    fallback_row.setdefault("cot_teacher", {})["soft_refine"] = {
                        "schema_version": REFINE_SCHEMA_VERSION,
                        "fallback_reason": validation.errors,
                        "base_teacher_source": (repaired_row.get("cot_teacher") or {}).get("teacher_source"),
                    }
                    out_f.write(json.dumps(fallback_row, ensure_ascii=False) + "\n")
                    report["written_rows"] += 1
                    report["fallback_rows"] += 1
                else:
                    report["dropped_rows"] += 1

            report["processed_rows"] = index + 1
            if (index + 1) % max(1, int(progress_interval)) == 0 or index + 1 == len(source_rows):
                elapsed = max(time.monotonic() - started_at, 1e-6)
                current_run_rows = max(int(report["processed_rows"]) - int(report["resumed_rows"]), 0)
                rate = float(current_run_rows) / elapsed
                remaining = max(len(source_rows) - int(report["processed_rows"]), 0)
                eta_seconds = remaining / rate if rate > 0 else None
                report["rate_rows_per_hour"] = rate * 3600.0
                report["eta_seconds"] = eta_seconds
                cot.write_report(report_json, report)
                eta_text = f"{eta_seconds / 3600.0:.2f}h" if eta_seconds is not None else "unknown"
                print(
                    "[cot-soft-refine] progress "
                    f"{report['processed_rows']}/{len(source_rows)} "
                    f"written={report['written_rows']} refined={report['refined_rows']} "
                    f"fallback={report['fallback_rows']} dropped={report['dropped_rows']} "
                    f"rate={report['rate_rows_per_hour']:.1f}/h eta={eta_text}",
                    flush=True,
                )

    report["status"] = "done"
    report["eta_seconds"] = 0.0
    cot.write_report(report_json, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine only soft natural-language fields in repaired SideWalk Phase1 CoT JSONL")
    parser.add_argument("--source-jsonl", required=True, help="Fixed geometry source JSONL")
    parser.add_argument("--repaired-cot-jsonl", required=True, help="Deterministically repaired CoT JSONL")
    parser.add_argument("--output-jsonl", required=True, help="Soft-refined CoT JSONL")
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--expected-points", type=int, default=16)
    parser.add_argument("--trajectory-tolerance", type=float, default=0.025)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--image-detail", default="auto", choices=["auto", "low", "high"])
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--fallback-policy", default="keep", choices=["keep", "drop", "error"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = asyncio.run(
        refine_soft_fields(
            args.source_jsonl,
            args.repaired_cot_jsonl,
            args.output_jsonl,
            report_json=args.report_json,
            limit=args.limit,
            expected_points=int(args.expected_points),
            trajectory_tolerance=float(args.trajectory_tolerance),
            max_retries=int(args.max_retries),
            image_detail=str(args.image_detail),
            progress_interval=int(args.progress_interval),
            fallback_policy=str(args.fallback_policy),
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
