#!/usr/bin/env python3
"""Filter unreasonable stop samples with the configured VLM.

The Phase1 geometry pipeline labels short or low-motion future paths as
``stop``.  Some of those samples are useful safety supervision, e.g. waiting at
a red light, yielding to pedestrians, or stopping before an obstacle.  Others
are weak labels caused by recording start/end, robot idle time, or odometry
artifacts.  This tool asks the existing Agent VLM client to classify only stop
rows, keeps non-stop rows unchanged, and writes accepted/rejected manifests for
manual review.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODEL_PROCESSING_DIR = Path(__file__).resolve().parent
if str(MODEL_PROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_PROCESSING_DIR))

from sidewalk_phase1_cot_teacher import (  # noqa: E402
    add_usage_to_report,
    parse_json_object,
    telemetry_usage,
)


SCHEMA_VERSION = "sidewalk_phase1_stop_vlm_filter_v1"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or f"{row.get('clip_id')}_{row.get('frame_idx')}")


def row_intent(row: dict[str, Any]) -> str:
    route_context = row.get("route_context")
    if isinstance(route_context, dict) and route_context.get("intent_gt"):
        return str(route_context["intent_gt"])
    sidewalk_meta = row.get("sidewalk_meta")
    if isinstance(sidewalk_meta, dict) and sidewalk_meta.get("intent_gt"):
        return str(sidewalk_meta["intent_gt"])
    return str(row.get("intent_gt") or "unknown")


def route_theta(row: dict[str, Any]) -> float | None:
    for container_name in ("route_context", "sidewalk_meta"):
        container = row.get(container_name)
        if isinstance(container, dict) and container.get("route_theta_deg") is not None:
            try:
                return float(container["route_theta_deg"])
            except Exception:
                return None
    return None


def load_existing_decisions(path: Path) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return decisions
    for row in read_jsonl(path):
        sample_id = str(row.get("id") or "")
        if sample_id:
            decisions[sample_id] = row
    return decisions


def split_api_keys(value: str) -> list[str]:
    keys: list[str] = []
    for raw in str(value or "").replace("\n", ",").split(","):
        key = raw.strip()
        if key and key not in keys:
            keys.append(key)
    return keys


def load_api_key_pool(api_key_pool: str | None, api_key_file: Path | None) -> list[str]:
    keys: list[str] = []
    for value in (
        api_key_pool,
        os.getenv("ARK_API_KEYS", ""),
        os.getenv("LLM_API_KEYS", ""),
    ):
        for key in split_api_keys(value or ""):
            if key not in keys:
                keys.append(key)

    if api_key_file is not None and api_key_file.is_file():
        for line in api_key_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for key in split_api_keys(stripped):
                if key not in keys:
                    keys.append(key)
    return keys


def build_stop_filter_system_prompt() -> str:
    return (
        "You are a strict sidewalk-navigation dataset quality reviewer. "
        "Your job is to decide whether a geometry label 'stop' is a reasonable "
        "navigation supervision target from the current FPV image and aligned BEV image. "
        "Return only one compact JSON object. Do not use markdown."
    )


def build_stop_filter_question(row: dict[str, Any]) -> str:
    rc = row.get("route_context") or {}
    meta = row.get("sidewalk_meta") or {}
    traj = row.get("trajectory_fpv_t") or row.get("pixel_trajectory") or []
    ego = row.get("trajectory_ego_t") or []
    payload = {
        "sample_id": row.get("id"),
        "clip_id": row.get("clip_id") or meta.get("clip_id"),
        "frame_idx": row.get("frame_idx") or meta.get("frame_idx"),
        "frame_name": row.get("frame_name") or meta.get("frame_name"),
        "route_context": rc,
        "sidewalk_meta": {
            "bev_time_delta_ms": meta.get("bev_time_delta_ms"),
            "trajectory_source": meta.get("trajectory_source"),
            "pose_source": meta.get("pose_source"),
        },
        "trajectory_summary": {
            "num_points": len(traj),
            "first_fpv": traj[0][:2] if isinstance(traj, list) and traj else None,
            "last_fpv": traj[-1][:2] if isinstance(traj, list) and traj else None,
            "last_ego": ego[-1][:2] if isinstance(ego, list) and ego else None,
        },
    }
    return (
        "Classify whether this stop-labeled sidewalk navigation sample should be kept.\n"
        "Use FPV as primary evidence for current-frame safety and BEV only for route/topology context.\n\n"
        "KEEP the stop sample only when the current scene contains credible reasons to stop or yield, such as:\n"
        "- red traffic light, crossing signal, intersection waiting, curb/road-crossing wait;\n"
        "- pedestrian or cyclist immediately blocking or crossing the intended path;\n"
        "- vehicle/scooter/obstacle blocking the near-field walkable route;\n"
        "- narrow/unsafe/occluded passage where a cautious robot should hold.\n\n"
        "DROP the stop sample when the image shows a clear path and the stop label likely comes from:\n"
        "- robot not started, recording idle, clip boundary, or operator pause;\n"
        "- no visible safety reason to stop;\n"
        "- far background object that does not block the near-field route;\n"
        "- odometry/trajectory artifact or ambiguous evidence.\n\n"
        "Return exactly one JSON object with this schema:\n"
        "{\n"
        "  \"decision\": \"keep|drop|unsure\",\n"
        "  \"reason_type\": \"red_light|pedestrian_yield|dynamic_obstacle|static_obstacle|intersection_wait|occlusion_or_narrow_passage|clear_path_idle|clip_boundary_or_robot_idle|artifact_or_bad_label|ambiguous\",\n"
        "  \"confidence\": \"high|medium|low\",\n"
        "  \"visible_stop_cue\": \"short description of the visual cue, or none\",\n"
        "  \"near_field_blocked\": true,\n"
        "  \"review_note\": \"one concise sentence\"\n"
        "}\n\n"
        "Decision policy:\n"
        "- Use decision=keep for high/medium confidence valid stop causes.\n"
        "- Use decision=drop for clear_path_idle, clip_boundary_or_robot_idle, artifact_or_bad_label, or no visible stop cue.\n"
        "- Use decision=unsure only when the images are genuinely ambiguous; these will be kept for human review by default.\n\n"
        "Evidence JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def load_agent_llm_helpers():
    from mcp_server.llm import VisionAnalyzer, local_image_path_to_data_url

    return VisionAnalyzer, local_image_path_to_data_url


async def call_vlm_stop_filter(
    row: dict[str, Any],
    *,
    image_detail: str,
    api_key: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    VisionAnalyzer, local_image_path_to_data_url = load_agent_llm_helpers()
    analyzer = VisionAnalyzer()
    if api_key:
        analyzer.settings = replace(analyzer.settings, api_key=api_key)
    if not analyzer.is_available():
        raise RuntimeError("VLM stop filter is not configured. Set ARK_API_KEY, LLM_API_KEY, ARK_API_KEYS, or LLM_API_KEYS.")

    fpv_url = local_image_path_to_data_url(str(row["fpv_image_path"]))
    extra_urls: list[str] = []
    if row.get("bev_image_path"):
        extra_urls.append(local_image_path_to_data_url(str(row["bev_image_path"])))
    result = await analyzer.analyze(
        image_url=fpv_url,
        extra_image_urls=extra_urls,
        question=build_stop_filter_question(row),
        system_prompt=build_stop_filter_system_prompt(),
        detail=image_detail,
    )
    answer = str(result.get("answer") or "").strip()
    parsed = parse_json_object(answer)
    if parsed is None:
        parsed = {
            "decision": "unsure",
            "reason_type": "ambiguous",
            "confidence": "low",
            "visible_stop_cue": "parse_failed",
            "near_field_blocked": False,
            "review_note": "VLM response could not be parsed as JSON.",
            "raw_answer": answer,
        }
    return parsed, result


def normalize_decision(parsed: dict[str, Any], *, unsure_policy: str, min_keep_confidence: str) -> dict[str, Any]:
    decision = str(parsed.get("decision") or "unsure").strip().lower()
    if decision not in {"keep", "drop", "unsure"}:
        decision = "unsure"
    confidence = str(parsed.get("confidence") or "low").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    confidence_rank = {"low": 0, "medium": 1, "high": 2}
    min_rank = confidence_rank.get(min_keep_confidence, 1)
    if decision == "keep" and confidence_rank.get(confidence, 0) < min_rank:
        decision = "unsure"
    if decision == "unsure":
        decision = unsure_policy

    out = dict(parsed)
    out["decision"] = decision
    out["confidence"] = confidence
    return out


def should_keep_decision(decision: dict[str, Any]) -> bool:
    return str(decision.get("decision") or "").lower() == "keep"


async def evaluate_stop_row(
    row: dict[str, Any],
    *,
    image_detail: str,
    api_key: str | None,
    unsure_policy: str,
    min_keep_confidence: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    async with semaphore:
        parsed, telemetry = await call_vlm_stop_filter(row, image_detail=image_detail, api_key=api_key)
    decision = normalize_decision(
        parsed,
        unsure_policy=unsure_policy,
        min_keep_confidence=min_keep_confidence,
    )
    usage = telemetry_usage(telemetry)
    sample_id = row_id(row)
    decision_row = {
        "id": sample_id,
        "clip_id": row.get("clip_id") or (row.get("sidewalk_meta") or {}).get("clip_id"),
        "frame_idx": row.get("frame_idx") or (row.get("sidewalk_meta") or {}).get("frame_idx"),
        "fpv_image_path": row.get("fpv_image_path"),
        "bev_image_path": row.get("bev_image_path"),
        "route_theta_deg": route_theta(row),
        "stop_filter": decision,
    }
    if usage:
        decision_row["usage"] = usage
    return sample_id, decision, decision_row, telemetry


async def filter_stop_samples(
    *,
    source_jsonl: Path,
    output_jsonl: Path,
    rejected_jsonl: Path,
    decisions_jsonl: Path,
    report_json: Path,
    limit: int | None,
    image_detail: str,
    unsure_policy: str,
    min_keep_confidence: str,
    keep_non_stop: bool,
    progress_interval: int,
    concurrency: int,
    api_key_pool: str | None,
    api_key_file: Path | None,
) -> dict[str, Any]:
    rows = list(read_jsonl(source_jsonl))
    if limit is not None:
        rows = rows[: int(limit)]

    existing_decisions = load_existing_decisions(decisions_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rejected_jsonl.parent.mkdir(parents=True, exist_ok=True)
    decisions_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Rebuild outputs from source + cached decisions so repeated runs are stable.
    output_jsonl.write_text("", encoding="utf-8")
    rejected_jsonl.write_text("", encoding="utf-8")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_jsonl": str(source_jsonl),
        "output_jsonl": str(output_jsonl),
        "rejected_jsonl": str(rejected_jsonl),
        "decisions_jsonl": str(decisions_jsonl),
        "input_rows": len(rows),
        "keep_non_stop": bool(keep_non_stop),
        "unsure_policy": unsure_policy,
        "min_keep_confidence": min_keep_confidence,
        "concurrency": max(1, int(concurrency)),
        "effective_concurrency": max(1, int(concurrency)),
        "api_key_count": 0,
        "counts": {},
        "decision_counts": {},
        "reason_type_counts": {},
    }
    counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    decisions_by_id: dict[str, dict[str, Any]] = {}
    pending_stop_rows: list[dict[str, Any]] = []
    api_keys = load_api_key_pool(api_key_pool, api_key_file)
    report["api_key_count"] = len(api_keys)

    for row in rows:
        sample_id = row_id(row)
        intent = row_intent(row)
        counts[f"input:{intent}"] += 1

        if intent != "stop":
            continue

        cached = existing_decisions.get(sample_id)
        if cached is not None:
            decisions_by_id[sample_id] = dict(cached.get("stop_filter") or {})
            counts["stop_cached_decision"] += 1
        else:
            pending_stop_rows.append(row)

    if pending_stop_rows:
        effective_concurrency = max(1, int(concurrency))
        if api_keys:
            effective_concurrency = min(effective_concurrency, len(api_keys))
        report["effective_concurrency"] = effective_concurrency
        semaphore = asyncio.Semaphore(effective_concurrency)
        tasks = [
            asyncio.create_task(
                evaluate_stop_row(
                    row,
                    image_detail=image_detail,
                    api_key=api_keys[(index - 1) % len(api_keys)] if api_keys else None,
                    unsure_policy=unsure_policy,
                    min_keep_confidence=min_keep_confidence,
                    semaphore=semaphore,
                )
            )
            for index, row in enumerate(pending_stop_rows, start=1)
        ]
        for completed, task in enumerate(asyncio.as_completed(tasks), start=1):
            sample_id, decision, decision_row, telemetry = await task
            decisions_by_id[sample_id] = decision
            add_usage_to_report(report, telemetry)
            append_jsonl(decisions_jsonl, decision_row)
            counts["stop_new_vlm_decision"] += 1
            if progress_interval > 0 and (completed % progress_interval == 0 or completed == len(tasks)):
                print(
                    json.dumps(
                        {
                            "vlm_progress": f"{completed}/{len(tasks)}",
                            "cached": counts.get("stop_cached_decision", 0),
                            "new_vlm": counts.get("stop_new_vlm_decision", 0),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    for idx, row in enumerate(rows, start=1):
        sample_id = row_id(row)
        intent = row_intent(row)
        if intent != "stop":
            if keep_non_stop:
                out = dict(row)
                out.setdefault("stop_filter", {"decision": "keep", "reason_type": "non_stop_passthrough"})
                append_jsonl(output_jsonl, out)
                counts[f"kept:{intent}"] += 1
            continue

        decision = decisions_by_id.get(sample_id)
        if decision is None:
            decision = {
                "decision": "keep" if unsure_policy == "keep" else "drop",
                "reason_type": "ambiguous",
                "confidence": "low",
                "visible_stop_cue": "missing_decision",
                "near_field_blocked": False,
                "review_note": "No VLM decision was available for this stop sample.",
            }
        decision_counts[str(decision.get("decision") or "unknown")] += 1
        reason_counts[str(decision.get("reason_type") or "unknown")] += 1

        out_row = dict(row)
        out_row["stop_filter"] = decision
        if should_keep_decision(decision):
            append_jsonl(output_jsonl, out_row)
            counts["kept:stop"] += 1
        else:
            append_jsonl(rejected_jsonl, out_row)
            counts["rejected:stop"] += 1

        if progress_interval > 0 and (idx % progress_interval == 0 or idx == len(rows)):
            print(
                json.dumps(
                    {
                        "write_progress": f"{idx}/{len(rows)}",
                        "kept_stop": counts.get("kept:stop", 0),
                        "rejected_stop": counts.get("rejected:stop", 0),
                        "new_vlm": counts.get("stop_new_vlm_decision", 0),
                        "cached": counts.get("stop_cached_decision", 0),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    report["counts"] = dict(sorted(counts.items()))
    report["decision_counts"] = dict(sorted(decision_counts.items()))
    report["reason_type_counts"] = dict(sorted(reason_counts.items()))
    write_json(report_json, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--rejected-jsonl", required=True)
    parser.add_argument("--decisions-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image-detail", default="auto", choices=["auto", "low", "high"])
    parser.add_argument("--unsure-policy", default="keep", choices=["keep", "drop"])
    parser.add_argument("--min-keep-confidence", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--drop-non-stop", action="store_true", help="Only write kept stop rows to output.")
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent VLM calls for uncached stop rows.")
    parser.add_argument(
        "--api-key-pool",
        default=None,
        help="Comma-separated API keys. Also supports ARK_API_KEYS or LLM_API_KEYS env vars.",
    )
    parser.add_argument(
        "--api-key-file",
        default=None,
        help="Text file containing one API key per line. Blank lines and # comments are ignored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = asyncio.run(
        filter_stop_samples(
            source_jsonl=Path(args.source_jsonl),
            output_jsonl=Path(args.output_jsonl),
            rejected_jsonl=Path(args.rejected_jsonl),
            decisions_jsonl=Path(args.decisions_jsonl),
            report_json=Path(args.report_json),
            limit=args.limit,
            image_detail=str(args.image_detail),
            unsure_policy=str(args.unsure_policy),
            min_keep_confidence=str(args.min_keep_confidence),
            keep_non_stop=not bool(args.drop_non_stop),
            progress_interval=int(args.progress_interval),
            concurrency=int(args.concurrency),
            api_key_pool=args.api_key_pool,
            api_key_file=Path(args.api_key_file) if args.api_key_file else None,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
