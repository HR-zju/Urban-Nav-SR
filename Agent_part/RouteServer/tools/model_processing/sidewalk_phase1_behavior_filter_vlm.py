#!/usr/bin/env python3
"""Filter SideWalk Phase1 sample rows by VLM visual interestingness.

The paper-style behavior filter classifies first-person robot observations as
``1`` for interesting or ``2`` for boring.  In the current dataset pipeline this
is a sample-level selector: it reads an already-built JSONL manifest and sends
the FPV frames embedded in each sample row.  It does not scan or filter raw
videos.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
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

from sidewalk_phase1_cot_teacher import add_usage_to_report, telemetry_usage  # noqa: E402


SCHEMA_VERSION = "sidewalk_phase1_behavior_filter_vlm_v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Read a JSONL file.

    Args:
        path: JSONL path.

    Returns:
        Iterator of decoded dictionary rows.

    Raises:
        ValueError: If any non-empty line is invalid JSON.
    """

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
            if isinstance(row, dict):
                yield row


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one row to a JSONL file.

    Args:
        path: Output JSONL path.
        row: JSON-serializable row.

    Returns:
        ``None``.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write an indented JSON object.

    Args:
        path: Output JSON path.
        payload: JSON-serializable payload.

    Returns:
        ``None``.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def row_id(row: dict[str, Any], index: int | None = None) -> str:
    """Return a stable manifest row id.

    Args:
        row: Source manifest row.
        index: Optional source line index for rows without ids.

    Returns:
        Stable id string.
    """

    raw = row.get("id")
    if raw:
        return str(raw)
    clip_id = row.get("clip_id") or (row.get("sidewalk_meta") or {}).get("clip_id")
    frame_idx = row.get("frame_idx") or (row.get("sidewalk_meta") or {}).get("frame_idx")
    if clip_id is not None and frame_idx is not None:
        return f"{clip_id}_{frame_idx}"
    return f"__index_{index}" if index is not None else "__missing_id"


def row_intent(row: dict[str, Any]) -> str:
    """Extract intent from common SideWalk manifest locations.

    Args:
        row: Manifest row.

    Returns:
        Intent string or ``unknown``.
    """

    route_context = row.get("route_context")
    if isinstance(route_context, dict) and route_context.get("intent_gt"):
        return str(route_context["intent_gt"])
    sidewalk_meta = row.get("sidewalk_meta")
    if isinstance(sidewalk_meta, dict) and sidewalk_meta.get("intent_gt"):
        return str(sidewalk_meta["intent_gt"])
    return str(row.get("intent_gt") or "unknown")


def split_api_keys(value: str) -> list[str]:
    """Split comma/newline-separated API keys.

    Args:
        value: Raw key list text.

    Returns:
        Unique keys in input order.
    """

    keys: list[str] = []
    for raw in str(value or "").replace("\n", ",").split(","):
        key = raw.strip()
        if key and key not in keys:
            keys.append(key)
    return keys


def load_api_key_pool(api_key_pool: str | None, api_key_file: Path | None) -> list[str]:
    """Load API keys from CLI and environment.

    Args:
        api_key_pool: Optional comma-separated keys.
        api_key_file: Optional text file containing one key per line.

    Returns:
        Unique keys in priority order.
    """

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


def load_existing_decisions(path: Path) -> dict[str, dict[str, Any]]:
    """Load cached VLM behavior decisions by id.

    Args:
        path: Existing decisions JSONL path.

    Returns:
        Mapping from sample id to decision row.
    """

    decisions: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return decisions
    for row in read_jsonl(path):
        sample_id = str(row.get("id") or "")
        if sample_id:
            decisions[sample_id] = row
    return decisions


def build_behavior_system_prompt() -> str:
    """Build the behavior-filter system prompt.

    Returns:
        System prompt for sample-level binary interestingness.
    """

    return (
        "You are an expert first-person robot navigation sample reviewer.\n"
        "Your task is to analyze a temporal FPV sample from a delivery robot "
        "in an urban environment.\n\n"
        "Determine if this sample is \"interesting\" or \"boring\" for navigation training.\n\n"
        "INTERESTING scenes include:\n"
        "- Interactions with pedestrians\n"
        "- Interactions with objects\n"
        "- Complex terrain navigation\n"
        "- Traffic situations\n"
        "- Environmental interactions\n"
        "- Dynamic obstacles or unexpected situations\n\n"
        "BORING scenes include:\n"
        "- Simple straight-line walking on sidewalks\n"
        "- Repetitive movement without interactions\n"
        "- Just following a path without any challenges\n\n"
        "Please respond with only \"1\" for interesting or \"2\" for boring."
    )


def build_behavior_question(row: dict[str, Any], frame_paths: list[str]) -> str:
    """Build the per-row behavior-filter user question.

    Args:
        row: Source manifest row.
        frame_paths: FPV keyframes sent to the VLM in temporal order.

    Returns:
        User question string.
    """

    payload = {
        "sample_id": row.get("id"),
        "clip_id": row.get("clip_id") or (row.get("sidewalk_meta") or {}).get("clip_id"),
        "frame_idx": row.get("frame_idx") or (row.get("sidewalk_meta") or {}).get("frame_idx"),
        "frame_count": len(frame_paths),
        "temporal_order": "oldest_to_newest",
        "view": "first_person_fpv_only",
    }
    return (
        "Classify the provided temporal FPV frames as one already-built training sample.\n"
        "The images are ordered from oldest to newest; the final image is the supervised current frame.\n"
        "Return only the single character 1 or 2.\n\n"
        "Metadata JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def parse_behavior_answer(answer: str) -> int | None:
    """Parse a binary interestingness answer.

    Args:
        answer: Raw VLM answer.

    Returns:
        ``1`` for interesting, ``2`` for boring, or ``None`` on parse failure.
    """

    text = str(answer or "").strip()
    if text in {"1", "2"}:
        return int(text)
    match = re.search(r"\b([12])\b", text)
    if match:
        return int(match.group(1))
    lowered = text.lower()
    if "interesting" in lowered and "boring" not in lowered:
        return 1
    if "boring" in lowered and "interesting" not in lowered:
        return 2
    return None


def normalize_behavior_decision(
    parsed_label: int | None,
    *,
    raw_answer: str,
    failure_policy: str,
) -> dict[str, Any]:
    """Normalize a raw VLM answer into a behavior decision object.

    Args:
        parsed_label: Parsed binary label.
        raw_answer: Raw answer text.
        failure_policy: Label policy for parse/API failure.

    Returns:
        Decision dictionary.
    """

    if parsed_label in {1, 2}:
        return {
            "schema_version": SCHEMA_VERSION,
            "label": int(parsed_label),
            "label_name": "interesting" if parsed_label == 1 else "boring",
            "valid": True,
            "raw_answer": raw_answer,
        }
    if failure_policy == "interesting":
        label = 1
    elif failure_policy == "boring":
        label = 2
    else:
        label = 0
    return {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "label_name": {0: "unknown", 1: "interesting", 2: "boring"}[label],
        "valid": False,
        "raw_answer": raw_answer,
        "failure_policy": failure_policy,
    }


def is_image_path(value: Any) -> bool:
    """Return whether a value looks like an image file path.

    Args:
        value: Candidate path.

    Returns:
        ``True`` if the suffix is a known image suffix.
    """

    return Path(str(value or "")).suffix.lower() in IMAGE_SUFFIXES


def looks_like_fpv_path(value: Any) -> bool:
    """Heuristically distinguish FPV image paths from BEV/map paths.

    Args:
        value: Candidate image path.

    Returns:
        ``True`` if the path appears to be FPV rather than BEV.
    """

    text = str(value or "").replace("\\", "/").lower()
    name = Path(text).name
    if not is_image_path(text):
        return False
    if "/bev/" in text or "bev_images" in text or name.endswith("_bev.jpg") or name.endswith("_bev.png"):
        return False
    return True


def uniformly_subsample(items: list[str], max_items: int) -> list[str]:
    """Uniformly subsample a list while preserving order.

    Args:
        items: Input item list.
        max_items: Maximum number of output items.

    Returns:
        Ordered subsample.
    """

    if max_items <= 0 or len(items) <= max_items:
        return list(items)
    if max_items == 1:
        return [items[-1]]
    selected: list[str] = []
    last = len(items) - 1
    used: set[int] = set()
    for i in range(max_items):
        idx = int(round(i * last / float(max_items - 1)))
        if idx in used:
            idx = min(last, max(0, idx + 1))
        used.add(idx)
        selected.append(items[idx])
    return selected


def collect_row_fpv_images(row: dict[str, Any], *, max_frames: int) -> list[str]:
    """Collect FPV image paths from common SideWalk row fields.

    Args:
        row: Manifest row.
        max_frames: Maximum returned images.

    Returns:
        FPV image paths in temporal order when known.
    """

    candidates: list[str] = []
    for key in ("fpv_image_paths", "history_fpv_image_paths"):
        value = row.get(key)
        if isinstance(value, list):
            candidates.extend(str(item) for item in value if looks_like_fpv_path(item))
    if row.get("fpv_image_path") and looks_like_fpv_path(row.get("fpv_image_path")):
        candidates.append(str(row["fpv_image_path"]))

    if not candidates and isinstance(row.get("images"), list):
        candidates.extend(str(item) for item in row["images"] if looks_like_fpv_path(item))
        if not candidates:
            candidates.extend(str(item) for item in row["images"] if is_image_path(item))

    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return uniformly_subsample(deduped, max_frames)


def collect_behavior_frames(
    row: dict[str, Any],
    *,
    sample_id: str,
    max_frames: int,
) -> list[str]:
    """Collect sample-row FPV frames used by the behavior filter.

    Args:
        row: Manifest row.
        sample_id: Stable sample id.
        max_frames: Maximum frames to return.

    Returns:
        Existing sample image paths in temporal order.
    """

    _ = sample_id
    return collect_row_fpv_images(row, max_frames=max_frames)


def resolve_frame_paths(frame_paths: list[str], *, manifest_path: Path) -> list[str]:
    """Resolve relative image paths against the source manifest directory.

    Args:
        frame_paths: Raw frame paths collected from a row.
        manifest_path: Source manifest JSONL path.

    Returns:
        Frame paths suitable for local file loading.
    """

    resolved: list[str] = []
    for raw in frame_paths:
        path = Path(str(raw))
        if path.is_absolute():
            resolved.append(str(path))
            continue
        manifest_relative = (manifest_path.parent / path).resolve()
        if manifest_relative.exists():
            resolved.append(str(manifest_relative))
        else:
            resolved.append(str(path))
    return resolved


def load_agent_llm_helpers():
    """Load RouteServer VLM helper classes lazily.

    Returns:
        ``(VisionAnalyzer, local_image_path_to_data_url)``.
    """

    from mcp_server.llm import VisionAnalyzer, local_image_path_to_data_url

    return VisionAnalyzer, local_image_path_to_data_url


async def call_vlm_behavior_filter(
    row: dict[str, Any],
    *,
    frame_paths: list[str],
    image_detail: str,
    api_key: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call the configured VLM for one behavior-filter decision.

    Args:
        row: Source manifest row.
        frame_paths: FPV frame paths in temporal order.
        image_detail: Provider image detail setting.
        api_key: Optional per-worker API key.

    Returns:
        ``(decision, telemetry_result)``.

    Raises:
        RuntimeError: If the VLM client is not configured.
        ValueError: If no frame paths are available.
    """

    if not frame_paths:
        raise ValueError("No FPV frames available for behavior filtering.")

    VisionAnalyzer, local_image_path_to_data_url = load_agent_llm_helpers()
    analyzer = VisionAnalyzer()
    if api_key:
        analyzer.settings = replace(analyzer.settings, api_key=api_key)
    if not analyzer.is_available():
        raise RuntimeError("VLM behavior filter is not configured. Set ARK_API_KEY, LLM_API_KEY, ARK_API_KEYS, or LLM_API_KEYS.")

    urls = [local_image_path_to_data_url(path) for path in frame_paths]
    result = await analyzer.analyze(
        image_url=urls[0],
        extra_image_urls=urls[1:],
        question=build_behavior_question(row, frame_paths),
        system_prompt=build_behavior_system_prompt(),
        detail=image_detail,
    )
    answer = str(result.get("answer") or "").strip()
    decision = normalize_behavior_decision(
        parse_behavior_answer(answer),
        raw_answer=answer,
        failure_policy="unknown",
    )
    return decision, result


def should_keep_decision(decision: dict[str, Any], keep_labels: set[int]) -> bool:
    """Return whether a normalized decision should be kept.

    Args:
        decision: Behavior decision dictionary.
        keep_labels: Integer labels to keep.

    Returns:
        ``True`` when the decision label is in ``keep_labels``.
    """

    try:
        label = int(decision.get("label"))
    except Exception:
        label = 0
    return label in keep_labels


async def evaluate_behavior_row(
    row: dict[str, Any],
    *,
    sample_id: str,
    frame_paths: list[str],
    image_detail: str,
    api_key: str | None,
    api_key_index: int | None,
    failure_policy: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """Evaluate one row with bounded VLM concurrency.

    Args:
        row: Source manifest row.
        sample_id: Stable row id.
        frame_paths: FPV frame paths.
        image_detail: Provider image detail setting.
        api_key: Optional API key for this task.
        api_key_index: 1-based API key index.
        failure_policy: Label policy for API/parse failure.
        semaphore: Async concurrency semaphore.

    Returns:
        ``(sample_id, decision, decision_row, telemetry)``.
    """

    raw_error = None
    telemetry: dict[str, Any] | None = None
    try:
        async with semaphore:
            decision, telemetry = await call_vlm_behavior_filter(
                row,
                frame_paths=frame_paths,
                image_detail=image_detail,
                api_key=api_key,
            )
        if not decision.get("valid"):
            decision = normalize_behavior_decision(
                None,
                raw_answer=str(decision.get("raw_answer") or ""),
                failure_policy=failure_policy,
            )
    except Exception as exc:
        raw_error = f"{type(exc).__name__}:{str(exc)[:240]}"
        decision = normalize_behavior_decision(None, raw_answer="", failure_policy=failure_policy)
        decision["error"] = raw_error

    usage = telemetry_usage(telemetry)
    decision_row: dict[str, Any] = {
        "id": sample_id,
        "clip_id": row.get("clip_id") or (row.get("sidewalk_meta") or {}).get("clip_id"),
        "frame_idx": row.get("frame_idx") or (row.get("sidewalk_meta") or {}).get("frame_idx"),
        "intent_gt": row_intent(row),
        "fpv_frame_paths": frame_paths,
        "behavior_filter": decision,
        "api_key_index": api_key_index,
    }
    if usage:
        decision_row["usage"] = usage
    if raw_error:
        decision_row["error"] = raw_error
    return sample_id, decision, decision_row, telemetry


async def filter_behavior_samples(
    *,
    source_jsonl: Path,
    output_jsonl: Path,
    rejected_jsonl: Path,
    decisions_jsonl: Path,
    report_json: Path,
    limit: int | None,
    image_detail: str,
    max_frames_per_call: int,
    keep_labels: set[int],
    failure_policy: str,
    progress_interval: int,
    concurrency: int,
    api_key_pool: str | None,
    api_key_file: Path | None,
    retry_invalid_cached: bool,
) -> dict[str, Any]:
    """Run behavior filtering over a source manifest.

    Args:
        source_jsonl: Source manifest JSONL.
        output_jsonl: Kept-row output JSONL.
        rejected_jsonl: Rejected-row output JSONL.
        decisions_jsonl: Decision cache/audit JSONL.
        report_json: Summary report path.
        limit: Optional input row cap.
        image_detail: Provider image detail setting.
        max_frames_per_call: Maximum FPV frames sent per sample.
        keep_labels: Integer labels to keep.
        failure_policy: Label policy for failures.
        progress_interval: Progress print interval.
        concurrency: Requested concurrent VLM calls.
        api_key_pool: Optional comma-separated API keys.
        api_key_file: Optional key file.
        retry_invalid_cached: Retry cached decisions whose ``valid`` flag is false.

    Returns:
        Summary report dictionary.
    """

    rows = list(read_jsonl(source_jsonl))
    if limit is not None:
        rows = rows[: int(limit)]

    existing_decisions = load_existing_decisions(decisions_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rejected_jsonl.parent.mkdir(parents=True, exist_ok=True)
    decisions_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("", encoding="utf-8")
    rejected_jsonl.write_text("", encoding="utf-8")

    api_keys = load_api_key_pool(api_key_pool, api_key_file)
    effective_concurrency = max(1, int(concurrency))
    if api_keys:
        effective_concurrency = min(effective_concurrency, len(api_keys))

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_jsonl": str(source_jsonl),
        "output_jsonl": str(output_jsonl),
        "rejected_jsonl": str(rejected_jsonl),
        "decisions_jsonl": str(decisions_jsonl),
        "input_rows": len(rows),
        "limit": limit,
        "image_detail": image_detail,
        "max_frames_per_call": int(max_frames_per_call),
        "keep_labels": sorted(keep_labels),
        "failure_policy": failure_policy,
        "concurrency": max(1, int(concurrency)),
        "effective_concurrency": effective_concurrency,
        "api_key_count": len(api_keys),
        "retry_invalid_cached": bool(retry_invalid_cached),
        "counts": {},
        "label_counts": {},
        "intent_counts": {},
    }

    counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    decisions_by_id: dict[str, dict[str, Any]] = {}
    pending: list[tuple[int, dict[str, Any], str, list[str]]] = []

    for index, row in enumerate(rows, start=1):
        sample_id = row_id(row, index=index)
        intent_counts[row_intent(row)] += 1
        cached = existing_decisions.get(sample_id)
        if isinstance(cached, dict) and isinstance(cached.get("behavior_filter"), dict):
            cached_decision = dict(cached["behavior_filter"])
            if retry_invalid_cached and cached_decision.get("valid") is False:
                counts["retry_invalid_cached"] += 1
            else:
                decisions_by_id[sample_id] = cached_decision
                counts["cached_decision"] += 1
                continue
        frames = collect_behavior_frames(
            row,
            sample_id=sample_id,
            max_frames=max(1, int(max_frames_per_call)),
        )
        frames = resolve_frame_paths(frames, manifest_path=source_jsonl)
        if not frames:
            decisions_by_id[sample_id] = normalize_behavior_decision(
                None,
                raw_answer="",
                failure_policy=failure_policy,
            )
            decisions_by_id[sample_id]["error"] = "no_fpv_frames"
            counts["missing_frames"] += 1
            append_jsonl(
                decisions_jsonl,
                {
                    "id": sample_id,
                    "clip_id": row.get("clip_id") or (row.get("sidewalk_meta") or {}).get("clip_id"),
                    "frame_idx": row.get("frame_idx") or (row.get("sidewalk_meta") or {}).get("frame_idx"),
                    "intent_gt": row_intent(row),
                    "fpv_frame_paths": [],
                    "behavior_filter": decisions_by_id[sample_id],
                    "error": "no_fpv_frames",
                },
            )
            continue
        pending.append((index, row, sample_id, frames))

    if pending:
        semaphore = asyncio.Semaphore(effective_concurrency)
        tasks = []
        for task_index, (_source_index, row, sample_id, frames) in enumerate(pending, start=1):
            api_key = api_keys[(task_index - 1) % len(api_keys)] if api_keys else None
            api_key_index = ((task_index - 1) % len(api_keys)) + 1 if api_keys else None
            tasks.append(
                asyncio.create_task(
                    evaluate_behavior_row(
                        row,
                        sample_id=sample_id,
                        frame_paths=frames,
                        image_detail=image_detail,
                        api_key=api_key,
                        api_key_index=api_key_index,
                        failure_policy=failure_policy,
                        semaphore=semaphore,
                    )
                )
            )
        for completed, task in enumerate(asyncio.as_completed(tasks), start=1):
            sample_id, decision, decision_row, telemetry = await task
            decisions_by_id[sample_id] = decision
            add_usage_to_report(report, telemetry)
            append_jsonl(decisions_jsonl, decision_row)
            counts["new_vlm_decision"] += 1
            if progress_interval > 0 and (completed % progress_interval == 0 or completed == len(tasks)):
                print(
                    json.dumps(
                        {
                            "behavior_progress": f"{completed}/{len(tasks)}",
                            "cached": counts.get("cached_decision", 0),
                            "new_vlm": counts.get("new_vlm_decision", 0),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    for index, row in enumerate(rows, start=1):
        sample_id = row_id(row, index=index)
        decision = decisions_by_id.get(sample_id)
        if decision is None:
            decision = normalize_behavior_decision(None, raw_answer="", failure_policy=failure_policy)
            decision["error"] = "missing_decision"
        label = str(decision.get("label_name") or decision.get("label") or "unknown")
        label_counts[label] += 1
        out_row = dict(row)
        out_row["behavior_filter"] = decision
        if should_keep_decision(decision, keep_labels):
            append_jsonl(output_jsonl, out_row)
            counts["kept"] += 1
        else:
            append_jsonl(rejected_jsonl, out_row)
            counts["rejected"] += 1

    report["counts"] = dict(sorted(counts.items()))
    report["label_counts"] = dict(sorted(label_counts.items()))
    report["intent_counts"] = dict(sorted(intent_counts.items()))
    report["kept_rows"] = counts.get("kept", 0)
    report["rejected_rows"] = counts.get("rejected", 0)
    write_json(report_json, report)
    return report


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed argparse namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--rejected-jsonl", required=True)
    parser.add_argument("--decisions-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image-detail", default="auto", choices=["auto", "low", "high"])
    parser.add_argument("--max-frames-per-call", type=int, default=5)
    parser.add_argument(
        "--keep-labels",
        default="1",
        help="Comma-separated labels to keep. Use 1 for interesting, 2 for boring, 0 for unknown.",
    )
    parser.add_argument(
        "--failure-policy",
        default="unknown",
        choices=["unknown", "interesting", "boring"],
        help="Decision label used when a VLM call or parse fails.",
    )
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=1)
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
    parser.add_argument(
        "--retry-invalid-cached",
        action="store_true",
        help="Retry cached decisions whose behavior_filter.valid is false, such as API failures.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the behavior-filter CLI."""

    args = parse_args()
    keep_labels = {
        int(item.strip())
        for item in str(args.keep_labels).split(",")
        if item.strip() in {"0", "1", "2"}
    }
    if not keep_labels:
        raise SystemExit("--keep-labels must contain at least one of 0,1,2")
    report = asyncio.run(
        filter_behavior_samples(
            source_jsonl=Path(args.source_jsonl),
            output_jsonl=Path(args.output_jsonl),
            rejected_jsonl=Path(args.rejected_jsonl),
            decisions_jsonl=Path(args.decisions_jsonl),
            report_json=Path(args.report_json),
            limit=args.limit,
            image_detail=str(args.image_detail),
            max_frames_per_call=int(args.max_frames_per_call),
            keep_labels=keep_labels,
            failure_policy=str(args.failure_policy),
            progress_interval=int(args.progress_interval),
            concurrency=int(args.concurrency),
            api_key_pool=args.api_key_pool,
            api_key_file=Path(args.api_key_file) if args.api_key_file else None,
            retry_invalid_cached=bool(args.retry_invalid_cached),
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
